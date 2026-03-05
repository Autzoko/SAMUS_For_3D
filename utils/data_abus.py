"""ABUS 2D slice dataset for SAMUS/AutoSAMUS training.

Supports two data loading modes:
  1. WebDataset shards (recommended for HPC):
     shard_dir/{split}/shard-NNNNN.tar
  2. Direct file loading (fallback):
     data_root/{split}/images/{volume_id}/slice_XXXX.png
     data_root/{split}/masks/{volume_id}/slice_XXXX.png
"""

import io
import json
import logging
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_us import (
    JointTransform2D,
    correct_dims,
    random_click,
    fixed_click,
    random_bbox,
    fixed_bbox,
)

logger = logging.getLogger(__name__)


# =============================================================================
# WebDataset-based loader (primary)
# =============================================================================

def build_abus_wds_loader(shard_dir, split, joint_transform, img_size=256,
                          batch_size=8, num_workers=4, epoch_length=None):
    """Build a WebDataset-based DataLoader for ABUS segmentation.

    Args:
        shard_dir: Directory containing {split}/shard-NNNNN.tar files.
        split: One of 'train', 'val', 'test'.
        joint_transform: JointTransform2D instance.
        img_size: Target image size for bbox generation.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        epoch_length: If set, artificially set dataset length (for training).

    Returns:
        (dataset, dataloader) tuple.
    """
    import webdataset as wds

    split_dir = os.path.join(shard_dir, split)
    shard_pattern = sorted([
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".tar")
    ])

    if not shard_pattern:
        raise FileNotFoundError(f"No .tar shards found in {split_dir}")

    # Read index for dataset size
    index_path = os.path.join(split_dir, "index.json")
    n_samples = None
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        n_samples = index.get("n_samples")

    is_train = "train" in split

    def decode_sample(sample):
        """Decode a WebDataset sample into SAMUS-compatible format."""
        # Decode image
        img_bytes = sample["image.png"]
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

        # Decode mask
        mask_bytes = sample["mask.png"]
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

        # Binarize mask
        mask[mask > 0] = 1

        # Metadata
        metadata = json.loads(sample["json"])
        sample_name = sample["__key__"]

        # Correct dimensions
        image, mask = correct_dims(image, mask)

        # Apply augmentation
        image, mask_t, low_mask = joint_transform(image, mask)

        # Generate prompts
        mask_np = np.array(mask_t)
        class_id = 1
        if is_train:
            pt, point_label = random_click(mask_np, class_id)
            bbox = random_bbox(mask_np, class_id, img_size)
        else:
            pt, point_label = fixed_click(mask_np, class_id)
            bbox = fixed_bbox(mask_np, class_id, img_size)

        # Binarize for target class
        mask_t[mask_t != class_id] = 0
        mask_t[mask_t == class_id] = 1
        low_mask[low_mask != class_id] = 0
        low_mask[low_mask == class_id] = 1

        point_labels = np.array(point_label)
        low_mask = low_mask.unsqueeze(0)
        mask_t = mask_t.unsqueeze(0)

        return {
            "image": image,
            "label": mask_t,
            "p_label": point_labels,
            "pt": pt,
            "bbox": bbox,
            "low_mask": low_mask,
            "image_name": sample_name + ".png",
            "class_id": class_id,
        }

    # Ensure enough shards for workers
    n_shards = len(shard_pattern)
    effective_workers = min(num_workers, n_shards)

    # Build pipeline
    dataset = wds.WebDataset(
        shard_pattern,
        shardshuffle=n_shards if is_train else False,
        nodesplitter=wds.split_by_node,
    )

    if is_train:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(decode_sample)

    # with_epoch(n) yields n *samples*, then stops. DataLoader batches them.
    # So with_epoch(n_samples) → n_samples // batch_size batches per epoch.
    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)
    elif n_samples is not None:
        dataset = dataset.with_epoch(n_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=_collate_abus,
    )

    logger.info(f"ABUS WebDataset [{split}]: {n_shards} shards, "
                f"{n_samples or '?'} samples, workers={effective_workers}")

    return dataset, loader


def _collate_abus(batch):
    """Custom collate that handles mixed tensor/numpy fields."""
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(vals))
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals  # strings, etc.
    return result


# =============================================================================
# Direct file-based loader (fallback)
# =============================================================================

class ABUSDataset(Dataset):
    """Dataset for ABUS 2D slices with hierarchical directory structure.

    Args:
        data_root: Path to processed_abus_seg directory.
        split: One of 'train', 'val', 'test'.
        joint_transform: Augmentation transform (JointTransform2D instance).
        img_size: Target image size (used for bbox generation).
        prompt: Prompt type ('click').
    """

    def __init__(self, data_root, split="train", joint_transform=None,
                 img_size=256, prompt="click"):
        self.data_root = data_root
        self.split = split
        self.joint_transform = joint_transform
        self.img_size = img_size
        self.prompt = prompt
        self.class_id = 1  # binary: background=0, lesion=1

        self.samples = self._scan_samples()
        logger.info(f"ABUSDataset [{split}]: {len(self.samples)} slices "
                    f"from {self._count_volumes()} volumes")

    def _scan_samples(self):
        """Scan directory structure and build list of (img_path, mask_path, name)."""
        samples = []
        images_dir = os.path.join(self.data_root, self.split, "images")
        masks_dir = os.path.join(self.data_root, self.split, "masks")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        for volume_id in sorted(os.listdir(images_dir)):
            vol_img_dir = os.path.join(images_dir, volume_id)
            vol_mask_dir = os.path.join(masks_dir, volume_id)
            if not os.path.isdir(vol_img_dir):
                continue

            for fname in sorted(os.listdir(vol_img_dir)):
                if not fname.endswith(".png"):
                    continue
                img_path = os.path.join(vol_img_dir, fname)
                mask_path = os.path.join(vol_mask_dir, fname)

                if not os.path.isfile(mask_path):
                    continue

                slice_name = fname.replace(".png", "")
                sample_name = f"{volume_id}_{slice_name}"
                samples.append((img_path, mask_path, sample_name))

        if len(samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {images_dir}. "
                f"Check directory structure."
            )
        return samples

    def _count_volumes(self):
        volumes = set()
        for _, _, name in self.samples:
            vol_id = name.rsplit("_slice_", 1)[0]
            volumes.add(vol_id)
        return len(volumes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, sample_name = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        # Binarize mask
        mask[mask > 0] = 1

        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        else:
            raise RuntimeError("joint_transform is required")

        mask_np = np.array(mask)
        if "train" in self.split:
            pt, point_label = random_click(mask_np, self.class_id)
            bbox = random_bbox(mask_np, self.class_id, self.img_size)
        else:
            pt, point_label = fixed_click(mask_np, self.class_id)
            bbox = fixed_bbox(mask_np, self.class_id, self.img_size)

        mask[mask != self.class_id] = 0
        mask[mask == self.class_id] = 1
        low_mask[low_mask != self.class_id] = 0
        low_mask[low_mask == self.class_id] = 1

        point_labels = np.array(point_label)
        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {
            "image": image,
            "label": mask,
            "p_label": point_labels,
            "pt": pt,
            "bbox": bbox,
            "low_mask": low_mask,
            "image_name": sample_name + ".png",
            "class_id": self.class_id,
        }
