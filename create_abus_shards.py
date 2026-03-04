"""Create WebDataset shards from the processed ABUS segmentation dataset.

Usage:
    python create_abus_shards.py \
        --data_root /path/to/processed_abus_seg \
        --output_dir /path/to/shards \
        --samples_per_shard 500

Input structure:
    data_root/{split}/images/{volume_id}/slice_XXXX.png
    data_root/{split}/masks/{volume_id}/slice_XXXX.png

Output structure:
    output_dir/{split}/shard-{NNNNN}.tar
    output_dir/{split}/index.json
"""

import argparse
import io
import json
import logging
import os

import cv2
import numpy as np
import webdataset as wds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def scan_split(data_root, split):
    """Scan a split directory and return list of (img_path, mask_path, key)."""
    images_dir = os.path.join(data_root, split, "images")
    masks_dir = os.path.join(data_root, split, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    samples = []
    skipped_no_mask = 0
    skipped_empty_mask = 0

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
                skipped_no_mask += 1
                continue

            # Verify mask is non-empty
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.max() == 0:
                skipped_empty_mask += 1
                continue

            slice_name = fname.replace(".png", "")
            key = f"{volume_id}_{slice_name}"
            samples.append((img_path, mask_path, key, volume_id))

    if skipped_no_mask > 0:
        logger.warning(f"[{split}] Skipped {skipped_no_mask} slices: mask file missing")
    if skipped_empty_mask > 0:
        logger.warning(f"[{split}] Skipped {skipped_empty_mask} slices: empty mask")

    return samples


def create_shards(samples, output_dir, split, samples_per_shard=500):
    """Write samples to WebDataset .tar shards."""
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    pattern = os.path.join(split_dir, "shard-%05d.tar")
    n_shards = 0
    n_written = 0
    volumes = set()

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for img_path, mask_path, key, volume_id in samples:
            # Read raw PNG bytes
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()

            # Read image dimensions for metadata
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            metadata = {
                "volume_id": volume_id,
                "slice_name": key.split("_", 1)[1] if "_" in key else key,
                "height": h,
                "width": w,
                "split": split,
            }

            sink.write({
                "__key__": key,
                "image.png": img_bytes,
                "mask.png": mask_bytes,
                "json": json.dumps(metadata).encode(),
            })
            n_written += 1
            volumes.add(volume_id)

    # Count shards
    n_shards = len([f for f in os.listdir(split_dir) if f.endswith(".tar")])

    # Write index
    index = {
        "split": split,
        "n_samples": n_written,
        "n_volumes": len(volumes),
        "n_shards": n_shards,
        "samples_per_shard": samples_per_shard,
    }
    with open(os.path.join(split_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"[{split}] Wrote {n_written} samples in {n_shards} shards "
                f"({len(volumes)} volumes)")
    return index


def main():
    parser = argparse.ArgumentParser(description="Create ABUS WebDataset shards")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to processed_abus_seg directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--samples_per_shard", type=int, default=500,
                        help="Max samples per shard (default: 500)")
    args = parser.parse_args()

    all_stats = {}
    for split in ["train", "val", "test"]:
        logger.info(f"Processing {split}...")
        samples = scan_split(args.data_root, split)
        logger.info(f"  Found {len(samples)} valid samples")
        stats = create_shards(samples, args.output_dir, split,
                              args.samples_per_shard)
        all_stats[split] = stats

    # Write combined index
    with open(os.path.join(args.output_dir, "index.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    total = sum(s["n_samples"] for s in all_stats.values())
    logger.info(f"Done. Total: {total} samples across {len(all_stats)} splits.")


if __name__ == "__main__":
    main()
