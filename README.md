# SAMUS + AutoSAMUS for ABUS Lesion Segmentation

Fork of [SAMUS](https://arxiv.org/pdf/2309.06824.pdf) with additions for training
AutoSAMUS on 3D ABUS (Automated Breast Ultrasound) 2D slices.

```
Raw ABUS 3D volumes
    |
    v
Preprocessed 2D slices (only lesion-containing)
    |  create_abus_shards.py
    v
WebDataset .tar shards (train/val/test)
    |  train.py --modelname AutoSAMUS --task ABUS
    v
Trained AutoSAMUS checkpoint
    |  test.py --modelname AutoSAMUS --task ABUS
    v
Evaluation: Dice, IoU, Hausdorff distance
```

## Model Architecture

**SAMUS** adapts SAM (Segment Anything Model) for ultrasound:
- ViT-B image encoder (patch_size=8, 256x256 input)
- CNN embedding layer for grayscale ultrasound
- Adapter modules in transformer blocks
- Standard SAM prompt encoder + mask decoder

**AutoSAMUS** adds automatic prompt generation on top of frozen SAMUS:
- Frozen: image encoder, prompt encoder, mask decoder (loaded from SAMUS pretrained)
- Trainable: `Prompt_Embedding_Generator` (cross-attention with 50 object tokens) + `feature_adapter` (4-layer CNN)
- No manual clicks or bounding boxes needed at inference

Loss: Dice + BCEWithLogits on 128x128 low-resolution logits.

## Dataset

### ABUS Segmentation Data

Preprocessed ABUS data with train/val/test splits:

```
processed_abus_seg/
  train/          (100 volumes, ~3170 slices)
    images/{volume_id}/slice_XXXX.png    # 608x865, grayscale
    masks/{volume_id}/slice_XXXX.png     # binary lesion mask
  val/            (30 volumes, ~1110 slices)
  test/           (70 volumes, ~2376 slices)
  summary.csv
```

Only lesion-containing slices are included (no negative/background slices).
Total: ~6,656 slices across 200 volumes.

### WebDataset Shards

For HPC training, convert raw PNGs to WebDataset `.tar` shards to avoid
inode quota issues and enable streaming I/O:

```bash
python create_abus_shards.py \
    --data_root /path/to/processed_abus_seg \
    --output_dir /path/to/abus_shards \
    --samples_per_shard 500
```

Output:
```
abus_shards/
  train/shard-00000.tar ... shard-00006.tar
  train/index.json
  val/shard-00000.tar ... shard-00002.tar
  val/index.json
  test/shard-00000.tar ... shard-00004.tar
  test/index.json
  index.json
```

Each shard sample contains:
- `{key}.image.png` -- grayscale ultrasound slice
- `{key}.mask.png` -- binary lesion mask
- `{key}.json` -- metadata (volume_id, slice_name, dimensions)

## Checkpoints

Download pretrained weights before training:

```bash
mkdir -p checkpoints

# SAMUS pretrained (required for AutoSAMUS initialization)
pip install gdown
gdown 1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH -O checkpoints/samus_pretrained.pth

# SAM ViT-B (fallback)
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Training

### Quick Start (WebDataset, recommended)

```bash
# 1. Create shards
python create_abus_shards.py \
    --data_root /path/to/processed_abus_seg \
    --output_dir /path/to/abus_shards

# 2. Train AutoSAMUS
python train.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --shard_dir /path/to/abus_shards \
    --batch_size 8 \
    --base_lr 1e-4 \
    --warmup True \
    --warmup_period 250 \
    -keep_log True
```

### Direct File Loading (fallback)

```bash
python train.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --data_path /path/to/processed_abus_seg \
    --batch_size 8 \
    --base_lr 1e-4
```

### HPC (SLURM)

```bash
sbatch scripts/train_autosamus_abus.sbatch
```

The sbatch script automatically:
1. Downloads checkpoints if missing
2. Creates WebDataset shards if missing
3. Trains AutoSAMUS with warmup + TensorBoard logging

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 200 | ~165K iterations with bs=8 |
| Learning rate | 1e-4 | AdamW with warmup |
| Warmup | 250 iterations | Linear warmup |
| LR schedule | Polynomial decay (power=0.9) | After warmup |
| Batch size | 8 | Single A100 GPU |
| Input size | 256x256 | Grayscale -> 3ch RGB |
| Loss | Dice + BCE | On 128x128 low-res logits |
| Eval frequency | Every epoch | Slice-level Dice |

### What is Trainable

AutoSAMUS freezes most of the network. Only these modules have gradients:
- `prompt_generator` -- Prompt_Embedding_Generator (cross-attention)
- `feature_adapter` -- 4-layer CNN for dense prompt generation

## Testing

```bash
# Set load_path in utils/config.py to your best checkpoint, then:
python test.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --shard_dir /path/to/abus_shards

# Or with direct file loading:
python test.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --data_path /path/to/processed_abus_seg
```

Reports: Dice, Hausdorff distance, IoU, Accuracy, Sensitivity, Specificity
(per-slice with mean and std).

## File Structure

```
SAMUS/
  models/
    segment_anything/              # Original SAM
    segment_anything_samus/        # SAMUS (ultrasound-adapted SAM)
    segment_anything_samus_autoprompt/  # AutoSAMUS (+ auto prompt generator)
    model_dict.py                  # Model registry
  utils/
    config.py                      # Dataset configs (US30K, TN3K, BUSI, CAMUS, ABUS)
    data_us.py                     # Original SAMUS dataset + augmentations
    data_abus.py                   # ABUS dataset (WebDataset + direct file loading)
    evaluation.py                  # Dice/IoU/HD metrics
    loss_functions/sam_loss.py     # Dice + BCE loss
    generate_prompts.py            # Click/bbox prompt generation
  create_abus_shards.py            # ABUS -> WebDataset shard conversion
  train.py                         # Training script
  test.py                          # Evaluation script
  scripts/
    train_autosamus_abus.sbatch    # SLURM job script
```

## Citation

```
@misc{lin2023samus,
      title={SAMUS: Adapting Segment Anything Model for Clinically-Friendly
             and Generalizable Ultrasound Image Segmentation},
      author={Xian Lin and Yangyang Xiang and Li Zhang and Xin Yang
              and Zengqiang Yan and Li Yu},
      year={2023},
      eprint={2309.06824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
