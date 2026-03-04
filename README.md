# SAMUS + AutoSAMUS for ABUS Lesion Segmentation

Fork of [SAMUS](https://arxiv.org/pdf/2309.06824.pdf) with additions for training
AutoSAMUS on 3D ABUS (Automated Breast Ultrasound) 2D slices.

## Two-Stage Training Pipeline

```
Raw ABUS 3D volumes
    |
    v
Preprocessed 2D slices (only lesion-containing)
    |  create_abus_shards.py
    v
WebDataset .tar shards (train/val/test)
    |
    |  Stage 1: train.py --modelname SAMUS --task ABUS
    v
ABUS-fine-tuned SAMUS checkpoint
    |
    |  Stage 2: train.py --modelname AutoSAMUS --task ABUS --load_path <stage1_ckpt>
    v
Trained AutoSAMUS checkpoint (automatic segmentation, no prompts needed)
    |
    |  test.py --modelname AutoSAMUS --task ABUS --load_path <stage2_ckpt>
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
- Frozen: image encoder, prompt encoder, mask decoder
- Trainable: `Prompt_Embedding_Generator` (cross-attention with 50 object tokens) + `feature_adapter` (4-layer CNN)
- No manual clicks or bounding boxes needed at inference

Loss: Dice + BCEWithLogits on 128x128 low-resolution logits.

### Checkpoint Chain

```
SAM ViT-B (sam_vit_b_01ec64.pth)
    |  pretrained on SA-1B
    v
SAMUS pretrained (samus_pretrained.pth)
    |  fine-tuned on US30K (30K ultrasound images)
    |  Trainable: cnn_embed, Adapter, upneck, rel_pos
    v
Stage 1: SAMUS fine-tuned on ABUS (SAMUS_*.pth)
    |  fine-tuned on ABUS train set (~3170 slices)
    |  Trainable: same adapter modules as above
    v
Stage 2: AutoSAMUS on ABUS (AutoSAMUS_*.pth)
    |  loads Stage 1 checkpoint, freezes everything
    |  Trainable: prompt_generator + feature_adapter only
    v
Final model for automatic ABUS lesion segmentation
```

### What is Trainable at Each Stage

**Stage 1 (SAMUS fine-tuning)**:
- Frozen: prompt_encoder, mask_decoder
- Frozen in image_encoder: most ViT blocks
- Trainable in image_encoder: `cnn_embed`, `post_pos_embed`, `Adapter`, global attention `rel_pos` (layers 2,5,8,11), `upneck`

**Stage 2 (AutoSAMUS training)**:
- Frozen: entire image_encoder, prompt_encoder, mask_decoder
- Trainable: `prompt_generator` (Prompt_Embedding_Generator, cross-attention) + `feature_adapter` (4-layer CNN)

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

# SAMUS pretrained (required for both stages)
pip install gdown
gdown 1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH -O checkpoints/samus_pretrained.pth

# SAM ViT-B (fallback)
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Training

### Quick Start (two-stage, WebDataset)

```bash
# 0. Create shards
python create_abus_shards.py \
    --data_root /path/to/processed_abus_seg \
    --output_dir /path/to/abus_shards

# 1. Stage 1: Fine-tune SAMUS on ABUS (adapts encoder to ABUS domain)
python train.py \
    --modelname SAMUS \
    --task ABUS \
    --sam_ckpt checkpoints/samus_pretrained.pth \
    --shard_dir /path/to/abus_shards \
    --batch_size 8 \
    --base_lr 1e-4 \
    --warmup True \
    --warmup_period 250 \
    -keep_log True

# 2. Stage 2: Train AutoSAMUS (auto prompt generator on frozen SAMUS)
#    Use the best SAMUS checkpoint from Stage 1
python train.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --load_path checkpoints/ABUS/SAMUS_best.pth \
    --shard_dir /path/to/abus_shards \
    --batch_size 8 \
    --base_lr 1e-4 \
    --warmup True \
    --warmup_period 250 \
    -keep_log True
```

### Direct File Loading (fallback)

```bash
# Stage 1
python train.py \
    --modelname SAMUS \
    --task ABUS \
    --sam_ckpt checkpoints/samus_pretrained.pth \
    --data_path /path/to/processed_abus_seg \
    --batch_size 8 --base_lr 1e-4

# Stage 2
python train.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --load_path checkpoints/ABUS/SAMUS_best.pth \
    --data_path /path/to/processed_abus_seg \
    --batch_size 8 --base_lr 1e-4
```

### HPC (SLURM)

```bash
# Stage 1: Fine-tune SAMUS on ABUS
sbatch scripts/train_samus_abus.sbatch

# Stage 2: Train AutoSAMUS (after Stage 1 completes)
# Automatically finds the latest SAMUS checkpoint, or set explicitly:
SAMUS_CKPT=checkpoints/ABUS/SAMUS_best.pth sbatch scripts/train_autosamus_abus.sbatch
```

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

## Testing

```bash
# Test SAMUS (Stage 1, with manual prompts)
python test.py \
    --modelname SAMUS \
    --task ABUS \
    --load_path checkpoints/ABUS/SAMUS_best.pth \
    --shard_dir /path/to/abus_shards

# Test AutoSAMUS (Stage 2, automatic segmentation)
python test.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --load_path checkpoints/ABUS/AutoSAMUS_best.pth \
    --shard_dir /path/to/abus_shards
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
    train_samus_abus.sbatch        # Stage 1: SAMUS fine-tuning on ABUS
    train_autosamus_abus.sbatch    # Stage 2: AutoSAMUS training on ABUS
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
