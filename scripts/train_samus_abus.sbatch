#!/bin/bash
#SBATCH --job-name=samus-abus
#SBATCH --output=logs/samus_abus_%j.out
#SBATCH --error=logs/samus_abus_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Stage 1: Fine-tune SAMUS on ABUS
#
# Loads SAMUS pretrained checkpoint (trained on US30K) and fine-tunes the
# adapter modules (cnn_embed, Adapter, upneck, rel_pos) on ABUS data.
# Prompt encoder and mask decoder remain frozen.
#
# Output: checkpoints/ABUS/SAMUS_<timestamp>_<epoch>_<dice>.pth
# This checkpoint is used as input for Stage 2 (AutoSAMUS training).
# =============================================================================

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl
module load cuda/11.8.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated successfully"
echo "Python path: $(which python)"
echo ""

# Change to project directory
PROJECT_DIR=/scratch/ll5582/Projects/SAMUS_For_3D
DATA_PATH=/scratch/ll5582/Data/Ultrasound/processed_abus_seg
SHARD_DIR=/scratch/ll5582/Data/Ultrasound/Shards_ABUS

cd ${PROJECT_DIR}
echo "Changed to directory: $(pwd)"
echo ""

# Create directories
mkdir -p logs checkpoints

# Download checkpoints if needed
if [ ! -f checkpoints/samus_pretrained.pth ]; then
    echo "Downloading SAMUS pretrained checkpoint..."
    pip install -q gdown
    gdown 1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH -O checkpoints/samus_pretrained.pth
fi

if [ ! -f checkpoints/sam_vit_b_01ec64.pth ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    wget -q -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# Create WebDataset shards if not already done
if [ ! -d "${SHARD_DIR}/train" ]; then
    echo "Creating WebDataset shards..."
    python create_abus_shards.py \
        --data_root ${DATA_PATH} \
        --output_dir ${SHARD_DIR} \
        --samples_per_shard 500

    if [ $? -ne 0 ]; then
        echo "Error: Shard creation failed"
        exit 1
    fi
    echo "Shards created successfully"
else
    echo "Using existing shards at ${SHARD_DIR}"
fi
echo ""

echo "=== Stage 1: Fine-tuning SAMUS on ABUS ==="
echo "Checkpoint: checkpoints/samus_pretrained.pth"
echo "Shard dir: ${SHARD_DIR}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --modelname SAMUS \
    --task ABUS \
    --sam_ckpt checkpoints/samus_pretrained.pth \
    --shard_dir ${SHARD_DIR} \
    --batch_size 2 \
    --base_lr 1e-4 \
    --warmup True \
    --warmup_period 10 \
    -keep_log True

if [ $? -eq 0 ]; then
    echo ""
    echo "Stage 1 training completed successfully!"
    echo "Best checkpoint saved to: checkpoints/ABUS/SAMUS_*.pth"
    echo "Use this checkpoint for Stage 2 (AutoSAMUS) training."
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
