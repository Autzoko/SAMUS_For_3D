#!/bin/bash
#SBATCH --job-name=autosamus-abus
#SBATCH --output=logs/autosamus_abus_%j.out
#SBATCH --error=logs/autosamus_abus_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Stage 2: Train AutoSAMUS on ABUS
#
# Loads the ABUS-fine-tuned SAMUS checkpoint from Stage 1 and trains the
# auto prompt generator (Prompt_Embedding_Generator + feature_adapter).
# Image encoder, prompt encoder, and mask decoder are all frozen.
#
# Prerequisites: Run Stage 1 (train_samus_abus.sbatch) first to get the
# ABUS-fine-tuned SAMUS checkpoint.
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
SHARD_DIR=/scratch/ll5582/Data/Ultrasound/Shards_ABUS

cd ${PROJECT_DIR}
echo "Changed to directory: $(pwd)"
echo ""

# Create directories
mkdir -p logs checkpoints

# ---- Set the Stage 1 checkpoint path ----
# After running Stage 1, find the best SAMUS checkpoint and set it here.
# Example: SAMUS_CKPT=checkpoints/ABUS/SAMUS_03041230_150_0.7823.pth
SAMUS_CKPT=${SAMUS_CKPT:-""}

if [ -z "${SAMUS_CKPT}" ]; then
    # Auto-find: use the most recent SAMUS checkpoint in checkpoints/ABUS/
    SAMUS_CKPT=$(ls -t checkpoints/ABUS/SAMUS_*.pth 2>/dev/null | head -1)
    if [ -z "${SAMUS_CKPT}" ]; then
        echo "Error: No Stage 1 SAMUS checkpoint found in checkpoints/ABUS/"
        echo "Run Stage 1 (train_samus_abus.sbatch) first, or set SAMUS_CKPT env var."
        exit 1
    fi
fi

if [ ! -f "${SAMUS_CKPT}" ]; then
    echo "Error: SAMUS checkpoint not found: ${SAMUS_CKPT}"
    exit 1
fi

echo "=== Stage 2: Training AutoSAMUS on ABUS ==="
echo "Stage 1 checkpoint: ${SAMUS_CKPT}"
echo "Shard dir: ${SHARD_DIR}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --modelname AutoSAMUS \
    --task ABUS \
    --load_path ${SAMUS_CKPT} \
    --shard_dir ${SHARD_DIR} \
    --batch_size 2 \
    --base_lr 1e-4 \
    --warmup True \
    --warmup_period 10 \
    -keep_log True

if [ $? -eq 0 ]; then
    echo ""
    echo "Stage 2 training completed successfully!"
    echo "Best checkpoint saved to: checkpoints/ABUS/AutoSAMUS_*.pth"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
