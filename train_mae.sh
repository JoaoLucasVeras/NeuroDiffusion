#!/bin/bash
#SBATCH --job-name=NeuroDiff_Stage1
#SBATCH --output=/home/015555345/NeruoDiffusion/logs/stage1_%j.out
#SBATCH --error=/home/015555345/NeruoDiffusion/logs/stage1_%j.err
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Stage 1: masked EEG pre-training (self-supervised).
# This stage never touches the stimulus images, so it can run before the
# stimulus set is complete.

set -euo pipefail

module load cuda/11.8 2>/dev/null || true
module load cudnn/8.6.0 2>/dev/null || true

ROOT=/home/015555345/NeruoDiffusion
export PYTHON=$ROOT/../.conda/envs/neurodiffusion/bin/python
[ -x "$PYTHON" ] || export PYTHON=/home/015555345/.conda/envs/neurodiffusion/bin/python
export PATH=$(dirname "$PYTHON"):$PATH

# Compute nodes are air-gapped.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p "$ROOT/logs" "$ROOT/results/eeg_pretrain"
cd "$ROOT/code"

DATASET=$ROOT/datasets/imagination_5_95_std.pth
SPLITS=$ROOT/datasets/imagination_5_95_std_splits_subject.pth

# Refuse to burn an allocation on a corrupt dataset.
echo "### PREFLIGHT ###"
$PYTHON check_data.py --dataset "$DATASET" --splits "$SPLITS" --skip_image_check

echo "### STAGE 1: masked EEG pre-training ###"
$PYTHON -u stageA1_eeg_pretrain.py \
    --root_path "$ROOT/" \
    --batch_size 64 \
    --num_epoch 400 \
    --lr 1.5e-4 \
    --patch_size 4 \
    --embed_dim 1024 \
    --decoder_embed_dim 512

echo "### DONE. Latest checkpoint: ###"
ls -td "$ROOT"/results/eeg_pretrain/*/ | head -1
