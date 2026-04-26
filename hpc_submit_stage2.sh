#!/bin/bash
#SBATCH --job-name=NeuroDiff_Stage2
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# 1. Load Modules (Adjust based on SJSU HPC system)
module load cuda/11.8
module load cudnn/8.6.0

# 2. Activate Environment
# Assuming you create a conda env named 'neuro'
source activate neuro

# 3. Navigate to Code
cd code/

# 4. Run Training
# We use the optimized settings from our Colab sessions
python eeg_ldm.py \
    --dataset EEG \
    --batch_size 16 \
    --num_epoch 200 \
    --lr 5.3e-5 \
    --eeg_signals_path ../datasets/eeg_5_95_std.pth \
    --splits_path ../datasets/block_splits_by_image_single.pth \
    --pretrain_gm_path ../pretrains/ \
    --pretrain_mbm_path ../results/eeg_pretrain/latest/checkpoint.pth \
    --use_time_cond True \
    --clip_tune True
