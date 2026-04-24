#!/bin/bash
#SBATCH --job-name=neuro_mae
#SBATCH --output=logs/res_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Load modules (Adjust based on SJSU HPC system specifics if known)
# Common for many HPCs:
module load anaconda3 || module load miniconda3
source activate neurodiffusion

# Create logs directory if it doesn't exist
mkdir -p ../results/logs

# Move to the code directory
cd code

# Run the Stage 1 Pretraining
# Note: On HPC, we can use the original high performance config
# You may want to edit config.py before running to increase batch size to 128
python stageA1_eeg_pretrain.py
