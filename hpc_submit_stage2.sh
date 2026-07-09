#!/bin/bash
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00


export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1

~/.conda/envs/neurodiffusion/bin/python code/eeg_ldm.py --checkpoint_path ./exps/results/generation/01-07-2026-18-31-38/checkpoint_best.pth
