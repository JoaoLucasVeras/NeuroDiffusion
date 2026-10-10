#!/bin/bash
#SBATCH --job-name=s13_eegnet
#SBATCH --output=logs/s13_eegnet_%A_%a.out
#SBATCH --array=1-22%6
#SBATCH --time=04:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# The authors' EEGNet on the Gao data, scored four ways per session and task:
# as published (max test accuracy over epochs), the same on shuffled labels (what that
# procedure reports with nothing to find), honest shuffled 5-fold, honest time-ordered.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export OMP_NUM_THREADS=4
cd "$ROOT/code"
R="$ROOT/results/s13_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"
$PYTHON -u gao_eegnet.py --subject "$S" --epochs 500 ${EXTRA_ARGS:-} \
    --json "$R/gao_eegnet_s$(printf %02d "$S").json"
