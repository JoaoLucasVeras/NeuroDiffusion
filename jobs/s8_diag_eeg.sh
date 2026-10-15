#!/bin/bash
#SBATCH --job-name=s8_diag_eeg
#SBATCH --output=logs/s8_diag_eeg_%A_%a.out
#SBATCH --array=1-4
#SBATCH --time=06:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Backbone x head diagnostic, EEGNet arm. One array task per subject.
#   eeg_cls : the small encoder, identical in every respect except a 33-way head and
#             cross-entropy in place of the CLIP head and InfoNCE.
# Compare against eeg_clip from jobs 84326 / 84567. One subject took 2h21m there, so the
# 6h limit leaves room; 84326 was killed by a limit sized too tightly.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s8_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"

echo "### SUBJECT $S, alpha 8-13 Hz, eeg_cls, 200 permutations"
$PYTHON -u head_backbone_diag.py --subject "$S" --band 8 13 --arms eeg_cls \
    --permute 200 --json "$R/diag_eeg_s${S}.json"
echo "### DONE -> $R/diag_eeg_s${S}.json"
