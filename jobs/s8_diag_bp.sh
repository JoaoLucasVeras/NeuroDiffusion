#!/bin/bash
#SBATCH --job-name=s8_diag_bp
#SBATCH --output=logs/s8_diag_bp_%A_%a.out
#SBATCH --array=1-4
#SBATCH --time=06:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail

# Backbone x head diagnostic, band-power arms. One array task per subject.
#   bp_cls  : band power + logistic regression, now with selection inside the null.
#             Should reproduce 3.25 / 2.25 / 2.75 / 3.00x; the p-values are the corrected ones.
#   bp_clip : the same features and optimiser, landing in CLIP space with InfoNCE.
#             The cell that matters: fixed features against the CLIP target.
# Pure CPU work, so it runs here rather than holding a GPU.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s8_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"

echo "### SUBJECT $S, alpha 8-13 Hz, bp_cls + bp_clip, 200 permutations"
$PYTHON -u head_backbone_diag.py --subject "$S" --band 8 13 --arms bp_cls bp_clip \
    --permute 200 --json "$R/diag_bp_s${S}.json"
echo "### DONE -> $R/diag_bp_s${S}.json"
