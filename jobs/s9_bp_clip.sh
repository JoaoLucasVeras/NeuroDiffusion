#!/bin/bash
#SBATCH --job-name=s9_bp_clip
#SBATCH --output=logs/s9_bp_clip_%A_%a.out
#SBATCH --array=1-4
#SBATCH --time=03:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail

# Train and save the band-power -> CLIP encoder per subject, for the conditioning step.
# l2 is chosen on training windows only, so the saved test predictions never saw the
# test set. The 200-shuffle null re-runs that inner choice on every shuffle.
# CPU only: about 45 minutes per subject, judging by the band-power arms of job 84691.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s9_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"

echo "### SUBJECT $S, alpha 8-13 Hz, band power -> CLIP, inner l2 choice, 200 permutations"
$PYTHON -u bp_clip_encoder.py --subject "$S" --band 8 13 --permute 200 \
    --save "$R/bp_clip_s${S}.pt" --json "$R/bp_clip_s${S}.json"
echo "### DONE -> $R"
