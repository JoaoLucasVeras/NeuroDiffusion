#!/bin/bash -l
# Full CPU run: submit only after separate approval following the smoke test.
#SBATCH --job-name=s19_visual_axes
#SBATCH --output=logs/s19_visual_axes_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=nsfq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1
set -euo pipefail

# The lead must set APPROVED_BY_PROJECT_LEAD=1 after reviewing the protocol.
if [[ "${APPROVED_BY_PROJECT_LEAD:-0}" != "1" ]]; then
    echo "Cluster execution is pending project lead approval." >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit through sbatch; do not run on a login node}"

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
cd "$ROOT"
branch="$(git symbolic-ref --short HEAD)"
if [[ "$branch" != "aditya-individual-branch" ]]; then
    echo "Refusing to run outside aditya-individual-branch (found $branch)." >&2
    exit 2
fi
module load python3/3.12.12
module load ml/torch/2.6
PYTHON="${NEURODIFFUSION_PYTHON:-/app/rl9.x/ml/torch/2.6/bin/python}"
# Inputs default to this checkout; point them at shared copies with DATASET= and CACHE=.
DATASET="${DATASET:-$ROOT/datasets/imagination_5_95_std.pth}"
CACHE="${CACHE:-$ROOT/datasets/clip_stimulus_emb.pt}"
# PERMUTE=25 gives a short trial run to check the job and time it before the full 1000.
PERMUTE="${PERMUTE:-1000}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
echo "CPU run; job $SLURM_JOB_ID; branch $branch; permutations $PERMUTE"
hostname
git log -1 --oneline
full_started=$SECONDS
"$PYTHON" -u studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py \
    --dataset "$DATASET" \
    --cache "$CACHE" \
    --allow_missing_embeddings \
    --output "$ROOT/results/s19_${SLURM_JOB_ID}" \
    --seed 2022 --permute "$PERMUTE"
echo "FULL_RUN_SECONDS=$((SECONDS - full_started))"
