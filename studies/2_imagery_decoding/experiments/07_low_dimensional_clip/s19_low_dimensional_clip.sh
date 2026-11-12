#!/bin/bash
# REVIEW DRAFT: cluster execution must wait for the project lead.
#SBATCH --job-name=s19_visual_axes
#SBATCH --output=logs/s19_visual_axes_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=REPLACE_WITH_NSF_QUEUE   # advisor rule: NSF queue only; run `sinfo -s` for its name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail

# The lead must set APPROVED_BY_PROJECT_LEAD=1 after reviewing the protocol.
if [[ "${APPROVED_BY_PROJECT_LEAD:-0}" != "1" ]]; then
    echo "Cluster execution is pending project lead approval." >&2
    exit 2
fi

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
# Inputs default to this checkout; point them at shared copies with DATASET= and CACHE=.
DATASET="${DATASET:-$ROOT/datasets/imagination_5_95_std.pth}"
CACHE="${CACHE:-$ROOT/datasets/clip_stimulus_emb.pt}"
# PERMUTE=25 gives a short trial run to check the job and time it before the full 1000.
PERMUTE="${PERMUTE:-1000}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
"$PYTHON" -u studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py \
    --dataset "$DATASET" \
    --cache "$CACHE" \
    --allow_missing_embeddings \
    --output "$ROOT/results/s19_${SLURM_JOB_ID}" \
    --seed 2022 --permute "$PERMUTE"
