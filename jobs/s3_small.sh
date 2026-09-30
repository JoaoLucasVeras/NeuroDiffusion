#!/bin/bash
#SBATCH --job-name=s3_small
#SBATCH --output=logs/s3_small_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s3_$SLURM_JOB_ID"
mkdir -p "$R"

banner () { echo ""; echo "#################################################################"
            echo "### $*"; echo "#################################################################"; }

# Alpha arm. The band the classical baseline proves carries imagery signal, and the
# one both Stage 2 runs failed on.
banner "ALPHA 8-13 Hz, subject 1, 200-permutation null"
$PYTHON -u small_encoder.py --subject 1 --band 8 13 --epochs 200 --permute 200 \
    --json "$R/small_s1_alpha.json"

# Broadband arm. The control that matters: if the small encoder gains the same way the
# 85M model did, its advantage is the recording fingerprint rather than imagery. The
# cross-subject test says that advantage should not be real.
banner "BROADBAND 1-50 Hz, subject 1, 200-permutation null"
$PYTHON -u small_encoder.py --subject 1 --band 0 0 --epochs 200 --permute 200 \
    --json "$R/small_s1_broad.json"

echo ""; echo "### DONE -> $R ###"
ls -la "$R"
