#!/bin/bash
#SBATCH --job-name=s15_eyechk
#SBATCH --output=logs/s15_eyechk_%A_%a.out
#SBATCH --array=1-22%10
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Imagined shapes: brain or eyes? One array task per subject, CPU only. Alpha power by
# head region and by second, plus eye signals alone, train one session / test the other,
# 200-shuffle null. Runs on our stored trials, then on the authors' ICA-cleaned data
# (eye components removed). Subjects 9 and 10 have one session and exit immediately.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT/code"
R="$ROOT/results/s15_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"
SS=$(printf %02d "$S")
$PYTHON -u gao_eyecheck.py --subject "$S" --task FVI --permute 200 --json "$R/eyecheck_ours_s$SS.json"
$PYTHON -u gao_eyecheck.py --subject "$S" --task FVI --permute 200 --data ../datasets/gao_ica \
    --json "$R/eyecheck_ica_s$SS.json"
