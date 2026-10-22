#!/bin/bash
#SBATCH --job-name=s16_gaorepro
#SBATCH --output=logs/s16_gaorepro_%A_%a.out
#SBATCH --array=1-22%10
#SBATCH --time=10:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Gao reproduction, stage 1. One array task per person, CPU only, no GPU. Exports the
# animals recordings in the files the authors' EEGNet.py expects, then runs their script
# unchanged on real labels and on scrambled labels. About 47 min per recording per label
# set: 4 runs for people with two sessions (~3 h), 2 for sub-09 and sub-10.
# Runs in the gaoprep env: torch 2.3.1 (accepts the scheduler's verbose argument the
# script passes) and scikit-learn (its StratifiedKFold and metrics).

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${GAOPREP_PYTHON:-$HOME/.conda/envs/gaoprep/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
cd "$ROOT"
S="$SLURM_ARRAY_TASK_ID"
SS=$(printf %02d "$S")
OUT="$ROOT/datasets/gao_repro"
$PYTHON -u studies/2_imagery_decoding/experiments/03_gao_reproduction/gao_repro_export.py --data datasets/gao_ica --out "$OUT" --subject "$S"
for KIND in real scrambled; do
    $PYTHON -u studies/2_imagery_decoding/experiments/03_gao_reproduction/gao_repro_run.py --folder "$OUT/$KIND/sub-$SS"
done
