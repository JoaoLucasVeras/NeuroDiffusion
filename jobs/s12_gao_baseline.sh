#!/bin/bash
#SBATCH --job-name=s12_gaobl
#SBATCH --output=logs/s12_gaobl_%A_%a.out
#SBATCH --array=1-22%10
#SBATCH --time=02:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Classical baseline on the Gao et al. data, one array task per subject: train on
# session 1, test on session 2, within each task, alpha band power and 1-15 Hz waveform,
# imagery and perception windows, 200-shuffle null. Subjects 9 and 10 have one session
# and exit immediately. All settings were fixed before any Gao result was seen.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT/code"
R="$ROOT/results/s12_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"
# EXTRA_ARGS="--align session" standardises features within each session (no labels used)
$PYTHON -u gao_baseline.py --subject "$S" --band 8 13 --permute 200 ${EXTRA_ARGS:-} \
    --json "$R/gao_baseline_s$(printf %02d "$S").json"
