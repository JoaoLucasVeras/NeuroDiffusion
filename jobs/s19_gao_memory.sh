#!/bin/bash
#SBATCH --job-name=s19_memory
#SBATCH --output=logs/s19_memory_%A_%a.out
#SBATCH --array=1-22%10
#SBATCH --time=02:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail

# Imagery as memory replay: pattern match to the other day's seeing templates (test 1)
# and the same with flexible timing (test 2), shapes, animals and objects, 500-shuffle
# null. Settings fixed in gao_memory.py's docstring. CPU only, a few minutes per person.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$ROOT/code"
R="$ROOT/results/s19_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"
$PYTHON -u gao_memory.py --subject "$S" --permute 500 --json "$R/memory_s$(printf %02d "$S").json"
