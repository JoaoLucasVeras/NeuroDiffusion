#!/bin/bash
#SBATCH --job-name=s17_enhance
#SBATCH --output=logs/s17_enhance_%A_%a.out
#SBATCH --array=1-22%10
#SBATCH --time=06:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Enhancement round (B) on imagined shapes, one array task per person, CPU only.
# Methods, targets (45% three-way, 65% two-way) and the decision rule are fixed in
# gao_enhance.py's docstring, written before any result. Subjects 9 and 10 have one
# session and exit immediately.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
R="$ROOT/results/s17_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$R"
S="$SLURM_ARRAY_TASK_ID"
$PYTHON -u studies/2_imagery_decoding/experiments/05_shapes_enhance/gao_enhance.py --subject "$S" --permute 200 --permute_transfer 100 \
    --json "$R/enhance_s$(printf %02d "$S").json"
