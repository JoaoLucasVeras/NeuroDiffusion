#!/bin/bash
#SBATCH --job-name=s6_subj4
#SBATCH --output=logs/s6_subj4_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Subject 4 alone. Job 84326 trained and evaluated this subject fine but was killed by its
# 8h wall limit after 60 of 200 permutations, so subject 4 is the one blank cell in the
# table. One subject took ~2h20m in 84326, so 4h is ample.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s6_$SLURM_JOB_ID"
mkdir -p "$R"

echo "### SUBJECT 4, alpha 8-13 Hz, 200 permutations, both metrics"
echo "### expecting R@1 0.0530 (1.75x) and pooled 0.0606 (2.00x) to reproduce"
$PYTHON -u small_encoder.py --subject 4 --band 8 13 --epochs 200 --permute 200 \
    --json "$R/small_s4_alpha.json"

echo ""; echo "### DONE -> $R ###"
ls -la "$R"
