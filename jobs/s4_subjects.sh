#!/bin/bash
#SBATCH --job-name=s4_subjects
#SBATCH --output=logs/s4_subjects_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# The small encoder reached 3.25x chance at p = 0.005 on subject 1's alpha band, matching
# what band power plus logistic regression achieves, but in CLIP space. This asks whether
# that holds for the other three people or was a subject-1 artifact.
#
# Subject 4 is the one to watch: significant in the classical baseline at 3.0x chance,
# yet exactly at chance in the 85M-parameter deep model.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$ROOT/code"
R="$ROOT/results/s4_$SLURM_JOB_ID"
mkdir -p "$R"

for S in 2 3 4; do
  echo ""
  echo "#################################################################"
  echo "### SUBJECT $S, alpha 8-13 Hz, 200-permutation null"
  echo "#################################################################"
  $PYTHON -u small_encoder.py --subject "$S" --band 8 13 --epochs 200 --permute 200 \
      --json "$R/small_s${S}_alpha.json"
done

echo ""; echo "### DONE -> $R ###"
ls -la "$R"
