#!/bin/bash
#SBATCH --job-name=s11_gao
#SBATCH --output=logs/s11_gao_%A_%a.out
#SBATCH --array=1-22%6
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail

# Convert one Gao et al. subject from the downloaded zip to datasets/gao/gao_sub-XX.pt.
# The raw BDF files are unpacked to a per-subject folder, converted, then deleted, so no
# more than six subjects' raw data (the array limit) sit on /home at once. The zip itself
# is kept until every subject has converted.
# Needs jobs/fetch_gao.sh to have run on coe-hpc1 first.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
S=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
ZIP="$ROOT/datasets/gao_raw/sub-$S.zip"
TMP="$ROOT/datasets/gao_raw/unpacked_sub-$S"
[ -f "$ZIP" ] || { echo "FATAL: $ZIP missing; run jobs/fetch_gao.sh on coe-hpc1" >&2; exit 1; }
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT/code"
echo "### sub-$S: unpacking"
$PYTHON -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$TMP"
echo "### sub-$S: converting"
$PYTHON -u prepare_gao_data.py --bids "$TMP" --subject "$SLURM_ARRAY_TASK_ID" --out "$ROOT/datasets/gao"
echo "### sub-$S: done"
