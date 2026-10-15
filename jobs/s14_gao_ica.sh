#!/bin/bash
#SBATCH --job-name=s14_gaoica
#SBATCH --output=logs/s14_gaoica_%A_%a.out
#SBATCH --array=1-22%8
#SBATCH --time=03:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail

# Option B: preprocess one Gao subject with the authors' own code (ICA + ICLabel) in the
# separate gaoprep environment, writing datasets/gao_ica/gao_sub-XX.pt. Unpacks the zip
# to a per-subject folder and deletes it afterwards.
# Needs jobs/fetch_gao.sh and jobs/setup_gao_prep.sh to have run on coe-hpc1.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PREP="$HOME/.conda/envs/gaoprep/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
ZIP="$ROOT/datasets/gao_raw/sub-$S.zip"
TMP="$ROOT/datasets/gao_raw/unpacked_ica_sub-$S"
[ -f "$ZIP" ] || { echo "FATAL: $ZIP missing" >&2; exit 1; }
[ -x "$PREP" ] || { echo "FATAL: gaoprep environment missing" >&2; exit 1; }
trap 'rm -rf "$TMP"' EXIT
cd "$ROOT/code"
$PREP -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$TMP"
$PREP -u gao_ica_prep.py --bids "$TMP" --subject "$SLURM_ARRAY_TASK_ID" --out "$ROOT/datasets/gao_ica"
