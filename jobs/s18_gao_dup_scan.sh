#!/bin/bash
#SBATCH --job-name=s18_dupscan
#SBATCH --output=logs/s18_dupscan_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail

# Checksum the EEG data section and events file of every recording in the published Gao
# zips, and report any shared by more than one recording. Read-only, one CPU.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
cd "$ROOT/code"
mkdir -p "$ROOT/results"
$PYTHON -u gao_dup_scan.py --raw ../datasets/gao_raw --json "$ROOT/results/s18_dupscan_${SLURM_JOB_ID}.json"
