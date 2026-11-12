#!/bin/bash -l
# CPU-only smoke test using the site's existing shared Python/PyTorch modules.
# Submit only after approval of this script and its exact resource request.
#SBATCH --job-name=s19_smoke
#SBATCH --partition=nsfq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1
#SBATCH --output=logs/s19_smoke_%A_%a.out
set -euo pipefail

if [[ "${APPROVED_BY_PROJECT_LEAD:-0}" != "1" ]]; then
    echo "Submission requires approval of the CPU smoke test." >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit through sbatch; do not run on a login node}"

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
cd "$ROOT"
branch="$(git symbolic-ref --short HEAD)"
if [[ "$branch" != "aditya-individual-branch" ]]; then
    echo "Refusing to run outside aditya-individual-branch (found $branch)." >&2
    exit 2
fi

module load python3/3.12.12
module load ml/torch/2.6
PYTHON="/app/rl9.x/ml/torch/2.6/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

echo "CPU smoke test; job $SLURM_JOB_ID; branch $branch"
hostname
git log -1 --oneline
"$PYTHON" -c 'import sys, numpy, scipy, torch; print("Python:", sys.executable, sys.version); print("NumPy:", numpy.__version__, "SciPy:", scipy.__version__, "PyTorch:", torch.__version__); print("Torch CPU threads:", torch.get_num_threads())'

smoke_started=$SECONDS
"$PYTHON" -u studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py \
    --dataset "$ROOT/datasets/imagination_5_95_std.pth" \
    --cache "$ROOT/datasets/clip_stimulus_emb.pt" \
    --allow_missing_embeddings \
    --output "$ROOT/results/s19_hpc_smoke_${SLURM_JOB_ID}" \
    --seed 2022 --permute 25
echo "SMOKE_SECONDS=$((SECONDS - smoke_started))"
