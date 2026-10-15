#!/bin/bash
#SBATCH --job-name=s10_imagevar
#SBATCH --output=logs/s10_imagevar_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Generator B: SD Image Variations on the band-power encoder's test predictions, then
# eval_report.py on every arm. Generator A is hpc_submit_stage2.sh with --cond_encoder;
# both write samples.npz and are scored by the same code.
#
#   ENC_DIR=$HOME/NeuroDiffusion/results/s9_<jobid> sbatch jobs/s10_imagevar.sh
#
# Needs jobs/setup_imagevar.sh to have run once on coe-hpc1.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL="$ROOT/pretrains/sd-image-variations"
: "${ENC_DIR:?set ENC_DIR to the bp_clip_encoder results directory}"
[ -d "$MODEL" ] || { echo "FATAL: $MODEL missing; run jobs/setup_imagevar.sh on coe-hpc1" >&2; exit 1; }
cd "$ROOT/code"
OUT="$ROOT/results/imagevar_${SLURM_JOB_ID}"

echo "### generating: encoders from $ENC_DIR"
$PYTHON -u generate_imagevar.py --model "$MODEL" --encoders "$ENC_DIR"/bp_clip_s*.pt --out "$OUT" \
    ${EXTRA_ARGS:-}

echo "### scoring every arm with eval_report.py"
for d in "$OUT"/s*_*/; do
    echo "--- $(basename "$d")"
    $PYTHON eval_report.py --samples "${d}samples.npz" --output "${d}report"
done
echo "### DONE -> $OUT"
