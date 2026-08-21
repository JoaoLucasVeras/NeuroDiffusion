#!/bin/bash
#SBATCH --job-name=stage3_eval
#SBATCH --output=/home/015555345/NeruoDiffusion/logs/stage3_%j.out
#SBATCH --error=/home/015555345/NeruoDiffusion/logs/stage3_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Stage 3: generate from a trained checkpoint and produce the quantitative report.
#
# Usage:
#   sbatch hpc_submit_eval.sh /path/to/checkpoint.pth
#   sbatch hpc_submit_eval.sh            # scores the most recent Stage 2 run

set -euo pipefail

ROOT=/home/015555345/NeruoDiffusion
export PYTHON=/home/015555345/.conda/envs/neurodiffusion/bin/python
export PATH=$(dirname "$PYTHON"):$PATH

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

cd "$ROOT/code"

EXPERIMENT=${EXPERIMENT:-imagination}
PROTOCOL=${PROTOCOL:-subject}
# "_avail" splits exclude trials whose stimulus image could not be sourced.
# Set SPLIT_SUFFIX="" once you have a full ILSVRC2012 copy.
SPLIT_SUFFIX=${SPLIT_SUFFIX-_avail}
DATASET=$ROOT/datasets/${EXPERIMENT}_5_95_std.pth
SPLITS=$ROOT/datasets/${EXPERIMENT}_5_95_std_splits_${PROTOCOL}${SPLIT_SUFFIX}.pth
IMAGENET=$ROOT/datasets/imageNet_images

MODEL=${1:-}
if [ -z "$MODEL" ]; then
    RUN=$(ls -td "$ROOT"/results/generation/*/ | head -1)
    echo "No checkpoint given, using latest run: $RUN"
else
    RUN=$(dirname "$MODEL")/
fi

# If the run already produced samples, just score them. Otherwise generate first.
if [ ! -f "${RUN}samples.npz" ]; then
    [ -n "$MODEL" ] || { echo "FATAL: no samples.npz and no checkpoint given"; exit 1; }
    echo "### GENERATING ###"
    $PYTHON -u gen_eval_eeg.py \
        --root "$ROOT/" \
        --dataset EEG \
        --model_path "$MODEL" \
        --eeg_signals_path "$DATASET" \
        --splits_path "$SPLITS" \
        --imagenet_path "$IMAGENET" \
        --config_patch "$ROOT/pretrains/models/config15.yaml"
    RUN=$(ls -td "$ROOT"/results/eval/*/ | head -1)
fi

echo "### EVALUATION ###"
$PYTHON eval_report.py --samples "${RUN}samples.npz" --output "${RUN}report"

echo "### REPORT ###"
cat "${RUN}report.md"
