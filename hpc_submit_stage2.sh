#!/bin/bash
#SBATCH --job-name=stage2_train
#SBATCH --output=/home/015555345/NeuroDiffusion/logs/stage2_%j.out
#SBATCH --error=/home/015555345/NeuroDiffusion/logs/stage2_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Stage 2: fine-tune the latent diffusion model on EEG conditioning.
#
# This stage DOES require the stimulus images. The preflight below aborts if any
# are missing -- previously a missing file silently became a black square, which
# made every training target identical and the whole run worthless.

set -euo pipefail

ROOT=/home/015555345/NeuroDiffusion
export PYTHON=/home/015555345/.conda/envs/neurodiffusion/bin/python
export PATH=$(dirname "$PYTHON"):$PATH

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "$ROOT/logs" "$ROOT/results/generation"
cd "$ROOT/code"

# --- dataset selection -------------------------------------------------------
# imagination : 3200 trials, 40 stimuli (1 per class), 20 windows per recording.
#               Use protocol=subject (leave-one-subject-out). Class-level metrics only.
# visual      : 7987 trials, 2000 distinct stimuli. Use protocol=image.
EXPERIMENT=${EXPERIMENT:-imagination}
PROTOCOL=${PROTOCOL:-subject}
# "_avail" splits exclude trials whose stimulus image could not be sourced.
# Set SPLIT_SUFFIX="" once you have a full ILSVRC2012 copy.
SPLIT_SUFFIX=${SPLIT_SUFFIX-_avail}

# Tunables. BATCH_SIZE=8 fits the 48h walltime with room to spare; if it OOMs on the
# 40GB A100, drop to 4 (still ~21h for visual) then 2. The previous run used 2, which
# would NOT have finished the visual set in time. GEN_LIMIT caps end-of-training
# sampling; that happens after the model is already checkpointed, so capping it only
# trades test-set size for the job finishing inside the walltime.
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCH=${NUM_EPOCH:-200}
DDIM_STEPS=${DDIM_STEPS:-250}
GEN_LIMIT=${GEN_LIMIT:-200}
# torch 1.12.1 ships no CUDA bfloat16 kernel for upsample_nearest2d, which the VAE
# decoder hits during PLMS sampling ("not implemented for 'BFloat16'"). fp16 AMP is
# supported and fast on A100 tensor cores. Fall back to 32 if fp16 produces NaNs.
PRECISION=${PRECISION:-16}

DATASET=$ROOT/datasets/${EXPERIMENT}_5_95_std.pth
SPLITS=$ROOT/datasets/${EXPERIMENT}_5_95_std_splits_${PROTOCOL}${SPLIT_SUFFIX}.pth
IMAGENET=$ROOT/datasets/imageNet_images

LATEST_PRETRAIN=$(ls -td "$ROOT"/results/eeg_pretrain/*/ | head -1)
CHECKPOINT="${LATEST_PRETRAIN}checkpoints/checkpoint.pth"

echo "### CONFIG ###"
echo "  experiment : $EXPERIMENT"
echo "  protocol   : $PROTOCOL"
echo "  dataset    : $DATASET"
echo "  splits     : $SPLITS"
echo "  stage1 ckpt: $CHECKPOINT"
echo "  batch/epoch: $BATCH_SIZE / $NUM_EPOCH   ddim: $DDIM_STEPS   gen_limit: $GEN_LIMIT   precision: $PRECISION"
[ -f "$CHECKPOINT" ] || { echo "FATAL: no Stage 1 checkpoint at $CHECKPOINT"; exit 1; }

# --- GPU sanity check -------------------------------------------------------
# History: the `training` branch carried CUDA_VISIBLE_DEVICES=1 because GPU 0 on one
# node was broken. Hardcoding an index is fragile -- it silently targets the wrong
# device (or none at all) depending on how SLURM scopes the allocation. Instead, let
# SLURM assign the GPU and verify here that it actually works, so a bad device fails
# in seconds rather than after hours of queueing.
echo "### GPU CHECK ###"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 || true
$PYTHON - <<'PYGPU'
import sys
try:
    import torch
except Exception as e:
    sys.exit("FATAL: cannot import torch (%s)" % e)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False -- no usable GPU in this allocation")
n = torch.cuda.device_count()
print("  visible GPUs: %d" % n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print("   [%d] %s  %.1f GB" % (i, p.name, p.total_memory / 1e9))
try:
    x = torch.randn(2048, 2048, device="cuda")
    float((x @ x).sum())
    torch.cuda.synchronize()
    print("  matmul on cuda:0 OK")
except Exception as e:
    sys.exit("FATAL: GPU present but unusable (%s: %s)" % (type(e).__name__, e))
PYGPU

echo "### PREFLIGHT ###"
$PYTHON check_data.py --dataset "$DATASET" --splits "$SPLITS" --imagenet_path "$IMAGENET"

echo "### STAGE 2: LDM fine-tuning ###"
$PYTHON -u eeg_ldm.py \
    --root_path "$ROOT/" \
    --dataset EEG \
    --batch_size "$BATCH_SIZE" \
    --num_epoch "$NUM_EPOCH" \
    --ddim_steps "$DDIM_STEPS" \
    --generate_limit "$GEN_LIMIT" \
    --lr 5.3e-5 \
    --precision "$PRECISION" \
    --eeg_signals_path "$DATASET" \
    --splits_path "$SPLITS" \
    --imagenet_path "$IMAGENET" \
    --pretrain_gm_path "$ROOT/pretrains/" \
    --pretrain_mbm_path "$CHECKPOINT" \
    --use_time_cond True \
    --clip_tune True \
    --strict_images True

RUN=$(ls -td "$ROOT"/results/generation/*/ | head -1)
echo "### DONE. Run directory: $RUN ###"

# Score the generations against the noise and shuffled-pairing baselines.
if [ -f "${RUN}samples.npz" ]; then
    echo "### EVALUATION ###"
    $PYTHON eval_report.py --samples "${RUN}samples.npz" --output "${RUN}report"
fi
