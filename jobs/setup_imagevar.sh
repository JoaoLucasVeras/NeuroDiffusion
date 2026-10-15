#!/bin/bash
# One-time setup for generator B (code/generate_imagevar.py).
# Run on coe-hpc1, which has internet; the GPU nodes behind coe-hpc3 do not.
# This is not a Slurm job.
set -euo pipefail
ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
DEST="$ROOT/pretrains/sd-image-variations"

# 1. diffusers on its own. Everything it depends on (transformers, huggingface_hub,
#    safetensors, Pillow) is already in this environment, so --no-deps adds exactly one
#    package and changes none of the existing ones.
echo "### installing diffusers 0.31.0 (no dependency changes)"
$PYTHON -m pip install --no-deps "diffusers==0.31.0"
$PYTHON -c "import torch, transformers, diffusers; from diffusers import StableDiffusionImageVariationPipeline; print('ok: torch', torch.__version__, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__)"

# 2. The model, revision v2.0 (the diffusers port). About 5.0 GB without the safety
#    checker, which only blanks images and is not needed for research output.
echo "### downloading lambdalabs/sd-image-variations-diffusers v2.0 -> $DEST"
$PYTHON - "$DEST" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("lambdalabs/sd-image-variations-diffusers", revision="v2.0",
                  local_dir=sys.argv[1], ignore_patterns=["safety_checker/*"])
PYEOF
du -sh "$DEST"
echo "### done. Generator B is ready: see jobs/s10_imagevar.sh"
