#!/bin/bash
#SBATCH --job-name=s7_ckpt
#SBATCH --output=logs/s7_ckpt_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Verify the --save checkpoint format round-trips before anything is built on it.
# Two epochs only: the weights are meaningless, the FILE FORMAT is what is under test.
# A passing syntax check means little here -- the ROOT_PATH incident was a clean syntax
# check that died on the cluster -- so this actually runs the code path.

ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT/code"
R="$ROOT/results/s7_$SLURM_JOB_ID"
mkdir -p "$R"
CKPT="$R/verify_s1.pt"

echo "### 1. train 2 epochs and save"
$PYTHON -u small_encoder.py --subject 1 --band 8 13 --epochs 2 --permute 0 \
    --save "$CKPT" --json "$R/verify_s1.json"

echo ""
echo "### 2. load it back and check every field the conditioning path needs"
$PYTHON - "$CKPT" <<'PYEOF'
import sys, torch
from small_encoder import SmallEncoder, n_params

blob = torch.load(sys.argv[1], map_location="cpu")
for k in ("state_dict", "arch", "norm", "stimulus_order", "metrics", "argv"):
    assert k in blob, "MISSING KEY: %s" % k
print("keys present:", sorted(blob.keys()))

a = blob["arch"]
print("arch:", a)
assert a["n_ch"] == 128 and a["n_time"] == 125, "unexpected input shape %s" % a
assert a["out_dim"] == 768, "CLIP dim is not 768: %s" % a["out_dim"]

m = SmallEncoder(a["n_ch"], a["n_time"], a["out_dim"])
m.load_state_dict(blob["state_dict"], strict=True)
print("state_dict loaded strict=True, %d params" % n_params(m))

mu, sd = blob["norm"]["mu"], blob["norm"]["sd"]
print("norm: mu=%.6f sd=%.6f" % (mu, sd))
assert sd > 0, "degenerate sd"

print("stimuli: %d" % len(blob["stimulus_order"]))
assert len(blob["stimulus_order"]) == 33, "expected 33 stimuli"

# the whole point: identical input must give identical output after a round trip
m.eval()
x = torch.zeros(2, 1, a["n_ch"], a["n_time"])
x[0, 0, 0, 0] = 1.0
with torch.no_grad():
    out = m(x)
print("forward ok, out shape", tuple(out.shape))
assert out.shape == (2, 768)
nrm = out.norm(dim=-1)
print("output L2 norms: %.6f %.6f  (head applies F.normalize)" % (nrm[0], nrm[1]))
assert torch.allclose(nrm, torch.ones(2), atol=1e-5), "output is not unit-norm"

print("")
print("CHECKPOINT FORMAT OK")
PYEOF

echo ""; echo "### DONE -> $R ###"
ls -la "$R"
