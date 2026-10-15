#!/bin/bash
# Download the Gao et al. (2026) imagery dataset: 22 subject zips (11.9 GB) and the ten
# stimulus images. Run on coe-hpc1, which has internet; not a Slurm job. Each file is
# checked against the size figshare reports, and files already complete are skipped,
# so the script can simply be re-run after an interruption.
# Then convert with: sbatch jobs/s11_prepare_gao.sh
set -euo pipefail
ROOT="${NEURODIFFUSION_ROOT:-$HOME/NeuroDiffusion}"
PYTHON="${NEURODIFFUSION_PYTHON:-$HOME/.conda/envs/neurodiffusion/bin/python}"
RAW="$ROOT/datasets/gao_raw"
mkdir -p "$RAW" "$ROOT/datasets/gao/stimuli"

$PYTHON - "$RAW" "$ROOT/datasets/gao/stimuli" <<'PYEOF'
import json, os, sys, time, urllib.request
raw, stim = sys.argv[1], sys.argv[2]
meta = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/30227503", timeout=60))
print("figshare record: %s, version %s, %s" % (meta["title"], meta["version"], meta["license"]["name"]))
files = [f for f in meta["files"] if f["name"].endswith(".zip") or f["name"].endswith(".jpg")]
for f in sorted(files, key=lambda f: f["name"]):
    dest = os.path.join(stim if f["name"].endswith(".jpg") else raw, f["name"])
    if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
        print("  have  %-22s" % f["name"]); continue
    for attempt in range(3):
        t = time.time()
        urllib.request.urlretrieve(f["download_url"], dest + ".part")
        if os.path.getsize(dest + ".part") == f["size"]:
            os.replace(dest + ".part", dest)
            print("  got   %-22s %7.1f MB in %.0f s" % (f["name"], f["size"] / 1e6, time.time() - t)); break
        print("  size mismatch on %s, retrying" % f["name"])
    else:
        sys.exit("FATAL: could not download %s intact" % f["name"])
print("all %d files present and complete" % len(files))
PYEOF
du -sh "$RAW"
