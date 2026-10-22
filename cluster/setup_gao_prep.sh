#!/bin/bash
# One-time setup for reproducing the Gao et al. preprocessing (option B): a SEPARATE
# conda environment with MNE, mne-icalabel and pyprep, so the main neurodiffusion
# environment is never touched. Run on coe-hpc1 (internet), not through Slurm.
set -euo pipefail
CONDA=/opt/ohpc/pub/apps/anaconda/3.9/bin/conda
ENV=$HOME/.conda/envs/gaoprep
if [ ! -x "$ENV/bin/python" ]; then
    $CONDA create -y -p "$ENV" python=3.10
fi
"$ENV/bin/python" -m pip install --quiet "mne==1.8.0" "mne-icalabel==0.7.0" "pyprep==0.4.3" \
    "onnxruntime" "python-picard" "scipy" "numpy<2" "torch==2.3.1" --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/python" -c "import mne, mne_icalabel, pyprep, picard; print('ok: mne', mne.__version__, '| mne-icalabel', mne_icalabel.__version__, '| pyprep', pyprep.__version__)"
