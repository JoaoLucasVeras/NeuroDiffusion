# Shared toolbox

Code that more than one study uses. **Import it, don't copy it**: if you need a change,
add an option, so every study keeps getting the same, tested behaviour.

| Folder | Script | Does |
|---|---|---|
| `data/` | `prepare_gao_data.py` | Converts the Gao et al. 2026 dataset (22 people) to `datasets/gao/gao_sub-XX.pt`, verifying every event |
| | `gao_ica_prep.py` | The same, cleaned with the Gao authors' own ICA preprocessing → `datasets/gao_ica/` |
| | `prepare_shimizu_data.py`, `make_splits.py` | Convert the first dataset (Shimizu & Srinivasan 2022) and build leakage-free splits |
| | `check_data.py` | Pre-flight check of a dataset; run it before any job on that data |
| | `fetch_stimuli.py`, `download_stimuli.py` | Get the first dataset's ImageNet stimulus pictures |
| `decoding/` | `classical_baseline.py` | **The core toolbox:** band-pass filter, band power, Riemannian features, logistic regression, a small EEGNet |
| | `gao_baseline.py` | Gao data loader, splits (across days, within a day, seeing → imagining), features and shuffled-label null |
| `generation/` | `generate_imagevar.py` | **Generator B:** Stable Diffusion Image Variations, drawing a picture from 768 CLIP numbers. Arms: brain / true picture / average |
| | `precache_models.py` | Downloads models on the login node (compute nodes have no internet) |
| `evaluation/` | `eval_report.py` | Scores generated images against no-information baselines |
| | `clip_metrics.py`, `eval_metrics.py`, `compare_reports.py` | The metric functions; side-by-side comparison of runs |
| `third_party/` | `gao_EEGNet.py`, `gao_preprocess.py` | The Gao authors' scripts, byte for byte (CC BY 4.0). **Do not edit** |
| (top level) | `paths.py` | Where the repository, `datasets/` and `results/` are |

## Using it from your own script

Put these lines at the top of any script in `studies/` or `research/`. They find `shared/`
from the script's own location, so the script runs from any folder:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "shared" / "paths.py").exists()) / "shared"))
import paths  # noqa: E402  shared/ on the import path; paths.ROOT is the repo root

from classical_baseline import bandpass, logreg
from gao_baseline import TASK_CLASSES

DATA = paths.DATASETS / "gao"
```
