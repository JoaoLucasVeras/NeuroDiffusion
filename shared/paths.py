"""
Where the repository and its data live, for every script in shared/ and studies/.

Each script starts with the same lines, which find this file by walking up from the
script's own folder, so a script runs the same from any folder:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                                if (p / "shared" / "paths.py").exists()) / "shared"))
    import paths

Importing it also puts every shared/ subfolder on the import path, so
`from classical_baseline import logreg` works from anywhere. Default data and output
locations are built from paths.DATASETS and paths.RESULTS, never from "../".
"""
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent
ROOT = SHARED.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"

for _sub in ("data", "decoding", "generation", "evaluation"):
    _p = str(SHARED / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
