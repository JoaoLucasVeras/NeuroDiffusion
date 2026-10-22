"""
Gao reproduction, stage 1: run the authors' EEGNet.py exactly as written.

third_party/gao_EEGNet.py is their file, byte-for-byte. Its main block reads a folder
hard-coded as r"..\\ani_s1_signal_sub_newset_del_ch". This runs that same source as
__main__ with that one string replaced by --folder, and changes nothing else: same
1,500 epochs, same 5-fold split with seed 42, same scoring. The replacement is checked
to have happened exactly once, and the source's SHA-256 is printed so the log shows
which file ran.

The script prints, per recording, "Average Accuracy" (mean over folds of each fold's
best test accuracy over epochs) and "Overall Best Accuracy" (the single best). Both are
collected afterwards from the log by summarise in gao_repro_summary.py.

Usage
-----
    python gao_repro_run.py --folder datasets/gao_repro/real/sub-01
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "shared" / "paths.py").exists()) / "shared"))
import paths  # noqa: E402  shared/ on the import path; paths.ROOT is the repo root
import argparse
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = str(paths.SHARED / "third_party" / "gao_EEGNet.py")
FOLDER_LITERAL = 'r"..\\ani_s1_signal_sub_newset_del_ch"'


def main():
    ap = argparse.ArgumentParser(description="Run the authors' EEGNet.py unchanged")
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    raw = open(SOURCE, "rb").read()
    print("source %s sha256 %s" % (SOURCE, hashlib.sha256(raw).hexdigest()))
    src = raw.decode("utf-8")
    if src.count(FOLDER_LITERAL) != 1:
        raise SystemExit("FATAL: expected the data folder literal exactly once in their script")
    src = src.replace(FOLDER_LITERAL, repr(os.path.abspath(args.folder)))
    print("data folder %s" % os.path.abspath(args.folder), flush=True)
    exec(compile(src, SOURCE, "exec"), {"__name__": "__main__", "__file__": SOURCE})


if __name__ == "__main__":
    main()
