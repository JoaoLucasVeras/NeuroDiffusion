"""
Gao reproduction, stage 1: write the files the authors' EEGNet.py expects.

Their Preprocess.py saves, per recording (one person, one session, one task), a
<name>_data.npy of shape (trials, 32, 1000) and a <name>_label.npy of event codes, with
the trials grouped by class in the order dog, bird, fish (its generate_data_label walks
event_id in that order). This writes exactly that from datasets/gao_ica, which was made
by running their own preprocessing functions on the raw recordings (gao_ica_prep.py).

For the animals task their events and ours are the same trials with the same codes, so
these files are what their pipeline would have produced. Stored as float32: their script
converts to float32 on loading (torch.Tensor), so the network sees identical numbers.

Two copies, one folder per person so the cluster can run people in parallel:
  real/sub-XX/        their labels
  scrambled/sub-XX/   the same recordings (data linked, not copied) with the labels
                      shuffled within each recording, seed fixed per person and session.
                      The control: what their procedure reports when no label
                      information exists.

Usage
-----
    python gao_repro_export.py --data datasets/gao_ica --out datasets/gao_repro --subject 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "shared" / "paths.py").exists()) / "shared"))
import paths  # noqa: E402  shared/ on the import path; paths.ROOT is the repo root
import argparse
import glob
import os

import numpy as np
import torch

CODES = {"dog": 1, "bird": 2, "fish": 3}          # task-AVI_events.json, their event_id


def main():
    ap = argparse.ArgumentParser(description="Export Gao animals data for their EEGNet.py")
    ap.add_argument("--data", default=str(paths.DATASETS / "gao_ica"))
    ap.add_argument("--out", default=str(paths.DATASETS / "gao_repro"))
    ap.add_argument("--task", default="AVI")
    ap.add_argument("--subject", type=int, default=None, help="one person only (default: all)")
    args = ap.parse_args()

    pattern = "gao_sub-%02d.pt" % args.subject if args.subject else "gao_sub-*.pt"
    for path in sorted(glob.glob(os.path.join(args.data, pattern))):
        d = torch.load(path, map_location="cpu")
        if d.get("variant") != "ica":
            raise SystemExit("FATAL: %s is not the ICA-cleaned variant" % path)
        sub = int(os.path.basename(path)[8:10])
        trials = [t for t in d["dataset"] if t["task"] == args.task]
        for ses in sorted({t["session"] for t in trials}):
            tt = [t for t in trials if t["session"] == ses]
            ordered = [t for c in CODES for t in sorted(tt, key=lambda t: t["trial"]) if t["class"] == c]
            X = np.stack([t["eeg"].numpy() for t in ordered]).astype(np.float32)
            y = np.array([CODES[t["class"]] for t in ordered])
            name = "sub-%02d_ses-%02d_task-%s_eeg" % (sub, ses, args.task)
            rng = np.random.default_rng(1000 * sub + ses)
            real = os.path.join(args.out, "real", "sub-%02d" % sub)
            scr = os.path.join(args.out, "scrambled", "sub-%02d" % sub)
            for folder in (real, scr):
                os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(real, name + "_data.npy"), X)
            np.save(os.path.join(real, name + "_label.npy"), y)
            link = os.path.join(scr, name + "_data.npy")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(os.path.join(real, name + "_data.npy")), link)
            np.save(os.path.join(scr, name + "_label.npy"), rng.permutation(y))
            print("%s: %s, classes %s" % (name, X.shape, dict(zip(*np.unique(y, return_counts=True)))))


if __name__ == "__main__":
    main()
