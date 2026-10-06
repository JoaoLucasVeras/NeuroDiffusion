"""
Classical baseline on the Gao et al. (2026) imagery dataset: band power plus logistic
regression, the same instrument that found the signal in the Shimizu data.

What is fixed in advance, before any Gao result was seen
--------------------------------------------------------
- Split: train on session 1, test on session 2. The headset comes off and goes back on
  between them, so a recording fingerprint cannot carry over. Subjects 9 and 10 have one
  session and are skipped in this mode.
- Decoding stays within a task: AVI is 3-way, FVI 3-way, OVI 4-way. The tasks were
  recorded as separate runs, so a 10-way classifier could tell runs apart instead of
  pictures.
- Regularisation l2 = 0.01, the middle of classical_baseline's grid. No selection, so
  nothing to correct the null for.
- Two windows from the same trials. imagery: 0 to 4 s after the marker, black screen.
  perception: -5.5 to -1.5 s, the picture on screen. Perception is the positive control:
  if it decodes at chance, the method is broken rather than the brain signal.
- Two feature sets. power: log band power per channel, the imagery instrument.
  waveform: the 1-15 Hz signal averaged in 100 ms bins per channel. Seeing a picture
  mostly produces a time-locked waveform, which band power discards by design, so the
  perception control is only meaningful with waveform features. Both run on both windows.

Trials here are separate events, so the permutation null shuffles labels across training
trials directly. That is the correct null for this dataset; it would not have been for
Shimizu, where windows of one recording shared a label.

Usage
-----
    python gao_baseline.py --subject 1 --band 8 13 --permute 200 --json out.json
"""
import argparse
import json
import os

import numpy as np
import torch

from classical_baseline import bandpass, feat_bandpower, logreg

TASK_CLASSES = {"AVI": ["dog", "bird", "fish"],
                "FVI": ["circle", "pentagram", "square"],
                "OVI": ["scissor", "watch", "cup", "chair"]}
WINDOWS = {"imagery": (0.0, 4.0), "perception": (-5.5, -1.5)}
L2 = 0.01


def load(path):
    d = torch.load(path, map_location="cpu")
    return d["dataset"], d["fs"], d["tmin"]


def features(trials, window, band, fs, tmin, kind="power"):
    a = int(round((window[0] - tmin) * fs))
    b = int(round((window[1] - tmin) * fs))
    if kind == "waveform":
        band = (1.0, 15.0)
    X = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        x = x - x.mean(axis=-1, keepdims=True)
        if band is not None:
            x = bandpass(x, band[0], band[1], fs=fs)        # filter the whole 12 s first
        X.append(x[:, a:b])                                  # then cut, so no edge effects
    if kind == "power":
        return feat_bandpower(X)
    step = int(0.1 * fs)                                     # 100 ms bins
    X = np.stack(X)
    n = X.shape[-1] // step
    return X[:, :, :n * step].reshape(len(X), X.shape[1], n, step).mean(-1).reshape(len(X), -1)


def score(Ftr, ytr, Fte, yte, n):
    p = logreg(Ftr, ytr, Fte, n, L2)
    return float((p.argmax(1) == yte).mean())


def main():
    ap = argparse.ArgumentParser(description="Gao et al. classical baseline")
    ap.add_argument("--data", default="../datasets/gao")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 13.0], help="0 0 = broadband")
    ap.add_argument("--permute", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    trials, fs, tmin = load(os.path.join(args.data, "gao_sub-%02d.pt" % args.subject))
    band = None if args.band[0] <= 0 else tuple(args.band)
    sessions = sorted({t["session"] for t in trials})
    if len(sessions) < 2:
        print("sub-%02d has only session(s) %s; cross-session test not possible, skipping"
              % (args.subject, sessions))
        return
    print("sub-%02d | band %s | train session %d, test session %d | l2 %g fixed in advance"
          % (args.subject, band, sessions[0], sessions[1], L2))

    rng = np.random.default_rng(args.subject)
    res = {"subject": args.subject, "band": band, "l2": L2, "tasks": {}}
    for task, classes in TASK_CLASSES.items():
        tr = [t for t in trials if t["task"] == task and t["session"] == sessions[0]]
        te = [t for t in trials if t["task"] == task and t["session"] == sessions[1]]
        ytr = np.array([classes.index(t["class"]) for t in tr])
        yte = np.array([classes.index(t["class"]) for t in te])
        n = len(classes)
        res["tasks"][task] = {"chance": 1.0 / n, "n_train": len(tr), "n_test": len(te)}
        for kind in ("power", "waveform"):
            for wname, w in WINDOWS.items():
                Ftr = features(tr, w, band, fs, tmin, kind)
                Fte = features(te, w, band, fs, tmin, kind)
                acc = score(Ftr, ytr, Fte, yte, n)
                null = [score(Ftr, rng.permutation(ytr), Fte, yte, n) for _ in range(args.permute)]
                pval = (np.sum(np.array(null) >= acc) + 1.0) / (len(null) + 1.0) if null else None
                res["tasks"][task]["%s_%s" % (kind, wname)] = {
                    "accuracy": acc, "x_chance": acc * n,
                    "null_mean": float(np.mean(null)) if null else None, "p_value": pval}
                print("  %s %-8s %-10s %d-way  acc %.3f  (%.2fx chance)  p %s"
                      % (task, kind, wname, n, acc, acc * n,
                         "%.4f" % pval if pval is not None else "-"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
