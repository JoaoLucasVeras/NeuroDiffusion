"""
Is the imagined-shapes signal brain activity, or eye movement?

Imagined shapes (FVI: circle, pentagram, square) are the one imagery result that holds up
on the Gao data (jobs 87790, 88171, 88200). Imagining a pentagram can make the eyes trace
its outline, and eye movements are large, shape-dependent electrical signals strongest at
the front of the head. So before anything is built on the shapes result, this asks where
on the head and when in the 4 s it comes from, and whether eye signals alone carry it.

Everything is fixed before any result is seen, and matches job 87790 (the cross-day
shapes result): train on one session, test on the other, both orders, mean of the two;
features standardised within each session (no labels); logistic regression, l2 = 0.01;
the null shuffles training labels in both orders.

Conditions, all on the imagery window (0-4 s after the cue, black screen):
  alpha_all        8-13 Hz log power, all 32 channels. The reference: should replicate 87790.
  alpha_posterior  the same, back of the head only (parietal and occipital, 10 channels).
                   Visual cortex: where imagery is expected.
  alpha_frontal    the same, front of the head only (Fp and F rows, 8 channels). Nearest
                   the eyes.
  alpha_middle     the same, the 14 central and temporal channels in between.
  alpha_t0..t3     all channels, one second at a time (0-1, 1-2, 2-3, 3-4 s). A tracing eye
                   moves throughout; a cue-locked brain response is often early.
  eye_proxy        eye signals only: horizontal (F7 - F8) and vertical (mean of Fpz, Fp1,
                   Fp2) under 4 Hz, averaged in 250 ms bins. If these alone decode shapes
                   as well as alpha_all does, the shapes result may be the eyes.

Reading it: a brain signal predicts alpha_posterior >= alpha_frontal, and eye_proxy near
chance. An eye signal predicts the reverse. The groups differ in size (10, 8, 14
channels), so compare them with that in mind.

--data ../datasets/gao_ica runs the alpha conditions on the authors' ICA-cleaned data,
from which eye components were removed. eye_proxy is skipped there: that data is
band-passed at 4-80 Hz, so the slow eye signal is removed by construction.

Usage
-----
    python gao_eyecheck.py --subject 1 --task FVI --permute 200 --json out.json
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from classical_baseline import bandpass, feat_bandpower
from gao_baseline import TASK_CLASSES, score, zscore

GROUPS = {
    "posterior": ["Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2"],
    "frontal": ["Fpz", "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"],
    "middle": ["FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T7", "T8",
               "CP3", "CP4", "TP7", "TP8"],
}
IMAGERY = (0.0, 4.0)
SECONDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def cut(fs, tmin, w):
    return int(round((w[0] - tmin) * fs)), int(round((w[1] - tmin) * fs))


def alpha_epochs(trials, fs, tmin):
    """8-13 Hz on the whole stored trial, so the cut windows carry no filter edges."""
    out = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        x = x - x.mean(axis=-1, keepdims=True)
        out.append(bandpass(x, 8.0, 13.0, fs=fs))
    return np.stack(out)


def eye_features(trials, channels, fs, tmin):
    """Horizontal and vertical eye signals under 4 Hz, 250 ms bins over 0-4 s."""
    ix = {c: channels.index(c) for c in ("F7", "F8", "Fpz", "Fp1", "Fp2")}
    b, a = butter(4, 4.0 / (fs / 2.0), btype="low")
    a0, a1 = cut(fs, tmin, IMAGERY)
    step = int(0.25 * fs)
    F = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        h = x[ix["F7"]] - x[ix["F8"]]
        v = (x[ix["Fpz"]] + x[ix["Fp1"]] + x[ix["Fp2"]]) / 3.0
        sig = filtfilt(b, a, np.stack([h, v]), axis=-1)[:, a0:a1]
        sig = sig - sig[:, :step].mean(axis=-1, keepdims=True)    # relative to the first bin
        n = sig.shape[-1] // step
        F.append(sig[:, :n * step].reshape(2, n, step).mean(-1).ravel())
    return np.stack(F)


def session_features(trials, channels, fs, tmin, with_eyes):
    A = alpha_epochs(trials, fs, tmin)
    a0, a1 = cut(fs, tmin, IMAGERY)
    full = feat_bandpower(list(A[:, :, a0:a1]))
    F = {"alpha_all": full}
    for g, names in GROUPS.items():
        F["alpha_" + g] = full[:, [channels.index(c) for c in names]]
    for k, w in enumerate(SECONDS):
        b0, b1 = cut(fs, tmin, w)
        F["alpha_t%d" % k] = feat_bandpower(list(A[:, :, b0:b1]))
    if with_eyes:
        F["eye_proxy"] = eye_features(trials, channels, fs, tmin)
    return {k: zscore(v) for k, v in F.items()}


def main():
    ap = argparse.ArgumentParser(description="Shapes imagery: brain or eyes?")
    ap.add_argument("--data", default="../datasets/gao")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--task", default="FVI", choices=sorted(TASK_CLASSES))
    ap.add_argument("--permute", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = torch.load(os.path.join(args.data, "gao_sub-%02d.pt" % args.subject), map_location="cpu")
    trials, fs, tmin, channels = d["dataset"], d["fs"], d["tmin"], d["channels"]
    variant = d.get("variant") or "ours"
    sessions = sorted({t["session"] for t in trials})
    if len(sessions) < 2:
        print("sub-%02d has one session; cross-session test not possible, skipping" % args.subject)
        return
    classes = TASK_CLASSES[args.task]
    n = len(classes)
    s1, s2 = sessions[:2]
    by_ses = {s: [t for t in trials if t["task"] == args.task and t["session"] == s] for s in (s1, s2)}
    missing = [c for c in classes if any(not any(t["class"] == c for t in tt) for tt in by_ses.values())]
    res = {"subject": args.subject, "task": args.task, "data": variant, "chance": 1.0 / n,
           "l2": 0.01, "conditions": {}}
    if missing:
        res["skipped"] = "absent from one session: " + ", ".join(missing)
        print("sub-%02d %s skipped: %s" % (args.subject, args.task, res["skipped"]))
    else:
        print("sub-%02d | %s %d-way | data %s | train one session, test the other, both orders"
              % (args.subject, args.task, n, variant))
        F = {s: session_features(by_ses[s], channels, fs, tmin, variant != "ica") for s in (s1, s2)}
        y = {s: np.array([classes.index(t["class"]) for t in by_ses[s]]) for s in (s1, s2)}
        rng = np.random.default_rng(args.subject)
        pairs = [(s1, s2), (s2, s1)]
        for cond in F[s1]:
            def run(labels):
                return float(np.mean([score(F[a][cond], labels[a], F[b][cond], y[b], n)
                                      for a, b in pairs]))
            acc = run(y)
            null = [run({s: rng.permutation(y[s]) for s in y}) for _ in range(args.permute)]
            p = (np.sum(np.array(null) >= acc) + 1.0) / (len(null) + 1.0) if null else None
            res["conditions"][cond] = {"accuracy": acc, "x_chance": acc * n, "n_features":
                                       int(F[s1][cond].shape[1]), "null_mean":
                                       float(np.mean(null)) if null else None, "p_value": p}
            print("  %-16s acc %.3f (%.2fx)  p %s" % (cond, acc, acc * n,
                                                    "%.4f" % p if p is not None else "-"))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
