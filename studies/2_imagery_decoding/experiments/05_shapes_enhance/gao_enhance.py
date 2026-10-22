"""
Enhancement round (B): can imagined shapes be decoded better than 37%?

One batch of stronger decoders on the Gao imagery data, every method, setting and target
fixed before any result was seen (agreed 2026-10-19). Imagery window 0-4 s after the cue.
Unless stated, every method trains on one session and tests on the other, both orders,
mean of the two, with features standardised within each session (no labels used), as in
the cross-day shapes result (job 87790). Logistic regression, l2 = 0.01. Null: training
labels shuffled, 200 times, the whole method repeated each time.

Three-way methods (chance 33%):
  alpha           8-13 Hz log power per channel. The reference: should replicate 87790.
  multiband       theta 4-8, alpha 8-13 and beta 13-30 Hz log power, concatenated.
  riemann         covariance of 4-30 Hz activity, mapped to the tangent space at each
                  session's own mean covariance (unsupervised re-centring, the standard
                  remedy for session shifts in Riemannian decoding).
  multiband_time  multiband power in each 1-s slice of the window, concatenated.
  pooled          alpha, but training on the other session AND the first 80% of this
                  session (recording order), testing on this session's last 20%; both
                  sessions. More training data, still no test trial before a training one.

Two-way (chance 50%): alpha, multiband and riemann on each pair of shapes (circle vs
pentagram, circle vs square, pentagram vs square); the figure of merit is the mean over
the three pairs, and its null shuffles within each pair.

Time-matched transfer: alpha power in four 1-s slices of the seeing window (-5.5 to -1.5 s)
and four of the imagining window; train on one seeing slice in one session, test on one
imagining slice in the other; 16 slice pairs. Its p-value uses the maximum over the 16
pairs under the null (100 shuffles), so looking at the best pair is paid for.

Decision rule, set in advance: the round succeeds if some method's mean accuracy across
people reaches 45% three-way, or 65% two-way (mean over pairs), AND its Fisher-combined
p-value across people is below 0.05 divided by the number of methods in that family
(5 three-way, 3 two-way). Then the shapes-to-image demo is built; otherwise effort moves
to designing our own recordings.

Usage
-----
    python gao_enhance.py --subject 1 --permute 200 --json out.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "shared" / "paths.py").exists()) / "shared"))
import paths  # noqa: E402  shared/ on the import path; paths.ROOT is the repo root
import argparse
import json
import os

import numpy as np
import torch

from classical_baseline import _cov, _logm_spd, _tangent, bandpass, feat_bandpower
from gao_baseline import score, zscore

CLASSES = ["circle", "pentagram", "square"]
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
IMAGERY = (0.0, 4.0)
SEEING = (-5.5, -1.5)


def cut(fs, tmin, a, b):
    return int(round((a - tmin) * fs)), int(round((b - tmin) * fs))


def filtered(trials, lo, hi, fs):
    out = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        out.append(bandpass(x - x.mean(axis=-1, keepdims=True), lo, hi, fs=fs))
    return np.stack(out)


def power(X, fs, tmin, a, b):
    i, j = cut(fs, tmin, a, b)
    return feat_bandpower(list(X[:, :, i:j]))


def riemann_session(X, fs, tmin):
    """Tangent vectors at this session's own log-Euclidean mean covariance."""
    i, j = cut(fs, tmin, *IMAGERY)
    covs = [_cov(x[:, i:j]) for x in X]
    ref = np.mean([_logm_spd(c) for c in covs], axis=0)
    w, v = np.linalg.eigh(ref)
    return _tangent(covs, (v * np.exp(w)) @ v.T)


def session_features(trials, fs, tmin):
    B = {k: filtered(trials, lo, hi, fs) for k, (lo, hi) in BANDS.items()}
    wide = filtered(trials, 4.0, 30.0, fs)
    F = {"alpha": power(B["alpha"], fs, tmin, *IMAGERY)}
    F["multiband"] = np.hstack([power(B[k], fs, tmin, *IMAGERY) for k in BANDS])
    F["riemann"] = riemann_session(wide, fs, tmin)
    F["multiband_time"] = np.hstack([power(B[k], fs, tmin, s, s + 1.0)
                                     for s in (0.0, 1.0, 2.0, 3.0) for k in BANDS])
    T = {"see%d" % n: power(B["alpha"], fs, tmin, SEEING[0] + n, SEEING[0] + n + 1.0) for n in range(4)}
    T.update({"img%d" % n: power(B["alpha"], fs, tmin, float(n), n + 1.0) for n in range(4)})
    return {k: zscore(v) for k, v in F.items()}, {k: zscore(v) for k, v in T.items()}


def p_value(obs, null):
    return float((np.sum(np.array(null) >= obs) + 1.0) / (len(null) + 1.0)) if null else None


def main():
    ap = argparse.ArgumentParser(description="Enhancement round on imagined shapes")
    ap.add_argument("--data", default=str(paths.DATASETS / "gao"))
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--permute", type=int, default=200)
    ap.add_argument("--permute_transfer", type=int, default=100)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = torch.load(os.path.join(args.data, "gao_sub-%02d.pt" % args.subject), map_location="cpu")
    trials, fs, tmin = d["dataset"], d["fs"], d["tmin"]
    sessions = sorted({t["session"] for t in trials if t["task"] == "FVI"})
    res = {"subject": args.subject, "task": "FVI", "l2": 0.01, "three_way": {}, "two_way": {}}
    if len(sessions) < 2:
        print("sub-%02d has one session; cross-session test not possible, skipping" % args.subject)
        if args.json:
            json.dump(dict(res, skipped="one session"), open(args.json, "w"), indent=2)
        return
    s1, s2 = sessions[:2]
    by = {s: sorted([t for t in trials if t["task"] == "FVI" and t["session"] == s],
                    key=lambda t: t["trial"]) for s in (s1, s2)}
    if any(len({t["class"] for t in by[s]}) < 3 for s in by):
        print("sub-%02d: a shape is missing from one session, skipping" % args.subject)
        return
    print("sub-%02d | shapes | train one session, test the other | settings fixed in advance" % args.subject)
    feats = {s: session_features(by[s], fs, tmin) for s in (s1, s2)}
    F = {s: feats[s][0] for s in feats}
    T = {s: feats[s][1] for s in feats}
    y = {s: np.array([CLASSES.index(t["class"]) for t in by[s]]) for s in (s1, s2)}
    rng = np.random.default_rng(args.subject)
    pairs = [(s1, s2), (s2, s1)]

    def cross(feat, labels, keep=None, n=3, src=None, dst=None):
        accs = []
        for a, b in pairs:
            ia = keep[a] if keep else slice(None)
            ib = keep[b] if keep else slice(None)
            Xa = (F[a][feat] if src is None else T[a][src])[ia]
            Xb = (F[b][feat] if dst is None else T[b][dst])[ib]
            accs.append(score(Xa, labels[a][ia], Xb, y[b][ib], n))
        return float(np.mean(accs))

    def shuffled(keep=None):
        out = {}
        for s in y:
            lab = y[s].copy()
            idx = keep[s] if keep else np.arange(len(lab))
            lab[idx] = rng.permutation(lab[idx])
            out[s] = lab
        return out

    # three-way, cross-session
    for m in ("alpha", "multiband", "riemann", "multiband_time"):
        acc = cross(m, y)
        null = [cross(m, shuffled()) for _ in range(args.permute)]
        res["three_way"][m] = {"accuracy": acc, "p_value": p_value(acc, null), "null_mean": float(np.mean(null))}
        print("  3-way %-15s acc %.3f  p %.4f" % (m, acc, res["three_way"][m]["p_value"]))

    # three-way, pooled days with a time-ordered test block in each session
    def pooled(labels):
        accs = []
        for test, other in pairs[::-1]:
            n = len(y[test]); cut80 = int(round(0.8 * n))
            Xtr = np.vstack([F[other]["alpha"], F[test]["alpha"][:cut80]])
            ytr = np.concatenate([labels[other], labels[test][:cut80]])
            accs.append(score(Xtr, ytr, F[test]["alpha"][cut80:], y[test][cut80:], 3))
        return float(np.mean(accs))
    acc = pooled(y)
    null = [pooled(shuffled()) for _ in range(args.permute)]
    res["three_way"]["pooled"] = {"accuracy": acc, "p_value": p_value(acc, null), "null_mean": float(np.mean(null))}
    print("  3-way %-15s acc %.3f  p %.4f" % ("pooled", acc, res["three_way"]["pooled"]["p_value"]))

    # two-way, mean over the three pairs of shapes
    pair_ids = [(0, 1), (0, 2), (1, 2)]
    for m in ("alpha", "multiband", "riemann"):
        def two(labels):
            per = []
            for c1, c2 in pair_ids:
                keep = {s: np.where((y[s] == c1) | (y[s] == c2))[0] for s in y}
                lab = {s: np.where(labels[s] == c1, 0, 1) for s in y}
                yy = {s: np.where(y[s] == c1, 0, 1) for s in y}
                accs = []
                for a, b in pairs:
                    accs.append(score(F[a][m][keep[a]], lab[a][keep[a]], F[b][m][keep[b]], yy[b][keep[b]], 2))
                per.append(float(np.mean(accs)))
            return float(np.mean(per)), per

        acc, per = two(y)
        null = []
        for _ in range(args.permute):
            # shuffle the training labels within each pair's trials separately
            nulls = []
            for c1, c2 in pair_ids:
                keep = {s: np.where((y[s] == c1) | (y[s] == c2))[0] for s in y}
                accs = []
                for a, b in pairs:
                    la = np.where(y[a][keep[a]] == c1, 0, 1)
                    la = rng.permutation(la)
                    accs.append(score(F[a][m][keep[a]], la, F[b][m][keep[b]],
                                      np.where(y[b][keep[b]] == c1, 0, 1), 2))
                nulls.append(float(np.mean(accs)))
            null.append(float(np.mean(nulls)))
        res["two_way"][m] = {"accuracy": acc, "pairs": dict(zip(["circle-pentagram", "circle-square", "pentagram-square"], per)),
                             "p_value": p_value(acc, null), "null_mean": float(np.mean(null))}
        print("  2-way %-15s acc %.3f  p %.4f  pairs %s" % (m, acc, res["two_way"][m]["p_value"],
                                                             " ".join("%.3f" % v for v in per)))

    # time-matched transfer, seeing slice -> imagining slice, max over 16 under the null
    def grid(labels):
        out = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                accs = [score(T[a]["see%d" % i], labels[a], T[b]["img%d" % j], y[b], 3) for a, b in pairs]
                out[i, j] = np.mean(accs)
        return out
    g = grid(y)
    null_max = [float(grid(shuffled()).max()) for _ in range(args.permute_transfer)]
    res["transfer_time"] = {"grid": g.tolist(), "max": float(g.max()),
                            "p_max": p_value(float(g.max()), null_max), "null_max_mean": float(np.mean(null_max))}
    print("  transfer seeing->imagining, best slice pair %.3f  p(max) %.4f" % (g.max(), res["transfer_time"]["p_max"]))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
