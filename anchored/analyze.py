"""
The anchored-imagery analysis, as fixed in PROTOCOL.md sections 8 and 9.

Per participant: alpha (8-13 Hz) log power at every EEG channel in the imagery window
(3.5-7.5 s), plus a posterior lateralization index (right minus left back of the head),
standardised within each session without labels. Logistic regression with l2 fixed at
0.01, trained on two sessions and tested on the third, all rotations, mean accuracy.
Null: training labels shuffled, the whole pipeline repeated.

  H1   move_left vs move_right, anchored trials
  H2   all four changes, anchored trials
  H3   anchored minus unanchored accuracy, left vs right and four-way
  eyes the same decoders on the eye channels alone (HEOG/VEOG: mean voltage in 0.5 s bins
       and alpha power); if they decode as well as the brain, the result is not imagery

The group summary combines p-values with Fisher's method, applies Bonferroni over the
three hypotheses, and prints each gate of PROTOCOL.md section 9 as PASS or FAIL.

Usage
-----
    python analyze.py sim_sub-01.pt sim_sub-02.pt --permute 1000 --json results.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from classical_baseline import bandpass, logreg   # noqa: E402

IMG = (3.5, 7.5)
LEFT_BACK = ["P7", "P3", "PO7", "PO3", "O1"]
RIGHT_BACK = ["P8", "P4", "PO8", "PO4", "O2"]
L2 = 0.01


def zscore(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-8)


def features(d):
    ch, fs, tmin = d["channels"], d["fs"], d["tmin"]
    eog = [ch.index(c) for c in d["eog"]]
    eeg = [i for i in range(len(ch)) if i not in eog]
    a, b = int(round((IMG[0] - tmin) * fs)), int(round((IMG[1] - tmin) * fs))
    brain, eyes = [], []
    for t in d["dataset"]:
        x = t["eeg"].numpy().astype(np.float64)
        al = bandpass(x - x.mean(axis=-1, keepdims=True), 8.0, 13.0, fs=fs)[:, a:b]
        p = np.log(al.var(axis=-1) + 1e-12)
        lat = np.mean([p[ch.index(c)] for c in RIGHT_BACK]) - np.mean([p[ch.index(c)] for c in LEFT_BACK])
        brain.append(np.append(p[eeg], lat))
        raw = x[eog, a:b] - x[eog, a - int(0.5 * fs):a].mean(axis=-1, keepdims=True)   # relative to cue end
        step = int(0.5 * fs)
        bins = raw[:, :raw.shape[1] // step * step].reshape(len(eog), -1, step).mean(-1).ravel()
        eyes.append(np.concatenate([bins, p[eog]]))
    sess = np.array([t["session"] for t in d["dataset"]])
    out = {}
    for name, F in (("brain", np.array(brain)), ("eyes", np.array(eyes))):
        G = np.empty_like(F)
        for s in np.unique(sess):
            G[sess == s] = zscore(F[sess == s])
        out[name] = G
    return out, sess


def cross_session(F, y, sess, n):
    accs = []
    for s in np.unique(sess):
        tr, te = sess != s, sess == s
        accs.append(float((logreg(F[tr], y[tr], F[te], n, L2).argmax(1) == y[te]).mean()))
    return float(np.mean(accs))


def test(F, y, sess, n, rng, n_perm):
    acc = cross_session(F, y, sess, n)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(sess):                       # shuffle within session
            m = sess == s
            yp[m] = rng.permutation(y[m])
        null.append(cross_session(F, yp, sess, n))
    return {"accuracy": acc, "chance": 1.0 / n, "null_mean": float(np.mean(null)),
            "p_value": float((np.sum(np.array(null) >= acc) + 1.0) / (n_perm + 1.0))}


def participant(path, n_perm):
    d = torch.load(path, map_location="cpu")
    feats, sess = features(d)
    change = np.array([t["change"] for t in d["dataset"]])
    anchor = np.array([t["anchor"] for t in d["dataset"]])
    rng = np.random.default_rng(int(d.get("participant", 0)))
    res = {"file": os.path.basename(path), "scenario": d.get("scenario"), "tests": {}}
    for cond in ("anchored", "unanchored"):
        for task, classes in (("left_right", ["move_left", "move_right"]),
                              ("four_way", ["move_left", "move_right", "stretch", "shrink"])):
            m = (anchor == cond) & np.isin(change, classes)
            y = np.array([classes.index(c) for c in change[m]])
            for src in ("brain", "eyes"):
                res["tests"]["%s_%s_%s" % (task, cond, src)] = test(feats[src][m], y, sess[m], len(classes), rng, n_perm)
    return res


def fisher(ps):
    return float(stats.combine_pvalues(ps, method="fisher").pvalue)


def gates(results):
    def col(key, field="accuracy"):
        return np.array([r["tests"][key][field] for r in results])
    alpha3 = 0.05 / 3
    lr, lr_p = col("left_right_anchored_brain"), col("left_right_anchored_brain", "p_value")
    eye_lr_p = col("left_right_anchored_eyes", "p_value")
    fw, fw_p = col("four_way_anchored_brain"), col("four_way_anchored_brain", "p_value")
    diff_lr = lr - col("left_right_unanchored_brain")
    h3_p = float(stats.ttest_1samp(diff_lr, 0.0, alternative="greater").pvalue) if len(results) > 1 else 1.0
    g = {
        "pilot": {"rule": "left/right >= 60% across days in >= 2 people, eyes-only at chance",
                  "pass": bool(((lr >= 0.60) & (lr_p < 0.05)).sum() >= 2 and fisher(eye_lr_p) >= 0.05)},
        "H1": {"rule": "left/right group p < 0.0167 and eyes-only not significant",
               "value": "%.3f mean, p %.2g, eyes p %.2g" % (lr.mean(), fisher(lr_p), fisher(eye_lr_p)),
               "pass": bool(fisher(lr_p) < alpha3 and fisher(eye_lr_p) >= 0.05)},
        "H2": {"rule": "four-way mean >= 35% and group p < 0.0167",
               "value": "%.3f mean, p %.2g" % (fw.mean(), fisher(fw_p)),
               "pass": bool(fw.mean() >= 0.35 and fisher(fw_p) < alpha3)},
        "H3": {"rule": "anchored better than unanchored (left/right), p < 0.0167",
               "value": "%+.3f mean difference, p %.2g" % (diff_lr.mean(), h3_p),
               "pass": bool(h3_p < alpha3)},
    }
    return g


def main():
    ap = argparse.ArgumentParser(description="Anchored-imagery analysis")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--permute", type=int, default=1000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for f in args.files:
        r = participant(f, args.permute)
        t = r["tests"]
        print("%s | L/R anchored %.3f (p %.3f) unanchored %.3f | 4-way anchored %.3f | eyes L/R %.3f (p %.3f)"
              % (r["file"], t["left_right_anchored_brain"]["accuracy"], t["left_right_anchored_brain"]["p_value"],
                 t["left_right_unanchored_brain"]["accuracy"], t["four_way_anchored_brain"]["accuracy"],
                 t["left_right_anchored_eyes"]["accuracy"], t["left_right_anchored_eyes"]["p_value"]), flush=True)
        results.append(r)
    g = gates(results)
    for k, v in g.items():
        print("  %-5s %s  %s  [%s]" % (k, "PASS" if v["pass"] else "FAIL", v.get("value", ""), v["rule"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"participants": results, "gates": g}, fh, indent=2)


if __name__ == "__main__":
    main()
