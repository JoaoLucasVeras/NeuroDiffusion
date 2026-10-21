"""
Imagery as memory replay: does imagining a shape re-create the pattern of seeing it?

Memory research detects a recalled item by how much the brain pattern during recall
resembles the pattern recorded when the item was seen (encoding-retrieval similarity),
rather than by training a classifier, and it allows the replay to happen at a different
moment, or speed, than the original. Two tests on the Gao data, settings fixed before
any result was seen (2026-10-20):

Test 1, pattern match. For each class, a "seeing template": the mean seeing-window
pattern of that class minus the mean over all classes, taken from the OTHER session.
Each imagining trial's pattern (minus its session's mean) is correlated with every
template across channels; the trial is assigned to the class it resembles most. No
classifier is trained. Templates come from the other day, so a trial's own seeing
period, or drift shared within a session, cannot produce the match.

Test 2, flexible timing. The same, in 500 ms windows every 250 ms: 15 windows of seeing,
15 of imagining. A trial's score for a class is its best match over every
(seeing window, imagining window) pair, so the replay may come at any moment and any
offset; the trial goes to the class with the best score. The search uses no labels, and
the null repeats it in full, so it pays for looking everywhere.

Features (fixed in advance):
  power  8-20 Hz log power per channel. Alpha and low beta power DEcreases carry memory
         content (the information-via-desynchronization account, Hanslmayr et al.), and
         alpha is where our imagery signal has been.
  volt   1-8 Hz voltage averaged in each window: the slow, time-locked part. Test 2 only.
Each feature is standardised within its session and window (no labels).

Primary: shapes (FVI). Secondary: animals (AVI), objects (OVI). Both session orders,
mean of the two. Null: the template session's labels shuffled, 500 times.
Excluded as duplicated in the published data (job 93110): sub-21 shapes and animals,
sub-20 objects. Subjects 9 and 10 have one session.

Usage
-----
    python gao_memory.py --subject 1 --permute 500 --json out.json
"""
import argparse
import json
import os

import numpy as np
import torch

from classical_baseline import bandpass

TASK_CLASSES = {"FVI": ["circle", "pentagram", "square"],
                "AVI": ["dog", "bird", "fish"],
                "OVI": ["scissor", "watch", "cup", "chair"]}
EXCLUDE = {("FVI", 21), ("AVI", 21), ("OVI", 20)}
SEEING = (-5.5, -1.5)
IMAGERY = (0.0, 4.0)
WIN, STEP = 0.5, 0.25


def windows(span):
    starts = np.arange(span[0], span[1] - WIN + 1e-9, STEP)
    return [(s, s + WIN) for s in starts]


def idx(fs, tmin, a, b):
    return int(round((a - tmin) * fs)), int(round((b - tmin) * fs))


def features(trials, fs, tmin, span, kind, whole=False):
    """(trials, windows, channels), standardised per session, window and channel."""
    lo, hi = (8.0, 20.0) if kind == "power" else (1.0, 8.0)
    X = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        X.append(bandpass(x - x.mean(axis=-1, keepdims=True), lo, hi, fs=fs))
    X = np.stack(X)
    spans = [span] if whole else windows(span)
    out = []
    for a, b in spans:
        i, j = idx(fs, tmin, a, b)
        seg = X[:, :, i:j]
        out.append(np.log(seg.var(axis=-1) + 1e-12) if kind == "power" else seg.mean(axis=-1))
    F = np.stack(out, axis=1)
    return (F - F.mean(0)) / (F.std(0) + 1e-8)


def unit(v):
    """Centre over channels and scale to unit length, so a dot product is a correlation."""
    v = v - v.mean(axis=-1, keepdims=True)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def templates(S, labels, n):
    """(classes, windows, channels): class mean minus the mean of all trials."""
    grand = S.mean(0)
    return unit(np.stack([S[labels == c].mean(0) - grand for c in range(n)]))


def predict(I, T):
    """I (trials, wi, ch), T (classes, ws, ch) -> best class per trial over all window pairs."""
    sim = np.einsum("nic,ksc->nkis", unit(I - I.mean(0)), T)
    return sim.reshape(sim.shape[0], sim.shape[1], -1).max(-1).argmax(1), sim


def run(feat, y, n, pairs, rng, n_perm):
    def acc(labels_for_templates):
        out = []
        for a, b in pairs:                      # imagining from a, seeing templates from b
            pred, _ = predict(feat[a]["img"], templates(feat[b]["see"], labels_for_templates[b], n))
            out.append(float((pred == y[a]).mean()))
        return float(np.mean(out))
    obs = acc(y)
    null = [acc({s: rng.permutation(y[s]) for s in y}) for _ in range(n_perm)]
    return {"accuracy": obs, "x_chance": obs * n, "null_mean": float(np.mean(null)),
            "p_value": float((np.sum(np.array(null) >= obs) + 1.0) / (n_perm + 1.0))}


def ers_map(feat, y, n, pairs):
    """Mean (own-class minus other-class) similarity for every seeing x imagining window."""
    maps = []
    for a, b in pairs:
        _, sim = predict(feat[a]["img"], templates(feat[b]["see"], y[b], n))
        own = sim[np.arange(len(y[a])), y[a]]
        others = (sim.sum(1) - own) / (n - 1)
        maps.append((own - others).mean(0))
    return np.mean(maps, axis=0)               # (imagining windows, seeing windows)


def main():
    ap = argparse.ArgumentParser(description="Imagery as memory replay, two tests")
    ap.add_argument("--data", default="../datasets/gao")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--permute", type=int, default=500)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = torch.load(os.path.join(args.data, "gao_sub-%02d.pt" % args.subject), map_location="cpu")
    trials, fs, tmin = d["dataset"], d["fs"], d["tmin"]
    res = {"subject": args.subject, "tasks": {}}
    rng = np.random.default_rng(args.subject)
    for task, classes in TASK_CLASSES.items():
        sessions = sorted({t["session"] for t in trials if t["task"] == task})
        if len(sessions) < 2 or (task, args.subject) in EXCLUDE:
            res["tasks"][task] = {"skipped": "one session" if len(sessions) < 2 else "duplicated recording"}
            continue
        n = len(classes)
        s1, s2 = sessions[:2]
        by = {s: [t for t in trials if t["task"] == task and t["session"] == s] for s in (s1, s2)}
        if any(len({t["class"] for t in by[s]}) < n for s in by):
            res["tasks"][task] = {"skipped": "class missing"}
            continue
        y = {s: np.array([classes.index(t["class"]) for t in by[s]]) for s in by}
        pairs = [(s1, s2), (s2, s1)]
        out = {"chance": 1.0 / n}
        whole = {s: {"see": features(by[s], fs, tmin, SEEING, "power", whole=True),
                     "img": features(by[s], fs, tmin, IMAGERY, "power", whole=True)} for s in by}
        out["test1_power"] = run(whole, y, n, pairs, rng, args.permute)
        for kind in ("power", "volt"):
            feat = {s: {"see": features(by[s], fs, tmin, SEEING, kind),
                        "img": features(by[s], fs, tmin, IMAGERY, kind)} for s in by}
            out["test2_" + kind] = run(feat, y, n, pairs, rng, args.permute)
            out["map_" + kind] = ers_map(feat, y, n, pairs).tolist()
        res["tasks"][task] = out
        print("sub-%02d %s %d-way | test1 power %.3f p %.3f | test2 power %.3f p %.3f | test2 volt %.3f p %.3f"
              % (args.subject, task, n, out["test1_power"]["accuracy"], out["test1_power"]["p_value"],
                 out["test2_power"]["accuracy"], out["test2_power"]["p_value"],
                 out["test2_volt"]["accuracy"], out["test2_volt"]["p_value"]), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
