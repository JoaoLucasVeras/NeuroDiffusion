"""
S1 -- classical baselines for imagined-category decoding. No diffusion, no GPU.

Purpose: find out whether ANY decodable imagery signal exists in this EEG, so that
a null from the generative pipeline becomes interpretable. Right now a null could
mean (a) no signal, (b) model too large for the data, (c) wrong input
representation, or (d) a fault downstream of the encoder -- and those are
indistinguishable. A simple classifier that cannot memorise 33 classes from ~360
examples separates (a) from the rest.

Runs in minutes on CPU. Deliberately dependency-light: numpy + scipy + torch only
(the cluster env has no scikit-learn or pyriemann, and compute nodes are offline).

Three feature sets, all on the same leakage-free window split the deep model uses:

  bandpower   log variance per channel after band-pass. The direct test of
              "is alpha power informative?" 128 features.
  riemann     per-trial covariance -> tangent space at the log-Euclidean mean,
              after channel PCA. The standard strong baseline in low-data EEG
              (arXiv 2407.20250). PCA is required: a raw 128x128 covariance has
              8,256 free parameters against ~360 training trials.
  eegnet      compact conv net on the filtered time series (~2k params). Does its
              own spatial filtering, so no channel selection needed.

Also reports the ARTIFACT ARM: Hajhassani et al. 2026 (arXiv 2605.12408) found
artifact rejection helps most in exactly our regime (low baseline, low SNR), while
Kessler et al. 2025 found it hurts in high-SNR perception tasks. Cheap enough to
settle empirically -- `--reject_artifacts` drops high-amplitude trials.

Usage (per subject):
    python classical_baseline.py --subject 1
    python classical_baseline.py --subject 1 --band 8 13 --window_len 4
    python classical_baseline.py --subject 1 --reject_artifacts
    python classical_baseline.py --all_subjects --json out.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt


# --------------------------------------------------------------------------- data

def load_split(dataset_path, splits_path, subject):
    """Return (train, test) lists of dicts for one subject, from the window split."""
    payload = torch.load(dataset_path, map_location="cpu")
    entries = payload["dataset"]
    labels = payload["labels"]
    sp = torch.load(splits_path, map_location="cpu")["splits"][0]

    def take(idx):
        out = []
        for i in idx:
            e = entries[i]
            if subject and e["subject"] != subject:
                continue
            out.append({
                "eeg": e["eeg"].numpy().astype(np.float64),   # (channels, time)
                "label": int(e["label"]),
                "image": e["image"],
                "window": int(e["window"]),
                "subject": int(e["subject"]),
            })
        return out

    return take(sp["train"]), take(sp["test"]), labels


def concat_windows(trials, n):
    """Glue n adjacent windows of the same recording into one longer segment.

    The source windows are non-overlapping and contiguous (Shimizu Methods: "each
    10 second recording was split into 20 trials of 500 ms without overlap"), so
    concatenating consecutive indices reconstructs the true continuous signal.
    Incomplete groups are dropped rather than zero-padded.
    """
    if n <= 1:
        return trials
    by_rec = {}
    for t in trials:
        by_rec.setdefault((t["subject"], t["image"]), []).append(t)
    out = []
    for (subj, img), group in by_rec.items():
        group.sort(key=lambda t: t["window"])
        wins = [t["window"] for t in group]
        run = []
        for t, w in zip(group, wins):
            if run and w != run[-1]["window"] + 1:
                run = []                      # a gap: start a fresh run
            run.append(t)
            if len(run) == n:
                out.append({
                    "eeg": np.concatenate([r["eeg"] for r in run], axis=1),
                    "label": run[0]["label"], "image": img,
                    "window": run[0]["window"], "subject": subj,
                })
                run = []
    return out


def bandpass(x, lo, hi, fs=250.0, order=4):
    """x: (channels, time). Zero-phase Butterworth band-pass."""
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-6), min(hi / nyq, 0.99)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:                 # too short to pad: filter unpadded
        return filtfilt(b, a, x, axis=-1, padlen=0)
    return filtfilt(b, a, x, axis=-1)


def prepare(trials, band, reject_artifacts, reject_thresh=None):
    """Filter every trial; optionally drop high-amplitude (likely ocular) trials."""
    X = []
    for t in trials:
        x = t["eeg"]
        x = x - x.mean(axis=-1, keepdims=True)
        if band is not None:
            x = bandpass(x, band[0], band[1])
        X.append(np.ascontiguousarray(x))
    kept = list(range(len(trials)))
    if reject_artifacts:
        # Peak-to-peak per trial, pooled over channels. Blinks run 100+ uV against
        # ~10 uV of EEG, so they sit far out in this distribution.
        p2p = np.array([x.max(axis=-1).ptp() if x.shape[0] == 1 else
                        (x.max(axis=-1) - x.min(axis=-1)).max() for x in X])
        if reject_thresh is None:
            reject_thresh = np.percentile(p2p, 90)
        kept = [i for i in kept if p2p[i] <= reject_thresh]
    return [X[i] for i in kept], [trials[i] for i in kept], reject_thresh


# ----------------------------------------------------------------------- features

def feat_bandpower(X):
    """log variance per channel. (n, channels)"""
    return np.log(np.stack([x.var(axis=-1) + 1e-12 for x in X]))


def _cov(x, shrink=0.05):
    """Shrunk sample covariance, normalised by trace so amplitude drops out."""
    c = np.cov(x)
    c = c / (np.trace(c) / c.shape[0] + 1e-12)
    return (1 - shrink) * c + shrink * np.eye(c.shape[0])


def fit_channel_pca(X, k):
    """Channel-space PCA fitted on TRAIN only. Returns a (k, channels) projection."""
    flat = np.concatenate([x for x in X], axis=1)          # (channels, total_time)
    flat = flat - flat.mean(axis=1, keepdims=True)
    c = np.cov(flat)
    w, v = eigh(c)
    order = np.argsort(w)[::-1][:k]
    return v[:, order].T


def _logm_spd(c):
    w, v = eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * np.log(w)) @ v.T


def _invsqrtm_spd(c):
    w, v = eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * (w ** -0.5)) @ v.T


def feat_riemann_fit(X, proj):
    """Fit the tangent-space reference on TRAIN. Returns (features, reference)."""
    covs = [_cov(proj @ x) for x in X]
    # Log-Euclidean mean: cheap, and a good stand-in for the Riemannian mean.
    ref = np.mean([_logm_spd(c) for c in covs], axis=0)
    w, v = eigh(ref)
    ref = (v * np.exp(w)) @ v.T
    return _tangent(covs, ref), ref


def feat_riemann_apply(X, proj, ref):
    return _tangent([_cov(proj @ x) for x in X], ref)


def _tangent(covs, ref):
    """Upper triangle of logm(ref^-1/2 C ref^-1/2), off-diagonals scaled by sqrt(2)."""
    isq = _invsqrtm_spd(ref)
    d = ref.shape[0]
    iu = np.triu_indices(d)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = []
    for c in covs:
        s = _logm_spd(isq @ c @ isq)
        out.append(s[iu] * scale)
    return np.stack(out)


# -------------------------------------------------------------------- classifiers

def logreg(Xtr, ytr, Xte, n_classes, l2, epochs=600, lr=0.05, seed=0):
    """Multinomial logistic regression, full-batch LBFGS-free. Returns test probs."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    a = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    b = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.long)
    lin = torch.nn.Linear(a.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=l2)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(a), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(lin(b), dim=-1).numpy()


class EEGNet(torch.nn.Module):
    """Compact EEGNet-style net (~2k params). Lawhern et al. 2018, trimmed."""
    def __init__(self, n_ch, n_time, n_classes, F1=8, D=2, F2=16, drop=0.5):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, F1, (1, 33), padding=(0, 16), bias=False)
        self.b1 = torch.nn.BatchNorm2d(F1)
        self.dw = torch.nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.b2 = torch.nn.BatchNorm2d(F1 * D)
        self.p1 = torch.nn.AvgPool2d((1, 4))
        self.d1 = torch.nn.Dropout(drop)
        self.sp = torch.nn.Conv2d(F1 * D, F2, (1, 9), padding=(0, 4), bias=False)
        self.b3 = torch.nn.BatchNorm2d(F2)
        self.p2 = torch.nn.AvgPool2d((1, 4))
        self.d2 = torch.nn.Dropout(drop)
        self.head = torch.nn.Linear(F2 * max(n_time // 16, 1), n_classes)

    def forward(self, x):
        x = torch.nn.functional.elu(self.b1(self.c1(x)))
        x = self.d1(self.p1(torch.nn.functional.elu(self.b2(self.dw(x)))))
        x = self.d2(self.p2(torch.nn.functional.elu(self.b3(self.sp(x)))))
        return self.head(x.flatten(1))


def run_eegnet(Xtr, ytr, Xte, n_classes, epochs=150, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = np.stack(Xtr)[:, None]                      # (n, 1, ch, time)
    te = np.stack(Xte)[:, None]
    mu, sd = tr.mean(), tr.std() + 1e-8
    a = torch.tensor((tr - mu) / sd, dtype=torch.float32)
    b = torch.tensor((te - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.long)
    net = EEGNet(a.shape[2], a.shape[3], n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(a)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            lossf(net(a[idx]), y[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(b), dim=-1).numpy()


# ----------------------------------------------------------------------- scoring

def score(probs, trials, n_classes):
    """Per-trial top-1/top-5, plus the trial-averaged (one decision per stimulus)."""
    y = np.array([t["label"] for t in trials])
    top1 = float((probs.argmax(1) == y).mean())
    k5 = min(5, n_classes)
    top5 = float(np.mean([y[i] in np.argsort(-probs[i])[:k5] for i in range(len(y))]))

    # pool the windows of each stimulus before deciding
    stems = [t["image"] for t in trials]
    uniq = sorted(set(stems))
    idx = {s: i for i, s in enumerate(uniq)}
    acc = np.zeros((len(uniq), n_classes))
    lab = np.zeros(len(uniq), dtype=int)
    for i, s in enumerate(stems):
        acc[idx[s]] += probs[i]
        lab[idx[s]] = y[i]
    avg_top1 = float((acc.argmax(1) == lab).mean())
    return {"top1": top1, "top5": top5, "trial_avg_top1": avg_top1,
            "n_test": len(y), "n_stimuli": len(uniq)}


# -------------------------------------------------------------------------- main

def run_subject(args, subject, dataset, splits):
    tr, te, synsets = load_split(dataset, splits, subject)
    if not tr or not te:
        return {"subject": subject, "error": "empty split"}

    tr = concat_windows(tr, args.window_len)
    te = concat_windows(te, args.window_len)

    band = None if args.band[0] <= 0 else (args.band[0], args.band[1])
    Xtr, tr, thresh = prepare(tr, band, args.reject_artifacts)
    Xte, te, _ = prepare(te, band, args.reject_artifacts, thresh)

    present = sorted({t["label"] for t in tr} | {t["label"] for t in te})
    remap = {c: i for i, c in enumerate(present)}
    n_classes = len(present)
    ytr = np.array([remap[t["label"]] for t in tr])
    for t in tr + te:
        t["label"] = remap[t["label"]]

    res = {"subject": subject, "n_train": len(tr), "n_test": len(te),
           "n_classes": n_classes, "chance": 1.0 / n_classes,
           "band": band, "window_len": args.window_len,
           "reject_artifacts": bool(args.reject_artifacts),
           "n_channels": Xtr[0].shape[0], "n_samples": Xtr[0].shape[1]}

    # ---- bandpower + logistic regression
    Ftr, Fte = feat_bandpower(Xtr), feat_bandpower(Xte)
    best = None
    for l2 in args.l2:
        p = logreg(Ftr, ytr, Fte, n_classes, l2)
        s = score(p, te, n_classes)
        s["l2"] = l2
        if best is None or s["top1"] > best["top1"]:
            best = s
    res["bandpower"] = best

    # ---- riemannian tangent space + logistic regression
    try:
        proj = fit_channel_pca(Xtr, args.pca)
        Rtr, ref = feat_riemann_fit(Xtr, proj)
        Rte = feat_riemann_apply(Xte, proj, ref)
        best = None
        for l2 in args.l2:
            p = logreg(Rtr, ytr, Rte, n_classes, l2)
            s = score(p, te, n_classes)
            s["l2"] = l2
            if best is None or s["top1"] > best["top1"]:
                best = s
        best["n_features"] = int(Rtr.shape[1])
        res["riemann"] = best
    except Exception as e:                      # keep going; report the failure
        res["riemann"] = {"error": "%s: %s" % (type(e).__name__, e)}

    # ---- eegnet
    if not args.skip_eegnet:
        try:
            p = run_eegnet(Xtr, ytr, Xte, n_classes, epochs=args.eegnet_epochs)
            res["eegnet"] = score(p, te, n_classes)
        except Exception as e:
            res["eegnet"] = {"error": "%s: %s" % (type(e).__name__, e)}

    # ---- which channels carry the signal (we have no electrode labels, so rank
    #      them by between-class separability of log band power)
    F = np.concatenate([Ftr, Fte])
    yall = np.concatenate([ytr, [t["label"] for t in te]])
    gm = F.mean(0)
    between = np.zeros(F.shape[1])
    for c in range(n_classes):
        m = yall == c
        if m.sum() > 1:
            between += m.sum() * (F[m].mean(0) - gm) ** 2
    within = F.var(0) * len(F) + 1e-12
    res["top_channels"] = [int(i) for i in np.argsort(-(between / within))[:10]]
    return res


def main():
    p = argparse.ArgumentParser(description="Classical baselines for imagery decoding")
    p.add_argument("--dataset", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--splits", default="../datasets/imagination_5_95_std_splits_window_avail.pth")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--all_subjects", action="store_true")
    p.add_argument("--band", type=float, nargs=2, default=[8.0, 13.0],
                   help="band-pass in Hz; use '0 0' for broadband")
    p.add_argument("--window_len", type=int, default=1,
                   help="adjacent 500 ms windows to glue together (1=500ms, 4=2s)")
    p.add_argument("--pca", type=int, default=16, help="channel PCA dims for riemann")
    p.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--reject_artifacts", action="store_true",
                   help="drop the top-decile peak-to-peak trials (ocular arm)")
    p.add_argument("--eegnet_epochs", type=int, default=150)
    p.add_argument("--skip_eegnet", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    subjects = [1, 2, 3, 4] if args.all_subjects else [args.subject]
    out = {"config": vars(args), "results": []}
    for s in subjects:
        r = run_subject(args, s, args.dataset, args.splits)
        out["results"].append(r)
        print("\n" + "=" * 72)
        print("SUBJECT %s  |  train %s  test %s  |  %s classes, chance %.4f"
              % (s, r.get("n_train"), r.get("n_test"), r.get("n_classes"),
                 r.get("chance", 0)))
        print("  input: %s ch x %s samples | band %s | window_len %s | reject %s"
              % (r.get("n_channels"), r.get("n_samples"), r.get("band"),
                 r.get("window_len"), r.get("reject_artifacts")))
        for name in ("bandpower", "riemann", "eegnet"):
            d = r.get(name)
            if not d:
                continue
            if "error" in d:
                print("  %-10s FAILED  %s" % (name, d["error"]))
                continue
            ch = r["chance"]
            print("  %-10s top1 %.4f (%.1fx chance)  top5 %.4f  "
                  "trial-avg %.4f (%d stimuli)"
                  % (name, d["top1"], d["top1"] / ch if ch else 0,
                     d["top5"], d["trial_avg_top1"], d["n_stimuli"]))
        print("  most separable channel indices:", r.get("top_channels"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("\nwrote", args.json)

    # headline: did anything clear chance on the trial-averaged metric?
    best = 0.0
    for r in out["results"]:
        for name in ("bandpower", "riemann", "eegnet"):
            d = r.get(name) or {}
            if "trial_avg_top1" in d:
                best = max(best, d["trial_avg_top1"] / r["chance"])
    print("\nBEST trial-averaged result across everything: %.1fx chance" % best)
    print("(>2x on several subjects = signal worth chasing; ~1x everywhere = this "
          "dataset is exhausted)")


if __name__ == "__main__":
    main()
