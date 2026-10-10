"""
The Gao et al. (2026) EEGNet result, re-run four ways to separate signal from scoring.

The published 75.8% (animals, within session) comes from the authors' EEGNet.py. Reading
that script, two things decide what the number means:

1. The reported accuracy is the best test-fold accuracy over all training epochs. Each
   fold trains for 500 to 1500 epochs, evaluates the held-out fold after every epoch,
   and keeps the maximum. With about 24 test trials per fold, the maximum of hundreds of
   noisy evaluations sits well above chance even when nothing is learned.
2. The learning-rate scheduler steps on that same test-fold accuracy, so test data also
   shapes training.

So the same network, settings and folds are scored four ways, per session and task:

  published   shuffled 5-fold, scored as published: max test accuracy over epochs,
              scheduler on test accuracy
  null        exactly the same, with the training labels shuffled. What the published
              procedure reports when there is nothing to find. The decisive row.
  honest      shuffled 5-fold, fixed epochs, no scheduler, final-epoch accuracy
  honest_time first 80% of the run in recording order trains, last 20% tests, scored as
              honest. No epoch selection and no shared drift.

The network is EEGNetModel copied from the authors' EEGNet.py, minus its max-norm helper,
which their training loop defines but never calls, so behaviour is unchanged; optimiser,
learning rate, weight decay, batch size, per-channel standardisation on training stats
and 500 epochs (their paper's Table 2) are theirs too. Preprocessing differs: theirs is
ICA-cleaned, ours is average reference and a 4-80 Hz band-pass on the stored trials,
because ICA needs the continuous raw recording. Window: 0-4 s after the marker, as
theirs.

Usage
-----
    python gao_eegnet.py --subject 1 --json out.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from classical_baseline import bandpass

TASK_CLASSES = {"AVI": ["dog", "bird", "fish"],
                "FVI": ["circle", "pentagram", "square"],
                "OVI": ["scissor", "watch", "cup", "chair"]}


# ---- copied unchanged from the authors' EEGNet.py (figshare 30227503 v3, CC BY 4.0) ----
class EEGNetModel(nn.Module):
    def __init__(self, chans=32, classes=5, time_points=1000, temp_kernel=25,
                 f1=8, f2=16, d=2, pk1=16, pk2=8, dropout_rate=0.5, max_norm1=1, max_norm2=1):
        super(EEGNetModel, self).__init__()
        linear_size = (time_points // (pk1 * pk2)) * f2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
# ----------------------------------------------------------------------------------------


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare(trials, fs, tmin):
    """Average reference, 4-80 Hz on the full 12 s trial, then the 0-4 s imagery window."""
    a, b = int(round(-tmin * fs)), int(round((4.0 - tmin) * fs))
    X = []
    for t in trials:
        x = t["eeg"].numpy().astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        x = bandpass(x, 4.0, 80.0, fs=fs)
        X.append(x[:, a:b])
    return np.stack(X).astype(np.float32)


def run_fold(Xtr, ytr, Xte, yte, n_cls, published, epochs, seed):
    """One training run, the authors' settings. Returns (max over epochs, final epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr = torch.from_numpy(Xtr).unsqueeze(1)
    xte = torch.from_numpy(Xte).unsqueeze(1)
    mean = xtr.mean(dim=(0, 3), keepdim=True)
    std = xtr.std(dim=(0, 3), keepdim=True) + 1e-8
    xtr, xte = ((xtr - mean) / std).to(DEV), ((xte - mean) / std).to(DEV)
    ttr = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    tte = torch.as_tensor(yte, dtype=torch.long).to(DEV)
    loader = DataLoader(TensorDataset(xtr, ttr), batch_size=64, shuffle=True)

    model = EEGNetModel(chans=Xtr.shape[1], classes=n_cls, time_points=Xtr.shape[2]).to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.09)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30) \
        if published else None
    best, acc = 0.0, 0.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        if published or _ == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = float((model(xte).argmax(1) == tte).float().mean())
            best = max(best, acc)
            if sched is not None:
                sched.step(acc)                 # as published: the scheduler sees test accuracy
    return best, acc


def stratified_folds(y, n_cls, k, rng):
    folds = [[] for _ in range(k)]
    for c in range(n_cls):
        for j, i in enumerate(rng.permutation(np.where(y == c)[0])):
            folds[j % k].append(i)
    every = np.arange(len(y))
    return [(np.setdiff1d(every, f), np.array(sorted(f))) for f in folds]


def main():
    ap = argparse.ArgumentParser(description="Gao EEGNet, four ways")
    ap.add_argument("--data", default="../datasets/gao")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = torch.load(os.path.join(args.data, "gao_sub-%02d.pt" % args.subject), map_location="cpu")
    trials, fs, tmin = d["dataset"], d["fs"], d["tmin"]
    rng = np.random.default_rng(args.subject)
    res = {"subject": args.subject, "epochs": args.epochs, "device": str(DEV), "sessions": {}}
    print("sub-%02d | %d epochs | device %s" % (args.subject, args.epochs, DEV))

    for ses in sorted({t["session"] for t in trials}):
        res["sessions"][ses] = {}
        for task, classes in TASK_CLASSES.items():
            tt = [t for t in trials if t["task"] == task and t["session"] == ses]
            if not tt or any(not any(t["class"] == c for t in tt) for c in classes):
                res["sessions"][ses][task] = {"skipped": "stimulus missing"}
                continue
            tt = sorted(tt, key=lambda t: t["trial"])                 # recording order
            X = prepare(tt, fs, tmin)
            y = np.array([classes.index(t["class"]) for t in tt])
            n = len(classes)
            folds = stratified_folds(y, n, 5, rng)
            y_null = rng.permutation(y)          # one fixed shuffle per session and task
            out = {"chance": 1.0 / n, "n": len(tt)}

            pub, nul, hon = [], [], []
            for f, (a, b) in enumerate(folds):
                pub.append(run_fold(X[a], y[a], X[b], y[b], n, True, args.epochs, f)[0])
                nul.append(run_fold(X[a], y_null[a], X[b], y_null[b], n, True, args.epochs, f)[0])
                hon.append(run_fold(X[a], y[a], X[b], y[b], n, False, args.epochs, f)[1])
            cut = int(round(0.8 * len(tt)))
            ht = run_fold(X[:cut], y[:cut], X[cut:], y[cut:], n, False, args.epochs, 0)[1]

            out.update({"published": float(np.mean(pub)), "null": float(np.mean(nul)),
                        "honest": float(np.mean(hon)), "honest_time": ht})
            res["sessions"][ses][task] = out
            print("  ses %d %s %d-way | published %.3f  null %.3f  honest %.3f  honest_time %.3f"
                  % (ses, task, n, out["published"], out["null"], out["honest"], ht))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
