"""
Which half of the small encoder loses the signal: the learned backbone, or the CLIP target?

Why this exists
---------------
On subject 1 the small encoder matches band power plus logistic regression exactly, 3.25x
chance. On subjects 2-4 it falls to 0.50x, 2.00x and 1.75x, while the baseline reported
2.25x, 2.75x and 3.00x on the same split. The small encoder changes two things at once
relative to the baseline, a learned EEGNet backbone in place of fixed log-variance features
and a CLIP-space head in place of a 33-way classifier, so this crosses the two:

                     classification head      CLIP head
    band power       bp_cls                   bp_clip
    EEGNet           eeg_cls                  eeg_clip  (jobs 84326 / 84567, not rerun)

bp_clip is the cell that matters most for the project. If fixed features survive the CLIP
target, we have a tiny encoder that already speaks the generator's language.

What makes this an honest comparison
------------------------------------
1. Selection happens inside the null. The band-power arms try the same four regularisation
   strengths as classical_baseline and keep the best on test. classical_baseline then
   re-runs only the winning strength on each shuffle, so its observed score is a best-of-4
   facing a best-of-1 null, and its p-values are optimistic. Here every shuffle gets the
   full best-of-4 too.
2. Each arm differs from its neighbour in one thing only. bp_clip uses logreg's optimiser,
   epochs and grid exactly, with the small encoder's InfoNCE in place of cross-entropy.
   eeg_cls uses the small encoder's backbone, optimiser, epochs, batch size and seeds, with
   a 33-way head and cross-entropy in place of the CLIP head.
3. Train accuracy is reported for every arm, so an arm that failed to optimise cannot be
   mistaken for one that found no signal.

Usage
-----
    python code/head_backbone_diag.py --subject 2 --permute 200 --json results/diag_s2.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from classical_baseline import (
    load_split, concat_windows, prepare, feat_bandpower, logreg, _permute_by_stimulus,
)
from small_encoder import SmallEncoder, clip_targets, info_nce, n_params

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)        # the grid classical_baseline searches
ARMS = ("bp_cls", "bp_clip", "eeg_cls")


# ---------------------------------------------------------------------- scoring

def top1_and_pooled(scores, key):
    """One rule for both heads: higher score is better. Probabilities for the
    classifiers, cosine similarity for the CLIP heads. Pooled averages the scores of a
    recording's windows before deciding, which is what classical_baseline's trial_avg and
    small_encoder's pooled_retrieval both do."""
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    k = torch.as_tensor(np.asarray(key))
    top1 = (s.argmax(1) == k).float().mean().item()
    hits = [float(s[k == c].mean(0).argmax().item() == c) for c in sorted(set(k.tolist()))]
    return top1, float(np.mean(hits))


# ------------------------------------------------------------------- band power

def linear_clip(Ftr, ytr, Fte, cand, l2, epochs=600, lr=0.05, seed=0):
    """logreg's twin. Identical standardisation, optimiser, learning rate, epochs and
    weight decay; the linear map lands in CLIP space and trains with InfoNCE instead."""
    torch.manual_seed(seed)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    a = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
    b = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
    y = torch.as_tensor(ytr, dtype=torch.long)
    C = cand.float()
    lin = torch.nn.Linear(a.shape[1], C.shape[1])
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=l2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = info_nce(F.normalize(lin(a), dim=-1), C[y], y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((F.normalize(lin(a), dim=-1) @ C.T).numpy(),
                (F.normalize(lin(b), dim=-1) @ C.T).numpy())


def bp_arm(kind, Ftr, ytr, Fte, key_te, cand):
    """Best of the four strengths, chosen on test, exactly as the baseline chooses."""
    best = None
    for l2 in L2_GRID:
        if kind == "bp_cls":
            ptr = logreg(Ftr, ytr, Ftr, cand.shape[0], l2)
            pte = logreg(Ftr, ytr, Fte, cand.shape[0], l2)
        else:
            ptr, pte = linear_clip(Ftr, ytr, Fte, cand, l2)
        t1, pooled = top1_and_pooled(pte, key_te)
        if best is None or t1 > best["top1"]:
            tr1, _ = top1_and_pooled(ptr, ytr)
            best = {"top1": t1, "pooled": pooled, "train_top1": tr1, "l2": l2}
    return best


# ----------------------------------------------------------------------- EEGNet

class SmallClassifier(SmallEncoder):
    """The small encoder with its head emitting 33 class logits instead of a unit vector.
    Same layers, same parameter count bar the head's output width."""

    def forward(self, x):
        x = self.b1(self.c1(x))
        x = self.d1(self.p1(F.elu(self.b2(self.dw(x)))))
        x = self.d2(self.p2(F.elu(self.b3(self.sp(x)))))
        return self.head(x.flatten(1))


def eeg_cls_arm(Xtr, ytr, Xte, key_te, n_cls, epochs, lr, seed, verbose=False):
    """small_encoder.train_once, line for line, with cross-entropy in place of InfoNCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(Xtr).float().unsqueeze(1).to(dev)
    xe = torch.from_numpy(Xte).float().unsqueeze(1).to(dev)
    ktr = torch.as_tensor(ytr, dtype=torch.long).to(dev)   # numpy int is 32-bit on Windows

    model = SmallClassifier(xt.shape[2], xt.shape[3], n_cls).to(dev)
    if verbose:
        print("[eeg_cls] %d trainable parameters" % n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = xt.shape[0]
    bs = min(64, n)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(xt[idx]), ktr[idx]).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        ptr = torch.softmax(model(xt), -1).cpu().numpy()
        pte = torch.softmax(model(xe), -1).cpu().numpy()
    t1, pooled = top1_and_pooled(pte, key_te)
    tr1, _ = top1_and_pooled(ptr, ytr)
    return {"top1": t1, "pooled": pooled, "train_top1": tr1}


# ------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Backbone x head diagnostic")
    p.add_argument("--dataset", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--splits", default="../datasets/imagination_5_95_std_splits_window_avail.pth")
    p.add_argument("--imagenet", default="../datasets/imageNet_images")
    p.add_argument("--cache", default="../datasets/clip_stimulus_emb.pt")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--band", type=float, nargs=2, default=[8.0, 13.0])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--epochs", type=int, default=200, help="EEGNet arm only")
    p.add_argument("--lr", type=float, default=1e-3, help="EEGNet arm only")
    p.add_argument("--permute", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    # data preparation is small_encoder.main's, unchanged
    tr, te, _ = load_split(args.dataset, args.splits, args.subject)
    tr, te = concat_windows(tr, 1), concat_windows(te, 1)
    band = None if args.band[0] <= 0 else (args.band[0], args.band[1])
    Xtr, tr, thresh = prepare(tr, band, False)
    Xte, te, _ = prepare(te, band, False, thresh)

    cand, uniq = clip_targets([t["image"] for t in te], args.imagenet, args.cache)
    sidx = {s: i for i, s in enumerate(uniq)}
    keep = [i for i, t in enumerate(tr) if t["image"] in sidx]
    Xtr, tr = [Xtr[i] for i in keep], [tr[i] for i in keep]
    key_tr = np.array([sidx[t["image"]] for t in tr])
    key_te = np.array([sidx[t["image"]] for t in te])

    Xtr, Xte = np.stack(Xtr), np.stack(Xte)
    Ftr, Fte = feat_bandpower(Xtr), feat_bandpower(Xte)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    n_cls = len(uniq)
    chance = 1.0 / n_cls
    print("subject %d | band %s | train %d test %d | %d stimuli, chance %.4f | arms %s"
          % (args.subject, band, len(Xtr), len(Xte), n_cls, chance, " ".join(args.arms)))

    def run(arm, y, seed=0, verbose=False):
        if arm == "eeg_cls":
            return eeg_cls_arm(Ztr, y, Zte, key_te, n_cls, args.epochs, args.lr, seed, verbose)
        return bp_arm(arm, Ftr, y, Fte, key_te, cand)

    res = {"subject": args.subject, "band": band, "n_train": len(Xtr), "n_test": len(Xte),
           "n_stimuli": n_cls, "chance": chance, "l2_grid": list(L2_GRID), "arms": {}}

    for arm in args.arms:
        obs = run(arm, key_tr, verbose=True)
        print("")
        print("=== %s ===" % arm)
        print("  test Recall@1    %.4f  (%.2fx chance)%s" % (
            obs["top1"], obs["top1"] / chance,
            "   l2 %g" % obs["l2"] if "l2" in obs else ""))
        print("  pooled Recall@1  %.4f  (%.2fx chance)" % (obs["pooled"], obs["pooled"] / chance))
        print("  train Recall@1   %.4f  (optimisation check)" % obs["train_top1"])

        if args.permute > 0:
            null, null_pooled = [], []
            for k in range(args.permute):
                # same shuffles, in the same order, as small_encoder's permutation run
                yp = _permute_by_stimulus(tr, key_tr, seed=3000 + k)
                r = run(arm, yp, seed=k)
                null.append(r["top1"])
                null_pooled.append(r["pooled"])
                if (k + 1) % 50 == 0:
                    print("  %d/%d  null mean so far %.4f (pooled %.4f)"
                          % (k + 1, args.permute, float(np.mean(null)), float(np.mean(null_pooled))))

            def summarise(vals, o):
                v = np.array(vals)
                return {"n": args.permute, "null_mean": float(v.mean()),
                        "null_p95": float(np.percentile(v, 95)), "null_max": float(v.max()),
                        "p_value": float((np.sum(v >= o) + 1.0) / (len(v) + 1.0))}

            obs["permutation"] = summarise(null, obs["top1"])
            obs["permutation_pooled"] = summarise(null_pooled, obs["pooled"])
            print("  p = %.4f   pooled p = %.4f   (null mean %.4f, p95 %.4f)" % (
                obs["permutation"]["p_value"], obs["permutation_pooled"]["p_value"],
                obs["permutation"]["null_mean"], obs["permutation"]["null_p95"]))
        res["arms"][arm] = obs

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print("")
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
