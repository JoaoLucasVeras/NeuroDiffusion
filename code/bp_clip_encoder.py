"""
Train and save the band-power -> CLIP encoder, one per subject, for the conditioning step.

Why this exists
---------------
head_backbone_diag (jobs 84691 / 84692) crossed features against heads. Fixed band-power
features mapped linearly into CLIP space kept the signal on three of four subjects
(3.25, 2.25, 3.00, 2.50x chance), while every learned EEGNet variant lost it. This is
that encoder, trained once and saved, so the generation step has something to condition
on that is measurably not noise on more than one person.

How the regularisation strength is chosen
-----------------------------------------
The diagnostic picked l2 on the test set, which was acceptable there because the same
selection ran inside its permutation null. It is not acceptable here: images generated
for the test set must not depend on the test set. So l2 is chosen inside the training
data, by a split that mirrors the outer one. The outer split holds out the last four of
twenty windows of each recording with a one-window gap; here the last two training
windows are held out, again with a one-window gap. The test set is touched once, at the
end, to report what the saved encoder does.

Output, one file per subject
----------------------------
The linear map, the feature standardisation it was trained with, the band, the chosen
l2 and the inner scores behind that choice, the stimulus order, and the predicted CLIP
embedding for every test window. The generation step needs only those predictions.

Usage
-----
    python bp_clip_encoder.py --subject 1 --save ../results/bp_clip_s1.pt --permute 200
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from classical_baseline import (
    load_split, concat_windows, prepare, feat_bandpower, _permute_by_stimulus,
)
from small_encoder import clip_targets, info_nce
from head_backbone_diag import L2_GRID, top1_and_pooled


def fit(Ftr, ytr, cand, l2, epochs=600, lr=0.05, seed=0):
    """head_backbone_diag.linear_clip, returning the fitted map instead of predictions.
    Identical standardisation, optimiser, schedule and loss."""
    torch.manual_seed(seed)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    a = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
    y = torch.as_tensor(ytr, dtype=torch.long)
    C = cand.float()
    lin = torch.nn.Linear(a.shape[1], C.shape[1])
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=l2)
    for _ in range(epochs):
        opt.zero_grad()
        info_nce(F.normalize(lin(a), dim=-1), C[y], y).backward()
        opt.step()
    return lin, mu, sd


def predict(lin, mu, sd, Fx):
    with torch.no_grad():
        x = torch.tensor((Fx - mu) / sd, dtype=torch.float32)
        return F.normalize(lin(x), dim=-1)


def inner_split(windows, n_hold=2, gap=1):
    """Indices for an inner train / validation split by window position, mirroring
    the outer window protocol: the last n_hold windows are validation, the gap window
    before them is dropped so the two never touch in time."""
    w = sorted(set(windows))
    hold = set(w[-n_hold:])
    drop = set(w[-(n_hold + gap):-n_hold]) if gap else set()
    tr = [i for i, x in enumerate(windows) if x not in hold and x not in drop]
    va = [i for i, x in enumerate(windows) if x in hold]
    return np.array(tr), np.array(va)


def choose_l2(Ftr, ytr, windows, cand):
    """Best l2 on the inner validation windows. Training data only."""
    itr, iva = inner_split(windows)
    scores = {}
    for l2 in L2_GRID:
        lin, mu, sd = fit(Ftr[itr], ytr[itr], cand, l2)
        t1, _ = top1_and_pooled((predict(lin, mu, sd, Ftr[iva]) @ cand.float().T).numpy(), ytr[iva])
        scores[l2] = t1
    best = max(L2_GRID, key=lambda k: scores[k])     # ties go to the smaller l2
    return best, scores


def run(Ftr, ytr, windows, Fte, key_te, cand):
    l2, inner = choose_l2(Ftr, ytr, windows, cand)
    lin, mu, sd = fit(Ftr, ytr, cand, l2)
    pred = predict(lin, mu, sd, Fte)
    t1, pooled = top1_and_pooled((pred @ cand.float().T).numpy(), key_te)
    return {"l2": l2, "inner": inner, "top1": t1, "pooled": pooled,
            "lin": lin, "mu": mu, "sd": sd, "pred": pred}


def main():
    p = argparse.ArgumentParser(description="Band-power -> CLIP encoder, trained and saved")
    p.add_argument("--dataset", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--splits", default="../datasets/imagination_5_95_std_splits_window_avail.pth")
    p.add_argument("--imagenet", default="../datasets/imageNet_images")
    p.add_argument("--cache", default="../datasets/clip_stimulus_emb.pt")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--band", type=float, nargs=2, default=[8.0, 13.0])
    p.add_argument("--permute", type=int, default=0,
                   help="permutation null for this exact procedure, inner l2 choice included")
    p.add_argument("--save", required=True)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    # data preparation is head_backbone_diag's, unchanged
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
    win_tr = [t["window"] for t in tr]

    Ftr, Fte = feat_bandpower(np.stack(Xtr)), feat_bandpower(np.stack(Xte))
    chance = 1.0 / len(uniq)
    itr, iva = inner_split(win_tr)
    print("subject %d | band %s | train %d test %d | %d stimuli, chance %.4f"
          % (args.subject, band, len(Ftr), len(Fte), len(uniq), chance))
    print("inner split: train windows %s, validation windows %s"
          % (sorted({win_tr[i] for i in itr}), sorted({win_tr[i] for i in iva})))

    r = run(Ftr, key_tr, win_tr, Fte, key_te, cand)
    print("")
    print("inner Recall@1 by l2: " + "  ".join("%g: %.4f" % (k, v) for k, v in r["inner"].items()))
    print("chosen l2            %g  (on training data only)" % r["l2"])
    print("test Recall@1        %.4f  (%.2fx chance)" % (r["top1"], r["top1"] / chance))
    print("test pooled Recall@1 %.4f  (%.2fx chance)" % (r["pooled"], r["pooled"] / chance))

    res = {"subject": args.subject, "band": band, "chance": chance, "l2": r["l2"],
           "inner": {str(k): v for k, v in r["inner"].items()},
           "recall_at_1": r["top1"], "pooled_recall_at_1": r["pooled"]}

    if args.permute > 0:
        null, null_pooled = [], []
        for k in range(args.permute):
            yp = _permute_by_stimulus(tr, key_tr, seed=3000 + k)
            q = run(Ftr, yp, win_tr, Fte, key_te, cand)
            null.append(q["top1"])
            null_pooled.append(q["pooled"])
            if (k + 1) % 50 == 0:
                print("  %d/%d  null mean so far %.4f" % (k + 1, args.permute, float(np.mean(null))))

        def summarise(vals, o):
            v = np.array(vals)
            return {"n": args.permute, "null_mean": float(v.mean()),
                    "null_p95": float(np.percentile(v, 95)),
                    "p_value": float((np.sum(v >= o) + 1.0) / (len(v) + 1.0))}

        res["permutation"] = summarise(null, r["top1"])
        res["permutation_pooled"] = summarise(null_pooled, r["pooled"])
        print("p = %.4f   pooled p = %.4f" % (res["permutation"]["p_value"],
                                              res["permutation_pooled"]["p_value"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    torch.save({
        "kind": "bandpower_linear_clip",
        "weight": r["lin"].weight.detach().clone(),       # (768, channels)
        "bias": r["lin"].bias.detach().clone(),
        "feat_mu": torch.as_tensor(r["mu"]), "feat_sd": torch.as_tensor(r["sd"]),
        "band": band, "l2": r["l2"], "subject": args.subject,
        "stimulus_order": uniq,
        "test": {"pred": r["pred"], "key": torch.as_tensor(key_te),
                 "image": [t["image"] for t in te], "window": [t["window"] for t in te]},
        "metrics": res, "argv": vars(args),
    }, args.save)
    print("")
    print("[save] encoder and %d test predictions -> %s" % (len(Fte), args.save))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
