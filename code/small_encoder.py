"""
A small EEG -> CLIP encoder, as a direct replacement for the 85M-parameter Stage 1/2
conditioning path.

Why this exists
---------------
The classical baseline established that alpha-band EEG carries decodable imagery signal:
band power plus logistic regression reaches 3.25x chance on subject 1, significant under a
permutation null at p = 0.005. Two Stage 2 runs then failed to find it. Job 82864 served
alpha input to a broadband-pretrained encoder, and job 83354 removed that mismatch by
pretraining on alpha first. Both landed at or below the score the pipeline produces from
pure noise input.

What is NOT in question is whether a small model can classify alpha: the baseline's own
EEGNet arm does that at 2.0x chance. What is untested is whether a small model can produce
usable CLIP-space embeddings, which is what the diffusion model actually consumes and what
the large encoder is failing to deliver.

So this trains an EEGNet-class backbone with a projection head onto CLIP image embeddings,
and reports Recall@1 over the distinct stimuli -- the same quantity Stage 2 logs as
val/retrieval_top1, so the numbers sit in the same table.

Usage
-----
    python code/small_encoder.py --subject 1 --band 8 13 --permute 200 \
        --json results/small_encoder_s1.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from classical_baseline import (
    load_split, concat_windows, prepare, _permute_by_stimulus,
)


# ------------------------------------------------------------------ CLIP targets

def clip_targets(stems, imagenet_dir, cache_path=None):
    """CLIP image embedding for each distinct stimulus, L2-normalised.

    Cached to disk: the encoder is retrained many times during a permutation run and
    the images never change.
    """
    uniq = sorted(set(stems))
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("stems") == uniq:
            print("[clip] loaded %d cached stimulus embeddings" % len(uniq))
            return blob["emb"], uniq

    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    print("[clip] embedding %d stimuli" % len(uniq))
    proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()

    imgs = []
    for s in uniq:
        synset = s.split('_')[0] if '_' in s else s
        path = os.path.join(imagenet_dir, synset, s)
        if not os.path.exists(path) and not path.endswith('.JPEG'):
            path += '.JPEG'
        imgs.append(Image.open(path).convert("RGB"))

    with torch.no_grad():
        batch = proc(images=imgs, return_tensors="pt")
        emb = model.get_image_features(**batch)
    emb = F.normalize(emb.float(), dim=-1)

    if cache_path:
        torch.save({"stems": uniq, "emb": emb}, cache_path)
    return emb, uniq


# ------------------------------------------------------------------------ model

class SmallEncoder(torch.nn.Module):
    """EEGNet backbone (Lawhern et al. 2018) with a projection onto CLIP space.

    The backbone is the same shape as the one in the classical baseline, so a difference
    in results is about the objective and the head rather than about the architecture.
    """

    def __init__(self, n_ch, n_time, out_dim, F1=8, D=2, F2=16, drop=0.5):
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
        feat = F2 * max(n_time // 16, 1)
        self.head = torch.nn.Linear(feat, out_dim)

    def forward(self, x):                      # x: (n, 1, channels, time)
        x = self.b1(self.c1(x))
        x = self.d1(self.p1(F.elu(self.b2(self.dw(x)))))
        x = self.d2(self.p2(F.elu(self.b3(self.sp(x)))))
        x = x.flatten(1)
        return F.normalize(self.head(x), dim=-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------- training

def info_nce(pred, target, key, temperature=0.07):
    """Contrastive loss over the stimuli in the batch.

    Windows of the same recording share a stimulus, so they are positives for each other
    rather than distractors. Treating them as negatives would ask the model to separate
    two clips of the same imagined picture, which is the opposite of the task.
    """
    logits = (pred @ target.T) / temperature
    pos = (key[:, None] == key[None, :]).float()
    logp = F.log_softmax(logits, dim=1)
    return -(torch.logsumexp(logp + torch.log(pos.clamp_min(1e-12)), dim=1)).mean()


def retrieval(pred, cand, key_true):
    """Recall@1 and Recall@5 against the candidate stimulus embeddings."""
    sims = pred @ cand.T
    order = sims.argsort(dim=1, descending=True)
    top1 = (order[:, 0] == key_true).float().mean().item()
    k5 = min(5, cand.shape[0])
    top5 = (order[:, :k5] == key_true[:, None]).any(dim=1).float().mean().item()
    return top1, top5


def pooled_retrieval(pred, cand, key_true):
    """One decision per stimulus: average the predictions of its windows first."""
    out, lab = [], []
    for k in sorted(set(key_true.tolist())):
        m = key_true == k
        out.append(F.normalize(pred[m].mean(0, keepdim=True), dim=-1))
        lab.append(k)
    p = torch.cat(out)
    t = torch.tensor(lab)
    return ((p @ cand.T).argmax(1) == t).float().mean().item()


def train_once(Xtr, key_tr, Xte, key_te, cand, epochs, lr, seed, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    xt = torch.from_numpy(Xtr).float().unsqueeze(1).to(dev)
    xe = torch.from_numpy(Xte).float().unsqueeze(1).to(dev)
    ktr = torch.as_tensor(key_tr).to(dev)
    kte = torch.as_tensor(key_te)
    C = cand.to(dev)

    model = SmallEncoder(xt.shape[2], xt.shape[3], C.shape[1]).to(dev)
    if verbose:
        print("[model] %d trainable parameters" % n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = xt.shape[0]
    bs = min(64, n)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = info_nce(model(xt[idx]), C[ktr[idx]], ktr[idx])
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                t1, _ = retrieval(model(xe).cpu(), cand, kte)
            print("  epoch %3d  loss %.4f  test Recall@1 %.4f" % (ep + 1, loss.item(), t1))

    model.eval()
    with torch.no_grad():
        pe = model(xe).cpu()
    t1, t5 = retrieval(pe, cand, kte)
    pooled = pooled_retrieval(pe, cand, kte)
    return {"recall_at_1": t1, "recall_at_5": t5, "pooled_recall_at_1": pooled,
            "n_params": n_params(model)}


# ------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Small EEG->CLIP encoder")
    p.add_argument("--dataset", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--splits", default="../datasets/imagination_5_95_std_splits_window_avail.pth")
    p.add_argument("--imagenet", default="../datasets/imageNet_images")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--band", type=float, nargs=2, default=[8.0, 13.0],
                   help="input band in Hz; '0 0' for the data as shipped")
    p.add_argument("--window_len", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--permute", type=int, default=0,
                   help="permutation null: refits on stimulus-shuffled labels this many times")
    p.add_argument("--cache", default="../datasets/clip_stimulus_emb.pt")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    tr, te, _ = load_split(args.dataset, args.splits, args.subject)
    tr = concat_windows(tr, args.window_len)
    te = concat_windows(te, args.window_len)
    band = None if args.band[0] <= 0 else (args.band[0], args.band[1])
    Xtr, tr, thresh = prepare(tr, band, False)
    Xte, te, _ = prepare(te, band, False, thresh)

    # candidates are the stimuli present in the test split
    stems_te = [t["image"] for t in te]
    cand, uniq = clip_targets(stems_te, args.imagenet, args.cache)
    sidx = {s: i for i, s in enumerate(uniq)}

    keep = [i for i, t in enumerate(tr) if t["image"] in sidx]
    Xtr = [Xtr[i] for i in keep]
    tr = [tr[i] for i in keep]
    key_tr = np.array([sidx[t["image"]] for t in tr])
    key_te = np.array([sidx[t["image"]] for t in te])

    Xtr = np.stack(Xtr)
    Xte = np.stack(Xte)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    chance = 1.0 / len(uniq)
    print("subject %d | band %s | train %d test %d | %d stimuli, chance %.4f"
          % (args.subject, band, len(Xtr), len(Xte), len(uniq), chance))

    res = train_once(Xtr, key_tr, Xte, key_te, cand, args.epochs, args.lr, 0, verbose=True)
    res.update({"subject": args.subject, "band": band, "n_train": len(Xtr),
                "n_test": len(Xte), "n_stimuli": len(uniq), "chance": chance})

    print("\n=== RESULT ===")
    print("  parameters        %d" % res["n_params"])
    print("  Recall@1          %.4f  (%.2fx chance)" % (res["recall_at_1"], res["recall_at_1"] / chance))
    print("  Recall@5          %.4f" % res["recall_at_5"])
    print("  pooled Recall@1   %.4f  (%.2fx chance)" % (res["pooled_recall_at_1"], res["pooled_recall_at_1"] / chance))

    if args.permute > 0:
        print("\nrunning %d permutations" % args.permute)
        null, null_pooled = [], []
        for k in range(args.permute):
            yp = _permute_by_stimulus(tr, key_tr, seed=3000 + k)
            r = train_once(Xtr, yp, Xte, key_te, cand, args.epochs, args.lr, seed=k)
            null.append(r["recall_at_1"])
            null_pooled.append(r["pooled_recall_at_1"])
            if (k + 1) % 20 == 0:
                print("  %d/%d  null mean so far %.4f (pooled %.4f)"
                      % (k + 1, args.permute, float(np.mean(null)), float(np.mean(null_pooled))))

        def summarise(vals, obs):
            v = np.array(vals)
            return {"n": args.permute,
                    "null_mean": float(v.mean()),
                    "null_p95": float(np.percentile(v, 95)),
                    "null_max": float(v.max()),
                    "p_value": float((np.sum(v >= obs) + 1.0) / (len(v) + 1.0))}

        # Pooling the windows of a recording before deciding is what a real system would do,
        # so it is the more relevant test. It had no null until now.
        res["permutation"] = summarise(null, res["recall_at_1"])
        res["permutation_pooled"] = summarise(null_pooled, res["pooled_recall_at_1"])

        for label, key in (("Recall@1", "permutation"), ("pooled  ", "permutation_pooled")):
            pm = res[key]
            print("  %s  null mean %.4f (chance %.4f)  p95 %.4f  max %.4f  ->  p = %.4f"
                  % (label, pm["null_mean"], chance, pm["null_p95"], pm["null_max"], pm["p_value"]))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
