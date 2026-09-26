"""
Build leakage-free train/test splits for the Shimizu EEG datasets.

Why this file exists
--------------------
The previous splits were produced by shuffling all trials and slicing 80/20.
For the imagination set that is catastrophic: each stimulus was recorded once
per subject and then cut into 20 overlapping 500 ms windows, so a random split
puts windows from the *same* recording of the *same* image on both sides. The
model can then match test trials it has effectively already seen, and every
downstream metric is inflated.

Three honest protocols are provided. Pick based on the claim you want to make:

  subject  Leave-one-subject-out. Train on N-1 subjects, test on the held-out
           one. Claim: "decodes an unseen person's EEG". Stimuli repeat across
           the split (all subjects saw the same 40 images), so report
           class-level metrics only.

  window   Within-subject window holdout. Early windows train, late windows
           test, with a gap so the test windows do not overlap training ones.
           Claim: "decodes later timepoints of a recording". Weakest of the
           three -- adjacent windows are highly correlated.

  image    Stimulus-disjoint. No image appears on both sides. Claim: "generalises
           to unseen stimuli". Strictest. On the imagination set this is
           zero-shot over held-out classes (image identity == class identity),
           so expect near-chance results; it is the right protocol for the
           visual set, which has 2000 distinct stimuli.

Usage:
    python code/make_splits.py --dataset datasets/imagination_5_95_std.pth --protocol subject
    python code/make_splits.py --dataset datasets/visual_5_95_std.pth --protocol image
"""
import argparse
import collections
import os

import numpy as np
import torch


def resolve_imagenet_dir(path):
    if path is None:
        return None
    nested = os.path.join(path, 'imageNet_images')
    if os.path.isdir(nested) and any(
            d.startswith('n') for d in os.listdir(nested)
            if os.path.isdir(os.path.join(nested, d))):
        return nested
    return path


def filter_to_available(entries, imagenet_dir):
    """Return the indices whose stimulus file actually exists on disk.

    Partial stimulus coverage is common (image-net.org's winter21 release dropped some
    ILSVRC2012 images). Excluding those trials from the splits keeps every remaining
    trial honest, which is strictly better than letting dataset.py substitute a blank
    target -- that is the failure mode that silently invalidated an entire training run.
    """
    root = resolve_imagenet_dir(imagenet_dir)
    if root is None or not os.path.isdir(root):
        raise SystemExit("--require_images given but no ImageNet dir at %s" % imagenet_dir)
    ok, dropped_stims = [], set()
    for i, e in enumerate(entries):
        stem = e['image']
        if os.path.exists(os.path.join(root, stem.split('_')[0], stem + '.JPEG')):
            ok.append(i)
        else:
            dropped_stims.add(stem)
    return ok, dropped_stims


def split_by_subject(entries, test_subject, rng):
    subjects = sorted({e["subject"] for e in entries})
    if test_subject not in subjects:
        raise ValueError(f"test_subject {test_subject} not in {subjects}")
    train = [i for i, e in enumerate(entries) if e["subject"] != test_subject]
    test = [i for i, e in enumerate(entries) if e["subject"] == test_subject]
    return train, test, f"leave-one-subject-out (held out subject {test_subject})"


def split_by_window(entries, test_frac, gap, rng):
    windows = sorted({e["window"] for e in entries})
    if len(windows) < 4:
        raise ValueError(
            "window protocol needs per-trial window indices; this dataset has "
            f"only {len(windows)} distinct window value(s). Use --protocol subject or image."
        )
    n_test = max(1, int(round(len(windows) * test_frac)))
    test_windows = set(windows[-n_test:])
    # Drop `gap` windows before the test block so train/test do not overlap in time.
    excluded = set(windows[-(n_test + gap):-n_test]) if gap else set()
    train = [i for i, e in enumerate(entries)
             if e["window"] not in test_windows and e["window"] not in excluded]
    test = [i for i, e in enumerate(entries) if e["window"] in test_windows]
    return train, test, (f"window holdout (test windows {sorted(test_windows)}, "
                         f"{len(excluded)} window(s) dropped as a gap)")


def split_by_image(entries, test_frac, rng):
    images = sorted({e["image"] for e in entries})
    rng.shuffle(images)
    n_test = max(1, int(round(len(images) * test_frac)))
    test_images = set(images[:n_test])
    train = [i for i, e in enumerate(entries) if e["image"] not in test_images]
    test = [i for i, e in enumerate(entries) if e["image"] in test_images]
    return train, test, f"stimulus-disjoint ({len(test_images)}/{len(images)} images held out)"


def audit(entries, train, test, protocol):
    """Print exactly what leaks and what does not, so the protocol is never assumed."""
    def keys(idx, fn):
        return {fn(entries[i]) for i in idx}

    tr_img, te_img = keys(train, lambda e: e["image"]), keys(test, lambda e: e["image"])
    tr_sub, te_sub = keys(train, lambda e: e["subject"]), keys(test, lambda e: e["subject"])
    tr_cls, te_cls = keys(train, lambda e: e["label"]), keys(test, lambda e: e["label"])
    tr_rec = keys(train, lambda e: (e["subject"], e["image"]))
    te_rec = keys(test, lambda e: (e["subject"], e["image"]))

    print(f"\n  protocol        : {protocol}")
    print(f"  train / test    : {len(train)} / {len(test)} trials")
    print(f"  subjects        : train {sorted(tr_sub)} | test {sorted(te_sub)}")
    print(f"  classes         : train {len(tr_cls)} | test {len(te_cls)} | shared {len(tr_cls & te_cls)}")
    print(f"  stimuli         : train {len(tr_img)} | test {len(te_img)} | shared {len(tr_img & te_img)}")
    shared_rec = tr_rec & te_rec
    print(f"  (subject,stimulus) recordings on both sides: {len(shared_rec)}"
          f"{'   <-- WINDOW LEAKAGE' if shared_rec and 'window' not in protocol else ''}")
    if not (tr_cls & te_cls):
        print("  NOTE: no class overlap -- this is a zero-shot protocol, expect near-chance metrics.")
    if tr_img & te_img:
        print("  NOTE: stimuli repeat across the split. Report class-level metrics, "
              "not instance-level reconstruction scores.")
    chance = 1.0 / max(1, len(te_cls))
    print(f"  chance accuracy : {chance:.4f} ({len(te_cls)}-way)")


def main():
    p = argparse.ArgumentParser(description="Build leakage-free EEG splits")
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["subject", "window", "image"], default="subject")
    p.add_argument("--test_subject", type=int, default=4)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--gap", type=int, default=1, help="windows dropped between train and test blocks")
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--output", default=None)
    p.add_argument("--imagenet_path", default="datasets/imageNet_images")
    p.add_argument("--require_images", action="store_true",
                   help="exclude trials whose stimulus file is not on disk")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    payload = torch.load(args.dataset, map_location="cpu")
    entries = payload["dataset"]
    print(f"Loaded {len(entries)} trials from {args.dataset}")

    usable = None
    if args.require_images:
        usable, dropped = filter_to_available(entries, args.imagenet_path)
        n_cls_before = len({e['label'] for e in entries})
        n_cls_after = len({entries[i]['label'] for i in usable})
        print(f"  stimulus filter: {len(usable)}/{len(entries)} trials usable "
              f"({len(dropped)} stimuli absent)")
        print(f"  classes retained: {n_cls_after}/{n_cls_before}")
        if not usable:
            raise SystemExit("No trials have an available stimulus -- fetch images first.")

    if args.protocol == "subject":
        train, test, desc = split_by_subject(entries, args.test_subject, rng)
    elif args.protocol == "window":
        train, test, desc = split_by_window(entries, args.test_frac, args.gap, rng)
    else:
        train, test, desc = split_by_image(entries, args.test_frac, rng)

    if usable is not None:
        keep = set(usable)
        train = [i for i in train if i in keep]
        test = [i for i in test if i in keep]

    if not train or not test:
        raise SystemExit(f"Empty split (train={len(train)}, test={len(test)}) -- check arguments.")

    audit(entries, train, test, desc)

    suffix = f"_splits_{args.protocol}" + ("_avail" if args.require_images else "")
    out = args.output or args.dataset.replace(".pth", f"{suffix}.pth")
    torch.save(
        {
            "splits": [{"train": train, "test": test}],
            "protocol": args.protocol,
            "description": desc,
            "source_dataset": os.path.basename(args.dataset),
            "seed": args.seed,
            "require_images": bool(args.require_images),
        },
        out,
    )
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
