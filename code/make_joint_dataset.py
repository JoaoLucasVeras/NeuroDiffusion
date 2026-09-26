"""
Build a joint perception + imagination dataset and a matching split.

Shimizu & Srinivasan (2022) report that training on both paradigms together raised
imagined-image classification from 13.4% to 25.2% (40-way, within subject). Both
sets share the cap, the four subjects and the 40 classes, so the 7,987 visual trials
are the largest source of extra training signal available without new recordings.

The split keeps the imagination test set exactly as in the per-paradigm protocol
(so results are comparable) and adds visual trials to TRAIN only:

    subject protocol : train = imagination + visual from the non-test subjects;
                       test  = imagination trials of the held-out subject.
                       Visual trials of the test subject are excluded as well, so
                       the encoder never sees that person's EEG statistics.
    window protocol  : train = imagination early windows + ALL usable visual trials;
                       test  = imagination late windows (with the gap), as before.

Each entry gets a 'paradigm' field ('visual' | 'imagination'). Visual trials have
window = -1 so the validation carve-out (dataset.split_val_from_train), which holds
out the LAST windows, never selects them.

Usage (from code/):
    python make_joint_dataset.py --protocol subject --require_images \\
        --imagenet_path ../datasets/imageNet_images
"""
import argparse
import collections
import os

import torch

from make_splits import filter_to_available, split_by_subject, split_by_window, audit


def main():
    p = argparse.ArgumentParser(description="Joint perception+imagination dataset and split")
    p.add_argument("--visual", default="../datasets/visual_5_95_std.pth")
    p.add_argument("--imagination", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--output", default="../datasets/joint_5_95_std.pth")
    p.add_argument("--protocol", choices=["subject", "window"], default="subject")
    p.add_argument("--test_subject", type=int, default=4)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--imagenet_path", default="../datasets/imageNet_images")
    p.add_argument("--require_images", action="store_true")
    # Per-subject ("Imagine"/"Mix") replication of Shimizu & Srinivasan 2022: the
    # imagination train/test both come from ONE subject; the perception data that is
    # mixed in comes from every subject, as in their Mix model.
    p.add_argument("--imagination_subject", type=int, default=0, help="0 = all subjects")
    p.add_argument("--visual_subject", type=int, default=0, help="0 = all subjects (paper's Mix)")
    args = p.parse_args()

    vis = torch.load(args.visual, map_location="cpu")
    ima = torch.load(args.imagination, map_location="cpu")
    if list(vis["labels"]) != list(ima["labels"]):
        raise SystemExit("visual and imagination class lists differ; labels would not be comparable")

    dataset = []
    for e in ima["dataset"]:
        d = dict(e); d["paradigm"] = "imagination"; dataset.append(d)
    n_ima = len(dataset)
    for e in vis["dataset"]:
        d = dict(e); d["paradigm"] = "visual"; d["window"] = -1; dataset.append(d)
    print("imagination %d + visual %d = %d trials" % (n_ima, len(vis["dataset"]), len(dataset)))

    images = sorted({e["image"] for e in dataset})
    torch.save({"dataset": dataset, "labels": ima["labels"], "images": images}, args.output)
    print("wrote", args.output, "(%d unique stimuli)" % len(images))

    # ---- split: imagination decides train/test; visual joins train only ----
    import numpy as np
    rng = np.random.RandomState(args.seed)
    ima_entries = dataset[:n_ima]
    if args.protocol == "subject":
        tr, te, desc = split_by_subject(ima_entries, args.test_subject, rng)
        vis_train = [n_ima + i for i, e in enumerate(dataset[n_ima:]) if e["subject"] != args.test_subject]
    else:
        tr, te, desc = split_by_window(ima_entries, args.test_frac, args.gap, rng)
        vis_train = list(range(n_ima, len(dataset)))
    if args.imagination_subject:
        S = args.imagination_subject
        tr = [i for i in tr if dataset[i]["subject"] == S]
        te = [i for i in te if dataset[i]["subject"] == S]
        desc += "; imagination restricted to subject %d" % S
    if args.visual_subject:
        V = args.visual_subject
        vis_train = [i for i in vis_train if dataset[i]["subject"] == V]
        desc += "; perception restricted to subject %d" % V
    train = tr + vis_train
    test = te
    desc = "joint perception+imagination; imagination " + desc

    if args.require_images:
        usable = set(filter_to_available(dataset, args.imagenet_path)[0])
        train = [i for i in train if i in usable]
        test = [i for i in test if i in usable]

    audit(dataset, train, test, desc)
    par = collections.Counter(dataset[i]["paradigm"] for i in train)
    print("  train paradigms :", dict(par))
    print("  test paradigms  :", dict(collections.Counter(dataset[i]["paradigm"] for i in test)))

    suffix = "_splits_%s" % args.protocol
    if args.imagination_subject:
        suffix += "_s%d" % args.imagination_subject
    suffix += "_avail" if args.require_images else ""
    out = args.output.replace(".pth", suffix + ".pth")
    torch.save({"splits": [{"train": train, "test": test}], "protocol": args.protocol,
                "description": desc, "source_dataset": os.path.basename(args.output),
                "seed": args.seed, "require_images": bool(args.require_images)}, out)
    print("  wrote", out)


if __name__ == "__main__":
    main()
