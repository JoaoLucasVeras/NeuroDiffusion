"""
Convert the Shimizu (2022) EEG .mat files into the .pth format the training code expects.

Both experiments live in the same MATLAB v7.3 layout:

    data          (125, 128, N)   float64   time x channel x trial
    labels        (N, 1)          int64     class index, 0..39
    images        (N*30, 1)       uint16    ASCII codes, 30 chars per trial
    window_label  (N, 1)          int64     imagination only: window index 0..19

The `images` field is the part that has historically been misread. It is NOT a
lookup index -- it is a flattened fixed-width character matrix. Reshaping to
(N, 30) and mapping each row through chr() recovers the literal stimulus path,
e.g. "n11939491/n11939491_20025.JPEG" (short names are NUL-padded to 30).

Every conversion is gated on a hard assertion: the synset recovered from the
decoded path must equal SYNSETS[label] for all N trials. If the decode were
wrong the agreement would be at chance, so a clean pass is proof the EEG trials
and stimulus filenames are correctly paired -- not an assumption.

Usage:
    python code/prepare_shimizu_data.py --datasets_dir datasets
"""
import argparse
import collections
import os

import h5py
import numpy as np
import torch

NAME_WIDTH = 30  # fixed character width of the encoded stimulus paths


def decode_image_names(images_raw, n_trials):
    """Decode the uint16 ASCII matrix into one 'synset/file.JPEG' string per trial."""
    flat = np.asarray(images_raw).flatten()
    if flat.size != n_trials * NAME_WIDTH:
        raise ValueError(
            f"images field has {flat.size} entries, expected {n_trials * NAME_WIDTH} "
            f"({n_trials} trials x {NAME_WIDTH} chars)"
        )
    rows = flat.reshape(n_trials, NAME_WIDTH)
    names = []
    for row in rows:
        s = "".join(chr(int(c)) for c in row)
        names.append(s.rstrip("\x00").strip())
    return names


def validate_pairing(names, labels, synsets):
    """Assert decoded synsets agree with the integer labels on every trial."""
    bad = [
        (i, n, int(l))
        for i, (n, l) in enumerate(zip(names, labels))
        if n.split("/")[0] != synsets[int(l)]
    ]
    if bad:
        preview = "\n    ".join(
            f"trial {i}: decoded '{n}' but label {l} -> '{synsets[l]}'" for i, n, l in bad[:5]
        )
        raise AssertionError(
            f"Stimulus/label disagreement on {len(bad)}/{len(names)} trials. "
            f"The image decode is wrong -- refusing to write a corrupt dataset.\n    {preview}"
        )


def load_subject(path, expect_window_label):
    with h5py.File(path, "r") as f:
        data = np.array(f["data"])                       # (125, 128, N)
        eeg = np.transpose(data, (2, 1, 0))              # (N, 128, 125)
        labels = np.array(f["labels"]).flatten().astype(int)
        n = len(labels)
        names = decode_image_names(f["images"], n)
        if expect_window_label:
            if "window_label" not in f:
                raise KeyError(f"{path} has no window_label field")
            windows = np.array(f["window_label"]).flatten().astype(int)
        else:
            windows = np.zeros(n, dtype=int)
    return eeg, labels, names, windows


def convert(mat_paths, output_file, expect_window_label, imagenet_dir):
    loaded = []
    for path in mat_paths:
        subject_id = int("".join(ch for ch in os.path.basename(path) if ch.isdigit())[:3])
        loaded.append((subject_id, load_subject(path, expect_window_label)))
        print(f"  read {os.path.basename(path)}: {len(loaded[-1][1][1])} trials")

    # Canonical class list: the 40 synsets, sorted. Labels index into this.
    synsets = sorted({n.split("/")[0] for _, (_, _, names, _) in loaded for n in names})
    if len(synsets) != 40:
        raise AssertionError(f"Expected 40 synsets, decoded {len(synsets)}")

    dataset = []
    for subject_id, (eeg, labels, names, windows) in loaded:
        validate_pairing(names, labels, synsets)
        print(f"  subject {subject_id}: stimulus/label agreement {len(names)}/{len(names)} OK")
        for i in range(len(labels)):
            stem = os.path.splitext(os.path.basename(names[i]))[0]
            dataset.append(
                {
                    "eeg": torch.from_numpy(eeg[i]).float(),
                    "label": int(labels[i]),
                    "image": stem,                 # e.g. "n11939491_20025"
                    "subject": subject_id,
                    "window": int(windows[i]),
                }
            )

    images = sorted({e["image"] for e in dataset})
    payload = {"dataset": dataset, "labels": synsets, "images": images}
    torch.save(payload, output_file)

    print(f"\n  wrote {output_file}")
    print(f"    trials          : {len(dataset)}")
    print(f"    unique stimuli  : {len(images)}")
    print(f"    subjects        : {sorted({e['subject'] for e in dataset})}")
    print(f"    trials/stimulus : {sorted(collections.Counter(e['image'] for e in dataset).values())[0]}"
          f"..{sorted(collections.Counter(e['image'] for e in dataset).values())[-1]}")
    report_missing(images, imagenet_dir)
    return payload


def report_missing(images, imagenet_dir):
    """Report which stimulus files are absent, since a miss silently poisons training."""
    if imagenet_dir is None or not os.path.isdir(imagenet_dir):
        print(f"    stimulus files  : SKIPPED (no imagenet dir at {imagenet_dir})")
        return
    missing = [
        s for s in images
        if not os.path.exists(os.path.join(imagenet_dir, s.split("_")[0], s + ".JPEG"))
    ]
    have = len(images) - len(missing)
    print(f"    stimulus files  : {have}/{len(images)} present in {imagenet_dir}")
    if missing:
        print(f"    !! {len(missing)} MISSING -- training targets for these trials would be blank.")
        print(f"       run: python code/check_data.py  for the full manifest")


def resolve_imagenet_dir(root):
    """Tolerate the common double-nested imageNet_images/imageNet_images layout."""
    outer = os.path.join(root, "imageNet_images")
    inner = os.path.join(outer, "imageNet_images")
    if os.path.isdir(inner):
        return inner
    return outer if os.path.isdir(outer) else None


def main():
    p = argparse.ArgumentParser(description="Convert Shimizu EEG .mat files to .pth")
    p.add_argument("--datasets_dir", default="datasets")
    p.add_argument("--experiment", choices=["imagination", "visual", "both"], default="both")
    args = p.parse_args()

    imagenet_dir = resolve_imagenet_dir(args.datasets_dir)

    if args.experiment in ("imagination", "both"):
        print("=== Imagination Experiment ===")
        paths = sorted(
            os.path.join(args.datasets_dir, "Imagination Experiment", f)
            for f in os.listdir(os.path.join(args.datasets_dir, "Imagination Experiment"))
            if f.endswith(".mat")
        )
        convert(paths, os.path.join(args.datasets_dir, "imagination_5_95_std.pth"),
                expect_window_label=True, imagenet_dir=imagenet_dir)

    if args.experiment in ("visual", "both"):
        print("\n=== Visual Experiment ===")
        paths = sorted(
            os.path.join(args.datasets_dir, "Visual Experiment", f)
            for f in os.listdir(os.path.join(args.datasets_dir, "Visual Experiment"))
            if f.endswith(".mat")
        )
        convert(paths, os.path.join(args.datasets_dir, "visual_5_95_std.pth"),
                expect_window_label=False, imagenet_dir=imagenet_dir)


if __name__ == "__main__":
    main()
