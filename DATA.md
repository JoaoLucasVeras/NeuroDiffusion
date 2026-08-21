# Data pipeline

Everything in this document is checkable with `python code/check_data.py`. Run it before
submitting any job.

## What the raw data actually contains

Both Shimizu experiments ship as MATLAB v7.3 (HDF5) files with the same layout:

| field | shape | meaning |
|---|---|---|
| `data` | `(125, 128, N)` | time x channel x trial. 125 samples @ 250 Hz = one 500 ms window |
| `labels` | `(N, 1)` | class index, **0-39** |
| `images` | `(N*30, 1)` | **ASCII character codes**, 30 chars per trial, NUL-padded |
| `window_label` | `(N, 1)` | imagination only: window index 0-19 |

### The `images` field

This is the field that caused the silent Stage 2 failure. It is *not* a lookup index into
an image list. It is a flattened fixed-width character matrix. Reshape to `(N, 30)` and map
each row through `chr()`:

```
[110 49 49 57 51 57 52 57 49 47 110 ...] -> "n11939491/n11939491_20025.JPEG"
```

An earlier converter read `images[::30]`, which takes only the *first character* of every
row — always `110` (`'n'`). Every trial was therefore labelled `img_110`, that path never
resolved, and `dataset.py` substituted a black 512x512 square. All 3200 trials trained
against an identical blank target while the loss curve looked healthy.

The decode is verified rather than assumed: `prepare_shimizu_data.py` asserts that the
synset embedded in every decoded filename equals `SYNSETS[label]`. A wrong decode would
agree at chance; the actual agreement is **800/800 per imagination subject** and
**2000/2000 per visual subject**.

## Dataset shapes

| | trials | unique stimuli | structure |
|---|---|---|---|
| imagination | 3200 | **40** (one per class) | 4 subjects x 40 stimuli x 20 sliding windows |
| visual | 7987 | 2000 (50 per class) | 4 subjects x ~2000 stimuli, 1 trial each |

The imagination set has exactly one stimulus per class, so **image identity is class
identity**. Instance-level reconstruction cannot be measured on it — only class-level
decoding. The visual set has 2000 distinct stimuli and does support stimulus-disjoint
evaluation.

## Splits

`make_splits.py` offers three protocols. It prints a leakage audit for each.

| protocol | claim it supports | notes |
|---|---|---|
| `subject` | "decodes an unseen person's EEG" | leave-one-subject-out. Default. Stimuli repeat across the split, so report class-level metrics only. |
| `window` | "decodes later timepoints" | within-subject, late windows held out with a gap. Weakest — adjacent windows are highly correlated. |
| `image` | "generalises to unseen stimuli" | strictest. Correct choice for the **visual** set. On the imagination set it is zero-shot over held-out classes, so expect near-chance results. |

The previous `imagination_splits.pth` shuffled all 3200 trials and sliced 80/20. Because
each recording is cut into 20 near-duplicate windows, **154 of 154 test recordings also
appeared in training**. That file has been removed.

## Rebuilding from scratch

```bash
python code/prepare_shimizu_data.py --datasets_dir datasets
python code/make_splits.py --dataset datasets/imagination_5_95_std.pth --protocol subject
python code/make_splits.py --dataset datasets/visual_5_95_std.pth      --protocol image
python code/check_data.py  --dataset datasets/imagination_5_95_std.pth \
                           --splits  datasets/imagination_5_95_std_splits_subject.pth
```

## Outstanding blocker: the stimulus images

`datasets/imageNet_images` is the DreamDiffusion ImageNet subset. It is **not** the Shimizu
stimulus set:

| needed by | unique stimuli | present on disk |
|---|---|---|
| imagination | 40 | **1** |
| visual | 2000 | **74** |

Stage 2 cannot produce a meaningful model until these are supplied. Stage 1 (masked EEG
pre-training) is self-supervised and runs fine without them.

`check_data.py` writes the exact manifest of what is missing, and can import the files from
an ImageNet tree you already have:

```bash
python code/check_data.py --dataset datasets/imagination_5_95_std.pth \
    --import_from /path/to/ImageNet/train
```

Many university clusters host ImageNet-1k in a shared read-only directory — check there
before downloading anything.
