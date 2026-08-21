# Runbook

Order of operations, and what each stage needs. See [DATA.md](DATA.md) for why the data
pipeline looks the way it does.

## Stage 0 — build and validate the data (no GPU)

```bash
python code/prepare_shimizu_data.py --datasets_dir datasets
python code/make_splits.py --dataset datasets/imagination_5_95_std.pth --protocol subject --require_images
python code/make_splits.py --dataset datasets/visual_5_95_std.pth      --protocol image --require_images
python code/check_data.py  --dataset datasets/imagination_5_95_std.pth \
                           --splits  datasets/imagination_5_95_std_splits_subject_avail.pth
```

`prepare_shimizu_data.py` refuses to write a dataset whose decoded stimulus filenames
disagree with the integer labels, so a silent mis-pairing cannot reach training.

Then, **on the login node only** (compute nodes are air-gapped):

```bash
python code/precache_models.py
```

Four different classes are loaded from `openai/clip-vit-large-patch14` across the codebase
— the dataset's processor, the CLIP-tune vision encoder, SD1.5's text conditioning, and
the evaluation model — and they pull different files. With `TRANSFORMERS_OFFLINE=1` set on
the compute nodes, anything not already cached raises at job startup.

## Stage 1 — masked EEG pre-training (needs NO stimulus images)

```bash
sbatch train_mae.sh
```

**Why this works without the images.** Stage 1 trains a masked autoencoder on the EEG
signal alone: it masks a fraction of the EEG patches and learns to reconstruct them from
the surrounding ones. The objective is EEG-to-EEG. The stimulus image is never an input
and never a target — it is only needed in Stage 2, where the EEG embedding has to be
aligned to an image. `stageA1_eeg_pretrain.py` now passes `load_images=False`, so the
dataset skips image IO entirely (which also removes a CLIP preprocess from every sample,
making the epochs faster). The preflight in `train_mae.sh` runs with `--skip_image_check`
for the same reason.

So the encoder you get out of Stage 1 is complete and reusable — nothing about it will
need redoing once the images arrive.

Output: `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth`

## Stage 1.5 — fetch the stimulus images (blocking for Stage 2)

You need 1926 specific ImageNet files across 40 synsets (~250 MB on disk), listed in
`missing_stimuli.txt`. The 40 imagination stimuli are a subset of the 2000 visual ones,
so one fetch covers both experiments.

### Option A: download directly (no credentials)

```bash
python code/download_stimuli.py
```

image-net.org serves whole-synset tars at `/data/winter21_whole/` without login, and they
preserve the original ImageNet filenames this dataset references. Most redistributions
(HuggingFace `imagenet-1k`, resized mirrors) re-shard into parquet and drop those names,
so they cannot satisfy the manifest — and `ILSVRC/imagenet-1k` is gated (401) anyway.

Each tar is streamed and discarded; only wanted members are written. Expect several GB of
transfer for ~250 MB of images, since the wanted files are spread through each archive.
Resumable — re-running fetches only what is still outstanding.

**Coverage is partial.** The winter21 release dropped some ILSVRC2012 images. Measured
result of a full run: **1590 of 1926 files (82.6%)**, 5.73 GB transferred, ~58 minutes.

| | stimuli | trials | classes |
|---|---|---|---|
| visual | 1700/2000 (85%) | 6789/7987 | **40/40** |
| imagination | 33/40 (82.5%) | 2640/3200 | **33/40** |

The visual experiment is unaffected. The imagination experiment loses 7 classes outright,
because it has exactly one stimulus per class -- a missing image means a missing class.
Chance goes from 2.5% (40-way) to 3.03% (33-way). The 7 files are listed in
`imagination_missing.txt`; they are genuinely absent from winter21 (verified by a full
re-read of those 7 tars), so only a real ILSVRC2012 copy will recover them.

### Option B: from an ImageNet copy you already have

```bash
python code/fetch_stimuli.py --from_dir  /path/to/ImageNet/train           # extracted tree
python code/fetch_stimuli.py --from_tars /path/to/ILSVRC2012_img_train.tar # tars
```

A real ILSVRC2012 copy has full coverage. If you have one (or the cluster does), prefer it.

### Partial coverage

Do **not** train with missing stimuli — a blank target is what invalidated the earlier run.
Instead, exclude those trials from the splits so every remaining trial is honest:

```bash
python code/make_splits.py --dataset datasets/imagination_5_95_std.pth \
    --protocol subject --require_images
python code/make_splits.py --dataset datasets/visual_5_95_std.pth \
    --protocol image --require_images
```

This writes `..._splits_<protocol>_avail.pth` and reports how many trials and classes
survive. The dataset `.pth` stays canonical; only the splits narrow. The SLURM scripts
default to these via `SPLIT_SUFFIX=_avail`; set `SPLIT_SUFFIX=""` once you have full
coverage.

Confirm before spending GPU time:

```bash
python code/check_data.py --dataset datasets/imagination_5_95_std.pth \
    --splits datasets/imagination_5_95_std_splits_subject_avail.pth
python code/check_data.py --dataset datasets/visual_5_95_std.pth \
    --splits datasets/visual_5_95_std_splits_image_avail.pth
```

The image check is scoped to the trials the splits actually reach, so an `_avail` split
passes cleanly while the unfiltered one still fails.

## Stage 2 — LDM fine-tuning (both experiments, side by side)

```bash
sbatch --export=ALL,EXPERIMENT=visual,PROTOCOL=image        hpc_submit_stage2.sh
sbatch --export=ALL,EXPERIMENT=imagination,PROTOCOL=subject hpc_submit_stage2.sh
```

Each job preflights its data and aborts before allocating the GPU if anything is wrong.
Each writes `samples.npz` and runs `eval_report.py` on completion.

| | visual | imagination |
|---|---|---|
| trials | 7987 | 3200 |
| stimuli | 2000 | 40 (one per class) |
| protocol | stimulus-disjoint | leave-one-subject-out |
| usable trials | 6789 | 2640 |
| classes | 40 | 33 |
| chance | 2.50% | 3.03% |
| supports | instance-level reconstruction | class-level decoding only |
| role | sanity ceiling | the result of interest |

## Stage 3 — evaluation and the comparison table

Each Stage 2 job already produces its own report. To score an arbitrary checkpoint:

```bash
sbatch hpc_submit_eval.sh /path/to/checkpoint.pth
```

Then put the two runs side by side:

```bash
python code/compare_reports.py \
    --reports results/generation/<visual_run>/report.json \
              results/generation/<imag_run>/report.json \
    --names visual imagination \
    --output results/comparison
```

### What the metrics mean

`eval_report.py` scores the model against two controls:

- **noise** — random pixels. Floor for "did anything render at all".
- **shuffled** — your own generations scored against the *wrong* ground truth. This is the
  one that matters: it holds image quality constant and removes only the EEG-stimulus
  correspondence. If the model does not beat shuffled, it is producing plausible pictures
  unrelated to the brain signal.

The verdict line fires when the model exceeds the shuffled control by more than 2 pooled
SEMs. The harness was validated on synthetic controls: same-class generations scored
22.3 SEM (verdict: signal present), random-class generations scored 1.6 SEM (verdict: not
distinguishable).

## Things that will bite you

- `argparse` booleans use a real parser now, so `--use_time_cond False` means False. It
  previously evaluated to `True` because `bool("False")` is `True`.
- A missing stimulus file used to become a black square. It now raises. Pass
  `--strict_images False` only if you have deliberately accepted the loss.
- Never use a random trial-level split on the imagination set: each recording is cut into
  20 overlapping windows, so a shuffle puts near-duplicates on both sides.
