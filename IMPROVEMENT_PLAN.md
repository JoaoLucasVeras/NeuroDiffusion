# Improvement plan: imagination decoding

Written 2026-09-21, after the first full Stage 1 + Stage 2 runs on the Shimizu data.

## Where we stand

| Run | Protocol | Result |
|---|---|---|
| Visual (`02-09-2026-23-07-56`) | stimulus-disjoint, all 4 subjects | **Positive.** CLIP sim 0.585 vs shuffled 0.548 (+4.3 SEM); 40-way 9.5% vs 2.5% chance |
| Imagination (`02-09-2026-20-29-28`) | leave-one-subject-out (LOSO) | First eval invalid (10 stimuli x 20 windows). Stratified re-eval = job 80178 |
| Imagination, window protocol | within-subject, late windows held out | Training = job 80181 (gpuql). Upper-bound control |

The pipeline is validated by the visual result. The open question is imagination.

## What the first imagination run tells us (evidence, not guesses)

1. **The CLIP alignment loss was memorised, not learned.**
   `train/loss_clip` went 0.22 -> 0.0001 by epoch 35. A diagnostic on 48 random
   stimuli (`code/clip_embed_diag.py`) shows CLIP ViT-L/14 image embeddings have
   mean pairwise cosine 0.54, and predicting the *mean* embedding scores
   `1 - cos = 0.26` -- exactly where training started. Driving it to 1e-4 means the
   network learned which of the 33 training targets each window belongs to.
   Cross-subject, that cannot transfer.

2. **Capacity vs data is absurd.** Stage 2 trains the entire 24-layer, 1024-d EEG
   encoder (~300M params) plus the U-Net cross-attention (`attn2`, `norm2`,
   `time_embed_condition`) on ~2,000 windows drawn from 33 stimuli.

3. **No regularisation reaches the optimiser.** `config.weight_decay = 0.05`
   exists but `configure_optimizers` builds `AdamW(params, lr=lr)` -- weight decay
   defaults to 0.01 and the config value is ignored. No dropout, no drop-path.

4. **Training is blind.** `validation_step` only evaluates `cls_net`, which is not
   trained (`cls_tune=False`). No held-out `loss_simple` or `loss_clip` is ever
   logged. We used the epoch-199 checkpoint with no basis for choosing it.

5. **The evaluation is expensive and coarse.** 250 PLMS steps x 200 trials ~ 4 h
   on an A100, and the readout is a CLIP similarity delta. That is fine for a
   final number, not for iterating.

## Priority list

Ordered by (expected gain) / (effort). Each item names the question it answers.

### P0 -- Cheap retrieval metric on the held-out split (no diffusion)
*Answers: does the EEG encoder carry any stimulus information at all?*

Add to `validation_step`: encode the held-out batch, map to CLIP space via
`mapping`, and compute (a) `val/loss_clip`, (b) n-way retrieval accuracy against
the CLIP embeddings of the test stimuli (33-way for imagination). Log every epoch.
Costs seconds. Also lets us pick the checkpoint by held-out retrieval rather than
"last epoch". This is the instrument every later experiment is measured with.

Files: `code/dc_ldm/models/diffusion/ddpm.py` (`validation_step`,
`full_validation`), `code/eeg_ldm.py` (val dataloader, `check_val_every_n_epoch`).

### P1 -- Regularise Stage 2
*Answers: is the null caused by overfitting rather than absence of signal?*

- Pass `weight_decay` through to AdamW (one-line fix).
- Freeze the first N encoder blocks (start with 18 of 24), train the last blocks +
  `channel_mapper` + `dim_mapper` + `mapping` + cross-attention.
- EEG augmentation in `EEGDataset.__getitem__` during training only:
  Gaussian noise (sigma ~ 0.1 of channel std), random channel dropout (10%),
  the existing temporal jitter, amplitude scaling (0.9-1.1). Mixup between
  windows of the *same* stimulus.
- Early stopping on the P0 retrieval metric with patience ~20 epochs.
  200 epochs is almost certainly far past the optimum.

### P2 -- Contrastive CLIP loss with an explicit weight
*Answers: does a batch-relative objective generalise better than pointwise cosine?*

`clip_loss()` (InfoNCE, symmetric) is already in `ldm_for_eeg.py` but unused.
Switch `get_clip_loss` to it with a learnable temperature, and add
`config.clip_weight` (start 1.0, sweep {1, 5, 10}). Because batches of 8 will
often contain several windows of the same stimulus, mask same-stimulus pairs out
of the negatives (supervised-contrastive style) -- otherwise the loss punishes
correct matches.

### P3 -- Stage 1 on all EEG (imagination + visual = 11,187 trials)
*Answers: does a better subject-invariant encoder help cross-subject transfer?*

`train_mae.sh` currently pretrains on imagination only (3,200 trials). Same
cap, same subjects, same preprocessing -- concatenate the two `.pth` files.
Keep the LOSO test subject out of pretraining too, or the encoder sees the
test subject's statistics. 3.5x more pretraining data at zero labelling cost.

### P4 -- Subject handling
*Answers: is the gap a subject-shift problem?*

- Per-subject, per-channel z-scoring (check what `5_95_std` already does).
- Few-shot calibration: after LOSO training, fine-tune only `channel_mapper` /
  `dim_mapper` on the test subject's *training* windows (windows 0-14),
  evaluate on windows 16-19. This is exactly the setting for your own data:
  a pretrained model plus a short calibration session.
- Run all 4 LOSO folds. One held-out subject is n=1; the honest number is the
  mean over folds.

### P5 -- Evaluation power
- Average predictions across the 20 windows of a stimulus before scoring
  ("trial averaging"). Standard in EEG decoding; boosts SNR ~ sqrt(20).
- Report per-class results; some categories (faces, scenes) are known to be
  far more decodable than others. A whole-set null can hide a strong subset.
- Bootstrap CIs on the CLIP delta instead of pooled-SEM.

### P6 -- Sampling
- `cfg_scale=8` was chosen for the imagination paradigm without a sweep.
  Sweep {3, 5, 8, 12} on a fixed checkpoint once P0 exists.
- 250 PLMS steps -> 50 DDIM steps for iteration runs (5x faster, negligible
  quality loss). Keep 250 for final numbers.

## What "the results we want" requires, honestly

Your end goal is decoding *your own* imagery. What this dataset can prove:

- **Window protocol (running now):** can the pipeline decode imagined category
  from EEG at all when subject and session are fixed? If this is null after P0-P2,
  the Shimizu imagination signal is below what this architecture can extract, and
  the path forward is your own data with a design built for decodability
  (few classes, many trials, interleaved order, session holdout).
- **LOSO protocol:** does it transfer to a new person zero-shot? Nice to have;
  not required for the personal goal. P4 few-shot calibration is the realistic
  version.

The visual result already shows the architecture extracts category information
from this EEG cap. Whether imagery survives the same treatment is what the next
two runs answer.

## What the field has learned (literature check, 2026-09-21)

Sources: Shimizu & Srinivasan 2022 (the dataset paper); Li et al. NeurIPS 2024
(ATM-S, THINGS-EEG); "Interpretable EEG-to-Image Generation with Semantic Prompts"
(2025); the 2026 cross-subject EEG survey (arXiv 2604.27033); EEG foundation
models (LaBraM, CBraMod, REVE). Links in the conversation log / commit message.

### Directly actionable for this dataset

- **Train on perception + imagination jointly.** This is the headline result of
  the dataset paper itself: imagination-only classification 13.4%, joint
  training 25.2% (40-way, within-subject). We train Stage 2 on imagination
  only. The visual set is 7,987 trials of the *same* 40 classes, same cap, same
  four people. Joint training is a data-side lever ~3.5x larger than anything
  else available without recording. -> new item **P3b**, above P3 in priority.
- **Imagery lives in low-frequency (<15 Hz) temporal-cortex activity;
  perception in high-frequency (>35 Hz) occipital activity** (same paper,
  attention analysis). Our encoder sees raw broadband 128-ch. Cheap experiments:
  band-limit the input to <15 Hz for imagination; try temporal-channel subsets.
  If the imagery signal is alpha/mu-band, a 250-sample window at 250 Hz is
  only 2 alpha cycles -- longer windows (1-2 s) may matter more than any model change.
- **Trial averaging is standard.** THINGS-EEG results average up to 80
  repetitions per test image; single-trial accuracy is far lower. Our 20 windows
  per imagined stimulus are the analogue. Now built into the val metric
  (`retrieval_top1_avg`) and should be added to eval_report.
- **Small encoders win on small data.** ATM-S is a few million parameters
  (channel attention + temporal-spatial conv + projector) trained with
  contrastive loss against CLIP embeddings; it beats large pretrained encoders on
  THINGS-EEG. Our Stage 2 fine-tunes ~300M. A ~5M-parameter encoder trained from
  scratch with the contrastive loss is a legitimate ablation and may simply be
  better here. -> **P7**.
- **Contrastive alignment, not pointwise regression,** is used by every
  competitive EEG-to-image method since 2023. Implemented (`clip_loss=contrastive`).

### Field-level context worth knowing

- **THINGS-EEG is the benchmark** (10 subjects, 16,540 training images x 4
  repeats, 200 test images x 80 repeats, 63 ch). Subject-dependent 200-way
  top-1 is ~27% (ATM-S) up to ~47-78% in 2026 papers; cross-subject is far lower.
  Nobody reaches those numbers with 40 images per subject.
- **The EEGCVPR / Spampinato dataset results (79%+) are block-design
  contaminated** (Li et al. 2020) and should not be used as a bar. Several
  2025 papers still report on it.
- **Cross-subject transfer toolbox** (survey): Euclidean/Riemannian alignment
  per subject, adversarial subject-invariance (GRL), subject-conditioned
  contrastive learning (same-stimulus-different-subject as positives), meta-
  learning, and few-shot calibration. Of these, per-subject alignment and
  few-shot calibration are cheap and map directly onto your own-data plan.
- **EEG foundation models exist** (LaBraM: 2,000 h; CBraMod: 9,000 h TUEG,
  19 channels 10-20; REVE: 25,000 subjects, any montage). They are pretrained
  on clinical EEG, not visual tasks, and most assume 10-20 montages. Worth a
  try as an encoder initialisation (REVE handles arbitrary channel sets) but
  not a first-order lever.
- **All published imagery results are within-subject.** Cross-subject imagery
  decoding is essentially an open problem; a null there is not a failure of
  this project.

### Added priorities

- **P3b -- Joint perception+imagination Stage 2** (highest data-side value).
  Concatenate the visual and imagination training splits (both `_avail`), keep
  the imagination test split as the test set. Requires a combined `.pth` + splits.
- **P7 -- Small encoder ablation** (ATM-S-style, ~5M params, contrastive only,
  no diffusion) as a fast classifier baseline. If it beats the 300M MAE on the
  val retrieval metric, the diffusion conditioning should come from it.
- **P8 -- Input preprocessing sweep**: <15 Hz band-limit; longer windows;
  per-subject Euclidean alignment.

## Run queue

| Job | What | Status |
|---|---|---|
| 80178 | Imagination LOSO, stratified 200-trial eval on existing checkpoint | running, cs001 |
| 80181 | Imagination window-protocol Stage 2 (200 epochs) + eval | running, cs003, gpuql |
| next | P0 + P1 implemented and smoke-tested locally on hpc, then LOSO v2 | not started |
| after | P2, P3 as separate A/B runs against v2 | not started |

Change one thing per run. Every run is scored with the same stratified
`eval_report.py` so numbers are comparable.
