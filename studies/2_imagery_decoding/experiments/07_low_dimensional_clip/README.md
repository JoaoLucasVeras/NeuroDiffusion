# 07 · Low-dimensional visual axes for unseen imagined objects

**Question.** Does a training-derived, low-dimensional visual target improve
EEG retrieval of previously unseen imagined objects?

**Status:** implementation and local synthetic validation; scientific results **not run**.
**Cluster job numbers:** **pending lead approval**. Do not submit or execute on the
cluster until the project lead approves. The adjacent job also requires an explicit
`APPROVED_BY_PROJECT_LEAD=1` environment setting. This is a guard against accidental
execution, not a mechanism for requesting or verifying approval.

## Rationale and limits

[Wadia et al., Science (2026)](https://doi.org/10.1126/science.adt8343)
found shared perception/imagery tuning in individual VTC neurons using a visual feature
space derived from a deep network. Here PCA of frozen CLIP embeddings is a pragmatic
target-space hypothesis, not a reproduction of their neural recording method or their
AlexNet feature space. Scalp EEG does not measure individual VTC neurons.

The Shimizu data has four people, 128 channels at 250 Hz, and one 10-second imagery
event per class per person, stored as twenty 500 ms windows. Roughly 33 of 40 stimuli
have available images. An unseen stimulus is also an unseen class. Windows do not
constitute independent trials. Older per-subject encoder results used early/late
windows of the same event and are not directly comparable to this protocol.

## Fixed protocol

- Reuse `shared/decoding/classical_baseline.py` for 8–13 Hz filtering and log band
  power, with no artifact rejection. Average the twenty window features into one
  vector per `(subject, image)` event. Require windows 0–19 exactly once and a
  consistent channel/time shape. No concatenation or random window split.
- Use the common cached stimulus set across all included subjects; record every
  exclusion. Missing embeddings fail by default. The explicit
  `--allow_missing_embeddings` flag permits exclusions (needed for the existing
  partial ImageNet cache). Require at least 15 common stimuli. No downloads occur.
- Seed 2022 defines five outer stimulus folds, shared by every subject. Four inner
  stimulus folds within each outer training set select hyperparameters. Every object
  is tested once per subject. The manifest is saved before any model fitting.
- Fit both arms using ridge regression: mean squared error plus `l2 * ||W||²`, with
  an unpenalized intercept, feature standardization, and centered image targets.
  Compare full 768-dimensional CLIP targets with PCA dimensions `{2, 4, 8, 16}`.
  Reject dimensions above any inner-training target rank. The same l2 grid is used
  for both arms: `{0.0001, 0.001, 0.01, 0.1, 1}`. Select by mean inner pairwise
  accuracy; ties within `1e-12` prefer fewer dimensions, then stronger l2.
- Every scaler, target mean, and PCA basis uses its own training partition only.
  Decode reduced predictions back to 768 dimensions, restore the target mean and
  L2-normalize. Full predictions are normalized identically. Holdout embeddings
  are used only for scoring, never target fitting or model selection.
- Retrieve against the held-out images in that fold, in both arms. Pairwise accuracy
  gives half credit to cosine ties within `1e-12` (chance 0.5). Top-1 gives fractional
  credit for a tied best candidate (chance `1 / fold size`); reciprocal rank averages
  over tied ranks. Cosine margin is correct similarity minus the largest distractor
  similarity. Average stimulus scores within person, then weight people equally.
- Controls: the true test image projected through the training-derived PCA basis
  (oracle; explicitly uses test images only at scoring), and the training-target mean
  repeated for every test item (no EEG information). The oracle is diagnostic and
  is never used for model selection or a success gate.
- Primary statistic: mean across people of `(reduced − full)` pairwise accuracy.
  Repeat 1,000 permutations of the outer-training EEG→image assignment. Use the
  same permutation for every subject in a fold; leave outer-test assignments fixed.
  Inner validation uses the permuted assignments, with all transforms and selection
  repeated. No permuted target comes from outer test. Use one-sided Monte Carlo
  `p = (1 + count(null_delta >= observed_delta)) / (1 + permutations)`.
- Record both paired deltas and reduced-arm accuracies under permutation. This null
  asks whether the paired gain exceeds gains from broken EEG–target correspondence;
  it is not a population-level test of estimator superiority. Four subjects and
  the small set of shared stimuli limit inference. Do not treat windows or folds
  as additional people or independent replication.
- Follow-up gate: positive primary difference, paired-null p ≤ .05, at least three
  people with positive differences, and at least 1,000 permutations. A zero/short
  permutation run cannot pass. Passing warrants lead review of a subsequent
  perception-plus-imagery experiment, not automatic implementation or generation.

## Inputs and outputs

Run from the repository root. Inputs are trusted repository `.pth` files and a local
frozen CLIP ViT-L/14 cache with `{"stems": [...], "emb": tensor(N, 768)}` as produced
by Experiment 01's `clip_targets`. The cache may be a superset; stems must be unique
and vectors finite and nonzero. The model cache is not downloaded or regenerated.
Loading `.pth` uses pickle: only use trusted files. Dependencies: NumPy, SciPy, PyTorch
(with the `weights_only` load option; recommended 2.x); pytest for local tests.
`requirements-local.txt` records the exact local validation versions (Python 3.13,
Windows). It does not replace or upgrade the cluster environment.

```powershell
.\.venv-vtc\Scripts\python.exe -m pytest studies/2_imagery_decoding/experiments/07_low_dimensional_clip -q
```

Scientific run command, **for project-lead review only**:

```bash
python studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py \
  --dataset datasets/imagination_5_95_std.pth \
  --cache datasets/clip_stimulus_emb.pt --allow_missing_embeddings \
  --output results/s19_reviewed_run --seed 2022 --permute 1000
```

After explicit approval, the lead can prepare `logs/` and submit the adjacent CPU job:

```bash
mkdir -p logs
sbatch --export=ALL,APPROVED_BY_PROJECT_LEAD=1 studies/2_imagery_decoding/experiments/07_low_dimensional_clip/s19_low_dimensional_clip.sh
```

The job is one CPU process for all subjects to keep the permutations synchronized.
The 12-hour walltime is a provisional budget; actual cluster runtime is unmeasured.
The runner never submits jobs. CPU numerical thread limits are set in the job script.

The output directory must be new. Under ignored `results/`, save:

- `manifest.json`: every outer/inner index, ordered stimulus identities, event counts,
  exclusions, arguments, versions, Git commit/dirty status, and input SHA-256 hashes.
- `observed.json`: distinct checkpoint kind `vtc_visual_axes_ridge_v1`, all fitted
  coefficients, scalers, target means/PCA bases, tuning scores, fold test indices,
  predictions for all four arms, and per-subject metrics. All arrays are JSON lists;
  this is intentionally not a generator-compatible encoder checkpoint.
- `null_progress.json`: completed permutation count and null distributions, refreshed
  every 25 permutations. An interrupted run is incomplete; rerun into a new directory.
- `summary.json`: final participant/group results, permutation distributions, p-value,
  provenance, limitations and follow-up gate. Its absence means the run is incomplete.

## Result and record

Local validation on 2026-11-10: **15 synthetic tests passed**, including full CLI
artifacts, training/test isolation for both arms and the follow-up gate. Python
parsing and Git Bash shell syntax checks passed. These checks do not measure the
scientific hypothesis. Details and package versions are in the research log.

| Item | Status |
|---|---|
| Real-data retrieval and paired difference | **not run** |
| Chance | Pairwise 50%; top-1 depends on fold candidate count |
| Permutation p-value | **not run** |
| Cluster job numbers | **pending lead approval** |

Research record: `research/aditya/FINDINGS.md`; append-only activity history:
`research/aditya/LOG.md`. No real EEG or model weights are committed.
