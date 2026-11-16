# 07 · Low-dimensional visual axes for unseen imagined objects

**Question.** Does a training-derived, low-dimensional visual target improve
EEG retrieval of previously unseen imagined objects?

**Status:** complete; 1,000-permutation result is a **null**, with no supported
benefit from additional target-rank truncation.
**Cluster job numbers:** smoke **113015**, full **113017**, both completed.
Any further submission requires separate approval. The adjacent job also requires an explicit
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

**What "full" means here.** Ridge predictions are linear combinations of the training
targets, so with about 26 training images per fold the full-CLIP arm already predicts
within the span of those images: at most about 25 dimensions, not 768. The comparison
is therefore top 2–16 principal directions of the training images versus all of them,
a reduced-rank truncation. That is still a fair test of the hypothesis; read results
in that light.

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
  people with positive differences, at least 1,000 permutations, **and** the reduced
  arm's group pairwise accuracy above chance under the same training-label null
  (p ≤ .05; added 2026-11-11 in review, before any real-data run, so two chance-level
  arms cannot pass on noise alone). A zero/short
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

After the project lead approves, from `coe-hpc3` in the repository root:

1. Queue `nsfq` was confirmed for this personal-account run. The script enforces
   `aditya-individual-branch` and loads the validated shared Python/PyTorch modules.
2. Point `DATASET` and `CACHE` at the shared copies if they are not in your own checkout.
3. Run the smoke job first; obtain separate approval before submitting the full run:

```bash
mkdir -p logs
sbatch --export=ALL,APPROVED_BY_PROJECT_LEAD=1 studies/2_imagery_decoding/experiments/07_low_dimensional_clip/s19_hpc_smoke.sh
sbatch --export=ALL,APPROVED_BY_PROJECT_LEAD=1 studies/2_imagery_decoding/experiments/07_low_dimensional_clip/s19_low_dimensional_clip.sh
```

Add `DATASET=<path>,CACHE=<path>` to `--export` when using shared copies.

The job is one CPU process for all subjects to keep the permutations synchronized.
The full-run proposal now requests 4 GB RAM and a one-hour limit, based on smoke
job 113015's approximately 1.1 GB peak RAM and 52-second runner time for 25 permutations.
Expected full compute time is approximately 25–35 minutes, plus unknown queue wait.
The site allocated/billed two logical CPUs for the smoke job despite requesting one;
allow at most two billed CPU-hours for the one-hour full-job limit. Scientific code
uses one thread and no GPU. The runner never submits jobs.

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

### Personal-account HPC smoke test (completed 2026-11-11)

`s19_hpc_smoke.sh` is a separate one-task, CPU-only test for Aditya's checkout.
It uses `nsfq`, one CPU, 16 GB RAM, zero requested GPUs, and a 30-minute limit
(requested maximum 0.5 CPU-hours; site allocation was two logical CPUs). It loads the existing site modules
`python3/3.12.12` and `ml/torch/2.6`, with Python at
`/app/rl9.x/ml/torch/2.6/bin/python`. Read-only inspection confirmed NumPy 2.3.5,
SciPy 1.16.3 and PyTorch 2.6.0+cu126 package directories. Imports and node compatibility
were successfully tested in batch job 113015 on g29; no login-node analysis was performed.
Although the shared PyTorch build includes CUDA support, this job requests no GPU,
hides CUDA devices and executes the NumPy/ridge pipeline on one CPU thread.

The script enforces `aditya-individual-branch`, uses the uploaded personal inputs,
and runs only 25 permutations. Prepare `logs/` before submission. After approval
of the upload and this exact submission, submit **from coe-hpc3**, inside the
personal checkout:

```bash
sbatch --export=ALL,APPROVED_BY_PROJECT_LEAD=1 studies/2_imagery_decoding/experiments/07_low_dimensional_clip/s19_hpc_smoke.sh
```

No full run is launched by this script. Record the returned job ID in the research
log and use its measured runtime and memory to prepare the next resource request.
Queue wait is unknown for future jobs. Completed smoke job **113015** ran in 58
seconds (52 seconds in the runner), exited 0:0 and used 1,117,304 KB maximum RSS
(about 1.1 GB). No GPU was allocated. The small summary, manifest and log were
downloaded to ignored `results/s19_hpc_smoke_113015/`.

Local validation on 2026-11-10: **15 synthetic tests passed**, including full CLI
artifacts, training/test isolation for both arms and the follow-up gate. Python
parsing and Git Bash shell syntax checks passed. These checks do not measure the
scientific hypothesis. Details and package versions are in the research log.

| Item | Status |
|---|---|
| Real-data retrieval and paired difference | Full run: full 52.2475%, reduced 52.2980%, paired delta +0.0505 percentage points |
| Chance | Pairwise 50%; top-1 depends on fold candidate count |
| Permutation p-value | 1,000 permutations: delta p .477522, reduced-above-chance p .122877 |
| Cluster job numbers | Smoke **113015** and full **113017**, both completed (CPU only) |
| Full-run resources | 25 min 56 s elapsed; approximately 1.1 GB peak RSS; no GPU |
| Follow-up gate | **FAIL**; only 2/4 participants improve and neither permutation requirement passes |

Conclusion: additional PCA target-rank truncation did not improve retrieval under
this fixed protocol. This result does not refute the single-neuron VTC finding.
Do not proceed to generation or shared perception/imagery training based on it.
Downloaded reports are in ignored `results/s19_hpc_full_113017/`; full checkpoint
and predictions remain in the personal HPC checkout at `results/s19_113017/`.

Research record: `research/aditya/FINDINGS.md`; append-only activity history:
`research/aditya/LOG.md`. No real EEG or model weights are committed.
