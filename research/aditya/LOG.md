# Aditya research activity log

Append new dated entries; preserve earlier entries and correct them with a new entry.
Scientific runs and local software checks must be labeled separately.

## 2026-11-10 — Experiment 07 scope and initial implementation

- User authorized implementation on `aditya-individual-branch`, including a research
  log. User requires all cluster execution to wait for the project lead. No cluster
  connection, transfer, submission, or analysis has been performed.
- Starting HEAD: `f4c77933945408d130798e7a3931b22c9e456b34`; worktree was clean.
- Chosen scientific claim: unseen imagined objects within each person. The existing
  band-power encoder uses early/late windows of the same event; it is not an
  independent-event baseline. With roughly 33 available images, the initial
  suggestion of 64 training-only PCA dimensions is impossible. Fixed dimensions
  are now 2/4/8/16, limited by each inner-training numerical rank.
- Added Experiment 07 runner, README, synthetic tests and adjacent CPU SLURM draft.
  Reuses shared alpha filtering and log-power functions. All twenty windows are
  averaged into one event; five outer and four inner splits operate on image identity.
- Full and reduced arms use the same ridge estimator and l2 grid. Targets/scalers
  are fitted within training partitions only; held-out targets enter only scoring.
  Training permutations are synchronized across people and repeat model selection.
- Outputs are JSON model checkpoints and reports under ignored `results/`, with
  split manifests persisted before fitting. No generator integration or new dataset
  download is part of this experiment. Missing embeddings require explicit opt-in
  to consistently exclude and record unavailable objects.
- Statistical qualification: a paired training-label null measures gains relative
  to broken EEG–image correspondence, not population-level model superiority.
- Local setup: `python -m venv .venv-vtc`; added this environment to `.gitignore`.
  Initial dependency installation was blocked by restricted network sockets.
  Retried through the approved escalation mechanism to install NumPy, SciPy,
  PyTorch and pytest into this local isolated environment. Validation follows below.
- Scientific results: **not run**. Cluster job numbers: **pending lead approval**.

## 2026-11-10 — Local validation completed (synthetic data only)

- Isolated environment installation completed: Python 3.13, NumPy 2.5.4, SciPy
  1.18.1, PyTorch 2.15.0+cpu, pytest 9.2.0. Exact top-level versions are recorded
  in the experiment's `requirements-local.txt`. The cluster environment was not
  inspected or changed.
- Final test command, from the repository root, with numerical libraries limited
  to one thread:

  ```powershell
  $env:OMP_NUM_THREADS='1'
  $env:MKL_NUM_THREADS='1'
  $env:OPENBLAS_NUM_THREADS='1'
  .\.venv-vtc\Scripts\python.exe -m pytest studies/2_imagery_decoding/experiments/07_low_dimensional_clip -q
  ```

- Outcome: **15 passed in 35.45 seconds**. Coverage includes deterministic nested
  stimulus folds, a planted low-dimensional signal, pure noise and shuffled labels,
  hand-checkable retrieval/tie metrics, rank rejection, regularization/dimension
  tie-breaking, full/reduced fit invariance to changed test data, synchronized
  permutations confined to outer training, event averaging, missing assets and
  explicit cache exclusions, duplicate/incomplete windows and inconsistent shapes.
- The end-to-end synthetic check wrote temporary manifests, fitted parameters,
  predictions and summary JSON, repeated the nested pipeline for one permutation,
  and verified that the final success gate stays false for such a short run. It
  also verified that existing output directories cannot be overwritten and that
  the manifest exists before fitting begins. Temporary test artifacts are not
  scientific data and are not committed.
- `git diff --check` passed; both new Python sources parsed successfully.
- Git Bash `bash -n` passed for `s19_low_dimensional_clip.sh`. A local invocation
  without the approval flag emitted “Cluster execution is pending project lead
  approval” and stopped before its execution path. No cluster was contacted.
- Saved provenance includes input hashes, hashes of the runner and imported shared
  source files, Git commit and worktree state, arguments, and numerical versions.
  This captures uncommitted implementation content without implying it is in HEAD.
- Scientific results remain **not run**. Cluster job numbers remain **pending lead
  approval**. No EEG data, CLIP weights, scientific permutation run or generator
  was executed or downloaded. Only software dependencies were downloaded locally.
- Next action: project-lead review of Experiment 07 README, cached image coverage,
  current cluster dependency compatibility and provisional CPU job resources. The
  README includes exact proposed run/submission commands for that review.

## 2026-11-10 — Attempted local real-data smoke run; inputs absent

- User requested continuing Experiment 07 and running it locally if possible.
  The project-lead approval requirement for cluster execution remains in effect.
- Checked the checkout for existing `.pth`, `.pt`, and `.mat` assets, excluding
  the isolated environment and Git internals. No matching input assets were found;
  there is no local `datasets/` directory.
- Invoked the real runner from the repository root, using the isolated CPU
  environment and one numerical thread:

  ```powershell
  .\.venv-vtc\Scripts\python.exe studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py --dataset datasets/imagination_5_95_std.pth --cache datasets/clip_stimulus_emb.pt --output results/s19_local_smoke_20261110 --seed 2022 --permute 0
  ```

- Outcome: runner exit code **2** with `Missing local asset:
  datasets/imagination_5_95_std.pth. No download was attempted.` Independently
  confirmed both the EEG dataset and the CLIP cache are absent. No output directory
  was created, no fitting began, and there is no scientific result to interpret.
- Existing synthetic validation remains 15 passing tests; unchanged code was not
  retested as a substitute for the missing scientific run.
- Next input needed: local paths to trusted `imagination_5_95_std.pth` and
  `clip_stimulus_emb.pt` files, or the project lead's provision of those assets.
  The runner supports arbitrary local paths. No cluster connection, data transfer,
  download, job submission or remote execution was performed.

## 2026-11-11 — Review changes by the project lead (Joao), before any real-data run

Made directly on this branch after review, so the experiment can run on the cluster.
No real data has been run; the protocol change below is fixed before any result.

- **Gate:** the follow-up gate now also requires the reduced arm's group pairwise
  accuracy to beat chance under the same training-label null (p ≤ .05), reported as
  `reduced_above_chance_p`; the paired-delta part is kept as `delta_gate`. Without it,
  two chance-level arms could pass on noise. `decision_gate` itself is unchanged.
- **Loading:** `load_trusted` falls back when PyTorch has no `weights_only` argument
  (before 1.13); the cluster's main environment pins 1.12.1.
- **Job script:** partition set to the placeholder `REPLACE_WITH_NSF_QUEUE` (advisor rule:
  NSF queue only; `sbatch` rejects the placeholder without using resources); `DATASET`,
  `CACHE` and `PERMUTE` can be set at submission, for shared data paths and a short
  trial run (`PERMUTE=25`).
- **README:** the gate change, how to run, and a note that the full-CLIP arm's ridge
  predictions already lie within the roughly 25-dimensional span of each fold's training
  images, so the comparison is a reduced-rank truncation rather than 768 against 2–16.
- **Tests:** two added (above-chance p; loader fallback). Local result recorded below.
- **Local check:** 17 tests passed (Python 3.11, NumPy 1.26.4, PyTorch 2.9.0 CPU), shell
  syntax check passed, runner parses as Python 3.8. Scientific results remain **not run**.
