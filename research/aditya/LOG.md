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

## 2026-11-11 — HPC execution authorized; SSH setup unavailable locally

- User reports project-lead approval to run Experiment 07 on the HPC and explicitly
  requires all code work to remain on `aditya-individual-branch`.
- Verified current branch and commit `dcaa6e4` (Experiment 07 implementation), tracking
  `origin/aditya-individual-branch` at the same commit. An unrelated untracked file
  with an unusual name is present and has been left untouched.
- Read the complete user-provided `Downloads/HPC_AGENT_ACCESS.md`. It specifies a
  dedicated SSH key, the `sjsu-hpc1` alias, read-only access verification, confirmation
  of the NSF queue, and explicit approval for each submission/resource request.
  The existing `cpuqs` job draft must not be submitted unchanged under these rules.
- Local prerequisite checks: `C:\Users\adiak\.ssh\config` and
  `C:\Users\adiak\.ssh\id_neurodiffusion_hpc` are absent according to filesystem
  checks. `ssh -G sjsu-hpc1` leaves the hostname as the literal alias and reports no
  dedicated identity or BatchMode configuration, so the required alias is not set
  up in the current execution context.
- No SSH connection, key creation, remote checkout change, data transfer, environment
  change or job submission was attempted. Private key contents were not accessed.
- Next action: human completes Part 1 of the access guide (VPN and dedicated-key
  setup) or provides the actual configured SSH alias/key location. Then perform
  the read-only checks, confirm shared input/Python paths and NSF queue, and present
  one small CPU task with exact resources for submission approval.

## 2026-11-11 — Pulled lead review fixes into the personal branch

- At the user's request, fetched origin and fast-forwarded only
  `aditya-individual-branch` from `dcaa6e4` to `71c55fc` (Joao's review fixes).
  Local HEAD matches `origin/aditya-individual-branch`; no main-branch changes.
- Preserved the uncommitted HPC setup entry using a targeted stash. Both the lead's
  review entry and the local entry appended at the same place; resolved that log-only
  conflict by keeping both. Local log additions remain unstaged, and the targeted
  stash remains available as a backup. The unrelated untracked file was untouched.
- Local validation after pulling: **17 tests passed in 33.06 seconds**, using the
  isolated environment and one numerical thread; pytest's cache provider was
  disabled. `git diff --check` passed, and no unresolved Git entries remain.
- No cluster connection, scientific run, job submission, remote file change, or
  push was performed during this update.

## 2026-11-11 — Local OSF data setup blocked by invalid ZIP

- User requested setting up local data from
  `C:\Users\adiak\Downloads\2fgks-osfstorage-archive.zip`.
- Checked the exact archive: 58,820,590 bytes, last modified November 10.
  Python's standard ZIP validator returned false; opening the archive raised
  `BadZipFile: File is not a zip file`. This is the same archive previously found
  to have a ZIP starting header but no end-of-archive record.
- The original archive was left untouched. No files were extracted, and no EEG
  tensor or embedding cache was generated from this invalid input.
- The local isolated environment has NumPy, SciPy and PyTorch; the repository's
  MATLAB converter also needs h5py, which is not installed yet. Deferred installing
  conversion/image dependencies until a valid input is available.
- Next action: obtain a complete archive or the four `Imagine_Sub001.mat` through
  `Imagine_Sub004.mat` files individually from the same OSF project. Then place
  them in the ignored `datasets/Imagination Experiment/` directory, convert with
  the existing repository script, and audit subjects, stimulus names and windows.
  The CLIP image-embedding cache remains a separate required asset.
- No HPC connection, transfer, submission or remote change was performed.

## 2026-11-11 — Four local MATLAB files validated and imagery dataset built

- User supplied `Imagine_Sub001.mat` through `Imagine_Sub004.mat` individually in
  Downloads. All four open successfully as HDF5 MATLAB files. Each contains data
  shaped `(125, 128, 800)`, 800 labels, decoded stimulus-name characters and window
  indices 0 through 19. These valid individual files supersede the unusable ZIP.
- Installed `h5py==3.16.0` into the existing ignored `.venv-vtc` environment to
  enable the repository converter. No cluster environment was changed.
- Copied the four originals into ignored `datasets/Imagination Experiment/`, after
  checking no destination files existed. Downloads originals remain untouched.
- The first converter attempt exposed an existing `UnboundLocalError`: the local
  `paths` list in `main()` shadowed the imported shared `paths` module before
  `paths.DATASETS` was read. Renamed the local lists to `mat_paths` for both
  experiment branches; no decoding, filtering or target logic was changed.
- Successfully ran:

  ```powershell
  .\.venv-vtc\Scripts\python.exe shared/data/prepare_shimizu_data.py --datasets_dir datasets --experiment imagination
  .\.venv-vtc\Scripts\python.exe shared/data/check_data.py --dataset datasets/imagination_5_95_std.pth --skip_image_check
  ```

- Output: `datasets/imagination_5_95_std.pth`, 205,856,293 bytes. Every subject's
  stimulus/label check passed 800/800. Dataset has four subjects, 40 unique stimuli,
  40 classes and 3,200 stored windows shaped `(128, 125)`.
- Additional complete audit passed: all 3,200 EEG tensors finite; 160 distinct
  `(subject, stimulus)` events; every event contains exactly windows 0–19 once;
  event labels consistent; the same 40 stimuli appear in every participant.
- EEG-only preflight passed with its expected warning that no split file was
  supplied. Experiment 07 creates its own nested stimulus splits; no historical
  early/late-window split was generated. The preflight's image check was explicitly
  skipped and does not establish readiness for EEG-to-image retrieval.
- Confirmed the copied MATLAB files and converted tensor are Git-ignored.
- Remaining local input: `datasets/clip_stimulus_emb.pt` is absent, and local
  stimulus images are not installed. No embeddings were synthesized or substituted.
  Experiment 07 scientific results remain **not run**. No HPC access occurred.

## 2026-11-11 — CLIP cache installed; local real-data smoke test completed

- User supplied `Downloads/clip_stimulus_emb.pt`, identified by the project lead as
  the Shimizu imagination cache covering the 33 stimulus images found online.
- Validated the trusted file: keys `stems` and `emb`, 33 unique stems, embedding
  shape `(33, 768)`, all values finite, nonzero unit norms, and all filenames
  match the locally converted EEG dataset. The seven missing images exactly
  match `shared/data/imagination_missing.txt`.
- Copied the cache into `datasets/clip_stimulus_emb.pt` without overwriting any
  existing file. Added `datasets/*.pt` to `.gitignore` to protect the cache from
  accidental commits. Cache SHA-256:
  `6ca2710ed9c6becf1f3976f757f1bdfe5ec31547d9a20eb00b21ddef2e23df83`.
- Local smoke command, with numerical libraries restricted to one CPU thread:

  ```powershell
  .\.venv-vtc\Scripts\python.exe -u studies/2_imagery_decoding/experiments/07_low_dimensional_clip/low_dimensional_clip.py --dataset datasets/imagination_5_95_std.pth --cache datasets/clip_stimulus_emb.pt --allow_missing_embeddings --output results/s19_local_smoke_20261111_33images --seed 2022 --permute 25
  ```

- Completed successfully in **40.61 seconds**. Effective sample is four people
  with 33 events each (132 independent imagery events); the seven absent image
  targets were explicitly excluded across participants. Outputs and cache are ignored.
- Smoke results: mean full/reduced pairwise accuracy 0.522475 / 0.522980 (chance
  0.5), delta +0.000505; two of four participants improve. The short null gave
  paired-delta p 0.5 and reduced-above-chance p 0.153846. The follow-up gate is
  false. These are a 25-permutation smoke test, not a completed fixed analysis;
  no settings were changed in response to the observed results.
- Started the fixed 1,000-permutation local run under the prior local-run
  authorization, then stopped it when the user requested switching execution to
  HPC. Partial output is preserved at `results/s19_local_full_20261111_33images/`:
  25 permutations checkpointed, no final `summary.json`. Do not report it as complete.

## 2026-11-11 — Read-only HPC readiness checks for CPU-only execution

- User requests execution on HPC with no GPU, and an estimate of when results
  will be available. HPC access guide still requires queue confirmation and exact
  resource approval for each submission, plus approval before remote checkout edits.
- Dedicated Windows SSH key reaches the login node; nested SSH reaches the submit
  node `g17.hpc.coe` (coe-hpc3). The user's job queue was empty.
- `sinfo -s` identifies NSF partitions `nsfq` and `nsfqm`. Both list 13 allocated
  nodes and no idle nodes at inspection; this does not give a reliable queue-wait
  estimate or prove that no CPUs can be scheduled on allocated nodes.
- Own-account checks confirmed these paths do not exist:
  `~/NeuroDiffusion`, its expected `datasets/` input files, and
  `~/.conda/envs/neurodiffusion/bin/python`. No other user's home
  directory was inspected. Shared input/environment paths must come from the lead.
- A multiline read-only check encountered Windows/SSH quoting issues. Corrected
  to a direct remote `ls` for the four exact expected paths; no remote files were
  modified. No SSH config or keys were read or changed on the server.
- Disk report: shared `/home` is 95% full, approximately 6.7 TB available. No data
  duplication, environment creation, checkout creation/change or submission occurred.
- Preliminary compute estimate from the local smoke test is roughly 20–25 minutes
  for 1,000 permutations, subject to CPU speed and fixed overhead. HPC queue wait,
  environment readiness and transfer/setup time are unknown. A short HPC task
  must measure actual runtime before a reliable completion estimate is possible.
- Next inputs: confirm NSF queue name (candidate `nsfq`), shared dataset/cache and
  Python executable paths, and approve creating an isolated own-account checkout
  on `aditya-individual-branch`. Then request approval for a proposed one-task CPU
  smoke job (1 CPU, 16 GB RAM, 0 GPUs, 30-minute limit; maximum 0.5 CPU-hours).

## 2026-11-11 — NSF queue confirmed and personal HPC checkout created

- User confirmed `nsfq` and approved creating the own-account checkout on
  `aditya-individual-branch`. This does not supply missing shared paths or approve
  an additional dataset copy/environment installation.
- On the login node, shallow-cloned only `aditya-individual-branch` from the
  project repository into `~/NeuroDiffusion`. Verified clean branch
  `aditya-individual-branch` at `71c55fc`; no main checkout, pull or merge occurred.
- Read-only verification found the login node's Git lacks `-C`; used `cd` into
  the approved checkout for compatible status/log checks. Available command paths
  are `/usr/bin/python` and `/usr/bin/python3`; no conda executable was found on
  PATH. These system Python paths have not been validated for scientific dependencies.
- Explained that the shared paths are the cluster locations of the existing EEG
  tensor, CLIP cache, and project Python environment. Those locations must be
  supplied by the lead; none is inferred from or copied out of another user's home.
- No datasets were uploaded, no Python environment was created or changed, no
  job script was edited remotely, and no job was submitted.

## 2026-11-11 — Uploaded validated inputs to the personal HPC account

- User explicitly approved uploading our local data to the personal HPC account.
  Announced exact destination and total size before transfer: 205,959,120 bytes
  (about 196 MB) to `~/NeuroDiffusion/datasets/`.
- Read-only preflight confirmed the checkout is on `aditya-individual-branch`,
  neither destination file exists, and shared `/home` has approximately 6.7 TB
  free (95% used). Created only the missing datasets directory, then copied the
  two requested files with SCP. No raw MATLAB files were uploaded.
- Uploaded and verified byte-for-byte using local and remote SHA-256 hashes:

  | File | Bytes | SHA-256 |
  |---|---:|---|
  | `imagination_5_95_std.pth` | 205856293 | `baf1ec1c55e41c18be9e390f540c8c4f369d8bfa101491250f9cc58ccd4c16b7` |
  | `clip_stimulus_emb.pt` | 102827 | `6ca2710ed9c6becf1f3976f757f1bdfe5ec31547d9a20eb00b21ddef2e23df83` |

- Confirmed the remote checkout remains on the personal branch. The remote branch's
  committed `.gitignore` ignores `.pth` but does not yet include our local
  `datasets/*.pt` addition, so remote status reports untracked `datasets/` for
  the cache. No data was staged or committed. The local cache is Git-ignored.
- No remote tracked files or environment were modified, and no job was submitted.
  Next prerequisite is a validated Python environment on compute nodes, followed
  by approval for the exact CPU smoke-job submission.

## 2026-11-11 — Shared site environment found; CPU smoke script ready for approval

- User said to continue after the personal-account data upload. Inspected modules
  and configuration read-only on coe-hpc3; did not create an environment or submit.
- System Python on the scheduler is 3.6.8, but site modules provide Python 3.12.12
  and `ml/torch/2.6`. The latter uses `/app/rl9.x/ml/torch/2.6/bin/python` and
  requires the Python 3.12.12 module. Package directories confirm NumPy 2.3.5,
  SciPy 1.16.3, and PyTorch 2.6.0+cu126. Did not import them on the scheduler.
- Read-only `scontrol show partition nsfq` on coe-hpc3 confirms NSF nodes g20–32,
  maximum time 1 day and 1 minute, and partition state UP. An initial command
  quoting issue ran a partition query on the wrong login context; corrected using
  a script passed on stdin through both SSH hops and confirmed hostname g17.hpc.coe.
- Prepared local `s19_hpc_smoke.sh` next to Experiment 07. Login-shell batch script
  loads the site modules, enforces the personal branch, hides CUDA devices, restricts
  CPU threads to one, logs package versions, and runs the unchanged runner on the
  uploaded inputs with seed 2022 and 25 permutations. No GPU resource is requested.
- Resource proposal: `nsfq`, one node, one task, one CPU, 16 GB memory, zero GPUs,
  array size one, 30-minute limit (maximum 0.5 CPU-hours). Script does not start a
  full run. A completed smoke job must precede any separate full-run approval.
- Local Git Bash `bash -n` and `git diff --check` pass. No model logic changed.
- Pending explicit approval to upload the new script to the personal checkout,
  create its `logs/` output directory and submit this exact one-task job. No remote
  file edit/upload or `sbatch` was performed in this environment-discovery step.

## 2026-11-11 — Approved HPC smoke job 113015 submitted

- User explicitly approved uploading the prepared smoke script, creating `logs/`,
  and submitting the one-task CPU smoke job described above.
- Confirmed the remote checkout is on `aditya-individual-branch` and the destination
  script does not exist. Created `logs/`, uploaded only `s19_hpc_smoke.sh`, verified
  shell syntax on coe-hpc3, and submitted exactly one job with
  `APPROVED_BY_PROJECT_LEAD=1`. Returned job/array ID: **113015**, task **113015_1**.
- Job started on **g29**, in **nsfq**, approximately one second after submission.
  Script enforces the personal branch; log confirms runner commit `71c55fc`.
- Requested one node, one task, one CPU, 16 GB memory, no GPUs, 30 minutes, 25
  permutations. Scheduler reports requested `cpu=1` but allocated `cpu=2` and
  billing=2 (site allocation granularity). No GPU is present in allocated TRES.
  Code is restricted to one CPU thread, confirmed by `torch.get_num_threads()`.
  Thus maximum billed allocation could be 1 CPU-hour rather than the requested
  0.5 CPU-hour estimate; actual usage will be recorded after completion.
- Compute-node imports succeeded using the shared site environment: Python
  3.12.12, NumPy 2.3.5, SciPy 1.16.3, PyTorch 2.6.0+cu126. No installation was needed.
- Job log: `~/NeuroDiffusion/logs/s19_smoke_113015_1.out`.
  Expected output directory: `results/s19_hpc_smoke_113015/`.
- At the first status check the job was RUNNING, with all environment checks passed.
  No full scientific job has been submitted. Completion and results follow below.

## 2026-11-11 — HPC smoke 113015 completed; reports retrieved

- Job **113015_1** completed successfully on g29 with exit code **0:0**. Elapsed
  allocation 58 seconds; runner measured 52 seconds. Maximum batch RSS **1,117,304K**
  (approximately 1.1 GB). Requested one CPU, allocated two logical CPUs; allocation
  accounting is approximately `2 * 58 / 3600 = 0.0322` CPU-hours. No GPU was allocated.
- All 25 permutations completed. Group pairwise full/reduced accuracy is
  0.5224747475 / 0.5229797980; paired delta **+0.0005050505** (about +0.0505 percentage
  points), paired-null p **0.5**, reduced-above-chance p **0.153846**. Follow-up gate
  is false. Mean delta exactly matches the local smoke result. These smoke results
  do not constitute the planned 1,000-permutation inference.
- Downloaded only the small `summary.json`, `manifest.json` and
  `s19_smoke_113015_1.out` into ignored `results/s19_hpc_smoke_113015/`. The large
  observed model/prediction checkpoint remains on HPC; no dataset was downloaded.
- Prepared a local update to the existing full-run job script for separate review:
  confirmed `nsfq`, one node/task/array element, one requested CPU, 4 GB memory,
  zero GPUs, one-hour limit, 1,000 permutations. It loads the same verified modules,
  enforces the personal branch, and logs elapsed time. Shell syntax passes.
- This reduces memory/time from the original unmeasured 16 GB / 12-hour draft.
  Expected full compute is approximately 25–35 minutes, based on this smoke job;
  queue wait and CPU variation remain unknown. Maximum billed allocation may be
  two CPU-hours if the scheduler again allocates two logical CPUs for one requested.
- No full script was uploaded/replaced on HPC and no second job was submitted.
  Await explicit approval for overwriting the existing full-run script in the
  personal checkout and submitting that exact one-task full run.

## 2026-11-11 — Full HPC job 113017 approved and started

- User approved replacing the full-run script and submitting the exact one-task
  CPU job proposed above. Checked the remote script had no unexpected edits before
  replacement; branch remains `aditya-individual-branch` at runner commit `71c55fc`.
- Linux pre-submission `bash -n` caught CRLF line endings after the initial upload.
  `set -e` stopped the submission script before `sbatch`, so no job/resources were
  consumed by that preflight. Confirmed the CR characters with a read-only byte/
  line-ending inspection; normalized the exact local script using dos2unix,
  re-uploaded the same approved content with LF endings, and passed Linux syntax.
- Added local `.gitattributes` with `*.sh text eol=lf` to prevent recurrence across
  Windows/Linux checkouts. Scientific code, settings and resource limits unchanged.
- Submitted exactly one full job from coe-hpc3: **113017**, task **113017_1**.
  Explicitly exported `PERMUTE=1000`, the personal checkout root and the validated
  shared Python executable. No GPU was requested.
- Initial scheduler verification: RUNNING on **g29**, partition **nsfq**, requested
  one CPU, allocated two logical CPUs, **4 GB RAM**, one-hour limit; allocated TRES
  contains no GPU. Maximum billed allocation is two CPU-hours. Job started about
  two seconds after submission; log confirms the personal branch and 1,000 permutations.
- Log: `~/NeuroDiffusion/logs/s19_visual_axes_113017_1.out`.
  Results: `~/NeuroDiffusion/results/s19_113017/`.
- Watching this job read-only; no additional submission or cancellation is authorized.
  Completion, accounting and final reports will be recorded below.

## 2026-11-11 — Monitoring stopped at the user's request; full job left running

- User asked to leave the job running and check again when they return in about
  30 minutes. Stopped only the read-only SSH monitoring session; did not run
  `scancel`, signal the batch process, edit its files, or submit another job.
- Last observed status: **113017_1 RUNNING on g29**, 4 minutes 38 seconds
  elapsed, with **150/1,000 permutations** recorded in the log.
- The Slurm batch job runs independently of the monitoring connection. On the
  user's next request, inspect job 113017 accounting/logs, then retrieve the small
  final summary if it exists. No scientific completion or final p-value is claimed.

## 2026-11-11 — Full job 113017 completed; final result is a null

- User requested checking the job again. Read-only Slurm accounting reports
  **113017_1 COMPLETED**, exit **0:0**, elapsed **00:25:56**, allocated two logical
  CPUs and requested memory 4 GB. Maximum batch RSS **1,114,092K** (about 1.1 GB).
  Runner measured **1,553 seconds**. Approximate billed CPU-hours from allocation:
  `2 * 1556 / 3600 = 0.8644`. No GPU was allocated.
- `squeue` no longer finds the finished job; `sacct` and the final log independently
  confirm successful completion, not a missing/failed submission. All **1,000**
  permutations finished and `summary.json` exists.
- Downloaded the approximately 61 KB summary, small manifest and text log into
  ignored `results/s19_hpc_full_113017/`. The large observed checkpoint remains
  on the cluster. No datasets or model weights were copied back.
- Final group pairwise accuracy: full target **0.5224747475**, reduced target
  **0.5229797980**, chance **0.5**. Paired difference **+0.0005050505**, equivalent
  to **+0.0505 percentage points**. Paired training-label-null p **0.4775224775**.
- Reduced-arm above-chance training-label-null p **0.1228771229**. Only **2 of 4**
  participants improve. Both permutation requirements and the directional
  participant requirement fail, so **follow_up_gate = false**.
- Secondary metrics: full/reduced top-1 0.181818 / 0.166667, with weighted fold
  chance 0.151515. True-target PCA oracle pairwise 0.732071; training-mean control
  pairwise 0.5. Oracle uses target images only as a scoring diagnostic and must
  not be represented as EEG decoding performance.
- Paired per-person differences: subject 1 +0.031313; subject 2 -0.030303;
  subject 3 +0.023232; subject 4 -0.022222. Small gains cancel across participants.
- Recorded final results and limitations in FINDINGS.md and the experiment README.
  Conclusion is specific to this alpha-power/ridge/CLIP-PCA protocol and small
  dataset; it is not evidence against the original single-neuron VTC finding.
- No settings were tuned in response, no follow-up job was submitted, and no
  remote source files were edited during this check. Next action is lead review
  of the null before designing a distinct follow-up experiment.

## 2026-11-11 — Prepared completed findings for branch publication

- User requested pushing the findings to the personal branch and advice on next
  steps. Refreshed origin refs and confirmed no branch divergence before staging.
- Included final findings, full activity history, experiment README, the actual
  CPU smoke/full scripts, local conversion dependency and converter startup fix,
  cache ignore rule, and Linux shell line-ending attributes for reproducibility.
- Public research notes use home-relative HPC paths rather than the account's
  numeric identifier. No raw EEG, embedding cache, model checkpoint, local result
  directory, SSH access document, keys or unrelated untracked file is included.
- Scientific code and experiment settings were not changed after seeing results.
  Existing validation comprises 17 tests, the real-data conversion/audit and
  completed HPC smoke/full jobs; final shell syntax and whitespace checks pass.
- Next step is Joao's review of the null and a separate diagnostic proposal,
  rather than automatic perception/imagery training or image generation. No new
  HPC job is requested or submitted as part of publication.

## 2026-11-12 - Prepared the findings PR and main-page summary

- User requested the immediate publication step after Joao reviewed the work and
  invited a PR. Work remains on `aditya-individual-branch`; no push to main.
- Fetched origin and merged its two newer main commits without conflicts,
  preserving the team's GitHub Pages setup and 12 November Gao dataset update.
- Added the completed Experiment 07 result to the main documentation's status,
  experiment ledger and dated journal. Added the experiment to Study 2's index
  and updated the root README's experiment count and current-status summary.
- Reported full/reduced outer-test pairwise accuracy, chance, sample size, paired
  and above-chance permutation results, controls, job numbers and limitations.
  Explicitly distinguished successful execution from successful EEG decoding;
  this null neither establishes intrinsically faint imagery nor disproves VTC.
- Validation: reran the existing local suite with
  `.venv-vtc/Scripts/python.exe -m pytest studies/2_imagery_decoding/experiments/07_low_dimensional_clip/test_low_dimensional_clip.py -q -p no:cacheprovider`:
  **17 passed in 49.61 seconds**. HTML parsing, unique IDs, newest-first journal
  chronology and existence of linked review files passed. `git diff --check`
  passed. An initial inline HTML-check command hit PowerShell quoting errors;
  rerunning the check through standard input succeeded without changing files.
- No scientific code or settings changed, no data or generated artifacts added,
  no cluster connection or run, and no new experiment started. The unrelated
  untracked file was left untouched. Next action: open the personal-branch PR
  against main for Joao's review; do not merge it automatically.

## 2026-11-12 - Opened findings PR 17

- Pushed documentation commit `c517412` to the personal branch and opened
  [PR 17](https://github.com/JoaoLucasVeras/NeuroDiffusion/pull/17) from
  `aditya-individual-branch` to `main`, including methods, controls, exact results,
  validation and the retained legacy branch-name exception to the PR template.
- GitHub confirmed the PR is **open**, not merged. No auto-merge enabled and no
  new HPC work performed. An empty-array handling issue in the initial API check
  was corrected before creation; exactly one PR was created.
- Next action: Joao reviews/merges the PR so the main documentation page receives
  the findings. Agree on a distinct follow-up question before implementing or
  submitting another experiment.
