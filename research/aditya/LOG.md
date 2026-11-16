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

## 2026-11-14 - Reviewed four papers for possible follow-up experiments

- Compared the papers' methods and evaluation with Experiment 07, the Gao
  baseline, the stronger-decoder tests and the anchored-imagery protocol.
  This was a literature/code review, not a new scientific run.
- SGDM (https://arxiv.org/html/2604.22649v1): useful distinction between semantic
  and structural predictions, but its tangram protocol records viewing, not
  eyes-closed imagery. Image-level holdouts do not establish unseen-person
  generalization. No project code/checkpoint release was verified from its text.
- GAN-titled baseline (https://arxiv.org/html/2609.22282v1): its main evidence is
  THINGS perception retrieval with repeated-trial averaging, not faithful pixel
  reconstruction. Its single shuffled-pair scoring control is not a training-label
  permutation test. The linked implementation could not be inspected successfully.
- ATM (https://arxiv.org/html/2403.07721 and the official NeurIPS proceedings
  entry): reusable encoder/alignment components, not a plug-in imagery solution.
  The conference PDF exceeded the browser size limit; used the authors' full-text
  arXiv version and released code, without assuming identical versions. The
  develop launcher uses 63 channels, 250 samples and ViT-H/14 by default. Its
  training entry point uses condition-stratified within-person validation,
  random validation for joint/LOSO modes, and documents a test-loss fallback when
  validation is disabled. Reuse needs explicit grouped validation and no fallback.
  Source: https://github.com/dongyangli-del/EEG_Image_decode/tree/develop .
- PMC11534409 is a narrative review of psychiatric EEG biomarkers, not an
  image-generation method or evidence for visual-content decoding.
  Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11534409/ .
- Most promising cheap proposal: measure the effect of averaging independent
  imagery events on cross-day Gao shapes decoding at fixed repetition budgets,
  retaining the existing simple decoder. Class-conditioned bundling is an
  instructed-repeat/offline protocol, not blind grouping or unseen-object retrieval.
  Do not treat overlapping bundles or different regrouping seeds as independent
  samples. Shimizu windows cannot substitute for independent repeated events.
- Other proposals: separately test structural targets before any ControlNet
  integration; audit/pilot ATM on THINGS only if a reusable large-data perception
  encoder is an explicit goal. Neither proposal is evidence of imagery transfer.
- No experiment was implemented, no data downloaded, no cluster contacted and
  no GitHub content changed. Notes remain local on the personal branch. Any
  numerical follow-up needs a fixed protocol, versioned/audited data and fresh
  execution approval. Reusing previously inspected data makes this a follow-up,
  not an untouched independent confirmation.

## 2026-11-14 - Independent-repetition feasibility audit

- User requested proceeding with the feasibility audit. Stayed on
  `aditya-individual-branch`; preserved the earlier uncommitted literature notes
  and unrelated untracked file. No cluster connection, download or submission.
- Code-level finding: `shared/data/prepare_gao_data.py` documents 40 events per
  class/session and stores subject, session, task, trial ordinal, class/image,
  sampling rate and epoch boundaries. It verifies source event codes against BDF
  triggers. A usable trial key is (subject, session, task, trial), NOT trial alone.
  Trial ordinals are assigned before exclusions, so gaps must be preserved.
- Provenance limitation: converted trials do not retain source filename, raw
  event latency or archive checksums. Their IDs allow deterministic disjoint
  grouping but do not prove non-overlapping raw time intervals, correct source
  version, or independence from recording drift. Source records are needed for
  that part of the audit; duplicate IDs and repeated EEG content must be checked.
- Novelty within inspected code: Gao baseline and enhancement scripts score
  individual test events, not fixed bundles of 2/4/8 events. Existing pooling in
  `classical_baseline.score` and `eval_report.py` groups Shimizu windows by image;
  that is not independent Gao repetition averaging. No equivalent Gao budget
  experiment was found in the checked-in code/docs searched.
- Important exclusion mismatch: experiment 06's README says exclusions are in
  each script's EXCLUDE set, but the inspected Gao baseline and enhancement
  entry points contain no such subject/task guard. Historical corrected results
  may have excluded people during aggregation or a separate run; that mechanism
  is not established here. New code must explicitly reject sub-21 FVI for v3
  rather than assuming the imported loader enforces it. Sub-09/sub-10 lack the
  second session. Expected eligible count is at most 19 of 22, subject to audit.
- Version hazard: the converter's source string says v3, but `cluster/fetch_gao.sh`
  queries the unversioned latest record and only size-checks cached downloads.
  That string is not proof of source version. The 12 November project journal
  reports v4 replacement files, including sub-21. Do not mix versions, trust the
  label alone, or restore excluded participants without checksum/duplicate audit.
  Direct Figshare metadata retrieval was unavailable through the web tool during
  this audit; no current release state was independently verified.
- Local asset inventory: datasets contains only the Shimizu MAT folder, converted
  imagination tensor and CLIP cache. There is no local Gao dataset directory and
  no local Gao numerical result directory. Actual per-class counts, trial quality,
  duplicate recordings and the data-level eligibility manifest remain unverified.
- Ran an ephemeral Python metadata-only check using the existing isolated local
  interpreter. With synthetic 40-event/class inputs in each of two sessions,
  budgets 1/2/4/8 yield 120/60/30/15 decisions per session (40/20/10/5 per class).
  Every budget uses 120 distinct event IDs once, every bundle stays inside one
  session/class, and no partial bundle occurs. Assertions passed. This verifies
  arithmetic and grouping feasibility ONLY, not real EEG counts or performance.
- Protocol corrections before implementation: use four attempts as the sole
  primary budget, compare both arms on identical retained source events, weight
  participants equally after averaging directions, and do not count directions
  or regroupings as independent people. Keep the same decoder/settings, choose
  one pooling operation in advance, and repeat both arms under paired training-
  label shuffles. Any session-wide label-free normalization is transductive
  calibration, not online performance; preserve and disclose it in both arms.
- Class-based test grouping is an instructed-repeat/offline analysis. Existing
  trials re-present the picture before each imagery event; this is not proof of
  four consecutive uncued mental attempts, unseen-object retrieval, or successful
  image generation. Four 4-second imagery epochs imply at least 16 seconds of
  imagery evidence, plus cues, picture presentation and breaks; actual latency
  cannot be claimed without source timing.
- Decision: CONDITIONAL GO for drafting an experiment, NOT cleared for a real-
  data run. Next requirement is the location of the existing audited Gao assets
  and their provenance. Verify data-level feasibility before freezing success
  criteria or writing/submitting a cluster job. No classifier or scientific
  analysis was run, no code changed, and no GitHub content was modified.

## 2026-11-14 - Attempted read-only HPC data inventory

- User authorized checking their own HPC account for existing Gao assets. Tried
  the previously configured SSH host with strict host-key checking, batch-mode
  authentication and a bounded filename search limited to the account's home.
  The planned search prunes SSH, Git, package-cache and environment directories.
- Connection failed before authentication: `coe-hpc1.sjsu.edu` could not be
  resolved. No remote command ran, so the presence or absence of Gao data on
  the account remains unknown. Did not retry in a loop or bypass host checking.
- A local VPN-agent process is running, but this does not establish an active
  campus VPN session. Adapter inspection was denied in the local sandbox;
  sandboxed existence checks for SSH files were also denied and do not show that
  the key is missing. No private key contents were requested or displayed.
- The previously supplied access-guide path in Downloads was unavailable.
  Used only the existing read-only connection settings; no setup was changed.
- Next action: restore campus/VPN DNS connectivity, then repeat the authorized
  read-only inventory. No data transfer, environment changes, cluster analysis,
  submission, cancellation, commit or push occurred.

## 2026-11-14 - Completed read-only HPC filename inventory after VPN reconnect

- User confirmed VPN connectivity and authorized retry. SSH succeeded with the
  existing dedicated key and strict host-key verification. The personal remote
  checkout remains on `aditya-individual-branch`; nothing was fetched or changed.
- The personal checkout's datasets directory contains only
  `imagination_5_95_std.pth` and `clip_stimulus_emb.pt`: the Experiment 07 Shimizu
  inputs. No Gao dataset directory or Gao-related conversion/result logs were
  found in the inspected checkout locations.
- Bounded searches of the account's home (maximum depth six, no traversal of
  symlink targets, excluding credentials/package-cache/environment directories)
  found Gao source scripts but no `gao_sub-*.pt`, `sub-*.zip`, FVI-named source
  files or Figshare-record-named assets. This is not a cluster-wide absence claim;
  other accounts, shared storage and deeper/unmatched paths were not searched.
- One inventory command stopped before its file checks because the remote Git
  predates `git branch --show-current`; reran with the compatible read-only
  `git symbolic-ref --short HEAD`, then completed the checks successfully.
- Outcome: no Gao data located within the authorized inspected locations.
  Next requirement is a shared dataset path with permission to read it and source
  provenance, or a separately approved download/conversion workflow. No transfer,
  numerical analysis, job, remote edit, commit or GitHub change occurred.

## 2026-11-14 - Started local Gao version 4 download

- User requested downloading the dataset locally. The official Figshare API
  reports version 4, CC BY 4.0, 44 files totaling approximately 12.038 GB,
  including 22 subject archives. Local disk check found approximately 227 GB free.
- Added `research/aditya/download_gao.py`, a standard-library downloader pinned
  to the version-4 metadata endpoint. Destination is the ignored
  `datasets/gao_raw/figshare_v4/` directory. Saved the full source manifest before
  archive transfers. The script checks size and official MD5, records SHA-256,
  preserves incomplete files as .part, and never overwrites mismatching files.
- Command: `.venv-vtc/Scripts/python.exe -u research/aditya/download_gao.py`.
  Two concurrent transfers; downloaded source scripts are stored, never executed.
  Syntax and Git whitespace checks passed; destination ignore rule verified.
- Initial requests were slow to transfer bytes. A diagnostic HEAD request
  redirected to storage and returned 403, but a one-byte GET succeeded with 206;
  the actual downloader then verified both sub-01.zip and sub-02.zip (1.142 GB).
  HEAD failure was not evidence that GET downloads were blocked.
- This starts a new versioned local dataset, not an exact copy of the project's
  historical version-3 inputs. Updated participants require fresh provenance,
  event-code and duplicate checks before analysis; do not infer that v4 fixes
  every documented issue or reuse the converter's hardcoded v3 string as proof.
- Download completion and verification will be recorded separately. No extraction,
  conversion, model fitting, cluster connection, job or GitHub update performed.

## 2026-11-14 - Local Gao version 4 download completed

- User requested a completion check. Downloader exited successfully and its final
  report confirms `complete: true`, 44 verified source files, zero errors, and
  approximately 12.038 GB downloaded, including all 22 subject ZIP archives.
- Every source file passed byte-size and official Figshare MD5 checks; SHA-256
  values are also recorded. Source metadata and the verification report remain
  under ignored `datasets/gao_raw/figshare_v4/`. No partial files remain.
- No extraction, conversion, downloaded-code execution, model fitting, cluster
  action, commit or push occurred. Next step is an archive/event/duplicate audit,
  focused first on FVI and version-4 replacements, before preparing EEG inputs.

## 2026-11-14 - Audited version-4 archives, events and shapes triggers locally

- User requested the archive/event/duplicate audit. No full extraction, conversion,
  fitting, HPC connection, job, commit or push. Work stays on the personal branch.
- Reused experiment 06's `gao_dup_scan.py` to stream BDF payload hashes (excluding
  headers) and event-file hashes directly from the 22 ZIPs. Its results are stored
  in ignored `results/gao_v4_audit/duplicates.json`; final outcome recorded below.
- Added `research/aditya/audit_gao_archives.py` for all event TSVs and BDF headers,
  plus streaming Status-channel verification on FVI runs. Exact command:
  `.venv-vtc/Scripts/python.exe -u research/aditya/audit_gao_archives.py --raw datasets/gao_raw/figshare_v4 --out results/gao_v4_audit/events.json`.
- Event/header audit completed in 68.05 seconds: 125 runs across all tasks, no
  parser errors or recorded structural issues, and no type/numeric-code label
  disagreements in the inspected metadata. This is not a signal-quality audit.
- All 42 FVI runs contain 120 events (40 per shape), totaling 5,040 events. All
  have unique within-run trial IDs, increasing/non-overlapping 4-second intervals,
  4-second declared durations, and fit the converter's full -7 to +5 second bounds.
  Minimum marker separation is 17.09 seconds. All share one 33-channel layout
  (32 EEG plus Status) at 1,000 Hz. TSV onset equals latency/fs exactly.
- Verified FVI event codes against raw BDF Status transitions in all 42 runs;
  every run matches within one sample, inside the converter's two-sample tolerance.
  This verifies trigger correspondence, not whether the imagery window is free
  of perceptual carryover or timing differences across participants.
- Confirmed 20 participants have two FVI sessions; sub-09 and sub-10 have one.
  This gives at most 4,800 cross-session-candidate events before cohort/quality
  decisions. Version-4 duplicate results must be considered before using sub-21.
- Ran metadata-only grouping assertions on actual events: chronological same-code
  groups of four use all 120 distinct trials exactly once within each run, yielding
  30 decisions/run (10 per shape). Saved membership/timing for all 1,260 bundles
  under ignored `results/gao_v4_audit/four_attempt_grouping.json`. The 42-run
  inventory includes the two single-session participants; it is not a final cohort.
- Important latency correction: from the first event marker to four seconds after
  the fourth, those retrospective groups span median 158.1955 seconds, minimum
  55.316 and maximum 820.279, including intervening other-class trials. Do not
  advertise this as a 16-second online decoder or use labels to group unknown
  incoming trials. A future repeated-intention protocol would need new validation.
- Synthetic in-memory BDF checks passed for header/channel layout, record order,
  low-16-bit Status decoding (ignoring upper-byte flags), and truncated-payload
  rejection. Git whitespace checks passed. No downloaded Python code was executed.

- Full duplicate scan completed successfully: **125 BDF data sections and 125
  event files hashed; zero duplicate groups**. Independently checked the saved
  report's row counts and empty duplicate list. The previously flagged whole-
  recording copies are therefore not byte-identical in the downloaded version 4.
  This does not establish absence of partial duplication, artifact contamination
  or other data problems, nor does it alter historical version-3 findings.
- Outcome: structural/event-level feasibility is established for a shapes
  repetition analysis, with 20 paired-session candidates in v4 (19 if retaining
  the historical sub-21 exclusion). Freeze the cohort decision before fitting;
  do not mix a new-cohort result with historical figures as if directly comparable.
  Next step is a short repetition protocol and version-aware FVI preparation,
  preserving source hashes, event IDs and timings. No scientific accuracy or
  repetition benefit has been measured by this audit.

## 2026-11-14 - Defined Experiment 08 and started local shapes preparation

- User asked to keep moving and explain the data. Confirmed the actual FVI star
  stimulus is a five-pointed outlined star (dataset name: pentagram); the other
  two classes are circle and square. This is three known stimuli repeated over
  trials, not a large collection of distinct natural images or new-object decoding.
- Added Study 2 experiment `08_repetition_budget` with a prospective protocol.
  Primary cohort: 20 v4 paired-session participants, including audited v4 sub-21;
  sensitivity summary excludes sub-21. No accuracy-based cohort selection.
- Fixed the comparison to mean class-probability pooling of four events versus
  individual-event correctness on the same held-out events, using the same fitted
  shared alpha-power/logistic-regression decoder. Both session directions, equal
  participant weighting, seed 2022, fixed estimator settings, 1,000 paired
  training-label shuffles and participant bootstrap uncertainty are specified.
  Statistical and 45%-accuracy practical criteria are kept separate; 2/8-attempt
  budgets are descriptive and cannot rescue the primary outcome.
- Explicitly documented class-informed offline grouping, full-test-session
  transductive normalization, observed acquisition latency, prior use of these
  datasets, and potential perceptual carryover. A 16-second or online decoding
  claim is not supported. Scientific results are **not run**, job numbers **none**.
- Added an experiment-specific preparer importing `convert_run` from the existing
  shared converter rather than copying its EEG implementation. It uses one
  temporary FVI run at a time, preserves original event IDs/onsets/sample indices
  and source hashes, writes v4 provenance instead of the shared CLI's hardcoded
  v3 string, and refuses to overwrite converted subjects. No downloaded code runs.
- Five preparation tests passed in 20.85 seconds: original IDs/timings, label
  disagreement, duplicate source IDs, wrong-version rejection and no-overwrite.
  Root-level CLI help and Git whitespace checks passed.
- Started `.venv-vtc/Scripts/python.exe -u studies/2_imagery_decoding/experiments/08_repetition_budget/prepare_shapes.py --raw datasets/gao_raw/figshare_v4 --out datasets/gao/v4_fvi`.
  First converted participant independently checked: 240 unique finite epochs,
  32x3000 samples, 250 Hz, -7 to +5 seconds, 40 events/class/session and preserved
  raw event timing. Full-cohort completion will be recorded separately.
- Bundled dataset_description.json says CC0 while the official v4 manifest and
  README say CC BY 4.0. Preserved these source documents and noted the inconsistency;
  no data redistribution or new permissions claim is made.
- No scientific fitting, cluster use, GitHub edits, commit or push. Experiment 07
  and its findings remain unchanged; the new experiment is indexed in Study 2.

## 2026-11-14 - Completed local FVI preparation and corrected session terminology

- All 20 paired-session participants converted successfully into ignored
  `datasets/gao/v4_fvi/`: 4,800 events across 20 files, 1,844,897,900 bytes total.
  Independently reloaded every output and checked cohort membership, v4 provenance,
  unique source-event IDs, 40 events per shape/session, raw onset/sample agreement,
  finite EEG and the expected 32x3000 shape at 250 Hz. All checks passed.
- SHA-256 of each complete prepared EEG epoch found **zero exact duplicate epoch
  groups** among 4,800 events. This extends the whole-record duplicate audit but
  does not exclude near-duplicates, artifacts, confounds or inadequate imagery.
  Saved the input-validation summary in `results/gao_v4_audit/prepared_fvi.json`.
- Timing terminology correction before any fitting: 19/20 paired FVI participants
  have matching BDF header dates between sessions; only sub-21 differs. These can
  be export dates, so they establish neither same-day nor different-day acquisition.
  The inspected sub-01 EEG sidecars contain no acquisition-date field. Changed
  the new protocol/index to **cross-session**, not verified cross-day. Actual
  acquisition days and cap replacement need independent protocol confirmation.
  Historical project claims are not silently rewritten on this evidence alone.
- Protocol SHA-256 at input-validation time:
  `861f119e353838d3a22f984408b1849cca1fea420d65f4750c8a6c072aeb4c10`.
- All raw archives are retained; only temporary single-run extraction copies were
  automatically cleaned up. Converted data remain outside Git. No scientific
  accuracy, permutation result or benefit from averaging has been measured.
- Next step is implementing and synthetically testing the repetition evaluator
  against the written protocol. No cluster job, GitHub edit, commit or push.

## 2026-11-14 - Implemented and validated the repetition evaluator

- User requested the next implementation step. Added `repetition_budget.py` and
  evaluator tests under Experiment 08. It imports shared Gao alpha-power extraction
  and logistic regression; no shared scientific code or Experiment 07 was changed.
- Implemented chronological disjoint 1/2/4/8-event groups, paired single/four-event
  evaluation with one fit per direction, deterministic class ties, both session
  directions, equal participant weighting, mean-probability pooling and constant
  controls. Test labels supply offline grouping/scoring only; they never train
  the source-session estimator or determine settings.
- Implemented 1,000-default training-label permutations, participant bootstrap
  intervals, separate statistical/practical gates and the predefined no-sub21
  sensitivity. Subset, low-permutation and noncanonical-seed runs cannot advance
  the primary gate. No scientific instance of these procedures was executed.
- Outputs preserve pre-fit membership/provenance, input/code/protocol hashes,
  seed/arguments/library versions, features/scaling, observed probabilities and
  compact per-person null results. Resume checks code, protocol, data, settings,
  library versions and cohort, and restarts after the last complete permutation.
  Fitted weights are not exposed by the shared estimator; documented that this
  format saves predictions and reproducibility inputs, not reusable model weights.
- Default invocation requires an explicit mode. `--validate-only` does not call
  feature extraction or an estimator; scientific execution requires `--run`.
  Existing output directories are refused except for explicit compatible resume.
- Final suite command:
  `.venv-vtc/Scripts/python.exe -m pytest studies/2_imagery_decoding/experiments/08_repetition_budget -q -p no:cacheprovider`.
  **24 passed in 8.43 seconds**, including the five preparation tests. Evaluator
  tests cover known signal, pure noise, shuffled training labels, metric arithmetic,
  balanced constant control/ties, grouping/order/counts, one fit per direction,
  source-fit isolation from target EEG/labels, deterministic paired shuffles,
  checkpoint consistency and interrupted synthetic-run resume equivalence.
- Synthetic end-to-end orchestration tests assert membership manifest creation
  precedes fitting and that changed fingerprints reject resume before another fit.
  Validation-mode tests make feature extraction and fitting raise if called.
- Root CLI help and Git whitespace checks passed. Ran real-input validation only:
  `.venv-vtc/Scripts/python.exe -u studies/2_imagery_decoding/experiments/08_repetition_budget/repetition_budget.py --validate-only --data datasets/gao/v4_fvi --out results/exp08_validation_20261114_final`.
  All **20 participants / 4,800 events passed**. Saved exact event/group membership
  before any scientific fit. The final-cohort median four-event span is **158.183
  seconds**, not 16 seconds. Output explicitly records scientific_models_fitted=0.
- An earlier validation-only snapshot is retained at
  `results/exp08_validation_20261114/`; the final snapshot adds library versions
  to the resume signature. Both fitted zero scientific models. Generated reports
  and datasets are ignored by Git. No scientific accuracies or p-values exist yet.
- Updated the experiment README with explicit CLI modes, artifacts, resume rules
  and synthetic test command; updated the Study 2 index. No cluster access/job,
  GitHub modification, commit or push. Next step is a separately requested small
  scientific smoke run before considering the full permutation workload.

## 2026-11-14 - Completed the first real-data repetition smoke run locally

- User requested proceeding. Ran exactly a two-participant, five-permutation
  CPU diagnostic, not the full cohort or full inference workload:
  `.venv-vtc/Scripts/python.exe -u studies/2_imagery_decoding/experiments/08_repetition_budget/repetition_budget.py --run --data datasets/gao/v4_fvi --out results/exp08_local_smoke_20261114 --subjects 1 2 --seed 2022 --permute 5`.
- Both input files validated, both session directions fitted, and all five
  training-label permutations completed. Runner exited zero. The permutation
  loop took 2.7 seconds. All expected manifest, feature/scaling, prediction,
  progress and summary artifacts exist under ignored results storage.
- Diagnostic participant-averaged single/four-event test accuracy:
  **33.3333% / 28.3333%**, three-way chance **33.3333%**; paired delta **-5.0 pp**.
  Subject 1: 28.75% / 20.00% (delta -8.75 pp). Subject 2: 37.9167% /
  36.6667% (delta -1.25 pp). Both directions and secondary budgets are saved.
- Both diagnostic permutation p-values are 1.0, but five permutations and two
  participants cannot support the planned inference. Canonical-complete,
  statistical-advance and practical-advance flags are correctly false by design.
  This is an end-to-end execution check, not the completed Experiment 08 result.
- Independently checked saved null count (five), both directions' balanced
  constant-probability controls (1/3), gate flags and presence of saved artifacts.
  No fitting settings, pooling rule, primary budget or cohort changed after this
  diagnostic. Do not use the eight-event score to rescue the primary comparison.
- Rough linear scaling of the permutation loop: 2.7/5 * (20/2) * 1000 =
  5,400 seconds, approximately 90 minutes locally, plus overhead and variation.
  This is not a promised completion time or a cluster resource measurement.
- Updated the experiment README to distinguish completed smoke from the still-
  unrun full-cohort analysis. This documentation update changes its fingerprint;
  the old smoke directory remains an immutable record of its original inputs/code/
  protocol. Use a fresh directory for the full run, not --resume on the smoke.
- No full-cohort run, HPC connection, job submission, commit, push or GitHub edit.
  Next step is the fixed full 20-person, 1,000-permutation analysis if requested;
  local CPU execution is feasible and does not require a GPU.

## 2026-11-15 - Started the full Experiment 08 analysis locally

- User explicitly ruled out HPC, then requested proceeding. Started the fixed
  20-participant, 1,000-permutation analysis on the local Windows CPU only.
  No SSH connection, transfer, cluster submission or remote operation.
- Command: `.venv-vtc/Scripts/python.exe -u studies/2_imagery_decoding/experiments/08_repetition_budget/repetition_budget.py --run --data datasets/gao/v4_fvi --out results/exp08_local_full_20261115 --seed 2022 --permute 1000`.
- Launched with a hidden window and separate ignored stdout/stderr files:
  `results/exp08_local_full_20261115.stdout.log` and
  `results/exp08_local_full_20261115.stderr.log`. Launcher PID 10096, observed
  interpreter PID 27480. The runner limits PyTorch to one CPU thread; no GPU used.
- Refused pre-existing output/log paths before launch. Confirmed the personal
  branch and preserved existing uncommitted work and the unrelated untracked file.
  Early stdout confirms input validation is progressing; no stderr observed.
- The runner saves a complete group-permutation checkpoint after each iteration.
  Preserve its code, shared helpers and protocol README while it runs so strict
  resume fingerprints remain usable. Research-log appends do not change them.
- Approximately 90 minutes was the smoke-based estimate, not a completion promise.
  Completion and final inference remain pending; no partial score was used to
  alter settings or select budgets. Keep the machine awake; interrupted work can
  resume with the identical command plus `--resume` after verifying no active copy.
- No commit, push or GitHub update. The two-person smoke stays a separate diagnostic;
  this fresh directory is the full-cohort analysis.

## 2026-11-15 - Local full-run progress check

- User requested an update. Local interpreter remains active; stdout and saved
  progress agree on **218/1,000 permutations**. Loop elapsed 1,572.2 seconds
  (about 26.2 minutes). Stderr is empty and no final summary exists yet.
- Average-rate estimate is approximately 94 more minutes; recent iterations vary,
  so this is not a completion guarantee. Process working set is about 384 MB.
- Did not inspect partial scores, change settings, restart, interrupt, access HPC,
  or modify the runner/protocol. The current iteration checkpoints remain intact.

## 2026-11-15 - Local full run still active at 648 permutations

- User asked whether finished. Stdout and saved checkpoint agree on
  **648/1,000 permutations**; process is active, stderr empty, no summary yet.
  Permutation-loop elapsed is 8,746.6 seconds (about 2 hours 26 minutes).
- The earlier runtime estimate was optimistic; iteration speed has varied
  substantially. No firm completion time is claimed. All 648 completed group
  permutations remain saved. No partial accuracy was inspected or tuning done.
- No restart, interruption, code/protocol change, remote access or HPC use.

## 2026-11-15 - Paused the local full run at the user's request

- User explicitly requested stopping now and resuming later. Confirmed the known
  worker/launcher PIDs were Python processes with matching launch times, then
  stopped only interpreter 27480 and launcher 10096. No other process or HPC job
  was stopped. Both identified processes are no longer present.
- The atomic checkpoint remains valid: **660/1,000 complete group permutations**,
  each containing all 20 participants. Manifest and observed predictions exist;
  no final summary yet. No checkpoint, dataset or model artifact was deleted.
- Resume later with the identical command plus `--resume`, using
  `results/exp08_local_full_20261115` after checking no other copy is active.
  It will retain 660 completed permutations and run the remaining 340. Any
  unfinished iteration at interruption is recomputed, not counted as complete.
- Do not edit the runner, shared helpers, protocol README, dependencies or inputs
  before resuming; fingerprint changes deliberately reject incompatible resume.
  This log-only append does not change those fingerprints. No automatic restart.

## 2026-11-15 - Resumed the local full analysis on request

- User requested continuing. Verified the saved 660 complete group permutations,
  all 20 participants per permutation, and unchanged runner/protocol/shared-code
  fingerprints. No final summary exists. Process command-line inspection was
  sandbox-denied; a supported process listing confirmed zero active python.exe
  processes before launch, preventing a second concurrent copy.
- Restarted the exact full-analysis command with `--resume`, retaining seed 2022,
  20 participants, target 1,000 permutations and the original output directory.
  It will recompute input features/verify signatures, reuse observed predictions
  and the 660 saved permutations, then continue the remaining work.
- Hidden local launcher PID 27676. Fresh stdout/stderr logs use suffix
  `exp08_local_full_20261115.resume_20261115` under ignored results storage; original
  logs are preserved. No settings, protocol, dependencies or data were changed.
- Execution is local CPU only. No HPC connection, upload, job, commit or push.
  Successful continuation beyond the checkpoint will be confirmed separately.

- Confirmed successful resume: all input checks/signatures passed, stdout reports
  permutation **661/1,000**, and progress.json independently records 661 complete
  iterations. Resumed stderr is empty. Interpreter PID 4892 remains active; no
  restart from zero or duplicate scientific run occurred.

## 2026-11-15 - Resumed run progress at 898 permutations

- User requested an update. Resumed stdout and the valid saved checkpoint agree:
  **898/1,000 permutations complete**, 102 remaining. Interpreter PID 4892 is
  active, stderr empty, and the final summary has not been generated.
- Resumed-loop elapsed is 1,735.1 seconds for 238 additional permutations. Recent
  iterations range roughly 12-27 seconds, slower than the resumed average; any
  remaining-time estimate is approximate. No partial scores inspected, settings
  changed, process interrupted, HPC accessed or GitHub content modified.

## 2026-11-15 - Resumed run progress at 987 permutations

- Read-only status check on request: **987/1,000 complete**, 13 remaining;
  checkpoint matches stdout. Local interpreter remains active, stderr empty,
  and summary.json is not present yet. Recent iterations take about 10-13 seconds.
- No partial results inspected, settings changed, restart or remote operation.

## 2026-11-15 - Full local Experiment 08 completed; no supported repetition gain

- User requested completion check. Resumed log reaches **1,000/1,000** and reports
  the saved summary; the local worker/launcher have exited. Both stderr logs are
  empty. Checkpoint has exactly 1,000 complete permutations over all 20 people.
- Participant-averaged single/four-event held-out test accuracy is
  **34.8541667% / 34.3333333%**, chance **33.3333333%**. Primary difference
  **-0.5208333 percentage points**; one-sided paired training-label p **.6823176823**.
  Four-event above-chance p **.2277722278**. Statistical and practical gates fail.
- Improvement 95% participant-bootstrap interval: **-2.7291667 to +1.7708333 pp**.
  Four-event accuracy interval: **30.1666667% to 38.4166667%**. Nine of 20 people
  improve. Two/eight-event descriptive accuracies: 35.6250% / 35.8333333%; neither
  replaces the primary four-event comparison or supports an advance claim.
- Predefined no-sub21 sensitivity: single/four 35.0219298% / 34.9122807%,
  delta -0.1096491 pp, paired p .5154845155, four-event above-chance p .1198801199.
  Excluding the replaced participant does not change the primary interpretation.
- Independently reconstructed all primary event/group accuracies from saved
  probabilities/membership and both p-values from the saved null arrays. Verified
  full cohort, completion flags and constant controls. No post-result tuning.
- Preserved the exact original README as ignored `protocol_at_run.md`, verifying
  its SHA-256 against the run manifest BEFORE adding completed results to the
  current README. Updated the experiment README and Study 2 index; Experiment 07
  remains unchanged. Outputs remain under `results/exp08_local_full_20261115/`.
- Conclusion is a protocol-specific null: four-attempt probability pooling did
  not improve cross-session shape decoding here. It is not a finding that all
  imagery decoding is impossible. No generation follow-up is supported by this test.
- All execution was local CPU. No HPC access/job, upload, GitHub edit, commit or
  push. Next action: discuss this completed result before choosing another test.

## 2026-11-15 - Prepared Experiment 08 closeout for publication

- User requested completing documentation, publication and branch push before
  considering another idea. Confirmed `aditya-individual-branch`, refreshed origin,
  and found no divergence from its remote. Current main is already an ancestor;
  no merge or changes to main were needed. PR 17 remains open for this branch.
- Updated FINDINGS.md with Experiment 08's methods, audit, completed null and
  limitations while retaining Experiment 07's detailed results. Added the result
  to the project journal/status/ledger and root README; updated study counts and
  the study/experiment indexes. Older dated journal entries remain unchanged.
- Included local download/archive-audit helpers, version-aware FVI preparation,
  evaluator and tests. No new scientific method or run was started. Data and
  generated artifacts remain ignored; the unrelated untracked file stays untouched.
- Validation: **41 tests passed in 23.90 seconds** across Experiments 07 and 08.
  Python source syntax, HTML parsing/unique IDs/newest-first journal chronology,
  published headline values against summary.json and Git whitespace checks passed.
  Inspected the explicit publication file list for data, private account paths
  and credentials. No datasets, keys, weights or run artifacts are included.
- Next action is committing/pushing this closeout only to the personal branch and
  refreshing the existing PR description so it covers both completed experiments.
  No automatic merge, new experiment, model run or HPC action is part of this work.
