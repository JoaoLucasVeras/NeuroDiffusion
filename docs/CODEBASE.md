# Codebase guide

A map of the repository for anyone starting their own experiment. It says what each part
does, which parts are current and which are legacy, where your work plugs in, and the
rules every result has to follow.

Read the [project documentation](journal.html) first if you have not: it explains the
science and the results. This page is about the code.

---

## 1. The pipeline in one picture

```
 brain recording ──► 1 DATA ──► 2 FEATURES ──► 3 DECODER ──► 4 CLIP SPACE ──► 5 GENERATOR ──► image
 (EEG .mat / .bdf)    convert,    band power,     classifier     768 numbers       Stable Diffusion
                      clean,      waveforms,      or linear      describing a      draws from the
                      split       covariance      map            picture           768 numbers
                                                                      │
                               6 EVALUATION ◄──────────────────────────┘
                     honest tests: other day / later trials, shuffled-label null
```

Every experiment changes one box and keeps the rest, so its effect can be measured.

## 2. Folder map

The work is split into **studies** (one research question each, with its own approach)
and **experiments** (one test with one result, inside a study). See
[studies/README.md](../studies/README.md) for the definitions.

| Folder | What lives there | Status |
|---|---|---|
| `shared/` | The toolbox every study imports: data converters, decoders, generators, evaluation. See [shared/README.md](../shared/README.md) | **Current, reuse it** |
| `shared/third_party/` | The Gao authors' own scripts, byte for byte (CC BY 4.0) | **Do not edit**: kept unchanged on purpose |
| `studies/1_dreamdiffusion/` | The original DreamDiffusion model and its jobs (85M-parameter encoder + Stable Diffusion 1.5) | Finished, kept as it was. Do not edit unless your experiment is about this model |
| `studies/2_imagery_decoding/` | The main study: 6 numbered experiments, each folder with its scripts, job scripts and a README | Current |
| `studies/3_anchored_imagery/` | Protocol, stimulus program, simulator and analysis for the anchored-imagery study | Current, no hardware yet |
| `cluster/` | One-time setup: environments, downloads, data conversion | Run once |
| `docs/` | Documentation; `journal.html` is the main page. `docs/archive/` holds September planning notes | Read |
| `research/` | **Your space.** `research/<your-name>/` for notes and write-ups, from `TEMPLATE.md` | Yours |
| `datasets/`, `results/`, `logs/`, `pretrains/` | Data, outputs, job logs, model weights | **Not in git.** On the cluster |

**Run everything from the repository root**, e.g.
`python shared/decoding/classical_baseline.py --help`. Every script finds `shared/`,
`datasets/` and `results/` from its own location (`shared/paths.py`), so no `cd` is needed.

## 3. Every script, by stage

### Data: convert, check, split (`shared/data/`)

| Script | Does | Status |
|---|---|---|
| `prepare_gao_data.py` | Converts the Gao et al. 2026 dataset (22 people) to `.pt`, verifying every event | **Current main dataset** |
| `gao_ica_prep.py` | The same, cleaned with the Gao authors' own ICA preprocessing | Current |
| `prepare_shimizu_data.py` | Converts the first dataset (Shimizu & Srinivasan 2022, 4 people) to `.pth` | Current |
| `make_splits.py` | Leakage-free train/test splits for that dataset | Current |
| `fetch_stimuli.py`, `download_stimuli.py` | Get its ImageNet stimulus pictures | Current |
| `check_data.py` | Pre-flight check: run before any job | **Always** |

### Features and decoders (`shared/decoding/` and the study 2 experiments)

Experiment folders are under `studies/2_imagery_decoding/experiments/`.

| Script | Does | Status |
|---|---|---|
| `shared/decoding/classical_baseline.py` | **The shared toolbox.** Band-pass filter, band power, Riemannian features, logistic regression, a small EEGNet. Most scripts import from it | **Current, reuse it** |
| `shared/decoding/gao_baseline.py` | Gao decoding: across days, within a day (shuffled vs time-ordered), seeing-to-imagining transfer | Current |
| `01_shimizu_signal/small_encoder.py` | A small learned network (91k parameters), first dataset | Superseded by the next |
| `01_shimizu_signal/bp_clip_encoder.py` | **Band power → CLIP numbers**, a linear map (about 99k parameters). The current best encoder for image generation | **Current** |
| `01_shimizu_signal/head_backbone_diag.py` | Which part of a model loses the signal | Done |
| `03_gao_reproduction/gao_eegnet.py`, `gao_repro_*.py` | Re-running the Gao paper's EEGNet result | Done (not reproduced) |
| `04_shapes_eye_check/gao_eyecheck.py` | Is a signal brain or eye movement? By region, by second, eye channels alone | Done (not the eyes) |
| `05_shapes_enhance/gao_enhance.py` | Six stronger decoders on imagined shapes, with targets set in advance | Done (none beat alpha power) |
| `06_duplicate_scan/gao_dup_scan.py` | Finds duplicated recordings in the Gao files | Done; exclude sub-20 objects, sub-21 shapes/animals |
| `studies/1_dreamdiffusion/stageA1_eeg_pretrain.py`, `eeg_ldm.py` | The original DreamDiffusion training | Legacy, but `eeg_ldm.py --cond_encoder` can use our new encoder |

### Generation (`shared/generation/`)

| Script | Does | Status |
|---|---|---|
| `generate_imagevar.py` | **Generator B:** Stable Diffusion Image Variations, draws from CLIP numbers. Arms: brain / true picture / average | **Current default** |
| `studies/1_dreamdiffusion/eeg_ldm.py`, `gen_eval_eeg.py` | **Generator A:** DreamDiffusion | Kept, swappable |
| `precache_models.py` | Downloads models on the login node (compute nodes have no internet) | Run once |

### Evaluation (`shared/evaluation/`)

| Script | Does |
|---|---|
| `eval_report.py` | Scores generated images against no-information baselines |
| `clip_metrics.py`, `eval_metrics.py` | The metric functions |
| `compare_reports.py` | Puts runs side by side |

## 4. The data you will use

Both converted datasets are on the cluster under `~/NeuroDiffusion/datasets/`, never in git.

**Gao et al. 2026, the main dataset:** `datasets/gao/gao_sub-XX.pt`, one file per person (22).

```python
d = torch.load("datasets/gao/gao_sub-01.pt")
d["fs"]        # 250 samples per second
d["tmin"]      # -7.0: each trial runs from 7 s before to 5 s after the imagery cue
d["channels"]  # 32 electrode names, Fpz ... O2
trial = d["dataset"][0]
trial["eeg"]      # tensor (32, 3000): channels x samples
trial["task"]     # "AVI" animals | "FVI" shapes | "OVI" objects
trial["class"]    # e.g. "circle"
trial["session"]  # 1 or 2 (different days); sub-09 and sub-10 have only 1
trial["trial"]    # order within the recording
```

Time windows inside a trial: **seeing** the picture is −5.5 to −1.5 s, **imagining** is 0 to 4 s.
`datasets/gao_ica/` holds the same trials after the authors' cleaning (imagery window only).

**Shimizu & Srinivasan 2022, the first dataset:** `datasets/imagination_5_95_std.pth` and
`visual_5_95_std.pth`, with splits from `make_splits.py`. Four people; each picture imagined
once. See [DATA.md](DATA.md) for its traps.

## 5. Where your experiment plugs in

| If your idea is about… | Start from | Change |
|---|---|---|
| Cleaning the signal (artifacts, ICA, filtering) | `shared/data/prepare_gao_data.py`, `gao_ica_prep.py` | Write a new converter that outputs the same `.pt` format, then run `gao_baseline.py --data <yours>` on it |
| New features (bands, connectivity, time windows) | `shared/decoding/gao_baseline.py` `features()`, `classical_baseline.py` | Add a feature function; keep the split and null |
| A new decoder or model | `05_shapes_enhance/gao_enhance.py` (a template for "several methods, one fair test") | Add a method; compare against `alpha` as the reference |
| Brain → CLIP mapping | `01_shimizu_signal/bp_clip_encoder.py` | Change the map; keep its saved-checkpoint format so generators can load it |
| Image generation | `shared/generation/generate_imagevar.py` | Swap the pipeline; keep the brain / true picture / average arms |
| A new dataset | `shared/data/prepare_gao_data.py` | Convert it to the same `.pt` layout, so every script works on it unchanged |
| The anchored experiment | `studies/3_anchored_imagery/` | Read its README and PROTOCOL first |

**The fastest start:** copy the experiment folder closest to your idea (for example
`05_shapes_enhance/`) to a new number, change one thing, run it on two or three people,
then submit the full run. Import from `shared/` rather than copying it.

## 6. The rules every result follows

These come from mistakes this field (and this project) has already made. A result that
breaks one is not accepted.

1. **Test on data the model never saw: a different day, or later trials.** Never shuffle
   trials from one recording into both train and test; drift makes that look like signal.
2. **Run a shuffled-label null.** Repeat the whole method with the labels scrambled. If the
   score barely drops, it was never brain signal.
3. **Fix settings before seeing the test result.** No picking the best of many settings,
   epochs or methods on the test data; if you try several, correct for it.
4. **Hunt for shortcuts.** Eye movements, blinks, recording drift, the cue on screen.
   `gao_eyecheck.py` shows how.
5. **State chance level and sample size** with every number.
6. **Exclude the duplicated recordings:** sub-21 shapes and animals, sub-20 objects.

## 7. Running on the cluster

```bash
ssh <you>@coe-hpc1.sjsu.edu          # login node, has internet (VPN needed)
ssh coe-hpc3                         # scheduler: submit and check jobs here
cd ~/NeuroDiffusion
sbatch studies/2_imagery_decoding/experiments/02_gao_baseline/s12_gao_baseline.sh
                                     # submit; EXTRA_ARGS="..." passes flags through
squeue -u <you>                      # your jobs
```

- **Template:** copy `05_shapes_enhance/s17_gao_enhance.sh` (CPU, one task per person) or
  `03_gao_reproduction/s13_gao_eegnet.sh` (GPU). Job scripts `cd` to the repository root
  and call the script by its path. Keep the job script next to the experiment it runs.
- **Prefer CPU** (`--partition=cpuqs`). Most of our analyses do not need a GPU.
- **GPU:** one GPU per job, at most 2–3 at once, and never the A100/H100 nodes unless needed.
  `--exclude=cs002` (a faulty GPU).
- Outputs go to `results/<job>_<id>/`, logs to `logs/`. The shared home drive is nearly
  full: delete what you do not need.
- More in [RUNBOOK.md](RUNBOOK.md).

## 8. Your workflow

1. Branch from `main`: `git switch -c <name>/<topic> origin/main`.
2. New experiment: a new numbered folder in the study it belongs to, with a README (copy one).
   New question: a new study folder. Notes and write-ups go in `research/<name>/` (from `TEMPLATE.md`).
3. Test on fake or small data first (see how `studies/3_anchored_imagery/simulate.py` plants a known effect).
4. Run it; keep the result JSON and the job number.
5. Write up in `research/<name>/`: question, method, result with chance level, what it means.
6. Open a pull request into `main`. **Add, don't replace:** new experiments beside the old ones.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the git details and
[team-guide.html](team-guide.html) for how independent projects work.
