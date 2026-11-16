# Findings: imagery decoding experiments 07 and 08

*Researcher:* Aditya · *Branch:* `aditya-individual-branch` · *Last updated:* 2026-11-15

## Latest result: four imagery attempts did not improve shape decoding

Experiment 08 asked whether averaging predictions from four separate imagery
events improves classification across recording sessions. We kept the alpha-power
features and logistic-regression model fixed, so both arms used the same fitted
models and source events. Details: [Experiment 08](../../studies/2_imagery_decoding/experiments/08_repetition_budget/).

### Data and checks

- Gao dataset, Figshare 30227503 version 4: circle, square and a five-pointed star
  (called pentagram). Twenty participants with two sessions, 40 events per
  shape/session: **4,800 events**. These are three repeated stimuli, not 4,800 images.
- Downloaded 44 files with official size/MD5 verification. Audited 125 EEG payloads
  and 125 event files; no exact whole-recording duplicates in this v4 release.
  All 42 shapes runs passed trigger/event checks. No exact duplicate prepared
  epochs were found. These checks do not establish artifact-free or high-quality EEG.
- Version-4 sub-21 was included in the fixed primary cohort; a predefined
  sensitivity excluded it. Historical version-3 results are not replaced.
- Trained on one session and tested on the other, then reversed; averaged
  directions within person, then weighted people equally. No hyperparameter search.
- Grouped four events of the same known shape chronologically. This is an offline,
  instructed-repeat analysis, not a way to group unknown intentions online. Test-
  session normalization was label-free but transductive. Median group span was
  **158.183 seconds**, not a demonstrated 16-second decision latency.
- Use **cross-session**, not verified cross-day: header dates may be export dates
  and do not independently establish acquisition days or cap replacement.

### Held-out results

| Measure | Result |
|---|---|
| Single-event / four-event accuracy | 34.8542% / 34.3333% |
| Three-way chance | 33.3333% |
| Four minus single | -0.5208 percentage points |
| Paired training-label permutation p | .682318; 1,000 permutations, one-sided |
| Four-event above-chance permutation p | .227772; 1,000 permutations, one-sided |
| 95% participant-bootstrap interval for improvement | -2.7292 to +1.7708 percentage points |
| Participants improving | 9/20 |
| Statistical / practical advance | Both fail; four-event accuracy also below the fixed 45% benchmark |
| Sensitivity without sub-21 | 35.0219% single / 34.9123% four; paired p .515485 |
| Execution | Local Windows CPU only; 1,000 permutations completed with checkpointed resume; no HPC job |
| Validation | 24 local tests; input checks on all 4,800 events; saved scores and p-values independently reconstructed |

The balanced constant-probability control gave 33.3333%. Secondary two/eight-event
accuracies were 35.6250% / 35.8333%; they do not replace the primary comparison.

Four-event probability pooling did not provide a supported gain under this fixed
protocol. This does not show that all imagery is undecodable or identify a single
limiting component. We did not retune after inspecting the results, and this test
does not justify moving to image generation.

Code, tests and the full protocol are in the experiment folder. The append-only
[log](LOG.md) records commands, audits, interruptions and checks. Generated
artifacts remain in ignored `results/exp08_local_full_20261115/`, including the
original protocol snapshot, provenance manifest, predictions, scaling and nulls.

## Experiment 07: low-dimensional visual targets

### The question

Does predicting a training-derived low-dimensional visual space improve held-out
imagery retrieval compared with predicting full 768-dimensional CLIP embeddings?
This tests whether target complexity contributes to the weak EEG-to-image bridge.

### Which part of the pipeline

Alpha-band EEG features → visual representation → retrieval. Image generation and
new data collection are deferred. The approved CPU smoke and full cluster runs
are complete; any new submission requires separate approval.

### If you changed a method

- Adds Experiment 07 in Study 2: `studies/2_imagery_decoding/experiments/07_low_dimensional_clip/`.
- Compares full-CLIP ridge with PCA-target ridge using identical regularization choices
  and nested stimulus holdouts. Uses shared filtering and band-power functions.
- The reference is recomputed on the same splits. Historical early/late window
  results cannot serve as the baseline for an unseen-object claim.
- Every imagery event is represented once by averaging its twenty window features.

### If you used a dataset or paper

- Inspiration: Wadia et al. (2026), [A shared code for perceiving and imagining objects
  in human ventral temporal cortex](https://doi.org/10.1126/science.adt8343).
  Intracranial single-neuron evidence motivates the hypothesis; scalp EEG and CLIP PCA
  do not replicate their measurements or original visual representation.
- Dataset: Shimizu & Srinivasan (2022), [OSF](https://osf.io/2fgks/), already used by
  NeuroDiffusion. Four participants, 128 electrodes including posterior coverage,
  250 Hz converted data. One imagery event per image/class per participant; twenty
  stored windows are not twenty repetitions. Approximately 33 images are available.
- Data/image redistribution and license permissions are not changed or extended;
  the implementation consumes existing trusted local data and commits none of it.

### What I did

Prepared nested, whole-stimulus cross-validation with training-only feature scaling,
target PCA and model selection. Added synchronized training-label permutations,
oracle and mean-target controls, saved manifests and model parameters, local synthetic
checks, and a guarded CPU job script. Details and exact
commands are in the experiment README and chronological LOG.md.

Ran the reviewed protocol on the uploaded, validated local inputs using the shared
HPC environment: smoke job 113015 followed by full job 113017, both on nsfq with
no GPU allocation. All 1,000 training-label permutations completed in the full run.

### The result

| What | Value |
|---|---|
| Full-target pairwise accuracy | 52.2475% |
| Reduced-target pairwise accuracy | 52.2980% |
| Scientific pairwise improvement | +0.0505 percentage points; no evidence of improvement under the paired training-label null |
| Chance | Pairwise 50%; top-1 averages 15.1515% across held-out fold candidate sets |
| Paired-delta permutation p-value | 0.477522 (1,000 permutations, one-sided) |
| Reduced-arm above-chance permutation p-value | 0.122877 (1,000 permutations, one-sided) |
| Participants improving | 2 of 4; required at least 3 |
| Follow-up gate | **FAIL** |
| Sample size | Local EEG verified: four people, 40 stimuli each; matched cache covers 33 stimuli/person, giving 132 usable imagery events |
| Test set or validation set? | Five outer unseen-stimulus folds; four inner selection folds |
| Cluster job numbers | Smoke 113015 and full 113017 completed successfully on nsfq, g29; no GPU allocation |
| Full-run resources | 25 min 56 s; approximately 1.1 GB peak RSS; two logical CPUs allocated, code limited to one thread |
| Local verification | 17 synthetic tests passed; all 3,200 local EEG windows and 160 complete events audited. See LOG.md |

### What it means for us

Reducing the visual target's rank did not provide a meaningful or statistically
supported retrieval gain under this fixed protocol. The reduced model also failed
the above-chance permutation requirement. Do not advance to perception-plus-imagery
training or image generation on the strength of this result.

This is a null for this particular combination of alpha-power event features,
ridge regression and training-derived CLIP PCA on four people with 33 usable imagery
events each. It does not disprove shared VTC codes or rule out other representations.
The full-target ridge arm already predicts within the span of training images, so
the actual comparison is additional rank truncation of a small-data linear model.
The paired training-label null does not establish population-level model superiority.

Secondary checks: full/reduced top-1 retrieval was 18.1818% / 16.6667%, respectively.
The PCA oracle achieved 73.2071% pairwise accuracy; it uses true target images at
scoring and is an oracle diagnostic, not EEG performance. The mean-target
control achieved 50% pairwise accuracy, as expected.

Per-participant pairwise accuracy (full / reduced): subject 1, 55.56% / 58.69%;
subject 2, 54.14% / 51.11%; subject 3, 47.47% / 49.80%; subject 4, 51.82% / 49.60%.
Small directional gains in two people were offset by losses in the other two.

### Closeout

Experiment 08 is now complete as a separate follow-up and also found no supported
gain. Preserve both results and their fixed settings; do not tune on these test
outcomes and present the resulting best score as confirmation. Any later test
needs a separate question, numbered experiment and criteria set before fitting.

The small downloaded reports are in ignored `results/s19_hpc_full_113017/`:
`summary.json`, `manifest.json`, and `s19_visual_axes_113017_1.out`. Full fitted models
and predictions remain in the personal HPC checkout at `~/NeuroDiffusion/results/s19_113017/`.
