# Findings: Do low-dimensional visual targets improve unseen-object imagery decoding?

*Researcher:* Aditya · *Branch:* `aditya-individual-branch` · *Last updated:* 2026-11-11

## The question

Does predicting a training-derived low-dimensional visual space improve held-out
imagery retrieval compared with predicting full 768-dimensional CLIP embeddings?
This tests whether target complexity contributes to the weak EEG-to-image bridge.

## Which part of the pipeline

Alpha-band EEG features → visual representation → retrieval. Image generation and
new data collection are deferred. The approved CPU smoke and full cluster runs
are complete; any new submission requires separate approval.

## If you changed a method

- Adds Experiment 07 in Study 2: `studies/2_imagery_decoding/experiments/07_low_dimensional_clip/`.
- Compares full-CLIP ridge with PCA-target ridge using identical regularization choices
  and nested stimulus holdouts. Uses shared filtering and band-power functions.
- The reference is recomputed on the same splits. Historical early/late window
  results cannot serve as the baseline for an unseen-object claim.
- Every imagery event is represented once by averaging its twenty window features.

## If you used a dataset or paper

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

## What I did

Prepared nested, whole-stimulus cross-validation with training-only feature scaling,
target PCA and model selection. Added synchronized training-label permutations,
oracle and mean-target controls, saved manifests and model parameters, local synthetic
checks, and a CPU job draft guarded for project-lead approval. Details and exact
commands are in the experiment README and chronological LOG.md.

Ran the reviewed protocol on the uploaded, validated local inputs using the shared
HPC environment: smoke job 113015 followed by full job 113017, both on nsfq with
no GPU allocation. All 1,000 training-label permutations completed in the full run.

## The result

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
| Local verification | 17 synthetic tests passed after lead review; all 3,200 local EEG windows and 160 complete events audited. See LOG.md |

## What it means for us

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

## What I would do next

Review the completed null with Joao before choosing a different hypothesis. Preserve
this result and the fixed settings; do not tune on these held-out folds and present
the resulting best score as confirmation. Any follow-up should be a separate
numbered experiment with its own advance criteria and approval.

The small downloaded reports are in ignored `results/s19_hpc_full_113017/`:
`summary.json`, `manifest.json`, and `s19_visual_axes_113017_1.out`. Full fitted models
and predictions remain in the personal HPC checkout at `~/NeuroDiffusion/results/s19_113017/`.
