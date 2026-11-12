# Findings: Do low-dimensional visual targets improve unseen-object imagery decoding?

*Researcher:* Aditya · *Branch:* `aditya-individual-branch` · *Last updated:* 2026-11-10

## The question

Does predicting a training-derived low-dimensional visual space improve held-out
imagery retrieval compared with predicting full 768-dimensional CLIP embeddings?
This tests whether target complexity contributes to the weak EEG-to-image bridge.

## Which part of the pipeline

Alpha-band EEG features → visual representation → retrieval. Image generation and
new data collection are deferred. Cluster work waits for the project lead.

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

## The result

| What | Value |
|---|---|
| Scientific pairwise improvement | **not run** |
| Chance | Pairwise 50%; top-1 1 / held-out candidates in each fold |
| Permutation p-value | **not run** |
| Sample size | Expected four people; exact shared cached stimulus/event count pending preflight |
| Test set or validation set? | Five outer unseen-stimulus folds; four inner selection folds |
| Cluster job numbers | **pending lead approval** |
| Local verification | 15 synthetic tests passed; Python and shell syntax checks passed. See LOG.md |

## What it means for us

No evidence of improvement has been measured yet. The implementation provides an
explicit test of the target-space hypothesis. The paired training-label null does
not establish population-level model superiority, especially with only four people.

## What I would do next

Have the project lead review the fixed protocol, cache coverage, job resources and
run commands before any cluster execution. A follow-up requires positive mean gain,
paired-null p ≤ .05 with at least 1,000 permutations and directional improvement in
at least three participants. Even a passing gate requires review; it triggers no
automatic perception/imagery training or image generation.
