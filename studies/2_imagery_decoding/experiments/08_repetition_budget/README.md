# 08 - Can four imagery attempts improve cross-session shape decoding?

**Status:** completed locally on CPU, including all 1,000 training-label permutations.
Four-attempt pooling did not improve the primary metric. No HPC used.
Cluster job numbers: **none**. No job script or submission is part of this step.

## Question

Does averaging predictions from four independent, instructed imagery events improve
cross-session classification compared with single-event predictions on the same events?
This is a known-three-shape task, not unseen-object retrieval or image reconstruction.

## Data and cohort, fixed before fitting

- Gao visual imagery dataset, Figshare record 30227503, **version 4**. Official
  metadata and README state CC BY 4.0; the bundled dataset_description.json instead
  says CC0. Preserve both documents and attribution rather than silently resolving
  that inconsistency or redistributing the data here.
- FVI only: circle, square and five-pointed outlined star (called pentagram).
- Primary cohort: the **20 people with two FVI sessions**, excluding sub-09/sub-10.
  Includes version-4 sub-21: the archive audit found no whole-recording duplicates
  in v4, unlike historical v3. Predefined sensitivity: repeat the group summary
  without sub-21; this cannot rescue a failed primary test.
- 40 distinct events per shape/session: 120 events/session, 240/person,
  **4,800 candidate events**. Eligibility/technical failures must be reported;
  no participant selection based on accuracy. Signal-quality validation remains
  distinct from the completed header/event/whole-record duplicate checks.
- The picture is presented before each imagery period. Existing source markers and
  trigger checks do not by themselves rule out perceptual carryover.
- Use **cross-session**, not verified cross-day: 19 of 20 paired FVI participants
  have matching BDF header dates across sessions. These may be export dates and
  do not establish same-day acquisition either. Different acquisition days and
  independent cap replacement need source-protocol confirmation, not inference
  from session names alone.

## Fixed comparison

1. Import shared Gao feature extraction and the shared logistic-regression estimator.
   Use 8-13 Hz log power per electrode from 0-4 seconds after the recorded marker.
   No ICA, new architecture, target PCA, feature search or hyperparameter search.
2. Preserve the existing label-free per-session feature standardization. It uses
   the complete test session without labels, so this is **offline/transductive
   calibration**, not a demonstration of real-time deployment.
3. Train on session 1, test session 2; then reverse. Fixed l2 .01, 600 Adam steps,
   learning rate .05, seed 2022, using the shared estimator's training scaling.
4. Save source-event membership before fitting. Within each participant/session/
   true shape, sort by event latency and partition into disjoint groups of four.
   Require exactly 40 events/shape; fail rather than silently drop an incomplete
   group or choose favorable events. Test labels determine instructed-repeat
   membership only, never estimator fitting, tuning or normalization.
5. **Single-event arm:** mean correctness of individual predictions on all 120
   held-out events. **Four-event arm:** average the three class probabilities
   within each group, then argmax, producing 30 decisions (10 per shape).
   Use identical fitted models, events and probability predictions for both arms.
   Resolve prediction ties by the fixed order circle, pentagram, square.
6. Average directions within each person, then weight people equally. Primary
   outcome: group mean of participant-wise four-event minus single-event accuracy.
   Chance for both arms: 1/3. Report both directions, all people and actual counts.

## Controls, uncertainty and decision

- 1,000 training-label permutations. Within each source session/person, shuffle
  event labels, refit, and reuse the same fit for both arms. For each permutation,
  keep held-out true labels and membership fixed. Repeat both directions and the
  group statistic; permute neither individual windows nor already scored outputs.
- Report one-sided Monte Carlo p=(1 + count(null >= observed))/1001 for the paired
  improvement and four-event above-chance performance. Include their null arrays.
- Bootstrap participants (10,000 draws, seed 2022) for 95% intervals. Directions,
  bundles and alternate regroupings are not independent participants.
- Statistical advance requires positive primary improvement and both p-values
  <= .05. Separately label practical usefulness against **45% three-way accuracy
  within four attempts**, a project benchmark, not a clinical/deployment threshold.
  A statistical pass without that level is a limited gain, not a practical success.
- Secondary budgets 2 and 8 use predefined disjoint groups and are descriptive;
  do not select the best budget or use them to rescue the primary result.
- A constant class/probability control must give balanced 1/3 accuracy. Synthetic
  and leakage tests must pass before any scientific fitting.

## Interpretation and latency

These are retrospective groups of known same-shape events interleaved with other
shapes. Four events contain 16 seconds of imagery-window EEG, but their observed
marker-to-final-window span is median **158.1955 seconds** across all 42 FVI runs
(including single-session participants), range 55.316-820.279 seconds. Report
timing again on the final cohort. This is not a 16-second online decoder, four
uncued consecutive attempts, or a method for grouping unknown incoming intentions.

Earlier Gao results have already informed this question; this is a prospective
follow-up analysis on previously studied data, not independent confirmation.
If it fails, report it without changing the frozen protocol. If it succeeds,
validate a realistic repeated-intention recording protocol before a live demo.
Image generation remains a separate question.

## Local preparation (no model fitting)

From the repository root:

```powershell
.venv-vtc/Scripts/python.exe studies/2_imagery_decoding/experiments/08_repetition_budget/prepare_shapes.py --raw datasets/gao_raw/figshare_v4 --out datasets/gao/v4_fvi
```

The preparer imports the existing shared converter, temporarily unpacks one FVI
run at a time, and saves 250-Hz, 32-channel, 12-second epochs (-7 to +5 seconds).
It preserves v4 provenance, source checksums, original event IDs and latencies.
No downloaded scripts are executed. Existing converted outputs are never overwritten.
Raw/converted datasets and generated reports stay outside Git.

## Evaluator usage

Run from the repository root; only load the trusted tensors produced above.
Input validation writes all event/group assignments without extracting EEG features
or fitting any estimator:

```powershell
.venv-vtc/Scripts/python.exe studies/2_imagery_decoding/experiments/08_repetition_budget/repetition_budget.py --validate-only --data datasets/gao/v4_fvi --out results/exp08_validation
```

Scientific execution requires an explicit, separate `--run` invocation. This
command shows the completed run configuration (the actual output directory was
`results/exp08_local_full_20261115`):

```powershell
.venv-vtc/Scripts/python.exe studies/2_imagery_decoding/experiments/08_repetition_budget/repetition_budget.py --run --data datasets/gao/v4_fvi --out results/exp08_full --seed 2022 --permute 1000
```

`--subjects` allows subset diagnostics; subset runs, fewer than 1,000 permutations
or a different seed cannot pass the canonical advance gate. Existing outputs are
refused unless `--resume` is supplied for a scientific run. Resume requires the
same input hashes, code, protocol, cohort, permutation count and settings, and
continues after the last complete group permutation. It does not overwrite inputs.
No automatic downloads, cluster connections or submissions are performed.

Saved artifacts include the pre-fit membership/provenance manifest, Git commit
and code hashes, arguments and library versions, raw/standardized features and
session scaling parameters, observed event/group probabilities, compact per-person
permutation results, participant-bootstrap intervals, controls and decision gates.
The shared estimator returns probabilities, not fitted weight checkpoints; inputs,
settings and seeds are retained for rerunning it. Generated files are ignored by Git.

Run local tests without fitting the real EEG:

```powershell
.venv-vtc/Scripts/python.exe -m pytest studies/2_imagery_decoding/experiments/08_repetition_budget -q -p no:cacheprovider
```

## Diagnostic smoke run (not the primary result)

Local run `results/exp08_local_smoke_20261114/`, participants 1 and 2, seed 2022,
five training-label permutations. Single/four-event mean test accuracy was
33.3333% / 28.3333%, chance 33.3333%; difference -5.0000 percentage points.
Both participants' differences were negative. These two participants and five
permutations are an execution check, not a full-cohort conclusion. The canonical
gate is disabled for this subset/count. Do not retune settings, choose the better
secondary budget, or select participants based on this diagnostic.

Predictions, features/scaling, pre-fit manifest and all five permutation records
were saved; constant-control and smoke-gate checks passed. The permutation loop
took 2.7 seconds. Linear scaling suggests about 90 minutes for 20 participants
and 1,000 permutations on this CPU, excluding preparation/overheads; this is an
estimate, not a runtime guarantee. No cluster job was used.

## Full result - 15 November 2026

Twenty participants, 4,800 held-out event predictions across both session
directions; directions averaged within person, then people weighted equally.
These are cross-session tests on the same three shapes, not unseen-object tests.

| Measure | Result |
|---|---|
| Single-event accuracy | 34.8542% |
| Four-event accuracy | 34.3333% |
| Chance | 33.3333% |
| Four minus single | -0.5208 percentage points |
| Paired training-label permutation p | .682318 (1,000 permutations, one-sided) |
| Four-event above-chance permutation p | .227772 (1,000 permutations, one-sided) |
| 95% participant-bootstrap interval for improvement | -2.7292 to +1.7708 percentage points |
| 95% participant-bootstrap interval for four-event accuracy | 30.1667% to 38.4167% |
| Participants improving | 9/20 |
| Statistical / practical advance | Both fail |

Predefined sensitivity without sub-21 also found no supported benefit:
35.0219% single versus 34.9123% four, paired p .515485; four-event above-chance
p .119880. Secondary two/eight-event accuracies were 35.6250% / 35.8333%; these
are descriptive, not replacement primary outcomes.

All 1,000 permutations completed across an initial local run and a checkpointed
resume. Constant controls passed; primary accuracies and p-values were independently
reconstructed from saved predictions, group assignments and null records. The
run's exact protocol text was preserved as `protocol_at_run.md` beside its original
manifest before adding this result section. Large artifacts remain ignored by Git.

Conclusion: four-event probability averaging did not yield a supported gain for
this fixed alpha-power/logistic-regression protocol. This does not establish
that imagery is intrinsically undecodable or rule out other recordings/methods.
Do not retune on these results or proceed to generation on their strength.
