# Study 2: imagery decoding

**Question.** Can we read *which* picture someone is imagining from their EEG, when the
test is honest: a different day, or later trials, and a shuffled-label null?

**Data.** First the Shimizu & Srinivasan (2022) set (4 people, each picture imagined once),
then the Gao et al. (2026) set (22 people, 3 tasks: shapes, animals, objects, two days each).
Both are converted once by the scripts in [`shared/data/`](../../shared/data/).

**Answer so far.** Only imagined **shapes** decode, at about 1.05 to 1.1 times chance,
across days and within a day. The published 75.8% did not reproduce with the authors' own
code, and no stronger decoder lifted the signal. That is why
[study 3](../3_anchored_imagery/) designs a new kind of recording.

## Experiments, in the order we ran them

| # | Experiment | Question | Result |
|---|---|---|---|
| 01 | [shimizu_signal](experiments/01_shimizu_signal/) | Where did the first model lose the signal? Does it reach the images? | Alpha band power holds it (3 of 4 people); the generator works, but the signal is too faint to reach the images |
| 02 | [gao_baseline](experiments/02_gao_baseline/) | On the 22-person set: across days, within a day, seeing to imagining | Shapes 35.0% vs 33.3% chance across days (p .0002); animals and objects at chance |
| 03 | [gao_reproduction](experiments/03_gao_reproduction/) | Does the paper's 75.8% hold up with their own code? | No: 42.8% with real labels vs 41.7% with scrambled ones |
| 04 | [shapes_eye_check](experiments/04_shapes_eye_check/) | Is the shapes signal brain or eye movement? | Not slow eye movement; spread across the head |
| 05 | [shapes_enhance](experiments/05_shapes_enhance/) | Can six stronger decoders lift it? | No: best 37.7% three-way, targets (45%) missed |
| 06 | [duplicate_scan](experiments/06_duplicate_scan/) | Are any published recordings duplicated? | Yes: sub-21 shapes/animals, sub-20 objects. Excluded everywhere |

Shared code these experiments import: [`classical_baseline.py`](../../shared/decoding/classical_baseline.py)
(features, logistic regression, EEGNet) and [`gao_baseline.py`](../../shared/decoding/gao_baseline.py)
(the Gao loader, splits and null).

## Running one

Every experiment runs from the repository root, on the cluster:

```bash
cd ~/NeuroDiffusion
sbatch studies/2_imagery_decoding/experiments/05_shapes_enhance/s17_gao_enhance.sh
```

Job scripts keep their original names (`s12_...`, `s17_...`), so they match the job numbers
and log files quoted in the journal.
