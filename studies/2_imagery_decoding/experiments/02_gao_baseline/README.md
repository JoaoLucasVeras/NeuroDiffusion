# 02 · Gao baseline: does imagery decode across days?

**Question.** On the Gao et al. 2026 set (22 people; shapes, animals, objects; two days),
can a simple decoder tell which picture is imagined, tested on a different day? Within a
day, does the split matter? Can a decoder trained on *seeing* read *imagining*?

**Script.** [`shared/decoding/gao_baseline.py`](../../../../shared/decoding/gao_baseline.py), which is shared
because experiments 04 and 05 import it. Alpha band power plus logistic regression, with a
200-shuffle null. The data is converted once by `cluster/s11_prepare_gao.sh`.

| Run | Job | Result |
|---|---|---|
| `s12_gao_baseline.sh` (train day 1, test day 2) | 87331 | Seeing decodes on all tasks (the positive control passes); imagery at chance |
| `EXTRA_ARGS="--align session"` (per-day standardising, no labels) | 87790 | Shapes imagery decodes across days, 1.11×, p .0018; animals and objects at chance |
| `EXTRA_ARGS="--split within"` (shuffled vs time-ordered) | 87812 | Shuffled splits inflate seeing results (16–19 vs 5–9 significant sessions) |
| `EXTRA_ARGS="--split transfer"` (seeing → imagining) | 88200 | Shapes only, p .019 |

**Corrected after the duplicate scan (experiment 06), sub-21 excluded:**
- shapes across days: 35.0% vs 33.3% chance (p .0002);
- same day, time-ordered: 37.1% (p .005);
- seeing → imagining: p .013.

Imagined shapes are decodable, faintly.
