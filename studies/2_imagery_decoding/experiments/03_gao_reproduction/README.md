# 03 · Gao reproduction: does the paper's 75.8% hold up?

**Question.** The Gao et al. paper reports 75.8% for imagined animals with EEGNet. Do we get
that with their own code and cleaning?

| Script | Does |
|---|---|
| `gao_eegnet.py` | Their EEGNet, scored four ways: as published (best test accuracy over epochs), the same on scrambled labels, honest shuffled 5-fold, honest time-ordered |
| `gao_repro_export.py` | Exports our converted data in the folder layout their script expects (real and scrambled labels) |
| `gao_repro_run.py` | Runs their `EEGNet.py` **byte for byte** (checksum printed), with only the data folder swapped in |
| `gao_repro_summary.py` | Collects the accuracies their script prints |

Their scripts are in [`shared/third_party/`](../../../../shared/third_party/), unchanged (CC BY 4.0).

| Job script | Job | Result |
|---|---|---|
| `s13_gao_eegnet.sh` (option A, our cleaning) | 87908 | As published 43%, scrambled labels also 43%: the scoring itself produces the number |
| `s14_gao_ica.sh`, then `s13` with `--data datasets/gao_ica` (option B, their cleaning) | 88067, 88171 | Published 43% = scrambled 44%; honest at chance, except shapes time-ordered 37.3% |
| `s16_gao_repro.sh` (their exact script, animals, all 22 people) | 92377 | 42.8% real vs 41.7% scrambled (paired p .12); the paper reports 76.6% / 74.4% |

**Conclusion.** Not reproduced. Their script reports the best test accuracy over all
epochs and steps its learning rate on test accuracy, which inflates the score even with
scrambled labels. The remaining gap to 75.8% is unexplained.
