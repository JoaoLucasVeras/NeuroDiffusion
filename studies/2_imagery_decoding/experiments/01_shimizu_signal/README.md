# 01 · Shimizu signal: where did the signal go?

**Question.** The DreamDiffusion model (study 1) produced chance-level images from imagined
EEG. Is there imagery signal in this data at all, which part of the model lost it, and does
it reach generated images once we keep it?

**Data.** Shimizu & Srinivasan 2022, 4 people, 33 imagined pictures each.

| Script | Does |
|---|---|
| `small_encoder.py` | A small learned encoder (91k parameters) into CLIP space |
| `head_backbone_diag.py` | Swaps backbone (band power vs learned) and head (classifier vs CLIP) to find which loses the signal |
| `bp_clip_encoder.py` | **Band power → CLIP**, a linear map (99k parameters), l2 chosen on training data only. Saves the encoder the generators load |
| `clip_embed_diag.py` | How far apart the stimulus pictures are in CLIP space |
| `make_joint_dataset.py` | Merges seeing and imagining trials (an early idea, not used in the results) |

| Job script | Jobs | Result |
|---|---|---|
| `s3_small.sh`, `s4_subjects.sh`, `s6_subj4.sh` | 84326, 84567 | Small encoder: signal on alpha band power, subject 1 at 3.25× chance |
| `s7_ckpt_verify.sh` | | Checks the saved encoder file format |
| `s8_diag_bp.sh`, `s8_diag_eeg.sh` | 84691, 84692 | Band power keeps the signal on 3 of 4 people; every learned backbone loses it |
| `s9_bp_clip.sh` | 85801 | Honest encoder: 2.25 / 2.25 / 1.25 / 2.25× chance, p .050 / .035 / .41 / .035 |
| `s10_imagevar.sh` | 85810 | Generator B from true pictures beats shuffled by 33–34 SEM (the generator works); from brain predictions 0.3 / 1.1 / 0.0 / 0.3 SEM (the signal does not reach the images) |

**Conclusion.** The signal is real but too faint, and this dataset too small (one imagined
trial per picture), to drive image generation. Next: a bigger dataset (experiment 02).

Generator B itself is shared: [`shared/generation/generate_imagevar.py`](../../../../shared/generation/generate_imagevar.py).
