# Signal quality research: how to get more out of the EEG we have

Compiled 2026-09-22, after the corrected imagination baseline came back null and the
project pivoted from cross-subject to per-subject decoding.

Companion to [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md), which holds the engineering
backlog. This file holds the *evidence* -- what the literature says about where the
imagery signal actually is, and what that implies for our input representation.

---

## 0. The reframe: effective sample size

Each subject imagined each picture **once**, for 10 seconds. Cutting that into 20
windows produces 20 rows in a file, but not 20 measurements. The number that governs
what any model can learn is the count of independent imagery events.

| Dataset | Imagery events per person | Rows per person |
|---|---|---|
| Shimizu 2022 (ours), after stimulus filtering | **33** | 660 |
| Gao et al. visual-imagery dataset (Sci Data 2026) | **800** | 800 |

~24x difference. No architecture, loss or regulariser closes that gap. Everything
below is about extracting more from 33 events, and it has a low ceiling.

---

## 1. Where the imagery signal lives (three independent sources agree)

| Source | Finding |
|---|---|
| [Xie et al., *Current Biology* 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7342016/) | Imagery and perception share a representational code **only in the alpha band (8-13 Hz)** -- absent in theta and beta. Present in **posterior electrodes only** (parieto-occipital). Imagery information spans **600-2280 ms**, peak 1340 ms. Cross-decoding perception<->imagery works bidirectionally. |
| [Stecher & Kaiser, *Sci Rep* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11150249/) | Imagined scenes and their properties decodable from **8-13 Hz, peaking at 11 Hz**. Scene properties decodable *exclusively* from alpha. Used multitapers because "imagery data tends to be noisy". |
| [Shimizu & Srinivasan 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9491577/) (our dataset) | Attention analysis: imagination classified by **low-frequency (<15 Hz)** activity over temporal cortex; perception by **high-frequency (>35 Hz)** occipital/frontal. |

### Why this matters for us

The signal is **induced** oscillatory power -- not phase-locked to the cue. Both
imagery papers extract it with Morlet wavelets or multitapers.

What we currently feed the model:

- raw broadband time-domain signal
- all 128 channels, no posterior weighting
- 500 ms windows (5 alpha cycles -- marginal for power estimation)
- upsampled 125 -> 512 samples, which adds no information

A time-domain network *could* learn to compute band power, but discovering that from
~360 training examples is not realistic. Supplying alpha power directly is a large
representational shortcut for modest effort.

### Window length

500 ms is short relative to what the imagery literature uses (2-4 s). Xie's
600-2280 ms window also implies Shimizu's earliest windows may carry perception
after-effects rather than imagery. Worth testing:

- 2 s windows (5 per episode) instead of 500 ms (20 per episode) -- fewer rows,
  better SNR per row
- a per-window analysis: which part of the 10 s actually carries signal?

---

## 2. Do NOT add artifact removal

[Kessler, Enge & Skeide, *Communications Biology* 2025](https://www.nature.com/articles/s42003-025-08464-3)
([preprint](https://arxiv.org/abs/2410.14453)) systematically varied preprocessing and
measured decoding accuracy:

- **every artifact-correction step reduced decoding accuracy**
- higher high-pass cutoffs consistently improved results
- baseline correction helped EEGNet; lower low-pass cutoffs helped time-resolved models

Their own caveat is the important part: uncorrected artifacts inflate accuracy because
the model exploits structured noise rather than neural signal. Ocular artifacts in
particular can covary with stimulus category.

**Our policy:** do not add ICA. If we obtain a positive result, it owes an
ocular-artifact control before we believe it.

---

## 3. Small models beat large ones at this data scale

- [Gao et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886826/): **EEGNet 75.8%**
  (animals), 75.1% (figures), 62.0% (objects) vs **CSP+KNN ~50%/~41%**; chance 33%/25%.
  EEGNet is a few thousand parameters.
- Shimizu got 13.4% (40-way, chance 2.5%) with Sinc-EEGNet, also small.
- [ATM-S / THINGS-EEG, NeurIPS 2024](https://arxiv.org/abs/2403.07721): a few-million-parameter
  encoder is state of the art; large pretrained encoders do not win.
- [Riemannian geometry review](https://arxiv.org/pdf/2407.20250): covariance +
  tangent-space methods "only require small training samples" -- the standard strong
  baseline in low-data EEG.

Our Stage 2 trains **85M parameters on ~360 trials per subject**. This is the wrong
side of that literature.

---

## 4. Proposals, ordered by information per hour

### S1 -- Classical baselines on alpha-band features (cheapest, highest information)
*Answers: is there ANY decodable imagery signal in this data?*

Per subject, on the same leakage-free window split:

1. Bandpass 8-13 Hz, posterior channels, compute induced power per window
2. Classifier A: Riemannian covariance -> tangent space -> logistic regression
3. Classifier B: EEGNet on the band-limited signal
4. Report 33-way accuracy vs 3.03% chance, and the trial-averaged version

Hours of work, minutes of CPU, **no GPU**. If both come back at chance, no amount of
diffusion work will rescue this dataset, and we learn that in a day rather than in
GPU-weeks. If either beats chance, we have a target for the conditioning encoder to
match, and a signal we know exists.

### S2 -- Alpha-band / time-frequency input representation
Feed the main pipeline band power (or a compact time-frequency map) rather than raw
broadband samples. Targets the representation gap in section 1.

### S3 -- Window-length experiment
2 s windows vs 500 ms; plus per-window analysis of where in the 10 s the signal is.

### S4 -- Small encoder (P7 in the main plan)
Informed by whatever S1 reports. EEGNet- or ATM-S-class, ~10^4-10^6 params.

### S5 -- Second dataset: Gao et al. 2026
See section 5.

---

## 5. Gao et al. visual imagery dataset (Scientific Data, 2026)

Gao J, Liu Y, Li Z, Huang K, Wang F, Xu J, Zhao L, Li T, Fu Y. *An EEG Dataset for Visual Imagery-Based Brain-Computer Interface.* Scientific Data 13, 194 (2026). [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886826/) ·
[Data: figshare 10.6084/m9.figshare.30227503](https://doi.org/10.6084/m9.figshare.30227503) ·
CC BY-NC-ND 4.0

| | |
|---|---|
| Participants | 22 (19 completed both sessions) |
| Sessions | 2 per participant |
| Channels / rate | 32 (10-20 system) / 1000 Hz |
| Classes | 10, in 3 groups: figures (circle, square, pentagram), animals (dog, fish, bird), objects (cup, chair, watch, scissors) |
| Trials | **40 per class per session**; 400 per session; **800 per participant** |
| Trial structure | 17 s: 3 s fixation, 4 s image viewing, mask, **4 s imagery**, 4 s rest |
| Baselines | EEGNet 75.8% / 75.1% / 62.0%; CSP+KNN ~50% / ~41%; chance 33% / 25% |

### Why it matters to this project

1. **It has the design we lack.** Few classes, many real repetitions, two sessions.
   Session-level holdout is possible, which is the strictest honest protocol short of
   cross-subject.
2. **It has a published, reproducible target.** 75.8% with EEGNet on 3-class is a
   number to reproduce, which validates our pipeline against a known-good result
   instead of against silence.
3. **4-second imagery windows** match the literature's timing rather than 500 ms.
4. **It is the template for our own recordings.** 10 classes, 40 trials each, two
   sessions, 32 channels -- this is what a collection protocol should look like, and
   we can copy it rather than invent one.
5. **Cue-then-imagine structure** matches ours, so a model trained here is a plausible
   initialisation for our own data later.

### Caveats

- Only 32 channels vs our 128 (fine -- posterior coverage is what matters).
- 1000 Hz vs 250 Hz; needs downsampling to match.
- No perception-only trials, so the Shimizu "Mix" trick does not directly transfer.
- Different classes from ours, so models are not directly interchangeable -- but the
  *encoder* and the *method* are.
- Licence is NonCommercial-NoDerivatives: fine for research, check before
  redistributing anything derived from it.

---

## Decisions pending

- [ ] Approve S1 (classical baselines) as a detour from the diffusion pipeline
- [ ] Decide whether to pull in the Gao et al. dataset as a second track
- [ ] Fold approved items into IMPROVEMENT_PLAN.md's priority ordering

---

## 6. Source protocol details (verified from the paper, 2026-09-24)

Quoted verbatim from Shimizu & Srinivasan 2022 Methods. These settle several
questions that were previously assumed.

| Detail | What the paper says | Why it matters |
|---|---|---|
| Eyes during imagery | *"told to imagine the image with eyes opened"* | **Eyes OPEN.** A secondhand summary circulating as "eyes closed" is wrong. Eyes closed would flood occipital alpha with the idle rhythm and swamp the content-specific modulation we are trying to decode. Open is the condition the imagery literature works in. |
| Windowing | *"each 10 second recording was split into 20 trials of 500 ms without overlap"* | Non-overlapping and contiguous, so the 20 windows tile the full 10 s. **We can concatenate adjacent windows to build 2 s or 4 s segments** from the published files -- this unblocks S3 with no new data. |
| Acquisition | *"128-channel NeuroScan EEG system with 2 kHz sampling rate"* | The raw 2 kHz data is NOT what is published. |
| Downsampling | *"After removing artifacts, all trials were downsampled to 250 Hz"* | We have the cleaned, decimated version. 125 samples = 500 ms. |
| Filtering | *"filtered by a band-pass filter with frequency of 1-50 Hz"* | Alpha (8-13 Hz) is intact, so the S1/S2 strategy is safe. Shimizu's own >35 Hz perception finding is only partly preserved. |

**Inherited caveat.** Artifact removal was applied before the data were shared.
Kessler, Enge & Skeide (section 2) found every artifact-correction step reduces
decoding accuracy. Our numbers may be slightly depressed by 2022 preprocessing
choices, and we cannot undo them. Worth stating in any writeup.

---

## 7. Candidate datasets evaluated and rejected

Four criteria a candidate must meet for this project: **(a) visual imagery, not
perception or motor imagery; (b) multiple real repetitions per class; (c)
occipital / parieto-occipital coverage; (d) ideally >1 session for an honest holdout.**

A fifth, practical: our encoder starts with `Conv1d(128 -> 1024)`, so the channel
count is baked into the architecture. A different montage is not "more data" -- it
requires rebuilding the input layer and retraining Stage 1. The bar is high.

| Dataset | What it is | Verdict |
|---|---|---|
| [MindBigData ImageNet](https://www.mindbigdata.com/opendb/imagenet.html) | **One subject** (the author), Emotiv Insight 5 ch @ 128 Hz: AF3, AF4, T7, T8, Pz. 14,012 images, ~1 rep each, 3 s while **viewing**. ODbL. | **No.** Disqualifying: **no occipital electrodes at all** -- Pz is parietal midline; O1/O2/Oz absent. We would be looking for alpha imagery signal where the device cannot see. Also perception not imagery, and 1 rep per image repeats our own dataset's flaw. Note: the advertised "70,060 brain signals" is 14,012 trials x 5 channels, not 70,060 trials. |
| [OpenNeuro ds007162](https://openneuro.org/datasets/ds007162/versions/1.0.0) | "Adaptive recruitment of cortex-wide recurrence for visual object recognition". 34 participants, 1 session, 64 ch actiCHamp @ 1000 Hz. 242 images (121 "challenge" selected where humans beat AlexNet, 121 control). RSVP: 200 ms image + 100 ms blank, sequences of 14; task was **paperclip detection**. CC0. [preprint](https://www.biorxiv.org/content/10.1101/2025.10.17.682937v2) | **No.** Excellent dataset, wrong problem: 100% perception, and participants were not even attending to object category. 200 ms presentations are the opposite of the 2-4 s windows imagery needs. Our perception pipeline already works; perception is not where we are blocked. |
| [PhysioNet EEG MMI](https://www.physionet.org/content/eegmmidb/1.0.0/) | 109 subjects, 64 ch @ 160 Hz, BCI2000, EDF+. Imagining fist / foot movements. ODC-BY. | **No.** *Motor* imagery -- mu/beta over sensorimotor cortex. The decoded output is "left hand", not image content; there is nothing to reconstruct. Fine dataset, unrelated problem. |
| [meagmohit/EEG-Datasets](https://github.com/meagmohit/EEG-Datasets) | Curated list. Categories: motor imagery, emotion, ErrP, VEP, ERP, SCP, resting state, music, eye movements, misc, clinical. | **Largely exhausted.** There is **no visual-imagery category**; the only image-related entries are MindBigData and a working-memory set. That scarcity is itself informative -- public *visual* imagery EEG barely exists, which is why Gao et al. 2026 stands out. |

Still the only recommended addition: **Gao et al. 2026** (section 5).
