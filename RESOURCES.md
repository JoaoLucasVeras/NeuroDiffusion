# Resources: datasets and papers

Running record of every dataset and paper the team has proposed or found, with a
plain verdict on whether it helps *this* project.

A "not suitable" verdict is about fit, not quality. Most of the datasets below are
good science, carefully collected, and several are better built than the one we are
using. They are recorded here with the specific reason they do not fit, so nobody
has to re-derive it and so the reasoning can be challenged if it is wrong.

Each entry is tagged with who surfaced it, so credit and follow-up questions land
in the right place.

Companions: [SIGNAL_QUALITY_RESEARCH.md](SIGNAL_QUALITY_RESEARCH.md) (the evidence
on where the imagery signal lives) and [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)
(the engineering backlog).

---

## What we are actually looking for

The project's goal is to reconstruct an *imagined* image from EEG. That narrows the
field far more than "EEG + images" does. A candidate dataset needs:

1. **Visual imagery** — not perception, and not motor imagery. The imagery has to
   have image content that could in principle be reconstructed.
2. **Multiple real repetitions per class** — repetitions are what let us average
   away noise. Slicing one long recording into many windows does not count; those
   are pieces of one event, not separate measurements.
3. **Parieto-occipital coverage** — three independent papers place imagery content
   in alpha-band activity over posterior electrodes. Without those electrodes the
   signal is not recoverable regardless of method.
4. **More than one session**, ideally — so a whole recording day can be held out.
   This is the strictest honest test short of cross-subject.

And one practical constraint that drives several verdicts below: our encoder begins
with `Conv1d(128 channels -> 1024)`. **The channel count is baked into the
architecture.** A dataset with a different montage is not simply "more data" — it
requires rebuilding the input layer and retraining Stage 1 from scratch. EEG
datasets are not interchangeable the way image datasets are.

---

# Datasets

## In use

### Shimizu & Srinivasan (2022) — our current dataset
*Found by: Joao*
[Data (OSF)](https://osf.io/2fgks/) · [Paper](https://doi.org/10.1371/journal.pone.0274847) · [Code](https://github.com/shimihirouci/Improve_Imagination)

4 subjects (3M/1F), 128-channel NeuroScan acquired at 2 kHz, shared at 250 Hz after
artifact removal and 1–50 Hz band-pass. Two paradigms: a **visual** experiment
(7,987 trials, 2,000 distinct stimuli) and an **imagination** experiment (3,200
trials = 4 subjects x 40 stimuli x 20 windows).

Protocol: grey 20x20 cm canvas, fixation cross, stimulus for 500 ms, 500 ms noise
mask to flush buffered activity, then **10 seconds of imagining with eyes open**.
Each 10 s recording split into 20 non-overlapping 500 ms windows.

**Verdict: in use, and the source of the project's one positive result.** The
perception half works — generated images beat the shuffled-pairing control by
4.3 SEM on a stimulus-disjoint split. The imagination half is the hard part, for a
structural reason: each person imagined each picture exactly once, so there are
**33 imagery events per person**, not 660. The 20 windows are one event sliced thin.

Two details worth knowing, both verified from the Methods:
- Windows are **non-overlapping and contiguous**, so 2 s and 4 s segments can be
  built by concatenating adjacent windows. No new data needed.
- Artifact removal was applied **before** the data were shared. We cannot undo it,
  and per Kessler et al. below that may slightly depress our decoding numbers.

---

## Recommended addition

### Gao et al. (2026) — EEG dataset for visual imagery BCI
*Found by: Joao*
[Paper](https://doi.org/10.1038/s41597-025-06512-5) · [Data (figshare)](https://doi.org/10.6084/m9.figshare.30227503) · CC BY-NC-ND 4.0

22 participants (19 completed both sessions), 32 channels at 1000 Hz, 10 classes in
three groups (figures / animals / objects). **40 trials per class per session**, 400
per session, **800 per participant**. Trial structure: 3 s fixation, 4 s viewing,
mask, **4 s imagery**, 4 s rest. Published baselines: EEGNet 75.8% / 75.1% / 62.0%;
CSP+KNN ~50% / ~41%; chance 33% / 25%.

**Verdict: the one dataset worth adding.** It is the only candidate that clears all
four criteria, and the reason it stands out is a single number: **40 repetitions per
class**. Averaging 40 repeats improves signal-to-noise by roughly sqrt(40) ~ 6x
before any modelling. Our dataset offers no equivalent.

Three further reasons:
- **A target to debug against.** 75.8% with a small model is a number we can try to
  reproduce. You cannot debug a pipeline against a null result; you can debug it
  against a published one.
- **Two sessions** make a whole-day holdout possible — the honest test for anything
  a person would actually use.
- **It is the template for our own recordings.** Few classes, many repetitions, two
  sessions, 32 channels. Validated and published. Copying it beats inventing one.

Caveats: different classes from ours, so weights are not transferable (the method
is). 32 channels vs our 128 means a Stage 1 rebuild. Needs downsampling to match.
No perception trials, so the Shimizu joint-training trick does not carry over.

---

## Evaluated, not suitable for this project

### MindBigData "ImageNet of the Brain"
*Found by: Aditya*
[Site](https://www.mindbigdata.com/opendb/imagenet.html) · ODbL 1.0

One subject (the dataset's author), Emotiv Insight consumer headset, **5 channels at
128 Hz: AF3, AF4, T7, T8, Pz**. 14,012 images, roughly one repetition each, 3 s
recorded while viewing.

*Note on the headline figure:* the advertised "70,060 brain signals" is 14,012
trials x 5 channels, not 70,060 trials. 14,012 x 5 = 70,060, and the stated 26.85M
data points match 70,060 x 384 samples (3 s at 128 Hz).

**Verdict: not suitable.** The instinct was sound — it is by far the largest
single-person image-EEG collection that exists, and for a subject-specific project
that is genuinely the right thing to look for. The blocker is the montage:
**there are no occipital electrodes.** Pz is parietal midline; O1, O2 and Oz are
absent. Having just committed to the alpha-band parieto-occipital strategy, we would
be searching for the signal in the one place this device cannot see. Secondary
issues: the 3 s window is during viewing (perception, not a separate imagery
period), and one repetition per image reproduces the exact limitation we are trying
to escape.

### OpenNeuro ds007162 — cortex-wide recurrence in object recognition
*Found by: Aditya*
[Dataset](https://openneuro.org/datasets/ds007162/versions/1.0.0) · [Preprint](https://www.biorxiv.org/content/10.1101/2025.10.17.682937v2) · CC0

34 participants, 1 session, 64-channel EASYCAP / BrainVision actiCHamp at 1000 Hz
(10-10). 242 images: 121 "challenge" (selected where human performance beats
AlexNet) and 121 control. RSVP sequences of 14 images, 200 ms each plus 100 ms
blank; the participants' task was **detecting whether a paperclip appeared**.
Derivatives include time-resolved object-identity decoding.

**Verdict: not suitable for the imagery goal — but the best-built dataset anyone has
proposed.** 64 channels, 1000 Hz, CC0, 34 participants, decoding baselines included:
if this project were about perception it would be a strong candidate, and it is
worth remembering for that reason. Two blockers here: it is 100% perception with no
imagery period at all, and the participants were attending to paperclips rather than
object category. The 200 ms presentations are also the opposite of the 2–4 s windows
imagery needs. Our perception pipeline already works; perception is not where we are
stuck.

### PhysioNet EEG Motor Movement/Imagery
*Found by: Aditya*
[Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/) · ODC-BY 1.0

109 subjects, 64 channels at 160 Hz, BCI2000, EDF+. 14 runs per subject.
Imagining opening/closing fists and feet.

**Verdict: not suitable.** The word "imagery" is doing a lot of work across two
different literatures, and this is the other one. Motor imagery lives in the mu/beta
rhythm over sensorimotor cortex (C3/C4), and what gets decoded is "left hand" or
"both feet" — there is no image content, so there is nothing for a diffusion model
to reconstruct. It is the standard motor-BCI benchmark and an excellent dataset; it
simply answers a different question. Worth knowing it exists if the project ever
branches into control rather than reconstruction.

### meagmohit/EEG-Datasets — curated list
*Found by: Aditya*
[Repository](https://github.com/meagmohit/EEG-Datasets)

Categories: motor imagery, emotion recognition, error-related potentials, VEPs,
ERPs, slow cortical potentials, resting state, music, eye blinks/movements,
miscellaneous, clinical.

**Verdict: largely exhausted for our purposes — and that is itself a finding.**
There is **no visual-imagery category at all**. The only image-related entries are
MindBigData and a working-memory set. Public *visual* imagery EEG barely exists.
That explains why Gao et al. stands out as much as it does, and it is a concrete
argument for collecting our own data rather than continuing to search.

### THINGS-EEG (OpenNeuro ds003825) — noted, not formally evaluated
*Found by: Joao*
[Dataset](https://openneuro.org/datasets/ds003825)

50 subjects, 22,248 images from 1,854 object concepts, RSVP at 10 Hz. The field's
main EEG visual-decoding benchmark; most 2024–2026 papers report on it.

**Verdict: perception, so not directly useful — but worth keeping in view** as the
benchmark our methods would be compared against if we ever publish on the perception
side.

---

# Papers

## The dataset paper and its result we are measured against

*Found by: Joao*

**Shimizu H, Srinivasan R (2022).** *Improving classification and reconstruction of
imagined images from EEG signals.* PLOS ONE 17(9): e0274847.
[doi](https://doi.org/10.1371/journal.pone.0274847)

Reports 13.4% ± 4.2 on imagination (40-way, chance 2.5%) with a small Sinc-EEGNet,
rising to **25.2% ± 2.2 when perception data is added to training** — the single
most transferable result for us. Also reports, via attention analysis, that
imagination is classified by sub-15 Hz activity while perception uses >35 Hz
occipital activity.

**Useful: yes, centrally.** Two caveats we have to carry: all their models are
per-subject with no held-out-subject test, and with one recording per person-picture
their 80/10/10 split necessarily puts windows from the *same* 10-second recording on
both sides of the split — including the windows immediately adjacent in time. That
is why our numbers are lower on the same data.

## Where the imagery signal lives

*Found by: Joao*

**Xie S, Kaiser D, Cichy RM (2020).** *Visual Imagery and Perception Share Neural
Representations in the Alpha Frequency Band.* Current Biology 30(13), 2621–2627.e5.
[doi](https://doi.org/10.1016/j.cub.2020.04.074)

Imagery and perception share a representational code **only in the alpha band
(8–13 Hz)** — absent in theta and beta — and **only in posterior electrodes**.
Imagery information spans 600–2280 ms, peaking at 1340 ms. Cross-decoding works in
both directions.

**Useful: yes, this is the most actionable paper we have found.** It says our input
representation is probably wrong: we feed raw broadband signal from all 128 channels
in 500 ms windows, when the target is induced alpha power over posterior channels
across seconds.

*Found by: Joao*

**Stecher R, Kaiser D (2024).** *Representations of imaginary scenes and their
properties in cortical alpha activity.* Scientific Reports 14, 12796.
[doi](https://doi.org/10.1038/s41598-024-63320-4)

Independent confirmation: imagined scenes and their properties decodable from
8–13 Hz, peaking at 11 Hz, and scene properties decodable *exclusively* from alpha.
Used multitapers specifically because "imagery data tends to be noisy."

**Useful: yes.** Second independent source for the same conclusion, plus a concrete
feature-extraction recipe.

## Preprocessing

*Found by: Joao*

**Kessler R, Enge A, Skeide MA (2025).** *How EEG preprocessing shapes decoding
performance.* Communications Biology.
[journal](https://www.nature.com/articles/s42003-025-08464-3) ·
[preprint](https://arxiv.org/abs/2410.14453)

Systematic sweep of preprocessing choices against decoding accuracy. **Every
artifact-correction step reduced decoding accuracy.** Higher high-pass cutoffs
helped; baseline correction helped EEGNet. Their own caveat matters most:
uncorrected artifacts can *inflate* accuracy because the model exploits structured
noise, and ocular artifacts in particular covary with stimulus category.

**Useful: yes, as a policy.** Do not add artifact correction hoping for better
numbers. But if we obtain a positive result, it owes an ocular control.

*Found by: Aditya*

**Singh B, Wagatsuma H (2017).** *A removal of eye movement and blink artifacts from
EEG data using morphological component analysis.* Computational and Mathematical
Methods in Medicine 2017, 1861645. [doi](https://doi.org/10.1155/2017/1861645)

Morphological Component Analysis for ocular artifacts: decomposes the signal by
sparsity across three dictionaries (UDWT for slow drift, DST for oscillations,
DIRAC for spikes) rather than relying on statistical independence the way ICA does.
Reconstruction correlation > 0.98; validated on simulations plus 8 participants at
23 channels / 500 Hz.

**Useful: yes — but as a control, not as an enhancement, and not yet.** This is a
legitimate and well-validated technique, and the concern behind it is real: subjects
imagined with their **eyes open**, staring at a grey square for ten seconds, so they
blinked, and blinks are 100+ µV against roughly 10 µV of EEG. It is entirely
plausible that imagining a complex scene produces different eye behaviour than
imagining a simple object — which would hand us a spurious result.

Three reasons it is not a preprocessing step to add now: our data is already
artifact-corrected upstream; the Kessler result says correction reduces decoding
accuracy; and the mechanism behind that result (artifacts carrying task-correlated
information) is something we want removed rather than exploited.

Where it belongs: **the moment a classical baseline or a model comes back positive,
strip ocular artifacts and show the result survives.** If it does, we have neural
decoding. If it vanishes, we caught ourselves before publishing an eye-movement
detector. MCA is arguably better suited to that job than ICA, since it needs less
prior knowledge about which component is the artifact.

One honest open question in the paper's favour: Kessler et al. tested *ICA-based*
correction. MCA is a different mechanism and may be more surgical. Nobody appears to
have tested it for decoding specifically, so that is a genuine unknown rather than a
settled point.

## Artifact removal

A coherent set surfaced by Aditya, all on ICA and artifact rejection. Read together
with the Kessler entry above, which pulls in the opposite direction, and the
Hajhassani entry below, which resolves the tension.

*Found by: Aditya*
**Jung T-P, Makeig S, Humphries C, Lee T-W, McKeown MJ, Iragui V, Sejnowski TJ
(2000).** *Removing electroencephalographic artifacts by blind source separation.*
Psychophysiology 37:163-178. [doi](https://doi.org/10.1111/1469-8986.3720163)

**The foundational paper** for ICA-based EEG artifact removal -- eye movements,
blinks, cardiac, muscle and line noise. Everything downstream, MCA included, is a
response to this.

**Useful: yes, as background.** Worth reading regardless of what we decide, because
it defines the vocabulary the rest of the artifact literature uses.

*Found by: Aditya*
**Jiang X, Bian G-B, Tian Z (2019).** *Removal of artifacts from EEG signals: A
review.* Sensors 19(5):987. [doi](https://doi.org/10.3390/s19050987)

Survey of regression, wavelets, PCA/ICA/CCA, EMD and hybrid approaches. Concludes
"there is no optimal choice for remove all types of artifacts": ICA handles diverse
artifacts, CCA and EMD do better on muscle.

**Useful: yes, as an orientation map** for choosing a method once we decide we want
one.

*Found by: Aditya*
**Automatic removal of the eye blink artifact from EEG using an ICA-based template
matching approach** (2006). [PubMed 16537983](https://pubmed.ncbi.nlm.nih.gov/16537983/)

Automates the awkward part of ICA -- deciding *which* component is the blink -- by
matching component scalp topographies against a fixed template. Validated on 18
subjects.

**Useful: yes, if we go the ICA route.** Manual component selection does not scale
and is a source of experimenter bias; this removes both problems.

*Found by: Aditya*
**Denoising of EEG signal based on word imagination using ICA for artifact and noise
removal on unspoken speech** (2021).
[ResearchGate](https://www.researchgate.net/publication/350817495)

ICA denoising applied to imagined *speech*. Could not fetch the full text
(ResearchGate blocks automated access), so this summary is from the title alone.

**Useful: possibly.** A different imagery modality with the same technique. Worth a
skim for whether their artifact handling transfers to visual imagery.

*Found by: Joao* (while following up Aditya's batch)
**Hajhassani D, Aristimunha B, Graignic P-A, Mellot A, Kusch L, Delorme A, Semah T,
Caillet AH (2026).** *From EEG cleaning to decoding: The role of artifact rejection
in MI-based BCIs.* [arXiv:2605.12408](https://arxiv.org/abs/2605.12408)

Benchmarks artifact rejection against decoding accuracy across **13 public
datasets**. Key finding, verbatim: *"Rejection effects are strongly subject- and
regime-dependent, with the largest gains in low-baseline/low-SNR conditions, so it
should be used adaptively."*

**Useful: yes -- and it changed our plan.** Kessler et al. found cleaning hurts, but
their tasks were perception and visual search, which are high-SNR. Ours is imagery:
low SNR, low baseline, subjects sitting at chance. That is precisely the regime where
this paper says cleaning helps. So "do not add artifact removal" was too strong a
policy, drawn from evidence in the wrong regime. The classical baseline (S1) now runs
**both arms** -- cleaned and uncleaned -- and lets the data decide. Aditya's reading
directly changed the experimental design.

## Architecture and method

*Found by: Joao*

**Li D, et al. (2024).** *Visual Decoding and Reconstruction via EEG Embeddings with
Guided Diffusion (ATM-S).* NeurIPS 2024. [arXiv:2403.07721](https://arxiv.org/abs/2403.07721)

State of the art on THINGS-EEG with a **few-million-parameter** encoder (channel
attention + temporal-spatial convolution + projector), trained contrastively against
CLIP image embeddings, feeding a two-stage SDXL/IP-Adapter pipeline.

**Useful: yes, as an architecture target.** Large pretrained encoders do not win
here. Our Stage 2 trains 85M parameters on ~360 examples per subject.

*Found by: Joao*

**Riemannian Geometry-Based EEG Approaches: A Literature Review** (2024).
[arXiv:2407.20250](https://arxiv.org/pdf/2407.20250)

Covariance matrices as points on an SPD manifold; tangent-space mapping then
ordinary Euclidean classifiers. These methods "only require small training samples."

**Useful: yes — this is the basis of the S1 classical baseline.** It is the standard
strong baseline in low-data EEG and runs in minutes on CPU.

*Found by: Joao*

**Interpretable EEG-to-Image Generation with Semantic Prompts** (2025).
[arXiv:2507.07157](https://arxiv.org/abs/2507.07157)

Aligns EEG to LLM-generated multilevel semantic captions rather than directly to
images. Reports a transformer EEG encoder with spatial and temporal attention, and
that CLIP-style contrastive loss beat MSE in their ablations.

**Useful: partly.** The contrastive-over-regression finding matches what we
independently implemented. The caption-conditioning idea is interesting but their
headline numbers are on the EEGCVPR dataset, which has known block-design problems.

*Found by: Joao*

**Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion
Models** (2024). [arXiv:2410.02780](https://arxiv.org/abs/2410.02780)

ControlNet-based conditioning of a latent diffusion model on EEG, with minimal
preprocessing.

**Useful: marginally.** A simpler conditioning path than ours, worth a look if we
rebuild the conditioning stage.

*Found by: Joao*

**Cross-Subject Generalization for EEG Decoding: A Survey.**
[arXiv:2604.27033](https://arxiv.org/html/2604.27033v1)

Catalogues the transfer toolbox: Euclidean/Riemannian alignment, adversarial
subject-invariance, subject-conditioned contrastive learning, meta-learning,
few-shot calibration.

**Useful: deferred.** We deliberately dropped cross-subject transfer as too
ambitious for now. Two items stay relevant: per-subject alignment, and few-shot
calibration — which is exactly the recipe for adapting a pretrained model to a new
person with a short calibration session.

*Found by: Joao*

**EEG foundation models** — [CBraMod](https://github.com/wjq-learning/cbramod)
(ICLR 2025, 9,000 h TUEG, 19 channels), LaBraM (2,000 h), REVE (25,000 subjects,
arbitrary montages).

**Useful: not yet.** All pretrained on clinical EEG rather than visual tasks, and
most assume 10-20 montages. REVE handles arbitrary channel sets, which makes it the
one worth revisiting if the montage-mismatch problem ever becomes the bottleneck.

*Found by: Joao*

**The Perils and Pitfalls of Block Design for EEG Classification Experiments**
(Li et al., IEEE TPAMI ~2021) — *citation not re-verified in this session.*

The critique that invalidated several high-profile EEG image-decoding results:
when all trials of a class are recorded in one contiguous block, a classifier can
learn slow drift rather than stimulus content.

**Useful: yes, as a design constraint.** This is why the recording protocol for our
own data must randomise presentation order within each session.

---

## Proposing a new resource

Check it against the four criteria at the top. The two that eliminate the most
candidates are **real repetitions per class** and **posterior coverage** — both are
easy to check from a dataset description and both are hard blockers.

If a candidate clears all four, it is worth a serious look even at the cost of a
Stage 1 rebuild.
