# Pinboard

Ideas that are worth doing but not now. Nothing here is rejected — everything on this
page is parked, with the reason and the specific thing that would unpark it.

Anything genuinely dead belongs in `RESOURCES.md` with a verdict instead. Anything
active belongs in `IMPROVEMENT_PLAN.md` or the queue in the documentation.

**Format:** what it is, why it is parked, what would unpark it, and when it came up.

---

## 1. Commercialising this, with users collecting their own data

*Raised 2026-09-25, Joao.*

**The idea.** Ship a product where the user records their own imagery data at home and it
trains their own model. Asked specifically whether users could annotate their own data,
and whether neural-data sensitivity kills the idea.

**What we worked out.**

- **Labelling is nearly free.** The label comes from the cue, not from the user. Show a
  picture, record while they imagine it, and the trial is labelled by construction. No
  annotation step exists. This is the standard BCI calibration session.
- **Sensitivity is a cost, not a wall.** Neural data is now specifically regulated
  (Colorado 2024 biological/neural data, California SB 1223 adding neural data to CCPA
  sensitive info, Montana 2025, Chile constitutionally, UNESCO neurotech guidance). Under
  GDPR it would almost certainly be special-category. All of that is workable with opt-in
  consent, purpose limitation, real deletion, and no secondary use. Needs actual counsel;
  this area moves fast.
- **The per-subject architecture is accidentally the privacy-friendly one.** Because
  cross-subject transfer does not work (measured, see the cross-subject test), you build
  per-user models. That means no pooled raw neural data, "delete my data" is satisfiable
  by deleting one model rather than trying to unlearn from shared weights, and training
  can plausibly run on-device.

**Why it is parked.** The science is not there. 3x chance on 33 classes with 33 events per
person is not a product, and shipping it would mean selling something that does not work.
Three hard blockers sit above the business question:

1. **Hardware.** Imagery lives in posterior, parieto-occipital electrodes. Consumer
   headsets are frontal-weighted because dry electrodes through hair is the unsolved
   engineering problem. The Muse has no occipital coverage — the exact defect that got
   MindBigData rejected.
2. **Calibration burden.** At ~40 repetitions per class per person, ten classes is roughly
   1.5 to 2 hours across two sessions. Consumer onboarding tolerance is about ten minutes.
3. **Session-to-session drift.** Every time the headset goes on, electrode positions
   change. Whether a model survives a new session is unknown for visual imagery, and it
   is the difference between "quick recalibration" and "unusable".

**What would unpark it.** In order:

- Per-user calibration cost dropping from hours to minutes, which depends on whether
  pretrained *representations* transfer across people even though trained *classifiers*
  do not (see pin 2).
- A headset with real occipital contact that a person will actually wear.
- Evidence that a model survives a second session on the same person. The Gao et al.
  dataset can answer this without any hardware.

**The nearer-term version.** Not self-service consumer, but a **consented, compensated
participant programme** with proper hardware and a protocol built for depth over breadth.
Cleaner ethics, cleaner regulation, and it unblocks the research rather than depending on
it. This is the shape to keep in mind when designing our own recording protocol.

**Two liabilities to settle before any launch.** EEG can incidentally reveal health
information such as epileptiform activity or sleep pathology, which raises a
duty-to-inform question. And claiming any health or wellness benefit potentially makes
this a regulated medical device.

---

## 2. Cross-subject pretraining, as opposed to cross-subject transfer

*Raised 2026-09-25, following the cross-subject test.*

**The idea.** Pretrain the encoder on everyone's data, then fine-tune a small head per
person. Different mechanism from what we just tested and killed.

**Why it is parked.** We measured that a trained *classifier* does not transfer between
people: all pairs sat inside the permutation null. But that says nothing about whether
*representations* transfer. Our Stage 1 masked autoencoder already works this way, so the
machinery partly exists. It is parked rather than active because with 33 events per person
we cannot tell a real fine-tuning gain from noise.

**What would unpark it.** The Gao et al. dataset, where 22 people with 40 repetitions
each makes a pretrain-then-fine-tune comparison actually measurable. This is the highest
value item on this page, because it is the lever behind pin 1's calibration cost.

---

## 3. ICA-based artifact removal

*Raised 2026-09-22, Aditya's reading batch.*

**The idea.** Use ICA to strip eye blinks and movement artifacts before decoding.

**Why it is parked.** Condition D of the classical baseline tested trial rejection on our
own data and found it **neutral** — sometimes up, sometimes down, no consistent
direction. That settles a genuine disagreement in the literature (Kessler et al. say
cleaning hurts, Hajhassani et al. say it helps in low-signal regimes) for our data
specifically. Not worth effort now.

**What would unpark it.** Our own recordings. Then we control the noise, we know what
artifacts our setup produces, and the calculation changes. Aditya's papers are the right
starting point when that day comes.

---

## 4. EEG foundation models

*Raised 2026-09-22, Joao.*

**The idea.** Use a large pretrained EEG model (CBraMod and similar) as the encoder
instead of training our own.

**Why it is parked.** They are pretrained on clinical EEG — sleep staging, seizure
detection, abnormality screening — rather than visual tasks, and at different electrode
counts and montages. Li et al. (2024) also found large pretrained encoders do not
automatically win on small EEG datasets, which is precisely our regime.

**What would unpark it.** A foundation model pretrained on visual-task EEG, or evidence
that clinical pretraining transfers to visual decoding. Worth re-checking periodically;
this field moves quickly.

---

## 5. Perception-data pretraining via THINGS-EEG

*Raised 2026-09-22, Joao.*

**The idea.** Pretrain the encoder on a large, well-built perception dataset
(THINGS-EEG / OpenNeuro ds003825), then adapt to imagery.

**Why it is parked.** Related to pin 2 and blocked on the same uncertainty. Also, our own
Mix experiment is a warning: training subject 1 on imagination and perception together
made them *worse*, 6.1% against 26.3%, because 6,789 perception trials swamped 363
imagination trials. Naive mixing of perception and imagery data actively hurt us.

**What would unpark it.** A staged approach rather than mixing — pretrain on perception,
then fine-tune on imagery alone, with the ratio controlled rather than accidental.

---

## 6. Longer analysis windows

*Raised 2026-09-24, from condition C of the baseline.*

**The idea.** Use 2 to 4 second segments instead of 500 ms. At 10 Hz, half a second is
only five cycles of the rhythm we are measuring, and the imagery literature uses longer.

**Why it is parked.** Condition C tried it and was inconclusive, not because the idea is
wrong but because gluing windows into 2-second segments leaves 99 training and 33 test
samples. The measurement is too noisy to read.

**What would unpark it.** Any dataset with real repetitions. This is a genuinely open
question that our data simply cannot answer.

---

## 7. Reconstructing image detail rather than category

*Standing ambition, worth writing down explicitly.*

**The idea.** The stated goal is "turn imagination into an image". What the pipeline
actually does is extract a weak category hint and let a generator draw a generic member of
that category. Nearly all visual detail comes from the generator, not the brain.

**Why it is parked.** Physics and information content. Scalp EEG has poor spatial
resolution and poor signal-to-noise; the detail is not in the recording to be extracted.
fMRI work that reconstructs recognisable images uses a signal with orders of magnitude
more spatial information.

**What would unpark it.** Honestly, a different measurement modality. Worth keeping on
the page anyway, because it is the thing the project is ultimately about, and because
being clear that we are not doing it is what keeps the claims honest.
