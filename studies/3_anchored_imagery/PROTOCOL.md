# Anchored imagery: experiment protocol

Draft 1, 20 October 2026. Written before any data exists. The analysis plan and success
criteria in sections 8 and 9 are fixed now, so that results cannot shape them.

## 1. Aim

Test whether EEG can tell which **change** a person imagines being applied to a shape
they are looking at, and whether an anchor helps compared with imagining from memory.

## 2. Hypotheses

- **H1.** Imagined **left vs right** movement of an on-screen shape can be decoded above
  chance on a day the decoder never saw, and is not explained by eye movements.
- **H2.** All four imagined changes (move left, move right, stretch, shrink) can be decoded
  above chance (25%) across days.
- **H3.** Changes are decoded better when the shape stays on screen (**anchored**) than
  when it disappears and must be imagined from memory (**unanchored**).

## 3. Participants

- 12–20 healthy adults, normal or corrected vision, no neurological history.
- Each completes the Vividness of Visual Imagery Questionnaire (VVIQ, scores 16–80). It is
  recorded, not used to exclude; scores of 16–32 (little or no imagery) are analysed
  separately.
- 3 sessions per person, on different days, within two weeks.

## 4. Equipment

- EEG with at least 32 channels, covering parietal and occipital sites: P3, Pz, P4, P7, P8,
  PO3, PO4, PO7, PO8, O1, Oz, O2.
- **Eye recording, required:** horizontal and vertical EOG (4 electrodes), or an eye
  tracker.
- Sampling rate 500 Hz or more. Event markers from the stimulus computer, timed to the
  millisecond (Lab Streaming Layer or a parallel port).
- Monitor at a fixed distance (about 60 cm), chin rest if available.

## 5. Stimuli

- **Anchors:** a square, a circle and a triangle, white outlines on a dark grey
  background, about 4° of visual angle, centred, with a central fixation dot.
- **Changes**, each with its own cue symbol:

| Change | Cue | What to imagine |
|---|---|---|
| move left | ← | the shape sliding left |
| move right | → | the shape sliding right |
| stretch | ⟷ | the shape growing larger |
| shrink | →← | the shape becoming smaller |

## 6. One trial (about 10 s)

| Time | Screen | Participant |
|---|---|---|
| 0.0–1.0 s | fixation dot | keeps eyes on the dot |
| 1.0–2.5 s | anchor shape appears | looks at it |
| 2.5–3.5 s | cue symbol above the shape | reads the instruction |
| 3.5–7.5 s | **anchored:** shape stays · **unanchored:** shape gone, dot only | imagines the change, eyes on the dot |
| 7.5–9.5 s | blank (jittered 1.5–2.5 s) | rests, may blink |

Eyes stay on the central dot throughout. Blinking is encouraged during the rest period.

## 7. Session structure (about 60 minutes of recording)

1. **Seeing localiser** (5 min): the actual change is animated on screen, 4 changes × 15
   repeats, randomised. It gives "what seeing the change looks like" for comparison.
2. **Main task:** 8 blocks of 40 trials = 320 trials per session.
   - Every block holds all 4 changes × both anchor conditions × 5 repeats, shuffled.
   - No more than 3 consecutive trials share a change, and none share both change and
     anchor condition.
   - Anchor shapes rotate evenly across changes.
   - Per session: 80 trials per change, 40 of them anchored and 40 unanchored.
3. **Catch trials:** after 10% of trials, chosen at random, the participant reports the
   imagined change with a key press. Below 80% correct on these flags a session.
4. Breaks between blocks; the VVIQ before session 1.

Across three sessions: 960 trials per person, 240 per change.

## 8. Analysis plan (fixed before data)

- **Window:** the 4 s imagery period (3.5–7.5 s), excluding the cue.
- **Primary feature:** alpha (8–13 Hz) log power at every channel, plus the posterior
  lateralization index (right minus left over PO7/PO8, P7/P8, O1/O2), standardised within
  each session without labels.
- **Classifier:** logistic regression, fixed regularisation (l2 = 0.01). No tuning on test data.
- **Test:** train on two sessions, test on the third, all three rotations. Never random
  trial splits.
- **Null:** training labels shuffled, 1,000 times, whole pipeline repeated.
- **Eye check:** the same decoder on EOG signals alone. If eyes decode the change as well
  as the brain channels, the result is not accepted as imagery.
- **H3:** anchored and unanchored trials decoded separately, compared within person.
- **Group:** Fisher-combined p across people. Bonferroni across H1–H3.

## 9. Success criteria (set in advance)

| Gate | Passes if | Then |
|---|---|---|
| **Pilot** (2–3 people) | Left vs right ≥ 60% across days in at least 2 people, EOG-only at chance | Recruit the main sample |
| **H1** | Left vs right above chance across days, group p < 0.05/3, EOG-only not significant | Build the live demo for left/right |
| **H2** | Four changes ≥ 35% (chance 25%), group p < 0.05/3 | Extend the demo to four changes |
| **H3** | Anchored better than unanchored, group p < 0.05/3 | Evidence for the anchor idea itself |

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Eyes follow the imagined movement | Fixation dot, EOG or eye tracker, EOG-only decoding as a control |
| Attention, not imagery, drives left/right | Expected and useful for a control signal. Stretch vs shrink tests imagery without a side |
| Fatigue over 320 trials | Short blocks, breaks, catch trials to monitor engagement |
| Drift across the session | Cross-session testing, randomised order, per-session standardisation |
| Some people have little imagery | VVIQ recorded, analysed separately |

## 11. Ethics notes (for the IRB application)

- Non-invasive scalp EEG, minimal risk. Gel may need washing out.
- About 90 minutes per visit including setup, 3 visits.
- Written consent; participants can stop at any time.
- Data stored under codes, no names; consent covers sharing anonymised data.
