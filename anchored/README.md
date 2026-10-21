# Anchored imagery

A subsection of NeuroDiffusion, started 20 October 2026.

## The idea

Asking someone to "imagine a dog" mostly measures **recall**: they replay a dog they have
seen. Real imagination does more. It takes something you can see, holds it in memory, and
**changes** it, or builds something new out of remembered pieces.

> imagination = **seeing** (an anchor) + **memory** (hold it) + **imagery** (change it)

Without an anchor, imagery drifts toward daydreaming. Without a change, it is only memory.

## Why it should be easier to read from EEG

The main project asks the brain to produce a whole picture from nothing, then tries to
read the whole picture. On the Gao data that reaches about 35% for three shapes, where
guessing gives 33%.

Anchored imagery flips this:

1. **Show** an image. Seeing gives a strong, stable signal.
2. Ask for **one specific change**: move left, move right, stretch, shrink.
3. **Decode only the change.** There are few options, and they are timed by the cue.
4. **Edit the image** on screen to match. Image editing is a solved problem for software.

Imagining something moving left or right also shifts attention to that side, and the
resulting alpha lateralization is one of the most reliable signals EEG can read.

## The ladder

| Step | Task | Decode | Status |
|---|---|---|---|
| 1. Recall | "Imagine a square" | which object | where the main project is |
| 2. **Anchored change** | see a square, imagine it move or stretch | which change | **this subsection starts here** |
| 3. Attribute building | longer legs, then a puffier tail, one at a time | each attribute in turn | if step 2 works |
| 4. Free construction | "a chihuahua with dreads and long legs" | the whole combination | long term |

The brain signal chooses one piece at a time, and the generator assembles them.

## What is here

| File | What it is |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | The experiment design: trials, timing, sessions, controls, analysis plan and the success criteria, set in advance. The core of an ethics (IRB) application |
| [`trials.py`](trials.py) | Builds each session's randomised trial list from the protocol's rules, and checks them |
| [`run_experiment.py`](run_experiment.py) | The stimulus program (PsychoPy): shows anchors and cues, sends event markers. Has a dry-run mode |
| [`simulate.py`](simulate.py) | Makes fake EEG that follows the protocol, with known effects, so the analysis can be tested before any recording |
| [`analyze.py`](analyze.py) | The planned analysis: decode the imagined change across days, check for eye movements, compare anchored with unanchored |
| [`edit_shape.py`](edit_shape.py) | Draws the anchor shape and applies a decoded change: the "watch it change" output |

## Status

- No hardware yet. Everything here runs on a laptop with simulated data.
- Waiting on: access to an EEG system with eye channels, and ethics approval.
- The main project's memory-replay tests are paused and will resume later.
