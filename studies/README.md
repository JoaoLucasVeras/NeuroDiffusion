# Studies

NeuroDiffusion is one project with one goal: turn imagination into images. The work
toward it is split into **studies** and **experiments**.

| Term | Meaning | Example |
|---|---|---|
| **Study** | One big research question with its own approach. Contains many experiments. | "Can we decode an imagined *change* to a shape we just saw?" |
| **Experiment** | One test with one result, inside a study | "Is the shapes signal really eye movement? No." |

**Where does a new idea go?** A new test of an existing question is a new experiment inside
that study. A new question, or a new way of collecting data, is a new study.

## The studies so far

| Study | Question | Status |
|---|---|---|
| [1_dreamdiffusion](1_dreamdiffusion/) | Can the DreamDiffusion model, built for *seeing*, generate images from *imagined* EEG? | Finished. Generation worked on seeing, imagination stayed at chance. Kept unchanged |
| [2_imagery_decoding](2_imagery_decoding/) | Can imagined pictures be read from EEG at all, when tested honestly? | Main study, 6 experiments. Only imagined *shapes* decode, at about 1.05 to 1.1 times chance |
| [3_anchored_imagery](3_anchored_imagery/) | Can we decode an imagined *change* to a shape the person just saw? | Designed and simulated. Waiting on EEG hardware and ethics approval |

## Starting a new study or experiment

1. Branch from main: `git switch -c <name>/<topic> origin/main`.
2. **New experiment:** copy the closest folder in `2_imagery_decoding/experiments/`, give it
   the next number, and change one thing. **New study:** make `studies/<n>_<name>/` with a
   README that states the question, the method, and how you will know it worked.
3. Reuse the toolbox in [`shared/`](../shared/) by importing it, never by copying it.
4. Every experiment folder has a README with the question, how to run it, the job numbers,
   and the result with its chance level. Fill in the result when it comes back, good or bad.

The rules every result has to follow are in [docs/CODEBASE.md](../docs/CODEBASE.md#6-the-rules-every-result-follows).
