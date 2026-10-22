# 04 · Shapes eye check: brain or eyes?

**Question.** Imagining a circle vs a triangle could make the eyes trace the shape. Is the
shapes signal from experiment 02 really the brain?

**Script.** `gao_eyecheck.py` decodes from:
- each head region;
- each second of the imagery window;
- the eye signals alone (frontal channels below 4 Hz, and F7 − F8).

It runs on our data and on the authors' ICA-cleaned data (eye components removed). Job script `s15_gao_eyecheck.sh`, job 90544.

**Result.** Not the eyes:
- Eye signals alone decode nothing (p .47).
- The shapes signal survives removing the eye components (p .002).
- It is spread across the head rather than confined to visual cortex, at about 1.05–1.10× chance.
