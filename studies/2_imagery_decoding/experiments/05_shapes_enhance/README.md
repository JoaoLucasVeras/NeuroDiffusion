# 05 · Shapes enhance: can a stronger decoder lift it?

**Question.** With the shapes signal confirmed, do stronger methods raise it to a useful
level? The targets were set **before** running: 45% three-way and 65% two-way, Bonferroni-corrected.

**Script.** `gao_enhance.py` compares:
- alpha power (the reference);
- multiband power;
- Riemannian tangent space with per-day re-centring;
- per-second multiband;
- both days pooled, with a time-ordered test block;
- every pair of shapes;
- seeing → imagining transfer slice by slice, with a max-null.

Job script `s17_gao_enhance.sh`, job 92390 (sub-21 excluded).

**Result.** Every method misses its target by a wide margin:
- best three-way 37.7% (pooled, p .006), against the 45% target;
- best two-way 52.0%, against 65%;
- nothing beats plain alpha power (35.0%).

By the rule set beforehand, effort moves from this dataset to new recordings.
