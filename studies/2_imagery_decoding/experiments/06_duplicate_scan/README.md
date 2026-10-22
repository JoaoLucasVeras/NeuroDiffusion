# 06 · Duplicate scan: are any recordings copies?

**Question.** One person scored 100% on imagined shapes, which is not believable. Are any of
the published recordings duplicates of each other?

**Script.** `gao_dup_scan.py` takes an MD5 of the EEG data section (after the file header) and of
every events file, in all 22 published zips. Job script `s18_gao_dup_scan.sh`, job 93110. It needs the
raw zips in `datasets/gao_raw/` (from `cluster/fetch_gao.sh`).

**Result.** Two sets of identical recordings:

| Person | Identical files |
|---|---|
| sub-21 | day 1 shapes = day 2 shapes = day 2 animals |
| sub-20 | day 1 objects = day 2 objects |

These are excluded from every later analysis (the `EXCLUDE` set in each script). Experiment
02's corrected numbers leave them out.
