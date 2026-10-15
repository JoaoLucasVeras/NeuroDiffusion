"""
Gao reproduction, stage 1: collect the authors' printed numbers from the run logs.

Reads every log of a run, takes each recording's "Average Accuracy" and "Overall Best
Accuracy" exactly as their script printed them, and sets real labels beside scrambled
ones. The paper's figure to compare against: animals, 76.6% session 1, 74.4% session 2
(Table 3), 75.8% in the text.

Usage
-----
    python gao_repro_summary.py --logs '../logs/s16_gaorepro_90600_*.out' --json out.json
"""
import argparse
import glob
import json
import re

import numpy as np
from scipy import stats

SUBJECT = re.compile(r"^Subject: (sub-\d+_ses-\d+_task-\w+?)_eeg_data\.npy")
FOLDER = re.compile(r"^data folder .*/gao_repro/(real|scrambled)/")
AVERAGE = re.compile(r"^Average Accuracy: ([\d.]+)%")
BEST = re.compile(r"^Overall Best Accuracy: ([\d.]+)%")


def parse(paths):
    rows = {}
    for path in sorted(paths):
        kind = rec = None
        for line in open(path, errors="replace"):
            line = line.strip()
            if FOLDER.match(line):
                kind = FOLDER.match(line).group(1)
            elif SUBJECT.match(line):
                rec = SUBJECT.match(line).group(1)
            elif AVERAGE.match(line) and kind and rec:
                rows.setdefault(rec, {})[kind + "_average"] = float(AVERAGE.match(line).group(1)) / 100
            elif BEST.match(line) and kind and rec:
                rows.setdefault(rec, {})[kind + "_best"] = float(BEST.match(line).group(1)) / 100
    return rows


def main():
    ap = argparse.ArgumentParser(description="Summarise the Gao reproduction")
    ap.add_argument("--logs", required=True, help="glob of the run's log files")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = parse(glob.glob(args.logs))
    keys = ("real_average", "scrambled_average", "real_best", "scrambled_best")
    done = {r: v for r, v in rows.items() if all(k in v for k in keys)}
    print("%d recordings complete, %d incomplete" % (len(done), len(rows) - len(done)))
    print("%-24s %8s %8s %8s %8s" % ("recording", "real", "scramb", "realBst", "scrBst"))
    for r in sorted(done):
        v = done[r]
        print("%-24s %8.3f %8.3f %8.3f %8.3f" % (r, *(v[k] for k in keys)))
    out = {"n": len(done), "recordings": done}
    for ses in ("ses-01", "ses-02", ""):
        sel = [v for r, v in done.items() if ses in r]
        if not sel:
            continue
        m = {k: float(np.mean([v[k] for v in sel])) for k in keys}
        diff = [v["real_average"] - v["scrambled_average"] for v in sel]
        p = float(stats.ttest_1samp(diff, 0.0).pvalue) if len(sel) > 1 else None
        out[ses or "all"] = dict(m, n=len(sel), p_real_vs_scrambled=p)
        print("%-8s n=%2d | average: real %.3f scrambled %.3f (paired p %s) | best: real %.3f scrambled %.3f"
              % (ses or "all", len(sel), m["real_average"], m["scrambled_average"],
                 "%.3g" % p if p is not None else "-", m["real_best"], m["scrambled_best"]))
    print("paper (Table 3): animals 0.766 session 1, 0.744 session 2; chance 0.333")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
