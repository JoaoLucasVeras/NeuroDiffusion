"""
Build one session's trial list for the anchored-imagery experiment (PROTOCOL.md section 7).

Main task: 8 blocks x 40 trials. Each block holds every change x anchor condition five
times, shuffled so that no more than 3 trials in a row share a change and no two in a row
share both change and anchor condition. Anchor shapes rotate evenly within each change.
exactly 10% of trials, chosen at random, are followed by a catch question. Rest periods are
jittered between 1.5 and 2.5 s. A seeing localiser (4 changes x 15, shuffled) comes first.

Usage
-----
    python trials.py --participant 1 --session 1 --out sub-01_ses-1_trials.csv
"""
import argparse
import csv
import itertools

import numpy as np

CHANGES = ["move_left", "move_right", "stretch", "shrink"]
ANCHOR = ["anchored", "unanchored"]
SHAPES = ["square", "circle", "triangle"]
BLOCKS, REPEATS = 8, 5
CATCH_RATE = 0.10
LOCALISER_REPEATS = 15
MAX_RUN = 3


def ok_order(seq):
    run = 1
    for prev, cur in zip(seq, seq[1:]):
        if prev == cur:
            return False                        # same change and same anchor condition
        run = run + 1 if prev[0] == cur[0] else 1
        if run > MAX_RUN:
            return False
    return True


def block_order(rng, tries=10000):
    cells = [c for c in itertools.product(CHANGES, ANCHOR) for _ in range(REPEATS)]
    for _ in range(tries):
        seq = [cells[i] for i in rng.permutation(len(cells))]
        if ok_order(seq):
            return seq
    raise RuntimeError("no valid order found")


def session(participant, session_no):
    rng = np.random.default_rng(1000 * participant + session_no)
    shape_cycle = {c: 0 for c in CHANGES}
    shape_start = {c: int(rng.integers(len(SHAPES))) for c in CHANGES}
    rows = []
    loc = [c for c in CHANGES for _ in range(LOCALISER_REPEATS)]
    for k, ch in enumerate(rng.permutation(loc)):
        rows.append({"part": "localiser", "block": 0, "trial": k + 1, "change": ch, "anchor": "seen",
                     "shape": SHAPES[k % len(SHAPES)], "catch": 0, "rest": round(float(rng.uniform(1.5, 2.5)), 3)})
    n = 0
    for b in range(1, BLOCKS + 1):
        for ch, an in block_order(rng):
            n += 1
            shape = SHAPES[(shape_start[ch] + shape_cycle[ch]) % len(SHAPES)]
            shape_cycle[ch] += 1
            rows.append({"part": "main", "block": b, "trial": n, "change": ch, "anchor": an, "shape": shape,
                         "catch": 0, "rest": round(float(rng.uniform(1.5, 2.5)), 3)})
    main = [r for r in rows if r["part"] == "main"]
    for i in rng.choice(len(main), int(round(CATCH_RATE * len(main))), replace=False):
        main[i]["catch"] = 1
    return rows


def check(rows):
    main = [r for r in rows if r["part"] == "main"]
    assert len(main) == BLOCKS * len(CHANGES) * len(ANCHOR) * REPEATS
    for b in range(1, BLOCKS + 1):
        blk = [(r["change"], r["anchor"]) for r in main if r["block"] == b]
        assert ok_order(blk), "order rule broken in block %d" % b
        for cell in itertools.product(CHANGES, ANCHOR):
            assert blk.count(cell) == REPEATS
    for ch in CHANGES:
        counts = [sum(1 for r in main if r["change"] == ch and r["shape"] == s) for s in SHAPES]
        assert max(counts) - min(counts) <= 1, "shapes unbalanced for %s: %s" % (ch, counts)
    loc = [r["change"] for r in rows if r["part"] == "localiser"]
    assert all(loc.count(c) == LOCALISER_REPEATS for c in CHANGES)
    assert sum(r["catch"] for r in main) == int(round(CATCH_RATE * len(main)))
    return {"main_trials": len(main), "catch": sum(r["catch"] for r in main),
            "per_change": {c: sum(1 for r in main if r["change"] == c) for c in CHANGES}}


def main():
    ap = argparse.ArgumentParser(description="Build an anchored-imagery session")
    ap.add_argument("--participant", type=int, required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = session(args.participant, args.session)
    print(check(rows))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
