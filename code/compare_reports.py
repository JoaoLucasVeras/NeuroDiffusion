"""
Put two or more evaluation runs side by side in one table.

The intended use is the visual-vs-imagination comparison: the visual run is a sanity
ceiling (2000 distinct stimuli, stimulus-disjoint split, a well-posed reconstruction
problem) and the imagination run is the result you actually care about. Reading them
together is what makes the imagination number interpretable -- on its own, a modest
40-way accuracy is hard to place.

Usage:
    python code/compare_reports.py \\
        --reports results/generation/<visual_run>/report.json \\
                  results/generation/<imag_run>/report.json \\
        --names visual imagination \\
        --output results/comparison
"""
import argparse
import json
import os


ROWS = [
    ("CLIP similarity (model)", lambda r: r["model"]["clip_image_similarity"]["mean"], "%.4f"),
    ("CLIP similarity (shuffled ctrl)", lambda r: r["baseline_shuffled"]["clip_image_similarity"]["mean"], "%.4f"),
    ("CLIP similarity (noise ctrl)", lambda r: r["baseline_noise"]["clip_image_similarity"]["mean"], "%.4f"),
    ("Delta vs shuffled (SEMs)", lambda r: r["headline"]["delta_in_sems"], "%.1f"),
    ("Zero-shot top-1", lambda r: r["model"]["clip_zeroshot_top1"], "%.4f"),
    ("Zero-shot top-5", lambda r: r["model"]["clip_zeroshot_top5"], "%.4f"),
    ("2-way", lambda r: r["model"].get("clip_2_way"), "%.4f"),
    ("10-way", lambda r: r["model"].get("clip_10_way"), "%.4f"),
    ("40-way", lambda r: r["model"].get("clip_40_way"), "%.4f"),
    ("Chance", lambda r: r["chance_accuracy"], "%.4f"),
    ("Test trials", lambda r: r["n_trials"], "%d"),
]


def fmt(fn, rep, spec):
    try:
        v = fn(rep)
    except (KeyError, TypeError):
        return "-"
    if v is None:
        return "-"
    return spec % v


def main():
    p = argparse.ArgumentParser(description="Side-by-side comparison of evaluation reports")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--output", default="comparison")
    args = p.parse_args()

    names = args.names or [os.path.basename(os.path.dirname(r)) or "run%d" % i
                           for i, r in enumerate(args.reports)]
    if len(names) != len(args.reports):
        raise SystemExit("--names must have one entry per --reports entry")

    reps = []
    for path in args.reports:
        with open(path) as f:
            reps.append(json.load(f))

    lines = ["# Evaluation comparison\n"]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for label, fn, spec in ROWS:
        lines.append("| %s | %s |" % (label, " | ".join(fmt(fn, r, spec) for r in reps)))

    lines.append("\n## Verdicts\n")
    for name, r in zip(names, reps):
        lines.append("- **%s**: %s" % (name, r["headline"]["verdict"]))

    lines.append("\n## Reading this table\n")
    lines.append("- The *shuffled* control holds image quality fixed and removes only the "
                 "EEG-stimulus correspondence. Beating it is the minimum bar for claiming "
                 "the model uses the brain signal at all.")
    lines.append("- Where stimuli repeat across a split (the imagination set has one image "
                 "per class), only the class-level rows are meaningful; CLIP similarity "
                 "there is bounded by how well the model reproduces one fixed image.")
    lines.append("- `Delta vs shuffled (SEMs)` above ~2 is the threshold used for the verdict.")

    md = "\n".join(lines)
    with open(args.output + ".md", "w") as f:
        f.write(md)
    with open(args.output + ".json", "w") as f:
        json.dump({n: r for n, r in zip(names, reps)}, f, indent=2)
    print(md)
    print("\nWrote %s.md and %s.json" % (args.output, args.output))


if __name__ == "__main__":
    main()
