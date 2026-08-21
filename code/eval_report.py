"""
Produce the quantitative evaluation report: model scores next to the baselines that
say whether those scores mean anything.

Consumes the samples.npz written by eeg_ldm.py / gen_eval_eeg.py:
    gt      (n, h, w, 3)      uint8   ground-truth stimulus
    pred    (n, k, h, w, 3)   uint8   k generations per trial
    labels  (n,)              int     class index
    synsets (n_classes,)      str     synset id per class index

Usage:
    python code/eval_report.py --samples results/generation/<run>/samples.npz \\
        --output results/generation/<run>/report
"""
import argparse
import json
import os

import numpy as np

from clip_metrics import CLIPScorer, make_noise_baseline, shuffle_pairs, synset_to_name


def load_samples(path):
    d = np.load(path, allow_pickle=True)
    gt = d["gt"]
    pred = d["pred"]
    labels = d["labels"]
    synsets = [str(s) for s in d["synsets"]]
    if pred.ndim == 4:            # single generation per trial
        pred = pred[:, None]
    return gt, pred, labels, synsets


def summarise(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
        "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "n": int(n),
    }


def main():
    p = argparse.ArgumentParser(description="EEG-to-image evaluation report")
    p.add_argument("--samples", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--n_way", type=int, nargs="+", default=[2, 10, 40])
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--device", default=None)
    p.add_argument("--clip_model", default=None,
                   help="override the CLIP checkpoint (default: openai/clip-vit-large-patch14)")
    args = p.parse_args()

    gt, pred, labels, synsets = load_samples(args.samples)
    n, k = pred.shape[0], pred.shape[1]
    n_classes = len(synsets)
    print("Loaded %d trials, %d generation(s) each, %d classes" % (n, k, n_classes))

    scorer = CLIPScorer(device=args.device,
                        **({"model_name": args.clip_model} if args.clip_model else {}))
    results = {
        "n_trials": n,
        "n_generations_per_trial": k,
        "n_classes": n_classes,
        "chance_accuracy": 1.0 / n_classes,
        "samples_file": os.path.abspath(args.samples),
    }

    # ---- model: average the k generations per trial ----
    print("\nScoring model generations ...")
    sims, top1, probs_acc = [], [], []
    for j in range(k):
        pj = pred[:, j]
        sims.append(scorer.image_similarity(pj, gt))
        acc, probs = scorer.topk_accuracy(pj, labels, synsets, k=1)
        top1.append(acc)
        probs_acc.append(probs)
    model_sim = np.mean(sims, axis=0)
    mean_probs = np.mean(probs_acc, axis=0)

    results["model"] = {
        "clip_image_similarity": summarise(model_sim),
        "clip_zeroshot_top1": float(np.mean(top1)),
        "clip_zeroshot_top5": float(np.mean([
            int(l in np.argsort(-mean_probs[i])[:5]) for i, l in enumerate(labels)
        ])),
    }
    for w in args.n_way:
        if w <= n_classes:
            results["model"]["clip_%d_way" % w] = scorer.n_way_accuracy(
                mean_probs, labels, n_way=w, seed=args.seed)

    # ---- baseline: random noise ----
    print("Scoring noise baseline ...")
    noise = make_noise_baseline(n, gt.shape[1], gt.shape[2], seed=args.seed)
    noise_sim = scorer.image_similarity(noise, gt)
    noise_acc, noise_probs = scorer.topk_accuracy(noise, labels, synsets, k=1)
    results["baseline_noise"] = {
        "clip_image_similarity": summarise(noise_sim),
        "clip_zeroshot_top1": noise_acc,
    }
    for w in args.n_way:
        if w <= n_classes:
            results["baseline_noise"]["clip_%d_way" % w] = scorer.n_way_accuracy(
                noise_probs, labels, n_way=w, seed=args.seed)

    # ---- baseline: shuffled pairing (the one that matters) ----
    print("Scoring shuffled-pairing baseline ...")
    shuffled_gt, _ = shuffle_pairs(gt, labels, seed=args.seed)
    shuf_sim = scorer.image_similarity(pred[:, 0], shuffled_gt)
    results["baseline_shuffled"] = {
        "clip_image_similarity": summarise(shuf_sim),
        "note": "model generations scored against mismatched ground truth; "
                "isolates EEG-image correspondence from raw image quality",
    }

    # ---- headline: does the model beat the controls? ----
    m = results["model"]["clip_image_similarity"]
    s = results["baseline_shuffled"]["clip_image_similarity"]
    pooled_sem = np.sqrt(m["sem"] ** 2 + s["sem"] ** 2)
    delta = m["mean"] - s["mean"]
    results["headline"] = {
        "clip_similarity_delta_vs_shuffled": float(delta),
        "delta_in_sems": float(delta / pooled_sem) if pooled_sem > 0 else 0.0,
        "top1_over_chance": results["model"]["clip_zeroshot_top1"] - results["chance_accuracy"],
        "verdict": ("EEG-conditioned signal present"
                    if pooled_sem > 0 and delta > 2 * pooled_sem
                    else "NOT distinguishable from chance pairing"),
    }

    out = args.output or os.path.splitext(args.samples)[0] + "_report"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".json", "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("# EEG-to-Image Evaluation Report\n")
    lines.append("- Trials: **%d**, generations/trial: **%d**, classes: **%d**"
                 % (n, k, n_classes))
    lines.append("- Chance accuracy: **%.4f**\n" % results["chance_accuracy"])
    lines.append("## CLIP image similarity (higher is better)\n")
    lines.append("| Condition | Mean | SEM |")
    lines.append("|---|---|---|")
    for key, name in [("model", "Model"), ("baseline_shuffled", "Shuffled pairing"),
                      ("baseline_noise", "Random noise")]:
        d = results[key]["clip_image_similarity"]
        lines.append("| %s | %.4f | %.4f |" % (name, d["mean"], d["sem"]))
    lines.append("\n## Zero-shot classification of generated images\n")
    lines.append("| Metric | Model | Noise | Chance |")
    lines.append("|---|---|---|---|")
    lines.append("| top-1 | %.4f | %.4f | %.4f |" % (
        results["model"]["clip_zeroshot_top1"],
        results["baseline_noise"]["clip_zeroshot_top1"],
        results["chance_accuracy"]))
    for w in args.n_way:
        kk = "clip_%d_way" % w
        if kk in results["model"]:
            lines.append("| %d-way | %.4f | %.4f | %.4f |" % (
                w, results["model"][kk], results["baseline_noise"][kk], 1.0 / w))
    lines.append("\n## Verdict\n")
    lines.append("CLIP similarity exceeds the shuffled-pairing control by "
                 "**%.4f** (%.1f SEM).\n" % (delta, results["headline"]["delta_in_sems"]))
    lines.append("**%s**\n" % results["headline"]["verdict"])
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print("\nWrote %s.json and %s.md" % (out, out))


if __name__ == "__main__":
    main()
