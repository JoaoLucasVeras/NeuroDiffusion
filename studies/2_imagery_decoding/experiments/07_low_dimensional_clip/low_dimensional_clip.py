"""Experiment 07: training-only visual axes and unseen-stimulus EEG retrieval.

No model downloads, GPU use, or cluster submission. See README.md for the protocol.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "shared" / "paths.py").exists()) / "shared"))
import paths

import numpy as np
import torch
import scipy
from classical_baseline import prepare, feat_bandpower

KIND = "vtc_visual_axes_ridge_v1"
DIMS = (2, 4, 8, 16)
L2_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_trusted(path):
    """torch.load for trusted repository files, on PyTorch with or without weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location="cpu")


def load_events(dataset_path, cache_path, allow_missing_embeddings=False):
    """Read trusted repository .pth and {stems, emb} cache; never embed/download.

    Restrict to the common stimulus set across participants so every comparison
    and synchronized permutation operates on the same objects. Report exclusions.
    """
    for p in (dataset_path, cache_path):
        if not Path(p).is_file():
            raise ValueError("Missing local asset: %s. No download was attempted." % p)
    data = load_trusted(dataset_path)
    cache = load_trusted(cache_path)
    stems = cache.get("stems", [])
    if not stems or len(set(stems)) != len(stems):
        raise ValueError("Embedding cache needs unique nonempty stems and emb.")
    targets = as_numpy(cache["emb"]).astype(np.float64)
    if targets.shape != (len(stems), 768) or not np.isfinite(targets).all():
        raise ValueError("Expected finite cached CLIP embeddings of shape (stimuli, 768).")
    if np.any(np.linalg.norm(targets, axis=1) <= 1e-12):
        raise ValueError("Zero CLIP embedding in cache.")
    embedding = dict(zip(stems, normalize(targets)))
    groups, seen, shape = {}, set(), None
    entries = data.get("dataset", [])
    if not entries:
        raise ValueError("Dataset is empty.")
    for e in entries:
        key = (int(e["subject"]), str(e["image"]))
        window = int(e["window"])
        if window < 0 or key + (window,) in seen:
            raise ValueError("Duplicate or negative window identifier: %s" % (key + (window,),))
        seen.add(key + (window,))
        x = as_numpy(e["eeg"]).astype(np.float64)
        if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
            raise ValueError("Invalid EEG window: %s" % (key,))
        if shape is None:
            shape = x.shape
        if x.shape != shape:
            raise ValueError("Inconsistent recording shapes: %s versus %s" % (x.shape, shape))
        groups.setdefault(key, []).append({"eeg": x, "window": window,
                                          "label": int(e["label"]),
                                          "subject": key[0], "image": key[1]})
    missing = sorted({k[1] for k in groups} - set(embedding))
    if missing and not allow_missing_embeddings:
        raise ValueError("Missing embeddings for %d stimuli: %s. Supply a complete cache or "
                         "explicitly use --allow_missing_embeddings to record exclusions."
                         % (len(missing), missing))
    subjects = sorted({k[0] for k in groups})
    available = [{k[1] for k in groups if k[0] == s} & set(embedding) for s in subjects]
    common = sorted(set.intersection(*available))
    if len(common) < 15:
        raise ValueError("Need at least 15 common cached stimuli for nested 5/4 folds; found %d. "
                         "Missing embeddings: %s" % (len(common), missing))
    features, counts = {}, {}
    for s in subjects:
        rows, ns = [], []
        for stem in common:
            group = sorted(groups[s, stem], key=lambda t: t["window"])
            windows = [t["window"] for t in group]
            if windows != list(range(20)):
                raise ValueError("Expected all 20 Shimizu imagery windows (0..19): %s" % ((s, stem),))
            if len({t["label"] for t in group}) != 1:
                raise ValueError("Inconsistent labels within imagery event.")
            filtered, _, _ = prepare(group, (8.0, 13.0), False)
            rows.append(feat_bandpower(filtered).mean(axis=0))
            ns.append(len(group))
        features[s] = np.stack(rows)
        counts[s] = ns
    audit = {"subjects": subjects, "stimuli": common, "window_counts": counts,
             "window_shape": list(shape), "uncached_stimuli": missing,
             "excluded_by_subject": {s: sorted({k[1] for k in groups if k[0] == s} - set(common))
                                     for s in subjects},
             "sampling_rate_hz": 250, "band_hz": [8, 13]}
    return features, np.stack([embedding[s] for s in common]), audit


def partitions(indices, n_folds, seed):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2 * n_folds:
        raise ValueError("Each retrieval fold needs at least two candidate stimuli.")
    shuffled = np.random.default_rng(seed).permutation(indices)
    return [np.sort(x) for x in np.array_split(shuffled, n_folds)]


def make_manifest(stems, seed=2022):
    indices = np.arange(len(stems))
    manifest = {"seed": seed, "stimuli": list(stems), "outer": []}
    for fold, te in enumerate(partitions(indices, 5, seed)):
        tr = np.setdiff1d(indices, te)
        inner = []
        for va in partitions(tr, 4, seed + 100 + fold):
            it = np.setdiff1d(tr, va)
            inner.append({"train": it.tolist(), "validation": va.tolist()})
        manifest["outer"].append({"fold": fold, "train": tr.tolist(),
                                  "test": te.tolist(), "inner": inner})
    return manifest


def fit_ridge(x, y, l2, dim=None):
    """Minimize mean squared error + l2 * ||W||^2, intercept unpenalized.

    All transforms are fitted only on the supplied training rows. Reduced and
    full arms share the same centered ridge estimator and target mean.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    ym = y.mean(0)
    centered = y - ym
    basis = None
    rank = int(np.linalg.matrix_rank(centered))
    if dim is not None:
        if dim > rank:
            raise ValueError("PCA dimension %d exceeds training numerical rank %d" % (dim, rank))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:dim].T
        centered = centered @ basis
    # Dual solve is small: one row per independent imagery event.
    dual = np.linalg.solve(z @ z.T + len(z) * l2 * np.eye(len(z)), centered)
    return {"feature_mean": mu, "feature_scale": sd, "target_mean": ym,
            "basis": basis, "weight": z.T @ dual, "l2": l2,
            "dimension": dim, "training_target_rank": rank}


def predict(model, x):
    out = ((x - model["feature_mean"]) / model["feature_scale"]) @ model["weight"]
    if model["basis"] is not None:
        out = out @ model["basis"].T
    return normalize(out + model["target_mean"])


def retrieval(pred, candidates):
    """One prediction per candidate in the same order; ties get fractional credit."""
    sims = normalize(pred) @ normalize(candidates).T
    n = len(sims)
    if n < 2 or sims.shape != (n, n):
        raise ValueError("Retrieval needs matching predictions and >=2 candidates.")
    correct = np.diag(sims)
    others = sims[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    ties = np.isclose(others, correct[:, None], rtol=0, atol=1e-12)
    wins = (others < correct[:, None]) & ~ties
    ahead = ((others > correct[:, None]) & ~ties).sum(1)
    ntie = ties.sum(1)
    rr = [np.mean(1.0 / np.arange(a + 1, a + t + 2)) for a, t in zip(ahead, ntie)]
    return {"pairwise": ((wins.sum(1) + 0.5 * ntie) / (n - 1)).tolist(),
            "top1": np.where(ahead == 0, 1.0 / (ntie + 1), 0.0).tolist(),
            "reciprocal_rank": rr,
            "cosine_margin": (correct - others.max(1)).tolist(),
            "top1_chance": [1.0 / n] * n}


def choose(x, targets, fold, reduced, assignment):
    """Tune on outer-training identities only, including under label permutation.

    assignment maps EEG event rows to target rows; it is a bijection restricted
    to outer training. Inner validation is scored against these assigned labels.
    No outer-test target is read here.
    """
    inner = []
    for f in fold["inner"]:
        tr, va = np.array(f["train"]), np.array(f["validation"])
        y = targets[assignment[tr]]
        centered = y - y.mean(0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        tol = singular[0] * max(centered.shape) * np.finfo(float).eps if len(singular) else 0
        rank = int(np.sum(singular > tol))
        inner.append((tr, va, vt.T, rank))
    dims = tuple(d for d in DIMS if all(d <= row[3] for row in inner)) if reduced else (None,)
    vals = {(dim, l2): [] for dim in dims for l2 in L2_GRID}
    for tr, va, basis, _ in inner:
        for l2 in L2_GRID:
            full = fit_ridge(x[tr], targets[assignment[tr]], l2)
            for dim in dims:
                model = full
                if dim is not None:
                    # Ridge is linear in its targets: projection commutes with fitting.
                    # Reuse one solve per l2 instead of solving once per PCA dimension.
                    model = dict(full, basis=basis[:, :dim], dimension=dim,
                                 weight=full["weight"] @ basis[:, :dim])
                vals[dim, l2].extend(retrieval(predict(model, x[va]), targets[assignment[va]])["pairwise"])
    scores = [{"dimension": dim, "l2": l2, "score": float(np.mean(v))}
              for (dim, l2), v in vals.items()]
    if not scores:
        raise ValueError("No PCA candidate is supported by all inner-training ranks.")
    best_score = max(s["score"] for s in scores)
    tied = [s for s in scores if abs(s["score"] - best_score) <= 1e-12]
    best = min(tied, key=lambda s: (s["dimension"] or 0, -s["l2"]))
    tr = np.array(fold["train"])
    model = fit_ridge(x[tr], targets[assignment[tr]], best["l2"], best["dimension"])
    return model, scores


def evaluate(features, targets, manifest, permutation=None, retain=True):
    """Evaluate a full nested procedure; permutations are synchronized by fold."""
    all_subjects = {}
    for subject, x in sorted(features.items()):
        metrics = {k: {} for k in ("full", "reduced", "oracle", "mean")}
        models = []
        for fold in manifest["outer"]:
            tr, te = np.array(fold["train"]), np.array(fold["test"])
            assignment = np.arange(len(targets))
            if permutation is not None:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [manifest["seed"], 9001, permutation, fold["fold"]]))
                assignment[tr] = rng.permutation(tr)
            fitted, tuning, predictions = {}, {}, {}
            for arm in ("full", "reduced"):
                fitted[arm], tuning[arm] = choose(x, targets, fold, arm == "reduced", assignment)
                predictions[arm] = predict(fitted[arm], x[te])
            reduced = fitted["reduced"]
            basis, mean = reduced["basis"], reduced["target_mean"]
            predictions["oracle"] = normalize((targets[te] - mean) @ basis @ basis.T + mean)
            predictions["mean"] = np.repeat(normalize(mean[None]), len(te), axis=0)
            for arm, pred in predictions.items():
                r = retrieval(pred, targets[te])
                for k, v in r.items():
                    metrics[arm].setdefault(k, []).extend(v)
            if retain:
                models.append({"fold": fold["fold"], "test_indices": te,
                               "training_target_indices": assignment[tr],
                               "models": fitted, "tuning": tuning, "predictions": predictions})
        summary = {arm: {k: float(np.mean(v)) for k, v in m.items()} for arm, m in metrics.items()}
        all_subjects[str(subject)] = {"metrics": summary,
                                     "delta": summary["reduced"]["pairwise"] - summary["full"]["pairwise"]}
        if retain:
            all_subjects[str(subject)]["folds"] = models
    delta = float(np.mean([s["delta"] for s in all_subjects.values()]))
    group = {arm: {metric: float(np.mean([s["metrics"][arm][metric] for s in all_subjects.values()]))
                   for metric in next(iter(all_subjects.values()))["metrics"][arm]}
             for arm in ("full", "reduced", "oracle", "mean")}
    return {"subjects": all_subjects, "group_metrics": group, "mean_delta": delta}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": h.hexdigest()}


def provenance():
    def git(*args):
        r = subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__,
            "source_hashes": [fingerprint(__file__),
                              fingerprint(paths.SHARED / "decoding" / "classical_baseline.py"),
                              fingerprint(paths.SHARED / "paths.py")]}


def decision_gate(observed, null, permutation_count):
    pvalue = float((1 + np.sum(np.array(null) >= observed["mean_delta"])) / (1 + len(null))) if null else None
    improvements = sum(s["delta"] > 0 for s in observed["subjects"].values())
    passed = bool(permutation_count >= 1000 and len(null) == permutation_count
                  and observed["mean_delta"] > 0 and pvalue <= 0.05 and improvements >= 3)
    return pvalue, improvements, passed


def above_chance_p(observed, null_reduced):
    """One-sided permutation p that the reduced arm's group pairwise accuracy beats chance.

    The paired-delta gate alone can pass when both arms sit at chance; this requires the
    reduced arm itself to beat the same training-label null. None without permutations.
    """
    if not null_reduced:
        return None
    obs = observed["group_metrics"]["reduced"]["pairwise"]
    return float((1 + np.sum(np.array(null_reduced) >= obs)) / (1 + len(null_reduced)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=paths.DATASETS / "imagination_5_95_std.pth")
    ap.add_argument("--cache", type=Path, required=True, help="Existing {stems, emb} CLIP cache, 768 dimensions")
    ap.add_argument("--output", type=Path, required=True, help="New run directory; existing directories are refused")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--permute", type=int, default=1000)
    ap.add_argument("--allow_missing_embeddings", action="store_true",
                    help="Explicitly exclude and log stimuli absent from the local cache")
    args = ap.parse_args()
    if args.permute < 0 or args.seed < 0:
        ap.error("seed and permutation count must be nonnegative")
    try:
        features, targets, audit = load_events(args.dataset, args.cache, args.allow_missing_embeddings)
        manifest = make_manifest(audit["stimuli"], args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))
    meta = {"kind": KIND, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "provenance": provenance(), "inputs": [fingerprint(args.dataset), fingerprint(args.cache)],
            "audit": audit, "dimensions": DIMS, "l2_grid": L2_GRID}
    def save(name, value):
        (args.output / name).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Persist every identity and split before the first fitted model.
    save("manifest.json", {**meta, "splits": manifest})
    observed = evaluate(features, targets, manifest)
    save("observed.json", {"kind": KIND, **observed})
    null = []
    above_chance_null = []
    for p in range(args.permute):
        result = evaluate(features, targets, manifest, permutation=p, retain=False)
        null.append(result["mean_delta"])
        above_chance_null.append(float(np.mean([s["metrics"]["reduced"]["pairwise"]
                                                for s in result["subjects"].values()])))
        if (p + 1) % 25 == 0:
            save("null_progress.json", {"completed": p + 1, "delta": null,
                                        "reduced_pairwise": above_chance_null})
            print("permutations %d/%d" % (p + 1, args.permute), flush=True)
    # Exploratory paired-label-null gate, not a population-level superiority test.
    pvalue, improvements, delta_gate = decision_gate(observed, null, args.permute)
    p_above = above_chance_p(observed, above_chance_null)
    gate = bool(delta_gate and p_above is not None and p_above <= 0.05)
    save("summary.json", {**meta, "subjects": {s: {k: v for k, v in r.items() if k != "folds"}
                                             for s, r in observed["subjects"].items()},
                          "group_metrics": observed["group_metrics"],
                          "mean_delta": observed["mean_delta"], "paired_training_label_null_p": pvalue,
                          "null_deltas": null, "null_reduced_pairwise": above_chance_null,
                          "participants_improved": improvements, "reduced_above_chance_p": p_above,
                          "delta_gate": delta_gate, "follow_up_gate": gate,
                          "limitations": ["Four participants; one imagery event per class per person.",
                                           "Permutation tests break EEG-target association; they do not establish population-level model superiority.",
                                           "One image per class: held-out image identity is also held-out class identity."]})
    print("Saved %s; mean pairwise delta %+.4f; permutation p %s; reduced above chance p %s; "
          "follow-up gate %s" % (args.output, observed["mean_delta"], pvalue, p_above, gate))


if __name__ == "__main__":
    main()
