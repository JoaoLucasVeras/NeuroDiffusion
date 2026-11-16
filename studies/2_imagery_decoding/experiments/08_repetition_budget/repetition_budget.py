"""Offline probability pooling of independent Gao FVI events.

Use --validate-only to check inputs/save membership without extracting features
or fitting models. --run is an explicit opt-in. Load only trusted prepared .pt files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared/paths.py").exists())
sys.path.insert(0, str(ROOT / "shared"))
import paths
from classical_baseline import logreg
from gao_baseline import features

import numpy as np
import scipy
import torch

CLASSES = ("circle", "pentagram", "square")
COHORT = tuple(s for s in range(1, 23) if s not in (9, 10))
BUDGETS = (1, 2, 4, 8)
KIND = "gao_repetition_probability_pool_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    """Atomically replace generated progress; never used for source files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def groups_for(labels, latencies, budget):
    labels, latencies = np.asarray(labels), np.asarray(latencies)
    if budget not in BUDGETS or len(labels) != 120 or len(latencies) != 120:
        raise ValueError("Expected 120 events and a fixed repetition budget")
    if not np.isfinite(latencies).all() or len(np.unique(latencies)) != 120:
        raise ValueError("Duplicate or nonfinite event latencies")
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Expected exactly three shape labels")
    groups = []
    for label in range(3):
        members = np.flatnonzero(labels == label)
        if len(members) != 40:
            raise ValueError("Require exactly 40 events per shape; no dropping")
        members = members[np.argsort(latencies[members], kind="stable")]
        groups.extend(members.reshape(-1, budget).tolist())
    result = np.asarray(groups, dtype=int)
    if len(np.unique(result)) != 120:
        raise ValueError("Repeated/missing group members")
    return result


def pool_scores(probabilities, labels, groups):
    probabilities, labels, groups = np.asarray(probabilities), np.asarray(labels), np.asarray(groups)
    if probabilities.shape != (len(labels), 3) or groups.ndim != 2:
        raise ValueError("Invalid prediction/group shape")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Nonfinite or negative probabilities")
    if not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    if groups.size != len(labels) or not np.array_equal(np.sort(groups.ravel()), np.arange(len(labels))):
        raise ValueError("Groups must cover each test event exactly once")
    target = labels[groups[:, 0]]
    if not (labels[groups] == target[:, None]).all():
        raise ValueError("Mixed-shape group")
    pooled = probabilities[groups].mean(1)
    # np.argmax deterministically prefers circle, then pentagram, then square.
    return {"accuracy": float(np.mean(pooled.argmax(1) == target)),
            "decisions": len(groups), "predicted": pooled.argmax(1).tolist(),
            "targets": target.tolist(), "probabilities": pooled.tolist()}


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid EEG features")
    mean, spread = values.mean(0), values.std(0) + 1e-8
    return (values - mean) / spread, mean, spread


def predict(source, labels, target, seed):
    return logreg(source, labels, target, 3, .01, epochs=600, lr=.05, seed=seed)


def evaluate_subject(sessions, seed, permutation=None, estimator=predict, keep_predictions=False):
    """Each direction fits once; both arms/budgets reuse its predictions."""
    if set(sessions) != {1, 2}:
        raise ValueError("Two sessions required")
    directions = []
    for source_id, target_id in ((1, 2), (2, 1)):
        source, target = sessions[source_id], sessions[target_id]
        labels = source["y"].copy()
        if permutation is not None:
            rng = np.random.default_rng(np.random.SeedSequence([seed, source["subject"], source_id, permutation + 1]))
            labels = rng.permutation(labels)
        probabilities = estimator(source["X"], labels, target["X"], seed)
        scores = {str(k): pool_scores(probabilities, target["y"], target["groups"][k]) for k in BUDGETS}
        row = {"train_session": source_id, "test_session": target_id,
               "accuracy": {key: value["accuracy"] for key, value in scores.items()},
               "decisions": {key: value["decisions"] for key, value in scores.items()}}
        if keep_predictions:
            row.update(event_probabilities=np.asarray(probabilities).tolist(),
                       event_targets=target["y"].tolist(), event_ids=target["ids"], pooled=scores)
            row["constant_probability_control"] = {
                str(k): pool_scores(np.full((120, 3), 1 / 3), target["y"], target["groups"][k])["accuracy"]
                for k in BUDGETS}
        directions.append(row)
    accuracy = {str(k): float(np.mean([d["accuracy"][str(k)] for d in directions])) for k in BUDGETS}
    return {"subject": sessions[1]["subject"], "accuracy": accuracy,
            "delta": accuracy["4"] - accuracy["1"], "directions": directions}


def p_value(observed, null):
    return float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (len(null) + 1)) if len(null) else None


def bootstrap(values, seed, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def validate_results(observed, null, subjects, requested):
    if len(null) > requested:
        raise ValueError("Checkpoint has excess permutations")
    for rows in [observed, *null]:
        if sorted(r["subject"] for r in rows) != sorted(subjects):
            raise ValueError("Checkpoint cohort mismatch")
        for row in rows:
            accuracy = row["accuracy"]
            if set(accuracy) != {str(k) for k in BUDGETS}:
                raise ValueError("Checkpoint budget mismatch")
            if not all(np.isfinite(a) and 0 <= a <= 1 for a in accuracy.values()):
                raise ValueError("Invalid checkpoint accuracy")
            if not np.isclose(row["delta"], accuracy["4"] - accuracy["1"], atol=1e-12):
                raise ValueError("Inconsistent checkpoint difference")


def summarize(observed, null, seed, requested):
    ids = [r["subject"] for r in observed]
    accuracy = {str(k): float(np.mean([r["accuracy"][str(k)] for r in observed])) for k in BUDGETS}
    delta = float(np.mean([r["delta"] for r in observed]))
    null_delta = [float(np.mean([r["delta"] for r in iteration])) for iteration in null]
    null_four = [float(np.mean([r["accuracy"]["4"] for r in iteration])) for iteration in null]
    pd, pf = p_value(delta, null_delta), p_value(accuracy["4"], null_four)
    complete = len(null) == requested
    canonical = set(ids) == set(COHORT) and seed == 2022 and len(null) >= 1000 and complete
    statistical = bool(canonical and delta > 0 and pd <= .05 and pf <= .05)
    return {"accuracy": accuracy, "chance": 1 / 3, "delta": delta,
            "p_delta": pd, "p_four_above_chance": pf,
            "null_delta": null_delta, "null_four": null_four,
            "delta_ci95_participant_bootstrap": bootstrap([r["delta"] for r in observed], seed),
            "four_ci95_participant_bootstrap": bootstrap([r["accuracy"]["4"] for r in observed], seed),
            "participants": len(ids), "participants_improving": sum(r["delta"] > 0 for r in observed),
            "permutations_completed": len(null), "canonical_complete_run": canonical,
            "statistical_advance": statistical, "practical_45_percent_met": accuracy["4"] >= .45,
            "practical_advance": bool(statistical and accuracy["4"] >= .45),
            "secondary_budgets_descriptive_only": [2, 8]}


def load_inputs(data, subjects, compute_features):
    manifest, cohort, cache = [], {}, {}
    epoch_hashes = set()
    channel_order = None
    for subject in sorted(subjects):
        file = data / f"gao_sub-{subject:02d}.pt"
        source_hash = digest(file)
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if (payload.get("format") != "gao_fvi_v4_with_event_provenance_v1"
                or payload.get("provenance", {}).get("version") != 4
                or (payload["fs"], payload["tmin"], payload["tmax"]) != (250, -7., 5.)):
            raise ValueError(f"Unexpected prepared format/version/timing: {file.name}")
        if channel_order is not None and channel_order != payload["channels"]:
            raise ValueError("Inconsistent channel order")
        channel_order = payload["channels"]
        if len(channel_order) != 32:
            raise ValueError("Expected 32 EEG channels")
        trials = payload["dataset"]
        if len(trials) != 240 or {t["session"] for t in trials} != {1, 2}:
            raise ValueError("Require 240 events in sessions 1 and 2")
        for t in trials:
            if t["subject"] != subject or t["task"] != "FVI" or t["class"] not in CLASSES:
                raise ValueError("Subject/task/class mismatch")
            if payload["classes"][t["label"]] != t["class"]:
                raise ValueError("Stored label/class disagreement")
            if t["eeg"].shape != (32, 3000) or not torch.isfinite(t["eeg"]).all():
                raise ValueError("Invalid epoch shape or nonfinite data")
            if (t["source_fs"] != 1000 or t["source_latency_samples"] / 1000 != t["source_onset_seconds"]
                    or t["source_archive_md5"] != payload["provenance"]["archive_md5"]):
                raise ValueError("Inconsistent source timing/provenance")
            h = hashlib.sha256(t["eeg"].numpy().tobytes()).hexdigest()
            if h in epoch_hashes:
                raise ValueError("Exact duplicate EEG epoch")
            epoch_hashes.add(h)
        sessions, session_manifest = {}, []
        for session in (1, 2):
            rows = sorted((t for t in trials if t["session"] == session), key=lambda t: t["source_latency_samples"])
            ids = [[subject, session, "FVI", t["trial"]] for t in rows]
            if len(rows) != 120 or len({tuple(i) for i in ids}) != 120:
                raise ValueError("Missing/duplicate source event IDs")
            y = np.array([CLASSES.index(t["class"]) for t in rows])
            lat = np.array([t["source_latency_samples"] for t in rows])
            if np.any(np.diff(lat) < 4000):
                raise ValueError("Overlapping imagery intervals")
            groups = {k: groups_for(y, lat, k) for k in BUDGETS}
            spans = (lat[groups[4][:, -1]] - lat[groups[4][:, 0]]) / 1000 + 4
            session_manifest.append({"session": session, "event_ids": ids, "labels": y.tolist(),
                                     "latencies_samples_1000Hz": lat.tolist(),
                                     "source_files": [t["source_file"] for t in rows],
                                     "groups_event_indices": {str(k): g.tolist() for k, g in groups.items()},
                                     "four_event_span_seconds": spans.tolist()})
            if compute_features:
                raw = features(rows, (0., 4.), (8., 13.), 250, -7., kind="power")
                X, mean, spread = standardize(raw)
                sessions[session] = {"X": X, "y": y, "ids": ids, "groups": groups, "subject": subject}
                for label, value in (("raw", raw), ("standardized", X), ("mean", mean), ("spread", spread)):
                    cache[f"sub{subject:02d}_session{session}_{label}"] = value
        manifest.append({"subject": subject, "prepared_file": file.name, "prepared_sha256": source_hash,
                         "source_provenance": payload["provenance"], "sessions": session_manifest})
        if compute_features:
            cohort[subject] = sessions
        print(f"VALIDATED subject {subject:02d}: 240 events", flush=True)
    return manifest, cohort, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--data", type=Path, default=paths.DATASETS / "gao/v4_fvi")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(COHORT))
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--permute", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.permute < 0 or args.seed < 0 or len(set(args.subjects)) != len(args.subjects)
            or not args.subjects or not set(args.subjects).issubset(COHORT)):
        parser.error("Invalid seed, permutations or paired-session subjects")
    if args.resume and not args.run:
        parser.error("Resume is only supported for --run")
    if args.out.exists() and not args.resume:
        parser.error("Output exists; choose a fresh directory or explicitly --resume a run")
    if args.resume and not (args.out / "manifest.json").is_file():
        parser.error("No saved run manifest to resume")
    torch.set_num_threads(1)
    manifest, cohort, cache = load_inputs(args.data, args.subjects, args.run)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    signature = {"kind": KIND, "seed": args.seed, "subjects": sorted(args.subjects),
                 "permutations": args.permute, "inputs": manifest,
                 "versions": {"python": sys.version, "numpy": np.__version__,
                              "torch": str(torch.__version__), "scipy": scipy.__version__},
                 "settings": {"class_order": list(CLASSES), "budgets": list(BUDGETS),
                              "alpha_hz": [8, 13], "window_seconds": [0, 4], "l2": .01,
                              "epochs": 600, "lr": .05, "pooling": "mean_probabilities",
                              "scaling": "offline label-free per-session then shared training-only",
                              "bootstrap_participant_draws": 10000},
                 "protocol_sha256": digest(Path(__file__).with_name("README.md")),
                 "runner_sha256": digest(__file__),
                 "shared_sha256": {name: digest(ROOT / "shared/decoding" / name) for name in ("gao_baseline.py", "classical_baseline.py")}}
    metadata = {"signature": signature, "git_commit": commit, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__,
                "working_tree_changes_possible": True, "mode": "run" if args.run else "validate_only"}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        old = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        if old["signature"] != signature or old["mode"] != "run":
            raise ValueError("Resume rejected: inputs, protocol, code or settings changed")
    else:
        save_json(args.out / "manifest.json", metadata)  # Before the first estimator fit.
    spans = [v for subject in manifest for session in subject["sessions"] for v in session["four_event_span_seconds"]]
    if args.validate_only:
        save_json(args.out / "validation.json", {"participants": len(manifest), "events": 240 * len(manifest),
                  "scientific_models_fitted": 0, "four_event_span_median_seconds": float(np.median(spans))})
        print("VALIDATION COMPLETE: no features extracted or models fitted", flush=True)
        return
    if not args.resume or not (args.out / "features_and_scaling.npz").exists():
        np.savez_compressed(args.out / "features_and_scaling.npz", **cache)
    observed_file = args.out / "observed.json"
    if args.resume and observed_file.exists():
        observed = json.loads(observed_file.read_text(encoding="utf-8"))
    else:
        observed = [evaluate_subject(cohort[s], args.seed, keep_predictions=True) for s in sorted(cohort)]
        save_json(observed_file, observed)
    progress_file = args.out / "progress.json"
    null = json.loads(progress_file.read_text(encoding="utf-8"))["null"] if args.resume and progress_file.exists() else []
    validate_results(observed, null, sorted(cohort), args.permute)
    start = time.monotonic()
    for iteration in range(len(null), args.permute):
        rows = [evaluate_subject(cohort[s], args.seed, permutation=iteration) for s in sorted(cohort)]
        # Only compact per-person means are needed for paired/group inference.
        null.append([{k: r[k] for k in ("subject", "accuracy", "delta")} for r in rows])
        save_json(progress_file, {"completed": len(null), "null": null})
        print(f"PERMUTATIONS {len(null)}/{args.permute}; elapsed this session {time.monotonic()-start:.1f}s", flush=True)
    summary = summarize(observed, null, args.seed, args.permute)
    kept = [r for r in observed if r["subject"] != 21]
    if kept:
        sensitivity = summarize(kept, [[r for r in iteration if r["subject"] != 21] for iteration in null], args.seed, args.permute)
        sensitivity["interpretation"] = "Predefined descriptive sensitivity; cannot rescue primary"
        summary["without_sub21"] = sensitivity
    summary.update(kind=KIND, individual_results=observed,
                   four_event_span_median_seconds=float(np.median(spans)),
                   transductive_session_calibration=True, label_informed_offline_grouping=True)
    save_json(args.out / "summary.json", summary)
    print(f"SAVED {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
