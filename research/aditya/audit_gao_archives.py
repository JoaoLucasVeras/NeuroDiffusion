"""Read Gao ZIPs in place: event metadata, BDF headers and FVI trigger integrity.

This does not extract files, convert EEG, or fit models. Full-record duplicate
hashing is provided separately by the existing experiment-06 scanner.
"""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np


MAPS = {"AVI": {1: "dog", 2: "bird", 3: "fish"},
        "FVI": {1: "pentagram", 2: "square", 3: "circle"},
        "OVI": {1: "scissor", 2: "watch", 3: "cup", 4: "chair"}}


def header(stream):
    fixed = stream.read(256)
    channels = int(fixed[252:256])
    size = int(fixed[184:192])
    records = int(fixed[236:244])
    seconds = float(fixed[244:252])
    if size != 256 * (channels + 1) or records < 1 or seconds <= 0:
        raise ValueError("Invalid/unsupported BDF header")
    fields = stream.read(size - 256).decode("latin-1")
    def values(offset, width):
        return [fields[offset * channels + i * width:offset * channels + (i+1) * width].strip()
                for i in range(channels)]
    names = values(0, 16)
    samples = [int(v) for v in values(216, 8)]
    if len(set(names)) != len(names) or min(samples) <= 0:
        raise ValueError("Duplicate channel labels or invalid samples")
    return {"header_bytes": size, "records": records, "record_seconds": seconds,
            "channels": names, "samples_per_record": samples}


def status(stream, info):
    """Read bounded chunks of BDF records, retaining only low-16-bit triggers."""
    names, samples = info["channels"], info["samples_per_record"]
    index = names.index("Status")
    count, start, total = samples[index], sum(samples[:index]), sum(samples)
    parts = []
    for offset in range(0, info["records"], 32):
        n = min(32, info["records"] - offset)
        block = stream.read(n * total * 3)
        if len(block) != n * total * 3:
            raise ValueError("Truncated BDF payload")
        raw = np.frombuffer(block, dtype=np.uint8).reshape(n, total, 3)
        selected = raw[:, start:start+count, :].astype(np.uint32)
        parts.append(((selected[:, :, 0] | selected[:, :, 1] << 8) & 0xffff).reshape(-1))
    return np.concatenate(parts)


def inspect(zf, name):
    prefix = name.removesuffix("_events.tsv")
    subject = int(prefix.split("sub-")[-1][:2])
    session = int(prefix.split("_ses-")[1][:2])
    task = prefix.split("_task-")[1].split("_")[0]
    rows = list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8-sig")), delimiter="\t"))
    lat = np.array([int(row["latency"]) for row in rows], dtype=np.int64)
    codes = np.array([int(row["value"]) for row in rows], dtype=np.int64)
    ordinals = [int(row["trial"]) for row in rows]
    bdf = prefix + "_eeg.bdf"
    with zf.open(bdf) as stream:
        info = header(stream)
        triggers = status(stream, info) if task == "FVI" else None
    expected = info["header_bytes"] + info["records"] * 3 * sum(info["samples_per_record"])
    fs = info["samples_per_record"][0] / info["record_seconds"]
    length = info["records"] * info["samples_per_record"][0]
    issues = []
    if not rows:
        issues.append("no_events")
    if zf.getinfo(bdf).file_size != expected:
        issues.append("bdf_length_mismatch")
    if len(set(ordinals)) != len(ordinals):
        issues.append("duplicate_trial_ids")
    if len(lat) > 1 and not (np.diff(lat) > 0).all():
        issues.append("nonincreasing_event_latencies")
    if len(lat) > 1 and (np.diff(lat) < 4 * fs).any():
        issues.append("overlapping_0_to_4_second_intervals")
    if any(sample != info["samples_per_record"][0] for sample in info["samples_per_record"]):
        issues.append("unequal_channel_rates")
    mapping = MAPS[task]
    counts = {mapping.get(int(code), f"unknown_{code}"): int(count) for code, count in Counter(codes).items()}
    if set(counts) != set(mapping.values()) or any(value != 40 for value in counts.values()):
        issues.append("class_counts_not_40_each")
    disagreement = sum(row["type"].strip() != mapping.get(int(row["value"])) for row in rows)
    onset_error = max((abs(float(row["onset"]) - int(row["latency"]) / fs) for row in rows), default=0)
    valid = (lat >= 7 * fs) & (lat + 5 * fs <= length)
    if not valid.all():
        issues.append("events_outside_converter_epoch_bounds")
    trigger_report = None
    if triggers is not None:
        if fs != info["samples_per_record"][info["channels"].index("Status")] / info["record_seconds"]:
            issues.append("status_sample_rate_differs")
        values, frequencies = np.unique(triggers, return_counts=True)
        base = values[np.argmax(frequencies)]
        positions = np.where((triggers != base) & (np.r_[base, triggers[:-1]] == base))[0]
        values = triggers[positions]
        keep = np.isin(values, np.unique(codes))
        positions, values = positions[keep], values[keep]
        matched = len(positions) == len(lat)
        max_error = int(np.abs(positions - lat).max()) if matched and len(lat) else None
        matched = bool(matched and max_error is not None and max_error <= 2 and np.array_equal(values, codes))
        if not matched:
            issues.append("trigger_event_mismatch")
        trigger_report = {"matched": matched, "count": len(positions), "max_latency_error_samples": max_error}
    retained_counts = Counter(mapping.get(int(code), "unknown") for code in codes[valid])
    bundles = {str(k): {label: count // k for label, count in retained_counts.items()} for k in (1, 2, 4, 8)}
    return {"file": name, "subject": subject, "session": session, "task": task,
            "events": len(rows), "class_counts": counts, "type_code_disagreements": disagreement,
            "unique_trial_ids": len(set(ordinals)), "fs": fs, "channels": info["channels"],
            "minimum_event_gap_seconds": float(np.diff(lat).min() / fs) if len(lat) > 1 else None,
            "onset_latency_max_error_seconds": onset_error,
            "duration_counts": dict(Counter(row["duration"] for row in rows)),
            "converter_usable_events": int(valid.sum()), "bundles_per_class": bundles,
            "trigger_check": trigger_report, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Audit output already exists; use a new filename")
    start = time.monotonic()
    runs, errors = [], []
    archives = sorted(args.raw.glob("sub-*.zip"))
    if not archives:
        raise FileNotFoundError("No subject ZIPs")
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if len(names) != len(set(names)):
                errors.append({"archive": archive.name, "error": "duplicate_zip_members"})
            for name in sorted(n for n in names if n.endswith("_events.tsv")):
                try:
                    result = inspect(zf, name)
                    runs.append(result)
                    print(f"{archive.name} {result['session']} {result['task']}: {result['events']} events; issues={result['issues']}", flush=True)
                except Exception as exc:
                    errors.append({"archive": archive.name, "file": name, "error": str(exc)})
                    print(f"ERROR {name}: {exc}", flush=True)
    report = {"archives": len(archives), "source_manifest": str(args.raw / "figshare_manifest.json"),
              "elapsed_seconds": round(time.monotonic() - start, 2), "runs": runs, "errors": errors,
              "scope": "All event TSVs/BDF headers; FVI Status triggers; no EEG model or conversion"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"SAVED {args.out}; {len(runs)} runs, {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
