"""Prepare versioned Gao FVI events locally, importing the shared converter."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "shared" / "data"))
import paths
from prepare_gao_data import convert_run, CLASSES, FS_OUT, TMIN, TMAX

import torch

SUBJECTS = tuple(s for s in range(1, 23) if s not in (9, 10))


def md5(path):
    result = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def annotate(trials, rows, source, archive_md5):
    for trial in trials:
        ordinal = trial["trial"]
        event = rows[ordinal - 1]
        if event["type"].strip() != trial["class"]:
            raise ValueError("Source event label differs from shared conversion")
        trial.update(source_file=source, source_archive_md5=archive_md5,
                     event_ordinal=ordinal, trial=int(event["trial"]),
                     source_latency_samples=int(event["latency"]),
                     source_onset_seconds=float(event["onset"]), source_fs=1000)
    keys = [(t["subject"], t["session"], t["task"], t["trial"]) for t in trials]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate source event IDs")


def prepare(raw, out, subjects):
    meta = json.loads((raw / "figshare_manifest.json").read_text(encoding="utf-8"))
    if meta["id"] != 30227503 or meta["version"] != 4:
        raise ValueError("Requires official Gao version-4 manifest")
    files = {entry["name"]: entry for entry in meta["files"]}
    destinations = [out / f"gao_sub-{subject:02d}.pt" for subject in subjects]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing converted subject")
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    converter_sha = hashlib.sha256((ROOT / "shared/data/prepare_gao_data.py").read_bytes()).hexdigest()
    preparer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((raw / "figshare_manifest.json").read_bytes()).hexdigest()
    for subject, destination in zip(subjects, destinations):
        name = f"sub-{subject:02d}.zip"
        archive = raw / name
        expected = files[name]
        digest = md5(archive)
        if archive.stat().st_size != expected["size"] or digest != expected["computed_md5"]:
            raise ValueError(f"Source archive integrity failed: {name}")
        entries, channels, run_sources = [], None, []
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            selected = sorted(n for n in names if n.endswith("_task-FVI_eeg.bdf"))
            required = [f"sub-{subject:02d}/ses-{session:02d}/eeg/sub-{subject:02d}_ses-{session:02d}_task-FVI_eeg.bdf" for session in (1, 2)]
            if selected != required or len(names) != len(set(names)):
                raise ValueError(f"Unexpected FVI members: {name}")
            for member in selected:
                session = int(member.split("_ses-")[1][:2])
                events_member = member.replace("_eeg.bdf", "_events.tsv")
                event_bytes = zf.read(events_member)
                rows = list(csv.DictReader(io.StringIO(event_bytes.decode("utf-8-sig")), delimiter="\t"))
                with tempfile.TemporaryDirectory(prefix="gao_fvi_") as temporary:
                    bdf = Path(temporary) / Path(member).name
                    with zf.open(member) as source, bdf.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    with bdf.with_name(bdf.name.replace("_eeg.bdf", "_events.tsv")).open("xb") as target:
                        target.write(event_bytes)
                    trials, ch, dropped, unit, exclusions = convert_run(str(bdf), subject, session, "FVI")
                if dropped or exclusions or len(trials) != 120:
                    raise ValueError(f"Unexpected lost/excluded FVI events: {member}")
                if channels is not None and channels != ch:
                    raise ValueError("Channel order differs between sessions")
                channels = ch
                annotate(trials, rows, member, digest)
                for label in ("circle", "pentagram", "square"):
                    if sum(t["class"] == label for t in trials) != 40:
                        raise ValueError("Expected exactly 40 events per shape")
                if any(t["eeg"].shape != (32, 3000) or not torch.isfinite(t["eeg"]).all() for t in trials):
                    raise ValueError("Unexpected epoch shape or nonfinite signal")
                entries.extend(trials)
                run_sources.append({"file": member, "events_sha256": hashlib.sha256(event_bytes).hexdigest(),
                                    "source_unit": unit, "retained_events": len(trials)})
        payload = {"dataset": entries, "classes": CLASSES, "channels": channels,
                   "fs": FS_OUT, "tmin": TMIN, "tmax": TMAX, "imagery_window": (0., 4.),
                   "exclusions": {}, "format": "gao_fvi_v4_with_event_provenance_v1",
                   "source": "Gao et al. 2026, figshare 30227503 version 4 (FVI only)",
                   "provenance": {"version": 4, "archive": name, "archive_md5": digest,
                                  "manifest_sha256": manifest_sha, "converter_sha256": converter_sha,
                                  "preparer_sha256": preparer_sha, "git_commit": commit,
                                  "working_tree_changes_possible": True, "runs": run_sources}}
        with destination.open("xb") as stream:
            torch.save(payload, stream)
        print(f"SAVED {destination}: {len(entries)} events, {len(channels)} channels; v4", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=paths.DATASETS / "gao_raw/figshare_v4")
    parser.add_argument("--out", type=Path, default=paths.DATASETS / "gao/v4_fvi")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    args = parser.parse_args()
    if len(args.subjects) != len(set(args.subjects)) or not set(args.subjects).issubset(SUBJECTS):
        parser.error("Choose unique subjects from the fixed paired-session cohort")
    torch.set_num_threads(1)
    prepare(args.raw, args.out, args.subjects)


if __name__ == "__main__":
    main()
