"""Download and checksum the official Gao v4 release; no extraction or analysis."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "datasets" / "gao_raw" / "figshare_v4"
URL = "https://api.figshare.com/v2/articles/30227503/versions/4"


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def fetch(entry):
    name = entry["name"]
    if Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Unsafe filename in manifest")
    path = DEST / name
    expected = entry["computed_md5"].lower()
    if path.exists():
        md5, sha256 = hashes(path)
        if path.stat().st_size != entry["size"] or md5 != expected:
            raise ValueError(f"Existing file differs from manifest; left untouched: {name}")
        print(f"VERIFIED existing {name}", flush=True)
    else:
        part = DEST / (name + ".part")
        # Never silently overwrite a partial file from another/interrupted run.
        with part.open("xb") as output:
            with urllib.request.urlopen(entry["download_url"], timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        md5, sha256 = hashes(part)
        if part.stat().st_size != entry["size"] or md5 != expected:
            raise ValueError(f"Checksum/size mismatch; retained partial file: {name}")
        if path.exists():
            raise FileExistsError(f"Destination appeared during download: {name}")
        part.rename(path)
        print(f"VERIFIED downloaded {name} ({entry['size'] / 1e6:.1f} MB)", flush=True)
    return {"name": name, "file_id": entry["id"], "bytes": entry["size"],
            "md5": md5, "sha256": sha256, "download_url": entry["download_url"]}


def main():
    started = time.monotonic()
    with urllib.request.urlopen(URL, timeout=60) as response:
        meta = json.load(response)
    if meta["id"] != 30227503 or meta["version"] != 4:
        raise ValueError("Unexpected dataset/version")
    files = meta["files"]
    names = [f["name"] for f in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames")
    expected_subjects = {f"sub-{i:02d}.zip" for i in range(1, 23)}
    if {name for name in names if name.endswith(".zip")} != expected_subjects:
        raise ValueError("Unexpected subject archive set")
    required = sum(f["size"] for f in files)
    if shutil.disk_usage(ROOT).free < required * 2 + 2_000_000_000:
        raise OSError("Insufficient free space including safety margin")
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST / "figshare_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        identity = lambda m: sorted((f["id"], f["name"], f["size"], f["computed_md5"]) for f in m["files"])
        if identity(old) != identity(meta):
            raise ValueError("Saved manifest differs; not overwriting")
    else:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2)
    print(f"Gao v4: {len(files)} files; {required / 1e9:.3f} GB; {DEST}", flush=True)
    verified = []
    errors = []
    # Two downloads at a time; no third-party dependencies or downloaded code runs.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(fetch, entry): entry["name"] for entry in files}
        for future in concurrent.futures.as_completed(pending):
            try:
                verified.append(future.result())
                total = sum(f["bytes"] for f in verified)
                print(f"PROGRESS {len(verified)}/{len(files)}; verified {total / 1e9:.3f} GB", flush=True)
            except Exception as exc:
                errors.append({"name": pending[future], "error": str(exc)})
                print(f"FAILED {pending[future]}: {exc}", flush=True)
    report = {"source_api": URL, "version": 4,
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "verified": sorted(verified, key=lambda item: item["name"]),
              "errors": errors, "complete": len(verified) == len(files) and not errors}
    report_path = DEST / f"verification_{time.time_ns()}.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"REPORT {report_path}", flush=True)
    if not report["complete"]:
        raise SystemExit("Download incomplete; see report. Existing files/partials preserved.")
    print(f"COMPLETE: {len(files)} verified files; no extraction or analysis performed.", flush=True)


if __name__ == "__main__":
    main()
