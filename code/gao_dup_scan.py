"""
Look for duplicated recordings in the published Gao et al. data.

Found by accident on 2026-10-20: in sub-21.zip three recordings hold byte-identical EEG
(ses-01 FVI, ses-02 FVI, ses-02 AVI); only the timestamps in their headers differ. This
checks every recording of every person the same way: an MD5 of each BDF file's data
section (everything after its header, whose length the header states at byte 184) and of
each events file, streamed straight out of the zips, then reports any data or events
shared by more than one recording.

Usage
-----
    python gao_dup_scan.py --raw ../datasets/gao_raw --json out.json
"""
import argparse
import glob
import hashlib
import json
import os
import zipfile


def md5_after(fh, skip):
    h = hashlib.md5()
    fh.read(skip)
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Scan the Gao zips for duplicated recordings")
    ap.add_argument("--raw", default="../datasets/gao_raw")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for z in sorted(glob.glob(os.path.join(args.raw, "sub-*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("_eeg.bdf"):
                    with zf.open(name) as fh:
                        head = fh.read(256)
                    hdr = int(head[184:192].decode("ascii").strip())
                    with zf.open(name) as fh:
                        rows.append({"file": name, "kind": "eeg", "header": hdr,
                                     "date": head[168:184].decode("ascii"),
                                     "md5": md5_after(fh, hdr)})
                elif name.endswith("_events.tsv"):
                    with zf.open(name) as fh:
                        rows.append({"file": name, "kind": "events", "md5": hashlib.md5(fh.read()).hexdigest()})
        print("%s: %d files hashed" % (os.path.basename(z), sum(1 for r in rows if r["file"].startswith(os.path.basename(z)[:6]))), flush=True)

    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["md5"]), []).append(r["file"])
    dups = [{"kind": k, "md5": m, "files": f} for (k, m), f in groups.items() if len(f) > 1]
    print("\n%d EEG recordings, %d events files hashed" % (sum(r["kind"] == "eeg" for r in rows),
                                                        sum(r["kind"] == "events" for r in rows)))
    print("%d duplicate groups" % len(dups))
    for d in dups:
        print("  %s %s:\n    %s" % (d["kind"], d["md5"][:12], "\n    ".join(d["files"])))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "duplicates": dups}, fh, indent=2)


if __name__ == "__main__":
    main()
