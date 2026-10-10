"""
Option B: the Gao et al. preprocessing, reproduced with the authors' own code.

Runs third_party/gao_preprocess.py (their Preprocess.py, CC BY 4.0) on the raw BDF:
pyprep bad-channel detection, 50 Hz notch, 1-100 Hz, average reference, Picard ICA with
ICLabel rejection of every component not labelled brain or other, bad-channel
interpolation, 4-80 Hz, 250 Hz, epochs -0.2 to 4 s baseline-corrected and cropped to
0-4 s. Their functions are called with their parameters; nothing in them is changed.

Two things are not taken from their script, because as written it would mislabel:
- Events. Their epoching names only dog, bird and fish (codes 1-3), so it would drop
  every OVI chair trial (code 4). Here every run uses the code table for its own task
  and the same exclusions as prepare_gao_data.py: sub-08 ses-01 OVI code 1 (scissor and
  watch shared it) and sub-13 ses-02 OVI code 5 (undocumented).
- Channel selection takes the EEG channels by type rather than by position.

Reproduced as published, and worth knowing: their pipeline multiplies the data by 1e-6
after MNE has already converted it to volts, so amplitudes are a million times too
small. ICA, ICLabel and the per-channel standardisation in EEGNet are largely
insensitive to a global scale, so it is kept to match their numbers, not fixed.

Runs in the separate gaoprep environment (jobs/setup_gao_prep.sh), never neurodiffusion.

Usage
-----
    ~/.conda/envs/gaoprep/bin/python gao_ica_prep.py --bids <unpacked dir> --subject 1 \\
        --out ../datasets/gao_ica
"""
import argparse
import csv
import glob
import os

import mne
import numpy as np
import torch

from prepare_gao_data import VALUE_MAP, PER_CLASS, IMAGE, CLASSES
from third_party.gao_preprocess import preprocess_eeg_data, remove_artifacts_ica

mne.set_log_level("WARNING")


def run_events(bdf, task):
    """Events from the events file, labelled by verified code, faulty codes excluded."""
    rows = list(csv.DictReader(open(bdf.replace("_eeg.bdf", "_events.tsv")), delimiter="\t"))
    lat = np.array([int(r["latency"]) for r in rows])
    val = np.array([int(r["value"]) for r in rows])
    vmap = VALUE_MAP[task]
    counts = {v: int((val == v).sum()) for v in np.unique(val)}
    keep = np.array([v in vmap and counts[v] <= PER_CLASS for v in val])
    excluded = int((~keep).sum())
    events = np.stack([lat[keep], np.zeros(keep.sum(), int), val[keep]], axis=1).astype(int)
    trial_no = np.arange(1, len(rows) + 1)[keep]
    return events, trial_no, excluded


def process(bdf, subject, session, task):
    raw = mne.io.read_raw_bdf(bdf, preload=True)
    events, trial_no, excluded = run_events(bdf, task)
    data = raw.get_data()
    data *= 1e-6                                       # as in their pipeline (see docstring)
    raw._data = data
    raw, events = raw.resample(sfreq=250, events=events)
    raw = preprocess_eeg_data(raw, low_freq=1.0, high_freq=100.0,
                              montage_name="standard_1020", reference="average")
    cleaned, ica = remove_artifacts_ica(raw, n_components=0.999999, low_freq=4.0, high_freq=80.0)
    vmap = VALUE_MAP[task]
    event_id = {vmap[c]: c for c in sorted(set(events[:, 2]))}
    ep = mne.Epochs(cleaned, events=events, event_id=event_id, tmin=-0.2, tmax=4.0,
                    baseline=(-0.2, 0), preload=True, picks="eeg", on_missing="warn")
    ep.crop(tmin=0)
    X = ep.get_data()[:, :, :1000].astype(np.float32)
    kept = ep.selection                                   # indices into events that survived
    out = []
    for x, code, k in zip(X, ep.events[:, 2], kept):
        name = vmap[int(code)]
        out.append({"eeg": torch.from_numpy(x), "label": CLASSES.index(name), "class": name,
                    "image": IMAGE[name], "task": task, "subject": subject,
                    "session": session, "trial": int(trial_no[k])})
    return out, ep.ch_names, excluded, len(events) - len(kept)


def main():
    ap = argparse.ArgumentParser(description="Gao et al. preprocessing, authors' code")
    ap.add_argument("--bids", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", default="../datasets/gao_ica")
    args = ap.parse_args()

    sub = "sub-%02d" % args.subject
    runs = sorted(glob.glob(os.path.join(args.bids, sub, "ses-*", "eeg", "*_eeg.bdf")))
    if not runs:
        raise SystemExit("FATAL: no BDF files under %s" % os.path.join(args.bids, sub))
    entries, channels = [], None
    for bdf in runs:
        base = os.path.basename(bdf)
        session = int(base.split("_ses-")[1][:2])
        task = base.split("_task-")[1].split("_")[0]
        trials, ch, excluded, lost = process(bdf, args.subject, session, task)
        channels = channels or ch
        counts = {}
        for t in trials:
            counts[t["class"]] = counts.get(t["class"], 0) + 1
        print("  %-38s %3d trials %s  excluded %d by code, %d dropped by epoching"
              % (base, len(trials), counts, excluded, lost))
        entries.extend(trials)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "gao_%s.pt" % sub)
    torch.save({"dataset": entries, "classes": CLASSES, "channels": channels, "fs": 250,
                "tmin": 0.0, "tmax": 4.0, "variant": "ica",
                "source": "Gao et al. 2026 figshare v3, preprocessed with the authors' code"}, out)
    print("%s: %d trials -> %s" % (sub, len(entries), out))


if __name__ == "__main__":
    main()
