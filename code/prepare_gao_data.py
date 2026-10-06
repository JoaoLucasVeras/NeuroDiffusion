"""
Convert the Gao et al. (2026) visual imagery dataset from raw BIDS/BDF into trials.

Source: figshare 10.6084/m9.figshare.30227503, version 3, CC BY 4.0. 22 participants,
32 channels at 1000 Hz, two sessions (sub-09 and sub-10 have one), three tasks recorded
as separate runs: AVI (dog, bird, fish), FVI (circle, pentagram, square) and OVI
(scissor, watch, cup, chair). 40 trials per class per session.

What is kept, and what is not
-----------------------------
- A 12 s window per trial, -7 to +5 s around the event marker. The marker sits inside
  the black-screen imagery period: on sub-09 the picture is on screen from about -5.8 to
  -1.7 s and the 4 s after the marker are black screen, measured from the occipital
  response and alpha power because the paper does not say. So each trial holds the
  perception period, the imagery period, and margin either side. The rest of each run,
  fixation and most of the rest period, is dropped.
- 1000 Hz resampled to 250 Hz with an anti-aliasing polyphase filter. Everything this
  project studies is below 50 Hz.
- Values in microvolts, float32, which holds the source's 24-bit precision exactly.
- No re-referencing, filtering or ICA. Cleaning choices stay open for experiments.

Every event is checked against the Status channel in the BDF, and the run is refused if
they disagree. The authors' example Preprocess.py reads only codes 1-3, which would drop
every OVI "chair" trial; this reads every code the events file defines.

Usage
-----
    python prepare_gao_data.py --bids <dir containing sub-XX/> --subject 9 \\
        --out ../datasets/gao
writes ../datasets/gao/gao_sub-09.pt
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch
from scipy.signal import resample_poly

FS_OUT = 250
TMIN, TMAX = -7.0, 5.0
TASKS = ("AVI", "FVI", "OVI")
# one global index per stimulus; the three tasks were recorded as separate runs, so
# decoding should stay within a task (3 or 4 way) or it can separate runs instead.
CLASSES = ["dog", "bird", "fish", "circle", "pentagram", "square",
           "scissor", "watch", "cup", "chair"]
IMAGE = {"dog": "Animal_dog", "bird": "Animal_bird", "fish": "Animal_fish",
         "circle": "Figure_circle", "pentagram": "Figure_pentagram", "square": "Figure_square",
         "scissor": "Object_scissor", "watch": "Object_watch", "cup": "Object_cup",
         "chair": "Object_chair"}


def read_bdf(path):
    """Minimal BDF reader: channel labels, sampling rates, and a function returning one
    channel in physical units. No dependencies beyond numpy."""
    raw = open(path, "rb").read()
    h = raw[:256].decode("latin-1")
    ns, nrec, rdur, hlen = int(h[252:256]), int(h[236:244]), float(h[244:252]), int(h[184:192])
    sh = raw[256:hlen].decode("latin-1")

    def fld(off, w):
        return [sh[off * ns + i * w: off * ns + (i + 1) * w].strip() for i in range(ns)]

    labels = fld(0, 16)
    pmin, pmax = [float(v) for v in fld(104, 8)], [float(v) for v in fld(112, 8)]
    dmin, dmax = [float(v) for v in fld(120, 8)], [float(v) for v in fld(128, 8)]
    units = fld(96, 8)
    nsamp = [int(v) for v in fld(216, 8)]
    rec = np.frombuffer(raw, np.uint8, offset=hlen)[:nrec * sum(nsamp) * 3]
    rec = rec.reshape(nrec, sum(nsamp), 3)

    def chan(name, physical=True):
        i = labels.index(name)
        s = sum(nsamp[:i])
        b = rec[:, s:s + nsamp[i], :].reshape(-1, 3).astype(np.int64)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64)
        if not physical:
            return v
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        x = (v - dmin[i]) * gain + pmin[i]
        scale = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6}.get(units[i].lower(), 1.0)
        return x * scale

    return labels, nsamp[0] / rdur, chan, units


def status_onsets(chan):
    s = chan("Status", physical=False).astype(np.int64) & 0xFFFF
    vals, counts = np.unique(s, return_counts=True)
    base = vals[np.argmax(counts)]
    on = np.where((s != base) & (np.r_[base, s[:-1]] == base))[0]
    return on, s[on]


def convert_run(bdf, subject, session, task):
    labels, fs, chan, units = read_bdf(bdf)
    eeg_names = [l for l in labels if l != "Status"]
    ev = list(csv.DictReader(open(bdf.replace("_eeg.bdf", "_events.tsv")), delimiter="\t"))
    lat = np.array([int(r["latency"]) for r in ev])
    val = np.array([int(r["value"]) for r in ev])
    names = [r["type"].strip() for r in ev]

    # the events file must agree with the triggers recorded in the BDF itself
    on, code = status_onsets(chan)
    keep = np.isin(code, np.unique(val))
    on, code = on[keep], code[keep]
    if len(on) != len(lat) or np.abs(on - lat).max() > 2 or not (code == val).all():
        raise SystemExit("FATAL: %s events disagree with its Status channel "
                         "(%d events, %d triggers)" % (os.path.basename(bdf), len(lat), len(on)))

    x = np.stack([chan(c) for c in eeg_names])                    # (32, time), microvolts
    lo, hi = int(round(TMIN * fs)), int(round(TMAX * fs))
    trials, dropped = [], 0
    for k, (l, name) in enumerate(zip(lat, names)):
        if l + lo < 0 or l + hi > x.shape[1]:
            dropped += 1
            continue
        seg = x[:, l + lo:l + hi]
        seg = resample_poly(seg, FS_OUT, int(fs), axis=1).astype(np.float32)
        trials.append({"eeg": torch.from_numpy(seg), "label": CLASSES.index(name),
                       "class": name, "image": IMAGE[name], "task": task,
                       "subject": subject, "session": session, "trial": k + 1})
    return trials, eeg_names, dropped, units[0]


def main():
    p = argparse.ArgumentParser(description="Gao et al. BDF -> trials")
    p.add_argument("--bids", required=True, help="directory containing sub-XX/")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--out", default="../datasets/gao")
    args = p.parse_args()

    sub = "sub-%02d" % args.subject
    runs = sorted(glob.glob(os.path.join(args.bids, sub, "ses-*", "eeg", "*_eeg.bdf")))
    if not runs:
        raise SystemExit("FATAL: no BDF files under %s" % os.path.join(args.bids, sub))

    entries, channels = [], None
    for bdf in runs:
        base = os.path.basename(bdf)
        session = int(base.split("_ses-")[1][:2])
        task = base.split("_task-")[1].split("_")[0]
        trials, ch, dropped, unit = convert_run(bdf, args.subject, session, task)
        if channels is None:
            channels = ch
        elif ch != channels:
            raise SystemExit("FATAL: channel list differs in %s" % base)
        counts = {}
        for t in trials:
            counts[t["class"]] = counts.get(t["class"], 0) + 1
        print("  %-38s %3d trials %s%s  (unit %s)" % (base, len(trials), counts,
              "  DROPPED %d at the run edges" % dropped if dropped else "", unit))
        entries.extend(trials)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "gao_%s.pt" % sub)
    torch.save({"dataset": entries, "classes": CLASSES, "channels": channels,
                "fs": FS_OUT, "tmin": TMIN, "tmax": TMAX,
                "imagery_window": (0.0, 4.0),
                "source": "Gao et al. 2026, figshare 10.6084/m9.figshare.30227503 v3, CC BY 4.0"},
               out)
    print("%s: %d trials, %d channels, %d samples each -> %s"
          % (sub, len(entries), len(channels), entries[0]["eeg"].shape[1], out))


if __name__ == "__main__":
    main()
