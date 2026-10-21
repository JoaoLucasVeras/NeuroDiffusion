"""
Simulated EEG for the anchored-imagery protocol, with known effects, to test the analysis
before any recording exists.

Each trial follows a session list from trials.py. The epoch covers 3.0-8.0 s of the trial
(the imagery period is 3.5-7.5 s), at 128 Hz, on 32 EEG channels plus HEOG and VEOG.
Background: noise plus a 9-11 Hz alpha rhythm strongest at the back of the head. Each
session gets its own channel gains, standing in for the cap being refitted.

Scenarios:
  imagery  Imagining a leftward move lowers alpha over the RIGHT back of the head (attention
           to the left), rightward the reverse; stretch lowers alpha on both sides, shrink
           raises it slightly. The effect is stronger when the shape stays on screen
           (anchored) than when it is gone (unanchored), and varies across people.
  eyes     No brain effect. The eyes drift toward the imagined side, which shows in HEOG and
           leaks weakly into the frontal channels. The analysis must catch this.
  null     No effect of any kind.

Usage
-----
    python simulate.py --scenario imagery --participant 1 --out sim_sub-01.pt
"""
import argparse

import numpy as np
import torch

from trials import session

EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
       "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2"]
CHANNELS = EEG + ["HEOG", "VEOG"]
LEFT_BACK = ["P7", "P3", "PO7", "PO3", "O1"]
RIGHT_BACK = ["P8", "P4", "PO8", "PO4", "O2"]
BACK = LEFT_BACK + RIGHT_BACK + ["Pz", "Oz"]
FRONT = ["Fp1", "Fp2", "F7", "F8"]
FS, TMIN, TMAX = 128, 3.0, 8.0
IMG = (3.5, 7.5)


def alpha_topography():
    w = np.full(len(EEG), 0.3)
    for c in BACK:
        w[EEG.index(c)] = 1.0
    return w


def trial_eeg(rng, gains, scenario, change, anchor, strength):
    n = int((TMAX - TMIN) * FS)
    t = TMIN + np.arange(n) / FS
    img = (t >= IMG[0]) & (t < IMG[1])
    x = rng.normal(0, 1.0, (len(CHANNELS), n))
    f = rng.uniform(9, 11)
    alpha = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # alpha strength varies a lot from trial to trial and channel to channel, as in real data
    amp = 2.0 * alpha_topography() * rng.lognormal(0.0, 0.35) * rng.lognormal(0.0, 0.2, len(EEG))
    mod = np.ones((len(EEG), n))
    if scenario == "imagery":
        e = strength * (1.0 if anchor == "anchored" else 0.4)
        sides = {"move_left": RIGHT_BACK, "move_right": LEFT_BACK, "stretch": LEFT_BACK + RIGHT_BACK}
        if change in sides:
            for c in sides[change]:
                mod[EEG.index(c), img] -= e * (0.6 if change == "stretch" else 1.0)
        else:                                               # shrink
            for c in LEFT_BACK + RIGHT_BACK:
                mod[EEG.index(c), img] += 0.3 * e
    x[:len(EEG)] += (amp[:, None] * mod) * alpha[None, :]
    if scenario == "eyes" and change in ("move_left", "move_right"):
        drift = (-1 if change == "move_left" else 1) * strength * 3.0
        ramp = np.clip((t - IMG[0]) / 1.0, 0, 1) * img
        x[CHANNELS.index("HEOG")] += drift * ramp
        for c, sgn in (("F7", 1), ("F8", -1), ("Fp1", 0.5), ("Fp2", -0.5)):
            x[EEG.index(c)] += 0.15 * sgn * drift * ramp
    x[:len(EEG)] *= gains[:, None]
    return x.astype(np.float32)


def simulate(scenario, participant, n_sessions=3):
    rng = np.random.default_rng(10_000 + participant)
    # imagery effects are small and differ between people; eye drift is large when present
    strength = {"imagery": rng.uniform(0.04, 0.14), "eyes": rng.uniform(0.5, 1.0), "null": 0.0}[scenario]
    data = []
    for s in range(1, n_sessions + 1):
        gains = rng.uniform(0.8, 1.2, len(EEG))
        for r in session(participant, s):
            if r["part"] != "main":
                continue
            data.append({"eeg": torch.from_numpy(trial_eeg(rng, gains, scenario, r["change"], r["anchor"], strength)),
                         "change": r["change"], "anchor": r["anchor"], "shape": r["shape"],
                         "session": s, "block": r["block"], "trial": r["trial"]})
    return {"dataset": data, "channels": CHANNELS, "eog": ["HEOG", "VEOG"], "fs": FS, "tmin": TMIN,
            "scenario": scenario, "participant": participant, "strength": float(strength)}


def main():
    ap = argparse.ArgumentParser(description="Simulate anchored-imagery EEG")
    ap.add_argument("--scenario", choices=["imagery", "eyes", "null"], default="imagery")
    ap.add_argument("--participant", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = simulate(args.scenario, args.participant)
    torch.save(d, args.out)
    print("%s participant %d: %d trials, strength %.2f -> %s" % (args.scenario, args.participant,
                                                                 len(d["dataset"]), d["strength"], args.out))


if __name__ == "__main__":
    main()
