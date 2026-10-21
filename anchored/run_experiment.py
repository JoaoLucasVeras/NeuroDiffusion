"""
Stimulus program for the anchored-imagery experiment (PROTOCOL.md sections 5-7).

Shows the fixation dot, the anchor shape, the cue and the imagery period for every trial
of a session list from trials.py, runs the seeing localiser first, asks the catch
questions, and sends an event marker at every phase change. Markers go to Lab Streaming
Layer when pylsl is installed (most EEG systems can record an LSL marker stream) and are
always written to a CSV log with the screen-flip time.

Marker codes: 1 fixation, 2 anchor on, 3 cue on, 4 imagery start, 5 imagery end, 6 rest,
plus 10 + change index (10 move_left, 11 move_right, 12 stretch, 13 shrink) sent with the
cue, 20 / 21 for anchored / unanchored, 30 localiser animation start, 40 catch response.

    --dry-run  windowed, short timings, no markers expected: for checking the program.

Needs PsychoPy (pip install psychopy). Untested on a recording setup until hardware exists.

Usage
-----
    python run_experiment.py --participant 1 --session 1 [--dry-run] [--fullscreen]
"""
import argparse
import csv
import os
import time

from trials import CHANGES, session

TIMING = {"fixation": 1.0, "anchor": 1.5, "cue": 1.0, "imagery": 4.0}
CUE_TEXT = {"move_left": "←", "move_right": "→", "stretch": "⟷", "shrink": "→←"}
KEYS = {"left": "move_left", "right": "move_right", "up": "stretch", "down": "shrink"}


class Markers:
    def __init__(self, log_path):
        self.outlet = None
        try:
            from pylsl import StreamInfo, StreamOutlet
            self.outlet = StreamOutlet(StreamInfo("AnchoredImageryMarkers", "Markers", 1, 0, "int32", "anchored-imagery"))
        except ImportError:
            print("pylsl not installed: markers go to the CSV log only")
        self.fh = open(log_path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["time", "code", "trial", "label"])

    def send(self, code, trial, label, t):
        if self.outlet is not None:
            self.outlet.push_sample([int(code)])
        self.w.writerow(["%.6f" % t, code, trial, label])

    def close(self):
        self.fh.close()


def build(win, visual):
    unit = 3.0                                               # degrees of visual angle
    shapes = {
        "square": visual.Rect(win, width=2 * unit, height=2 * unit, lineColor="white", fillColor=None, lineWidth=4, units="deg"),
        "circle": visual.Circle(win, radius=unit, lineColor="white", fillColor=None, lineWidth=4, units="deg", edges=96),
        "triangle": visual.ShapeStim(win, vertices=[(0, unit), (unit, -0.8 * unit), (-unit, -0.8 * unit)],
                                     lineColor="white", fillColor=None, lineWidth=4, units="deg"),
    }
    dot = visual.Circle(win, radius=0.15, fillColor=(0.82, 0.58, 0.35), lineColor=None, units="deg")
    cue = visual.TextStim(win, text="", pos=(0, 2 * unit), height=2.0, units="deg", color="white")
    msg = visual.TextStim(win, text="", height=0.8, units="deg", color="white", wrapWidth=30)
    return shapes, dot, cue, msg


def show(win, core, stims, seconds):
    clock = core.Clock()
    first = None
    while clock.getTime() < seconds:
        for s in stims:
            s.draw()
        t = win.flip()
        first = first if first is not None else t
    return first


def localiser_animation(win, core, shape, dot, change, seconds):
    start = core.getTime()
    t0 = None
    while (now := core.getTime() - start) < seconds:
        k = min(now / seconds, 1.0)
        shape.pos = ((-4 * k) if change == "move_left" else (4 * k) if change == "move_right" else 0, 0)
        shape.size = (1 + 0.6 * k) if change == "stretch" else (1 - 0.5 * k) if change == "shrink" else 1
        shape.draw()
        dot.draw()
        t = win.flip()
        t0 = t0 if t0 is not None else t
    shape.pos, shape.size = (0, 0), 1
    return t0


def main():
    ap = argparse.ArgumentParser(description="Anchored-imagery stimulus program")
    ap.add_argument("--participant", type=int, required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()
    from psychopy import core, event, monitors, visual         # imported here so trials.py stays testable

    scale = 0.25 if args.dry_run else 1.0
    timing = {k: v * scale for k, v in TIMING.items()}
    rows = session(args.participant, args.session)
    os.makedirs(args.out, exist_ok=True)
    tag = "sub-%02d_ses-%d" % (args.participant, args.session)
    markers = Markers(os.path.join(args.out, tag + "_markers.csv"))
    mon = monitors.Monitor("lab", width=53.0, distance=60.0)
    mon.setSizePix((1920, 1080))
    win = visual.Window(size=(1280, 720), fullscr=args.fullscreen, monitor=mon, color=(-0.68, -0.66, -0.6), units="deg")
    shapes, dot, cue, msg = build(win, visual)
    responses = open(os.path.join(args.out, tag + "_responses.csv"), "w", newline="")
    rw = csv.writer(responses)
    rw.writerow(["trial", "true_change", "reported_change", "correct"])

    msg.text = "Keep your eyes on the dot.\nImagine the change you are asked for.\n\nPress space to start."
    msg.draw(); win.flip(); event.waitKeys(keyList=["space"])

    for r in rows:
        if "escape" in event.getKeys():
            break
        shape = shapes[r["shape"]]
        n = r["trial"]
        if r["part"] == "localiser":
            markers.send(30, n, "localiser_" + r["change"], localiser_animation(win, core, shape, dot, r["change"], timing["imagery"]))
            markers.send(6, n, "rest", show(win, core, [dot], r["rest"] * scale))
            continue
        markers.send(1, n, "fixation", show(win, core, [dot], timing["fixation"]))
        markers.send(2, n, "anchor_" + r["shape"], show(win, core, [shape, dot], timing["anchor"]))
        cue.text = CUE_TEXT[r["change"]]
        t = show(win, core, [shape, cue, dot], timing["cue"])
        markers.send(3, n, "cue", t)
        markers.send(10 + CHANGES.index(r["change"]), n, r["change"], t)
        markers.send(20 if r["anchor"] == "anchored" else 21, n, r["anchor"], t)
        stims = [shape, dot] if r["anchor"] == "anchored" else [dot]
        markers.send(4, n, "imagery_start", show(win, core, stims, timing["imagery"]))
        markers.send(5, n, "imagery_end", core.getTime())
        if r["catch"]:
            msg.text = "Which change did you imagine?\n← left   → right   ↑ stretch   ↓ shrink"
            msg.draw(); win.flip()
            key = event.waitKeys(keyList=list(KEYS))[0]
            markers.send(40, n, "catch_" + KEYS[key], core.getTime())
            rw.writerow([n, r["change"], KEYS[key], int(KEYS[key] == r["change"])])
        markers.send(6, n, "rest", show(win, core, [], r["rest"] * scale))
        if r["part"] == "main" and n % 40 == 0 and n < 320:
            msg.text = "Block %d of 8 done. Take a break.\nPress space to continue." % (n // 40)
            msg.draw(); win.flip(); event.waitKeys(keyList=["space"])

    markers.close()
    responses.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
