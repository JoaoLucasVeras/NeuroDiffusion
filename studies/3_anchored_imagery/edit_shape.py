"""
Draw an anchor shape and apply a decoded change: the "watch it change" output.

For step 2 of the ladder the edit is geometric, so it is done directly: a shift for
move_left / move_right, a scale for stretch / shrink. `confidence` (0-1) sets how far the
change goes, so a weak decode makes a small change and a confident one a full change. An
AI image editor takes over at step 3, when the changes are attributes rather than geometry.

Usage
-----
    python edit_shape.py --shape square --change move_left --confidence 0.8 --out left.gif
"""
import argparse

from PIL import Image, ImageDraw

SIZE = 512
BG, FG = (40, 44, 52), (235, 235, 230)
FULL = {"move_left": ("dx", -150), "move_right": ("dx", 150), "stretch": ("scale", 1.6), "shrink": ("scale", 0.5)}


def draw(shape, dx=0.0, scale=1.0, radius=90):
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    cx, cy, r = SIZE / 2 + dx, SIZE / 2, radius * scale
    if shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=FG, width=8)
    elif shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=FG, width=8)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)], outline=FG, width=8)
    else:
        raise ValueError("unknown shape %r" % shape)
    d.ellipse([SIZE / 2 - 5, SIZE / 2 - 5, SIZE / 2 + 5, SIZE / 2 + 5], fill=(232, 147, 90))   # fixation dot
    return img


def frame(shape, change, amount):
    """amount 0 = the anchor as shown, 1 = the full change."""
    kind, full = FULL[change]
    if kind == "dx":
        return draw(shape, dx=full * amount)
    return draw(shape, scale=1.0 + (full - 1.0) * amount)


def animate(shape, change, confidence=1.0, steps=24):
    confidence = min(max(confidence, 0.0), 1.0)
    return [frame(shape, change, confidence * i / (steps - 1)) for i in range(steps)]


def main():
    ap = argparse.ArgumentParser(description="Apply a decoded change to an anchor shape")
    ap.add_argument("--shape", default="square", choices=["square", "circle", "triangle"])
    ap.add_argument("--change", default="move_left", choices=sorted(FULL))
    ap.add_argument("--confidence", type=float, default=1.0)
    ap.add_argument("--out", default="change.gif")
    args = ap.parse_args()
    frames = animate(args.shape, args.change, args.confidence)
    frames[0].save(args.out, save_all=True, append_images=frames[1:] + frames[-1:] * 12, duration=40, loop=0)
    print("wrote %s (%d frames)" % (args.out, len(frames)))


if __name__ == "__main__":
    main()
