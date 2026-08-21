"""
Preflight data validator. Run this before submitting any training job.

It exists because the failure that wasted the previous Stage 2 run was silent: every
trial pointed at a stimulus file that did not exist, the dataset substituted a black
square, and the loss curve looked entirely healthy while the model learned nothing.
Every check below turns one of those silent failures into a loud one.

Exit code is non-zero if anything fatal is found, so a SLURM script can gate on it:

    python code/check_data.py --dataset ... --splits ... || exit 1

It can also repair the most common problem -- missing stimulus images -- by pulling
them out of an ImageNet tree you already have:

    python code/check_data.py --dataset datasets/imagination_5_95_std.pth \\
        --import_from /path/to/ImageNet/train
"""
import argparse
import collections
import os
import shutil
import sys

import torch


class Report:
    def __init__(self):
        self.fatal = []
        self.warn = []

    def ok(self, msg):
        print("  [ OK ]  %s" % msg)

    def bad(self, msg):
        print("  [FAIL]  %s" % msg)
        self.fatal.append(msg)

    def soft(self, msg):
        print("  [WARN]  %s" % msg)
        self.warn.append(msg)


def resolve_imagenet_dir(path):
    if path is None:
        return None
    nested = os.path.join(path, 'imageNet_images')
    if os.path.isdir(nested) and any(
            d.startswith('n') for d in os.listdir(nested)
            if os.path.isdir(os.path.join(nested, d))):
        return nested
    return path


def check_dataset(entries, rep):
    print("\n--- dataset integrity ---")
    n = len(entries)
    rep.ok("%d trials" % n)

    required = {'eeg', 'label', 'image', 'subject'}
    missing_keys = required - set(entries[0].keys())
    if missing_keys:
        rep.bad("trial entries are missing keys: %s" % sorted(missing_keys))
        return

    images = {e['image'] for e in entries}
    if len(images) == 1:
        rep.bad("ALL %d trials share one stimulus (%r). The image field is degenerate -- "
                "this is the bug that produced a constant training target. Re-run "
                "code/prepare_shimizu_data.py." % (n, next(iter(images))))
    else:
        rep.ok("%d distinct stimuli across trials" % len(images))

    labels = {e['label'] for e in entries}
    rep.ok("%d distinct classes, labels %d..%d" % (len(labels), min(labels), max(labels)))

    # Stimulus/label consistency: the synset embedded in the filename must be a
    # deterministic function of the label, or the pairing is scrambled.
    by_label = collections.defaultdict(set)
    for e in entries:
        by_label[e['label']].add(e['image'].split('_')[0])
    scrambled = {l: s for l, s in by_label.items() if len(s) > 1}
    if scrambled:
        rep.bad("%d label(s) map to multiple synsets, e.g. %s -- EEG/stimulus pairing is wrong"
                % (len(scrambled), dict(list(scrambled.items())[:2])))
    else:
        rep.ok("every label maps to exactly one synset (pairing is self-consistent)")

    shapes = {tuple(e['eeg'].shape) for e in entries}
    if len(shapes) == 1:
        rep.ok("EEG tensors uniformly shaped %s" % (shapes.pop(),))
    else:
        rep.soft("EEG tensors have %d distinct shapes: %s" % (len(shapes), sorted(shapes)[:4]))

    flat = torch.stack([e['eeg'].flatten()[:64] for e in entries[:512]])
    if torch.isnan(flat).any():
        rep.bad("EEG data contains NaNs")
    elif float(flat.std()) < 1e-8:
        rep.bad("EEG data is constant (std=%.2e) -- signal is dead" % float(flat.std()))
    else:
        rep.ok("EEG signal is live (sampled std=%.3f)" % float(flat.std()))

    subs = collections.Counter(e['subject'] for e in entries)
    rep.ok("subjects: %s" % dict(sorted(subs.items())))


def check_images(entries, imagenet_dir, rep, manifest_path, scope_idx=None):
    print("\n--- stimulus files ---")
    if imagenet_dir is None or not os.path.isdir(imagenet_dir):
        rep.bad("ImageNet directory not found: %s" % imagenet_dir)
        return []
    rep.ok("resolved image root: %s" % imagenet_dir)

    # Only trials reachable through the splits matter. A split built with
    # --require_images has already dropped the trials whose stimulus is absent.
    if scope_idx is not None:
        entries = [entries[i] for i in scope_idx]
        rep.ok("checking the %d trials reachable through the splits" % len(entries))

    wanted = sorted({e['image'] for e in entries})
    missing = [s for s in wanted
               if not os.path.exists(os.path.join(imagenet_dir, s.split('_')[0], s + '.JPEG'))]
    have = len(wanted) - len(missing)

    if not missing:
        rep.ok("all %d stimulus images present" % len(wanted))
        _verify_readable(wanted, imagenet_dir, entries, rep)
    else:
        n_trials_hit = sum(1 for e in entries if e['image'] in set(missing))
        rep.bad("%d/%d stimulus images missing -- %d/%d trials (%.1f%%) would train against a "
                "BLANK target" % (len(missing), len(wanted), n_trials_hit, len(entries),
                                  100.0 * n_trials_hit / len(entries)))
        with open(manifest_path, 'w') as f:
            for s in missing:
                f.write("%s/%s.JPEG\n" % (s.split('_')[0], s))
        print("          manifest of missing files written to %s" % manifest_path)
        print("          fix with: --import_from /path/to/ImageNet/train")
    return missing


def _verify_readable(wanted, imagenet_dir, entries, rep):
    """Existence is not enough: a zero-byte or truncated file passes os.path.exists but
    blows up in the DataLoader mid-epoch. Decode every stimulus once, here, where the
    failure is cheap."""
    try:
        from PIL import Image
    except ImportError:
        rep.soft("Pillow unavailable, skipping image decode check")
        return
    corrupt = []
    for stem in wanted:
        path = os.path.join(imagenet_dir, stem.split('_')[0], stem + '.JPEG')
        try:
            with Image.open(path) as im:
                im.load()
        except Exception as e:
            corrupt.append((stem, type(e).__name__))
    if corrupt:
        hit = sum(1 for e in entries if e['image'] in {c[0] for c in corrupt})
        rep.bad("%d stimulus file(s) exist but cannot be decoded (%d trials affected): %s"
                % (len(corrupt), hit, ', '.join("%s [%s]" % c for c in corrupt[:3])))
    else:
        rep.ok("all %d stimulus images decode cleanly" % len(wanted))


def check_splits(entries, splits_path, rep):
    print("\n--- split integrity ---")
    if not splits_path:
        rep.soft("no --splits given, skipping leakage audit")
        return
    if not os.path.exists(splits_path):
        rep.bad("splits file not found: %s" % splits_path)
        return

    loaded = torch.load(splits_path, map_location='cpu')
    rep.ok("protocol: %s" % loaded.get('description', 'UNSPECIFIED (legacy file)'))
    tr = loaded['splits'][0]['train']
    te = loaded['splits'][0]['test']

    if set(tr) & set(te):
        rep.bad("%d trial indices appear in BOTH train and test" % len(set(tr) & set(te)))
    else:
        rep.ok("train/test trial indices are disjoint (%d / %d)" % (len(tr), len(te)))

    oob = [i for i in list(tr) + list(te) if i >= len(entries)]
    if oob:
        rep.bad("%d split indices are out of range for a %d-trial dataset -- the splits file "
                "does not match this dataset" % (len(oob), len(entries)))
        return

    has_window = 'window' in entries[0]

    def key(e):
        # A "recording" is one subject viewing one stimulus. Sliding windows of the
        # same recording are near-duplicates, so they must not span the split.
        return (e['subject'], e['image'])

    rec_tr = {key(entries[i]) for i in tr}
    rec_te = {key(entries[i]) for i in te}
    shared = rec_tr & rec_te
    protocol = loaded.get('protocol', '')
    if shared and protocol != 'window':
        rep.bad("%d/%d test (subject,stimulus) recordings also appear in train. If the trials "
                "are sliding windows of one recording, the test set is effectively seen data."
                % (len(shared), len(rec_te)))
    elif shared:
        rep.soft("%d/%d test recordings share a (subject,stimulus) with train -- expected for "
                 "the 'window' protocol, but it is the weakest of the three." % (len(shared), len(rec_te)))
    else:
        rep.ok("no (subject,stimulus) recording spans the split")

    cls_tr = {entries[i]['label'] for i in tr}
    cls_te = {entries[i]['label'] for i in te}
    img_tr = {entries[i]['image'] for i in tr}
    img_te = {entries[i]['image'] for i in te}
    if not (cls_tr & cls_te):
        rep.soft("zero class overlap -- zero-shot protocol, expect near-chance metrics")
    if img_tr & img_te:
        rep.soft("%d stimuli appear on both sides -- report CLASS-level metrics only, not "
                 "instance-level reconstruction scores" % len(img_tr & img_te))
    print("          chance accuracy for reporting: %.4f (%d-way)"
          % (1.0 / max(1, len(cls_te)), len(cls_te)))
    if has_window:
        w_te = collections.Counter(entries[i].get('window') for i in te)
        print("          test window indices: %s" % sorted(w_te))


def import_missing(missing, source_root, imagenet_dir, rep):
    """Locate missing stimulus files inside an existing ImageNet tree and copy them in."""
    print("\n--- importing missing stimulus files ---")
    print("  scanning %s ..." % source_root)
    wanted = {s + '.JPEG' for s in missing}
    found = {}
    for dirpath, _, filenames in os.walk(source_root):
        for fn in filenames:
            if fn in wanted and fn not in found:
                found[fn] = os.path.join(dirpath, fn)
        if len(found) == len(wanted):
            break
    print("  located %d/%d" % (len(found), len(wanted)))

    copied = 0
    for fn, src in found.items():
        synset = fn.split('_')[0]
        dst_dir = os.path.join(imagenet_dir, synset)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fn)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    print("  copied %d files into %s" % (copied, imagenet_dir))
    still = sorted(wanted - set(found))
    if still:
        rep.soft("%d files were NOT found in %s (e.g. %s)"
                 % (len(still), source_root, ', '.join(still[:3])))
    return copied


def main():
    p = argparse.ArgumentParser(description="Validate EEG dataset and splits before training")
    p.add_argument("--dataset", default="datasets/imagination_5_95_std.pth")
    p.add_argument("--splits", default=None)
    p.add_argument("--imagenet_path", default="datasets/imageNet_images")
    p.add_argument("--import_from", default=None,
                   help="path to an existing ImageNet tree to pull missing stimulus files from")
    p.add_argument("--manifest", default="missing_stimuli.txt")
    p.add_argument("--skip_image_check", action="store_true",
                   help="Stage 1 (masked EEG pre-training) is self-supervised and needs no stimuli")
    args = p.parse_args()

    rep = Report()
    print("=" * 72)
    print("PREFLIGHT: %s" % args.dataset)
    print("=" * 72)

    if not os.path.exists(args.dataset):
        print("  [FAIL]  dataset not found: %s" % args.dataset)
        print("          build it with: python code/prepare_shimizu_data.py")
        sys.exit(1)

    payload = torch.load(args.dataset, map_location='cpu')
    entries = payload['dataset']
    imagenet_dir = resolve_imagenet_dir(args.imagenet_path)

    check_dataset(entries, rep)

    # Determine which trials training will actually see, so the image check is
    # scoped to them rather than to the whole dataset.
    scope_idx = None
    if args.splits and os.path.exists(args.splits):
        loaded = torch.load(args.splits, map_location='cpu')
        sp = loaded['splits'][0]
        scope_idx = sorted(set(sp['train']) | set(sp['test']))
        if any(i >= len(entries) for i in scope_idx):
            scope_idx = None          # mismatched splits; check_splits will report it

    if args.skip_image_check:
        print("\n--- stimulus files ---")
        print("  [SKIP]  EEG-only run, stimulus images not required")
        missing = []
    else:
        missing = check_images(entries, imagenet_dir, rep, args.manifest, scope_idx)

    if missing and args.import_from:
        if import_missing(missing, args.import_from, imagenet_dir, rep):
            rep.fatal = [m for m in rep.fatal if 'stimulus images missing' not in m]
            missing = check_images(entries, imagenet_dir, rep, args.manifest, scope_idx)

    check_splits(entries, args.splits, rep)

    print("\n" + "=" * 72)
    if rep.fatal:
        print("RESULT: %d FATAL, %d warning(s). DO NOT SUBMIT THE TRAINING JOB." % (len(rep.fatal), len(rep.warn)))
        for m in rep.fatal:
            print("  - %s" % m)
        sys.exit(1)
    print("RESULT: PASS (%d warning(s)). Safe to train." % len(rep.warn))
    for m in rep.warn:
        print("  - %s" % m)
    sys.exit(0)


if __name__ == "__main__":
    main()
