"""
Pull the Shimizu stimulus images out of an ImageNet download into datasets/imageNet_images.

You need 1926 specific files spread over 40 synsets -- roughly 250 MB, not the full 138 GB
ImageNet. This script never extracts more than it has to.

ImageNet-1k train ships in three shapes; all three are handled:

  1. Extracted tree            .../train/n02106662/n02106662_10284.JPEG
     python code/fetch_stimuli.py --from_dir /path/to/ImageNet/train

  2. Per-synset tars           .../n02106662.tar, n02124075.tar, ...
     python code/fetch_stimuli.py --from_tars /path/to/synset_tars

  3. The single monolithic tar ILSVRC2012_img_train.tar (contains the per-synset tars)
     python code/fetch_stimuli.py --from_tars /path/to/ILSVRC2012_img_train.tar

Only the 40 synsets listed in needed_synsets.txt are touched. Afterwards:

  python code/check_data.py --dataset datasets/imagination_5_95_std.pth \\
      --splits datasets/imagination_5_95_std_splits_subject.pth
"""
import argparse
import io
import os
import shutil
import sys
import tarfile


def load_manifest(path):
    if not os.path.exists(path):
        sys.exit("Manifest %s not found. Generate it with:\n"
                 "  python code/check_data.py --dataset datasets/visual_5_95_std.pth" % path)
    with open(path) as f:
        entries = [ln.strip() for ln in f if ln.strip()]
    wanted = {}
    for rel in entries:
        synset = rel.split('/')[0]
        wanted.setdefault(synset, set()).add(os.path.basename(rel))
    return wanted


def resolve_dest(root):
    nested = os.path.join(root, 'imageNet_images')
    if os.path.isdir(nested):
        return nested
    return root


def copy_from_dir(wanted, source, dest):
    flat = {fn: syn for syn, fns in wanted.items() for fn in fns}
    found = 0
    print("Scanning %s ..." % source)
    for dirpath, _, filenames in os.walk(source):
        for fn in filenames:
            syn = flat.get(fn)
            if syn is None:
                continue
            out_dir = os.path.join(dest, syn)
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, fn)
            if not os.path.exists(out):
                shutil.copy2(os.path.join(dirpath, fn), out)
                found += 1
        if found and found % 200 == 0:
            print("  %d copied ..." % found)
    return found


def extract_from_synset_tar(fileobj, synset, filenames, dest):
    out_dir = os.path.join(dest, synset)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=fileobj, mode='r|*') as tf:
        for member in tf:
            base = os.path.basename(member.name)
            if base in filenames:
                out = os.path.join(out_dir, base)
                if not os.path.exists(out):
                    src = tf.extractfile(member)
                    if src is not None:
                        with open(out, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        n += 1
    return n


def copy_from_tars(wanted, source, dest):
    total = 0
    if os.path.isdir(source):
        for synset, filenames in sorted(wanted.items()):
            tar_path = os.path.join(source, synset + '.tar')
            if not os.path.exists(tar_path):
                print("  %s: tar not found, skipping" % synset)
                continue
            with open(tar_path, 'rb') as fh:
                n = extract_from_synset_tar(fh, synset, filenames, dest)
            print("  %s: %d/%d extracted" % (synset, n, len(filenames)))
            total += n
        return total

    # monolithic ILSVRC2012_img_train.tar: a tar of per-synset tars
    print("Streaming %s (this reads the archive once) ..." % source)
    with tarfile.open(source, mode='r|*') as outer:
        for member in outer:
            synset = os.path.splitext(os.path.basename(member.name))[0]
            if synset not in wanted:
                continue
            data = outer.extractfile(member)
            if data is None:
                continue
            buf = io.BytesIO(data.read())
            n = extract_from_synset_tar(buf, synset, wanted[synset], dest)
            print("  %s: %d/%d extracted" % (synset, n, len(wanted[synset])))
            total += n
    return total


def main():
    p = argparse.ArgumentParser(description="Fetch Shimizu stimulus images from an ImageNet copy")
    p.add_argument("--manifest", default="missing_stimuli.txt")
    p.add_argument("--from_dir", default=None, help="extracted ImageNet tree")
    p.add_argument("--from_tars", default=None,
                   help="directory of per-synset tars, or ILSVRC2012_img_train.tar")
    p.add_argument("--dest", default="datasets/imageNet_images")
    args = p.parse_args()

    if not (args.from_dir or args.from_tars):
        sys.exit("Give either --from_dir or --from_tars. See the docstring for the three layouts.")

    wanted = load_manifest(args.manifest)
    n_files = sum(len(v) for v in wanted.values())
    dest = resolve_dest(args.dest)
    print("Manifest: %d files across %d synsets" % (n_files, len(wanted)))
    print("Destination: %s\n" % dest)

    if args.from_dir:
        copied = copy_from_dir(wanted, args.from_dir, dest)
    else:
        copied = copy_from_tars(wanted, args.from_tars, dest)

    print("\nCopied %d files." % copied)
    still = [
        "%s/%s" % (syn, fn)
        for syn, fns in wanted.items() for fn in fns
        if not os.path.exists(os.path.join(dest, syn, fn))
    ]
    if still:
        print("Still missing %d file(s), e.g. %s" % (len(still), ', '.join(sorted(still)[:3])))
        with open(args.manifest, 'w') as f:
            for rel in sorted(still):
                f.write(rel + "\n")
        print("Manifest rewritten with just the outstanding files.")
        sys.exit(1)
    print("All stimulus files present. Re-run code/check_data.py to confirm.")


if __name__ == "__main__":
    main()
