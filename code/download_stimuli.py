"""
Download the Shimizu stimulus images directly from image-net.org.

The whole-synset tars under /data/winter21_whole/ are served without credentials and
preserve the original ImageNet filenames (n02106662_10284.JPEG), which is what this
dataset references. That makes them the one source that can satisfy the manifest
exactly -- most redistributions re-encode into shards and drop the original names.

Each tar is streamed and discarded as it goes: only the ~48 wanted members per synset
are written to disk. Nothing is stored except the images you need (~250 MB total),
though the transfer itself is several GB because the wanted files are spread through
each archive.

Resumable: files already present are skipped, so re-running after an interruption only
fetches what is left.

Usage:
    python code/download_stimuli.py
    python code/download_stimuli.py --manifest missing_stimuli.txt --dest datasets/imageNet_images
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

BASE_URL = "https://image-net.org/data/winter21_whole/%s.tar"
UA = {"User-Agent": "curl/8.0"}


def load_manifest(path):
    if not os.path.exists(path):
        sys.exit("Manifest %s not found. Generate it with code/check_data.py." % path)
    wanted = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            synset = ln.split("/")[0]
            wanted.setdefault(synset, set()).add(os.path.basename(ln))
    return wanted


def resolve_dest(root):
    nested = os.path.join(root, "imageNet_images")
    return nested if os.path.isdir(nested) else root


def fetch_synset(synset, filenames, dest, retries=3):
    """Stream one synset tar, writing only the wanted members. Stops once all are found."""
    out_dir = os.path.join(dest, synset)
    os.makedirs(out_dir, exist_ok=True)

    outstanding = {fn for fn in filenames if not os.path.exists(os.path.join(out_dir, fn))}
    if not outstanding:
        return 0, 0, 0.0

    url = BASE_URL % synset
    for attempt in range(1, retries + 1):
        got, read_bytes = 0, 0
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)
            counting = _CountingReader(resp)
            with tarfile.open(fileobj=counting, mode="r|") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if name not in outstanding:
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    tmp = os.path.join(out_dir, name + ".part")
                    with open(tmp, "wb") as fh:
                        fh.write(src.read())
                    os.replace(tmp, os.path.join(out_dir, name))
                    outstanding.discard(name)
                    got += 1
                    if not outstanding:
                        break          # every wanted file found; stop reading this tar
            read_bytes = counting.count
            resp.close()
            return got, read_bytes, time.time() - t0
        except (urllib.error.URLError, tarfile.TarError, OSError) as e:
            if attempt == retries:
                print("    %s: FAILED after %d attempts (%s)" % (synset, retries, type(e).__name__))
                return got, read_bytes, time.time() - t0
            print("    %s: %s, retrying (%d/%d)" % (synset, type(e).__name__, attempt, retries))
            time.sleep(2 * attempt)
    return 0, 0, 0.0


class _CountingReader:
    """Wraps a stream so we can report how many bytes were actually pulled."""

    def __init__(self, raw):
        self.raw = raw
        self.count = 0

    def read(self, n=None):
        chunk = self.raw.read(n) if n is not None else self.raw.read()
        self.count += len(chunk)
        return chunk


def main():
    p = argparse.ArgumentParser(description="Download stimulus images from image-net.org")
    p.add_argument("--manifest", default="missing_stimuli.txt")
    p.add_argument("--dest", default="datasets/imageNet_images")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these synsets")
    args = p.parse_args()

    wanted = load_manifest(args.manifest)
    if args.only:
        wanted = {k: v for k, v in wanted.items() if k in set(args.only)}
    dest = resolve_dest(args.dest)
    total_files = sum(len(v) for v in wanted.values())

    print("Manifest : %d files across %d synsets" % (total_files, len(wanted)))
    print("Dest     : %s" % dest)
    print("Source   : image-net.org/data/winter21_whole (no credentials required)\n")

    got_total, bytes_total, t_start = 0, 0, time.time()
    for i, synset in enumerate(sorted(wanted), 1):
        need = len(wanted[synset])
        got, nbytes, secs = fetch_synset(synset, wanted[synset], dest)
        got_total += got
        bytes_total += nbytes
        have = sum(1 for fn in wanted[synset]
                   if os.path.exists(os.path.join(dest, synset, fn)))
        print("[%2d/%2d] %s  %d/%d present  (+%d, %.0f MB, %.0fs)"
              % (i, len(wanted), synset, have, need, got, nbytes / 1e6, secs))

    elapsed = time.time() - t_start
    still = [
        "%s/%s" % (syn, fn)
        for syn, fns in wanted.items() for fn in fns
        if not os.path.exists(os.path.join(dest, syn, fn))
    ]
    print("\nFetched %d files, %.2f GB transferred, %.0f s"
          % (got_total, bytes_total / 1e9, elapsed))

    if still:
        print("Still missing %d file(s)." % len(still))
        with open(args.manifest, "w") as f:
            for rel in sorted(still):
                f.write(rel + "\n")
        print("Manifest rewritten with the outstanding files; re-run to retry.")
        sys.exit(1)
    print("All manifest files present. Now run code/check_data.py to confirm.")


if __name__ == "__main__":
    main()
