"""
Warm the HuggingFace cache on the LOGIN node before submitting jobs.

Compute nodes are air-gapped and the SLURM scripts export TRANSFORMERS_OFFLINE=1, so any
model not already in ~/.cache/huggingface raises the moment it is touched. That failure
would land at job startup (or worse, in the CLIP-tune loss on the first batch).

Four different classes are loaded from openai/clip-vit-large-patch14 across the codebase,
and they do not all pull the same files:

  AutoProcessor                 dataset.py            (image preprocessing)
  CLIPVisionModelWithProjection FrozenImageEmbedder   (clip_tune supervision)
  CLIPTextModel + CLIPTokenizer FrozenCLIPEmbedder    (SD1.5 conditioning)
  CLIPModel                     clip_metrics.py       (evaluation)

Run this once on the login node:
    python code/precache_models.py
"""
import sys

MODEL = "openai/clip-vit-large-patch14"


def main():
    from transformers import (AutoProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer,
                              CLIPVisionModelWithProjection)

    targets = [
        ("AutoProcessor", AutoProcessor),
        ("CLIPTokenizer", CLIPTokenizer),
        ("CLIPTextModel", CLIPTextModel),
        ("CLIPVisionModelWithProjection", CLIPVisionModelWithProjection),
        ("CLIPModel", CLIPModel),
    ]

    failed = []
    for name, cls in targets:
        try:
            cls.from_pretrained(MODEL)
            print("  [ OK ]  %s" % name)
        except Exception as e:
            print("  [FAIL]  %s -- %s: %s" % (name, type(e).__name__, e))
            failed.append(name)

    if failed:
        print("\n%d of %d failed. The login node needs internet for this step; do NOT run "
              "it inside a batch job." % (len(failed), len(targets)))
        sys.exit(1)
    print("\nAll CLIP components cached. Compute nodes can now run fully offline.")


if __name__ == "__main__":
    main()
