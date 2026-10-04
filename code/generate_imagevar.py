"""
Generator B: Stable Diffusion Image Variations, conditioned on predicted CLIP image
embeddings. The alternative to generator A (eeg_ldm.py --cond_encoder).

Why this exists
---------------
Generator A retrains Stable Diffusion's cross-attention to read our conditioning, on 33
stimuli. Image Variations (lambdalabs/sd-image-variations-diffusers) is Stable Diffusion
already fine-tuned by its authors to take a CLIP ViT-L/14 image embedding, the same
768-d space our encoders predict into. Nothing is trained here, so nothing can overfit,
and a failure points at the brain signal rather than at a generator that never learned.

Both generators read the same encoder checkpoint (bp_clip_encoder.py) and write the same
samples.npz, so eval_report.py scores either one, and switching costs one command.

Arms, one samples.npz each, all scored by eval_report.py
---------------------------------------------------------
  eeg      the encoder's prediction for each test window. The thing being tested.
  oracle   the true CLIP embedding of the stimulus. The ceiling: what this generator
           makes from a perfect brain signal.
  mean     the mean embedding of the stimuli, identical for every trial. What a
           prediction carrying no information at all produces.
The floor for eeg needs no generation of its own: eval_report.py re-pairs the eeg images
with other trials' stimuli.

Every arm uses the same starting noise for the same trial, so arms differ only in the
conditioning vector, and a gap between them is attributable to it.

Scale
-----
The encoder emits unit vectors; the variations model was trained on raw image_embeds,
whose norm is not 1. Every arm is rescaled to the mean norm of the real stimulus
embeddings, measured here with the model's own image encoder, so the arms differ in
direction only.

Usage
-----
    python generate_imagevar.py --model ../pretrains/sd-image-variations \\
        --encoders ../results/s9_X/bp_clip_s1.pt ../results/s9_X/bp_clip_s2.pt \\
        --out ../results/imagevar_X
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ARMS = ("eeg", "oracle", "mean")


def stimulus_path(imagenet_dir, stem):
    return os.path.join(imagenet_dir, stem.split("_")[0], stem + ".JPEG")


def load_pipe(model_dir, device, dtype):
    from diffusers import StableDiffusionImageVariationPipeline
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        model_dir, torch_dtype=dtype, safety_checker=None, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.no_grad()
def embed_images(pipe, paths, device):
    """Raw image_embeds from the generator's own encoder, preprocessed its own way."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    px = pipe.feature_extractor(images=imgs, return_tensors="pt").pixel_values
    enc = pipe.image_encoder
    return enc(px.to(device=device, dtype=enc.dtype)).image_embeds.float().cpu()


@torch.no_grad()
def generate(pipe, emb, seeds, steps, guidance, size, device):
    """The variations pipeline's own sampling, written out so the conditioning vector is
    passed in directly. Classifier-free guidance against a zero embedding, as the
    pipeline does. emb: (B, 768), already scaled. One image per row, one seed per row."""
    dtype = pipe.unet.dtype
    cond = emb.to(device=device, dtype=dtype)[:, None, :]           # (B, 1, 768)
    ctx = torch.cat([torch.zeros_like(cond), cond])
    lat_shape = (4, size // 8, size // 8)
    latents = torch.stack([
        torch.randn(lat_shape, generator=torch.Generator().manual_seed(int(s)))
        for s in seeds]).to(device=device, dtype=dtype)
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=device)
    latents = latents * sched.init_noise_sigma
    for t in sched.timesteps:
        inp = sched.scale_model_input(torch.cat([latents] * 2), t)
        eps_u, eps_c = pipe.unet(inp, t, encoder_hidden_states=ctx).sample.chunk(2)
        latents = sched.step(eps_u + guidance * (eps_c - eps_u), t, latents).prev_sample
    img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    img = ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return img.permute(0, 2, 3, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="Generator B: SD Image Variations")
    p.add_argument("--model", required=True, help="local sd-image-variations-diffusers directory")
    p.add_argument("--encoders", nargs="+", required=True, help="bp_clip_encoder.py checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="../datasets/imagination_5_95_std.pth")
    p.add_argument("--imagenet", default="../datasets/imageNet_images")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--limit", type=int, default=0, help="first N test trials only, for a smoke test")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipe(args.model, device, dtype)
    os.makedirs(args.out, exist_ok=True)

    payload = torch.load(args.dataset, map_location="cpu")
    synsets = [str(s) for s in payload["labels"]]
    label_of = {e["image"]: int(e["label"]) for e in payload["dataset"]}
    manifest = {"args": vars(args), "subjects": {}}

    for path in args.encoders:
        blob = torch.load(path, map_location="cpu")
        subj = int(blob["subject"])
        stems = list(blob["test"]["image"])
        pred = blob["test"]["pred"].float()
        if args.limit:
            stems, pred = stems[:args.limit], pred[:args.limit]
        order = list(blob["stimulus_order"])

        # the stimuli as the generator itself sees them
        true = embed_images(pipe, [stimulus_path(args.imagenet, s) for s in order], device)
        norm = float(true.norm(dim=-1).mean())
        cache = os.path.join(os.path.dirname(args.dataset), "clip_stimulus_emb.pt")
        ours = torch.load(cache, map_location="cpu") if os.path.exists(cache) else {}
        agree = None
        if ours.get("stems") == order:
            agree = float((F.normalize(true, dim=-1) * ours["emb"].float()).sum(-1).mean())
        print("subject %d | %d test trials | embedding norm %.3f | encoder agreement %s"
              % (subj, len(stems), norm,
                 "%.4f (cosine, ours vs the generator's)" % agree if agree is not None else "n/a"))
        if agree is not None and agree < 0.9:
            print("  WARNING: the generator's image encoder disagrees with the one the brain "
                  "encoder was trained against. Results for the eeg arm will be degraded.")

        sidx = {s: i for i, s in enumerate(order)}
        cond = {
            "eeg": F.normalize(pred, dim=-1) * norm,
            "oracle": F.normalize(true[[sidx[s] for s in stems]], dim=-1) * norm,
            "mean": F.normalize(true.mean(0, keepdim=True), dim=-1).expand(len(stems), -1) * norm,
        }
        gt = np.stack([np.asarray(Image.open(stimulus_path(args.imagenet, s)).convert("RGB")
                                  .resize((args.size, args.size), Image.BICUBIC)) for s in stems])
        seeds = [args.seed + 1000 * subj + i for i in range(len(stems))]

        for arm in args.arms:
            out = []
            for i in range(0, len(stems), args.batch):
                out.append(generate(pipe, cond[arm][i:i + args.batch], seeds[i:i + args.batch],
                                    args.steps, args.guidance, args.size, device))
            d = os.path.join(args.out, "s%d_%s" % (subj, arm))
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "samples.npz"),
                                gt=gt.astype(np.uint8),
                                pred=np.concatenate(out)[:, None].astype(np.uint8),
                                labels=np.array([label_of[s] for s in stems]),
                                stems=np.array(stems), synsets=np.array(synsets))
            print("  %-6s -> %s" % (arm, d))
        manifest["subjects"][subj] = {"encoder": path, "n_trials": len(stems),
                                      "embedding_norm": norm, "encoder_agreement": agree}

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("score each arm with: python eval_report.py --samples <dir>/samples.npz")


if __name__ == "__main__":
    main()
