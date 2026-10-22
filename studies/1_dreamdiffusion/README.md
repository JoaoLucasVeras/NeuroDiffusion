# Study 1: DreamDiffusion on imagined EEG

**Question.** DreamDiffusion (Bai et al., 2023) generates images from EEG recorded while
people *look* at pictures. Can the same model, adapted, generate images from EEG recorded
while people *imagine* them?

**Status: finished, kept unchanged.** This is where the project started. Its code is the
original DreamDiffusion codebase with our changes, and it still runs: generator A in
[study 2](../2_imagery_decoding/) can use it with our band-power encoder
(`eeg_ldm.py --cond_encoder`).

## Result (26 September 2026)

| | |
|---|---|
| **Seeing decodes** | Generated images beat a shuffled-pairing control by **4.3 standard errors** on a stimulus-disjoint split, on images and a subject the model never saw. The pipeline is wired correctly end to end. |
| **Imagination does not** | The same pipeline on imagery EEG sits at chance (+0.5 SEM). A measured null with working controls, not a crash. |
| **Imagery signal exists** | Classical methods find it in the alpha band at roughly **3× chance** (later corrected to 3 of 4 subjects significant). |
| **A known trap** | Broadband EEG scores about twice as well as alpha within a subject, and that advantage vanishes entirely across subjects. It is electrode drift acting as a per-recording fingerprint, not neural content. |

The binding constraint was data: each person imagined each picture exactly once, giving
**33 imagery events per person**. Study 2 starts by asking where the signal was lost.

## What is here

| Path | What |
|---|---|
| `stageA1_eeg_pretrain.py` | Stage 1: masked EEG pre-training (self-supervised encoder, 85M parameters) |
| `eeg_ldm.py` | Stage 2: fine-tunes Stable Diffusion 1.5 on EEG conditioning |
| `gen_eval_eeg.py` | Generates images from a trained checkpoint |
| `config.py`, `dataset.py` | Settings and data loading |
| `sc_mbm/`, `dc_ldm/` | The masked-EEG model and the latent diffusion model (vendored) |
| `jobs/` | The cluster jobs: `train_mae.sh` (stage 1), `hpc_submit_stage2.sh` (stage 2), `hpc_submit_eval.sh` (generation and scoring) |
| `notes/` | Our analysis of the DreamDiffusion codebase, phase by phase |
| `assets/` | Figures from the original DreamDiffusion paper |

## The original design

These were the planned upgrades to the baseline when the project began:

* **Optimized Temporal Masked Autoencoder (MAE):** the base EEG encoder modified to capture the rhythms of independent thought, rather than relying strictly on the visual cortex's reaction to a screen.
* **Advanced Semantic Bridging:** contrastive loss functions in the mapping layer, so the model learns to separate distinct imagined concepts before passing the vector to the generative model.
* **Hardware Flexibility:** adapting the pipeline from a strict 128-channel dependency to 64-channel (e.g. Brain Products actiCHamp) and 32-channel wet-sensor systems (e.g. EMOTIV EPOC Flex).

The roadmap was: (1) secure the signal with research-grade recordings, (2) fine-tune the
architecture to map those recordings into CLIP space, (3) "first light", the first images
driven entirely by imagination.

## Setup and running

Stable Diffusion v1.5: download `v1-5-pruned.ckpt` from the
[official page](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

```text
pretrains/
  models/        config15.yaml, v1-5-pruned.ckpt
  generation/    checkpoint_best.pth
  eeg_pretrain/  checkpoint.pth   (pre-trained EEG encoder)
datasets/
  imageNet_images/                 (subset of ImageNet)
  block_splits_by_image_single.pth
  eeg_5_95_std.pth
```

From the repository root:

```sh
python3 studies/1_dreamdiffusion/gen_eval_eeg.py \
  --dataset EEG \
  --model_path pretrains/models/checkpoint.pth \
  --splits_path "datasets/block_splits_by_image_single.pth" \
  --eeg_signals_path "datasets/eeg_5_95_std.pth" \
  --config_patch "pretrains/models/config15.yaml"
```

On the cluster: `sbatch studies/1_dreamdiffusion/jobs/train_mae.sh`, then
`hpc_submit_stage2.sh`, then `hpc_submit_eval.sh`. All three need a GPU.
