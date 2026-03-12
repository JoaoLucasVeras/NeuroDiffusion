# Phase 3: Latent Diffusion & Fine-Tuning

Once the Masked Autoencoder (Phase 2) is trained, we have an encoder that can extract a meaningful embedding (a complex array of numbers) from raw EEG time-series data. However, Stable Diffusion is designed to accept *Text* embeddings (specifically from the CLIP text encoder), not brainwave embeddings. 

This phase bridges that gap by swapping the Text Encoder for our EEG Encoder and gently fine-tuning Stable Diffusion’s Cross-Attention layers so it learns to "read" the EEG embeddings just like it reads text prompts.

Here is the line-by-line architectural breakdown:

## 1. `dc_ldm/ldm_for_eeg.py` (The Modified Diffusion Architecture)
This file defines the classes required to inject the EEG representations into the Stable Diffusion pipeline.

*   **`cond_stage_model` (Lines 29-90)**:
    *   In a normal Stable Diffusion model, the `cond_stage_model` is the CLIP Text Encoder.
    *   Here, they replace it entirely with the pre-trained `eeg_encoder` from Phase 2 (Loaded at line 34).
    *   **Mapping to CLIP Space (`self.mapping`)**: If `clip_tune = True`, a small neural network maps the 1024-dimensional EEG embedding into a 768-dimensional space (which is the exact size of a CLIP text embedding).
    *   **Dimensional Mapper (`self.dim_mapper`)**: A linear projection that maps the output up to `cond_dim` (1280), which is the exact dimensionality expected by Stable Diffusion 1.5's cross-attention layers.
    *   **Aligning with Images (`get_clip_loss`)**: To ensure the EEG encoding means the same thing as the image the user was looking at, the model uses Cosine Similarity against the actual image's CLIP embeddings (`image_embeds`). 
*   **`eLDM` (Lines 93-231)**:
    *   This is the wrapper class for the entire Stable Diffusion framework.
    *   **Initialization (Lines 95-133)**: It loads the standard Stable Diffusion `v1-5-pruned.ckpt` weights. It then forcefully replaces the original `cond_stage_model` with the new one we defined above (Line 112).
    *   **`finetune` method (Lines 135-170)**: 
        *   Crucial step: `self.model.freeze_first_stage()` (Freezes the VAE).
        *   It unfreezes the whole UNet but specifically focuses on optimizing the conditional encoders (`train_cond_stage_only = True`). This forces the UNet's attention layers to learn how to interpret the EEG signals without necessarily destroying the UNet's powerful pre-trained capability to draw shapes and textures.
    *   **`generate` method (Lines 173-231)**: The inference block. It takes an EEG embedding, passes it to the `PLMSSampler` (or DDIM solver) for a specified number of denoising steps (usually 250), and decodes the latent output back into a pixel image using the VAE (`decode_first_stage`).

## 2. `eeg_ldm.py` (The Executable Training Script)
This is the script the user runs to actually orchestrate the PyTorch Lightning training process.
*   **Metric Calculation (`get_eval_metric` Lines 43-69)**: It evaluates the generated images during training using SSIM (Structural Similarity Index) and PCC (Pearson Correlation Coefficient) against the ground-truth image (the image the subject was actually looking at when the EEG was recorded).
*   **Data Augmentations (`main` function Lines 125-140)**: Image datasets are normalized and randomly cropped.
*   **Orchestration (Lines 157-170)**: 
    *   It creates the `eLDM` instance.
    *   It instantiates a PyTorch Lightning Trainer (`create_trainer`, line 220).
    *   It calls `generative_model.finetune()`, which kicks off the training loops handled internally by PyTorch Lightning.
*   **Generation (Lines 71-94 & 173)**: After fine-tuning finishes, it automatically loops through the test set, generates the images from the EEG inputs, and saves `.png` files to the output directory so the researchers can visually inspect the results.

---
**Summary of Phase 3 Data Flow:**
1. Raw EEG -> `eeg_encoder` -> 1024-dim EEG embedding.
2. 1024-dim embedding -> `mapping` -> 768-dim space mapping (aligned via Cosine Similarity with the CLIP image embedding of the ground truth image).
3. 1024-dim embedding -> `dim_mapper` -> 1280-dim conditioning vector.
4. The UNet is fed Gaussian noise and the 1280-dim conditioning vector.
5. The UNet Cross-Attention layers (which are fine-tuned) use the conditioning vector to denoise the image step-by-step.
6. The resulting latent representation is decoded by the VAE into a 512x512 RGB Image.
