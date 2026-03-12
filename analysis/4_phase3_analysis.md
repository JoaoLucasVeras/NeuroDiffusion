# Phase 3: Latent Diffusion & Fine-Tuning

Once the Masked Autoencoder (Phase 2) is trained, we have an encoder that can extract a meaningful summary of a brainwave. However, Stable Diffusion is designed to accept *Text* descriptions, not brainwaves. This phase bridges that gap.

### 🧠 AI Concepts Explained Simply
*   **Stable Diffusion**: An AI that generates images by starting with complete television static (random noise). It denoises the static step-by-step until an image forms.
*   **Cross-Attention (The Steering Wheel)**: As Stable Diffusion clears the static, it needs to know *what* to draw. Cross-attention is the mechanism where the AI painting the image "looks over" at your text prompt (or in our case, the brainwave summary) and uses it as a guide. "Oh, the text says dog, let's make these pixels brown."
*   **CLIP Space**: OpenAI created a model called CLIP that reads an image and a text prompt and puts them in the same "mathematical room". The text "Golden Retriever" and a photo of a Golden Retriever sit exactly next to each other in this room. DreamDiffusion forces the brainwave to also sit in this exact same room!
*   **Fine-Tuning**: Stable Diffusion already knows what the world looks like. We don't want to destroy its ability to draw fur, eyes, and trees. We just want to "fine-tune" its steering wheel (cross-attention) so it listens to brainwaves instead of text.

---

## 1. `dc_ldm/ldm_for_eeg.py` (The Modified Diffusion Architecture)
This file defines the code required to rip out the text-reader in Stable Diffusion and replace it with our brainwave-reader.

*   **`cond_stage_model` (Lines 29-90)**:
    *   In a normal Stable Diffusion model, this part is the text reader. Here, they replace it entirely with the pre-trained `eeg_encoder` from Phase 2.
    *   **`self.mapping` (Translating to CLIP Space)**: The brainwave summary is 1024 numbers long. CLIP text summaries are exactly 768 numbers long. This small neural network acts as a translator, squishing the 1024 brain numbers into 768 numbers so they perfectly mimic text.
    *   **Aligning with Images (`get_clip_loss`)**: To make sure the translation worked, the code checks the brainwave against the *actual* image the subject was looking at. If the translating network put the brainwave in the same mathematical room as the image, it's correct!
    *   **`self.dim_mapper`**: Stable Diffusion 1.5 expects control signals to be exactly 1280 numbers long, so this stretches the final translation to fit the exact plug-in size required by Stable Diffusion.
*   **`eLDM` (Lines 93-231)**:
    *   This is the massive wrapper class for the entire Stable Diffusion framework.
    *   **Initialization (Lines 95-133)**: It loads the famous Stable Diffusion `v1-5-pruned.ckpt` weights. It then surgically swaps the original text-reader with our brainwave-reader.
    *   **`finetune` method (Lines 135-170)**: 
        *   Crucial step: `self.model.freeze_first_stage()`. This locks the core image-drawing capabilities so we don't accidentally make the AI forget how to draw.
        *   It only optimizes the conditional encoders (`train_cond_stage_only = True`). It strictly teaches the "steering wheel" how to interpret brainwaves.
    *   **`generate` method (Lines 173-231)**: The part that actually draws. It takes a brainwave, generates pure static, and loops 250 times (`ddim_steps`), slowly turning the static into the image that the person was thinking of.

## 2. `eeg_ldm.py` (The Executable Training Script)
This is the script the researcher physically runs on their server.
*   **Metric Calculation (`get_eval_metric`)**: While training, it generates test images and mathematically compares them to the true image the human looked at, giving the robot a report card.
*   **Data Augmentations**: The training images are randomly cropped and resized to prevent the AI from memorizing exact pixel layouts.
*   **Orchestration**: 
    *   It creates the `eLDM` (the AI model).
    *   It starts the PyTorch Lightning Trainer (the engine that runs the massive loops over days of training).
*   **Generation (Lines 71-94)**: After training, it grabs the test dataset (brainwaves the AI has never seen before), generates the images, and saves `.png` files to the hard drive so humans can be amazed by the results.
