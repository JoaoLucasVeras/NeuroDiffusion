# Phase 4: Inference & Evaluation

This is the final stage of the DreamDiffusion pipeline. After the model has been trained to map EEG signals horizontally to Stable Diffusion latent conditioning, it needs to be evaluated. This phase covers how a single EEG signal generates an image, and mathematically, how we determine if the generated image is "good".

Here is the line-by-line breakdown of the Inference and Evaluation files:

## 1. `gen_eval_eeg.py` (The Inference Script)
This is a standard standalone evaluation script. It does not train the model; it only performs the forward pass. 
*   **Initialization (Lines 72-108)**: 
    *   Loads the pre-trained weights from `args.model_path`.
    *   Sets up the `img_transform_test` which just resizes the images to 512x512 and normalizes them.
    *   Loads the datasets (both train and test splits) via `create_EEG_dataset`.
*   **Generation (Lines 110-131)**:
    *   It wraps the loaded weights in an inference-only class called `eLDM_eval` (imported from `ldm_for_eeg.py`).
    *   It generates batches of images (usually 10 `num_samples` at a time per EEG input).
    *   It runs 250 denoising steps (`ddim_steps`) natively.
    *   Finally, it converts the output PyTorch tensors back into `.png` pixel arrays and saves them to the disk so human researchers can visually compare the generated image against what the subject was looking at.

## 2. `eval_metrics.py` (The Quantitative Graders)
While looking at images is good, researchers need hard numbers to prove a model works. This file provides 5 different mathematical ways to grade the quality of the generated AI image compared to the original "ground truth" image.

*   **`mse_metric` (Line 18)**: Mean Squared Error. A very basic pixel-by-pixel difference calculation. Not great for AI generation since the AI might generate the correct *object* but in slightly different pixel locations.
*   **`pcc_metric` (Line 21)**: Pearson Correlation Coefficient. Measures the linear correlation between the pixels of the two images.
*   **`ssim_metric` (Line 24)**: Structural Similarity Index. Rather than raw pixels, this algorithm grades how similar the *structures* (edges, patterns, luminance) of the two images are.
*   **`psm_wrapper` (Lines 30-44)**: Learned Perceptual Image Patch Similarity (LPIPS). 
    *   This is a deep learning-based metric. 
    *   It passes both the generated image and the ground truth image through a pre-trained `AlexNet`. It then compares the hidden layer feature maps. If both images trigger similar neurons in AlexNet, it means they "look" perceptually similar to a human, even if the raw pixels don't perfectly align.
*   **`fid_wrapper` (Lines 46-56)**: Frechet Inception Distance.
    *   The gold standard metric for generative AI. 
    *   It passes the images through an `InceptionV3` classifier and measures the statistical distance (Frechet distance) between the distribution of the real images vs the fake images. A lower FID score is better.
*   **`get_n_way_top_k_acc` (Lines 124-146)**: $N$-way Top-$K$ Accuracy.
    *   This is a clever test. It passes both the generated image and the true image through a massive pre-trained Vision Transformer (`ViT_H_14`).
    *   It asks the ViT to classify both images. If the generated image is classified as the *exact same category* as the ground truth image (e.g., they both classify as a "Golden Retriever"), the model scores a point. This proves the EEG successfully captured the semantic *meaning* of the brainwave.

---
**Summary of the End-to-End Pipeline:**
1. **Target Image** -> Subject's Brain -> **Raw EEG recorded**.
2. **Phase 1**: Data is cleaned, sliced, and loaded.
3. **Phase 2 (MBM)**: Raw EEG is heavily masked. A Transformer Encoder learns to compress the EEG into a 1024-dim embedding, while a Decoder learns to fill in the missing brainwaves.
4. **Phase 3 (Latent Finetuning)**: The Decoder is thrown away. The 1024-dim EEG embedding is projected into a 1280-dim space. A pre-trained Stable Diffusion model's Cross-Attention layers are fine-tuned to accept this 1280-dim vector instead of text.
5. **Phase 4 (Inference)**: A new EEG signal is fed into the system. Stable diffusion generates an image. `eval_metrics.py` grades the generated image against the original target image using SSIM, LPIPS, FID, and ViT Semantic accuracy.
