# Phase 4: Inference & Evaluation

This is the final stage of the DreamDiffusion pipeline. After the model has been trained to map EEG signals horizontally to Stable Diffusion latent conditioning, it needs to be evaluated. This phase covers how a single EEG signal generates an image, and mathematically, how we determine if the generated image is "good".

### 🧠 AI Concepts Explained Simply
When you ask an AI to draw "a dog", and it draws a Husky, but you were thinking of a Poodle, is the AI right or wrong? Evaluating AI art is tricky because art is subjective. Scientists created mathematical formulas to score AI images like a report card.
*   **Pixel Grading (MSE/PCC)**: The dumbest way to grade an image. It just overlaps the real photo and the AI photo and subtracts the pixel colors. If the AI drew the dog 1 inch to the left, this score thinks the AI failed completely.
*   **Human-Eye Grading (LPIPS/SSIM)**: These formulas try to grade like a human. They look at the "structure" of the image (edges, blobs, lighting) instead of raw pixels. If the AI draws a Husky instead of a Poodle, the structure is still "dog-like", so it gets a passing grade.
*   **The Gold Standard (FID)**: Frechet Inception Distance. It uses *another* AI to look at 10,000 real photos and 10,000 AI photos. It measures how "fake" the AI photos feel statistically compared to reality. Lower scores mean it looks more real.
*   **Semantic Grading (ViT Accuracy)**: The smartest way to grade. It uses a massive AI to classify what is inside both pictures. If both pictures classify as "Dog", the AI successfully read your mind, even if the colors and shapes are different!

---

## 1. `gen_eval_eeg.py` (The Inference Script)
This is the script used when the training is entirely finished. It doesn't learn anymore; it just performs the magic trick.
*   **Initialization (Lines 72-108)**: 
    *   Loads the fully trained "brain" (`args.model_path`).
    *   Formats the test dataset (brainwaves it has never seen before).
*   **Generation (Lines 110-131)**:
    *   It wraps the loaded weights in an inference-only class called `eLDM_eval`.
    *   It takes a brainwave, generates complete static, and runs 250 denoising passes (`ddim_steps`).
    *   Finally, it converts the output into `.png` pixel arrays and saves them to the disk so human researchers can visually compare the original photo the person looked at against the photo generated purely from their brainwaves.

## 2. `eval_metrics.py` (The Quantitative Graders)
While looking at images is fun, scientists need hard numbers to get papers published. This file computes the 5 different mathematical report cards.

*   **`mse_metric` (Line 18)**: Mean Squared Error. The "dumb" pixel-by-pixel difference calculation.
*   **`pcc_metric` (Line 21)**: Pearson Correlation Coefficient. Measures the linear correlation between the raw pixels.
*   **`ssim_metric` (Line 24)**: Structural Similarity Index. Grades how similar the *patterns* (edges, luminance) of the two images are, ignoring exact pixel matches.
*   **`psm_wrapper` (Lines 30-44)**: Learned Perceptual Image Patch Similarity (LPIPS). 
    *   It passes both the generated image and the original image through an old AI called `AlexNet`. 
    *   If both images trigger the exact same "artificial neurons" inside AlexNet, it means they "look" perceptually identical to a human.
*   **`fid_wrapper` (Lines 46-56)**: Frechet Inception Distance.
    *   It passes the images through an `InceptionV3` classifier and measures the statistical distance (Frechet distance) between the blur, noise, and textures of real images vs the generated images. A lower FID score is better.
*   **`get_n_way_top_k_acc` (Lines 124-146)**: $N$-way Top-$K$ Accuracy.
    *   This is the "Semantic" test. It passes both the generated image and the true image through a massive pre-trained Vision Transformer (`ViT_H_14`).
    *   It asks the ViT to name the object. If the generated image is named the *exact same category* as the true image (e.g., they both classify as "Airplane"), the model scores a point. This proves the EEG successfully captured the core *meaning* of the brainwave.

---
**Summary of the End-to-End Pipeline:**
1. **Target Image** -> Subject's Brain -> **Raw EEG recorded**.
2. **Phase 1 (Data Prep)**: Data is cleaned, sliced to 512 milliseconds, and loaded.
3. **Phase 2 (MBM / Pre-training)**: Raw EEG is heavily masked. A self-supervised AI learns to compress the EEG into a 1024-number summary by practicing guessing the missing brainwaves.
4. **Phase 3 (Latent Finetuning)**: A pre-trained Stable Diffusion model has its "text steering wheel" ripped out and replaced with our EEG summary. It learns to draw the meaning of the brainwave.
5. **Phase 4 (Inference)**: A new EEG signal is fed into the system. Stable diffusion generates an image. `eval_metrics.py` grades the generated image against reality.
