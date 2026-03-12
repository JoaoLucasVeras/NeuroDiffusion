# Phase 1: Data & Configuration Layer Analysis

This document provides a line-by-line breakdown of the foundation logic of the DreamDiffusion framework. Before an AI can learn, it needs data fed to it in exactly the right format, and it needs "rules" dictating how it should learn. This phase handles both.

### 🧠 AI Concepts Explained Simply
*   **Hyperparameters**: If training an AI is like baking a cake, hyperparameters are the recipe. They tell the system how hot the oven should be (learning rate), how much flour to use (batch size), and how long to bake it (epochs).
*   **Tensors**: The AI doesn't understand "brainwaves" or "images". Everything is converted into a giant spreadsheet of numbers called a Tensor. 
*   **Batch Size**: The AI doesn't look at one brainwave at a time—that's too slow. It grabs a "batch" (e.g., 100 brainwaves) and studies them all simultaneously before making a single adjustment to itself.
*   **Epoch**: One full read-through of every single brainwave in the entire dataset. If we train for 500 epochs, the AI reads the entire book of brainwaves 500 times.

---

## 1. `config.py` (Hyperparameters & Setup)
This file defines configuration classes that hold all the hyperparameters, file paths, and model dimensions used across different stages of the pipeline.

### `Config_MBM_EEG` (Masked Brain Modeling)
This class configures the pre-training stage (Stage 1) where the AI first learns what an EEG signal "looks like".
*   **Training Params:** Uses a learning rate of `2.5e-4` (very small, careful steps) with warmup over `40` epochs, training for a total of `500` epochs. The batch size is `100`.
*   **Model Architecture Params:** 
    *   `mask_ratio = 0.1`: 10% of the EEG signal is hidden during training (the model must predict this missing data).
    *   `patch_size = 4` and `embed_dim = 1024`: The EEG signal is broken into tiny 4-timestep chunks, and each chunk is converted into a 1024-number list.
    *   `depth = 24`, `num_heads = 16`: The "brain" of the AI has 24 layers of depth, focusing on 16 different patterns simultaneously.

### `Config_EEG_finetune`
This class configures the fine-tuning stage where the pre-trained brain-reader is hooked up to other components.
*   Points to the datasets: `eeg_5_95_std.pth` (the brainwave spreadsheets) and `block_splits_by_image_all.pth` (which brainwaves are for study, and which are for the final test).
*   `mask_ratio` is increased to `0.5` (50% is hidden, making the test much harder).

### `Config_Generative_Model`
This is the most critical recipe for the **Image Generation** stage (Stage 2 & Inference).
*   **Finetune Params:** Very low learning rate (`5.3e-5`) for 500 epochs to gently adjust the Stable Diffusion painter without breaking its pre-learned ability to draw real-world textures.
*   **Diffusion Sampling:** `num_samples = 5` (draws 5 different random images per EEG input) and `ddim_steps = 250` (uses 250 slow strokes to paint the image for high quality).

---

## 2. `dataset.py` (Data Pipeline and Preprocessing)
This file handles loading the raw recordings from the hard drive and squeezing them into perfectly shaped Tensors for the AI to read.

### Utility Functions (Lines 1-98)
*   **`pad_to_patch_size` / `pad_to_length`**: The AI expects every brainwave to be exactly the same length. If a brainwave is too short, these functions add zeroes (silence) to the end until it fits perfectly.
*   **`img_norm`**: Images are stored as colors from 0 to 255. AI prefers numbers closer to zero. This squishes the colors into a range between `-1.0` and `1.0`.

### `eeg_pretrain_dataset` (Lines 108-156)
*   **Length Enforcement (`self.data_len = 512`)**: DreamDiffusion requires exactly 512 time steps (roughly half a second of brain activity).
    *   If the recording is longer than 512, it randomly crops out 512 steps.
    *   If it is shorter, it literally stretches the brainwave like a rubber band using math (`scipy.interpolate.interp1d`) to equal exactly 512 steps.
*   **Channel Enforcement (`self.data_chan = 128`)**: It enforces exactly 128 electrodes worth of data.

### `EEGDataset` (Lines 238-301)
Used during Phase 2 (Stable Diffusion Fine-Tuning) and final tests.
*   **Initialization:** Loads a single monolithic file (`eeg_5_95_std.pth`) containing all EEG signals and the name of the picture the person was looking at.
*   **Filtering:** Filters the dataset to only include data for a specific person (e.g., `subject=4`). Brain topography varies wildly between individuals, so AI trained on Bob's brain won't work on Alice's brain.
*   **`__getitem__` (The core extraction logic):**
    1.  Crops out specific timeframes: `eeg[20:460]`. It throws away the first few milliseconds (before the brain realizes what it's looking at) and the trailing data, capturing only the exact moment the visual cortex spikes.
    2.  Interpolates the length to exactly 512.
    3.  **Image Handling:** It loads the actual target image the user was looking at.
    4.  **CLIP Processing:** It runs the target image through a massive image-reader (`openai/clip-vit-large`) which acts as the "answer key" the AI can check against later.

### `Splitter` and `create_EEG_dataset` (Lines 302-342)
*   Because the dataset is a single large list, `Splitter` acts as a security guard. It decides which brainwaves the AI is allowed to see during "study time" (Train Split) and which brainwaves the AI is locked out from until the "final exam" (Test Split).
