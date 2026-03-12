# Phase 1: Data & Configuration Layer Analysis

This document provides a line-by-line breakdown and architectural overview of the Data and Configuration layers in the DreamDiffusion framework.

## 1. `config.py` (Hyperparameters & Setup)
This file defines configuration classes that hold all the hyperparameters, file paths, and model dimensions used across different stages of the pipeline.

### `Config_MBM_EEG` (Masked Brain Modeling)
This class configures the pre-training stage (Stage 1) where the Masked Autoencoder (MAE) learns to represent EEG signals.
*   **Training Params:** Uses a learning rate of `2.5e-4` with warmup over `40` epochs, training for a total of `500` epochs. Batch size is `100`.
*   **Model Architecture Params:** 
    *   `mask_ratio = 0.1`: 10% of the EEG signal is masked during training (the model must predict this).
    *   `patch_size = 4` and `embed_dim = 1024`: The EEG signal is broken into patches of size 4, and each patch is projected into a 1024-dimensional space.
    *   `depth = 24`, `num_heads = 16`: The transformer encoder has 24 layers and 16 attention heads.
*   **File Paths:** Defines where checkpoints should be saved (`output_path`).

### `Config_EEG_finetune`
This class configures the fine-tuning stage where the pre-trained MAE is fine-tuned for a specific downstream task (though often overridden by the Generative config).
*   Points to the datasets: `eeg_5_95_std.pth` (the EEG signals) and `block_splits_by_image_all.pth` (how the data is split for training/testing).
*   `mask_ratio` is increased to `0.5` (50%).

### `Config_Generative_Model`
This is the most critical configuration for the **Image Generation** stage (Stage 2 & Inference).
*   **Finetune Params:** Very low learning rate (`5.3e-5`) for 500 epochs to gently fine-tune the Stable Diffusion cross-attention mechanisms without destroying the pre-trained weights.
*   **Diffusion Sampling:** `num_samples = 5` (generates 5 images per EEG input) and `ddim_steps = 250` (uses 250 denoising steps for high quality).
*   `clip_tune = True`: Indicates that the CLIP text encoder is being fine-tuned or utilized alongside the EEG embeddings to better align the modalities.

---

## 2. `dataset.py` (Data Pipeline and Preprocessing)
This file handles loading the `.pth` files (which contain the PyTorch tensors of the EEG records) and preparing them for the neural networks.

### Utility Functions (Lines 1-98)
*   **`pad_to_patch_size` / `pad_to_length`**: Ensures the EEG time-series arrays match the exact dimensional requirements (e.g., divisible by the patch size like `4`).
*   **`process_voxel_ts` / `interpolate_voxels`**: Functions used specifically for fMRI/voxel data. While DreamDiffusion focuses on EEG, it's built on top of the `Mind-vis` repository (which did fMRI), so these legacy functions remain.
*   **`img_norm`**: Normalizes image tensors to a range of `[-1.0, 1.0]`, which is the standard input range required by Stable Diffusion.

### `eeg_pretrain_dataset` (Lines 108-156)
Used during Phase 1 (MBM Pre-training).
*   **Loading:** It scans a directory for `.npy` files containing raw EEG data.
*   **Length Enforcement (`self.data_len = 512`)**: DreamDiffusion requires exactly 512 time steps.
    *   If the data is longer than 512, it takes a random 512-step crop.
    *   If it is shorter, it uses `scipy.interpolate.interp1d` to artificially stretch (interpolate) the signal to exactly 512 steps.
*   **Channel Enforcement (`self.data_chan = 128`)**: It enforces exactly 128 channels. It will duplicate or crop channels as needed to force the shape to `[128, 512]`.

### `EEGDataset` (Lines 238-301)
Used during Phase 2 (Stable Diffusion Fine-Tuning) and Inference.
*   **Initialization:** Loads a single monolithic PyTorch file (`eeg_5_95_std.pth`) containing all EEG signals, labels, and corresponding image names.
*   **Filtering:** Filters the dataset to only include data for a specific subject (e.g., `subject=4`), as brain topography varies wildly between individuals.
*   **`__getitem__` (The core extraction logic):**
    1.  Extracts the raw EEG tensor and transposes it.
    2.  Crops out specific timeframes: `eeg[20:460, :]`. This removes the very beginning (baseline) and end of the recording to capture the peak visual stimuli response.
    3.  Interpolates the length to exactly 512 using `interp1d`.
    4.  **Image Handling:** If an ImageNet path is provided, it loads the actual target image the user was looking at.
    5.  **CLIP Processing:** It passes the target image through a pre-trained `AutoProcessor` (`openai/clip-vit-large-patch14`) to get the `image_raw` tensors, which will be used for CLIP alignment supervision.

### `Splitter` and `create_EEG_dataset` (Lines 302-342)
*   Because the dataset is a single large list, `Splitter` acts as a wrapper. It reads `block_splits_by_image_single.pth` which contains the exact indices of which EEG trials belong to the training set and which belong to the test set.
*   `create_EEG_dataset` packages the `EEGDataset` into train and test `Splitter` objects to be passed into PyTorch DataLoaders.
