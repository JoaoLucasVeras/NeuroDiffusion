# DreamDiffusion Codebase Analysis Plan

- [x] 1. **Data & Configuration Layer**
  - [x] Analyze `config.py` (Hyperparameters and settings)
  - [x] Analyze `dataset.py` (EEG signal and image data loading pipelines)
- [x] 2. **EEG Representation Learning (Masked Brain Modeling)**
  - [x] Analyze `sc_mbm/mae_for_eeg.py` (Masked Autoencoder architecture for EEG)
  - [x] Analyze `sc_mbm/trainer.py` & `sc_mbm/utils.py` (Training loop and utilities)
  - [x] Analyze `stageA1_eeg_pretrain.py` (Executable script for EEG pre-training)
- [x] 3. **Latent Diffusion & Fine-Tuning (Image Generation)**
  - [x] Analyze `dc_ldm/ldm_for_eeg.py` & `dc_ldm/utils.py` (Stable diffusion modified for EEG embeddings)
  - [x] Analyze `eeg_ldm.py` (Main script fine-tuning the diffusion model with EEG data)
- [x] 4. **Inference & Evaluation**
  - [x] Analyze `gen_eval_eeg.py` (Generating images from trained checkpoints)
  - [x] Analyze `eval_metrics.py` (Scoring the generated images - Inception Score, FID, etc.)
