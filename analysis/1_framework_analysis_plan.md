# DreamDiffusion Framework Analysis Plan

To fully understand this framework down to every line of code, we need to break it down logically. The codebase follows a sequential pipeline from loading raw brainwaves to generating the final images.

Here is the plan for how we will analyze the repository file-by-file in logical order:

## Phase 1: Data & Configuration Layer
Before any machine learning happens, we must load the data and set the rules.
*   **`config.py`**: Contains all hyperparameter configurations (learning rates, batch sizes, model dimensions) for the various training stages.
*   **`dataset.py`**: Handles loading the raw EEG signals, the ImageNet subsets, and extracting the corresponding features used for training alignment.

## Phase 2: EEG Representation Learning (Masked Brain Modeling)
Before we can generate images, the AI must learn how to "understand" EEG brainwaves robustly. This is done by masking parts of the EEG signal and forcing an autoencoder to predict the missing pieces.
*   **`sc_mbm/mae_for_eeg.py`**: The core architecture of the Masked Autoencoder (MAE) designed specifically for 1D EEG time-series data.
*   **`sc_mbm/trainer.py` & `sc_mbm/utils.py`**: The training loops, loss calculations, and utility functions for training the MAE.
*   **`stageA1_eeg_pretrain.py`**: The executable script that ties the MAE and trainer together to pre-train the EEG encoder.

## Phase 3: Latent Diffusion & Fine-Tuning (Image Generation)
Once the EEG signals can be encoded, we align them with a pre-trained Stable Diffusion model so those thoughts can be rendered as images.
*   **`dc_ldm/ldm_for_eeg.py`**: A modified Latent Diffusion Model. Instead of taking text as a prompt, this model is modified to take the encoded EEG signals (aligned using CLIP) as its conditioning input.
*   **`dc_ldm/utils.py`**: Utilities specific to the diffusion process.
*   **`eeg_ldm.py`**: The main executable script responsible for fine-tuning the Stable Diffusion cross-attention layers using the EEG embeddings.

## Phase 4: Inference & Evaluation
Finally, generating the results and determining how good they are.
*   **`gen_eval_eeg.py`**: The inference script. It takes a raw EEG signal, passes it through the trained EEG encoder, and into the fine-tuned Stable Diffusion model to generate an image.
*   **`eval_metrics.py`**: Contains functions for mathematical evaluation (like Inception Score or Frechet Inception Distance) to quantitatively measure the quality of the generated images.

---
**Next Step Recommendation:** 
I suggest we start with **Phase 1** and do a line-by-line breakdown of `config.py` and `dataset.py`. Once you are comfortable with those, we will move on to Phase 2.
