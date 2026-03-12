# Phase 2: EEG Representation Learning (Masked Brain Modeling)

To turn raw EEG brainwaves into something a stable diffusion model can understand, DreamDiffusion uses a **Masked Autoencoder (MAE)**. This stage is entirely self-supervised (it doesn't need labels or text). It works by taking an EEG signal, hiding (masking) a large portion of it, and forcing a neural network to guess what the missing pieces look like. By doing this, the network learns a deep, robust representation of brainwave patterns.

Here is the line-by-line architectural breakdown of the key files operating in this phase:

## 1. `sc_mbm/mae_for_eeg.py` (The Core Architecture)
This file defines the Transformer-based autoencoder. Because EEG data is a 1D time-series (unlike 2D images), the architecture uses 1D convolutions and sequences.

*   **`PatchEmbed1D` (Lines 11-29)**:
    *   Unlike images which are split into 16x16 squares, EEG signals are split into time "patches" using a 1D Convolution (`nn.Conv1d`).
    *   With `time_len=512` and `patch_size=4`, the 512 time steps are chunked into 128 patches.
    *   Each patch is projected into an `embed_dim` of 1024.
*   **`MAEforEEG` (Lines 31-335)**:
    *   **The Encoder (`forward_encoder`)**: Takes the patches, adds a class token (`[CLS]`), and adds Sine-Cosine Positional Embeddings so the network knows the chronological order of the patches.
    *   **`random_masking` (Lines 164-200)**: The most critical part. It generates random noise to select a subset of patches to keep (defined by `mask_ratio`, which is 10% in Stage 1 and 50% in Stage 2). The rest are discarded from the encoder to save memory and compute. There is also logic for `focus_range` to force the network to mask specific crucial timeframes of the brainwave more frequently.
    *   The unmasked patches are passed through 24 Transformer Blocks (`self.blocks`).
    *   **The Decoder (`forward_decoder`)**: Takes the encoded patches and inserts generic learnable `[MASK]` tokens into the missing empty spots. It then passes this full sequence through 8 Decoder Transformer Blocks.
    *   A linear projection (`self.decoder_pred`) attempts to reconstruct the original raw EEG values.
    *   **The Loss (`forward_loss`)**: The network is penalized using Mean Squared Error (MSE) only on the patches that were masked.
*   **`eeg_encoder` (Lines 337-426)**:
    *   This is a standalone version of the encoder *without* the decoder.
    *   Once the MAE finishes pre-training, only this class is used moving forward. The decoder is thrown away. This encoder will be what passes EEG embeddings to Stable Diffusion.

## 2. `sc_mbm/trainer.py` (The Training Loop)
This handles the logic of passing data through the MAE during the pre-training loop.
*   **`NativeScalerWithGradNormCount`**: Automatic Mixed Precision (AMP). It scales gradients so that PyTorch can train using `float16` precision without the gradients dropping to zero (underflow), which drastically speeds up training.
*   **`train_one_epoch`**:
    *   Iterates through the DataLoader.
    *   If `img_feature_extractor` is provided (It's not usually heavily used in Stage 1, but available), it can extract CLIP features from the image the subject was looking at.
    *   Passes the EEG `samples` into the `MAEforEEG` model.
    *   Calculates the gradient `loss_scaler(loss, optimizer, ...)` and steps the optimizer.
    *   **Pearson Correlation Calculation**: For logging purposes, it unpatchifies the predictions and calculates the Pearson correlation coefficient (`torch.corrcoef`) between the raw EEG and the predicted EEG to give humans a metric of how well it's learning.

## 3. `sc_mbm/utils.py`
Helper functions, primarily math operations and state management.
*   **`get_1d_sincos_pos_embed`**: Generates the sine and cosine waves used to inject positional awareness into the attention blocks.
*   **`adjust_learning_rate`**: Implements a "Half-cycle cosine learning rate decay with warmup". The learning rate starts near zero, ramps up linearly (`warmup_epochs`), and then gently curves downward following a cosine curve.
*   **`interpolate_pos_embed`**: If you load a pre-trained model but change the sequence length of the EEG data, this function uses PyTorch interpolation to stretch or squish the positional embeddings to fit the new length without breaking the learned weights.

---
**Summary of Phase 2 Data Flow:**
1. Raw EEG `(Batch, Channels=128, Time=512)` -> `PatchEmbed1D` -> Patches `(Batch, 128, 1024)`.
2. `random_masking` randomly drops most of the 128 patches.
3. The remaining patches pass through 24 Transformer Encoder layers.
4. `[MASK]` tokens are inserted back into the empty spots.
5. 8 Transformer Decoder layers attempt to recreate the dropped patches.
6. MSE Loss is calculated, and weights are updated.
