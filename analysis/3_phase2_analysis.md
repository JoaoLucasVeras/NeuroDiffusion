# Phase 2: EEG Representation Learning (Masked Brain Modeling)

To turn raw EEG brainwaves into something a stable diffusion model can understand, DreamDiffusion uses a **Masked Autoencoder (MAE)**. 

### 🧠 AI Concepts Explained Simply
Before looking at the code, let's understand the AI magic happening here:
*   **Self-Supervised Learning**: Imagine trying to learn a language by reading a book with random words blacked out. By staring at the surrounding words, you eventually guess the missing words. You teach yourself without a teacher. This is how the AI learns brainwaves here—we don't tell it "this brainwave means dog"; we just hide brainwaves and say "guess the missing part".
*   **Transformer Encoder**: Think of this as a highly skilled summarizer. It looks at a sequence (like a sentence, or in this case, a timeline of brainwaves) and figures out which parts are paying "attention" to each other to form a complete thought. It compresses the full brainwave into a dense summary (called an **embedding**).
*   **Autoencoder**: A system with two parts: an **Encoder** that compresses data into a smaller summary, and a **Decoder** that tries to recreate the original data from that short summary. If the recreated data looks exactly like the original, we know the Encoder is doing a perfect job summarizing.

---

## 1. `sc_mbm/mae_for_eeg.py` (The Core Architecture)
This file defines the neural network. Because EEG data is a 1D time-series (a single line waving up and down over time), the architecture is slightly different than image models.

*   **`PatchEmbed1D` (Lines 11-29)**:
    *   *Simple terms*: We don't feed the brainwave to the AI all at once. We chop the 512-millisecond recording into 128 tiny "patches" (chunks of time). 
    *   Each patch is translated (projected) into a list of 1024 numbers (`embed_dim`) so the AI has enough "room" to store complex patterns about that tiny split second.
*   **`MAEforEEG` (Lines 31-335)**:
    *   **The Encoder (`forward_encoder`)**: Takes the chopped-up patches. It adds a "Positional Embedding" (a mathematical timestamp) so the AI knows which chunk happened first, second, etc.
    *   **`random_masking` (Lines 164-200)**: *The most critical part.* It generates random noise to select a subset of patches to keep. The rest are completely hidden (masked) from the network. In Stage 1, it hides 10% of the brainwave. In Stage 2, it hides 50%.
    *   The unmasked patches are passed through 24 layers of the Transformer Encoder, deeply analyzing the surviving signals.
    *   **The Decoder (`forward_decoder`)**: It takes the summarized signal from the encoder, inserts blank `[MASK]` placeholders where the hidden data used to be, and tries to guess what the missing brainwaves looked like using 8 Decoder layers.
    *   **The Loss (`forward_loss`)**: The AI is penalized based on how "wrong" its guesses were compared to the actual hidden brainwaves (Mean Squared Error). Over 500 epochs (training cycles), it gets very good at guessing, which proves it deeply understands human brainwaves.
*   **`eeg_encoder` (Lines 337-426)**:
    *   Once the training is done, we throw away the Decoder. We no longer need the AI to guess missing brainwaves. We only keep the Encoder, which is now an expert at turning raw brainwaves into pure, summarized mathematical thoughts (embeddings).

## 2. `sc_mbm/trainer.py` (The Training Loop)
This script acts as the "Gym Coach" that forces the AI to train.
*   **`NativeScalerWithGradNormCount`**: A trick to make the AI train using less computer memory (Mixed Precision) without forgetting what it learned (preventing underflow).
*   **`train_one_epoch`**:
    *   Iterates through the dataset.
    *   Passes the EEG `samples` into the model.
    *   Calculates the penalty (`loss`) and mathematically tweaks the AI's brain (weights) using an `optimizer` so it performs better next time.
    *   **Pearson Correlation Calculation**: It calculates how closely the fake, predicted brainwave matches the real brainwave. This gives the human scientists a score to track how smart the AI is getting.

## 3. `sc_mbm/utils.py`
Helper functions, primarily math operations.
*   **`get_1d_sincos_pos_embed`**: Generates the sine and cosine waves (think of mathematical wave patterns) used to stamp standard timestamps onto the brainwave data.
*   **`adjust_learning_rate`**: The AI learns fast at first, and then slows down to make tiny, careful adjustments at the end (Cosine Learning Rate Decay).
