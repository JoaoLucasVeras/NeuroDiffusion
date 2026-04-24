# NeuroDiffusion: Dataset Integration & Replication Guide

This guide provides a step-by-step engineering workflow for integrating the PLOS One "Imagination Window" EEG dataset into the DreamDiffusion framework. Follow these steps to replicate the environment and prep the data for training.

---

## Phase 1: Environment & Weights Setup
Before processing any data, you must establish the "factory" and acquire the model's "brains."

1.  **Clone the Repository:**
    ```bash
    git clone <this-repo-url>
    cd NeuroDiffusion
    ```

2.  **Build the Conda Environment:**
    ```bash
    conda env create -f env.yaml
    conda activate neurodiffusion
    ```

3.  **Download Stable Diffusion Weights:**
    *   **File:** `v1-5-pruned.ckpt`
    *   **Source:** [HuggingFace (RunwayML)](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt)
    *   **Destination:** Create `pretrains/models/` and place the `.ckpt` file there.
    ```powershell
    mkdir -p pretrains/models
    # Use curl or your browser to download into this folder
    ```

4.  **Dataset Procurement:**
    *   Download the PLOS One "Imagination Window" dataset.
    *   Place the files in `datasets/Imagination Experiment/` (Active Recall tasks) and `datasets/Visual Experiment/` (Baselines).

---

## Phase 2: Exploratory Data Analysis (EDA)
Verify the data structure before feeding it into the model.

5.  **Install Required Tools:**
    The dataset uses MATLAB v7.3, which requires `h5py` for Python access.
    ```bash
    pip install h5py scipy numpy
    ```

6.  **Run the Inspection Script:**
    Use `analysis/inspect_mat.py` to verify the variable names and matrix shapes.
    ```bash
    python analysis/inspect_mat.py
    ```

7.  **Verify Findings (Checkpoint):**
    Ensure your output matches these expected dimensions:
    *   **`data`**: `(125, 128, 800)` -> [Time points, EEG Channels, Trials]
    *   **`labels`**: `(800, 1)` -> ImageNet Class IDs.
    *   Note: 125 time points @ 250Hz = Exactly 500ms (DreamDiffusion's standard).

---

## Phase 3: Data Aggregation & Preparation
Combine individual subject files into a single master training set.

8.  **Aggregate Subjects:**
    Write a script to loop through all `.mat` files in `datasets/Imagination Experiment/` and stack the `data` and `labels` arrays.
    *   **Target Shape:** `(Total_Trials, 128, 125)`
    *   *Note: You may need to transpose the axes from [Time, Channel, Trial] to [Trial, Channel, Time].*

9.  **Normalization:**
    Apply Z-score normalization across channels to stabilize the voltages for the MAE encoder.

---

## Phase 4: Formatting PyTorch Tensors
The DreamDiffusion engine requires two specific `.pth` files to run.

10. **Save EEG Tensor:**
    Convert the aggregated NumPy array to a PyTorch tensor and save it.
    *   **Filename:** `eeg_5_95_std.pth`

11. **Create Split Dictionary:**
    Create a mapping dictionary (`block_splits_by_image_single.pth`) that tells the dataloader which indices in the EEG tensor correspond to which ImageNet classes.

---

## Phase 5: Training & Validation
Run the Stage 1 and Stage 2 training scripts.

12. **Update Config:**
    Point `config.yaml` to your new `.pth` files.

13. **Run Stage 1 (MAE Pre-train):**
    ```bash
    python stageA1_eeg_pretrain.py
    ```

14. **Run Stage 2 (Diffusion Prep):**
    ```bash
    python eeg_ldm.py
    ```

15. **Generate Results:**
    Run `gen_eval_eeg.py` to see the first brain-to-image reconstructions.

---

## Phase 6: HPC Migration (Scaling Up)
When moving from local dev to the SJSU HPC cluster:

### 16. Preparation
- **Initialize Git**: Ensure your repo has the newly created `.gitignore`.
- **Push to Cloud**: `git push` your logic/code to your private remote.

### 17. Large File Transfer (SCP)
Do not download weights/data via HPC login nodes. Use `scp` from your local machine:
```bash
# Transfer the formatted tensors
scp -r ./datasets/*.pth username@hpc-login.sjsu.edu:~/NeuroDiffusion/datasets/

# Transfer the Stable Diffusion weights
scp ./pretrains/models/v1-5-pruned.ckpt username@hpc-login.sjsu.edu:~/NeuroDiffusion/pretrains/models/
```

### 18. Cluster Execution
- **Request a GPU**: Use the provided `code/train_mae.sh` script.
- **Submit Job**: `sbatch train_mae.sh`
- **Monitor Logs**: Check `results/logs/` for the `res_<job_id>.txt` output.
