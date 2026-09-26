# NeuroDiffusion: Generating High-Fidelity Images from Active Imagination

## Overview
**NeuroDiffusion** is an open-source deep learning framework designed to reconstruct high-fidelity images directly from human brainwaves. 

Developed by the AI/ML Club at San Jose State University, this project represents a specific paradigm shift in brain-computer interface (BCI) generative AI. While current state-of-the-art models focus on decoding *passive visual stimuli* (brainwaves recorded while a subject actively looks at a picture on a screen), NeuroDiffusion is engineered to decode **active cognitive recall**—translating pure, closed-eye imagination into the semantic latent space of generative AI.

## The Foundation & Proof of Concept
This framework is built upon the convergence of two major breakthroughs in BCI and machine learning:

1. **The Proof of Concept (The Science):** We rely on the foundational findings from the 2022 PLOS One paper, *"Improving classification and reconstruction of imagined images from EEG signals"* (Shimizu & Srinivasan). This research scientifically validated the "Imagination Window," proving that neural activity generated during active imagination can be successfully isolated, classified, and reconstructed.
2. **The Architecture (The Baseline):** We utilize **DreamDiffusion** (Bai et al., 2023) as our foundational codebase. DreamDiffusion pioneered the use of temporal Masked Autoencoders (MAE) and Latent Diffusion models (Stable Diffusion) to generate images from EEG signals, but it relied heavily on passive perception data. 

**Our Innovation:** NeuroDiffusion bridges the gap between these two works. We are taking the scientifically validated concept of decoding active imagination (from the PLOS One research) and engineering it into a state-of-the-art Latent Diffusion pipeline (an upgraded DreamDiffusion architecture).

## Key Architectural Enhancements
To achieve this shift from passive perception to active imagination, we are implementing several critical upgrades to the baseline pipeline:

* **Optimized Temporal Masked Autoencoder (MAE):** The base EEG encoder has been modified to capture the chaotic, unique rhythms of independent thought, rather than relying strictly on the visual cortex's reaction to external light and screens.
* **Advanced Semantic Bridging:** Guided by the PLOS One findings, we have integrated customized contrastive loss functions into our mapping layer. This creates strict training guardrails, ensuring the model learns to mathematically separate distinct imagined concepts (e.g., imagining a face vs. imagining a landscape) before passing the vector to the generative model.
* **Hardware Flexibility:** To increase research accessibility, the pipeline is being adapted from a strict 128-channel dependency to support high-density 64-channel systems (e.g., Brain Products actiCHamp) and 32-channel wet-sensor systems (e.g., EMOTIV EPOC Flex).

## Project Roadmap
Our immediate milestones leading to our V1.0 release include:
1. **Securing the Signal:** Conducting internal data collection trials using research-grade, high-density EEG equipment to build a robust "imagination window" dataset.
2. **Custom Fine-Tuning:** Training our modified PyTorch architecture to map these newly recorded temporal brainwaves accurately into the CLIP semantic space.
3. **"First Light":** Generating and evaluating our first batch of high-fidelity images driven entirely by active cognitive recall.

---

## Repository Structure & Setup
*(Note: Datasets and pre-trained models must be downloaded separately and placed in the root directory).*

For our generative backend, we utilize standard Stable Diffusion v1.5. You can download the `v1-5-pruned.ckpt` file from the [official Stability AI page](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

```text
/pretrains
┣ 📂 models
┃   ┗ 📜 config.yaml
┃   ┗ 📜 v1-5-pruned.ckpt
┣ 📂 generation  
┃   ┗ 📜 checkpoint_best.pth 
┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint.pth  (pre-trained EEG encoder)

/datasets
┣ 📂 imageNet_images (subset of Imagenet)
┗  📜 block_splits_by_image_single.pth 
┗  📜 eeg_5_95_std.pth  

/code
┣ 📂 sc_mbm
┃   ┗ 📜 mae_for_eeg.py (Modified for Imagination Encoding)
┃   ┗ 📜 trainer.py
┃   ┗ 📜 utils.py
┣ 📂 dc_ldm
┃   ┗ 📜 ldm_for_eeg.py (Semantic Bridge & Contrastive Loss logic)
┃   ┗ 📜 utils.py
┃   ┣ 📂 models
┃   ┣ 📂 modules
┗  📜 stageA1_eeg_pretrain.py   (main script for EEG pre-training)
┗  📜 eeg_ldm.py                (main script for fine-tuning stable diffusion)
┗  📜 gen_eval_eeg.py           (main script for generating images)
```

## Environment Setup
Create and activate the conda environment using the provided `env.yaml` file:

```sh
conda env create -f env.yaml
conda activate neurodiffusion
```

## Inference (Running the Baseline)
To test the baseline generation using standard checkpoint data, run the following command:

```sh
python3 code/gen_eval_eeg.py \
  --dataset EEG \
  --model_path pretrains/models/checkpoint.pth \
  --splits_path "datasets/block_splits_by_image_single.pth" \
  --eeg_signals_path "datasets/eeg_5_95_std.pth" \
  --config_patch "pretrains/models/config15.yaml"
```

---

## Acknowledgements & Citations
This codebase is heavily indebted to the brilliant work of the original authors of **DreamDiffusion**, which serves as the foundational architecture for our pipeline. We also explicitly acknowledge the pioneering scientific research presented in the **PLOS One** paper, which proved the viability of reconstructing imagined images and inspired our architectural direction.

If you utilize the baseline architecture or theoretical foundations that support this project, please cite the following original papers:

**The Structural Baseline:**
```bibtex
@article{bai2023dreamdiffusion,
  title={DreamDiffusion: Generating High-Quality Images from Brain EEG Signals},
  author={Bai, Yunpeng and Wang, Xintao and Cao, Yanpei and Ge, Yixiao and Yuan, Chun and Shan, Ying},
  journal={arXiv preprint arXiv:2306.16934},
  year={2023}
}
```

**The Scientific Proof of Concept:**
```bibtex
@article{shimizu2022improving,
  title={Improving classification and reconstruction of imagined images from EEG signals},
  author={Shimizu, Hirokatsu and Srinivasan, Ramesh},
  journal={PLOS One},
  volume={17},
  number={9},
  pages={e0274847},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
