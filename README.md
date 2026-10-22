# NeuroDiffusion: Generating High-Fidelity Images from Active Imagination

## Overview
**NeuroDiffusion** is an open-source deep learning framework designed to reconstruct high-fidelity images directly from human brainwaves. 

Developed by the AI/ML Club at San Jose State University, this project represents a specific paradigm shift in brain-computer interface (BCI) generative AI. While current state-of-the-art models focus on decoding *passive visual stimuli* (brainwaves recorded while a subject actively looks at a picture on a screen), NeuroDiffusion is engineered to decode **active cognitive recall**—translating voluntary visual imagery into the semantic latent space of generative AI.
## Where things are

```
NeuroDiffusion/
├── studies/        the research, one folder per study, each with its own README
│   ├── 1_dreamdiffusion/      DreamDiffusion on imagined EEG (finished)
│   ├── 2_imagery_decoding/    can imagined pictures be read honestly? (main study, 6 experiments)
│   └── 3_anchored_imagery/    decoding an imagined change to a seen shape (designed, needs hardware)
├── shared/         the toolbox every study imports: data, decoding, generation, evaluation
├── cluster/        one-time setup on the SJSU HPC: environments, downloads, data conversion
├── docs/           the project journal and guides
└── research/       each teammate's own notes and write-ups
```

A **study** is one research question with its own approach; an **experiment** is one test
inside a study, with one result. [studies/README.md](studies/README.md) explains how to add
either.

## Current status

Updated 22 October 2026. Full detail, with methods and controls, is in the
[project documentation](docs/journal.html).

| | |
|---|---|
| **Seeing decodes** | Generated images from EEG recorded while *looking* beat a shuffled control by 4.3 standard errors ([study 1](studies/1_dreamdiffusion/)). |
| **Imagining is faint** | On a 22-person dataset, only imagined *shapes* decode, at about 1.05 to 1.1 times chance, across days ([study 2](studies/2_imagery_decoding/)). Stronger decoders did not lift it. |
| **A published claim did not hold** | The dataset's paper reports 75.8% for imagined animals; the authors' own script gives 42.8% with real labels and 41.7% with scrambled ones. |
| **Next** | A new kind of recording, where people imagine a *change* to a shape they just saw ([study 3](studies/3_anchored_imagery/)). Designed and simulated; waiting on EEG hardware and ethics approval. |

## Documentation

- **[Project documentation](docs/journal.html)**: the main reference, readable start to finish, with a glossary
- [Codebase guide](docs/CODEBASE.md): every folder and script, the data format, the rules every result follows
- [Getting the data](docs/DATA.md): where to download it and how to build the tensors
- [Cluster runbook](docs/RUNBOOK.md): submitting and monitoring jobs
- [Datasets and papers](docs/RESOURCES.md), each with a verdict
- [How to contribute](CONTRIBUTING.md): fork, branch, pull request

## Setup

```sh
conda env create -f cluster/env.yaml
conda activate neurodiffusion
```

Run every script from the repository root, e.g.
`python shared/decoding/classical_baseline.py --help`. Scripts find `datasets/` and
`results/` themselves. Datasets and model weights are not in git; see
[DATA.md](docs/DATA.md).

## The Foundation & Proof of Concept
This framework is built upon the convergence of two major breakthroughs in BCI and machine learning:

1. **The Proof of Concept (The Science):** We rely on the foundational findings from the 2022 PLOS One paper, *"Improving classification and reconstruction of imagined images from EEG signals"* (Shimizu & Srinivasan). This research scientifically validated the "Imagination Window," proving that neural activity generated during active imagination can be successfully isolated, classified, and reconstructed.
2. **The Architecture (The Baseline):** We utilize **DreamDiffusion** (Bai et al., 2023) as our foundational codebase. DreamDiffusion pioneered the use of temporal Masked Autoencoders (MAE) and Latent Diffusion models (Stable Diffusion) to generate images from EEG signals, but it relied heavily on passive perception data. 

**Our Innovation:** NeuroDiffusion bridges the gap between these two works. We are taking the scientifically validated concept of decoding active imagination (from the PLOS One research) and engineering it into a state-of-the-art Latent Diffusion pipeline (an upgraded DreamDiffusion architecture).

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
