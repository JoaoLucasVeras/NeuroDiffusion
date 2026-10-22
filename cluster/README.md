# Cluster setup

One-time jobs on the SJSU COE HPC. Each has been run once already; you need them only for a
fresh copy of the project or a new dataset. Day-to-day job scripts live next to their
experiment in `studies/`.

| File | Where it runs | Does |
|---|---|---|
| `env.yaml`, `requirements_hpc.txt` | anywhere | The `neurodiffusion` conda environment |
| `setup_gao_prep.sh` | coe-hpc1 (internet) | A separate `gaoprep` environment (MNE, ICLabel) for the Gao authors' cleaning |
| `setup_imagevar.sh` | coe-hpc1 | Installs diffusers and downloads generator B's model (5 GB) |
| `fetch_gao.sh` | coe-hpc1 | Downloads the Gao et al. dataset (22 zips, 12 GB) into `datasets/gao_raw/` |
| `s11_prepare_gao.sh` | `sbatch` (CPU) | Converts each Gao zip to `datasets/gao/gao_sub-XX.pt` |

The cluster workflow and the resource rules (prefer CPU, one GPU per job, at most two or
three at once) are in [docs/CODEBASE.md](../docs/CODEBASE.md#7-running-on-the-cluster) and
[docs/RUNBOOK.md](../docs/RUNBOOK.md).
