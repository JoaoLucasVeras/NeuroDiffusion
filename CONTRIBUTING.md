# Contributing to NeuroDiffusion

Everyone with access works in this repository, on their own branch. You can push branches
freely. You cannot push to `main`: that needs a reviewed pull request.

---

> **Prefer a picture?** [`docs/workflow.html`](docs/workflow.html) shows the same thing as
> diagrams: where the permission boundary sits, and what happens to a pull request after
> you open it.

## One-time setup

Accept the collaborator invitation, then clone the repository directly.

```bash
git clone https://github.com/JoaoLucasVeras/NeuroDiffusion.git
cd NeuroDiffusion
conda env create -f env.yaml
conda activate neurodiffusion
```

The data is not in the repository and never will be. [`docs/DATA.md`](docs/DATA.md) has the
OSF link and the four commands that build the tensors and splits. Finish with:

```bash
python code/check_data.py     # exits non-zero if anything is wrong
```

Run that before submitting any job. It exists because a missing stimulus file once silently
became a black square, and an entire training run learned nothing while the loss curve
looked perfectly healthy.

## Branches

| Branch | What it is |
|---|---|
| `main` | Protected. Reviewed pull requests only. This branch should always work. |
| `hpc-dev` | Integration branch. The cluster pulls from here, so it moves fast and is not guaranteed stable. |
| `<yourname>/<topic>` | Your own work, for example `aditya/ica-preprocessing`. |

**Name your branch after yourself.** We all have write access, which means anyone *can*
push to anyone else's branch. GitHub has no per-branch permissions on a personal
repository, so the naming convention is what prevents that happening by accident. Please do
not push to a branch with someone else's name on it without asking them first.

```bash
git fetch origin
git checkout -b yourname/my-topic origin/main
```

**Push early.** Your branch being visible is the point of working this way: the owner can
follow what you are doing without waiting for a finished pull request.

```bash
git push -u origin yourname/my-topic
```

## Pull requests

Open the PR against `main`. The template asks four things: what changed, why, how you
checked it, and any numbers it produced. Please fill it in — "how it was checked" is the
field that matters most on a project where it is easy to produce a convincing-looking
result that means nothing.

A review from the repository owner is required before merge. That is not a formality about
trust. Results here are easy to misread, and a second pair of eyes on a number is worth a
great deal.

## Checking your work

The **classical baseline** is the cheap instrument. Five CPU-minutes, and it tells you
whether a preprocessing or representation change helped before anyone spends a GPU-day:

```bash
python code/classical_baseline.py --subject 1 --band 8 13 --permute 200
```

Most changes worth making can be evaluated this way first.

## Reporting results

If your change produces a number, three things must travel with it:

1. **The chance level.** 9% accuracy is meaningless until the reader knows chance is 3%.
2. **The sample size**, and whether those samples are independent. Twenty windows sliced
   from one 10-second recording are *not* twenty measurements of anything.
3. **Whether it is a validation or a test figure.** Validation numbers are peaks across
   many attempts and always flatter. Say which one you are quoting.

If a result looks surprisingly good, treat that as grounds for suspicion rather than
celebration. This field has a long history of findings that turned out to be recording
artifacts, and we have already caught one of our own that way.

## What not to commit

The `.gitignore` covers these, but to be explicit:

- Datasets, in any form (`datasets/*.pth`, `.mat` files, stimulus images)
- Model weights and checkpoints (`pretrains/models/`)
- Run outputs and logs (`results/`, `exps/`, `code/logs/`)
- Credentials, tokens, or SSH keys, ever

Large files are the one mistake that is genuinely painful to undo, since removing them
later means rewriting history for everybody. If you are unsure, ask before committing
rather than after.

## Working on the cluster

Most of the queue needs no GPU. The classical baseline and the small encoder both run on a
CPU in minutes, and the project's best current result came out of that lane. Only Stage 2
training needs the SJSU HPC cluster.

If you have an account, the scripts read two environment variables so they run from
anyone's home directory:

```bash
export NEURODIFFUSION_ROOT=$HOME/NeuroDiffusion
export NEURODIFFUSION_PYTHON=$HOME/.conda/envs/neurodiffusion/bin/python
```

Four traps, each of which has already cost this project real time:

- **Submit from `coe-hpc3`, not `coe-hpc1`.** The Slurm controller is unreachable from
  hpc1 and fails with a confusing connection error.
- **Use `--gres=gpu:1` unless you need the memory.** Asking for an A100 put one job 15
  hours back in the queue while three P100s and an H100 sat idle.
- **Put `$SLURM_JOB_ID` in every run directory.** Timestamps to the second are not unique
  enough; four simultaneous jobs once collided in pairs and overwrote each other's
  checkpoints.
- **Pin the Stage 1 checkpoint with `CHECKPOINT=`.** Picking whichever pretrain ran last
  once served alpha-band input to an encoder trained on the full signal, and cost a
  GPU-day to diagnose.

Cluster work happens on `hpc-dev`. If you need code of yours running there, say so in the
pull request and it will be merged forward — the cluster does not see your branch.

More detail in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Documentation

The project documentation is a single HTML file, [`docs/journal.html`](docs/journal.html),
also published as a live page. If your change alters a result or an explanation that
appears there, update it in the same pull request: add a dated entry to the Journal
chapter, add or amend the row in the ledger, and adjust the affected chapter.

Superseded numbers stay on the page, dated, rather than being deleted. The history of what
we believed and when is part of the record.
