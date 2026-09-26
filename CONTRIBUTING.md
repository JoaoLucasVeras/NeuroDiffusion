# Contributing to NeuroDiffusion

This project uses the **fork and pull request** model. You work in your own copy of the
repository and propose changes back. Nobody can modify anyone else's work, including by
accident, because your branches live in your repository rather than this one.

---

## One-time setup

**1. Fork the repository.** Press *Fork* at the top right of the GitHub page. You now have
`github.com/<you>/NeuroDiffusion`, which you control completely.

**2. Clone your fork** and point it at the original as `upstream`:

```bash
git clone https://github.com/<you>/NeuroDiffusion.git
cd NeuroDiffusion
git remote add upstream https://github.com/JoaoLucasVeras/NeuroDiffusion.git
git remote -v          # origin = your fork, upstream = the main project
```

That `upstream` remote is what you pull updates from. You will never push to it.

## The everyday loop

**1. Start from current `main`:**

```bash
git fetch upstream
git checkout -b my-topic upstream/main
```

Branch naming is up to you inside your own fork, but something descriptive like
`ica-preprocessing` or `alpha-input` helps when the pull request shows up.

**2. Do the work, commit as you go.**

**3. Push to your fork and open the pull request:**

```bash
git push origin my-topic
```

GitHub will print a link that opens a pull request against `JoaoLucasVeras/NeuroDiffusion`.
Target branch is `main`.

**4. Keep your fork current** while the PR is open, or before starting something new:

```bash
git fetch upstream
git rebase upstream/main        # or: git merge upstream/main
```

Forks do not update themselves. If yours has drifted a long way behind, GitHub's *Sync
fork* button on your fork's page does the same thing through the web interface.

## Pull requests

Target `main`. The template asks four things: what changed, why, how you checked it, and
any numbers it produced. Please actually fill it in — "how it was checked" is the field
that matters most in a project where it is easy to produce a convincing-looking result
that means nothing.

A review from the repository owner is required before merge. That is not about trust. It
is because results in this project are easy to misread, and a second pair of eyes on a
number is worth a great deal.

Leave *Allow edits by maintainers* ticked, which is the default. It lets the owner push a
small fix to your branch rather than sending the whole thing back over a typo.

## Reporting results

If your change produces a number, three things must travel with it:

1. **The chance level.** 9% accuracy is meaningless until the reader knows chance is 3%.
2. **The sample size**, and whether those samples are independent. Twenty windows sliced
   from one 10-second recording are *not* twenty measurements of anything.
3. **Whether it is a validation or a test figure.** Validation numbers are the peak across
   many training rounds and always flatter. Say which one you are quoting.

If a result looks surprisingly good, treat that as grounds for suspicion rather than
celebration. This field has a long history of findings that turned out to be recording
artifacts rather than brain activity. Read the leakage section of the project
documentation before getting attached to a number.

## What not to commit

The `.gitignore` covers these, but to be explicit, none of the following belong in git:

- Datasets, in any form (`datasets/*.pth`, `.mat` files, stimulus images)
- Model weights and checkpoints (`pretrains/models/`)
- Run outputs and logs (`results/`, `exps/`, `code/logs/`)
- Credentials, tokens, or SSH keys, ever

Large files are the one mistake that is genuinely painful to undo, since removing them
later means rewriting history for everybody. If you are unsure whether something belongs,
ask before committing rather than after.

## Working on the cluster

Cluster work happens on the `hpc-dev` branch of the main repository, which the cluster
pulls from directly. If you need code of yours running there, say so in the pull request
and it will be merged forward — do not expect the cluster to see your fork.

Three things that are easy to get wrong:

- **Submit jobs from `coe-hpc3`.** The Slurm controller is unreachable from `coe-hpc1`, so
  `sbatch` and `squeue` fail there with a confusing connection error.
- **Every run directory must include the job ID.** Timestamps to the second are not unique
  enough. Four simultaneous jobs once collided in pairs and silently overwrote each
  other's checkpoints, which cost us a set of test results.
- **`/home` runs close to full.** Check free space before writing a lot of samples.

## Documentation

The project documentation is a single HTML file, `analysis/journal.html`, published as a
live page. If your change alters a result or an explanation that appears there, update it
in the same pull request: add a dated entry to the Journal tab, add or amend the row in the
ledger, and adjust the affected step tab.

Superseded numbers stay on the page, dated, rather than being deleted. The history of what
we believed and when is part of the record.
