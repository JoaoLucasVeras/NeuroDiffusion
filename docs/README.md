# Documentation

Start with the [project documentation page](journal.html), which is also published as a
live page and is the best single place to understand what this project is and where it
stands. It reads in order, assumes no background in EEG or machine learning, and carries
a glossary.

Everything else here is reference material.

| File | What it is | Read it when |
|---|---|---|
| [journal.html](journal.html) | The main documentation: the signal, the data, the pipeline, results, the classical baseline, current status, a dated journal, and a glossary | You want to understand the project |
| [CODEBASE.md](CODEBASE.md) | A map of the code: studies and experiments, the shared toolbox, the data format, where a new experiment plugs in, and the rules every result follows | You are starting your own experiment |
| [workflow.html](workflow.html) | How to contribute, with diagrams: the three copies of the repo, the permission boundary, and what happens to a pull request | You are about to make your first change |
| [DATA.md](DATA.md) | Where to get the data, what the raw files contain, how splits are built, and the traps in them | You are setting up, or touching the data pipeline |
| [RUNBOOK.md](RUNBOOK.md) | Cluster operations: submitting jobs, monitoring them, recovering from the failures we have already hit | You are running something on the HPC |
| [RESOURCES.md](RESOURCES.md) | Every dataset and paper the team has proposed, each with a verdict and who found it | You are evaluating a new dataset or paper |
| [archive/](archive/) | September planning notes from the first dataset: the old work queue, parked ideas, and signal-quality research. Superseded by the journal, kept for the record | You want to know what was planned early on |

## Conventions

**Numbers carry their context.** An accuracy figure is meaningless without its chance
level, so every result here states chance, sample size, and whether it is a validation or
a test number.

**Superseded findings stay.** When a result is overturned, it is dated and corrected in
place rather than deleted. The record of what we believed and when is part of the work,
and it is how we avoid repeating a mistake.

**Surprisingly good results are treated as suspicious.** This field has a long history of
findings that turned out to be recording artifacts. If a number looks too good, the first
job is to try to break it. That instinct is what caught the recording-fingerprint shortcut described in
the documentation.
