# Documentation

Start with the [project documentation page](journal.html), which is also published as a
live page and is the best single place to understand what this project is and where it
stands. It reads in order, assumes no background in EEG or machine learning, and carries
a glossary.

Everything else here is reference material.

| File | What it is | Read it when |
|---|---|---|
| [journal.html](journal.html) | The main documentation: the signal, the data, the pipeline, results, the classical baseline, current status, a dated journal, and a glossary | You want to understand the project |
| [DATA.md](DATA.md) | Where to get the data, what the raw files contain, how splits are built, and the traps in them | You are setting up, or touching the data pipeline |
| [RUNBOOK.md](RUNBOOK.md) | Cluster operations: submitting jobs, monitoring them, recovering from the failures we have already hit | You are running something on the HPC |
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) | The prioritised work queue, with the reasoning behind the ordering | You are deciding what to do next |
| [RESOURCES.md](RESOURCES.md) | Every dataset and paper the team has proposed, each with a verdict and who found it | You are evaluating a new dataset or paper |
| [SIGNAL_QUALITY_RESEARCH.md](SIGNAL_QUALITY_RESEARCH.md) | Where the imagery signal lives, verified protocol details, and what actually improves signal quality | You are working on preprocessing or representation |
| [PINBOARD.md](PINBOARD.md) | Ideas that are parked rather than rejected, each with what would unpark it | You have an idea and want to know if it was already considered |

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
