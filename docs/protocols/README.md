# Experiment protocols

A protocol is written and merged to `main` **before** the run it governs (ADR-0004). It states:
question, candidates (one factor changed per variant), what is held fixed, the reproduction gate,
selection rule and tie rule, speed/cost constraint, decision rule for test, and what will be reported.
Changing a protocol after a run means writing a new protocol for a new run.

| Protocol | Milestone | Merged before run |
|---|---|---|
| M2-M3.1 | M2, M3, M3.1 | in issue bodies #3, #10, #18 (before this folder existed) |
| [m4_rerank.md](m4_rerank.md) | M4 | yes (this PR, before the Kaggle run) |
| [m5_generation.md](m5_generation.md) | M5 | yes (PR before the Kaggle run) |
