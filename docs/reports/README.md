# Milestone reports

One report per milestone, named after the release that closes it (`vX.Y.0.md`). A report is written
from the results files of that milestone; every number links back to a file under `results/`.
When a release PR is merged, the release workflow links the matching report from the GitHub Release.

| Milestone | Release | Report | Headline |
|---|---|---|---|
| M1 Scaffold | v0.1.0 | [v0.1.0.md](v0.1.0.md) | tooling, CI, tested chunker |
| M2 Evaluation harness and BM25 | v0.2.0 | [v0.2.0.md](v0.2.0.md) | BM25 on UIT-ViQuAD 2.0 validation: recall@5 0.856, MRR@10 0.744 |
| M3 Dense and hybrid retrieval | v0.3.0 | [v0.3.0.md](v0.3.0.md) | BM25 + e5-base RRF on test: MRR@10 0.776 (+0.059 [+0.053, +0.065] vs BM25) |
| M3.1 Ablations | v0.3.1 | [v0.3.1.md](v0.3.1.md) | segmentation lifts vietnamese-sbert MRR@10 0.420 -> 0.510; RRF k = 10 adds +0.006 MRR on test |

## Template

```markdown
# Milestone report: MN, <title>

| | |
|---|---|
| Release | vX.Y.0 |
| Pull requests | #.. |
| Results file | results/....json |
| Run | where, code commit, date |

## 1. Objective
## 2. Data (source, license, splits, leakage checks)
## 3. Method (fixed before running)
## 4. Results (copied from results files, with confidence intervals)
## 5. Error analysis
## 6. Threats to validity
## 7. Decisions and next steps
```

## Release procedure
1. Move the `Unreleased` CHANGELOG entries under `## [X.Y.Z] - YYYY-MM-DD`.
2. For a milestone release, add `docs/reports/vX.Y.0.md` and a row above.
3. Open a PR titled `chore(release): vX.Y.Z`; squash-merge it. The `release` workflow creates the
   annotated tag and the GitHub Release (pre-release while the version is 0.x).
