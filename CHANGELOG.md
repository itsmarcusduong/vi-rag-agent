# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Kaggle bridge: push a `kaggle/<job>` branch and GitHub Actions runs the job on Kaggle at that commit and commits its results (`.github/workflows/kaggle.yml`, `scripts/kaggle_job.py`).

## [0.2.1] - 2026-11-18

### Added
- Release automation: merging a PR titled `chore(release): vX.Y.Z` creates the annotated tag and the GitHub Release from this changelog (`.github/workflows/release.yml`, `scripts/release_notes.py`, tested).
- Milestone reports in `docs/reports/` (M1, M2) with a template and release procedure.

## [0.2.0] - 2026-11-18

### Added
- `vi_rag_agent.metrics`: recall@k, precision@k, hit@k, MRR@k, nDCG@k (graded) and `evaluate()` over a run, with hand-computed tests.
- `metrics.bootstrap_ci`: percentile bootstrap confidence interval over per-query values.
- `vi_rag_agent.text`: NFC normalisation and syllable tokenisation for Vietnamese.
- `vi_rag_agent.bm25`: vectorised Okapi BM25 (Lucene idf, deterministic tie-breaking), tested against a loop implementation of the formula.
- `vi_rag_agent.data`: UIT-ViQuAD 2.0 to passage-retrieval benchmark (dedup, stable IDs, answerable/unanswerable subsets).
- `scripts/run_bm25.py` and the `eval` workflow: run the M2 baseline on GitHub Actions and commit `results/m2_bm25.json`.
- M2 baseline results (BM25 on UIT-ViQuAD 2.0) in the README, with bootstrap confidence intervals and documented data caveats.

## [0.1.0] - 2026-11-18

### Added
- Project scaffold: `pyproject.toml`, ruff, pytest, GitHub Actions CI, PR and issue templates.
- `vi_rag_agent.chunking`: sentence-aware overlapping character chunker with unit tests.
