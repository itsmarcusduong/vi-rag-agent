# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `vi_rag_agent.metrics`: recall@k, precision@k, hit@k, MRR@k, nDCG@k (graded) and `evaluate()` over a run, with hand-computed tests.
- `metrics.bootstrap_ci`: percentile bootstrap confidence interval over per-query values.
- `vi_rag_agent.text`: NFC normalisation and syllable tokenisation for Vietnamese.
- `vi_rag_agent.bm25`: vectorised Okapi BM25 (Lucene idf, deterministic tie-breaking), tested against a loop implementation of the formula.
- `vi_rag_agent.data`: UIT-ViQuAD 2.0 to passage-retrieval benchmark (dedup, stable IDs, answerable/unanswerable subsets).
- `scripts/run_bm25.py` and the `eval` workflow: run the M2 baseline on GitHub Actions and commit `results/m2_bm25.json`.

## [0.1.0] - 2026-11-18

### Added
- Project scaffold: `pyproject.toml`, ruff, pytest, GitHub Actions CI, PR and issue templates.
- `vi_rag_agent.chunking`: sentence-aware overlapping character chunker with unit tests.
