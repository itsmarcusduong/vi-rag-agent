# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `vi_rag_agent.metrics`: recall@k, precision@k, hit@k, MRR@k, nDCG@k (graded) and `evaluate()` over a run, with hand-computed tests.

## [0.1.0] - 2026-11-18

### Added
- Project scaffold: `pyproject.toml`, ruff, pytest, GitHub Actions CI, PR and issue templates.
- `vi_rag_agent.chunking`: sentence-aware overlapping character chunker with unit tests.
