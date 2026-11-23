# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `scripts/run_rerank.py` (Kaggle job `m4_rerank`, #23): reranking experiment with stage status and reproduction gate; `results/m4_rerank.json`; ADR-0005; report `docs/reports/v0.4.0.md`.
- Kaggle workflow: fail fast when a kernel stays QUEUED; `[kaggle fetch]` commits only download the existing kernel's output (#25).
- `PLAN.md`, `docs/decisions/` (ADR-0001 to ADR-0004), `docs/protocols/`: solo-project workflow with protocols merged before runs (#21).

### Changed
- Branch, commit and PR conventions (ADR-0004); PR template sections: Summary, Changes, Evidence/Checks, Self-review, Limitations.

## [0.3.1] - 2026-11-21

M3.1: ablations of the M3 retrievers. Report: `docs/reports/v0.3.1.md`.

### Added
- `scripts/run_ablation_m3.py` (Kaggle job `m3_ablation`, #18): word segmentation for vietnamese-sbert and an RRF k/weight grid.
- Kaggle workflow: a commit whose message contains the skip marker (see `docs/KAGGLE.md`) does not start a GPU run.

### Changed
- `ENCODERS["vietnamese-sbert"]` now uses word segmentation (M3 results were produced with raw text).

### Results
- Word segmentation lifts vietnamese-sbert validation MRR@10 from 0.420 to 0.510 (still below BM25 0.744).
- RRF k = 10 (equal weights) beats the M3 default k = 60 on test by +0.006 MRR@10 [95% CI +0.004, +0.007]; adopted from M4 on.

## [0.3.0] - 2026-11-21

M3: dense and hybrid retrieval. Report: `docs/reports/v0.3.0.md`.

### Results
- Hybrid BM25 + multilingual-e5-base (RRF): test MRR@10 0.776 vs 0.717 for BM25 (paired ΔMRR +0.059, 95% CI [+0.053, +0.065]); on test questions with passages unseen in train, +0.067 [+0.060, +0.074].
- e5-base alone vs BM25 is inconclusive on validation; both PhoBERT-based encoders trail BM25.

### Added
- `results/repro/m2_bm25_kaggle.json`: M2 re-run on Kaggle via the bridge; all metrics identical to the GitHub Actions run.
- Kaggle bridge: push a `kaggle/<job>` branch and GitHub Actions runs the job on Kaggle at that commit and commits its results (`.github/workflows/kaggle.yml`, `scripts/kaggle_job.py`).
- `vi_rag_agent.dense`: exact cosine search over L2-normalised embeddings (NumPy or FAISS `IndexFlatIP`, identical results incl. ties) and per-model input handling (`ENCODERS`: e5 prefixes, word segmentation for the BKAI PhoBERT bi-encoder).
- `vi_rag_agent.fusion`: reciprocal rank fusion with deterministic tie-breaking.
- `metrics.paired_difference`: paired bootstrap comparison of two systems on the same queries.
- Release workflow refuses a version that is not greater than the latest tag.
- `scripts/run_dense.py` (Kaggle job `m3_dense`): three pre-registered encoders and RRF hybrids; results in `results/m3_dense.json`, report in `docs/reports/v0.3.0.md`.

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
