# vi-rag-agent

Vietnamese retrieval-augmented generation (RAG) and agentic question answering, built so that every
claim comes with a measurement.

> **Status: work in progress (M2).** Tested chunking, metrics and BM25 modules. Numbers in the table
> below are filled in only from real runs committed under `results/` (see
> [docs/EVALUATION.md](docs/EVALUATION.md)).

## Goals

- Measure retrieval honestly: recall@k, MRR and nDCG@k on a held-out Vietnamese question set.
- Compare BM25, dense (Vietnamese SBERT + FAISS), hybrid (reciprocal rank fusion) and cross-encoder reranking.
- Add a tool-using agent (LangGraph) and evaluate whether it beats plain RAG, including latency and cost.
- Ship it: FastAPI service, Docker image, CI, reproducible configs.

## Planned architecture

```text
question ──► query rewrite ──► retriever ──► reranker ──► LLM (OpenAI-compatible) ──► answer + sources
                               BM25 | dense | hybrid       cross-encoder              ▲
                                                                                      └── agent (LangGraph): tools, routing
```

## Results

Passage retrieval on UIT-ViQuAD 2.0: 5,317 unique passages from
184 Wikipedia articles; the gold document is the passage each question was written
on. Numbers are copied from [`results/m2_bm25.json`](results/m2_bm25.json) (run on GitHub Actions,
commit `bc8476d`); 95% intervals are percentile bootstrap over queries.

| Retriever | Split (n) | recall@1 | recall@5 [95% CI] | recall@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| BM25 (k1=1.5, b=0.75) | validation (2,653) | 0.663 | 0.856 [0.842-0.870] | 0.900 | 0.744 | 0.782 | 0.35 ms |
| BM25 (k1=1.5, b=0.75) | test (7,301) | 0.629 | 0.835 [0.827-0.843] | 0.883 | 0.717 | 0.757 | 0.35 ms |
| Dense (Vietnamese SBERT + FAISS) | | not measured | not measured | not measured | not measured | not measured | not measured |
| Hybrid (RRF) | | not measured | not measured | not measured | not measured | not measured | not measured |
| Hybrid + reranker | | not measured | not measured | not measured | not measured | not measured | not measured |

Validation uses answerable questions only. The test split as published on the Hugging Face mirror
marks every question as answerable, so the test row covers all its questions. Caveats: the corpus is
small (latency is not representative of a large index), and only one passage counts as correct even
when another passage of the same article also contains the answer, so recall is a lower bound.

## Quick start (development)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Optional extras: `.[retrieval]`, `.[agent]`, `.[api]` (added as the milestones land).

## Project layout

```text
src/vi_rag_agent/   library code (chunking today; retrieval, rerank, agent, api next)
tests/              pytest suite
docs/               roadmap and evaluation protocol
.github/            CI workflow, PR and issue templates
```

## Roadmap and reports

See [docs/ROADMAP.md](docs/ROADMAP.md). Each finished milestone has a report in
[docs/reports/](docs/reports/) (data, method, results with confidence intervals, threats to validity). Work is tracked as GitHub milestones, issues and pull requests;
releases are tagged with semantic versions (`v0.N.0` per milestone, `v1.0.0` when complete).

## Data and licenses

Corpus and evaluation data will be public datasets only. Each dataset's license is recorded in
[docs/EVALUATION.md](docs/EVALUATION.md) before it is used. Code in this repository is MIT licensed.
Parts of the design come from the author's own university coursework (Information Retrieval); nothing
is copied from third-party repositories.
