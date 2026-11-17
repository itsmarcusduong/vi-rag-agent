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

| Retriever | recall@5 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|
| BM25 | not measured | not measured | not measured | not measured |
| Dense (Vietnamese SBERT + FAISS) | not measured | not measured | not measured | not measured |
| Hybrid (RRF) | not measured | not measured | not measured | not measured |
| Hybrid + reranker | not measured | not measured | not measured | not measured |

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

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Work is tracked as GitHub milestones, issues and pull requests;
releases are tagged with semantic versions (`v0.N.0` per milestone, `v1.0.0` when complete).

## Data and licenses

Corpus and evaluation data will be public datasets only. Each dataset's license is recorded in
[docs/EVALUATION.md](docs/EVALUATION.md) before it is used. Code in this repository is MIT licensed.
Parts of the design come from the author's own university coursework (Information Retrieval); nothing
is copied from third-party repositories.
