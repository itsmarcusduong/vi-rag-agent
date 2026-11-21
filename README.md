# vi-rag-agent

Vietnamese retrieval-augmented generation (RAG) and agentic question answering, built so that every
claim comes with a measurement.

> **Status: work in progress (M3).** BM25, dense (FAISS) and hybrid (RRF) retrieval with measured results. Numbers in the table
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

Passage retrieval on UIT-ViQuAD 2.0: 5,317 unique passages from 184 Wikipedia
articles; the gold document is the passage each question was written on. Numbers are copied from
[`results/m3_dense.json`](results/m3_dense.json) (Kaggle T4 GPU via the GitHub Actions bridge, commit
`ef08c5b`) and [`results/m2_bm25.json`](results/m2_bm25.json). Brackets: 95% bootstrap CI;
ΔMRR is the paired difference against BM25 on the same questions.

**Validation (answerable, 2,653 questions): all candidates.** The best dense and
hybrid systems were selected here, by MRR@10, before looking at test.

| System | recall@1 | recall@5 | recall@10 | recall@100 | MRR@10 [95% CI] | ΔMRR vs BM25 [95% CI] |
|---|---|---|---|---|---|---|
| Hybrid RRF: BM25 + e5-base | 0.712 | 0.908 | 0.947 | 0.994 | 0.798 [0.785-0.811] | +0.054 [+0.044, +0.064] |
| Dense: multilingual-e5-base | 0.656 | 0.876 | 0.920 | 0.988 | 0.749 [0.735-0.763] | +0.005 [-0.009, +0.020] |
| BM25 (k1=1.5, b=0.75) | 0.663 | 0.856 | 0.900 | 0.974 | 0.744 [0.729-0.759] | reference |
| Hybrid RRF: BM25 + BKAI | 0.637 | 0.847 | 0.904 | 0.989 | 0.729 [0.714-0.744] | -0.015 [-0.029, -0.002] |
| Hybrid RRF: BM25 + vietnamese-sbert | 0.520 | 0.762 | 0.844 | 0.979 | 0.623 [0.606-0.639] | -0.121 [-0.136, -0.106] |
| Dense: BKAI vietnamese-bi-encoder | 0.522 | 0.737 | 0.791 | 0.929 | 0.611 [0.595-0.628] | -0.133 [-0.150, -0.115] |
| Dense: keepitreal/vietnamese-sbert | 0.320 | 0.555 | 0.641 | 0.862 | 0.420 [0.404-0.437] | -0.324 [-0.343, -0.306] |

**Test: only BM25 and the selected systems.**

| Questions (n) | System | recall@1 | recall@5 | recall@10 | MRR@10 [95% CI] | ΔMRR vs BM25 [95% CI] |
|---|---|---|---|---|---|---|
| test, all (7,301) | BM25 (k1=1.5, b=0.75) | 0.629 | 0.835 | 0.883 | 0.717 [0.707-0.725] | reference |
| test, all (7,301) | Dense: multilingual-e5-base | 0.637 | 0.853 | 0.906 | 0.730 [0.721-0.738] | +0.013 [+0.005, +0.023] |
| test, all (7,301) | Hybrid RRF: BM25 + e5-base | 0.688 | 0.890 | 0.934 | 0.776 [0.768-0.783] | +0.059 [+0.053, +0.065] |
| test, passage unseen in train (5,181) | BM25 (k1=1.5, b=0.75) | 0.609 | 0.822 | 0.873 | 0.699 [0.688-0.710] | reference |
| test, passage unseen in train (5,181) | Dense: multilingual-e5-base | 0.626 | 0.854 | 0.905 | 0.723 [0.712-0.733] | +0.024 [+0.013, +0.035] |
| test, passage unseen in train (5,181) | Hybrid RRF: BM25 + e5-base | 0.675 | 0.884 | 0.930 | 0.766 [0.756-0.775] | +0.067 [+0.060, +0.074] |

Findings: hybrid BM25 + multilingual-e5-base beats BM25 on validation and on both test sets, with
paired intervals that exclude zero. e5 alone ties BM25 on validation MRR (slightly ahead on recall@10) and is slightly
ahead on test. The two
Vietnamese PhoBERT-based encoders are below BM25 here; see [the M3 report](docs/reports/v0.3.0.md) for why
that comparison may be unfair to them (confirmed in [M3.1](docs/reports/v0.3.1.md): word segmentation lifts vietnamese-sbert
from 0.420 to 0.510 MRR@10, still below BM25; tuning RRF to k = 10 adds +0.006 MRR on test). Caveats: small corpus, one gold passage per question (recall is a
lower bound), and the test split marks every question as answerable.

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
