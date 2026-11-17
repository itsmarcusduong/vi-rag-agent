# Roadmap

Each milestone ends with a tag and a GitHub Release. A milestone is closed only when its numbers come from a
real run and are committed under `results/`. Everything below is **planned**, not done.

| Milestone | Tag | Scope | Done when |
|---|---|---|---|
| M1 Scaffold | v0.1.0 | tooling, CI, templates, chunking module | CI green on `main` |
| M2 Corpus and evaluation harness | v0.2.0 | public corpus + question set, recall@k / MRR / nDCG, BM25 baseline | baseline numbers committed |
| M3 Dense and hybrid retrieval | v0.3.0 | Vietnamese SBERT embeddings, FAISS, reciprocal rank fusion, ablation table | ablation table committed |
| M4 Reranking | v0.4.0 | cross-encoder rerank; compare an English MS MARCO model with a multilingual one; latency cost | rerank vs no-rerank table committed |
| M5 Generation and agent | v0.5.0 | OpenAI-compatible LLM client, LangGraph agent with tools, faithfulness and citation checks | agent vs plain RAG table committed |
| M6 Serving and release | v1.0.0 | FastAPI, Docker, demo, final write-up | `docker run` works from a clean clone |

## Porting notes (from the author's coursework)

Source: `Truy_xuat_du_lieu/Final` (offline RAG competition), `BT5` (LangGraph agent), `BT6` (ANN benchmark).

- **Reuse the ideas, rewrite the code.** Chunking was rewritten here; the original skips a stretch of text when a
  separator sits very close to the chunk start (reproduced on a synthetic case, fixed and covered by a test).
- **Drop course-specific glue:** the teacher-server protocol and the internal LLM proxy address.
- **Reranker:** the coursework used `cross-encoder/ms-marco-MiniLM-L-6-v2`, an English-trained model, and left it
  disabled. Do not assume it helps Vietnamese; M4 measures it against a multilingual alternative.
- **Do not port** the option that replays known public answer labels (`USE_PUBLIC_ANSWER_KEY` in BT5). It is
  an evaluation shortcut and has no place in a portfolio repository.
- **BT6** contains ANN ideas (Flat vs HNSW vs IVF-PQ); its report cells are a template. Verify real outputs exist
  before quoting any number from it.

## Suggested first issues

1. M2: choose the public corpus and record its license (blocking).
2. M2: question set + gold-passage mapping, with a fixed train/dev/test split made before any tuning.
3. M2: metrics module (recall@k, MRR, nDCG@k) with unit tests against hand-computed values.
4. M2: BM25 baseline and `results/m2_bm25.json`.
