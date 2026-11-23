# Plan

Milestones and done criteria, written before the work they describe (M4 onward) or recorded after the
fact (M1-M3.1, marked). Criteria are not changed after results are seen; a new criterion goes into a new
milestone. Status is updated only with a link to evidence.

| Milestone | Release | Done when | Status |
|---|---|---|---|
| M1 Scaffold | v0.1.0 | CI green on `main`; tested chunker | Done (recorded after the fact) - #1 |
| M2 Evaluation harness + BM25 | v0.2.0 | metrics tested against hand-computed values; BM25 numbers from a real run in `results/` | Done (recorded after the fact) - #4, #5, `results/m2_bm25.json` |
| M3 Dense + hybrid | v0.3.0 | pre-registered encoders compared on validation; selected systems run once on test incl. passage-unseen subset | Done (recorded after the fact) - #12, #16, `results/m3_dense.json` |
| M3.1 Ablations | v0.3.1 | segmentation and RRF questions answered under a pre-registered rule | Done (recorded after the fact) - #18, #19, `results/m3_ablation.json` |
| M4 Reranking | v0.4.0 | protocol `docs/protocols/m4_rerank.md` merged before the run; reproduction gate passes; all rerankers reported on validation; decision rule applied on test once; report `docs/reports/v0.4.0.md` | Done - #23, #27, `results/m4_rerank.json`, ADR-0005 (no reranker) |
| M4.1 Cheaper reranking (optional) | v0.4.1 | new protocol: bge-reranker-v2-m3 on the top 10 / 20 under the same 300 ms p95 limit | Planned |
| M5 Answer generation + agent | v0.5.0 | protocol merged first; end-to-end answer quality (EM/F1 on ViQuAD answers, faithfulness on a checked sample) for no-retrieval vs RAG vs agent, with cost and latency; downstream check that the M3/M4 retriever choice holds end to end | Planned (needs: which LLM backend - local Ollama, Kaggle-hosted open model, or an API) |
| M6 Serving + release | v1.0.0 | FastAPI service and Docker image run from a clean clone; README quick start verified; final report | Planned |

Out of scope unless requested: open-domain distractor corpus, encoder fine-tuning (would need a split
that excludes passages shared with test, see ADR-0001).
