# Protocol M4: cross-encoder reranking (issue #23)

Written and merged before the run (ADR-0004). Not to be changed after results exist.

## Question
Does reranking the top of the ADR-0003 hybrid retriever with a cross-encoder improve ranking quality
enough to justify its latency, and does an English MS MARCO reranker (used in the author's coursework)
work for Vietnamese?

## Base system (held fixed)
- Data, splits, metrics: ADR-0001 (UIT-ViQuAD 2.0; validation answerable n = 2,653; test n = 7,301;
  test passage-unseen-in-train n = 5,181).
- First stage: BM25 (k1 = 1.5, b = 0.75) + `intfloat/multilingual-e5-base`, RRF k = 10, equal weights,
  depth 100 (ADR-0003).

## Candidates (one factor varies: the reranker)
| ID | Reranker | License [DOCUMENT: model card, 2026-11-22] | Size | Notes |
|---|---|---|---|---|
| R0 | none | - | - | reference |
| R1 | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | 22.7M | English, trained on MS MARCO |
| R2 | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 0.6B | multilingual; Vietnamese not explicitly listed on the card |

Same settings for R1 and R2: rerank the top 50 hybrid passages per question; input pairs
(question, passage) as raw text; max length 512 tokens; float16 on a Kaggle T4; batch 64.
Passages ranked 51-100 keep their hybrid order after the reranked 50. Ties in reranker score keep the
hybrid order.

## Reproduction gate
Before any reranking, the script recomputes R0 on validation; its MRR@10 must equal the selected hybrid
in `results/m3_ablation.json` (0.8054) within 5e-5. Otherwise the run stops with status FAILED and no
comparison is reported.

## Selection (validation only)
- Eligibility (speed constraint): the reranker's added latency per question, p95, measured on the T4
  for 50 pairs, must be at most 300 ms.
- Among eligible rerankers, the one with the highest validation MRR@10 is selected. If the paired
  ΔMRR@10 between R1 and R2 has a 95% CI containing 0, the one with the lower p95 latency is selected.
- The selected reranker must beat R0 on validation (paired ΔMRR@10 CI above 0); otherwise the
  selection is R0 and test is run for R0 only.

## Decision rule (test, run once)
Adopt the selected reranker for M5 if its paired ΔMRR@10 against R0 has a 95% CI above 0 on both test
and test-unseen. Otherwise keep R0. The decision is recorded in ADR-0005.

## Reported
- Validation, all candidates: recall@1/5/10/50, MRR@10, nDCG@10, latency p50/p95, paired Δ vs R0.
- Test and test-unseen: R0 and the selected reranker only.
- Validation unanswerable: R0 and selected (exploratory).
- Not measured here: effect on answer quality (M5).

## Budget
Estimated T4 time below 60 minutes [INFERENCE: 50 pairs x ~10k questions per reranker; not measured].
Rerankers are scored on validation first; test is encoded only for the selected one.
