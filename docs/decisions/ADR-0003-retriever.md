# ADR-0003: Retriever for answer generation

Status: Accepted (recorded retrospectively on 2026-11-22; decided by the M3 and M3.1 rules)

## Context
M3 (`results/m3_dense.json`) selected BM25 + multilingual-e5-base fused with RRF on validation; on test it
beats BM25 by +0.059 MRR@10 [+0.053, +0.065]. M3.1 (`results/m3_ablation.json`) found RRF k = 10 better
than k = 60 on test by +0.006 [+0.004, +0.007], which met the pre-registered adoption rule.

## Decision
Default retriever: BM25 (k1 = 1.5, b = 0.75) + `intfloat/multilingual-e5-base` (exact cosine search),
reciprocal rank fusion with k = 10 and equal weights, depth 100.

## Consequences
- Downstream effect on answer quality is not yet measured (M5 must check it).
- M4 reranking is evaluated on top of this retriever.
