# ADR-0005: No reranker on top of the hybrid retriever (for now)

Status: Accepted (2026-11-24), decided by the rule in `docs/protocols/m4_rerank.md` (merged before the run)

## Context
M4 (`results/m4_rerank.json`) compared no reranking (R0), `cross-encoder/ms-marco-MiniLM-L6-v2` (R1) and
`BAAI/bge-reranker-v2-m3` (R2) on the top 50 of the ADR-0003 hybrid. Validation ΔMRR@10 vs R0: R1
-0.249 [-0.265, -0.233]; R2 +0.110 [+0.099, +0.121], with p95 latency 1030.63 ms against a pre-registered limit of 300 ms.

## Decision
Keep the ADR-0003 hybrid without reranking. Do not use English MS MARCO rerankers for Vietnamese.

## Consequences
- M5 builds on R0.
- R2's validation gain motivates M4.1 (fewer pairs under the same latency limit); if a configuration passes
  that protocol, a new ADR supersedes this one.
