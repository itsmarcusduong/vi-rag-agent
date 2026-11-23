# Protocol M5: RAG answer generation with open LLMs (issue #29)

Written and merged before any run (ADR-0004). Not to be changed after results exist.

## Questions
1. Which free, open-weight LLM that fits a Kaggle T4 answers Vietnamese ViQuAD questions best from
   retrieved passages?
2. How much does retrieval help over answering without context (closed-book)?
3. Downstream check of ADR-0003: does the hybrid retriever, better than BM25 on ranking metrics, also give
   better answers?

## Fixed settings
- Data: ADR-0001. Questions: answerable only. Gold answers: all `answers.text` of the question; EM and F1
  take the maximum over gold answers.
- Samples (to fit the GPU budget), drawn once with `numpy.random.default_rng(0)` without replacement:
  1,000 validation questions (selection and ablations) and 2,000 test questions (decision). The test
  sample is also reported on its passage-unseen-in-train subset.
- Retrieval: ADR-0003 hybrid (BM25 + e5-base, RRF k = 10) and BM25 alone; no reranker (ADR-0005).
- Generation: 4-bit NF4 quantisation (bitsandbytes 0.50.2, float16 compute) for every model, so models
  are compared at the precision used for local deployment; `transformers==5.24.0`; thinking disabled
  (`enable_thinking=False` in the chat template); greedy decoding; at most 48 new tokens; one fixed prompt.
- Prompt (system), with passages numbered in rank order and truncated to 1,200 characters each:
  > Bạn là trợ lý hỏi đáp tiếng Việt. Chỉ dùng thông tin trong các đoạn văn được cung cấp. Trả lời bằng một
  > cụm từ ngắn, trích nguyên văn từ đoạn văn nếu có thể, không giải thích.
  Closed-book system prompt: "Bạn là trợ lý hỏi đáp tiếng Việt. Trả lời bằng một cụm từ ngắn, không giải thích."
- Answer normalisation for EM/F1: NFC, lowercase, remove punctuation, collapse whitespace; F1 over syllable
  tokens (SQuAD-style, without English article removal).
- Uncertainty: paired cluster bootstrap over passages (questions about the same passage are resampled
  together), 2,000 resamples, 95% percentile interval.

## Reproduction gate
Before any generation, the hybrid retriever is rebuilt and its MRR@10 on the full validation set must equal
`results/m3_ablation.json` (0.8054) within 5e-5; otherwise the run stops with status FAILED.

## Stage A: model selection (validation sample, hybrid top-3)
| ID | Model | License [DOCUMENT: model card, 2026-11-24] | Size |
|---|---|---|---|
| L1 | `Qwen/Qwen3.5-4B` | Apache-2.0 | ~5B |
| L2 | `google/gemma-4-E4B-it` | Apache-2.0 | 8B incl. embeddings (4.5B effective) |
| L3 | `Qwen/Qwen3.5-9B` | Apache-2.0 | 9.7B |

Each model also answers closed-book on the same sample. A model that fails to load or run is recorded
under `failures` and excluded. Selection: highest F1 with hybrid top-3; if the paired F1 difference between
the best and the next model has a CI containing 0, the smaller model is selected.

## Stage B: context ablation (validation sample, selected model; one factor per variant)
| Variant | Context |
|---|---|
| B0 | hybrid top-3 (from stage A) |
| B1 | hybrid top-1 |
| B2 | hybrid top-5 |
| B3 | BM25 top-3 (downstream check of ADR-0003) |
| B4 | gold passage only (oracle; upper bound, not a deployable system) |
The number of passages k for the final system is chosen among B0-B2 by F1, ties (CI contains 0) to the
smaller k.

## Decision rule (test sample, run once)
Run on the test sample: closed-book, BM25 top-k, hybrid top-k with the selected model and k.
- RAG is adopted if hybrid top-k beats closed-book on F1 with a paired CI above 0.
- ADR-0003 is confirmed downstream if hybrid top-k beats BM25 top-k on F1 with a CI above 0; if the CI
  contains 0, the report states that the ranking gain did not translate into answer quality, and the
  retriever stays (it is not worse) pending a larger sample.
Both outcomes are recorded in an ADR.

## Reported
All stage A and B variants on validation (EM, F1, CIs, paired differences, mean generated tokens,
throughput, peak GPU memory, load failures); test results for the three decision systems on all sampled
test questions and on the unseen-passage subset; 30 sampled answers per variant (question ID, gold,
prediction) for error analysis; transformers/bitsandbytes/torch versions and model revisions.

## Not measured here
Faithfulness of free-form answers, unanswerable questions, agentic tool use (later milestones).

## Budget
About 16,000 generations [INFERENCE: estimated 1-2 T4 hours; not measured]. The workflow fails fast if the
kernel stays queued and can fetch the output later.
