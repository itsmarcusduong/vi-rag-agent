# Evaluation protocol

Rules fixed **before** running experiments, so results cannot be tuned after the fact.

## Data
- Public datasets only. Record name, URL, version/commit, license and any restriction here before use.
  License status of every candidate dataset: **not yet verified**.
- Split once, with a fixed seed, into train / dev / test. Tune on dev only. Report test once per final config.
- No test question or its gold passage may appear in the retrieval index used for tuning.

## Retrieval metrics
- recall@k (k in 1, 5, 10), MRR, nDCG@10, plus p50 / p95 latency per query on stated hardware.
- Every metric function has a unit test against hand-computed values.

## Experiments
- One config file per run; results written to `results/<experiment>.json` with config, git commit, seed and date.
- Always compare against the previous best and a trivial baseline (BM25).
- Report ablations (what each component adds), not only the final system.
- With small test sets, report a confidence interval (bootstrap) rather than a bare point estimate.

## Generation and agent
- Answer quality: exact match / F1 where gold answers exist; faithfulness (answer supported by retrieved text) and
  citation correctness on a manually checked sample.
- Compare agent vs plain RAG on the same questions, with cost (tokens) and latency.

## Reporting
- README numbers are copied from `results/*.json`, never typed from memory.
- State limitations: domain of the data, size of the test set, hardware.
