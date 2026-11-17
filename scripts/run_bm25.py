"""M2 baseline: BM25 passage retrieval on UIT-ViQuAD 2.0.

Writes ``results/m2_bm25.json``. Parameters are fixed in advance (no tuning), so the test split is
reported alongside validation. Run: ``python scripts/run_bm25.py`` (needs network and the
``retrieval`` and ``data`` extras).
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi_rag_agent import __version__
from vi_rag_agent.bm25 import BM25Params, BM25Retriever
from vi_rag_agent.data import HF_DATASET, load_viquad
from vi_rag_agent.metrics import bootstrap_ci, evaluate
from vi_rag_agent.text import tokenize

OUT = Path("results/m2_bm25.json")
EVAL_SPLITS = ("validation", "test")
DEPTH = 100  # retrieve deeper than the metric cut-offs to locate gold passages for error analysis
KS = (1, 5, 10, 20, 100)


def git_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gold_rank_buckets(run: dict, qrels: dict) -> dict[str, int]:
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-100": 0, ">100 or no lexical match": 0}
    for qid, rel in qrels.items():
        gold = next(iter(rel))
        ranking = run.get(qid, [])
        rank = ranking.index(gold) + 1 if gold in ranking else None
        if rank == 1:
            buckets["1"] += 1
        elif rank is not None and rank <= 5:
            buckets["2-5"] += 1
        elif rank is not None and rank <= 10:
            buckets["6-10"] += 1
        elif rank is not None:
            buckets["11-100"] += 1
        else:
            buckets[">100 or no lexical match"] += 1
    return buckets


def query_length_vs_hit(queries: dict, per_query: dict) -> dict:
    """Mean question length (syllables) for hits and misses at 10: are misses short questions?"""
    hits = [len(tokenize(queries[q])) for q, s in per_query.items() if s["hit@10"] == 1.0]
    misses = [len(tokenize(queries[q])) for q, s in per_query.items() if s["hit@10"] == 0.0]
    mean = lambda xs: round(statistics.fmean(xs), 2) if xs else None
    return {
        "hit@10_mean_len": mean(hits),
        "miss@10_mean_len": mean(misses),
        "n_miss@10": len(misses),
    }


def main() -> None:
    started = time.perf_counter()
    bench = load_viquad()
    load_s = time.perf_counter() - started

    params = BM25Params(k1=1.5, b=0.75)
    doc_ids = list(bench.corpus)
    t0 = time.perf_counter()
    retriever = BM25Retriever(doc_ids, [bench.corpus[d] for d in doc_ids], params)
    index_s = time.perf_counter() - t0

    results = {}
    for split in EVAL_SPLITS:
        for answerable in (True, False):
            queries, qrels = bench.subset(split, answerable)
            key = f"{split}/{'answerable' if answerable else 'unanswerable'}"
            if not queries:
                # e.g. the HF test split carries no unanswerable questions; record, don't crash.
                results[key] = {"n_queries": 0, "note": "no questions in this subset"}
                print(key, "empty", flush=True)
                continue
            run, latencies = {}, []
            for qid, question in queries.items():
                t = time.perf_counter()
                hits = retriever.search(question, k=DEPTH)
                latencies.append((time.perf_counter() - t) * 1000)
                run[qid] = [doc for doc, _ in hits]
            evaluation = evaluate(run, qrels, ks=KS, mrr_k=10, ndcg_k=10)
            ci = {
                name: [
                    round(v, 4)
                    for v in bootstrap_ci([s[name] for s in evaluation.per_query.values()])
                ]
                for name in ("recall@5", "recall@10", "mrr@10")
            }
            results[key] = {
                **evaluation.to_dict(),
                "metrics": {k: round(v, 4) for k, v in evaluation.metrics.items()},
                "ci95_bootstrap": ci,
                "latency_ms": {
                    "p50": round(float(np.percentile(latencies, 50)), 3),
                    "p95": round(float(np.percentile(latencies, 95)), 3),
                },
                "gold_rank_buckets": gold_rank_buckets(run, qrels),
                "query_length_vs_hit": query_length_vs_hit(queries, evaluation.per_query),
            }
            print(key, results[key]["metrics"], flush=True)

    payload = {
        "experiment": "m2_bm25",
        "description": "BM25 passage retrieval on UIT-ViQuAD 2.0 (gold: the source passage)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": __version__,
        "dataset": {"hf_id": HF_DATASET, **bench.stats()},
        "retriever": {
            "name": "bm25",
            "k1": params.k1,
            "b": params.b,
            "idf": "lucene: ln(1 + (N - n + 0.5) / (n + 0.5))",
            "tokenization": "NFC, lowercase, \\w+ syllables, no stopwords, no word segmentation",
            "query_terms": "unique",
            "vocab_size": len(retriever.vocab),
            "avg_doc_len_tokens": round(retriever.avgdl, 2),
        },
        "protocol": "parameters fixed before running; no tuning; metrics are means over queries",
        "results": results,
        "timing_s": {"load": round(load_s, 2), "index": round(index_s, 3)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "runner": os.environ.get("RUNNER_NAME") or os.environ.get("VI_RAG_RUNNER", "local"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
