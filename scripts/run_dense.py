"""M3: dense and hybrid retrieval on UIT-ViQuAD 2.0, compared with BM25.

Protocol (fixed in issue #10 before any run):
* Candidates: the encoders in ``vi_rag_agent.dense.ENCODERS``; hybrid = RRF (k = 60) of BM25 and a
  dense run, each truncated to depth 100.
* Every candidate is scored on validation (answerable). The best dense model and the best hybrid are
  chosen by validation MRR@10 alone.
* Test is reported only for BM25, the selected dense model and the selected hybrid, on all test
  questions and on the test questions whose passage never appears in train.
* Paired bootstrap differences against BM25 on the same queries.

Writes ``results/m3_dense.json``. Needs a GPU for reasonable speed (runs on Kaggle via the bridge).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi_rag_agent import __version__
from vi_rag_agent.bm25 import BM25Params, BM25Retriever
from vi_rag_agent.data import HF_DATASET, load_viquad
from vi_rag_agent.dense import ENCODERS, DenseRetriever, EncoderSpec, prepare
from vi_rag_agent.fusion import reciprocal_rank_fusion
from vi_rag_agent.metrics import bootstrap_ci, evaluate, paired_difference

OUT = Path("results/m3_dense.json")
DEPTH = 100
RRF_K = 60
KS = (1, 5, 10, 20, 100)
SELECT_ON = "mrr@10"
BATCH = 64


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def load_segmenter():
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize


def score(run: dict, qrels: dict) -> tuple[dict, dict]:
    """Rounded summary (with CIs) and the raw per-query scores."""
    ev = evaluate(run, qrels, ks=KS, mrr_k=10, ndcg_k=10)
    summary = {
        "n_queries": ev.n_queries,
        "n_missing_from_run": len(ev.missing_from_run),
        "metrics": {k: round(v, 4) for k, v in ev.metrics.items()},
        "ci95_bootstrap": {
            m: [round(x, 4) for x in bootstrap_ci([s[m] for s in ev.per_query.values()])]
            for m in ("recall@5", "recall@10", "mrr@10")
        },
    }
    return summary, ev.per_query


def paired(base: dict, other: dict) -> dict:
    out = {}
    for metric in ("recall@10", "mrr@10"):
        d = paired_difference(base, other, metric)
        out[metric] = {
            "mean_diff": round(d["mean_diff"], 4),
            "ci95": [round(x, 4) for x in d["ci95"]],
            "better": d["b_better"],
            "worse": d["a_better"],
            "tied": d["tied"],
        }
    return out


def truncation_rate(model, texts: list[str]) -> float:
    """Fraction of inputs longer than the model's max_seq_length (before truncation)."""
    tok = model.tokenizer
    limit = model.max_seq_length
    lengths = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
    return float(np.mean([n > limit for n in lengths]))


def encode_all(spec: EncoderSpec, passages: list[str], queries: dict[str, list[str]]):
    """Encode passages and every query set; returns embeddings, truncation stats and timings."""
    from sentence_transformers import SentenceTransformer

    segmenter = load_segmenter() if spec.word_segment else None
    model = SentenceTransformer(spec.model_name, device=device())
    model.max_seq_length = spec.max_seq_length
    prepared_passages = prepare(passages, spec, "passage", segmenter)
    t0 = time.perf_counter()
    passage_emb = model.encode(
        prepared_passages, batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True
    )
    passage_s = time.perf_counter() - t0
    query_emb = {}
    t0 = time.perf_counter()
    for name, texts in queries.items():
        query_emb[name] = model.encode(
            prepare(texts, spec, "query", segmenter),
            batch_size=BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    query_s = time.perf_counter() - t0
    stats = {
        "passages_truncated": round(truncation_rate(model, prepared_passages), 4),
        "encode_passages_s": round(passage_s, 1),
        "encode_queries_s": round(query_s, 1),
        "embedding_dim": int(passage_emb.shape[1]),
        "max_seq_length": spec.max_seq_length,
    }
    del model
    return passage_emb, query_emb, stats


def main() -> None:
    started = datetime.now(timezone.utc)
    bench = load_viquad()
    doc_ids = list(bench.corpus)
    passages = [bench.corpus[d] for d in doc_ids]

    # Query sets. "test_unseen": test questions whose passage is never used by a train question.
    sets = {}
    for split in ("validation", "test"):
        sets[f"{split}/answerable"] = bench.subset(split, True)
    sets["validation/unanswerable"] = bench.subset("validation", False)
    test_q, test_rel = sets["test/answerable"]
    unseen = {
        q for q, rel in test_rel.items() if "train" not in bench.passage_splits[next(iter(rel))]
    }
    sets["test_unseen/answerable"] = (
        {q: test_q[q] for q in unseen},
        {q: test_rel[q] for q in unseen},
    )
    query_ids = {name: list(qs) for name, (qs, _) in sets.items()}
    query_texts = {name: [qs[q] for q in query_ids[name]] for name, (qs, _) in sets.items()}

    bm25 = BM25Retriever(doc_ids, passages, BM25Params(1.5, 0.75))
    runs: dict[str, dict[str, dict]] = {"bm25": {}}
    for name in sets:
        runs["bm25"][name] = {
            q: [d for d, _ in bm25.search(t, k=DEPTH)]
            for q, t in zip(query_ids[name], query_texts[name], strict=True)
        }

    encoder_stats, failures = {}, {}
    for key, spec in ENCODERS.items():
        try:
            passage_emb, query_emb, stats = encode_all(spec, passages, query_texts)
        except Exception as exc:  # record and continue: one broken encoder must not hide the others
            failures[key] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            continue
        encoder_stats[key] = {"model": spec.model_name, "license": spec.license, **stats}
        retriever = DenseRetriever(doc_ids, passage_emb, backend="faiss")
        dense_runs = {}
        for name in sets:
            hits = retriever.search_embeddings(query_emb[name], k=DEPTH)
            dense_runs[name] = {
                q: [d for d, _ in h] for q, h in zip(query_ids[name], hits, strict=True)
            }
        runs[f"dense:{key}"] = dense_runs
        runs[f"hybrid:bm25+{key}"] = {
            name: {
                q: [
                    d
                    for d, _ in reciprocal_rank_fusion(
                        [runs["bm25"][name][q], dense_runs[name][q]], k=RRF_K, top_n=DEPTH
                    )
                ]
                for q in query_ids[name]
            }
            for name in sets
        }
        print(key, "done", stats, flush=True)

    # --- validation: every candidate ---
    val_name = "validation/answerable"
    val_qrels = sets[val_name][1]
    validation, per_query_val = {}, {}
    for system, by_set in runs.items():
        validation[system], per_query_val[system] = score(by_set[val_name], val_qrels)
    for system in validation:
        if system != "bm25":
            validation[system]["paired_vs_bm25"] = paired(
                per_query_val["bm25"], per_query_val[system]
            )

    def best(prefix: str) -> str | None:
        cands = [s for s in validation if s.startswith(prefix)]
        return max(cands, key=lambda s: (validation[s]["metrics"][SELECT_ON], s)) if cands else None

    selected = {"dense": best("dense:"), "hybrid": best("hybrid:")}

    # --- held-out sets: only BM25 and the selected systems ---
    held_out = {}
    for name in ("validation/unanswerable", "test/answerable", "test_unseen/answerable"):
        qrels = sets[name][1]
        held_out[name] = {}
        per_q = {}
        for system in ["bm25", *[s for s in selected.values() if s]]:
            held_out[name][system], per_q[system] = score(runs[system][name], qrels)
            if system != "bm25":
                held_out[name][system]["paired_vs_bm25"] = paired(per_q["bm25"], per_q[system])

    # --- error analysis on validation: who finds what the other misses (hit@10) ---
    complementarity = {}
    for system in validation:
        if system.startswith("dense:"):
            a, b = per_query_val["bm25"], per_query_val[system]
            complementarity[system] = {
                "both_hit": sum(a[q]["hit@10"] == 1 and b[q]["hit@10"] == 1 for q in a),
                "only_bm25": sum(a[q]["hit@10"] == 1 and b[q]["hit@10"] == 0 for q in a),
                "only_dense": sum(a[q]["hit@10"] == 0 and b[q]["hit@10"] == 1 for q in a),
                "neither": sum(a[q]["hit@10"] == 0 and b[q]["hit@10"] == 0 for q in a),
            }

    payload = {
        "experiment": "m3_dense",
        "created_utc": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": __version__,
        "dataset": {"hf_id": HF_DATASET, **bench.stats()},
        "protocol": {
            "selection_metric": f"validation/answerable {SELECT_ON}",
            "rrf_k": RRF_K,
            "depth": DEPTH,
            "bm25": {"k1": 1.5, "b": 0.75},
            "test_reported_for": "bm25 and the selected dense and hybrid systems only",
            "test_unseen": "test questions whose gold passage is not used by any train question",
        },
        "query_set_sizes": {name: len(ids) for name, ids in query_ids.items()},
        "encoders": encoder_stats,
        "encoder_failures": failures,
        "selected": selected,
        "validation": validation,
        "held_out": held_out,
        "complementarity_validation_hit@10": complementarity,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device(),
            "gpu": gpu_name(),
            "runner": os.environ.get("RUNNER_NAME") or os.environ.get("VI_RAG_RUNNER", "local"),
        },
        "wall_time_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "failures": failures}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
