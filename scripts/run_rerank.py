"""M4: cross-encoder reranking on top of the ADR-0003 hybrid retriever.

Follows docs/protocols/m4_rerank.md (merged before this run, issue #23). Stages:
  1. load      - data, BM25, e5-base embeddings, hybrid runs (RRF k = 10)
  2. gate      - reproduce validation MRR@10 of the hybrid from results/m3_ablation.json
  3. validate  - every reranker on validation; eligibility (p95 latency) and selection
  4. test      - R0 and the selected reranker on test and test-unseen; decision rule
Writes results/m4_rerank.json with a stage_status block; exits 1 if a stage fails.
Metric cut-offs come from run_dense.score (1, 5, 10, 20, 100); the protocol's recall@50 was
therefore not reported (deviation recorded in docs/reports/v0.4.0.md).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dense import device, encode_all, git_commit, gpu_name, paired, score  # noqa: E402

from vi_rag_agent import __version__  # noqa: E402
from vi_rag_agent.bm25 import BM25Params, BM25Retriever  # noqa: E402
from vi_rag_agent.data import HF_DATASET, load_viquad  # noqa: E402
from vi_rag_agent.dense import ENCODERS, DenseRetriever  # noqa: E402
from vi_rag_agent.fusion import reciprocal_rank_fusion  # noqa: E402
from vi_rag_agent.metrics import paired_difference  # noqa: E402

OUT = Path("results/m4_rerank.json")
REFERENCE = Path("results/m3_ablation.json")
DEPTH, RERANK_TOP, RRF_K = 100, 50, 10
MAX_LENGTH, BATCH = 512, 64
LATENCY_P95_LIMIT_MS = 300.0
GATE_TOLERANCE = 5e-5
RERANKERS = {
    "R1": {"model": "cross-encoder/ms-marco-MiniLM-L6-v2", "license": "Apache-2.0"},
    "R2": {"model": "BAAI/bge-reranker-v2-m3", "license": "Apache-2.0"},
}


def reproduction_target() -> float:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return ref["a2_held_out"]["validation/answerable"]["selected"]["metrics"]["mrr@10"]


def load_reranker(model_name: str):
    """CrossEncoder in float16 on the GPU; returns (predict_fn, dtype_str)."""
    import torch
    from sentence_transformers import CrossEncoder

    dev = device()
    try:
        model = CrossEncoder(
            model_name,
            max_length=MAX_LENGTH,
            device=dev,
            model_kwargs={"torch_dtype": torch.float16},
        )
    except TypeError:  # older sentence-transformers without model_kwargs
        model = CrossEncoder(model_name, max_length=MAX_LENGTH, device=dev)
        model.model.half()
    dtype = str(next(model.model.parameters()).dtype)

    def predict(pairs):
        scores = model.predict(pairs, batch_size=BATCH, show_progress_bar=False)
        if dev == "cuda":
            torch.cuda.synchronize()
        return np.asarray(scores, dtype=np.float64)

    return predict, dtype


def rerank_run(predict, questions: dict, run: dict, passages: dict) -> tuple[dict, list[float]]:
    """Rerank the top RERANK_TOP of each ranking; keep the rest in first-stage order."""
    out, latencies = {}, []
    for qid, ranking in run.items():
        head, tail = ranking[:RERANK_TOP], ranking[RERANK_TOP:]
        if not head:
            out[qid] = list(ranking)
            continue
        t0 = time.perf_counter()
        scores = predict([(questions[qid], passages[d]) for d in head])
        latencies.append((time.perf_counter() - t0) * 1000)
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))  # ties keep hybrid order
        out[qid] = [head[i] for i in order] + tail
    return out, latencies


def latency_summary(latencies: list[float]) -> dict:
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "n": len(latencies),
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    status: dict[str, str] = {}
    payload: dict = {
        "experiment": "m4_rerank",
        "issue": 23,
        "protocol": "docs/protocols/m4_rerank.md",
        "created_utc": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": __version__,
        "stage_status": status,
    }

    def finish(code: int) -> int:
        payload["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device(),
            "gpu": gpu_name(),
            "runner": os.environ.get("RUNNER_NAME") or os.environ.get("VI_RAG_RUNNER", "local"),
        }
        payload["wall_time_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage_status": status}, indent=2))
        return code

    # ---- 1. load ----
    try:
        bench = load_viquad()
        doc_ids = list(bench.corpus)
        texts_by_id = dict(bench.corpus)
        sets = {
            "validation/answerable": bench.subset("validation", True),
            "validation/unanswerable": bench.subset("validation", False),
            "test/answerable": bench.subset("test", True),
        }
        tq, trel = sets["test/answerable"]
        unseen = [q for q, r in trel.items() if "train" not in bench.passage_splits[next(iter(r))]]
        sets["test_unseen/answerable"] = ({q: tq[q] for q in unseen}, {q: trel[q] for q in unseen})
        ids = {n: list(qs) for n, (qs, _) in sets.items()}
        qtext = {n: [qs[q] for q in ids[n]] for n, (qs, _) in sets.items()}
        questions = {n: qs for n, (qs, _) in sets.items()}
        qrels = {n: rel for n, (_, rel) in sets.items()}

        bm25 = BM25Retriever(doc_ids, [texts_by_id[d] for d in doc_ids], BM25Params(1.5, 0.75))
        e5 = ENCODERS["multilingual-e5-base"]
        p_emb, q_emb, e5_stats = encode_all(e5, [texts_by_id[d] for d in doc_ids], qtext)
        dense = DenseRetriever(doc_ids, p_emb, backend="faiss")
        hybrid = {}
        for n in sets:
            hits = dense.search_embeddings(q_emb[n], k=DEPTH)
            hybrid[n] = {}
            for q, t, h in zip(ids[n], qtext[n], hits, strict=True):
                lexical = [d for d, _ in bm25.search(t, k=DEPTH)]
                semantic = [d for d, _ in h]
                fused = reciprocal_rank_fusion([lexical, semantic], k=RRF_K, top_n=DEPTH)
                hybrid[n][q] = [d for d, _ in fused]
        payload["first_stage"] = {"e5": e5_stats, "rrf_k": RRF_K, "depth": DEPTH}
        payload["query_set_sizes"] = {n: len(v) for n, v in ids.items()}
        payload["dataset"] = {"hf_id": HF_DATASET, **bench.stats()}
        status["load"] = "OK"
    except Exception as exc:
        status["load"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        return finish(1)

    # ---- 2. reproduction gate ----
    val = "validation/answerable"
    r0_val, r0_val_pq = score(hybrid[val], qrels[val])
    target = reproduction_target()
    got = r0_val["metrics"]["mrr@10"]
    payload["reproduction_gate"] = {"target_mrr@10": target, "reproduced_mrr@10": got}
    if abs(got - target) > GATE_TOLERANCE:
        status["gate"] = f"FAILED: reproduced {got} != reference {target}"
        return finish(1)
    status["gate"] = "OK"

    # ---- 3. validation: every reranker ----
    validation = {"R0": r0_val}
    per_query = {"R0": r0_val_pq}
    predictors = {}
    try:
        for rid, meta in RERANKERS.items():
            predict, dtype = load_reranker(meta["model"])
            predictors[rid] = predict
            run, lat = rerank_run(predict, questions[val], hybrid[val], texts_by_id)
            summary, pq = score(run, qrels[val])
            summary.update(
                {
                    "model": meta["model"],
                    "license": meta["license"],
                    "dtype": dtype,
                    "latency": latency_summary(lat),
                    "eligible": latency_summary(lat)["p95_ms"] <= LATENCY_P95_LIMIT_MS,
                    "paired_vs_R0": paired(r0_val_pq, pq),
                }
            )
            validation[rid], per_query[rid] = summary, pq
            print(rid, summary["metrics"]["mrr@10"], summary["latency"], flush=True)
        status["validate"] = "OK"
    except Exception as exc:
        status["validate"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload["validation"] = validation
        return finish(1)

    eligible = [r for r in RERANKERS if validation[r]["eligible"]]
    selected = "R0"
    reason = "no eligible reranker"
    if eligible:
        best = max(eligible, key=lambda r: validation[r]["metrics"]["mrr@10"])
        others = [r for r in eligible if r != best]
        if others:
            d = paired_difference(per_query[others[0]], per_query[best], "mrr@10")
            if d["ci95"][0] <= 0 <= d["ci95"][1]:
                best = min(eligible, key=lambda r: validation[r]["latency"]["p95_ms"])
                reason = "R1/R2 tie on validation (CI contains 0): lower p95 latency chosen"
            else:
                reason = "highest validation MRR@10 among eligible rerankers"
        else:
            reason = "only eligible reranker"
        beats_r0 = validation[best]["paired_vs_R0"]["mrr@10"]["ci95"][0] > 0
        if beats_r0:
            selected = best
        else:
            reason += "; but it does not beat R0 on validation (CI not above 0)"
    payload["validation"] = validation
    payload["selection"] = {"selected": selected, "eligible": eligible, "reason": reason}

    # ---- 4. held-out sets: R0 and the selected reranker only ----
    held_out = {}
    try:
        for n in ("test/answerable", "test_unseen/answerable", "validation/unanswerable"):
            base, base_pq = score(hybrid[n], qrels[n])
            held_out[n] = {"R0": base}
            if selected != "R0":
                run, lat = rerank_run(predictors[selected], questions[n], hybrid[n], texts_by_id)
                s, pq = score(run, qrels[n])
                s["latency"] = latency_summary(lat)
                s["paired_vs_R0"] = paired(base_pq, pq)
                held_out[n][selected] = s
        status["test"] = "OK"
    except Exception as exc:
        status["test"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload["held_out"] = held_out
        return finish(1)
    payload["held_out"] = held_out
    held_out["validation/unanswerable"]["note"] = "exploratory"

    adopt = selected != "R0" and all(
        held_out[n][selected]["paired_vs_R0"]["mrr@10"]["ci95"][0] > 0
        for n in ("test/answerable", "test_unseen/answerable")
    )
    payload["decision"] = {
        "adopt": adopt,
        "reranker": RERANKERS[selected]["model"] if adopt else None,
        "rule": "paired ΔMRR@10 vs R0 with CI95 above 0 on test and test-unseen",
    }
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
