"""M3.1 ablations (issue #18), protocol fixed before running.

A1: vietnamese-sbert on raw vs pyvi word-segmented text (validation, answerable), paired.
A2: RRF grid for BM25 + multilingual-e5-base over k and the dense weight, selected on validation
    MRR@10; the selected configuration and the M3 default (k = 60, weight 1) are then compared once
    on test (all questions and passage-unseen-in-train), paired.

Writes ``results/m3_ablation.json``. Reuses the encoding and scoring helpers of ``run_dense.py`` so
both experiments encode and score text identically.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dense import device, encode_all, git_commit, gpu_name, paired, score  # noqa: E402

from vi_rag_agent import __version__  # noqa: E402
from vi_rag_agent.bm25 import BM25Params, BM25Retriever  # noqa: E402
from vi_rag_agent.data import HF_DATASET, load_viquad  # noqa: E402
from vi_rag_agent.dense import ENCODERS, DenseRetriever  # noqa: E402
from vi_rag_agent.fusion import reciprocal_rank_fusion  # noqa: E402

OUT = Path("results/m3_ablation.json")
DEPTH = 100
RRF_KS = (10, 20, 40, 60, 100)
DENSE_WEIGHTS = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT = (60, 1.0)
SELECT_ON = "mrr@10"


def distance_from_default(k: int, w: float) -> tuple[float, float]:
    """Tie-break for the grid: prefer configurations closer to the pre-registered default."""
    return (abs(math.log(k / DEFAULT[0])), abs(math.log(w / DEFAULT[1])))


def dense_run(retriever: DenseRetriever, ids: list[str], embeddings) -> dict[str, list[str]]:
    hits = retriever.search_embeddings(embeddings, k=DEPTH)
    return {q: [d for d, _ in h] for q, h in zip(ids, hits, strict=True)}


def fuse(bm25_run: dict, dense: dict, k: int, w: float) -> dict[str, list[str]]:
    return {
        q: [
            d
            for d, _ in reciprocal_rank_fusion(
                [bm25_run[q], dense[q]], k=k, weights=[1.0, w], top_n=DEPTH
            )
        ]
        for q in bm25_run
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    bench = load_viquad()
    doc_ids = list(bench.corpus)
    passages = [bench.corpus[d] for d in doc_ids]

    sets = {
        "validation/answerable": bench.subset("validation", True),
        "test/answerable": bench.subset("test", True),
    }
    test_q, test_rel = sets["test/answerable"]
    unseen = [q for q, r in test_rel.items() if "train" not in bench.passage_splits[next(iter(r))]]
    sets["test_unseen/answerable"] = (
        {q: test_q[q] for q in unseen},
        {q: test_rel[q] for q in unseen},
    )
    ids = {n: list(qs) for n, (qs, _) in sets.items()}
    texts = {n: [qs[q] for q in ids[n]] for n, (qs, _) in sets.items()}
    qrels = {n: rel for n, (_, rel) in sets.items()}
    val = "validation/answerable"

    # ---------------- A1: word segmentation for vietnamese-sbert (validation only) ----------------
    raw_spec = ENCODERS["vietnamese-sbert"]
    seg_spec = replace(raw_spec, word_segment=True)
    a1, per_query_a1, encoders = {}, {}, {}
    for label, spec in (("raw", raw_spec), ("word_segmented", seg_spec)):
        p_emb, q_emb, stats = encode_all(spec, passages, {val: texts[val]})
        encoders[f"vietnamese-sbert/{label}"] = {"model": spec.model_name, **stats}
        run = dense_run(DenseRetriever(doc_ids, p_emb, backend="faiss"), ids[val], q_emb[val])
        a1[label], per_query_a1[label] = score(run, qrels[val])
    a1["word_segmented"]["paired_vs_raw"] = paired(
        per_query_a1["raw"], per_query_a1["word_segmented"]
    )

    # ---------------- A2: RRF grid for BM25 + e5-base ----------------
    bm25 = BM25Retriever(doc_ids, passages, BM25Params(1.5, 0.75))
    bm25_runs = {
        n: {
            q: [d for d, _ in bm25.search(t, k=DEPTH)]
            for q, t in zip(ids[n], texts[n], strict=True)
        }
        for n in sets
    }
    e5 = ENCODERS["multilingual-e5-base"]
    p_emb, q_emb, stats = encode_all(e5, passages, texts)
    encoders["multilingual-e5-base"] = {"model": e5.model_name, **stats}
    e5_retriever = DenseRetriever(doc_ids, p_emb, backend="faiss")
    e5_runs = {n: dense_run(e5_retriever, ids[n], q_emb[n]) for n in sets}

    grid = []
    for k in RRF_KS:
        for w in DENSE_WEIGHTS:
            summary, _ = score(fuse(bm25_runs[val], e5_runs[val], k, w), qrels[val])
            grid.append({"k": k, "dense_weight": w, **summary["metrics"]})
    best = max(
        grid,
        key=lambda g: (
            g[SELECT_ON],
            tuple(-x for x in distance_from_default(g["k"], g["dense_weight"])),
        ),
    )
    selected = (best["k"], best["dense_weight"])

    held_out = {}
    for n in ("validation/answerable", "test/answerable", "test_unseen/answerable"):
        default_s, default_pq = score(fuse(bm25_runs[n], e5_runs[n], *DEFAULT), qrels[n])
        tuned_s, tuned_pq = score(fuse(bm25_runs[n], e5_runs[n], *selected), qrels[n])
        tuned_s["paired_vs_default"] = paired(default_pq, tuned_pq)
        held_out[n] = {"default_k60_w1": default_s, "selected": tuned_s}
    held_out["validation/answerable"]["note"] = (
        "optimistic: the selected configuration was chosen on this set"
    )

    test_gain = held_out["test/answerable"]["selected"]["paired_vs_default"]["mrr@10"]
    adopt = selected != DEFAULT and test_gain["ci95"][0] > 0

    payload = {
        "experiment": "m3_ablation",
        "issue": 18,
        "created_utc": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": __version__,
        "dataset": {"hf_id": HF_DATASET, **bench.stats()},
        "protocol": {
            "a1": "vietnamese-sbert raw vs pyvi word-segmented, validation answerable only",
            "a2_grid": {
                "rrf_k": list(RRF_KS),
                "dense_weight": list(DENSE_WEIGHTS),
                "bm25_weight": 1.0,
            },
            "selection": f"validation/answerable {SELECT_ON}; ties: closest to k=60, w=1",
            "decision_rule": "adopt tuned RRF only if its test ΔMRR@10 vs default has CI95 above 0",
        },
        "query_set_sizes": {n: len(v) for n, v in ids.items()},
        "encoders": encoders,
        "a1_word_segmentation": a1,
        "a2_rrf_grid_validation": grid,
        "a2_selected": {"k": selected[0], "dense_weight": selected[1]},
        "a2_held_out": held_out,
        "decision": {"adopt_tuned_rrf": adopt, "test_mrr_gain_vs_default": test_gain},
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
    print(
        json.dumps(
            {
                "a1": {k: v["metrics"]["mrr@10"] for k, v in a1.items()},
                "selected": selected,
                "adopt": adopt,
            }
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
