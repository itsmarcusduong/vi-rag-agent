"""M5: RAG answer generation with open LLMs (docs/protocols/m5_generation.md, issue #29).

Stages: load -> gate -> stage_a (model selection) -> stage_b (context ablation) -> test (decision).
Writes results/m5_generation.json with a stage_status block; exits 1 if a stage fails.
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

from run_dense import device, encode_all, git_commit, gpu_name, score  # noqa: E402

from vi_rag_agent import __version__  # noqa: E402
from vi_rag_agent.bm25 import BM25Params, BM25Retriever  # noqa: E402
from vi_rag_agent.data import HF_DATASET, load_viquad  # noqa: E402
from vi_rag_agent.dense import ENCODERS, DenseRetriever  # noqa: E402
from vi_rag_agent.fusion import reciprocal_rank_fusion  # noqa: E402
from vi_rag_agent.generation import build_messages, clean_answer  # noqa: E402
from vi_rag_agent.metrics import paired_cluster_difference  # noqa: E402
from vi_rag_agent.qa import exact_match, f1_score  # noqa: E402

OUT = Path("results/m5_generation.json")
REFERENCE = Path("results/m3_ablation.json")
GATE_TOLERANCE = 5e-5
VAL_N, TEST_N, SEED = 1000, 2000, 0
DEPTH, RRF_K = 100, 10
LLMS = {
    "L1": "Qwen/Qwen3.5-4B",
    "L2": "google/gemma-4-E4B-it",
    "L3": "Qwen/Qwen3.5-9B",
}
SIZE_ORDER = ["L1", "L2", "L3"]  # smallest first (tie-break)
N_EXAMPLES = 30


def make_generator(model_name: str):  # pragma: no cover - replaced in tests
    from vi_rag_agent.generation import HFGenerator

    return HFGenerator(model_name)


def reproduction_target() -> float:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return ref["a2_held_out"]["validation/answerable"]["selected"]["metrics"]["mrr@10"]


def peak_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return None


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def library_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "bitsandbytes", "sentence_transformers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # not installed in the test environment
            out[name] = None
    return out


def sample_ids(ids: list[str], n: int, rng: np.random.Generator) -> list[str]:
    ids = sorted(ids)
    if n >= len(ids):
        return ids
    return sorted(ids[i] for i in rng.choice(len(ids), size=n, replace=False))


def answer(gen, questions: dict, qids: list[str], contexts: dict | None) -> dict:
    """Generate and score one variant; contexts maps qid -> list of passages, None = closed-book."""
    messages = [
        build_messages(questions[q], None if contexts is None else contexts[q]) for q in qids
    ]
    reset_peak_memory()
    t0 = time.perf_counter()
    raw, n_tokens = gen.generate(messages)
    elapsed = time.perf_counter() - t0
    preds = {q: clean_answer(r) for q, r in zip(qids, raw, strict=True)}
    return {
        "preds": preds,
        "tokens": dict(zip(qids, n_tokens, strict=True)),
        "seconds": elapsed,
        "peak_gb": peak_memory_gb(),
    }


def per_query_scores(preds: dict, golds: dict) -> dict:
    return {
        q: {"em": exact_match(p, golds[q]), "f1": f1_score(p, golds[q])} for q, p in preds.items()
    }


def summarise(
    variant: dict, golds: dict, groups: dict, examples_from: list[str]
) -> tuple[dict, dict]:
    pq = per_query_scores(variant["preds"], golds)
    zero = {q: {"em": 0.0, "f1": 0.0} for q in pq}
    summary = {"n": len(pq)}
    for m in ("em", "f1"):
        d = paired_cluster_difference(zero, pq, m, groups)
        summary[m] = round(d["mean_diff"], 4)
        summary[f"{m}_ci95"] = [round(x, 4) for x in d["ci95"]]
    tokens = list(variant["tokens"].values())
    summary["mean_generated_tokens"] = round(float(np.mean(tokens)), 2) if tokens else None
    summary["questions_per_second"] = (
        round(len(pq) / variant["seconds"], 3) if variant["seconds"] else None
    )
    summary["peak_gpu_gb"] = variant["peak_gb"]
    summary["examples"] = [
        {"qid": q, "gold": golds[q], "pred": variant["preds"][q]} for q in examples_from
    ]
    return summary, pq


def paired(pq_a: dict, pq_b: dict, groups: dict) -> dict:
    out = {}
    for m in ("em", "f1"):
        d = paired_cluster_difference(pq_a, pq_b, m, groups)
        out[m] = {
            "mean_diff": round(d["mean_diff"], 4),
            "ci95": [round(x, 4) for x in d["ci95"]],
            "better": d["b_better"],
            "worse": d["a_better"],
            "tied": d["tied"],
        }
    return out


def mean_f1(pq: dict) -> float:
    return float(np.mean([v["f1"] for v in pq.values()]))


def contains_zero(ci: list[float]) -> bool:
    return ci[0] <= 0 <= ci[1]


def select_model(candidates: list[str], pq: dict, groups: dict) -> tuple[str, str, dict | None]:
    """Stage A: best hybrid top-3 F1; if best vs next has a CI containing 0, the smaller model."""
    ranked = sorted(
        candidates, key=lambda c: (-mean_f1(pq[f"{c}/hybrid_top3"]), SIZE_ORDER.index(c))
    )
    if len(ranked) == 1:
        return ranked[0], "only model that loaded and ran", None
    best, nxt = ranked[0], ranked[1]
    d = paired(pq[f"{nxt}/hybrid_top3"], pq[f"{best}/hybrid_top3"], groups)["f1"]
    comparison = {"best": best, "next": nxt, "f1_diff": d}
    if contains_zero(d["ci95"]):
        smaller = min(best, nxt, key=SIZE_ORDER.index)
        return smaller, f"{best} vs {nxt}: F1 CI contains 0, smaller model {smaller}", comparison
    return best, "highest validation F1 with hybrid top-3 (CI excludes 0)", comparison


def select_k(pq_by_k: dict[int, dict], groups: dict) -> tuple[int, str]:
    """Protocol stage B: best F1 over k; a smaller k whose difference CI contains 0 wins."""
    best = max(sorted(pq_by_k), key=lambda k: mean_f1(pq_by_k[k]))  # ties in mean -> smaller k
    for k in sorted(pq_by_k):
        if k >= best:
            break
        d = paired(pq_by_k[k], pq_by_k[best], groups)["f1"]
        if contains_zero(d["ci95"]):
            return k, f"k={k} ties with best k={best} (F1 CI contains 0): smaller k"
    return best, "highest validation F1"


def main() -> int:
    started = datetime.now(timezone.utc)
    status: dict[str, str] = {}
    payload: dict = {
        "experiment": "m5_generation",
        "issue": 29,
        "protocol": "docs/protocols/m5_generation.md",
        "created_utc": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": __version__,
        "stage_status": status,
        "failures": {},
    }

    def finish(code: int) -> int:
        payload["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device(),
            "gpu": gpu_name(),
            "libraries": library_versions(),
            "runner": os.environ.get("RUNNER_NAME") or os.environ.get("VI_RAG_RUNNER", "local"),
        }
        payload["wall_time_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage_status": status}, indent=2), flush=True)
        return code

    # ---------------- load ----------------
    try:
        bench = load_viquad()
        doc_ids = list(bench.corpus)
        texts = [bench.corpus[d] for d in doc_ids]
        val_q, val_rel = bench.subset("validation", True)
        test_q, test_rel = bench.subset("test", True)
        golds = {**bench.answers["validation"], **bench.answers["test"]}
        gold_passage = {q: next(iter(r)) for q, r in {**val_rel, **test_rel}.items()}
        groups = gold_passage  # cluster unit for the bootstrap: the gold passage

        rng = np.random.default_rng(SEED)
        val_ids = sample_ids([q for q in val_q if golds.get(q)], VAL_N, rng)
        test_ids = sample_ids([q for q in test_q if golds.get(q)], TEST_N, rng)
        unseen = [q for q in test_ids if "train" not in bench.passage_splits[gold_passage[q]]]
        questions = {**val_q, **test_q}

        bm25 = BM25Retriever(doc_ids, texts, BM25Params(1.5, 0.75))
        full_val_ids = list(val_q)
        need = sorted(set(full_val_ids) | set(test_ids))
        e5 = ENCODERS["multilingual-e5-base"]
        p_emb, q_emb, e5_stats = encode_all(e5, texts, {"q": [questions[q] for q in need]})
        dense = DenseRetriever(doc_ids, p_emb, backend="faiss")
        hits = dense.search_embeddings(q_emb["q"], k=DEPTH)
        bm25_runs, hybrid_runs = {}, {}
        for q, h in zip(need, hits, strict=True):
            lexical = [d for d, _ in bm25.search(questions[q], k=DEPTH)]
            bm25_runs[q] = lexical
            fused = reciprocal_rank_fusion([lexical, [d for d, _ in h]], k=RRF_K, top_n=DEPTH)
            hybrid_runs[q] = [d for d, _ in fused]
        payload["samples"] = {
            "seed": SEED,
            "validation": len(val_ids),
            "test": len(test_ids),
            "test_unseen": len(unseen),
            "validation_passages": len({groups[q] for q in val_ids}),
        }
        payload["dataset"] = {"hf_id": HF_DATASET, **bench.stats()}
        payload["first_stage"] = {"e5": e5_stats, "rrf_k": RRF_K}
        status["load"] = "OK"
    except Exception as exc:
        status["load"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        return finish(1)

    # ---------------- gate ----------------
    gate, _ = score({q: hybrid_runs[q] for q in full_val_ids}, val_rel)
    target = reproduction_target()
    payload["reproduction_gate"] = {
        "target_mrr@10": target,
        "reproduced_mrr@10": gate["metrics"]["mrr@10"],
    }
    if abs(gate["metrics"]["mrr@10"] - target) > GATE_TOLERANCE:
        status["gate"] = f"FAILED: reproduced {gate['metrics']['mrr@10']} != reference {target}"
        return finish(1)
    status["gate"] = "OK"

    def ctx(run: dict, qids: list[str], k: int) -> dict:
        return {q: [bench.corpus[d] for d in run[q][:k]] for q in qids}

    oracle = {q: [bench.corpus[gold_passage[q]]] for q in val_ids}
    examples = val_ids[:N_EXAMPLES]

    # ---------------- stage A: model selection ----------------
    stage_a, pq_a, load_info = {}, {}, {}
    try:
        for lid, name in LLMS.items():
            gen = None
            try:
                gen = make_generator(name)
                info = {
                    "model": name,
                    "revision": getattr(gen, "revision", None),
                    "load_s": round(getattr(gen, "load_s", 0.0), 1),
                }
                runs = {}
                for cond, contexts in (
                    ("closed_book", None),
                    ("hybrid_top3", ctx(hybrid_runs, val_ids, 3)),
                ):
                    v = answer(gen, questions, val_ids, contexts)
                    runs[f"{lid}/{cond}"] = summarise(v, golds, groups, examples)
            except Exception as exc:  # load or OOM: recorded and excluded, as the protocol says
                payload["failures"][lid] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                continue
            finally:
                if gen is not None:
                    gen.close()
            load_info[lid] = info
            for key, (summary, pq) in runs.items():
                stage_a[key], pq_a[key] = summary, pq
                print(key, summary["f1"], flush=True)
        status["stage_a"] = "OK" if any(k.endswith("hybrid_top3") for k in stage_a) else "FAILED"
    except Exception as exc:
        status["stage_a"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload["stage_a"] = stage_a
        return finish(1)
    payload["stage_a"] = stage_a
    payload["models"] = load_info
    if status["stage_a"] != "OK":
        return finish(1)

    for lid in load_info:
        key = f"{lid}/hybrid_top3"
        stage_a[key]["paired_vs_closed_book"] = paired(
            pq_a[f"{lid}/closed_book"], pq_a[key], groups
        )
    selected, reason, comparison = select_model(list(load_info), pq_a, groups)
    if comparison:
        payload["selection_comparison"] = comparison
    payload["selected_model"] = {"id": selected, "model": LLMS[selected], "reason": reason}

    # ---------------- stage B: context ablation ----------------
    stage_b = {"B0_hybrid_top3": stage_a[f"{selected}/hybrid_top3"]}
    pq_b = {"B0_hybrid_top3": pq_a[f"{selected}/hybrid_top3"]}
    try:
        gen = make_generator(LLMS[selected])
        variants = {
            "B1_hybrid_top1": ctx(hybrid_runs, val_ids, 1),
            "B2_hybrid_top5": ctx(hybrid_runs, val_ids, 5),
            "B3_bm25_top3": ctx(bm25_runs, val_ids, 3),
            "B4_oracle": oracle,
        }
        for key, contexts in variants.items():
            v = answer(gen, questions, val_ids, contexts)
            stage_b[key], pq_b[key] = summarise(v, golds, groups, examples)
            print(key, stage_b[key]["f1"], flush=True)
        status["stage_b"] = "OK"
    except Exception as exc:
        status["stage_b"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload["stage_b"] = stage_b
        return finish(1)
    for key in stage_b:
        if key != "B0_hybrid_top3":
            stage_b[key]["paired_vs_B0"] = paired(pq_b["B0_hybrid_top3"], pq_b[key], groups)
    payload["stage_b"] = stage_b

    k_variants = {1: "B1_hybrid_top1", 3: "B0_hybrid_top3", 5: "B2_hybrid_top5"}
    chosen_k, k_reason = select_k({k: pq_b[v] for k, v in k_variants.items()}, groups)
    payload["selected_k"] = {"k": chosen_k, "reason": k_reason}

    # ---------------- test: decision ----------------
    test = {}
    try:
        systems = {
            "closed_book": None,
            f"bm25_top{chosen_k}": ctx(bm25_runs, test_ids, chosen_k),
            f"hybrid_top{chosen_k}": ctx(hybrid_runs, test_ids, chosen_k),
        }
        pq_t = {}
        for key, contexts in systems.items():
            v = answer(gen, questions, test_ids, contexts)
            test[key], pq_t[key] = summarise(v, golds, groups, test_ids[:N_EXAMPLES])
        gen.close()
        hyb, bm, cb = f"hybrid_top{chosen_k}", f"bm25_top{chosen_k}", "closed_book"
        for subset_name, ids in (("all", test_ids), ("unseen", unseen)):
            sub = {s: {q: pq_t[s][q] for q in ids} for s in pq_t}
            test[f"paired_{subset_name}"] = {
                "hybrid_vs_closed_book": paired(sub[cb], sub[hyb], groups),
                "hybrid_vs_bm25": paired(sub[bm], sub[hyb], groups),
                "f1": {
                    s: round(float(np.mean([v["f1"] for v in sub[s].values()])), 4) for s in sub
                },
            }
        status["test"] = "OK"
    except Exception as exc:
        status["test"] = f"FAILED: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload["test"] = test
        return finish(1)
    payload["test"] = test

    pa, pu = test["paired_all"], test["paired_unseen"]
    payload["decision"] = {
        "rag_adopted": pa["hybrid_vs_closed_book"]["f1"]["ci95"][0] > 0,
        "adr_0003_confirmed_downstream": pa["hybrid_vs_bm25"]["f1"]["ci95"][0] > 0,
        "unseen_hybrid_vs_closed_book_f1": pu["hybrid_vs_closed_book"]["f1"],
        "model": LLMS[selected],
        "k": chosen_k,
    }
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
