"""Ranking metrics for retrieval evaluation.

A *run* maps each query ID to a ranked list of document IDs (best first). *Qrels* map each query ID
to ``{doc_id: grade}``; a grade > 0 means relevant. Conventions follow trec_eval:

* recall@k = relevant retrieved in the top k / total relevant for the query
* precision@k = relevant retrieved in the top k / k (missing ranks count as non-relevant)
* hit@k = 1 if any relevant document is in the top k
* MRR@k = 1 / rank of the first relevant document within the top k, else 0
* nDCG@k uses linear gain (gain = grade) and discount 1 / log2(rank + 1)

Queries that have no relevant document in the qrels are excluded from the means and reported, so a
data problem cannot silently pull a metric towards 0. Duplicate IDs in a ranking raise an error: a
retriever that returns the same document twice is a bug, and counting it twice would inflate recall.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

Qrel = Mapping[str, int]


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _check_ranking(ranking: Sequence[str]) -> None:
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate document IDs")


def _relevant(qrel: Qrel) -> set[str]:
    return {doc_id for doc_id, grade in qrel.items() if grade > 0}


def recall_at_k(ranking: Sequence[str], qrel: Qrel, k: int) -> float:
    """Fraction of the query's relevant documents found in the top ``k``."""
    _check_k(k)
    _check_ranking(ranking)
    relevant = _relevant(qrel)
    if not relevant:
        raise ValueError("query has no relevant documents")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def precision_at_k(ranking: Sequence[str], qrel: Qrel, k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant document (denominator is always k)."""
    _check_k(k)
    _check_ranking(ranking)
    return len(_relevant(qrel).intersection(ranking[:k])) / k


def hit_at_k(ranking: Sequence[str], qrel: Qrel, k: int) -> float:
    """1.0 if at least one relevant document is in the top ``k``, else 0.0."""
    _check_k(k)
    _check_ranking(ranking)
    return 1.0 if _relevant(qrel).intersection(ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[str], qrel: Qrel, k: int) -> float:
    """1 / (1-based rank of the first relevant document within the top ``k``), else 0.0."""
    _check_k(k)
    _check_ranking(ranking)
    relevant = _relevant(qrel)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], qrel: Qrel, k: int) -> float:
    """Normalized discounted cumulative gain with linear gain and log2(rank + 1) discount."""
    _check_k(k)
    _check_ranking(ranking)
    dcg = sum(
        max(qrel.get(doc_id, 0), 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking[:k], start=1)
    )
    ideal_grades = sorted((g for g in qrel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_grades, start=1))
    if idcg == 0:
        raise ValueError("query has no relevant documents")
    return dcg / idcg


@dataclass
class EvaluationResult:
    """Mean metrics over evaluated queries plus bookkeeping about what was skipped."""

    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    n_queries: int
    skipped_no_relevant: list[str] = field(default_factory=list)
    missing_from_run: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Summary without per-query values (those can be large)."""
        return {
            "metrics": self.metrics,
            "n_queries": self.n_queries,
            "n_skipped_no_relevant": len(self.skipped_no_relevant),
            "n_missing_from_run": len(self.missing_from_run),
        }


def evaluate(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Qrel],
    ks: Iterable[int] = (1, 5, 10),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> EvaluationResult:
    """Score a run against qrels.

    Every query in ``qrels`` with at least one relevant document is evaluated. A query absent
    from ``run`` is scored as an empty ranking (all zeros) and listed in ``missing_from_run``: a
    retriever that fails to answer must be penalised, not dropped. Queries in ``run`` but not in
    ``qrels`` are ignored.
    """
    ks = sorted(set(ks))
    for k in (*ks, mrr_k, ndcg_k):
        _check_k(k)

    per_query: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    missing: list[str] = []
    for qid in sorted(qrels):
        qrel = qrels[qid]
        if not _relevant(qrel):
            skipped.append(qid)
            continue
        if qid not in run:
            missing.append(qid)
        ranking = list(run.get(qid, []))
        scores = {f"recall@{k}": recall_at_k(ranking, qrel, k) for k in ks}
        scores |= {f"hit@{k}": hit_at_k(ranking, qrel, k) for k in ks}
        scores[f"mrr@{mrr_k}"] = reciprocal_rank(ranking, qrel, mrr_k)
        scores[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranking, qrel, ndcg_k)
        per_query[qid] = scores

    if not per_query:
        raise ValueError("no query with relevant documents to evaluate")
    names = next(iter(per_query.values())).keys()
    means = {name: sum(s[name] for s in per_query.values()) / len(per_query) for name in names}
    return EvaluationResult(means, per_query, len(per_query), skipped, missing)


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-query values.

    Resamples queries with replacement; deterministic for a given ``seed``.
    """
    import numpy as np

    if not values:
        raise ValueError("values is empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_resamples, len(data)))
    means = data[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def paired_difference(
    per_query_a: Mapping[str, Mapping[str, float]],
    per_query_b: Mapping[str, Mapping[str, float]],
    metric: str,
    seed: int = 0,
) -> dict:
    """Mean of (b - a) over the queries both systems were scored on, with a bootstrap 95% CI.

    Pairing by query removes the between-query variance that dominates unpaired comparisons, so a
    small but consistent gain can be distinguished from noise.
    """
    shared = sorted(set(per_query_a) & set(per_query_b))
    if not shared:
        raise ValueError("no shared queries")
    diffs = [per_query_b[q][metric] - per_query_a[q][metric] for q in shared]
    low, high = bootstrap_ci(diffs, seed=seed)
    return {
        "metric": metric,
        "n": len(shared),
        "mean_diff": sum(diffs) / len(diffs),
        "ci95": [low, high],
        "b_better": sum(d > 0 for d in diffs),
        "a_better": sum(d < 0 for d in diffs),
        "tied": sum(d == 0 for d in diffs),
    }


def paired_cluster_difference(
    per_query_a: Mapping[str, Mapping[str, float]],
    per_query_b: Mapping[str, Mapping[str, float]],
    metric: str,
    groups: Mapping[str, str],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Mean of (b - a) over shared queries with a cluster bootstrap 95% CI.

    Queries in the same group (e.g. questions about the same passage) are resampled together, so the
    interval accounts for the correlation between them.
    """
    import numpy as np

    shared = sorted(set(per_query_a) & set(per_query_b))
    if not shared:
        raise ValueError("no shared queries")
    missing = [q for q in shared if q not in groups]
    if missing:
        raise ValueError(f"{len(missing)} queries have no group")
    by_group: dict[str, list[float]] = {}
    for q in shared:
        by_group.setdefault(groups[q], []).append(per_query_b[q][metric] - per_query_a[q][metric])
    keys = sorted(by_group)
    sums = np.array([sum(by_group[k]) for k in keys])
    counts = np.array([len(by_group[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_resamples, len(keys)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    diffs = [per_query_b[q][metric] - per_query_a[q][metric] for q in shared]
    return {
        "metric": metric,
        "n": len(shared),
        "n_groups": len(keys),
        "mean_diff": sum(diffs) / len(diffs),
        "ci95": [float(low), float(high)],
        "bootstrap_mean": float(means.mean()),
        "b_better": sum(d > 0 for d in diffs),
        "a_better": sum(d < 0 for d in diffs),
        "tied": sum(d == 0 for d in diffs),
    }
