"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

score(d) = sum over rankings r containing d of weight_r / (k + rank_r(d)), rank 1-based.
Only ranks are used, so BM25 scores and cosine similarities never need to be put on one scale.
Ties are broken by the best (lowest) rank the document reached in any input, then by document ID,
so the output is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: float = 60.0,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of document IDs into one list of (doc_id, fused score), best first."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking contains duplicate document IDs")
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    fused = sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
    return fused if top_n is None else fused[:top_n]
