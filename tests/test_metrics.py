"""Metric tests. Expected values are derived by hand in the comments, not by calling the code."""

import math

import pytest

from vi_rag_agent.metrics import (
    evaluate,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Relevant: d1 (rank 2) and d5 (rank 4). d9 is judged non-relevant (grade 0).
RANKING = ["d3", "d1", "d2", "d5"]
QREL = {"d1": 1, "d5": 1, "d9": 0}


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 0.5), (4, 1.0), (10, 1.0)])
def test_recall(k, expected):
    assert recall_at_k(RANKING, QREL, k) == expected


@pytest.mark.parametrize(
    ("k", "expected"), [(1, 0.0), (2, 1 / 2), (3, 1 / 3), (4, 2 / 4), (10, 2 / 10)]
)
def test_precision_uses_k_as_denominator(k, expected):
    assert precision_at_k(RANKING, QREL, k) == pytest.approx(expected)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (10, 1.0)])
def test_hit(k, expected):
    assert hit_at_k(RANKING, QREL, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (10, 0.5)])
def test_reciprocal_rank_is_one_based_and_cut_at_k(k, expected):
    assert reciprocal_rank(RANKING, QREL, k) == expected


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["d1", "d3"], QREL, 10) == 1.0


def test_ndcg_binary_at_3():
    # DCG@3 = 1/log2(3); IDCG@3 (two relevant docs) = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))  # 0.386853
    assert ndcg_at_k(RANKING, QREL, 3) == pytest.approx(expected)
    assert ndcg_at_k(RANKING, QREL, 3) == pytest.approx(0.386853, abs=1e-6)


def test_ndcg_binary_at_4():
    # DCG@4 = 1/log2(3) + 1/log2(5); IDCG unchanged
    assert ndcg_at_k(RANKING, QREL, 4) == pytest.approx(0.650921, abs=1e-6)


def test_ndcg_graded_uses_linear_gain_and_ideal_ordering():
    # ranking a(1), b(3), c(0): DCG = 1/1 + 3/log2(3) + 0
    # ideal grades sorted 3, 2, 1 (a, b, x): IDCG = 3/1 + 2/log2(3) + 1/log2(4)
    qrel = {"a": 1, "b": 3, "x": 2}
    assert ndcg_at_k(["a", "b", "c"], qrel, 3) == pytest.approx(0.607492, abs=1e-6)


def test_perfect_ranking_scores_one():
    assert ndcg_at_k(["d1", "d5", "d3"], QREL, 10) == pytest.approx(1.0)
    assert recall_at_k(["d1", "d5"], QREL, 2) == 1.0


def test_empty_ranking_scores_zero():
    assert recall_at_k([], QREL, 5) == 0.0
    assert reciprocal_rank([], QREL, 5) == 0.0
    assert ndcg_at_k([], QREL, 5) == 0.0


def test_duplicates_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        recall_at_k(["d1", "d1"], QREL, 2)


def test_invalid_k_and_no_relevant_raise():
    with pytest.raises(ValueError):
        recall_at_k(RANKING, QREL, 0)
    with pytest.raises(ValueError):
        recall_at_k(RANKING, {"d1": 0}, 5)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKING, {}, 5)


def test_evaluate_means_skips_and_missing():
    run = {"q1": RANKING, "q2": ["x"], "extra": ["d1"]}
    qrels = {
        "q1": QREL,  # recall@5 = 1, mrr = 0.5
        "q2": {"y": 1},  # nothing found: all zeros
        "q3": {"z": 0},  # no relevant document: skipped, not scored
        "q4": {"w": 1},  # absent from run: scored as empty ranking (zeros)
    }
    result = evaluate(run, qrels, ks=(1, 5), mrr_k=10, ndcg_k=10)
    assert result.n_queries == 3
    assert result.skipped_no_relevant == ["q3"]
    assert result.missing_from_run == ["q4"]
    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@5"] == pytest.approx(1 / 3)
    assert result.metrics["hit@5"] == pytest.approx(1 / 3)
    assert result.metrics["mrr@10"] == pytest.approx(0.5 / 3)
    assert result.metrics["ndcg@10"] == pytest.approx(0.650921 / 3, abs=1e-6)
    assert "extra" not in result.per_query
    assert result.to_dict()["n_skipped_no_relevant"] == 1


def test_evaluate_without_any_relevant_query_raises():
    with pytest.raises(ValueError):
        evaluate({"q": ["a"]}, {"q": {"a": 0}})
