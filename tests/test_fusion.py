import pytest

from vi_rag_agent.fusion import reciprocal_rank_fusion


def test_hand_computed_scores_and_order():
    # k = 60. a: 1/61 + 1/63; b: 1/62 + 1/61; c: 1/63; d: 1/62
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [d for d, _ in fused] == ["b", "a", "d", "c"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)


def test_exact_ties_break_by_best_rank_then_id():
    # x and y have identical scores (1/61 + 1/62) and the same best rank: ID decides.
    assert [d for d, _ in reciprocal_rank_fusion([["y", "x"], ["x", "y"]])] == ["x", "y"]
    # q: 1/61 + 1/63 = 0.032266 > p: 2/62 = 0.032258 > s: 1/61 > r: 1/63
    fused = reciprocal_rank_fusion([["q", "p", "r"], ["s", "p", "q"]])
    assert [d for d, _ in fused] == ["q", "p", "s", "r"]


def test_weights_and_top_n():
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=0, weights=[1.0, 3.0], top_n=1)
    assert fused == [("b", 3.0)]


def test_single_ranking_preserves_order():
    assert [d for d, _ in reciprocal_rank_fusion([["c", "a", "b"]])] == ["c", "a", "b"]


@pytest.mark.parametrize(
    ("rankings", "kwargs"),
    [
        ([["a", "a"]], {}),
        ([["a"]], {"weights": [1.0, 1.0]}),
        ([["a"]], {"k": -1}),
        ([["a"]], {"weights": [-1.0]}),
    ],
)
def test_invalid_input_raises(rankings, kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, **kwargs)


def test_equal_scores_prefer_document_with_better_best_rank():
    # k = 0: x = 1/1, w = 1/1, a = 1/2 + 1/2 = 1 (exact). a reached only rank 2, so it goes last
    # even though its ID sorts first.
    fused = reciprocal_rank_fusion([["x", "a"], ["w", "a"]], k=0)
    assert [d for d, _ in fused] == ["w", "x", "a"]
    assert all(score == 1.0 for _, score in fused)
