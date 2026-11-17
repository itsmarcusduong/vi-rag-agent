import math
import unicodedata

import numpy as np
import pytest

from vi_rag_agent.bm25 import BM25Params, BM25Retriever
from vi_rag_agent.text import normalize, tokenize

DOCS = {
    "d0": "Hà Nội là thủ đô của Việt Nam.",
    "d1": "Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam.",
    "d2": "Phở là món ăn nổi tiếng của Hà Nội.",
    "d3": "Sông Hồng chảy qua Hà Nội và đổ ra biển.",
}


def reference_scores(docs: list[str], query: str, k1: float, b: float) -> list[float]:
    """Direct transcription of the formula with plain loops (independent of the sparse code)."""
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(d) for d in tokenized) / n
    out = []
    for doc in tokenized:
        total = 0.0
        for term in set(tokenize(query)):
            df = sum(1 for d in tokenized if term in d)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = doc.count(term)
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        out.append(total)
    return out


def make(params=None):
    return BM25Retriever(list(DOCS), list(DOCS.values()), params)


@pytest.mark.parametrize("query", ["Hà Nội", "thủ đô Việt Nam", "món ăn", "biển Hà Nội Hà Nội"])
@pytest.mark.parametrize(("k1", "b"), [(1.5, 0.75), (0.9, 0.4), (1.2, 0.0), (2.0, 1.0)])
def test_scores_match_reference_formula(query, k1, b):
    retriever = make(BM25Params(k1, b))
    expected = reference_scores(list(DOCS.values()), query, k1, b)
    np.testing.assert_allclose(retriever.scores(query), expected, rtol=1e-12)


def test_hand_computed_single_term():
    # "phở": df = 1, N = 4 -> idf = ln(1 + 3.5 / 1.5). In d2: tf = 1, |d2| = 9 tokens,
    # doc lengths 8, 12, 9, 10 -> avgdl = 9.75 -> norm = 1.5 * (0.25 + 0.75 * 9 / 9.75)
    idf = math.log(1 + 3.5 / 1.5)
    norm = 1.5 * (0.25 + 0.75 * 9 / 9.75)
    expected = idf * 1 * 2.5 / (1 + norm)
    scores = make().scores("Phở")
    assert scores[2] == pytest.approx(expected, rel=1e-12)
    assert scores[0] == scores[1] == scores[3] == 0.0


def test_search_orders_by_score_and_drops_zero_scores():
    results = make().search("món ăn Hà Nội", k=10)
    ids = [d for d, _ in results]
    assert ids[0] == "d2"
    assert "d1" not in ids  # shares no term with the query
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_corpus_order():
    retriever = BM25Retriever(["b", "a", "c"], ["mèo", "mèo", "chó"])
    assert [d for d, _ in retriever.search("mèo", k=2)] == ["b", "a"]


def test_unknown_query_returns_nothing():
    assert make().search("blockchain", k=5) == []


def test_nfc_and_nfd_inputs_match():
    nfd_query = unicodedata.normalize("NFD", "Hà Nội")
    assert nfd_query != "Hà Nội"
    assert make().search(nfd_query, k=4) == make().search("Hà Nội", k=4)
    assert normalize(nfd_query) == "hà nội"


def test_repeated_query_terms_count_once():
    retriever = make()
    np.testing.assert_allclose(retriever.scores("Hà Nội Hà Nội"), retriever.scores("Hà Nội"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"doc_ids": [], "texts": []},
        {"doc_ids": ["a"], "texts": ["x", "y"]},
        {"doc_ids": ["a", "a"], "texts": ["x", "y"]},
    ],
)
def test_invalid_corpus_raises(kwargs):
    with pytest.raises(ValueError):
        BM25Retriever(**kwargs)


def test_invalid_params_and_k_raise():
    with pytest.raises(ValueError):
        BM25Params(k1=-1)
    with pytest.raises(ValueError):
        BM25Params(b=1.5)
    with pytest.raises(ValueError):
        make().search("Hà Nội", k=0)


def test_tokenize_drops_punctuation_and_lowercases():
    assert tokenize("Hà Nội, Việt Nam!") == ["hà", "nội", "việt", "nam"]
