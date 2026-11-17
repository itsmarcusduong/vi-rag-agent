import numpy as np
import pytest

from vi_rag_agent.dense import ENCODERS, DenseRetriever, EncoderSpec, l2_normalize, prepare

faiss = pytest.importorskip("faiss")

DOCS = ["d0", "d1", "d2", "d3"]
EMB = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.float32)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_returns_cosine_ranked_results(backend):
    retriever = DenseRetriever(DOCS, EMB, backend=backend)
    (hits,) = retriever.search_embeddings(np.array([[1.0, 0.2, 0.0]]), k=3)
    assert [d for d, _ in hits] == ["d0", "d2", "d1"]
    expected = np.array([1.0, 0.2, 0.0]) @ (EMB[0] / 1.0) / np.linalg.norm([1.0, 0.2, 0.0])
    assert hits[0][1] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_ties_broken_by_corpus_order(backend):
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    retriever = DenseRetriever(["b", "a", "c"], emb, backend=backend)
    (hits,) = retriever.search_embeddings(np.array([[1.0, 0.0]]), k=2)
    assert [d for d, _ in hits] == ["b", "a"]


def test_numpy_and_faiss_backends_agree_on_random_data():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, 16)).astype(np.float32)
    emb[10] = emb[11]  # exact duplicate passage -> exact tie
    queries = rng.normal(size=(25, 16)).astype(np.float32)
    queries[0] = emb[10]
    ids = [f"p{i}" for i in range(300)]
    a = DenseRetriever(ids, emb, "numpy").search_embeddings(queries, k=10)
    b = DenseRetriever(ids, emb, "faiss").search_embeddings(queries, k=10)
    assert [[d for d, _ in r] for r in a] == [[d for d, _ in r] for r in b]
    assert [d for d, _ in a[0][:2]] == ["p10", "p11"]


def test_k_larger_than_corpus_returns_all():
    (hits,) = DenseRetriever(DOCS, EMB).search_embeddings(np.array([[0.0, 0.0, 1.0]]), k=50)
    assert len(hits) == 4 and hits[0][0] == "d3"


def test_prepare_applies_prefixes_and_segmentation():
    e5 = ENCODERS["multilingual-e5-base"]
    assert prepare(["Hà Nội"], e5, "query") == ["query: Hà Nội"]
    assert prepare(["Hà Nội"], e5, "passage") == ["passage: Hà Nội"]
    bkai = ENCODERS["bkai-bi-encoder"]
    seg = lambda t: t.replace("Hà Nội", "Hà_Nội")  # noqa: E731
    assert prepare(["Hà Nội đẹp"], bkai, "query", segmenter=seg) == ["Hà_Nội đẹp"]
    with pytest.raises(ValueError, match="word-segmented"):
        prepare(["x"], bkai, "query")
    with pytest.raises(ValueError):
        prepare(["x"], EncoderSpec("m"), "document")


@pytest.mark.parametrize(
    ("ids", "emb", "backend"),
    [
        (["a"], np.ones((2, 3)), "numpy"),
        (["a", "a"], np.ones((2, 3)), "numpy"),
        (["a"], np.ones((1, 3)), "annoy"),
        (["a"], np.zeros((1, 3)), "numpy"),
        (["a"], np.ones(3), "numpy"),
    ],
)
def test_invalid_inputs_raise(ids, emb, backend):
    with pytest.raises(ValueError):
        DenseRetriever(ids, emb, backend=backend)


def test_l2_normalize_unit_norm():
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(EMB), axis=1), 1.0, rtol=1e-6)


def test_faiss_near_tie_at_cutoff_matches_numpy():
    # cos(d0) ~ 0.9999988 and cos(d1) ~ 0.9999992: d1 is higher in float32, but both round to
    # 0.999999, so the backends must treat them as tied and return d0 (corpus order). FAISS only
    # sees d0 if it searches deeper than k.
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = np.stack([np.array([1.0, 1.549e-3, 0.0]), np.array([1.0, 1.265e-3, 0.0]), [0, 1, 0]])
    query = base[None, :]
    a = DenseRetriever(["d0", "d1", "d2"], emb, "numpy").search_embeddings(query, k=1)
    b = DenseRetriever(["d0", "d1", "d2"], emb, "faiss").search_embeddings(query, k=1)
    assert [d for d, _ in a[0]] == [d for d, _ in b[0]] == ["d0"]
