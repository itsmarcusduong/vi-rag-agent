"""Dense retrieval with bi-encoders and exact inner-product search.

Embeddings are L2-normalised, so inner product equals cosine similarity. Search is exact: either a
NumPy matrix product or a FAISS ``IndexFlatIP`` (identical results; FAISS pays off once the corpus
outgrows memory-friendly matrix products). Approximate indexes (HNSW, IVF-PQ) are out of scope until
the corpus is large enough for them to matter.

Model-specific input handling lives in ``ENCODERS``: multilingual-e5 expects ``query: `` /
``passage: `` prefixes, and the BKAI bi-encoder (PhoBERT) expects word-segmented input. Getting this
wrong silently degrades retrieval, so it is configuration, not something each script re-implements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EncodeFn = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class EncoderSpec:
    """How to prepare text for one embedding model."""

    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    word_segment: bool = False
    max_seq_length: int = 256
    license: str = "not stated"


ENCODERS: dict[str, EncoderSpec] = {
    # PhoBERT vocabulary: needs word-segmented input (M3.1 ablation, #18). M3 used raw text.
    "vietnamese-sbert": EncoderSpec(
        "keepitreal/vietnamese-sbert", word_segment=True, max_seq_length=256
    ),
    "bkai-bi-encoder": EncoderSpec(
        "bkai-foundation-models/vietnamese-bi-encoder",
        word_segment=True,
        max_seq_length=256,
        license="Apache-2.0",
    ),
    "multilingual-e5-base": EncoderSpec(
        "intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=512,
        license="MIT",
    ),
}


def prepare(
    texts: Sequence[str],
    spec: EncoderSpec,
    kind: str,
    segmenter: Callable[[str], str] | None = None,
) -> list[str]:
    """Apply word segmentation (if the model needs it) and the query/passage prefix."""
    if kind not in ("query", "passage"):
        raise ValueError("kind must be 'query' or 'passage'")
    if spec.word_segment and segmenter is None:
        raise ValueError(f"{spec.model_name} needs word-segmented input; pass a segmenter")
    prefix = spec.query_prefix if kind == "query" else spec.passage_prefix
    out = []
    for text in texts:
        if spec.word_segment:
            text = segmenter(text)
        out.append(prefix + text)
    return out


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("expected a 2-D array of embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding vector cannot be normalised")
    return matrix / norms


class DenseRetriever:
    """Exact cosine-similarity search over pre-computed passage embeddings."""

    def __init__(self, doc_ids: Sequence[str], embeddings: np.ndarray, backend: str = "numpy"):
        if len(doc_ids) != len(embeddings):
            raise ValueError("one embedding per document is required")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")
        if backend not in ("numpy", "faiss"):
            raise ValueError("backend must be 'numpy' or 'faiss'")
        self.doc_ids = list(doc_ids)
        self.embeddings = l2_normalize(embeddings)
        self.backend = backend
        self._index = None
        if backend == "faiss":
            import faiss

            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)

    def search_embeddings(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Top ``k`` (doc_id, cosine) per query, best first; ties broken by corpus order."""
        if k <= 0:
            raise ValueError("k must be positive")
        queries = l2_normalize(query_embeddings)
        k = min(k, len(self.doc_ids))
        if self.backend == "faiss":
            # Ask FAISS for a few extra hits so ties at the cut-off are resolved by corpus order.
            depth = min(len(self.doc_ids), k + 8)
            scores, indices = self._index.search(queries, depth)
        else:
            scores = queries @ self.embeddings.T
            indices = None
        results = []
        for row in range(len(queries)):
            if indices is None:
                candidates = np.arange(len(self.doc_ids))
                cand_scores = scores[row]
            else:
                candidates = indices[row]
                cand_scores = scores[row]
            # Round to float32 precision so the two backends order near-ties identically.
            rounded = np.round(cand_scores.astype(np.float64), 6)
            order = np.lexsort((candidates, -rounded))[:k]
            results.append([(self.doc_ids[candidates[i]], float(cand_scores[i])) for i in order])
        return results
