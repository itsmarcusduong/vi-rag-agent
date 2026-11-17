"""Okapi BM25 retriever, vectorised with a sparse document-term matrix.

score(q, d) = sum over unique query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with the Lucene idf, idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)), which is always positive.
(The original Robertson idf can go negative for very common terms; ``rank_bm25`` patches that with
an epsilon floor, so its scores differ from ours by design.)

Per-(document, term) contributions are precomputed once, so a query is a sum of a few sparse
columns. Rankings are deterministic: ties are broken by document position in the corpus, and
documents with score 0 (no shared term) are never returned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from vi_rag_agent.text import tokenize


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be in [0, 1]")


class BM25Retriever:
    """Index a fixed corpus once, then answer ``search(query, k)`` calls."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        params: BM25Params | None = None,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")
        if not doc_ids:
            raise ValueError("corpus is empty")
        self.doc_ids = list(doc_ids)
        self.params = params or BM25Params()
        self.tokenizer = tokenizer

        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        for row, text in enumerate(texts):
            for token in tokenizer(text):
                rows.append(row)
                cols.append(vocab.setdefault(token, len(vocab)))
        self.vocab = vocab
        n_docs = len(texts)
        # Duplicate (row, col) pairs are summed on conversion, giving term frequencies.
        tf = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_docs, len(vocab))
        )
        tf.sum_duplicates()

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        self.avgdl = float(doc_len.mean()) if doc_len.size else 0.0
        doc_freq = np.bincount(tf.indices, minlength=len(vocab))
        self.idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        k1, b = self.params.k1, self.params.b
        norm = k1 * (1 - b + b * doc_len / self.avgdl) if self.avgdl > 0 else np.full(n_docs, k1)
        weights = tf.tocoo()
        values = (
            self.idf[weights.col] * weights.data * (k1 + 1) / (weights.data + norm[weights.row])
        )
        # Column-major layout makes "sum these term columns" cheap.
        self._weights = sparse.csc_matrix(
            (values, (weights.row, weights.col)), shape=(n_docs, len(vocab))
        )

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document (unique query terms; unknown terms ignored)."""
        term_ids = sorted({self.vocab[t] for t in self.tokenizer(query) if t in self.vocab})
        if not term_ids:
            return np.zeros(len(self.doc_ids))
        return np.asarray(self._weights[:, term_ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top ``k`` (doc_id, score) pairs, best first; only documents with score > 0."""
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.scores(query)
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        # Sort by score descending, then corpus position ascending (deterministic ties).
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in candidates[order]]
