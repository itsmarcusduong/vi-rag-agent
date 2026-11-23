"""Turn UIT-ViQuAD 2.0 into a passage-retrieval benchmark.

ViQuAD is a reading-comprehension dataset: each question is written about one Wikipedia passage.
For retrieval, every unique passage across all splits becomes a document, each question becomes a
query, and its gold document is the passage it was written about (binary relevance).

Choices fixed before any experiment:
* Passages are deduplicated on exact NFC text; the document ID is a hash of that text, so IDs are
  stable across runs and machines.
* All splits contribute to the corpus, so each query competes with the full passage pool.
* Answerable and unanswerable questions are kept apart. Unanswerable questions were written to look
  answerable from their passage, so the passage is still the right one to retrieve, but they are
  reported separately rather than mixed into the headline number.

Source: Hugging Face mirror ``taidng/UIT-ViQuAD2.0`` (Nguyen et al., COLING 2020; VLSP 2021).
The data is downloaded at run time and never stored in this repository.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

HF_DATASET = "taidng/UIT-ViQuAD2.0"
SPLITS = ("train", "validation", "test")


def passage_id(text: str) -> str:
    """Stable ID: first 16 hex chars of SHA-1 over the NFC-normalised, stripped text."""
    canonical = unicodedata.normalize("NFC", text).strip()
    return "p" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RetrievalBenchmark:
    corpus: dict[str, str] = field(default_factory=dict)  # doc_id -> passage text
    titles: dict[str, str] = field(default_factory=dict)  # doc_id -> article title
    queries: dict[str, dict[str, str]] = field(default_factory=dict)  # split -> qid -> question
    qrels: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)  # split -> qid -> rel
    answerable: dict[str, dict[str, bool]] = field(default_factory=dict)  # split -> qid -> bool
    passage_splits: dict[str, set[str]] = field(default_factory=dict)  # doc_id -> splits using it
    answers: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # split -> qid -> golds

    def subset(self, split: str, answerable: bool) -> tuple[dict[str, str], dict[str, dict]]:
        """Queries and qrels of one split, filtered by answerability."""
        keep = [q for q, a in self.answerable[split].items() if a is answerable]
        return (
            {q: self.queries[split][q] for q in keep},
            {q: self.qrels[split][q] for q in keep},
        )

    def stats(self) -> dict:
        """Counts that belong in every results file, so readers can judge the benchmark."""
        per_split = {}
        for split, flags in self.answerable.items():
            n_ans = sum(flags.values())
            per_split[split] = {
                "questions": len(flags),
                "answerable": n_ans,
                "unanswerable": len(flags) - n_ans,
                "passages": sum(1 for s in self.passage_splits.values() if split in s),
            }
        shared = sum(1 for s in self.passage_splits.values() if len(s) > 1)
        return {
            "n_passages": len(self.corpus),
            "n_titles": len(set(self.titles.values())),
            "passages_in_more_than_one_split": shared,
            "splits": per_split,
        }


def build_benchmark(records_by_split: Mapping[str, Iterable[Mapping]]) -> RetrievalBenchmark:
    """Build the benchmark from ViQuAD-style records (``id``, ``title``, ``context``,
    ``question``, ``is_impossible``)."""
    bench = RetrievalBenchmark()
    for split, records in records_by_split.items():
        bench.queries[split] = {}
        bench.qrels[split] = {}
        bench.answerable[split] = {}
        bench.answers[split] = {}
        for record in records:
            text = unicodedata.normalize("NFC", record["context"]).strip()
            doc_id = passage_id(text)
            bench.corpus.setdefault(doc_id, text)
            bench.titles.setdefault(doc_id, record["title"])
            bench.passage_splits.setdefault(doc_id, set()).add(split)

            qid = f"{split}:{record['id']}"
            if qid in bench.queries[split]:
                raise ValueError(f"duplicate question id {qid}")
            bench.queries[split][qid] = record["question"]
            bench.qrels[split][qid] = {doc_id: 1}
            bench.answerable[split][qid] = not bool(record["is_impossible"])
            golds = (record.get("answers") or {}).get("text") or []
            bench.answers[split][qid] = [g for g in golds if g and g.strip()]
    return bench


def load_viquad(splits: Iterable[str] = SPLITS) -> RetrievalBenchmark:  # pragma: no cover
    """Download ViQuAD 2.0 from the Hugging Face Hub (needs network and the ``data`` extra)."""
    from datasets import load_dataset

    columns = ("id", "title", "context", "question", "is_impossible", "answers")
    records = {}
    for split in splits:
        ds = load_dataset(HF_DATASET, split=split).select_columns(list(columns))
        records[split] = ds.to_list()
    return build_benchmark(records)
