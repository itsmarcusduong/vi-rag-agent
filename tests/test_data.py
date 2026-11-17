import unicodedata

import pytest

from vi_rag_agent.data import build_benchmark, passage_id

CTX_A = "Hà Nội là thủ đô của Việt Nam."
CTX_B = "Phở là món ăn nổi tiếng."


def rec(id_, ctx, q, impossible=False, title="T"):
    return {"id": id_, "title": title, "context": ctx, "question": q, "is_impossible": impossible}


RECORDS = {
    "train": [rec("1", CTX_A, "Thủ đô?"), rec("2", CTX_B, "Món gì?", title="Food")],
    "validation": [
        rec(
            "1", unicodedata.normalize("NFD", CTX_A) + "  ", "Ở đâu?"
        ),  # same passage, NFD + spaces
        rec("3", CTX_B, "Không trả lời được?", impossible=True, title="Food"),
    ],
}


def test_passages_are_deduplicated_across_splits_and_unicode_forms():
    bench = build_benchmark(RECORDS)
    assert len(bench.corpus) == 2
    assert bench.qrels["validation"]["validation:1"] == {passage_id(CTX_A): 1}
    assert bench.passage_splits[passage_id(CTX_A)] == {"train", "validation"}


def test_query_ids_are_namespaced_by_split():
    bench = build_benchmark(RECORDS)
    assert "train:1" in bench.queries["train"]
    assert "validation:1" in bench.queries["validation"]


def test_answerable_subset_and_stats():
    bench = build_benchmark(RECORDS)
    queries, qrels = bench.subset("validation", answerable=True)
    assert list(queries) == ["validation:1"]
    unanswerable, _ = bench.subset("validation", answerable=False)
    assert list(unanswerable) == ["validation:3"]
    stats = bench.stats()
    assert stats["n_passages"] == 2
    assert stats["passages_in_more_than_one_split"] == 2
    assert stats["splits"]["validation"] == {
        "questions": 2,
        "answerable": 1,
        "unanswerable": 1,
        "passages": 2,
    }


def test_passage_id_is_stable_and_normalised():
    assert passage_id(CTX_A) == passage_id(unicodedata.normalize("NFD", CTX_A) + "\n")
    assert passage_id(CTX_A) != passage_id(CTX_B)
    assert passage_id(CTX_A).startswith("p") and len(passage_id(CTX_A)) == 17


def test_duplicate_question_id_in_a_split_raises():
    with pytest.raises(ValueError):
        build_benchmark({"train": [rec("1", CTX_A, "a"), rec("1", CTX_B, "b")]})
