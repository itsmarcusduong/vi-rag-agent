import pytest

from vi_rag_agent.chunking import ChunkingConfig, chunk_text, clean_text

SENTENCES = " ".join(f"Câu số {i} nói về Hà Nội và các địa danh nổi tiếng." for i in range(40))


def test_empty_and_whitespace_return_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Hà Nội là thủ đô.") == ["Hà Nội là thủ đô."]


def test_chunks_respect_size_limit():
    config = ChunkingConfig(size=120, overlap=20, min_length=30)
    chunks = chunk_text(SENTENCES, config)
    assert len(chunks) > 1
    assert all(len(chunk) <= config.size for chunk in chunks)


def test_prefers_sentence_boundaries():
    chunks = chunk_text(SENTENCES)
    assert all(chunk.endswith(".") for chunk in chunks[:-1])


def test_consecutive_chunks_overlap():
    chunks = chunk_text(SENTENCES)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current[:20] in previous


def test_every_word_is_covered():
    chunks = chunk_text(SENTENCES)
    joined = "\n".join(chunks)
    assert all(word in joined for word in SENTENCES.split())


def test_text_without_separators_terminates_with_bounded_chunks():
    config = ChunkingConfig(size=100, overlap=10, min_length=20)
    chunks = chunk_text("a" * 1000, config)
    assert len(chunks) >= 10
    assert all(len(chunk) <= config.size for chunk in chunks)


def test_is_deterministic():
    assert chunk_text(SENTENCES) == chunk_text(SENTENCES)


def test_clean_text_collapses_whitespace_and_rules():
    assert clean_text("a   b\n\n\n\nc ---- d") == "a b\n\nc  d"


@pytest.mark.parametrize(
    "kwargs",
    [{"size": 0}, {"size": 10, "overlap": 10}, {"size": 10, "overlap": -1}, {"min_length": -1}],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


def test_chunks_start_at_word_boundaries():
    words = set(SENTENCES.split())
    chunks = chunk_text(SENTENCES)
    assert all(chunk.split()[0] in words for chunk in chunks)


def test_early_separator_does_not_cause_text_loss():
    # A separator right at the start must not produce a tiny dropped chunk that skips text.
    config = ChunkingConfig(size=100, overlap=20, min_length=20)
    text = "Ab. " + "x" * 500
    chunks = chunk_text(text, config)
    joined = "\n".join(chunks)
    cleaned = clean_text(text)
    assert all(cleaned[i : i + 10] in joined for i in range(len(cleaned) - 9))
