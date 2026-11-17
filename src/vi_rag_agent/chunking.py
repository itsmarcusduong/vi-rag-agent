"""Sentence-aware character chunking for Vietnamese text.

Chunks are at most ``size`` characters, prefer to end at a sentence or clause boundary, and
overlap by roughly ``overlap`` characters so a fact straddling a cut is still retrievable.
Characters, not tokens, are counted: this keeps the module dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters (characters).

    ``min_length`` drops chunks that are too short to be useful. With ``overlap >= min_length``
    the final chunk is always long enough to be kept, so no tail text is lost.
    """

    size: int = 256
    overlap: int = 48
    min_length: int = 40

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")


# (separator, characters of the separator that stay inside the chunk)
_BOUNDARIES: tuple[tuple[str, int], ...] = ((". ", 1), (", ", 1), (" ", 0))


def clean_text(text: str) -> str:
    """Collapse whitespace and strip markdown rules (``---``, ``***``)."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-{3,}|\*{3,}", "", text)
    return text.strip()


def _snap_to_boundary(text: str, start: int, end: int, min_span: int) -> int:
    """Move ``end`` back to the nearest sentence/clause/word boundary.

    A boundary is accepted only if the chunk keeps at least ``min_span`` characters; otherwise
    a tiny chunk would be dropped and the text after it silently skipped.
    """
    for separator, keep in _BOUNDARIES:
        index = text.rfind(separator, start, end)
        if index != -1 and index + keep - start >= min_span:
            return index + keep
    return end


def _align_to_word_start(text: str, position: int, limit: int) -> int:
    """Advance ``position`` to the next word start so a chunk never begins mid-word."""
    if position <= 0 or text[position - 1].isspace():
        return position
    next_space = text.find(" ", position, limit)
    return next_space + 1 if next_space != -1 else position


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """Split ``text`` into overlapping chunks. Returns ``[]`` for empty input."""
    config = config or ChunkingConfig()
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) < config.min_length:
        return [cleaned]

    chunks: list[str] = []
    length = len(cleaned)
    min_span = max(config.min_length, config.size // 2)
    start = 0
    while start < length:
        end = min(start + config.size, length)
        if end < length:
            end = _snap_to_boundary(cleaned, start, end, min_span)

        chunk = cleaned[start:end].strip()
        if len(chunk) >= config.min_length:
            chunks.append(chunk)
        if end >= length:
            break

        next_start = end - config.overlap
        if next_start <= start:
            next_start = start + max(1, config.size - config.overlap)
        start = _align_to_word_start(cleaned, next_start, end)
    return chunks
