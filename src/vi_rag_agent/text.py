"""Text normalisation and tokenisation for Vietnamese lexical retrieval.

Vietnamese text mixes precomposed (NFC) and decomposed (NFD) diacritics depending on the source, so
"Hà Nội" can be two different byte strings. Everything is normalised to NFC before tokenising,
otherwise BM25 would treat the same syllable as two terms.

Tokens are lowercase syllables (``\\w+`` runs). Vietnamese words are often multi-syllable
("Hà Nội"), so word segmentation (e.g. pyvi or underthesea) is a later ablation, not the baseline.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def normalize(text: str) -> str:
    """NFC-normalise and lowercase."""
    return unicodedata.normalize("NFC", text).lower()


def tokenize(text: str) -> list[str]:
    """Split normalised text into syllable tokens; punctuation is dropped."""
    return _TOKEN.findall(normalize(text))
