"""Answer-level metrics for extractive-style QA (SQuAD conventions adapted to Vietnamese).

Normalisation: NFC, lowercase, punctuation removed, whitespace collapsed. English article removal
from the SQuAD script is not applied. F1 is computed over syllable tokens. Both metrics take the
maximum over the gold answers of a question.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    if not golds:
        raise ValueError("at least one gold answer is required")
    norm = normalize_answer(prediction)
    return float(any(norm == normalize_answer(g) for g in golds))


def f1_score(prediction: str, golds: Sequence[str]) -> float:
    if not golds:
        raise ValueError("at least one gold answer is required")
    return max(_f1(prediction, g) for g in golds)
