import unicodedata

import pytest

from vi_rag_agent.qa import exact_match, f1_score, normalize_answer


def test_normalize_lowercases_strips_punctuation_and_nfc():
    nfd = unicodedata.normalize("NFD", "Hà Nội")
    assert normalize_answer(f"  {nfd}, Việt Nam!  ") == "hà nội việt nam"


def test_exact_match_after_normalisation_and_max_over_golds():
    assert exact_match("Hà Nội.", ["Thành phố Hà Nội", "hà nội"]) == 1.0
    assert exact_match("Hà Nội", ["Hải Phòng"]) == 0.0


def test_f1_hand_computed():
    # pred: "thủ đô hà nội" (4 tokens), gold: "hà nội" (2 tokens); overlap 2
    # precision 2/4, recall 2/2 -> F1 = 2 * 0.5 * 1 / 1.5 = 2/3
    assert f1_score("thủ đô Hà Nội", ["Hà Nội"]) == pytest.approx(2 / 3)


def test_f1_counts_repeated_tokens_with_multiplicity():
    # pred "nội nội" vs gold "hà nội": common = {nội: 1}; precision 1/2, recall 1/2 -> 0.5
    assert f1_score("nội nội", ["hà nội"]) == pytest.approx(0.5)


def test_f1_takes_max_over_golds_and_zero_without_overlap():
    assert f1_score("năm 1945", ["1945", "tháng 9 năm 1945"]) == pytest.approx(
        max(2 * (1 / 2) * 1 / (1 / 2 + 1), 2 * 1 * (2 / 4) / (1 + 2 / 4))
    )
    assert f1_score("Hải Phòng", ["Hà Nội"]) == 0.0


def test_empty_prediction_scores_zero_and_empty_gold_list_raises():
    assert f1_score("", ["Hà Nội"]) == 0.0
    assert exact_match("", ["Hà Nội"]) == 0.0
    with pytest.raises(ValueError):
        f1_score("x", [])


def test_f1_uses_token_multiplicity_on_both_sides():
    # pred "nội nội" (2), gold "nội nội hà" (3): common {nội: 2} -> P = 1, R = 2/3 -> F1 = 0.8
    assert f1_score("nội nội", ["nội nội hà"]) == pytest.approx(0.8)


def test_scores_use_the_best_gold_not_the_first():
    assert f1_score("Hà Nội", ["Hải Phòng", "Hà Nội"]) == 1.0
    assert exact_match("Hà Nội", ["Hải Phòng", "Hà Nội"]) == 1.0
