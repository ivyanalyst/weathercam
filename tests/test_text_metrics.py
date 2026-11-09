import pytest

from weathercam.core.text_metrics import (
    cer,
    corpus_cer,
    exact_match_rate,
    levenshtein,
    position_accuracy,
)


def test_levenshtein_identical():
    assert levenshtein("ABC123", "ABC123") == 0


def test_levenshtein_both_empty():
    assert levenshtein("", "") == 0


def test_levenshtein_one_empty():
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3


def test_levenshtein_kitten_sitting():
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_flaw_lawn():
    assert levenshtein("flaw", "lawn") == 2


def test_levenshtein_saturday_sunday():
    assert levenshtein("saturday", "sunday") == 3


def test_levenshtein_intention_execution():
    assert levenshtein("intention", "execution") == 5


def test_levenshtein_swap_costs_two():
    assert levenshtein("ab", "ba") == 2


def test_levenshtein_symmetric():
    assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")


def test_levenshtein_case_sensitive():
    assert levenshtein("ABC", "abc") == 3


def test_levenshtein_unicode():
    assert levenshtein("café", "cafe") == 1


def test_cer_perfect():
    assert cer("ABC123", "ABC123") == 0.0


def test_cer_one_substitution():
    assert cer("ABC123", "ABD123") == pytest.approx(1 / 6)


def test_cer_extra_characters():
    assert cer("ABC", "ABCDE") == pytest.approx(2 / 3)


def test_cer_can_exceed_one():
    assert cer("AB", "XYZW") == pytest.approx(2.0)


def test_cer_empty_hypothesis():
    assert cer("ABC", "") == pytest.approx(1.0)


def test_cer_empty_reference_raises():
    with pytest.raises(ValueError):
        cer("", "ABC")
    with pytest.raises(ValueError):
        cer("", "")


def test_corpus_cer_weights_by_length():
    assert corpus_cer(["AB", "CDEF"], ["AX", "CDEF"]) == pytest.approx(1 / 6)


def test_corpus_cer_empty_reference_item_allowed():
    assert corpus_cer(["", "AB"], ["X", "AB"]) == pytest.approx(0.5)


def test_corpus_cer_length_mismatch_raises():
    with pytest.raises(ValueError):
        corpus_cer(["AB"], ["AB", "CD"])


def test_corpus_cer_empty_input_raises():
    with pytest.raises(ValueError):
        corpus_cer([], [])


def test_corpus_cer_zero_reference_total_raises():
    with pytest.raises(ValueError):
        corpus_cer(["", ""], ["A", ""])


def test_exact_match_rate_all_equal():
    assert exact_match_rate(["AB", "CD"], ["AB", "CD"]) == 1.0


def test_exact_match_rate_fraction():
    assert exact_match_rate(["AB", "CD", "EF", "GH"], ["AB", "CX", "EF", "GX"]) == pytest.approx(0.5)


def test_exact_match_rate_none_equal():
    assert exact_match_rate(["AB"], ["XY"]) == 0.0


def test_exact_match_rate_unicode():
    assert exact_match_rate(["皖A12345"], ["皖A12345"]) == 1.0
    assert exact_match_rate(["皖A12345"], ["沪A12345"]) == 0.0


def test_exact_match_rate_length_mismatch_raises():
    with pytest.raises(ValueError):
        exact_match_rate(["AB"], ["AB", "CD"])


def test_exact_match_rate_empty_input_raises():
    with pytest.raises(ValueError):
        exact_match_rate([], [])


def test_position_accuracy_basic():
    assert position_accuracy(["ABC", "ABD"], ["ABC", "XBE"]) == pytest.approx([0.5, 1.0, 0.5])


def test_position_accuracy_short_hypothesis_counts_missing_as_wrong():
    assert position_accuracy(["ABC"], ["A"]) == pytest.approx([1.0, 0.0, 0.0])


def test_position_accuracy_long_hypothesis_ignores_extra_characters():
    assert position_accuracy(["AB"], ["ABCD"]) == pytest.approx([1.0, 1.0])


def test_position_accuracy_unicode_first_position():
    assert position_accuracy(["皖A", "沪B"], ["皖A", "皖B"]) == pytest.approx([0.5, 1.0])


def test_position_accuracy_different_reference_lengths_raises():
    with pytest.raises(ValueError):
        position_accuracy(["AB", "ABC"], ["AB", "ABC"])


def test_position_accuracy_empty_references_raises():
    with pytest.raises(ValueError):
        position_accuracy([], [])


def test_position_accuracy_empty_strings_raise():
    with pytest.raises(ValueError):
        position_accuracy([""], [""])


def test_position_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        position_accuracy(["AB"], ["AB", "CD"])
