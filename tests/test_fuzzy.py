import pytest

from weathercam.plates.fuzzy import match_plate, plate_distance

PLATE = "皖A12345"


def test_distance_identical_is_zero():
    assert plate_distance(PLATE, PLATE) == 0


def test_distance_one_substitution():
    assert plate_distance("皖A12346", PLATE) == 1


def test_distance_one_insertion():
    assert plate_distance("皖A123456", PLATE) == 1


def test_distance_one_deletion():
    assert plate_distance("皖A1235", PLATE) == 1


def test_distance_adjacent_swap_costs_two():
    assert plate_distance("皖A12354", PLATE) == 2


def test_match_exactly_at_limit_and_one_below():
    reading = "皖A12367"
    assert plate_distance(reading, PLATE) == 2
    assert match_plate(reading, PLATE, 2) is True
    assert match_plate(reading, PLATE, 1) is False


def test_empty_reading_never_matches():
    assert match_plate("", PLATE, 10) is False


def test_negative_limit_raises():
    with pytest.raises(ValueError):
        match_plate(PLATE, PLATE, -1)


def test_zero_limit_means_exact_match_only():
    assert match_plate(PLATE, PLATE, 0) is True
    assert match_plate("皖A12346", PLATE, 0) is False
