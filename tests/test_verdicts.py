import pytest

from weathercam.review.verdicts import (
    Record,
    append_record,
    count_verdicts,
    elements_for,
    latest,
    read_log,
    record_from_line,
    record_to_line,
)


def rec(**over):
    base = dict(
        source="veri",
        image="0002_c002_00030600_0.jpg",
        split="test",
        element="colour",
        model_value="red",
        verdict="correct",
        timestamp="2026-11-11T20:00:00",
    )
    base.update(over)
    return Record(**base)


def test_elements_veri():
    assert elements_for("veri") == (
        "vehicle_visible", "colour", "type", "brand", "view", "plate_box", "plate_text",
    )


def test_elements_ccpd():
    assert elements_for("ccpd") == ("detector_box", "label_box", "plate_text")


def test_elements_unknown_source_raises():
    with pytest.raises(ValueError):
        elements_for("dawn")


def test_valid_record_is_accepted():
    r = rec()
    assert r.verdict == "correct" and r.corrected is None


def test_element_not_in_source_rejected():
    with pytest.raises(ValueError):
        rec(source="ccpd", element="colour")


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        rec(source="dawn")


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        rec(split="holdout")


def test_empty_image_rejected():
    with pytest.raises(ValueError):
        rec(image="")


def test_empty_timestamp_rejected():
    with pytest.raises(ValueError):
        rec(timestamp="")


def test_unknown_verdict_rejected():
    with pytest.raises(ValueError):
        rec(verdict="maybe")


def test_not_applicable_can_never_be_written():
    with pytest.raises(ValueError):
        rec(verdict="not_applicable")


def test_wrong_requires_corrected():
    with pytest.raises(ValueError):
        rec(verdict="wrong")
    assert rec(verdict="wrong", corrected="blue").corrected == "blue"


def test_vehicle_visible_wrong_needs_no_corrected():
    r = rec(element="vehicle_visible", model_value="vehicle", verdict="wrong")
    assert r.corrected is None


def test_corrected_only_with_wrong_or_provided():
    for verdict in ("correct", "unsure"):
        with pytest.raises(ValueError):
            rec(verdict=verdict, corrected="blue")
    with pytest.raises(ValueError):
        rec(verdict="not_determinable", reason="blurred", corrected="blue")


def test_not_determinable_requires_reason():
    with pytest.raises(ValueError):
        rec(verdict="not_determinable")
    assert rec(verdict="not_determinable", reason="blurred").reason == "blurred"


def test_reason_must_be_from_list():
    with pytest.raises(ValueError):
        rec(verdict="not_determinable", reason="because")


def test_reason_only_with_not_determinable():
    with pytest.raises(ValueError):
        rec(verdict="correct", reason="blurred")


def test_no_model_value_rejects_correct_and_wrong():
    with pytest.raises(ValueError):
        rec(element="plate_text", model_value=None, verdict="correct")
    with pytest.raises(ValueError):
        rec(element="plate_text", model_value=None, verdict="wrong", corrected="A1")


def test_provided_needs_no_model_value_and_a_corrected_value():
    ok = rec(element="plate_text", model_value=None, verdict="provided", corrected="皖A12345")
    assert ok.corrected == "皖A12345"
    with pytest.raises(ValueError):
        rec(element="plate_text", model_value=None, verdict="provided")
    with pytest.raises(ValueError):
        rec(element="plate_text", model_value="皖A12345", verdict="provided", corrected="皖A12346")


def test_no_model_value_allows_unsure_and_not_determinable():
    assert rec(element="plate_box", model_value=None, verdict="unsure").verdict == "unsure"
    r = rec(element="plate_box", model_value=None, verdict="not_determinable", reason="plate_not_visible")
    assert r.reason == "plate_not_visible"


def test_box_correction_valid():
    r = rec(element="plate_box", model_value=None, verdict="provided", corrected=[10, 20, 110, 60])
    assert r.corrected == [10, 20, 110, 60]
    r = rec(source="ccpd", element="detector_box", model_value=[1, 2, 9, 9], verdict="wrong",
            corrected=[10.5, 20.5, 110.5, 60.5])
    assert r.verdict == "wrong"


def test_box_correction_invalid():
    for bad in ([10, 20, 5, 60], [10, 20, 110, 10], [10, 20, 110], "10 20 110 60"):
        with pytest.raises(ValueError):
            rec(element="plate_box", model_value=None, verdict="provided", corrected=bad)


def test_plate_text_correction_rules():
    for bad in ("", "A 12", " "):
        with pytest.raises(ValueError):
            rec(element="plate_text", model_value=None, verdict="provided", corrected=bad)


def test_json_round_trip_keeps_chinese_characters():
    r = rec(source="ccpd", element="plate_text", split="test", model_value="皖A12345",
            verdict="wrong", corrected="皖A12346", note="last digit blurred")
    line = record_to_line(r)
    assert "\n" not in line and "皖" in line
    assert record_from_line(line) == r


def test_append_keeps_earlier_lines(tmp_path):
    path = tmp_path / "sub" / "log.jsonl"
    a = rec()
    b = rec(element="type", model_value="sedan", verdict="unsure")
    append_record(path, a)
    append_record(path, b)
    text = path.read_text(encoding="utf-8")
    assert text == record_to_line(a) + "\n" + record_to_line(b) + "\n"


def test_read_missing_file_is_empty(tmp_path):
    assert read_log(tmp_path / "none.jsonl") == []


def test_read_bad_line_names_the_line(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(record_to_line(rec()) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_log(path)


def test_latest_wins_by_file_order():
    first = rec(verdict="correct", timestamp="2026-11-11T21:00:00")
    second = rec(verdict="wrong", corrected="blue", timestamp="2026-11-11T20:00:00")
    other = rec(element="type", model_value="sedan")
    result = latest([first, other, second])
    assert result[("veri", first.image, "colour")] == second
    assert len(result) == 2


def test_counts_use_latest_only():
    records = [rec(verdict="correct"), rec(verdict="wrong", corrected="blue")]
    counts = count_verdicts(records, "veri", "colour")
    assert set(counts) == {"correct", "wrong", "unsure", "not_determinable", "provided"}
    assert counts["wrong"] == 1 and counts["correct"] == 0


def test_counts_filter_by_source_and_element():
    records = [
        rec(image="a.jpg"),
        rec(image="b.jpg", verdict="unsure"),
        rec(image="c.jpg", element="type", model_value="sedan"),
        rec(source="ccpd", image="images/1.jpg", element="plate_text", model_value="皖A12345"),
    ]
    counts = count_verdicts(records, "veri", "colour")
    assert counts["correct"] == 1 and counts["unsure"] == 1
    assert sum(counts.values()) == 2
