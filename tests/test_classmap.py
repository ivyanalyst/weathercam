import pytest

from weathercam.core.types import Box
from weathercam.data.dawn import Annotation, LabeledBox, map_classes
from weathercam.detection.classmap import CLASS_NAMES, dawn_class_map, model_label_map


def test_class_names_are_the_seven_dawn_classes_in_fixed_order():
    assert CLASS_NAMES == ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck")


def test_dawn_class_map_gives_each_name_its_position():
    assert dawn_class_map() == {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 4,
        "train": 5,
        "truck": 6,
    }


def test_dawn_class_map_keeps_every_class():
    mapping = dawn_class_map()
    assert len(mapping) == 7
    assert None not in mapping.values()


def test_dawn_class_map_returns_a_fresh_dict_each_call():
    first = dawn_class_map()
    first["car"] = 99
    first["extra"] = 7
    assert dawn_class_map()["car"] == 2
    assert "extra" not in dawn_class_map()


def test_dawn_class_map_works_with_map_classes():
    annotation = Annotation(
        100,
        100,
        (
            LabeledBox("car", Box(0, 0, 10, 10)),
            LabeledBox("person", Box(5, 5, 20, 20)),
            LabeledBox("train", Box(1, 1, 2, 2)),
        ),
    )
    assert map_classes(annotation, dawn_class_map()) == [
        (2, Box(0, 0, 10, 10)),
        (0, Box(5, 5, 20, 20)),
        (5, Box(1, 1, 2, 2)),
    ]


def test_model_label_map_from_sequence_drops_other_classes():
    names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
    assert model_label_map(names) == {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6}


def test_model_label_map_from_sparse_mapping():
    names = {
        0: "person",
        2: "bicycle",
        3: "car",
        7: "motorcycle",
        8: "bus",
        9: "train",
        10: "truck",
        20: "zebra",
    }
    assert model_label_map(names) == {0: 0, 2: 1, 3: 2, 7: 3, 8: 4, 9: 5, 10: 6}


def test_model_label_map_with_placeholder_entries():
    names = ["N/A", "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light"]
    assert model_label_map(names) == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


def test_model_label_map_rejects_missing_class():
    names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    with pytest.raises(ValueError, match="train"):
        model_label_map(names)


def test_model_label_map_rejects_empty_names():
    with pytest.raises(ValueError):
        model_label_map([])
    with pytest.raises(ValueError):
        model_label_map({})


def test_model_label_map_rejects_duplicate_class_name():
    names = ["person", "bicycle", "car", "car", "motorcycle", "bus", "train", "truck"]
    with pytest.raises(ValueError, match="car"):
        model_label_map(names)


def test_model_label_map_rejects_non_string_name():
    names = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", 5]
    with pytest.raises(ValueError):
        model_label_map(names)


def test_model_label_map_is_case_sensitive():
    names = ["person", "bicycle", "Car", "motorcycle", "bus", "train", "truck"]
    with pytest.raises(ValueError, match="car"):
        model_label_map(names)


def test_model_label_map_rejects_negative_index():
    names = {-1: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "train", 6: "truck"}
    with pytest.raises(ValueError):
        model_label_map(names)


def test_model_label_map_rejects_non_integer_key():
    names = {"a": "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "train", 6: "truck"}
    with pytest.raises(ValueError):
        model_label_map(names)
