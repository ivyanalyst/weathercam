import pytest

from weathercam.data.attrlabels import AttrColumns, class_names, columns_of, encode, vehicle_classes
from weathercam.data.veri import Labels, VeriItem


def labels_of(*items):
    return Labels(declared=len(items), items=tuple(items))


def test_columns_keep_order():
    labels = labels_of(
        VeriItem("a.jpg", "v1", "c1", 3, 4),
        VeriItem("b.jpg", "v2", "c2", 1, 2),
    )
    columns = columns_of(labels)
    assert isinstance(columns, AttrColumns)
    assert columns.image_names == ("a.jpg", "b.jpg")
    assert columns.vehicle_ids == ("v1", "v2")
    assert columns.camera_ids == ("c1", "c2")
    assert columns.color_ids == (3, 1)
    assert columns.type_ids == (4, 2)


def test_columns_empty():
    columns = columns_of(labels_of())
    assert columns.image_names == ()
    assert columns.color_ids == ()


def test_vehicle_classes_one_pair_per_vehicle():
    labels = labels_of(
        VeriItem("a.jpg", "v1", "c1", 1, 2),
        VeriItem("b.jpg", "v1", "c2", 1, 2),
        VeriItem("c.jpg", "v2", "c1", 3, 4),
    )
    assert vehicle_classes(labels) == {"v1": (1, 2), "v2": (3, 4)}


def test_vehicle_classes_mixed_raises():
    labels = labels_of(
        VeriItem("a.jpg", "v1", "c1", 1, 2),
        VeriItem("b.jpg", "v1", "c2", 1, 5),
    )
    with pytest.raises(ValueError):
        vehicle_classes(labels)


def test_class_names_sorted_by_id():
    assert class_names({3: "green", 1: "yellow", 2: "orange"}) == ["yellow", "orange", "green"]


def test_encode_uses_sorted_id_order():
    names = {3: "green", 1: "yellow", 2: "orange"}
    assert encode([3, 1, 1], names) == [2, 0, 0]


def test_encode_unknown_raises():
    with pytest.raises(ValueError):
        encode([9], {1: "yellow"})
