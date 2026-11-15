import numpy as np
import pytest

from weathercam.retrieval.innersplit import veri_inner_split


def build(test_vehicles=5):
    split = []
    vehicles = []
    colors = []
    types = []
    for k in range(test_vehicles):
        for _ in range(2):
            split.append("test")
            vehicles.append(1000 + k)
            colors.append(1)
            types.append(1)
    for cls, (color, kind) in enumerate([(1, 1), (2, 3)]):
        for k in range(10):
            for _ in range(3):
                split.append("train")
                vehicles.append(cls * 100 + k)
                colors.append(color)
                types.append(kind)
    return {
        "split": np.array(split),
        "vehicle_ids": np.array(vehicles),
        "color_ids": np.array(colors),
        "type_ids": np.array(types),
    }


def test_sizes():
    fit, val = veri_inner_split(build(), 42)
    assert len(fit) == 48
    assert len(val) == 12


def test_rows_are_train_row_numbers():
    data = build()
    fit, val = veri_inner_split(data, 42)
    rows = np.concatenate([fit, val])
    assert (data["split"][rows] == "train").all()
    assert rows.min() == 10
    assert rows.max() == 69


def test_parts_partition_train_rows():
    fit, val = veri_inner_split(build(), 42)
    assert set(fit.tolist()) | set(val.tolist()) == set(range(10, 70))
    assert set(fit.tolist()) & set(val.tolist()) == set()


def test_no_vehicle_in_both_parts():
    data = build()
    fit, val = veri_inner_split(data, 42)
    fit_vehicles = set(data["vehicle_ids"][fit].tolist())
    val_vehicles = set(data["vehicle_ids"][val].tolist())
    assert fit_vehicles & val_vehicles == set()
    assert len(fit_vehicles) == 16
    assert len(val_vehicles) == 4


def test_validation_vehicles_stratified_by_class():
    data = build()
    _, val = veri_inner_split(data, 42)
    vehicles = set(data["vehicle_ids"][val].tolist())
    assert len([v for v in vehicles if v < 100]) == 2
    assert len([v for v in vehicles if v >= 100]) == 2


def test_same_seed_same_rows():
    data = build()
    first = veri_inner_split(data, 42)
    second = veri_inner_split(data, 42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_rows_ascending():
    fit, val = veri_inner_split(build(), 42)
    assert (np.diff(fit) > 0).all()
    assert (np.diff(val) > 0).all()


def test_vehicle_with_two_classes_raises():
    data = build()
    data["color_ids"][10] = 9
    with pytest.raises(ValueError):
        veri_inner_split(data, 42)
