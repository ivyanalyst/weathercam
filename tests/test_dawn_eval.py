import pytest

from weathercam.core.det_metrics import average_precision
from weathercam.core.types import Box, Detection
from weathercam.data.dawn import Annotation, LabeledBox
from weathercam.detection.classmap import CLASS_NAMES, dawn_class_map
from weathercam.detection.dawn_eval import (
    VEHICLE_LABEL,
    VEHICLE_SOURCE_LABELS,
    ClassScore,
    ImageSetScore,
    ground_truth_detections,
    merge_vehicle,
    score_by_weather,
    score_images,
)

PERSON = 0
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 4
TRAIN = 5
TRUCK = 6

A = Box(0.0, 0.0, 10.0, 10.0)
B = Box(20.0, 20.0, 30.0, 30.0)
C = Box(50.0, 50.0, 60.0, 60.0)
P = Box(70.0, 70.0, 80.0, 80.0)
E = Box(30.0, 30.0, 40.0, 40.0)
FAR = Box(100.0, 100.0, 110.0, 110.0)

GT_ONE = [
    Detection(A, CAR, 1.0),
    Detection(B, CAR, 1.0),
    Detection(C, TRUCK, 1.0),
    Detection(P, PERSON, 1.0),
]
GT_TWO = [Detection(A, CAR, 1.0), Detection(E, BUS, 1.0)]


def scored(detections, score=0.9):
    return [Detection(d.box, d.label, score) for d in detections]


def annotation(*items):
    return Annotation(
        1000, 800, tuple(LabeledBox(name, box) for name, box in items)
    )


def test_ground_truth_converts_with_score_one():
    ann = annotation(("car", A), ("person", P), ("car", B))
    result = ground_truth_detections(ann, dawn_class_map())
    assert result == [
        Detection(A, CAR, 1.0),
        Detection(P, PERSON, 1.0),
        Detection(B, CAR, 1.0),
    ]


def test_ground_truth_unknown_name_raises():
    with pytest.raises(ValueError):
        ground_truth_detections(annotation(("dog", A)), dawn_class_map())


def test_ground_truth_drops_none_mapped():
    ann = annotation(("car", A), ("dog", B))
    result = ground_truth_detections(ann, {"car": CAR, "dog": None})
    assert result == [Detection(A, CAR, 1.0)]


def test_ground_truth_empty_annotation():
    assert ground_truth_detections(annotation(), dawn_class_map()) == []


def test_vehicle_source_labels_match_class_map():
    names = ("car", "motorcycle", "bus", "truck")
    assert VEHICLE_SOURCE_LABELS == frozenset(CLASS_NAMES.index(n) for n in names)
    assert VEHICLE_SOURCE_LABELS == frozenset({CAR, MOTORCYCLE, BUS, TRUCK})


def test_merge_vehicle_relabels_filters_and_sorts():
    detections = [
        Detection(A, CAR, 0.9),
        Detection(P, PERSON, 0.95),
        Detection(B, BUS, 0.8),
        Detection(C, TRUCK, 0.7),
        Detection(FAR, MOTORCYCLE, 0.6),
        Detection(Box(120.0, 120.0, 130.0, 130.0), BICYCLE, 0.5),
        Detection(Box(140.0, 140.0, 150.0, 150.0), TRAIN, 0.4),
    ]
    result = merge_vehicle(detections)
    assert [d.label for d in result] == [VEHICLE_LABEL] * 4
    assert [d.score for d in result] == [0.9, 0.8, 0.7, 0.6]
    assert [d.box for d in result] == [A, B, C, FAR]


def test_merge_vehicle_removes_cross_class_duplicate():
    detections = [Detection(A, CAR, 0.9), Detection(A, TRUCK, 0.8)]
    assert merge_vehicle(detections) == [Detection(A, VEHICLE_LABEL, 0.9)]


def test_merge_vehicle_keeps_overlap_below_threshold():
    shifted = Box(5.0, 0.0, 15.0, 10.0)
    detections = [Detection(A, CAR, 0.9), Detection(shifted, TRUCK, 0.8)]
    assert len(merge_vehicle(detections)) == 2


def test_merge_vehicle_iou_parameter():
    shifted = Box(5.0, 0.0, 15.0, 10.0)
    detections = [Detection(A, CAR, 0.9), Detection(shifted, TRUCK, 0.8)]
    assert merge_vehicle(detections, nms_iou=0.3) == [
        Detection(A, VEHICLE_LABEL, 0.9)
    ]


def test_merge_vehicle_empty():
    assert merge_vehicle([]) == []


def test_perfect_predictions_score_one():
    gts = [GT_ONE, GT_TWO]
    result = score_images([scored(GT_ONE), scored(GT_TWO)], gts)
    assert isinstance(result, ImageSetScore)
    assert result.images == 2
    assert [c.label for c in result.classes] == [PERSON, CAR, BUS, TRUCK]
    assert [c.boxes for c in result.classes] == [1, 3, 1, 1]
    for item in result.classes:
        assert item.ap50 == pytest.approx(1.0)
        assert item.ap50_95 == pytest.approx(1.0)
    assert result.vehicle.boxes == 5
    assert result.vehicle.ap50 == pytest.approx(1.0)
    assert result.vehicle.ap50_95 == pytest.approx(1.0)


def test_no_predictions_score_zero():
    result = score_images([[], []], [GT_ONE, GT_TWO])
    assert [c.boxes for c in result.classes] == [1, 3, 1, 1]
    for item in result.classes:
        assert item.ap50 == 0.0
        assert item.ap50_95 == 0.0
    assert result.vehicle.boxes == 5
    assert result.vehicle.ap50 == 0.0


def test_missing_classes_score_zero_others_keep_score():
    cars_one = [d for d in GT_ONE if d.label == CAR]
    cars_two = [d for d in GT_TWO if d.label == CAR]
    result = score_images([scored(cars_one), scored(cars_two)], [GT_ONE, GT_TWO])
    by_label = {c.label: c for c in result.classes}
    assert by_label[CAR].ap50 == pytest.approx(1.0)
    assert by_label[CAR].boxes == 3
    for label in (PERSON, BUS, TRUCK):
        assert by_label[label].ap50 == 0.0
    assert result.vehicle.boxes == 5
    assert 0.0 < result.vehicle.ap50 < 1.0


def test_cross_class_duplicate_does_not_hurt_vehicle_ap():
    gts = [[Detection(A, CAR, 1.0), Detection(FAR, CAR, 1.0)]]
    preds = [
        [
            Detection(A, CAR, 0.9),
            Detection(A, TRUCK, 0.8),
            Detection(FAR, CAR, 0.7),
        ]
    ]
    result = score_images(preds, gts)
    assert result.vehicle.ap50 == pytest.approx(1.0)
    assert [c.label for c in result.classes] == [CAR]
    assert result.classes[0].ap50 == pytest.approx(1.0)
    raw_preds = [[Detection(d.box, VEHICLE_LABEL, d.score) for d in preds[0]]]
    raw_gts = [[Detection(d.box, VEHICLE_LABEL, d.score) for d in gts[0]]]
    assert average_precision(raw_preds, raw_gts, VEHICLE_LABEL, 0.5) < 0.9


def test_no_vehicle_ground_truth_gives_none():
    gts = [[Detection(P, PERSON, 1.0)]]
    result = score_images([scored(gts[0])], gts)
    assert [c.label for c in result.classes] == [PERSON]
    assert result.vehicle is None


def test_no_ground_truth_gives_empty():
    assert score_images([[], []], [[], []]) == ImageSetScore(2, (), None)


def test_loose_box_scores_by_threshold():
    gts = [[Detection(A, CAR, 1.0)]]
    preds = [[Detection(Box(0.0, 0.0, 10.0, 6.25), CAR, 0.9)]]
    result = score_images(preds, gts)
    assert result.classes[0].ap50 == pytest.approx(1.0)
    assert result.classes[0].ap50_95 == pytest.approx(0.3)
    assert result.vehicle.ap50 == pytest.approx(1.0)
    assert result.vehicle.ap50_95 == pytest.approx(0.3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_images([[]], [])


def test_scores_are_frozen():
    item = ClassScore(CAR, 1, 1.0, 1.0)
    with pytest.raises(AttributeError):
        item.boxes = 2


def test_by_weather_groups_and_scores():
    gts = [
        [Detection(A, CAR, 1.0)],
        [Detection(C, TRUCK, 1.0)],
        [Detection(B, CAR, 1.0), Detection(P, PERSON, 1.0)],
    ]
    preds = [scored(gts[0]), [], [Detection(B, CAR, 0.9)]]
    result = score_by_weather(["Fog", "Rain", "Fog"], preds, gts)
    assert list(result) == ["Fog", "Rain"]
    fog = result["Fog"]
    assert fog.images == 2
    assert [c.label for c in fog.classes] == [PERSON, CAR]
    assert [c.boxes for c in fog.classes] == [1, 2]
    assert fog.classes[0].ap50 == 0.0
    assert fog.classes[1].ap50 == pytest.approx(1.0)
    assert fog.classes[1].ap50_95 == pytest.approx(1.0)
    assert fog.vehicle.boxes == 2
    assert fog.vehicle.ap50 == pytest.approx(1.0)
    rain = result["Rain"]
    assert rain.images == 1
    assert [c.label for c in rain.classes] == [TRUCK]
    assert [c.boxes for c in rain.classes] == [1]
    assert rain.classes[0].ap50 == 0.0
    assert rain.vehicle.boxes == 1
    assert rain.vehicle.ap50 == 0.0


@pytest.mark.parametrize(
    "weathers,preds,gts",
    [
        (["Fog"], [[], []], [[], []]),
        (["Fog", "Fog"], [[]], [[], []]),
        (["Fog", "Fog"], [[], []], [[]]),
    ],
)
def test_by_weather_length_mismatch_raises(weathers, preds, gts):
    with pytest.raises(ValueError):
        score_by_weather(weathers, preds, gts)


@pytest.mark.parametrize("weather", ["", 3])
def test_bad_weather_name_raises(weather):
    with pytest.raises(ValueError):
        score_by_weather([weather], [[]], [[]])
