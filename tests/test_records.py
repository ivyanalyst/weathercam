import json

import pytest

from weathercam.core.types import Box, Detection
from weathercam.detection.records import (
    FORMAT_VERSION,
    ImagePrediction,
    PredictionSet,
    load_predictions,
    save_predictions,
)


def make_set():
    first = ImagePrediction(
        "a.jpg",
        640,
        480,
        (
            Detection(Box(1.5, 2.25, 100.0, 200.125), 2, 0.9),
            Detection(Box(0.1, 0.2, 0.30000000000000004, 0.4), 6, 1 / 3),
        ),
    )
    second = ImagePrediction("b.jpg", 1280, 720, ())
    settings = {
        "imgsz": 640,
        "iou": 0.5,
        "conf": 0.01,
        "tags": ["x", None, True],
        "nested": {"k": 1.5},
    }
    return PredictionSet("yolo11m.pt", settings, (first, second))


def saved(tmp_path):
    path = tmp_path / "out.json"
    save_predictions(path, make_set())
    return path


def payload_of(path):
    return json.loads(path.read_text())


def rewrite(path, payload):
    path.write_text(json.dumps(payload))


def first_image(payload):
    return payload["images"][0]


def first_detection(payload):
    return payload["images"][0]["detections"][0]


def test_round_trip_is_exact(tmp_path):
    assert load_predictions(saved(tmp_path)) == make_set()


def test_round_trip_keeps_order(tmp_path):
    loaded = load_predictions(saved(tmp_path))
    assert [image.name for image in loaded.images] == ["a.jpg", "b.jpg"]
    assert [d.label for d in loaded.images[0].detections] == [2, 6]


def test_empty_image_list_round_trips(tmp_path):
    path = tmp_path / "empty.json"
    original = PredictionSet("m", {}, ())
    save_predictions(path, original)
    assert load_predictions(path) == original


def test_file_declares_format_version(tmp_path):
    assert FORMAT_VERSION == 1
    assert payload_of(saved(tmp_path))["format"] == FORMAT_VERSION


def test_save_creates_missing_parent_folders(tmp_path):
    path = tmp_path / "a" / "b" / "out.json"
    save_predictions(path, make_set())
    assert load_predictions(path) == make_set()


def test_save_leaves_only_the_target_file(tmp_path):
    saved(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_saving_twice_gives_identical_bytes(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    save_predictions(first, make_set())
    save_predictions(second, make_set())
    assert first.read_bytes() == second.read_bytes()


def test_lists_become_tuples():
    detection = Detection(Box(0.0, 0.0, 1.0, 1.0), 0, 0.5)
    image = ImagePrediction("a", 1, 1, [detection])
    assert image.detections == (detection,)
    assert PredictionSet("m", {}, [image]).images == (image,)


def test_settings_tuples_become_lists():
    assert PredictionSet("m", {"s": (1, 2)}, ()).settings == {"s": [1, 2]}


def test_settings_are_copied():
    settings = {"a": 1}
    result = PredictionSet("m", settings, ())
    settings["a"] = 2
    assert result.settings == {"a": 1}


def test_duplicate_image_names_raise():
    with pytest.raises(ValueError):
        PredictionSet(
            "m",
            {},
            (ImagePrediction("a", 1, 1, ()), ImagePrediction("a", 2, 2, ())),
        )


@pytest.mark.parametrize(
    "width,height",
    [(0, 480), (-1, 480), (640, 0), (640, -5), (True, 480), (640.0, 480)],
)
def test_bad_image_size_raises(width, height):
    with pytest.raises(ValueError):
        ImagePrediction("a.jpg", width, height, ())


def test_empty_image_name_raises():
    with pytest.raises(ValueError):
        ImagePrediction("", 1, 1, ())


def test_empty_model_raises():
    with pytest.raises(ValueError):
        PredictionSet("", {}, ())


@pytest.mark.parametrize("settings", [[], "x", None])
def test_non_dict_settings_raise(settings):
    with pytest.raises(ValueError):
        PredictionSet("m", settings, ())


def test_non_string_settings_key_raises():
    with pytest.raises(ValueError):
        PredictionSet("m", {1: 2}, ())


@pytest.mark.parametrize(
    "settings",
    [{"a": {1, 2}}, {"a": float("nan")}, {"a": float("inf")}],
)
def test_unserialisable_settings_raise(settings):
    with pytest.raises(ValueError):
        PredictionSet("m", settings, ())


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "none.json")


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        load_predictions(path)


@pytest.mark.parametrize("text", ["[]", "3", "null"])
def test_load_non_object_raises(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_predictions(path)


@pytest.mark.parametrize("version", [0, 2, "1", True])
def test_load_wrong_version_raises(tmp_path, version):
    path = saved(tmp_path)
    payload = payload_of(path)
    payload["format"] = version
    rewrite(path, payload)
    with pytest.raises(ValueError):
        load_predictions(path)


@pytest.mark.parametrize("key", ["format", "model", "settings", "images"])
def test_load_missing_top_level_key_raises(tmp_path, key):
    path = saved(tmp_path)
    payload = payload_of(path)
    del payload[key]
    rewrite(path, payload)
    with pytest.raises(ValueError):
        load_predictions(path)


@pytest.mark.parametrize("key", ["name", "width", "height", "detections"])
def test_load_missing_image_key_raises(tmp_path, key):
    path = saved(tmp_path)
    payload = payload_of(path)
    del first_image(payload)[key]
    rewrite(path, payload)
    with pytest.raises(ValueError):
        load_predictions(path)


@pytest.mark.parametrize("key", ["box", "label", "score"])
def test_load_missing_detection_key_raises(tmp_path, key):
    path = saved(tmp_path)
    payload = payload_of(path)
    del first_detection(payload)[key]
    rewrite(path, payload)
    with pytest.raises(ValueError):
        load_predictions(path)


BAD_MUTATIONS = [
    pytest.param(lambda p: p.update(model=3), id="model_not_str"),
    pytest.param(lambda p: p.update(model=""), id="model_empty"),
    pytest.param(lambda p: p.update(settings=[]), id="settings_not_dict"),
    pytest.param(lambda p: p.update(images={}), id="images_not_list"),
    pytest.param(lambda p: p["images"].__setitem__(0, 3), id="image_not_object"),
    pytest.param(lambda p: first_image(p).update(name=3), id="name_not_str"),
    pytest.param(lambda p: first_image(p).update(width="640"), id="width_string"),
    pytest.param(lambda p: first_image(p).update(width=0), id="width_zero"),
    pytest.param(lambda p: first_image(p).update(width=True), id="width_bool"),
    pytest.param(
        lambda p: first_image(p).update(detections={}), id="detections_not_list"
    ),
    pytest.param(
        lambda p: p["images"].append(dict(first_image(p))), id="duplicate_name"
    ),
    pytest.param(
        lambda p: first_image(p)["detections"].__setitem__(0, 3),
        id="detection_not_object",
    ),
    pytest.param(
        lambda p: first_detection(p).update(box=[1, 2, 3]), id="box_three_values"
    ),
    pytest.param(
        lambda p: first_detection(p).update(box=["a", 0, 1, 1]), id="box_string_value"
    ),
    pytest.param(
        lambda p: first_detection(p).update(box=[None, 0, 1, 1]), id="box_null_value"
    ),
    pytest.param(
        lambda p: first_detection(p).update(box=[5, 0, 1, 1]), id="box_inverted"
    ),
    pytest.param(
        lambda p: first_detection(p).update(box=[float("nan"), 0, 1, 1]),
        id="box_nan",
    ),
    pytest.param(lambda p: first_detection(p).update(label=True), id="label_bool"),
    pytest.param(lambda p: first_detection(p).update(label=-1), id="label_negative"),
    pytest.param(lambda p: first_detection(p).update(label=1.5), id="label_float"),
    pytest.param(
        lambda p: first_detection(p).update(score=1.5), id="score_too_high"
    ),
    pytest.param(
        lambda p: first_detection(p).update(score=-0.1), id="score_negative"
    ),
    pytest.param(
        lambda p: first_detection(p).update(score="a"), id="score_string"
    ),
    pytest.param(lambda p: first_detection(p).update(score=True), id="score_bool"),
]


@pytest.mark.parametrize("mutate", BAD_MUTATIONS)
def test_load_rejects_bad_payload(tmp_path, mutate):
    path = saved(tmp_path)
    payload = payload_of(path)
    mutate(payload)
    rewrite(path, payload)
    with pytest.raises(ValueError):
        load_predictions(path)
