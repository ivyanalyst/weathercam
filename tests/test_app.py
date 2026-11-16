import base64
import math

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from weathercam.plates.ocr import PlateReading
from weathercam.service.app import OodBank, create_app
from weathercam.service.handlers import GalleryIndex
from weathercam.service.heads import LinearHead, MetricProjector
from weathercam.service.plate_client import PlateResult
from weathercam.service.worker_client import WorkerError

IMAGE_A = b"image-one"
IMAGE_B = b"image-two"
QUERY_ROW = np.array([1.0, 0.0], dtype=np.float32)
POSITIONS = (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93)
PLATE_HIT = PlateResult(640, 480, (10.0, 20.0, 110.0, 70.0), 0.9, PlateReading("XY12345", 0.93, POSITIONS))
PLATE_MISS = PlateResult(320, 240, None, None, None)


def b64(data):
    return base64.b64encode(data).decode("ascii")


def softmax(values):
    exps = [math.exp(v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class FakeEncoder:
    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def encode_images(self, blobs):
        self.image_calls.append(list(blobs))
        return np.tile(QUERY_ROW, (len(blobs), 1))

    def encode_text(self, texts):
        self.text_calls.append(list(texts))
        return np.tile(QUERY_ROW, (len(texts), 1))


class FailingEncoder:
    def encode_images(self, blobs):
        raise WorkerError("worker exited")

    def encode_text(self, texts):
        raise WorkerError("worker exited")


class FakeReader:
    def __init__(self):
        self.calls = []

    def read_plates(self, blobs):
        items = list(blobs)
        self.calls.append(items)
        return [(PLATE_HIT, PLATE_MISS)[i % 2] for i in range(len(items))]


class FailingReader:
    def read_plates(self, blobs):
        raise WorkerError("worker exited")


def make_head(bias, names, class_ids):
    return LinearHead(
        weights=np.zeros((2, len(bias))),
        bias=np.asarray(bias, dtype=np.float64),
        mean=np.zeros(2),
        std=np.ones(2),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        names=tuple(names),
    )


def weather_head():
    return make_head([0.0, 0.0, 0.0, 0.0], ("Fog", "Rain", "Sand", "Snow"), [0, 1, 2, 3])


def attribute_heads():
    return {
        "color": make_head([2.0, 1.0, 0.0, -1.0], ("red", "blue", "green", "gray"), [1, 2, 3, 4]),
        "type": make_head([0.0, 1.0], ("sedan", "suv"), [1, 2]),
    }


def ood_bank():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    return OodBank(bank, np.zeros(2), np.ones(2), 2)


def gallery():
    return GalleryIndex(["a", "b", "c"], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))


def build(**overrides):
    encoder = overrides.pop("encoder", FakeEncoder())
    reader = overrides.pop("plate_reader", FakeReader())
    options = {
        "weather_head": weather_head(),
        "attribute_heads": attribute_heads(),
        "ood": ood_bank(),
        "gallery": gallery(),
    }
    options.update(overrides)
    app = create_app(encoder=encoder, plate_reader=reader, **options)
    return TestClient(app), encoder, reader


def test_health_reports_features_and_caps():
    client, _, _ = build()
    body = client.get("/health").json()
    assert body["features"] == {"weather": True, "attributes": True, "ood": True, "plates": True, "search_text": True}
    assert body["gallery_size"] == 3
    assert body["max_images"] == 16
    assert body["max_image_bytes"] == 8388608
    bare = TestClient(create_app()).get("/health").json()
    assert bare["features"] == {key: False for key in body["features"]}
    assert bare["gallery_size"] is None


def test_weather_returns_every_class_for_every_image():
    client, _, _ = build()
    response = client.post("/weather", json={"images": [b64(IMAGE_A), b64(IMAGE_B)]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    for item in results:
        assert item["class_id"] == 0
        assert item["label"] == "Fog"
        assert item["probability"] == pytest.approx(0.25)
        assert [entry["label"] for entry in item["top"]] == ["Fog", "Rain", "Sand", "Snow"]
        assert [entry["class_id"] for entry in item["top"]] == [0, 1, 2, 3]
        assert [entry["probability"] for entry in item["top"]] == pytest.approx([0.25] * 4)


def test_weather_rejects_empty_or_missing_images():
    client, encoder, _ = build()
    assert client.post("/weather", json={"images": []}).status_code == 422
    assert client.post("/weather", json={}).status_code == 422
    assert encoder.image_calls == []


def test_invalid_base64_is_rejected():
    client, encoder, _ = build()
    response = client.post("/weather", json={"images": [b64(IMAGE_A), "not base64!!"]})
    assert response.status_code == 422
    assert encoder.image_calls == []


def test_decoded_bytes_reach_the_encoder():
    client, encoder, _ = build()
    client.post("/weather", json={"images": [b64(IMAGE_A), b64(IMAGE_B)]})
    assert encoder.image_calls == [[IMAGE_A, IMAGE_B]]


def test_attributes_return_every_head_with_capped_top():
    client, _, _ = build()
    response = client.post("/attributes", json={"images": [b64(IMAGE_A), b64(IMAGE_B)]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0] == results[1]
    assert set(results[0]) == {"color", "type"}
    color = results[0]["color"]
    expected = softmax([2.0, 1.0, 0.0, -1.0])
    assert (color["class_id"], color["label"]) == (1, "red")
    assert color["probability"] == pytest.approx(expected[0])
    assert [entry["label"] for entry in color["top"]] == ["red", "blue", "green"]
    assert [entry["probability"] for entry in color["top"]] == pytest.approx(expected[:3])
    kind = results[0]["type"]
    expected_type = softmax([0.0, 1.0])
    assert (kind["class_id"], kind["label"]) == (2, "suv")
    assert [entry["label"] for entry in kind["top"]] == ["suv", "sedan"]
    assert [entry["probability"] for entry in kind["top"]] == pytest.approx([expected_type[1], expected_type[0]])


def test_ood_returns_scores_without_a_verdict():
    client, _, _ = build()
    response = client.post("/ood", json={"images": [b64(IMAGE_A), b64(IMAGE_B)]})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"scores", "k"}
    assert body["k"] == 2
    assert body["scores"] == pytest.approx([math.sqrt(2.0)] * 2)


def test_ood_without_a_bank_is_unavailable():
    client, encoder, _ = build(ood=None)
    assert client.post("/ood", json={"images": [b64(IMAGE_A)]}).status_code == 503
    assert encoder.image_calls == []


def test_plates_return_readings_and_nulls():
    client, _, reader = build()
    response = client.post("/plates", json={"images": [b64(IMAGE_A), b64(IMAGE_B)]})
    assert response.status_code == 200
    body = response.json()
    assert body["matches"] is None
    hit, miss = body["results"]
    assert hit["index"] == 0
    assert (hit["width"], hit["height"]) == (640, 480)
    assert hit["box"] == [10.0, 20.0, 110.0, 70.0]
    assert hit["score"] == pytest.approx(0.9)
    assert hit["reading"]["text"] == "XY12345"
    assert hit["reading"]["confidence"] == pytest.approx(0.93)
    assert hit["reading"]["positions"] == pytest.approx(list(POSITIONS))
    assert miss == {"index": 1, "width": 320, "height": 240, "box": None, "score": None, "reading": None}
    assert reader.calls == [[IMAGE_A, IMAGE_B]]


def test_plates_match_within_max_edits():
    client, _, _ = build()
    images = [b64(IMAGE_A), b64(IMAGE_B)]
    close = client.post("/plates", json={"images": images, "query": "XY12346", "max_edits": 1})
    assert close.json()["matches"] == [{"index": 0, "distance": 1}]
    exact_only = client.post("/plates", json={"images": images, "query": "XY12346", "max_edits": 0})
    assert exact_only.json()["matches"] == []


def test_plate_query_and_max_edits_must_come_together():
    client, _, reader = build()
    images = [b64(IMAGE_A)]
    assert client.post("/plates", json={"images": images, "query": "XY12345"}).status_code == 422
    assert client.post("/plates", json={"images": images, "max_edits": 1}).status_code == 422
    assert client.post("/plates", json={"images": images, "query": "XY12345", "max_edits": -1}).status_code == 422
    assert reader.calls == []


def test_too_many_images_are_refused():
    client, encoder, reader = build(max_images=2)
    images = [b64(IMAGE_A), b64(IMAGE_B), b64(IMAGE_A)]
    assert client.post("/plates", json={"images": images}).status_code == 413
    assert client.post("/weather", json={"images": images}).status_code == 413
    assert client.post("/search/text", json={"texts": ["a", "b", "c"], "k": 1}).status_code == 413
    assert reader.calls == []
    assert encoder.image_calls == []
    assert encoder.text_calls == []
    assert client.post("/weather", json={"images": images[:2]}).status_code == 200


def test_oversized_images_are_refused():
    client, _, reader = build(max_image_bytes=8)
    assert client.post("/plates", json={"images": [b64(b"12345678")]}).status_code == 200
    assert client.post("/plates", json={"images": [b64(b"123456789")]}).status_code == 413
    assert len(reader.calls) == 1


def test_plates_without_a_reader_are_unavailable():
    client, _, _ = build(plate_reader=None)
    assert client.post("/plates", json={"images": [b64(IMAGE_A)]}).status_code == 503


def test_text_search_returns_names_and_scores():
    client, encoder, _ = build()
    response = client.post("/search/text", json={"texts": ["red car", "blue van"], "k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    for found in results:
        assert [entry["name"] for entry in found] == ["a", "c"]
        assert [entry["score"] for entry in found] == pytest.approx([1.0, 1.0 / math.sqrt(2.0)])
    assert encoder.text_calls == [["red car", "blue van"]]


def test_text_search_without_a_gallery_is_unavailable():
    client, encoder, _ = build(gallery=None)
    assert client.post("/search/text", json={"texts": ["red car"], "k": 1}).status_code == 503
    assert encoder.text_calls == []


def test_text_search_rejects_bad_requests():
    client, encoder, _ = build()
    assert client.post("/search/text", json={"texts": ["red car"], "k": 0}).status_code == 422
    assert client.post("/search/text", json={"texts": ["red car"], "k": 4}).status_code == 422
    assert client.post("/search/text", json={"texts": [], "k": 1}).status_code == 422
    assert client.post("/search/text", json={"texts": "red car", "k": 1}).status_code == 422
    assert encoder.text_calls == []
    assert client.post("/search/text", json={"texts": ["red car"], "k": 3}).status_code == 200


def test_worker_errors_become_bad_gateway():
    client, _, _ = build(encoder=FailingEncoder(), plate_reader=FailingReader())
    assert client.post("/weather", json={"images": [b64(IMAGE_A)]}).status_code == 502
    assert client.post("/plates", json={"images": [b64(IMAGE_A)]}).status_code == 502
    assert client.post("/search/text", json={"texts": ["red car"], "k": 1}).status_code == 502


def test_create_app_refuses_a_projected_gallery():
    projector = MetricProjector(np.eye(2), np.zeros(2), np.ones(2))
    projected = GalleryIndex(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]), projector=projector)
    with pytest.raises(ValueError):
        create_app(encoder=FakeEncoder(), gallery=projected)
