import numpy as np
import pytest

from weathercam.plates.ocr import PlateReading
from weathercam.retrieval.ood import knn_distances
from weathercam.service.handlers import (
    GalleryIndex,
    attributes,
    find_plates,
    ood_scores,
    read_plates,
    search_text,
    weather,
)
from weathercam.service.heads import LinearHead, MetricProjector
from weathercam.service.plate_client import PlateResult


def make_head(names, ids, dim):
    count = len(names)
    return LinearHead(
        np.eye(dim)[:, :count],
        np.zeros(count),
        np.zeros(dim),
        np.ones(dim),
        np.array(ids),
        tuple(names),
    )


def make_projector():
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    return MetricProjector(weights, np.zeros(3), np.ones(3))


class FakeEncoder:
    def __init__(self, table):
        self.table = table

    def encode_text(self, texts):
        return np.array([self.table[t] for t in texts], dtype=np.float32)

    def encode_images(self, blobs):
        raise AssertionError("not used")


class FakeReader:
    def __init__(self, results):
        self.results = results
        self.seen = None

    def read_plates(self, blobs):
        self.seen = list(blobs)
        return self.results


def plate(text):
    if text is None:
        return PlateResult(720, 1160, None, None, None)
    reading = PlateReading(text, 0.9, (0.9,) * 7)
    return PlateResult(720, 1160, (0.0, 0.0, 10.0, 10.0), 0.5, reading)


def test_weather_top_covers_all_classes():
    head = make_head(("Fog", "Rain", "Sand", "Snow"), [0, 1, 2, 3], 4)
    result = weather(head, np.array([[0.0, 0.0, 5.0, 0.0]]))
    assert len(result) == 1
    assert len(result[0].top) == 4
    assert result[0].label == "Sand"


def test_weather_maps_index_through_class_ids():
    head = make_head(("Fog", "Rain", "Sand", "Snow"), [10, 11, 12, 13], 4)
    result = weather(head, np.array([[0.0, 0.0, 0.0, 5.0]]))
    assert result[0].class_id == 13
    assert result[0].label == "Snow"


def test_attributes_returns_one_dict_per_row():
    heads = {
        "color": make_head(("red", "blue", "green"), [1, 2, 3], 3),
        "type": make_head(("sedan", "suv"), [1, 2], 3),
    }
    result = attributes(heads, np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]))
    assert len(result) == 2
    assert result[0]["color"].label == "red"
    assert result[0]["type"].class_id == 1
    assert result[1]["color"].class_id == 2
    assert result[1]["type"].label == "suv"


def test_attributes_caps_top_k_per_head():
    heads = {
        "color": make_head(("red", "blue", "green"), [1, 2, 3], 3),
        "type": make_head(("sedan", "suv"), [1, 2], 3),
    }
    result = attributes(heads, np.array([[5.0, 0.0, 0.0]]))
    assert len(result[0]["color"].top) == 3
    assert len(result[0]["type"].top) == 2


def test_attributes_requires_heads():
    with pytest.raises(ValueError):
        attributes({}, np.zeros((1, 3)))


def test_ood_scores_match_knn_on_standardised_input():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    queries = np.array([[2.0, 0.0], [1.0, 3.0]])
    got = ood_scores(bank, queries, np.zeros(2), np.ones(2), k=2)
    np.testing.assert_allclose(got, knn_distances(bank, queries, 2))


def test_ood_scores_standardise_first():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    got = ood_scores(bank, np.array([[3.0, 0.0]]), np.array([1.0, 0.0]), np.array([2.0, 1.0]), k=1)
    assert got[0] == pytest.approx(0.0, abs=1e-6)


def test_ood_scores_rank_far_above_near():
    bank = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]])
    queries = np.array([[1.0, 0.05], [0.0, 1.0]])
    got = ood_scores(bank, queries, np.zeros(2), np.ones(2), k=1)
    assert got[0] < got[1]


def test_ood_scores_reject_wrong_width():
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ood_scores(bank, np.ones((1, 3)), np.zeros(2), np.ones(2), k=1)


def test_gallery_without_projector_ranks_by_cosine():
    gallery = GalleryIndex(["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    result = gallery.search(np.array([[0.1, 0.0, 1.0]]), 2)
    assert [name for name, _ in result[0]] == ["b", "a"]
    assert result[0][0][1] == pytest.approx(0.995037, abs=1e-5)
    assert result[0][1][1] == pytest.approx(0.0995037, abs=1e-5)


def test_gallery_with_projector_uses_projection():
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 5.0]])
    gallery = GalleryIndex(["a", "b", "c"], embeddings, make_projector())
    result = gallery.search(np.array([[2.0, 0.1, 9.0]]), 3)
    assert [name for name, _ in result[0]] == ["a", "c", "b"]
    assert result[0][0][1] == pytest.approx(0.998752, abs=1e-5)
    assert result[0][1][1] == pytest.approx(0.741549, abs=1e-4)
    assert result[0][2][1] == pytest.approx(0.049938, abs=1e-5)


def test_gallery_returns_one_list_per_query():
    gallery = GalleryIndex(["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    result = gallery.search(np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0]]), 1)
    assert [row[0][0] for row in result] == ["a", "b"]


def test_gallery_rejects_name_count_mismatch():
    with pytest.raises(ValueError):
        GalleryIndex(["a"], np.ones((2, 3)))


def test_gallery_rejects_bad_k():
    gallery = GalleryIndex(["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError):
        gallery.search(np.ones((1, 3)), 3)
    with pytest.raises(ValueError):
        gallery.search(np.ones((1, 3)), 0)


def test_gallery_len_and_projected_flag():
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    plain = GalleryIndex(["a", "b"], embeddings)
    projected = GalleryIndex(["a", "b"], embeddings, make_projector())
    assert len(plain) == 2
    assert plain.projected is False
    assert projected.projected is True


def test_search_text_uses_encoder_output():
    gallery = GalleryIndex(["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    encoder = FakeEncoder({"x": [0.9, 0.1, 0.0], "y": [0.0, 1.0, 0.0]})
    result = search_text(encoder, gallery, ["x", "y"], 1)
    assert [row[0][0] for row in result] == ["a", "b"]
    assert result[1][0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_text_refuses_projected_gallery():
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gallery = GalleryIndex(["a", "b"], embeddings, make_projector())
    with pytest.raises(ValueError):
        search_text(FakeEncoder({"x": [1.0, 0.0, 0.0]}), gallery, ["x"], 1)


def test_search_text_rejects_empty_texts():
    gallery = GalleryIndex(["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError):
        search_text(FakeEncoder({}), gallery, [], 1)


def test_read_plates_passes_blobs_and_returns_results():
    results = [plate("AB12345"), plate(None)]
    reader = FakeReader(results)
    assert read_plates(reader, [b"a", b"b"]) == results
    assert reader.seen == [b"a", b"b"]


def test_read_plates_rejects_length_mismatch():
    with pytest.raises(ValueError):
        read_plates(FakeReader([plate("AB12345")]), [b"a", b"b"])


def test_read_plates_rejects_foreign_results():
    with pytest.raises(ValueError):
        read_plates(FakeReader(["x", "y"]), [b"a", b"b"])


def test_find_plates_returns_indices_and_distances():
    results = [plate("AB12345"), plate("AB12346"), plate("XY99999")]
    assert find_plates(results, "AB12345", 0) == [(0, 0)]
    assert find_plates(results, "AB12345", 1) == [(0, 0), (1, 1)]
    assert find_plates(results, "AB12345", 7) == [(0, 0), (1, 1), (2, 7)]


def test_find_plates_skips_unread_images():
    results = [plate(None), plate("AB12345")]
    assert find_plates(results, "AB12345", 1) == [(1, 0)]


def test_find_plates_rejects_negative_edits():
    with pytest.raises(ValueError):
        find_plates([], "AB12345", -1)
