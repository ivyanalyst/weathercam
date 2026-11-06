import json

import pytest

from weathercam.data.splitfile import load_split_paths

DATA = {
    "parts": {
        "train": {"paths": ["A/A/a1.jpg", "A/A/a2.jpg"]},
        "val": {"paths": ["B/B/b1.jpg"]},
        "test": {"paths": []},
    }
}


@pytest.fixture
def split_file(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(DATA))
    return path


@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_each_part(split_file, part):
    assert load_split_paths(split_file, part) == DATA["parts"][part]["paths"]


def test_order_is_kept(split_file):
    assert load_split_paths(split_file, "train") == ["A/A/a1.jpg", "A/A/a2.jpg"]


def test_returns_new_list(split_file):
    first = load_split_paths(split_file, "train")
    first.append("X")
    assert load_split_paths(split_file, "train") == ["A/A/a1.jpg", "A/A/a2.jpg"]


def test_unknown_part_raises(split_file):
    with pytest.raises(ValueError):
        load_split_paths(split_file, "holdout")


def test_unknown_part_raises_without_file(tmp_path):
    with pytest.raises(ValueError):
        load_split_paths(tmp_path / "missing.json", "holdout")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split_paths(tmp_path / "missing.json", "train")
