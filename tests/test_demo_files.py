"""Checks on the shipped demo gallery (demo/)."""
import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo"
COLUMNS = ["file", "colour", "type", "view", "scene"]
ZIP_PREFIX = "VeRi/image_test/"


def labels():
    with open(DEMO / "labels.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def gallery():
    return np.load(DEMO / "gallery_demo.npz")


def test_labels_complete():
    rows = labels()
    assert len(rows) == 39
    assert list(rows[0].keys()) == COLUMNS
    for row in rows:
        assert all(row[c].strip() for c in COLUMNS), row["file"]
    files = [row["file"] for row in rows]
    assert len(set(files)) == len(files)


def test_gallery_names_match_labels_and_images():
    names = sorted(str(n) for n in gallery()["names"])
    on_disk = sorted(p.name for p in (DEMO / "images").glob("*.jpg"))
    assert names == sorted(row["file"] for row in labels())
    assert names == on_disk


def test_embeddings_unit_norm():
    data = gallery()
    emb = data["embeddings"]
    assert emb.shape == (len(data["names"]), 512)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    norms = np.linalg.norm(emb, axis=1)
    assert np.abs(norms - 1.0).max() < 1e-5


def test_zip_layout_and_bytes():
    names = sorted(str(n) for n in gallery()["names"])
    with zipfile.ZipFile(DEMO / "demo_images.zip") as zf:
        members = sorted(zf.namelist())
        assert members == [ZIP_PREFIX + n for n in names]
        for member in members:
            disk = (DEMO / "images" / member.split("/")[-1]).read_bytes()
            assert zf.read(member) == disk


def test_images_share_one_size():
    image_module = pytest.importorskip("PIL.Image")
    sizes = set()
    for path in (DEMO / "images").glob("*.jpg"):
        with image_module.open(path) as im:
            sizes.add(im.size)
    assert sizes == {(640, 480)}
