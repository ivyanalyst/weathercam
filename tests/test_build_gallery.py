import subprocess
import sys
from pathlib import Path

import numpy as np

from weathercam.service.handlers import GalleryIndex

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gallery.py"
DIM = 512


def make_source(path, train=40, test=10, scale=1.0, names=None, drop=None):
    rng = np.random.default_rng(0)
    total = train + test
    vectors = rng.normal(size=(total, DIM))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True) * scale).astype(np.float32)
    if names is None:
        names = [f"img{i:03d}.jpg" for i in range(total)]
    arrays = {
        "image_names": np.array(names),
        "embeddings": vectors,
        "split": np.array(["train"] * train + ["test"] * test),
    }
    if drop is not None:
        del arrays[drop]
    np.savez(path, **arrays)
    return path


def run(source, output, split="train", count=None, seed=None):
    command = [sys.executable, str(SCRIPT), "--embeddings", str(source), "--split", split, "--output", str(output)]
    if count is not None:
        command += ["--count", str(count)]
    if seed is not None:
        command += ["--seed", str(seed)]
    return subprocess.run(command, capture_output=True, text=True)


def test_same_seed_gives_identical_output(tmp_path):
    source = make_source(tmp_path / "source.npz")
    assert run(source, tmp_path / "a.npz", count=5, seed=7).returncode == 0
    assert run(source, tmp_path / "b.npz", count=5, seed=7).returncode == 0
    a, b = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
    assert a["names"].tolist() == b["names"].tolist()
    assert np.array_equal(a["embeddings"], b["embeddings"])


def test_different_seed_changes_the_selection(tmp_path):
    source = make_source(tmp_path / "source.npz")
    assert run(source, tmp_path / "a.npz", count=5, seed=1).returncode == 0
    assert run(source, tmp_path / "b.npz", count=5, seed=2).returncode == 0
    assert np.load(tmp_path / "a.npz")["names"].tolist() != np.load(tmp_path / "b.npz")["names"].tolist()


def test_rows_are_exact_from_the_right_split_in_original_order(tmp_path):
    source = make_source(tmp_path / "source.npz")
    result = run(source, tmp_path / "out.npz", split="test", count=4, seed=3)
    assert result.returncode == 0
    original = np.load(source)
    out = np.load(tmp_path / "out.npz")
    position = {name: i for i, name in enumerate(original["image_names"].tolist())}
    rows = [position[name] for name in out["names"].tolist()]
    assert rows == sorted(rows)
    assert all(original["split"][row] == "test" for row in rows)
    assert np.array_equal(out["embeddings"], original["embeddings"][rows])
    assert out["embeddings"].dtype == np.float32
    assert out["embeddings"].shape == (4, DIM)


def test_no_count_takes_the_whole_split(tmp_path):
    source = make_source(tmp_path / "source.npz")
    assert run(source, tmp_path / "out.npz", split="test").returncode == 0
    out = np.load(tmp_path / "out.npz")
    assert out["embeddings"].shape == (10, DIM)
    assert out["names"].tolist() == [f"img{i:03d}.jpg" for i in range(40, 50)]


def test_non_unit_norms_are_refused(tmp_path):
    source = make_source(tmp_path / "source.npz", scale=2.0)
    result = run(source, tmp_path / "out.npz")
    assert result.returncode != 0
    assert "unit length" in result.stderr
    assert not (tmp_path / "out.npz").exists()


def test_duplicate_names_are_refused(tmp_path):
    source = make_source(tmp_path / "source.npz", names=["a.jpg"] * 50)
    result = run(source, tmp_path / "out.npz")
    assert result.returncode != 0
    assert "duplicate" in result.stderr


def test_count_larger_than_the_split_is_refused(tmp_path):
    source = make_source(tmp_path / "source.npz")
    result = run(source, tmp_path / "out.npz", count=41)
    assert result.returncode != 0
    assert "count must be in [1, 40]" in result.stderr


def test_unknown_split_is_refused(tmp_path):
    source = make_source(tmp_path / "source.npz")
    result = run(source, tmp_path / "out.npz", split="val")
    assert result.returncode != 0
    assert not (tmp_path / "out.npz").exists()


def test_missing_key_is_refused(tmp_path):
    source = make_source(tmp_path / "source.npz", drop="split")
    result = run(source, tmp_path / "out.npz")
    assert result.returncode != 0
    assert "no split" in result.stderr


def test_existing_output_is_not_overwritten(tmp_path):
    source = make_source(tmp_path / "source.npz")
    out = tmp_path / "out.npz"
    out.write_bytes(b"keep")
    result = run(source, out)
    assert result.returncode != 0
    assert "exists" in result.stderr
    assert out.read_bytes() == b"keep"


def test_output_builds_a_gallery_index(tmp_path):
    source = make_source(tmp_path / "source.npz")
    assert run(source, tmp_path / "out.npz").returncode == 0
    out = np.load(tmp_path / "out.npz")
    index = GalleryIndex(out["names"], out["embeddings"])
    assert len(index) == 40
    index.search(out["embeddings"][:1], 1)
