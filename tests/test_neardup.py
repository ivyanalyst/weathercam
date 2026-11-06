import numpy as np
import pytest
from PIL import Image

from weathercam.data.neardup import (
    UnionFind,
    dhash,
    group_by_distance,
    hash_distance,
    hash_file,
)

FULL = 2**256 - 1
X = np.arange(17)
Y = np.arange(16)


def gray(array):
    return Image.fromarray(np.asarray(array, dtype=np.uint8))


def decreasing():
    return gray(np.tile(255 - 15 * X, (16, 1)))


def increasing():
    return gray(np.tile(15 * X, (16, 1)))


def pattern():
    return (X[None, :] * 37 + Y[:, None] * 91) % 200 + 20


def test_distance_identical_is_zero():
    assert hash_distance(0b1011, 0b1011) == 0


def test_distance_one_bit():
    assert hash_distance(0b1010, 0b1000) == 1


def test_distance_symmetric():
    assert hash_distance(0b1100, 0b0110) == hash_distance(0b0110, 0b1100) == 2


def test_distance_all_bits():
    assert hash_distance(0, FULL) == 256


def test_hash_constant_image_is_zero():
    assert dhash(gray(np.full((16, 17), 100))) == 0


def test_hash_decreasing_is_all_ones():
    assert dhash(decreasing()) == FULL


def test_hash_increasing_is_zero():
    assert dhash(increasing()) == 0


def test_hash_one_row_sets_sixteen_bits():
    array = np.full((16, 17), 100)
    array[0] = 255 - 15 * X
    assert dhash(gray(array)).bit_count() == 16


def test_hash_ignores_constant_offset():
    base = dhash(gray(pattern()))
    assert base != 0
    assert dhash(gray(pattern() + 30)) == base


def test_hash_survives_nearest_upscale():
    image = gray(pattern())
    big = image.resize((34, 32), Image.Resampling.NEAREST)
    assert dhash(big) == dhash(image)


def test_hash_constant_rgb_is_zero():
    assert dhash(Image.new("RGB", (40, 30), (10, 200, 90))) == 0


@pytest.mark.parametrize("size", [0, -3])
def test_hash_rejects_bad_size(size):
    with pytest.raises(ValueError):
        dhash(decreasing(), size)


def test_union_find_starts_as_singletons():
    forest = UnionFind(4)
    assert [forest.find(i) for i in range(4)] == [0, 1, 2, 3]


def test_union_find_union_reports_merge():
    forest = UnionFind(3)
    assert forest.union(0, 1) is True
    assert forest.union(1, 0) is False


def test_union_find_is_transitive():
    forest = UnionFind(4)
    forest.union(0, 1)
    forest.union(1, 2)
    assert forest.find(0) == forest.find(2)
    assert forest.find(3) != forest.find(0)


def test_union_find_find_is_idempotent():
    forest = UnionFind(3)
    forest.union(0, 2)
    root = forest.find(2)
    assert forest.find(root) == root


def test_union_find_connected_pair_shares_root():
    forest = UnionFind(2)
    forest.union(0, 1)
    assert forest.find(0) == forest.find(1)


def test_group_empty():
    assert group_by_distance([], 5) == []


def test_group_single():
    assert group_by_distance([123], 5) == [0]


def test_group_identical_hashes():
    assert group_by_distance([7, 7, 7], 0) == [0, 0, 0]


@pytest.mark.parametrize("threshold,expected", [(3, [0, 0]), (2, [0, 1])])
def test_group_boundary_is_inclusive(threshold, expected):
    assert group_by_distance([0, 0b111], threshold) == expected


def test_group_chain_merges_through_middle():
    assert group_by_distance([0, 0b111, 0b111111], 3) == [0, 0, 0]


def test_group_id_is_smallest_index():
    assert group_by_distance([(1 << 10) - 1, 0, 0], 0) == [0, 1, 1]


def test_group_interleaved():
    assert group_by_distance([0, 1023, 0, 1023], 0) == [0, 1, 0, 1]


def test_group_rejects_negative_threshold():
    with pytest.raises(ValueError):
        group_by_distance([0, 1], -1)


def test_group_is_deterministic():
    hashes = [0, 5, 3, 0b1111, 2, 9]
    assert group_by_distance(hashes, 3) == group_by_distance(hashes, 3)


def test_hash_file_same_for_two_sizes(tmp_path):
    column = np.linspace(255, 0, 400).astype(np.uint8)
    image = Image.fromarray(np.tile(column, (300, 1))).convert("RGB")
    image.save(tmp_path / "a.jpg", quality=95)
    image.resize((200, 150)).save(tmp_path / "b.jpg", quality=95)
    assert hash_file(tmp_path / "a.jpg") == FULL
    assert hash_file(tmp_path / "b.jpg") == FULL


def test_hash_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "missing.jpg")
