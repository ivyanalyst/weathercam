import zipfile

import pytest

from weathercam.data.veri import ZIP_ROOT, VeriReader, read_image

CONTENT = {
    ("train", "0001_c001_00000001_0.jpg"): b"train-one",
    ("train", "0002_c002_00000002_0.jpg"): b"train-two",
    ("test", "0003_c003_00000003_1.jpg"): b"test-one",
    ("query", "0004_c004_00000004_2.jpg"): b"query-one",
}


def make_zip(path):
    with zipfile.ZipFile(path, "w") as archive:
        for (split, name), data in CONTENT.items():
            archive.writestr(f"{ZIP_ROOT}/image_{split}/{name}", data)
    return path


def make_folder(path):
    for (split, name), data in CONTENT.items():
        directory = path / f"image_{split}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    return path


@pytest.fixture(params=["zip", "folder"])
def source(request, tmp_path):
    if request.param == "zip":
        return make_zip(tmp_path / "VeRi.zip")
    return make_folder(tmp_path / "VeRi")


@pytest.fixture
def zip_path(tmp_path):
    return make_zip(tmp_path / "VeRi.zip")


@pytest.mark.parametrize("key", list(CONTENT))
def test_read_each_item(source, key):
    split, name = key
    with VeriReader(source) as reader:
        assert reader.read_image(split, name) == CONTENT[key]


def test_repeated_reads_equal(source):
    with VeriReader(source) as reader:
        first = reader.read_image("train", "0001_c001_00000001_0.jpg")
        second = reader.read_image("train", "0001_c001_00000001_0.jpg")
    assert first == second == b"train-one"


def test_unknown_split_raises_value_error(source):
    with VeriReader(source) as reader:
        with pytest.raises(ValueError):
            reader.read_image("val", "0001_c001_00000001_0.jpg")


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "../x"])
def test_bad_names_raise_value_error(source, name):
    with VeriReader(source) as reader:
        with pytest.raises(ValueError):
            reader.read_image("train", name)


def test_missing_item_raises_file_not_found(source):
    with VeriReader(source) as reader:
        with pytest.raises(FileNotFoundError):
            reader.read_image("train", "9999_c009_00000009_0.jpg")


def test_reader_usable_after_missing_item(source):
    with VeriReader(source) as reader:
        with pytest.raises(FileNotFoundError):
            reader.read_image("train", "9999_c009_00000009_0.jpg")
        assert reader.read_image("test", "0003_c003_00000003_1.jpg") == b"test-one"


def test_context_manager_closes(source):
    reader = VeriReader(source)
    assert not reader.closed
    with reader as entered:
        assert entered is reader
    assert reader.closed


def test_read_after_close_raises_value_error(source):
    reader = VeriReader(source)
    reader.close()
    with pytest.raises(ValueError):
        reader.read_image("train", "0001_c001_00000001_0.jpg")


def test_close_is_idempotent(source):
    reader = VeriReader(source)
    reader.close()
    reader.close()
    assert reader.closed


def test_accepts_str_source(source):
    with VeriReader(str(source)) as reader:
        assert reader.read_image("query", "0004_c004_00000004_2.jpg") == b"query-one"


def test_zip_opens_one_handle_for_many_reads(zip_path, monkeypatch):
    created = []
    original = zipfile.ZipFile

    class Counting(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(zipfile, "ZipFile", Counting)
    reader = VeriReader(zip_path)
    for _ in range(5):
        for split, name in CONTENT:
            reader.read_image(split, name)
    reader.close()
    assert len(created) == 1
    assert created[0].fp is None


def test_zip_matches_module_read_image(zip_path):
    with VeriReader(zip_path) as reader:
        for split, name in CONTENT:
            assert reader.read_image(split, name) == read_image(zip_path, split, name)


def test_zip_uppercase_suffix(tmp_path):
    path = make_zip(tmp_path / "VeRi.ZIP")
    with VeriReader(path) as reader:
        assert reader.read_image("train", "0002_c002_00000002_0.jpg") == b"train-two"


def test_zip_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VeriReader(tmp_path / "missing.zip")


def test_zip_not_a_zip_raises_value_error(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError):
        VeriReader(path)


def test_folder_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VeriReader(tmp_path / "nope")


def test_folder_source_is_a_file_raises(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        VeriReader(path)


def test_folder_matches_zip(tmp_path):
    zip_source = make_zip(tmp_path / "VeRi.zip")
    folder_source = make_folder(tmp_path / "VeRi")
    with VeriReader(zip_source) as from_zip, VeriReader(folder_source) as from_folder:
        for split, name in CONTENT:
            assert from_zip.read_image(split, name) == from_folder.read_image(split, name)
