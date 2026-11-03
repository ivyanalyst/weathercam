import zipfile

import pytest

from weathercam.data.veri import (
    Labels,
    NamedItem,
    VeriItem,
    named_items,
    read_labels,
    read_names,
    unlabeled_count,
)


def make_xml(items, declared=None):
    declared = len(items) if declared is None else declared
    lines = "".join(
        f'        <Item imageName="{n}" vehicleID="{v}" cameraID="{c}" colorID="{col}" typeID="{t}" />\r\n'
        for n, v, c, col, t in items
    )
    text = (
        '<?xml version="1.0" encoding="gb2312" ?>\r\n'
        '<TrainingImages Version="1.0">\r\n'
        f'    <Items number="{declared}">\r\n{lines}'
        "    </Items>\r\n</TrainingImages>\r\n"
    )
    return text.encode("gb2312")


TRAIN = make_xml([("a.jpg", "0001", "c001", 1, 4), ("b.jpg", "0002", "c002", 2, 1)], declared=3)
TEST = make_xml([("c.jpg", "0003", "c003", 10, 2)])
COLORS = b"1 yellow\r\n2 orange\r\n10 black"
TYPES = b"1 sedan\r\n2 suv\r\n4 hatchback"

FILES = {
    "train_label.xml": TRAIN,
    "test_label.xml": TEST,
    "list_color.txt": COLORS,
    "list_type.txt": TYPES,
}


@pytest.fixture
def zip_source(tmp_path):
    path = tmp_path / "VeRi.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in FILES.items():
            z.writestr(f"VeRi/{name}", data)
    return path


@pytest.fixture
def folder_source(tmp_path):
    folder = tmp_path / "veri"
    folder.mkdir()
    for name, data in FILES.items():
        (folder / name).write_bytes(data)
    return folder


@pytest.fixture(params=["zip", "folder"])
def source(request, zip_source, folder_source):
    return zip_source if request.param == "zip" else folder_source


@pytest.fixture
def empty_source(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("VeRi/readme.txt", b"x")
    return path


def test_read_labels_train(source):
    labels = read_labels(source, "train")
    assert labels.declared == 3
    assert [i.image_name for i in labels.items] == ["a.jpg", "b.jpg"]
    assert labels.items[0] == VeriItem("a.jpg", "0001", "c001", 1, 4)


def test_read_labels_test(source):
    labels = read_labels(source, "test")
    assert labels.declared == 1
    assert labels.items[0].vehicle_id == "0003"


def test_read_labels_query_raises(zip_source):
    with pytest.raises(ValueError):
        read_labels(zip_source, "query")


def test_read_labels_unknown_split_raises(folder_source):
    with pytest.raises(ValueError):
        read_labels(folder_source, "val")


def test_read_labels_missing_file_in_zip(empty_source):
    with pytest.raises(FileNotFoundError):
        read_labels(empty_source, "train")


def test_read_labels_missing_file_in_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path, "train")


def test_read_names(source):
    colors, types = read_names(source)
    assert colors == {1: "yellow", 2: "orange", 10: "black"}
    assert types == {1: "sedan", 2: "suv", 4: "hatchback"}


def test_read_names_missing_in_zip(empty_source):
    with pytest.raises(FileNotFoundError):
        read_names(empty_source)


def test_read_names_missing_in_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_names(tmp_path)


def test_unlabeled_count_train(source):
    assert unlabeled_count(read_labels(source, "train")) == 1


def test_unlabeled_count_test(source):
    assert unlabeled_count(read_labels(source, "test")) == 0


def test_unlabeled_count_negative_raises():
    items = (VeriItem("a.jpg", "0001", "c001", 1, 4), VeriItem("b.jpg", "0002", "c002", 2, 1))
    with pytest.raises(ValueError):
        unlabeled_count(Labels(1, items))


def test_named_items_maps_names_and_keeps_order(source):
    colors, types = read_names(source)
    result = named_items(read_labels(source, "train"), colors, types)
    assert result == (
        NamedItem("a.jpg", "0001", "c001", "yellow", "hatchback"),
        NamedItem("b.jpg", "0002", "c002", "orange", "sedan"),
    )


def test_named_items_empty():
    assert named_items(Labels(0, ()), {1: "yellow"}, {1: "sedan"}) == ()


def test_named_items_unknown_color_raises():
    labels = Labels(1, (VeriItem("a.jpg", "0001", "c001", 99, 1),))
    with pytest.raises(ValueError):
        named_items(labels, {1: "yellow"}, {1: "sedan"})


def test_named_items_unknown_type_raises():
    labels = Labels(1, (VeriItem("a.jpg", "0001", "c001", 1, 99),))
    with pytest.raises(ValueError):
        named_items(labels, {1: "yellow"}, {1: "sedan"})
