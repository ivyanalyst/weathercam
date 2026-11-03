import zipfile

import pytest

from weathercam.data.veri import (
    Labels,
    VeriItem,
    parse_labels,
    parse_names,
    read_image,
)


def make_xml(items, declared=None, extra=""):
    declared = len(items) if declared is None else declared
    lines = "".join(
        f'        <Item imageName="{n}" vehicleID="{v}" cameraID="{c}" colorID="{col}" typeID="{t}" />\r\n'
        for n, v, c, col, t in items
    )
    text = (
        '<?xml version="1.0" encoding="gb2312" ?>\r\n'
        '<TrainingImages Version="1.0">\r\n'
        f'    <Items number="{declared}">\r\n{extra}{lines}'
        "    </Items>\r\n</TrainingImages>\r\n"
    )
    return text.encode("gb2312")


ITEMS = [
    ("0001_c001_00016450_0.jpg", "0001", "c001", 1, 4),
    ("0002_c003_00084280_0.jpg", "0002", "c003", 10, 2),
]


def test_parse_labels_items():
    labels = parse_labels(make_xml(ITEMS))
    assert labels == Labels(
        2,
        (
            VeriItem("0001_c001_00016450_0.jpg", "0001", "c001", 1, 4),
            VeriItem("0002_c003_00084280_0.jpg", "0002", "c003", 10, 2),
        ),
    )


def test_parse_labels_keeps_zero_padded_vehicle_id():
    labels = parse_labels(make_xml(ITEMS))
    assert labels.items[0].vehicle_id == "0001"
    assert labels.items[0].camera_id == "c001"


def test_parse_labels_ids_are_ints():
    item = parse_labels(make_xml(ITEMS)).items[1]
    assert item.color_id == 10
    assert item.type_id == 2


def test_parse_labels_declared_differs_from_found():
    labels = parse_labels(make_xml(ITEMS, declared=5))
    assert labels.declared == 5
    assert len(labels.items) == 2


def test_parse_labels_no_items():
    labels = parse_labels(make_xml([], declared=0))
    assert labels.declared == 0
    assert labels.items == ()


def test_parse_labels_survives_gb2312_text():
    extra = "        <!-- \u8f66 -->\r\n"
    raw = make_xml(ITEMS, extra=extra)
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert len(parse_labels(raw).items) == 2


def test_parse_labels_missing_field_raises():
    raw = (
        '<?xml version="1.0" encoding="gb2312" ?>\r\n<Items number="1">\r\n'
        '<Item imageName="a.jpg" vehicleID="0001" cameraID="c001" colorID="1" />\r\n</Items>'
    ).encode("gb2312")
    with pytest.raises(ValueError):
        parse_labels(raw)


def test_parse_labels_non_integer_id_raises():
    raw = make_xml([("a.jpg", "0001", "c001", "x", 4)])
    with pytest.raises(ValueError):
        parse_labels(raw)


def test_parse_labels_missing_declared_raises():
    raw = (
        '<?xml version="1.0" encoding="gb2312" ?>\r\n'
        '<Item imageName="a.jpg" vehicleID="0001" cameraID="c001" colorID="1" typeID="4" />\r\n'
    ).encode("gb2312")
    with pytest.raises(ValueError):
        parse_labels(raw)


def test_parse_names_crlf_without_final_newline():
    assert parse_names(b"1 yellow\r\n2 orange\r\n3 green") == {1: "yellow", 2: "orange", 3: "green"}


def test_parse_names_lf_and_blank_lines():
    assert parse_names(b"1 sedan\n\n2 suv\n") == {1: "sedan", 2: "suv"}


def test_parse_names_non_integer_id_raises():
    with pytest.raises(ValueError):
        parse_names(b"x yellow\r\n")


def test_parse_names_missing_name_raises():
    with pytest.raises(ValueError):
        parse_names(b"1\r\n")


def make_zip(tmp_path):
    path = tmp_path / "VeRi.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("VeRi/image_train/a.jpg", b"train-bytes")
        z.writestr("VeRi/image_test/b.jpg", b"test-bytes")
        z.writestr("VeRi/image_query/c.jpg", b"query-bytes")
    return path


def test_read_image_from_zip(tmp_path):
    zip_path = make_zip(tmp_path)
    assert read_image(zip_path, "train", "a.jpg") == b"train-bytes"
    assert read_image(zip_path, "test", "b.jpg") == b"test-bytes"
    assert read_image(zip_path, "query", "c.jpg") == b"query-bytes"


def test_read_image_from_folder(tmp_path):
    folder = tmp_path / "veri"
    (folder / "image_test").mkdir(parents=True)
    (folder / "image_test" / "b.jpg").write_bytes(b"folder-bytes")
    assert read_image(folder, "test", "b.jpg") == b"folder-bytes"


def test_read_image_missing_in_zip(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(make_zip(tmp_path), "train", "nope.jpg")


def test_read_image_missing_in_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path, "train", "nope.jpg")


def test_read_image_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        read_image(make_zip(tmp_path), "val", "a.jpg")
