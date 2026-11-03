import pytest

from weathercam.core.types import Box
from weathercam.data.dawn import (
    Annotation,
    LabeledBox,
    load_annotation,
    map_classes,
    parse_voc_xml,
)


REAL_XML = """<annotation>
  <folder>input_fog</folder>
  <filename>foggy-001.jpg</filename>
  <path>/home/kenmo/v_ws/OpenLabeling/main/input_fog/foggy-001.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>1280</width>
    <height>675</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>car</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>462</xmin>
      <ymin>467</ymin>
      <xmax>601</xmax>
      <ymax>598</ymax>
    </bndbox>
  </object>
</annotation>
"""


def make_xml(objects, width=1280, height=675):
    body = "".join(objects)
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height>"
        f"<depth>3</depth></size>{body}</annotation>"
    )


def obj(name, xmin, ymin, xmax, ymax):
    return (
        f"<object><name>{name}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"
    )


def test_parse_real_file_structure():
    ann = parse_voc_xml(REAL_XML)
    assert ann == Annotation(1280, 675, (LabeledBox("car", Box(462, 467, 602, 599)),))


def test_parse_keeps_object_order_and_names():
    text = make_xml([obj("truck", 1, 2, 3, 4), obj("car", 10, 20, 30, 40)])
    ann = parse_voc_xml(text)
    assert [o.name for o in ann.objects] == ["truck", "car"]
    assert ann.objects[1].box == Box(10, 20, 31, 41)


def test_parse_no_objects():
    ann = parse_voc_xml(make_xml([]))
    assert ann.width == 1280
    assert ann.height == 675
    assert ann.objects == ()


def test_parse_box_touching_image_edges():
    ann = parse_voc_xml(make_xml([obj("car", 0, 0, 1279, 674)]))
    assert ann.objects[0].box == Box(0, 0, 1280, 675)


def test_parse_rejects_xmax_equal_width():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([obj("car", 0, 0, 1280, 10)]))


def test_parse_rejects_ymax_equal_height():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([obj("car", 0, 0, 10, 675)]))


def test_parse_rejects_negative_min():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([obj("car", -1, 0, 10, 10)]))


def test_parse_rejects_inverted_box():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([obj("car", 10, 0, 9, 10)]))


def test_parse_rejects_malformed_xml():
    with pytest.raises(ValueError):
        parse_voc_xml("<annotation><size>")


def test_parse_rejects_missing_size():
    with pytest.raises(ValueError):
        parse_voc_xml("<annotation></annotation>")


def test_parse_rejects_non_positive_size():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([], width=0))


def test_parse_rejects_missing_box_field():
    text = make_xml(["<object><name>car</name><bndbox><xmin>1</xmin></bndbox></object>"])
    with pytest.raises(ValueError):
        parse_voc_xml(text)


def test_parse_rejects_non_numeric_value():
    with pytest.raises(ValueError):
        parse_voc_xml(make_xml([obj("car", "abc", 0, 10, 10)]))


def test_parse_rejects_missing_name():
    text = make_xml(["<object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>"])
    with pytest.raises(ValueError):
        parse_voc_xml(text)


def test_load_annotation_reads_file(tmp_path):
    path = tmp_path / "a.xml"
    path.write_text(REAL_XML, encoding="utf-8")
    assert load_annotation(path) == parse_voc_xml(REAL_XML)


def test_load_annotation_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_annotation(tmp_path / "nope.xml")


def test_map_classes_maps_and_keeps_order():
    ann = parse_voc_xml(make_xml([obj("truck", 1, 2, 3, 4), obj("car", 10, 20, 30, 40)]))
    result = map_classes(ann, {"car": 0, "truck": 1})
    assert result == [(1, Box(1, 2, 4, 5)), (0, Box(10, 20, 31, 41))]


def test_map_classes_drops_none():
    ann = parse_voc_xml(make_xml([obj("person", 1, 2, 3, 4), obj("car", 10, 20, 30, 40)]))
    assert map_classes(ann, {"car": 0, "person": None}) == [(0, Box(10, 20, 31, 41))]


def test_map_classes_unknown_name_raises():
    ann = parse_voc_xml(make_xml([obj("zebra", 1, 2, 3, 4)]))
    with pytest.raises(ValueError):
        map_classes(ann, {"car": 0})


def test_map_classes_empty():
    assert map_classes(parse_voc_xml(make_xml([])), {"car": 0}) == []
