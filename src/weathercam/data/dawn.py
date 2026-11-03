from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from weathercam.core.boxes import from_voc
from weathercam.core.types import Box


@dataclass(frozen=True)
class LabeledBox:
    name: str
    box: Box


@dataclass(frozen=True)
class Annotation:
    width: int
    height: int
    objects: tuple[LabeledBox, ...]


def _number(parent, tag: str) -> float:
    element = parent.find(tag)
    if element is None or element.text is None:
        raise ValueError(f"missing <{tag}> in annotation")
    try:
        return float(element.text.strip())
    except ValueError as err:
        raise ValueError(f"non-numeric <{tag}>: {element.text!r}") from err


def _parse_object(element, width: float, height: float) -> LabeledBox:
    name_element = element.find("name")
    if name_element is None or not (name_element.text or "").strip():
        raise ValueError("object without a name")
    bndbox = element.find("bndbox")
    if bndbox is None:
        raise ValueError("object without a bndbox")
    box = from_voc(
        _number(bndbox, "xmin"),
        _number(bndbox, "ymin"),
        _number(bndbox, "xmax"),
        _number(bndbox, "ymax"),
    )
    if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
        raise ValueError(f"box {box} outside image {width}x{height}")
    return LabeledBox(name_element.text.strip(), box)


def parse_voc_xml(text: str) -> Annotation:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as err:
        raise ValueError(f"malformed XML: {err}") from err
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size> in annotation")
    width = _number(size, "width")
    height = _number(size, "height")
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive image size: {width}x{height}")
    objects = tuple(_parse_object(e, width, height) for e in root.findall("object"))
    return Annotation(int(width), int(height), objects)


def load_annotation(path) -> Annotation:
    with open(Path(path), "r", encoding="utf-8") as f:
        return parse_voc_xml(f.read())


def map_classes(annotation: Annotation, mapping: dict) -> list:
    result = []
    for item in annotation.objects:
        if item.name not in mapping:
            raise ValueError(f"class {item.name!r} not in mapping")
        label = mapping[item.name]
        if label is not None:
            result.append((label, item.box))
    return result
