import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "test", "query")
ZIP_ROOT = "VeRi"
FIELDS = ("imageName", "vehicleID", "cameraID", "colorID", "typeID")

ITEM_PATTERN = re.compile(r"<Item\s([^>]*?)/>")
DECLARED_PATTERN = re.compile(r'<Items\s+number="(\d+)"')


@dataclass(frozen=True)
class VeriItem:
    image_name: str
    vehicle_id: str
    camera_id: str
    color_id: int
    type_id: int


@dataclass(frozen=True)
class Labels:
    declared: int
    items: tuple[VeriItem, ...]


def _attribute(attrs: str, name: str) -> str:
    match = re.search(rf'\b{name}="([^"]*)"', attrs)
    if match is None:
        raise ValueError(f"item without {name}: {attrs.strip()[:80]}")
    return match.group(1)


def _integer(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f"non-integer {name}: {value!r}") from err


def parse_labels(data: bytes) -> Labels:
    text = data.decode("gb2312", errors="replace")
    declared = DECLARED_PATTERN.search(text)
    if declared is None:
        raise ValueError("missing <Items number=...>")
    items = []
    for match in ITEM_PATTERN.finditer(text):
        attrs = match.group(1)
        items.append(
            VeriItem(
                _attribute(attrs, "imageName"),
                _attribute(attrs, "vehicleID"),
                _attribute(attrs, "cameraID"),
                _integer(_attribute(attrs, "colorID"), "colorID"),
                _integer(_attribute(attrs, "typeID"), "typeID"),
            )
        )
    return Labels(int(declared.group(1)), tuple(items))


def parse_names(data: bytes) -> dict[int, str]:
    result = {}
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"line without a name: {line!r}")
        result[_integer(parts[0], "id")] = parts[1].strip()
    return result


def read_image(source, split: str, name: str) -> bytes:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    source = Path(source)
    if source.suffix.lower() == ".zip":
        member = f"{ZIP_ROOT}/image_{split}/{name}"
        with zipfile.ZipFile(source) as z:
            try:
                return z.read(member)
            except KeyError as err:
                raise FileNotFoundError(f"{member} not in {source}") from err
    return (source / f"image_{split}" / name).read_bytes()


LABEL_SPLITS = ("train", "test")


@dataclass(frozen=True)
class NamedItem:
    image_name: str
    vehicle_id: str
    camera_id: str
    color: str
    type: str


def _read_member(source, name: str) -> bytes:
    source = Path(source)
    if source.suffix.lower() == ".zip":
        member = f"{ZIP_ROOT}/{name}"
        with zipfile.ZipFile(source) as z:
            try:
                return z.read(member)
            except KeyError as err:
                raise FileNotFoundError(f"{member} not in {source}") from err
    return (source / name).read_bytes()


def read_labels(source, split: str) -> Labels:
    if split not in LABEL_SPLITS:
        raise ValueError(f"no label file for split: {split!r}")
    return parse_labels(_read_member(source, f"{split}_label.xml"))


def read_names(source) -> tuple[dict[int, str], dict[int, str]]:
    colors = parse_names(_read_member(source, "list_color.txt"))
    types = parse_names(_read_member(source, "list_type.txt"))
    return colors, types


def unlabeled_count(labels: Labels) -> int:
    missing = labels.declared - len(labels.items)
    if missing < 0:
        raise ValueError(f"found {len(labels.items)} items, declared {labels.declared}")
    return missing


def named_items(labels: Labels, colors: dict[int, str], types: dict[int, str]) -> tuple[NamedItem, ...]:
    result = []
    for item in labels.items:
        if item.color_id not in colors:
            raise ValueError(f"unknown colorID: {item.color_id}")
        if item.type_id not in types:
            raise ValueError(f"unknown typeID: {item.type_id}")
        result.append(
            NamedItem(
                item.image_name,
                item.vehicle_id,
                item.camera_id,
                colors[item.color_id],
                types[item.type_id],
            )
        )
    return tuple(result)


class VeriReader:
    def __init__(self, source):
        path = Path(source)
        self._zip = None
        self._folder = None
        self._closed = False
        if path.suffix.lower() == ".zip":
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            try:
                self._zip = zipfile.ZipFile(path)
            except zipfile.BadZipFile as err:
                raise ValueError(f"not a zip archive: {path}") from err
        else:
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            if not path.is_dir():
                raise ValueError(f"not a directory: {path}")
            self._folder = path

    @property
    def closed(self) -> bool:
        return self._closed

    def read_image(self, split: str, name: str) -> bytes:
        if self._closed:
            raise ValueError("reader is closed")
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"bad image name: {name!r}")
        if self._zip is not None:
            member = f"{ZIP_ROOT}/image_{split}/{name}"
            try:
                return self._zip.read(member)
            except KeyError as err:
                raise FileNotFoundError(f"{member} not in archive") from err
        return (self._folder / f"image_{split}" / name).read_bytes()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._zip is not None:
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
