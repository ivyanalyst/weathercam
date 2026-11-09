from typing import NamedTuple

PROVINCES = tuple("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学") + ("O",)
_LETTERS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALPHABETS = _LETTERS + ("O",)
ADS = _LETTERS + tuple("0123456789") + ("O",)


class PlateName(NamedTuple):
    vertices: tuple
    box: tuple
    field_box: tuple
    plate: str


def decode_plate(indices):
    if len(indices) != 7:
        raise ValueError(f"plate needs 7 indices, got {len(indices)}")
    tables = (PROVINCES, ALPHABETS) + (ADS,) * 5
    for table, index in zip(tables, indices):
        if not 0 <= index < len(table):
            raise ValueError(f"plate index out of range: {index}")
    return "".join(table[index] for table, index in zip(tables, indices))


def hull_box(vertices):
    if len(vertices) != 4:
        raise ValueError(f"expected 4 vertices, got {len(vertices)}")
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _points(field):
    points = []
    for part in field.split("_"):
        pair = part.split("&")
        if len(pair) != 2:
            raise ValueError(f"bad point: {part!r}")
        points.append((int(pair[0]), int(pair[1])))
    return points


def parse_name(path):
    name = path.split("/")[-1]
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    fields = stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"expected 7 fields, got {len(fields)}: {name!r}")
    corners = _points(fields[2])
    if len(corners) != 2:
        raise ValueError(f"expected 2 box corners, got {len(corners)}: {name!r}")
    vertices = _points(fields[3])
    if len(vertices) != 4:
        raise ValueError(f"expected 4 vertices, got {len(vertices)}: {name!r}")
    plate = decode_plate([int(token) for token in fields[4].split("_")])
    return PlateName(tuple(vertices), hull_box(vertices), (*corners[0], *corners[1]), plate)


def subset_of(path):
    for part in path.split("/")[:-1]:
        if part.startswith("ccpd_"):
            return part
    raise ValueError(f"no ccpd_ subset folder in path: {path!r}")
