"""Human review verdicts: record schema, per-source element rules, append-only log."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ELEMENTS = {
    "veri": ("vehicle_visible", "colour", "type", "brand", "view", "plate_box", "plate_text"),
    "ccpd": ("detector_box", "label_box", "plate_text"),
}
SPLITS = ("train", "val", "test")
VERDICTS = ("correct", "wrong", "unsure", "not_determinable", "provided")
REASONS = (
    "vehicle_too_small", "cut_off", "occluded", "blurred", "too_dark", "overexposed",
    "colour_ambiguous", "brand_not_identifiable", "brand_not_in_list", "view_unclear",
    "plate_not_visible", "other",
)
BOX_ELEMENTS = ("plate_box", "detector_box", "label_box")


def elements_for(source):
    if source not in ELEMENTS:
        raise ValueError(f"unknown source: {source!r}")
    return ELEMENTS[source]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_is_number(v) for v in value):
        raise ValueError(f"a box must be four finite numbers, got {value!r}")
    x1, y1, x2, y2 = value
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"a box needs x1 < x2 and y1 < y2, got {list(value)!r}")


def _check_plate_text(value):
    if not isinstance(value, str) or value == "" or any(c.isspace() for c in value):
        raise ValueError(f"plate text must be non-empty with no whitespace, got {value!r}")


@dataclass(frozen=True)
class Record:
    source: str
    image: str
    split: str
    element: str
    model_value: object
    verdict: str
    timestamp: str
    corrected: object = None
    reason: object = None
    note: object = None

    def __post_init__(self):
        if self.element not in elements_for(self.source):
            raise ValueError(f"element {self.element!r} does not apply to source {self.source!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split!r}")
        if not isinstance(self.image, str) or self.image == "":
            raise ValueError("image must be a non-empty string")
        if not isinstance(self.timestamp, str) or self.timestamp == "":
            raise ValueError("timestamp must be a non-empty string")
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict: {self.verdict!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError("note must be a string or None")

        if self.model_value is None and self.verdict in ("correct", "wrong"):
            raise ValueError("there is no model value, so the verdict cannot be correct or wrong")
        if self.verdict == "provided" and self.model_value is not None:
            raise ValueError("provided is only for elements with no model value")
        if self.verdict == "provided" and self.corrected is None:
            raise ValueError("provided needs your value in corrected")
        if self.verdict == "wrong" and self.corrected is None and self.element != "vehicle_visible":
            raise ValueError("wrong needs a corrected value")
        if self.corrected is not None and self.verdict not in ("wrong", "provided"):
            raise ValueError("corrected is only allowed with wrong or provided")

        if self.verdict == "not_determinable":
            if self.reason not in REASONS:
                raise ValueError(f"not_determinable needs a reason from the list, got {self.reason!r}")
        elif self.reason is not None:
            raise ValueError("reason is only allowed with not_determinable")

        if self.corrected is not None:
            if self.element in BOX_ELEMENTS:
                _check_box(self.corrected)
                object.__setattr__(self, "corrected", list(self.corrected))
            elif self.element == "plate_text":
                _check_plate_text(self.corrected)
            elif not isinstance(self.corrected, str) or self.corrected == "":
                raise ValueError("corrected must be a non-empty string")


def record_to_line(record):
    return json.dumps(asdict(record), ensure_ascii=False)


def record_from_line(line):
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("a log line must be a JSON object")
    try:
        return Record(**data)
    except TypeError as error:
        raise ValueError(str(error)) from error


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(record_to_line(record) + "\n")


def read_log(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip() == "":
                continue
            try:
                records.append(record_from_line(line))
            except ValueError as error:
                raise ValueError(f"{path}: line {number}: {error}") from error
    return records


def latest(records):
    """Latest record per (source, image, element); later in the list wins."""
    result = {}
    for record in records:
        result[(record.source, record.image, record.element)] = record
    return result


def count_verdicts(records, source, element):
    counts = {verdict: 0 for verdict in VERDICTS}
    for (src, _image, elem), record in latest(records).items():
        if src == source and elem == element:
            counts[record.verdict] += 1
    return counts
