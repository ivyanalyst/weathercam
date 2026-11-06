import json
from pathlib import Path

PARTS = ("train", "val", "test")


def load_split_paths(path, part):
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    data = json.loads(Path(path).read_text())
    return list(data["parts"][part]["paths"])
