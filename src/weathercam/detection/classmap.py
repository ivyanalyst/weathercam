from collections.abc import Mapping

CLASS_NAMES = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck")


def dawn_class_map() -> dict[str, int]:
    return {name: index for index, name in enumerate(CLASS_NAMES)}


def model_label_map(names) -> dict[int, int]:
    items = list(names.items()) if isinstance(names, Mapping) else list(enumerate(names))
    if not items:
        raise ValueError("model has no class names")
    label_of = dawn_class_map()
    result: dict[int, int] = {}
    seen: set[str] = set()
    for key, name in items:
        if isinstance(key, bool) or not isinstance(key, int) or key < 0:
            raise ValueError(f"class index must be a non-negative int, got {key!r}")
        if not isinstance(name, str):
            raise ValueError(f"class name must be a string, got {name!r}")
        if name not in label_of:
            continue
        if name in seen:
            raise ValueError(f"duplicate class name {name!r}")
        seen.add(name)
        result[key] = label_of[name]
    missing = [name for name in CLASS_NAMES if name not in seen]
    if missing:
        raise ValueError(f"model lacks classes: {missing}")
    return result
