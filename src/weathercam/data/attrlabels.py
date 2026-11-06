from dataclasses import dataclass


@dataclass(frozen=True)
class AttrColumns:
    image_names: tuple
    vehicle_ids: tuple
    camera_ids: tuple
    color_ids: tuple
    type_ids: tuple


def columns_of(labels):
    items = labels.items
    return AttrColumns(
        tuple(i.image_name for i in items),
        tuple(i.vehicle_id for i in items),
        tuple(i.camera_id for i in items),
        tuple(i.color_id for i in items),
        tuple(i.type_id for i in items),
    )


def vehicle_classes(labels):
    result = {}
    for item in labels.items:
        pair = (item.color_id, item.type_id)
        if item.vehicle_id in result and result[item.vehicle_id] != pair:
            raise ValueError(f"vehicle {item.vehicle_id} has two different classes")
        result[item.vehicle_id] = pair
    return result


def class_names(names):
    return [names[i] for i in sorted(names)]


def encode(ids, names):
    index = {key: k for k, key in enumerate(sorted(names))}
    result = []
    for value in ids:
        if value not in index:
            raise ValueError(f"unknown id: {value}")
        result.append(index[value])
    return result
