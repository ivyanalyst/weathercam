import numpy as np

from weathercam.data.split import group_split


def veri_inner_split(data, seed):
    train_rows = np.flatnonzero(data["split"] == "train")
    vehicles = data["vehicle_ids"][train_rows]
    classes = list(zip(data["color_ids"][train_rows].tolist(), data["type_ids"][train_rows].tolist()))
    fit_part, val_part = group_split(vehicles.tolist(), classes, (0.8, 0.2), seed)
    return train_rows[np.asarray(fit_part)], train_rows[np.asarray(val_part)]
