import random


def select_veri(vehicle_ids, camera_ids, split, part, n, seed):
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not (len(vehicle_ids) == len(camera_ids) == len(split)):
        raise ValueError("vehicle_ids, camera_ids and split must have the same length")
    rng = random.Random(seed)

    by_vehicle = {}
    for index, (vehicle, part_name) in enumerate(zip(vehicle_ids, split)):
        if part_name == part:
            by_vehicle.setdefault(vehicle, []).append(index)

    by_camera = {}
    for vehicle in sorted(by_vehicle):
        index = rng.choice(by_vehicle[vehicle])
        by_camera.setdefault(camera_ids[index], []).append(index)
    for camera in by_camera:
        rng.shuffle(by_camera[camera])

    picked = []
    cameras = sorted(by_camera)
    while len(picked) < n and any(by_camera[c] for c in cameras):
        for camera in cameras:
            if len(picked) < n and by_camera[camera]:
                picked.append(int(by_camera[camera].pop()))
    return sorted(picked)


def select_ccpd(test_files_by_subset, fixed, per_subset, seed):
    if per_subset < 0:
        raise ValueError(f"per_subset must be >= 0, got {per_subset}")
    known = {f for files in test_files_by_subset.values() for f in files}
    reasons = {}
    for reason, files in fixed.items():
        for name in files:
            if name not in known:
                raise ValueError(f"fixed file is not among the test files: {name}")
            found = reasons.setdefault(name, [])
            if reason not in found:
                found.append(reason)
    result = [(name, tuple(reasons[name])) for name in sorted(reasons)]

    rng = random.Random(seed)
    randoms = []
    for subset in sorted(test_files_by_subset):
        free = sorted(f for f in test_files_by_subset[subset] if f not in reasons)
        take = min(per_subset, len(free))
        randoms.extend(rng.sample(free, take))
    result.extend((name, ("random",)) for name in sorted(randoms))
    return result
