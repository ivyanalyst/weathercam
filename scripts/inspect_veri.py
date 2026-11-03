import argparse
import collections
import re
import zipfile

SPLITS = ("train", "test", "query")
ITEM_PATTERN = re.compile(r"<Item\s[^>]*/>")
ATTRIBUTE_PATTERN = re.compile(r'(\w+)="([^"]*)"')
DECLARED_PATTERN = re.compile(r'<Items\s+number="(\d+)"')


def read_text(archive, name, encoding):
    return archive.read(name).decode(encoding, errors="replace")


def read_names(archive, split):
    return read_text(archive, "VeRi/name_%s.txt" % split, "utf-8-sig").split()


def read_label_names(archive, name):
    labels = {}
    for line in read_text(archive, name, "utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) == 2:
            labels[parts[0]] = parts[1]
    return labels


def read_labels(archive, split):
    raw = read_text(archive, "VeRi/%s_label.xml" % split, "gb2312")
    match = DECLARED_PATTERN.search(raw)
    declared = int(match.group(1)) if match else None
    items = [dict(ATTRIBUTE_PATTERN.findall(tag)) for tag in ITEM_PATTERN.findall(raw)]
    return declared, items


def folder_files(entries, folder):
    prefix = "VeRi/%s/" % folder
    return [
        entry[len(prefix):]
        for entry in entries
        if entry.startswith(prefix) and entry.lower().endswith(".jpg")
    ]


def parse_name(name):
    parts = name.split("_")
    return parts[0], parts[1]


def count_table(counts, names):
    return " ".join(
        "%s=%d" % (names[key], counts.get(key, 0)) for key in sorted(names, key=int)
    )


def report_folders(entries, lists):
    print("== folders ==")
    unlisted_all = []
    for split in SPLITS:
        files = folder_files(entries, "image_%s" % split)
        listed = lists[split]
        unlisted = sorted(set(files) - set(listed))
        missing = sorted(set(listed) - set(files))
        unlisted_all.extend(unlisted)
        print(
            "%s: folder=%d listed=%d unlisted=%d missing=%d"
            % (split, len(files), len(listed), len(unlisted), len(missing))
        )
    per_vehicle = collections.Counter(parse_name(name)[0] for name in unlisted_all)
    print("unlisted images per vehicleID:", dict(sorted(per_vehicle.items())))


def report_labels(split, declared, items, color_names, type_names):
    vehicles = collections.Counter(item["vehicleID"] for item in items)
    cameras = {item["cameraID"] for item in items}
    print(
        "%s: declared=%s parsed=%d vehicles=%d cameras=%d"
        % (split, declared, len(items), len(vehicles), len(cameras))
    )
    print(
        "  images per vehicle: min=%d max=%d"
        % (min(vehicles.values()), max(vehicles.values()))
    )
    for key, names, label in (
        ("colorID", color_names, "color"),
        ("typeID", type_names, "type"),
    ):
        image_counts = collections.Counter(item[key] for item in items)
        pairs = {(item["vehicleID"], item[key]) for item in items}
        vehicle_counts = collections.Counter(value for _, value in pairs)
        values = collections.defaultdict(set)
        for vehicle, value in pairs:
            values[vehicle].add(value)
        mixed = sorted(vehicle for vehicle, found in values.items() if len(found) > 1)
        print("  %s images: %s" % (label, count_table(image_counts, names)))
        print("  %s vehicles: %s" % (label, count_table(vehicle_counts, names)))
        print("  %s ids not in list: %s" % (label, sorted(set(image_counts) - set(names))))
        print("  vehicles with more than one %s: %d %s" % (label, len(mixed), mixed[:5]))


def report_splits(entries, lists, labels):
    print("== splits ==")
    train_items = labels["train"][1]
    test_items = labels["test"][1]
    train_ids = {item["vehicleID"] for item in train_items}
    test_ids = {item["vehicleID"] for item in test_items}
    train_cameras = {item["cameraID"] for item in train_items}
    test_cameras = {item["cameraID"] for item in test_items}
    query_cameras = {parse_name(name)[1] for name in lists["query"]}
    print("vehicleID overlap train/test:", len(train_ids & test_ids))
    print("cameras only in train:", sorted(train_cameras - test_cameras))
    print("cameras only in test:", sorted(test_cameras - train_cameras))
    print(
        "query cameras: %d | not in test cameras: %s"
        % (len(query_cameras), sorted(query_cameras - test_cameras))
    )
    last = collections.Counter(
        name.rsplit("_", 1)[1].split(".")[0]
        for split in SPLITS
        for name in folder_files(entries, "image_%s" % split)
    )
    print("name last field:", dict(sorted(last.items())))


def check_index(queries, tests, gt_lines, jk_lines, offset):
    parsed = [parse_name(name) for name in tests]
    per_vehicle = collections.Counter(vehicle for vehicle, _ in parsed)
    out_of_range = gt_wrong = jk_wrong = coverage_wrong = 0
    for query, gt_line, jk_line in zip(queries, gt_lines, jk_lines):
        vehicle, camera = parse_name(query)
        gt = [int(token) - offset for token in gt_line.split()]
        jk = [int(token) - offset for token in jk_line.split()]
        if any(not 0 <= index < len(tests) for index in gt + jk):
            out_of_range += 1
            continue
        if any(parsed[i][0] != vehicle or parsed[i][1] == camera for i in gt):
            gt_wrong += 1
        if any(parsed[i] != (vehicle, camera) for i in jk):
            jk_wrong += 1
        if set(gt) & set(jk) or len(set(gt) | set(jk)) != per_vehicle[vehicle]:
            coverage_wrong += 1
    return out_of_range, gt_wrong, jk_wrong, coverage_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    arguments = parser.parse_args()
    with zipfile.ZipFile(arguments.zip_path) as archive:
        entries = archive.namelist()
        lists = {split: read_names(archive, split) for split in SPLITS}
        color_names = read_label_names(archive, "VeRi/list_color.txt")
        type_names = read_label_names(archive, "VeRi/list_type.txt")
        labels = {split: read_labels(archive, split) for split in ("train", "test")}
        gt_lines = read_text(archive, "VeRi/gt_index.txt", "utf-8-sig").splitlines()
        jk_lines = read_text(archive, "VeRi/jk_index.txt", "utf-8-sig").splitlines()
    report_folders(entries, lists)
    print("== labels ==")
    for split in ("train", "test"):
        declared, items = labels[split]
        report_labels(split, declared, items, color_names, type_names)
    report_splits(entries, lists, labels)
    print("== index files ==")
    print(
        "gt lines: %d jk lines: %d queries: %d"
        % (len(gt_lines), len(jk_lines), len(lists["query"]))
    )
    for offset in (1, 0):
        result = check_index(lists["query"], lists["test"], gt_lines, jk_lines, offset)
        print(
            "offset %d: out_of_range=%d gt_wrong=%d jk_wrong=%d coverage_wrong=%d"
            % ((offset,) + result)
        )


if __name__ == "__main__":
    main()
