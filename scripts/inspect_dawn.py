import argparse
import collections
import os
import statistics
import xml.etree.ElementTree as ET

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
FLAG_KEYS = (
    "xmin==0",
    "ymin==0",
    "xmax==width",
    "ymax==height",
    "xmax>width",
    "ymax>height",
    "degenerate",
    "non_integer",
)


def jpeg_size(path):
    with open(path, "rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = handle.read(1)
            if not byte:
                return None
            if byte != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                return None
            code = marker[0]
            if code == 0xD9:
                return None
            if code == 0x00 or code == 0x01 or 0xD0 <= code <= 0xD8:
                continue
            raw = handle.read(2)
            if len(raw) < 2:
                return None
            length = int.from_bytes(raw, "big")
            if length < 2:
                return None
            if code in SOF_MARKERS:
                data = handle.read(5)
                if len(data) < 5:
                    return None
                return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
            handle.seek(length - 2, 1)


def scan(root):
    records = []
    for directory, _, files in os.walk(root):
        relative_directory = os.path.relpath(directory, root)
        group = relative_directory.split(os.sep)[0]
        for name in files:
            stem, extension = os.path.splitext(name)
            records.append(
                (group, relative_directory, stem, extension.lower(), os.path.join(directory, name))
            )
    records.sort()
    return records


def child_number(element, tag):
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    try:
        return float(child.text)
    except ValueError:
        return None


def parse_voc(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    objects = []
    object_tags = []
    elements = root.findall("object")
    for element in elements:
        object_tags.append(tuple(sorted({child.tag for child in element})))
        box = element.find("bndbox")
        if box is None:
            continue
        coords = [child_number(box, tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
        if None in coords:
            continue
        name = element.findtext("name")
        objects.append((name.strip() if name else "", *coords))
    return {
        "root": root.tag,
        "children": tuple(sorted({child.tag for child in root})),
        "object_tags": object_tags,
        "object_count": len(elements),
        "width": child_number(size, "width") if size is not None else None,
        "height": child_number(size, "height") if size is not None else None,
        "depth": child_number(size, "depth") if size is not None else None,
        "objects": objects,
    }


def class_key(value):
    return int(value) if value.is_integer() else value


def parse_yolo(path):
    boxes = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                bad += 1
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                bad += 1
                continue
            boxes.append((values[0], *values[1:]))
    return boxes, bad


def yolo_to_xyxy(box, width, height):
    cx, cy, w, h = box[1:]
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def box_iou(a, b):
    inter_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_x * inter_y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def spread(values):
    if not values:
        return "n/a"
    return "median=%.3f min=%.3f max=%.3f" % (statistics.median(values), min(values), max(values))


def report_layout(records):
    print("== layout ==")
    print("extensions:", dict(sorted(collections.Counter(record[3] for record in records).items())))
    per_directory = collections.defaultdict(collections.Counter)
    for _, directory, _, extension, _ in records:
        per_directory[directory][extension] += 1
    for directory in sorted(per_directory):
        print("%s: %s" % (directory, dict(sorted(per_directory[directory].items()))))


def report_groups(records):
    print("== groups ==")
    for group in sorted({record[0] for record in records}):
        members = [record for record in records if record[0] == group]
        image_files = [record for record in members if record[3] in IMAGE_EXTENSIONS]
        images = {record[2] for record in image_files}
        xml_stems = {record[2] for record in members if record[3] == ".xml"}
        txt_stems = {record[2] for record in members if record[3] == ".txt"}
        print(
            "%s: image_files=%d unique_images=%d xml=%d txt=%d"
            % (group, len(image_files), len(images), len(xml_stems), len(txt_stems))
        )
        print(
            "  image_without_xml=%d xml_without_image=%d image_without_txt=%d txt_without_image=%d"
            % (
                len(images - xml_stems),
                len(xml_stems - images),
                len(images - txt_stems),
                len(txt_stems - images),
            )
        )
        print("  txt_without_image first 5:", sorted(txt_stems - images)[:5])


def report_voc(records):
    print("== voc ==")
    results = []
    failed = []
    for group, _, stem, extension, path in records:
        if extension != ".xml":
            continue
        try:
            results.append((group, stem, parse_voc(path)))
        except (ET.ParseError, OSError):
            failed.append(path)
    by_key = {(group, stem): data for group, stem, data in results}
    print(
        "xml parsed=%d failed=%d duplicate_stems=%d"
        % (len(results), len(failed), len(results) - len(by_key))
    )
    print("failed first 3:", failed[:3])
    print("root tags:", dict(collections.Counter(data["root"] for _, _, data in results)))
    print("root children:", dict(collections.Counter(data["children"] for _, _, data in results)))
    print(
        "object children:",
        dict(collections.Counter(tags for _, _, data in results for tags in data["object_tags"])),
    )
    print("size depth:", dict(collections.Counter(data["depth"] for _, _, data in results)))
    names = collections.defaultdict(collections.Counter)
    flags = collections.Counter()
    xs, ys, widths, heights = [], [], [], []
    boxes = empty = skipped = 0
    for group, _, data in results:
        skipped += data["object_count"] - len(data["objects"])
        if not data["objects"]:
            empty += 1
        width, height = data["width"], data["height"]
        for name, x1, y1, x2, y2 in data["objects"]:
            boxes += 1
            names[group][name] += 1
            xs.append(x1)
            ys.append(y1)
            widths.append(x2 - x1)
            heights.append(y2 - y1)
            flags["xmin==0"] += x1 == 0
            flags["ymin==0"] += y1 == 0
            flags["xmax==width"] += width is not None and x2 == width
            flags["ymax==height"] += height is not None and y2 == height
            flags["xmax>width"] += width is not None and x2 > width
            flags["ymax>height"] += height is not None and y2 > height
            flags["degenerate"] += x2 <= x1 or y2 <= y1
            flags["non_integer"] += any(value != int(value) for value in (x1, y1, x2, y2))
    for group in sorted(names):
        print("%s classes: %s" % (group, dict(sorted(names[group].items()))))
    print("boxes=%d images_without_objects=%d objects_skipped=%d" % (boxes, empty, skipped))
    print(
        "min xmin=%s min ymin=%s min width=%s min height=%s"
        % (min(xs, default=None), min(ys, default=None), min(widths, default=None), min(heights, default=None))
    )
    print(" ".join("%s=%d" % (key, flags[key]) for key in FLAG_KEYS))
    return by_key


def report_yolo(records):
    print("== yolo ==")
    image_stems = collections.defaultdict(set)
    for group, _, stem, extension, _ in records:
        if extension in IMAGE_EXTENSIONS:
            image_stems[group].add(stem)
    labels = {}
    files = bad_lines = 0
    for group, _, stem, extension, path in records:
        if extension == ".txt" and stem in image_stems[group]:
            boxes, bad = parse_yolo(path)
            labels[(group, stem)] = boxes
            bad_lines += bad
            files += 1
    classes = collections.Counter()
    boxes_total = empty = out_of_range = 0
    for boxes in labels.values():
        if not boxes:
            empty += 1
        for box in boxes:
            boxes_total += 1
            classes[class_key(box[0])] += 1
            if any(not 0.0 <= value <= 1.0 for value in box[1:]):
                out_of_range += 1
    print(
        "label files=%d unique=%d empty=%d bad_lines=%d"
        % (files, len(labels), empty, bad_lines)
    )
    print("boxes=%d coords_outside_0_1=%d" % (boxes_total, out_of_range))
    print("class ids:", dict(sorted(classes.items())))
    return labels


def report_cross(voc, labels):
    print("== voc vs yolo ==")
    shared = sorted(set(voc) & set(labels))
    equal = matched = unmatched = no_size = 0
    offsets = [[], [], [], []]
    mapping = collections.Counter()
    for key in shared:
        data = voc[key]
        yolo = labels[key]
        if len(yolo) == len(data["objects"]):
            equal += 1
        if data["width"] is None or data["height"] is None:
            no_size += 1
            continue
        converted = [
            (class_key(box[0]), yolo_to_xyxy(box, data["width"], data["height"])) for box in yolo
        ]
        for name, x1, y1, x2, y2 in data["objects"]:
            best = None
            best_iou = 0.0
            for class_id, box in converted:
                value = box_iou((x1, y1, x2, y2), box)
                if value > best_iou:
                    best = (class_id, box)
                    best_iou = value
            if best is None or best_iou <= 0.5:
                unmatched += 1
                continue
            matched += 1
            mapping[(name, best[0])] += 1
            for slot, (voc_value, yolo_value) in enumerate(zip((x1, y1, x2, y2), best[1])):
                offsets[slot].append(voc_value - yolo_value)
    print(
        "stems in both=%d equal_object_counts=%d without_size=%d"
        % (len(shared), equal, no_size)
    )
    print("voc boxes matched(IoU>0.5)=%d unmatched=%d" % (matched, unmatched))
    for label, values in zip(("xmin", "ymin", "xmax", "ymax"), offsets):
        print("voc-yolo offset %s: %s" % (label, spread(values)))
    print("voc name -> yolo id:", dict(sorted(mapping.items())))


def report_sizes(records, voc):
    print("== image sizes ==")
    sizes = collections.Counter()
    first_path = {}
    files = unreadable = 0
    for group, _, stem, extension, path in records:
        if extension not in IMAGE_EXTENSIONS:
            continue
        files += 1
        first_path.setdefault((group, stem), path)
        size = jpeg_size(path) if extension in (".jpg", ".jpeg") else None
        if size is None:
            unreadable += 1
        else:
            sizes[size] += 1
    print("image files=%d size_read=%d unreadable=%d" % (files, sum(sizes.values()), unreadable))
    print("distinct sizes=%d" % len(sizes))
    print("most common (width, height):", sizes.most_common(5))
    match = mismatch = swapped = unchecked = 0
    for key, data in voc.items():
        path = first_path.get(key)
        if path is None or data["width"] is None or data["height"] is None:
            unchecked += 1
            continue
        size = jpeg_size(path)
        if size is None:
            unchecked += 1
            continue
        declared = (int(data["width"]), int(data["height"]))
        if declared == size:
            match += 1
        elif declared == (size[1], size[0]):
            swapped += 1
        else:
            mismatch += 1
    print(
        "voc size vs image header: match=%d mismatch=%d swapped=%d unchecked=%d"
        % (match, mismatch, swapped, unchecked)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    arguments = parser.parse_args()
    records = scan(arguments.root)
    if not records:
        print("no files found under", arguments.root)
        return
    report_layout(records)
    report_groups(records)
    voc = report_voc(records)
    labels = report_yolo(records)
    report_cross(voc, labels)
    report_sizes(records, voc)


if __name__ == "__main__":
    main()
