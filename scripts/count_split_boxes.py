import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from weathercam.core.memory import MemoryGuard
from weathercam.data.dawn import load_annotation

PARTS = ("train", "val", "test")
ORIENTATION_TAG = 0x0112


def xml_path(dawn_dir, rel):
    path = Path(rel)
    weather = path.parts[0]
    return dawn_dir / path.parent / f"{weather}_PASCAL_VOC" / f"{path.stem}.xml"


def orientation(path):
    with Image.open(path) as image:
        value = image.getexif().get(ORIENTATION_TAG)
    return "none" if value is None else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/dawn_split/split_v2.json")
    parser.add_argument("--dawn-dir", default="data/DAWN")
    parser.add_argument("--memory-limit-mb", type=int, default=3000)
    args = parser.parse_args()
    dawn_dir = Path(args.dawn_dir)
    split = json.loads(Path(args.split).read_text())
    guard = MemoryGuard(args.memory_limit_mb)
    boxes = {part: Counter() for part in PARTS}
    images = {part: Counter() for part in PARTS}
    tags = {part: Counter() for part in PARTS}
    tag2_weather = Counter()
    tag0 = []
    missing = []
    seen = 0
    for part in PARTS:
        for rel in split["parts"][part]["paths"]:
            seen += 1
            if seen % 100 == 0:
                guard.check()
            xml = xml_path(dawn_dir, rel)
            if not xml.is_file():
                missing.append(rel)
                continue
            names = [obj.name for obj in load_annotation(xml).objects]
            boxes[part].update(names)
            images[part].update(set(names))
            tag = orientation(dawn_dir / rel)
            tags[part][tag] += 1
            if tag == 2:
                tag2_weather[(part, Path(rel).parts[0])] += 1
            if tag == 0:
                tag0.append((rel, part))
    names = sorted(
        set().union(*(boxes[part].keys() for part in PARTS)),
        key=lambda name: -sum(boxes[part][name] for part in PARTS),
    )
    width = max([len(name) for name in names] + [5])
    print(f"{'class':<{width}}  {'train':>6} {'val':>6} {'test':>6} {'total':>6}  |  {'tr_img':>6} {'va_img':>6} {'te_img':>6}")
    for name in names:
        counts = [boxes[part][name] for part in PARTS]
        imgs = [images[part][name] for part in PARTS]
        print(f"{name:<{width}}  {counts[0]:>6} {counts[1]:>6} {counts[2]:>6} {sum(counts):>6}  |  {imgs[0]:>6} {imgs[1]:>6} {imgs[2]:>6}")
    totals = [sum(boxes[part].values()) for part in PARTS]
    print(f"{'ALL':<{width}}  {totals[0]:>6} {totals[1]:>6} {totals[2]:>6} {sum(totals):>6}")
    print()
    for part in PARTS:
        ordered = sorted(tags[part].items(), key=lambda item: str(item[0]))
        print(f"EXIF orientation {part}: " + ", ".join(f"{key}: {value}" for key, value in ordered))
    print("tag 2 by part and weather:", dict(sorted(tag2_weather.items())))
    print("tag 0 images:", tag0)
    print(f"images seen: {seen}, missing XML: {len(missing)}")
    for rel in missing:
        print("missing:", rel)
    print(f"final RSS MB: {guard.check():.1f}")


if __name__ == "__main__":
    main()
