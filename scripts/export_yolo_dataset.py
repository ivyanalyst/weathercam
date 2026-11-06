import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

from weathercam.core.memory import MemoryGuard
from weathercam.data.dawn import load_annotation
from weathercam.data.yolo_export import yolo_label_text
from weathercam.detection.classmap import CLASS_NAMES

PARTS = ("train", "val", "test")
ORIENTATION_TAG = 0x0112
EXPECTED = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]


def xml_path(dawn_dir, rel):
    path = Path(rel)
    weather = path.parts[0]
    return dawn_dir / path.parent / f"{weather}_PASCAL_VOC" / f"{path.stem}.xml"


def write_image(src, dst, flip):
    if not flip:
        shutil.copyfile(src, dst)
        return
    with Image.open(src) as image:
        ImageOps.mirror(image.convert("RGB")).save(dst, "JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/dawn_split/split_v2.json")
    parser.add_argument("--dawn-dir", default="data/DAWN")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--voc-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=3000)
    args = parser.parse_args()
    assert list(CLASS_NAMES) == EXPECTED, list(CLASS_NAMES)
    ids = {name: index for index, name in enumerate(EXPECTED)}
    out = Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    dawn_dir = Path(args.dawn_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    voc_dir = Path(args.voc_dir) if args.voc_dir else dawn_dir
    split = json.loads(Path(args.split).read_text())
    guard = MemoryGuard(args.memory_limit_mb)
    errors = []
    seen = 0
    for part in PARTS:
        (out / "images" / part).mkdir(parents=True)
        (out / "labels" / part).mkdir(parents=True)
        paths = split["parts"][part]["paths"]
        if args.limit:
            paths = paths[: args.limit]
        stems = set()
        flipped = 0
        empty = 0
        boxes = 0
        for rel in paths:
            seen += 1
            if seen % 100 == 0:
                guard.check()
            stem = Path(rel).stem
            if stem in stems:
                errors.append((rel, "duplicate stem"))
                continue
            stems.add(stem)
            src = image_dir / Path(rel).name if image_dir is not None else dawn_dir / rel
            annotation = load_annotation(xml_path(voc_dir, rel))
            with Image.open(src) as image:
                tag = image.getexif().get(ORIENTATION_TAG)
                size = image.size
            if size != (annotation.width, annotation.height):
                errors.append((rel, f"size {size} vs xml"))
                continue
            items = [(ids[obj.name], obj.box) for obj in annotation.objects]
            try:
                text = yolo_label_text(items, annotation.width, annotation.height)
            except ValueError as error:
                errors.append((rel, str(error)))
                continue
            if any(obj.box.width == 0 or obj.box.height == 0 for obj in annotation.objects):
                errors.append((rel, "zero-size box"))
            flip = tag == 2
            write_image(src, out / "images" / part / f"{stem}.jpg", flip)
            (out / "labels" / part / f"{stem}.txt").write_text(text)
            flipped += int(flip)
            empty += int(not items)
            boxes += len(items)
        print(f"{part}: images {len(stems)}, boxes {boxes}, flipped {flipped}, empty labels {empty}")
    lines = ["train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(EXPECTED)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"errors: {len(errors)}")
    for rel, reason in errors:
        print("error:", rel, reason)
    print(f"final RSS MB: {guard.check():.1f}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
