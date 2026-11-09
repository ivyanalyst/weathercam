import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from weathercam.core.boxes import from_yolo
from weathercam.core.types import Box
from weathercam.data.yolo_export import yolo_label_text

WIDTH, HEIGHT = 720, 1160
PARTS = ("train", "val")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def place(job):
    source, target, mode = job
    if mode == "symlink":
        os.symlink(source, target)
    else:
        shutil.copyfile(source, target)


def main():
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    selected = []
    for part in PARTS:
        items = [item for item in manifest["items"] if item["part"] == part]
        selected += [(part, item) for item in (items[: args.limit] if args.limit else items)]
    jobs = []
    for part in PARTS:
        (out / "images" / part).mkdir(parents=True)
        (out / "labels" / part).mkdir(parents=True)
    for part, item in selected:
        stem = Path(item["file"]).stem
        label = yolo_label_text([(0, Box(*item["box"]))], WIDTH, HEIGHT)
        (out / "labels" / part / f"{stem}.txt").write_text(label)
        jobs.append((str((src / item["file"]).resolve()), out / "images" / part / f"{stem}.jpg", args.mode))
    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(place, jobs))
    (out / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: plate\n")
    worst = 0.0
    for part, item in selected:
        stem = Path(item["file"]).stem
        cx, cy, w, h = (float(v) for v in (out / "labels" / part / f"{stem}.txt").read_text().split()[1:])
        back = from_yolo(cx, cy, w, h, WIDTH, HEIGHT)
        worst = max(worst, max(abs(a - b) for a, b in zip(back.as_tuple(), item["box"])))
    print("images:", {part: sum(1 for p, _ in selected if p == part) for part in PARTS}, "| mode:", args.mode)
    print("worst label round-trip error (px): %.5f" % worst)
    print("wrote", out)


if __name__ == "__main__":
    main()
