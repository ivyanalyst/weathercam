"""Prepare the demo gallery: resize, rename, draft labels.

Reads images from a folder and writes demo_NN.jpg plus a draft
labels.csv. Source names are not written to the output folder; a
private map from demo name to source name goes to --map-out (keep it
outside the repository). Labels are guesses from file names only.
With --canvas WxH every image is shrunk to fit, then pasted centred
on a neutral grey canvas of that size (no crop, no stretch).
"""
import argparse
import csv
import os
import sys

from PIL import Image

EXTS = (".jpg", ".jpeg", ".png")
COLOURS = {
    "white", "black", "blue", "grey", "darkgrey", "silver", "red",
    "darkred", "green", "orange", "yellow", "pink", "champagne",
}
TYPES = ("bus", "truck", "hatchback", "suv", "sedan")
COUPE_WORDS = ("gt", "nsx", "spectre")
SCENES = (
    "alpine_pass", "mountain_road", "avenue", "overpass", "treeroad",
    "city_street",
)
GREY = (128, 128, 128)


def draft_labels(name):
    low = name.lower()
    first = name.split("_")[0]
    colour = first if first in COLOURS else ""
    tokens = set(low.replace("-", "_").split("_"))
    kind = ""
    for word in TYPES:
        if word in tokens:
            kind = word
            break
    if not kind and any(w in tokens for w in COUPE_WORDS):
        kind = "coupe"
    view = ""
    if "rear_three_quarter" in low:
        view = "rear34"
    elif "front34" in low:
        view = "front34"
    scene = ""
    for key in SCENES:
        if key in low:
            scene = key
            break
    return colour, kind, view, scene


def parse_canvas(text):
    try:
        w, h = (int(p) for p in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("use WxH, for example 640x480")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("canvas sides must be positive")
    return w, h


def shrink(src, dst, size, canvas=None):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if canvas is None:
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
        else:
            im.thumbnail(canvas, Image.Resampling.LANCZOS)
            base = Image.new("RGB", canvas, GREY)
            base.paste(im, ((canvas[0] - im.width) // 2,
                            (canvas[1] - im.height) // 2))
            im = base
        im.save(dst, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--map-out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--canvas", type=parse_canvas, default=None)
    args = ap.parse_args()
    labels_path = os.path.join(args.out, "labels.csv")
    if os.path.exists(labels_path):
        sys.exit("refusing to overwrite " + labels_path)
    names = sorted(
        (n for n in os.listdir(args.src) if n.lower().endswith(EXTS)),
        key=str.lower,
    )
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows, pairs = [], []
    for i, name in enumerate(names, start=1):
        out_name = "demo_%02d.jpg" % i
        shrink(os.path.join(args.src, name), os.path.join(img_dir, out_name),
               args.size, args.canvas)
        rows.append((out_name,) + draft_labels(name))
        pairs.append((out_name, name))
    with open(labels_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("file", "colour", "type", "view", "scene"))
        w.writerows(rows)
    with open(args.map_out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(pairs)
    blanks = [sum(1 for r in rows if not r[k]) for k in (1, 2, 3, 4)]
    print(len(rows), "images; blank colour/type/view/scene:", blanks)


if __name__ == "__main__":
    main()
