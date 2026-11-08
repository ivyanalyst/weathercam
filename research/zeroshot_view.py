import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from weathercam.attributes.zeroshot import (
    aggregate_tracks,
    class_vectors,
    image_probabilities,
    track_predictions,
)

CELL = 160
LABEL_W = 90


def load_thumb(zf, name):
    with zf.open("VeRi/image_train/" + name) as f:
        im = Image.open(f).convert("RGB")
    im.thumbnail((CELL, CELL))
    canvas = Image.new("RGB", (CELL, CELL), (128, 128, 128))
    canvas.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    return canvas


def pick_frames(names, k):
    names = sorted(names)
    if len(names) <= k:
        return names
    idx = np.unique(np.linspace(0, len(names) - 1, k).round().astype(int))
    return [names[i] for i in idx]


def make_sheet(zf, tids, shown, counts, frames):
    sheet = Image.new("RGB", (LABEL_W + frames * CELL, len(tids) * CELL), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (tid, names, n) in enumerate(zip(tids, shown, counts)):
        draw.text((6, r * CELL + 6), tid, fill=(0, 0, 0))
        draw.text((6, r * CELL + 22), "n=%d" % n, fill=(0, 0, 0))
        for c, name in enumerate(names):
            sheet.paste(load_thumb(zf, name), (LABEL_W + c * CELL, r * CELL))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="outputs/embeddings/veri_vitb32_openai_qg.npz")
    ap.add_argument("--text", default="outputs/view/view_text_vitb32_openai_qg.npz")
    ap.add_argument("--zip", default="/path/to/VeRi.zip")
    ap.add_argument("--out-dir", default="outputs/view")
    ap.add_argument("--tracks", type=int, default=100)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--per-sheet", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=100.0)
    args = ap.parse_args()

    d = np.load(args.emb)
    m = d["split"] == "train"
    names = [str(x) for x in d["image_names"][m]]
    keys = list(zip(map(str, d["vehicle_ids"][m]), map(str, d["camera_ids"][m])))
    t = np.load(args.text)
    classes = [str(c) for c in t["classes"]]
    probs = image_probabilities(d["embeddings"][m], class_vectors(t["text"]), args.scale)
    tkeys, means, counts = aggregate_tracks(probs, keys)
    labels, conf = track_predictions(means)

    eligible = np.flatnonzero(counts >= 2)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(eligible, size=min(args.tracks, len(eligible)), replace=False)
    members = {tkeys[i]: [] for i in pick}
    for n, k in zip(names, keys):
        if k in members:
            members[k].append(n)

    tids = ["t%03d" % (j + 1) for j in range(len(pick))]
    shown = [pick_frames(members[tkeys[i]], args.frames) for i in pick]
    pcounts = [int(counts[i]) for i in pick]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        for s in range(0, len(pick), args.per_sheet):
            sl = slice(s, s + args.per_sheet)
            sheet = make_sheet(zf, tids[sl], shown[sl], pcounts[sl], args.frames)
            sheet.save(out / ("sheet_%02d.png" % (s // args.per_sheet + 1)))

    with open(out / "sample_tracks.txt", "w") as f:
        for tid, i, n, sh in zip(tids, pick, pcounts, shown):
            v, c = tkeys[i]
            f.write("\t".join([tid, v, c, str(n), ";".join(sh)]) + "\n")
    np.savez(
        out / "sample_predictions.npz",
        tids=np.array(tids),
        pred=labels[pick],
        conf=conf[pick],
        mean_probs=means[pick],
        classes=np.array(classes),
    )

    print("train images", len(names), "tracks", len(tkeys), "eligible", len(eligible))
    print("sample", len(pick), "sheets", -(-len(pick) // args.per_sheet))
    print("row sum max deviation", float(np.abs(probs.sum(axis=1) - 1).max()))
    print("all-track predicted counts", dict(zip(classes, np.bincount(labels, minlength=len(classes)).tolist())))
    print("all-track mean confidence", round(float(conf.mean()), 4))


if __name__ == "__main__":
    main()
