import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from weathercam.data.ccpd import parse_name, subset_of, to_latin_plate
from weathercam.data.splitfile import PARTS, load_split_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--listing", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_paths(listing, count):
    by_name = collections.defaultdict(list)
    for path in listing:
        by_name[path.split("/")[-1]].append(path)
    pairs = [sorted(v) for _, v in sorted(by_name.items()) if len(v) == 2]
    return pairs[:: max(1, len(pairs) // count)][:count]


def main():
    args = parse_args()
    items = [(part, path) for part in PARTS for path in load_split_paths(args.split, part)]
    lines = Path(args.listing).read_text(encoding="utf-8").split("\n")
    pairs = pair_paths([x for x in lines if x.lower().endswith(".jpg")], args.pairs)
    members = [path for _, path in items] + [path for pair in pairs for path in pair]
    out = Path(args.out)
    free = shutil.disk_usage(out.parent).free
    print("images:", {part: sum(1 for p, _ in items if p == part) for part in PARTS}, "| pair checks:", len(pairs))
    print("members: %d, estimated %.2f GB (about 37 KB each), free %.1f GB" % (len(members), len(members) * 37e3 / 1e9, free / 1e9))
    if args.dry_run:
        print("first members:", *members[:3], sep="\n  ")
        return
    if free < 5e9:
        raise SystemExit("less than 5 GB free")
    os.makedirs(out, exist_ok=False)
    (out / "images").mkdir()
    staging = out / "staging"
    staging.mkdir()
    (out / "members.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
    command = ["nice", "-n", "19", "tar", "-xJf", args.archive, "-C", str(staging), "--no-wildcards", "-T", str(out / "members.txt")]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    same = sum(1 for a, b in pairs if file_sha256(staging / a) == file_sha256(staging / b))
    print("same name, same bytes:", same, "of", len(pairs))
    sizes = collections.Counter()
    records = []
    for k, (part, path) in enumerate(items):
        source = staging / path
        digest = file_sha256(source)
        target = out / "images" / f"{k:06d}.jpg"
        os.replace(source, target)
        with Image.open(target) as image:
            sizes[f"{image.size[0]}x{image.size[1]}"] += 1
        info = parse_name(path)
        records.append(
            {
                "file": f"images/{k:06d}.jpg",
                "path": path,
                "part": part,
                "subset": subset_of(path),
                "plate": info.plate,
                "plate_latin": to_latin_plate(info.plate),
                "box": list(info.box),
                "vertices": [list(v) for v in info.vertices],
                "sha256": digest,
            }
        )
        if (k + 1) % 5000 == 0:
            print("flattened", k + 1, flush=True)
    manifest = {
        "split_sha256": file_sha256(args.split),
        "listing_sha256": file_sha256(args.listing),
        "pair_checks": {"pairs": len(pairs), "identical": same},
        "image_sizes": dict(sizes),
        "items": records,
    }
    with open(out / "manifest.json", "x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    shutil.rmtree(staging)
    print("done:", len(records), "images, sizes:", dict(sizes))
    print("manifest sha256:", file_sha256(out / "manifest.json"))


if __name__ == "__main__":
    main()
