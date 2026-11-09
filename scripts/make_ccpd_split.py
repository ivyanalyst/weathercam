import argparse
import collections
import hashlib
import json
from pathlib import Path

from weathercam.data.ccpd import parse_name, subset_of
from weathercam.data.ccpd_split import PARTS, ccpd_split, duplicates

RULES = [
    "names without 7 annotation fields are dropped (ccpd_np)",
    "a file name present in more than one subset is dropped everywhere",
    "base keeps one image per plate; the pool is sampled from those",
    "train/val: group split by plate, stratified by province character",
    "a non-base image is dropped if its plate occurs in any base image",
    "test keeps one image per plate across all subsets, then a per-subset cap",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--listing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool", type=int, default=20000)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-cap", type=int, default=2000)
    return p.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(paths):
    provinces = collections.Counter(parse_name(p).plate[0] for p in paths)
    top, count = provinces.most_common(1)[0] if paths else ("", 0)
    return {
        "images": len(paths),
        "by_subset": dict(sorted(collections.Counter(subset_of(p) for p in paths).items())),
        "top_province": top,
        "top_province_share": round(count / len(paths), 4) if paths else 0.0,
    }


def main():
    args = parse_args()
    lines = [x for x in Path(args.listing).read_text(encoding="utf-8").split("\n") if x.lower().endswith(".jpg")]
    result = ccpd_split(lines, args.seed, args.pool, args.val_fraction, args.test_cap)
    check = duplicates(result)
    if check != {"names": 0, "plates": 0}:
        raise SystemExit(f"duplicate check failed: {check}")
    parts = {name: {**describe(result[name]), "paths": result[name]} for name in PARTS}
    payload = {
        "protocol": "CCPD2019 overlap-free plate-disjoint split v1",
        "seed": args.seed,
        "pool": args.pool,
        "val_fraction": args.val_fraction,
        "test_cap": args.test_cap,
        "listing_sha256": sha256(args.listing),
        "listing_jpg": len(lines),
        "rules": RULES,
        "removed": result["removed"],
        "duplicates": check,
        "parts": parts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("listing jpg:", len(lines), "removed:", result["removed"])
    for name in PARTS:
        info = {k: v for k, v in parts[name].items() if k != "paths"}
        print(name, info)
    print("file sha256:", sha256(out))


if __name__ == "__main__":
    main()
