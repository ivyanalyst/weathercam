import argparse
import json
import resource
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.data.neardup import group_by_distance, hash_distance, hash_file
from weathercam.data.split import group_split

THRESHOLDS = (0, 5, 10, 15, 20, 30, 40)
PARTS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)


def list_images(dawn_dir):
    return sorted(p.relative_to(dawn_dir).as_posix() for p in dawn_dir.rglob("*.jpg"))


def compute_hashes(dawn_dir, paths, size, guard):
    hashes = []
    for i, rel in enumerate(paths):
        hashes.append(hash_file(dawn_dir / rel, size))
        if (i + 1) % 100 == 0:
            print("hashed %d of %d, rss %.0f MB" % (i + 1, len(paths), guard.check()))
    guard.check()
    return hashes


def nearest_distances(hashes):
    best = [10**9] * len(hashes)
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hash_distance(hashes[i], hashes[j])
            if d < best[i]:
                best[i] = d
            if d < best[j]:
                best[j] = d
    return best


def summarize(groups, weathers):
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    mixed = sum(1 for m in members.values() if len({weathers[i] for i in m}) > 1)
    ordered = sorted(members.values(), key=lambda m: (-len(m), m[0]))
    return ordered, mixed


def report(paths, weathers, hashes):
    near = Counter(d // 5 for d in nearest_distances(hashes))
    print("nearest-neighbour distance histogram (bins of 5 bits)")
    for b in sorted(near):
        print("  %3d-%3d: %d" % (b * 5, b * 5 + 4, near[b]))
    for t in THRESHOLDS:
        ordered, mixed = summarize(group_by_distance(hashes, t), weathers)
        singles = sum(1 for m in ordered if len(m) == 1)
        print(
            "threshold %d: groups %d, largest %d, singletons %d, mixed-weather groups %d"
            % (t, len(ordered), len(ordered[0]), singles, mixed)
        )
        for m in ordered[:3]:
            print("    size %d: %s" % (len(m), ", ".join(paths[i] for i in m[:6])))


def build_split(paths, weathers, hashes, threshold, seed, size):
    groups = group_by_distance(hashes, threshold)
    classes = [weathers[g] for g in groups]
    ordered, mixed = summarize(groups, weathers)
    parts = group_split(groups, classes, FRACTIONS, seed)
    group_sets = [{groups[i] for i in part} for part in parts]
    for a, b in combinations(group_sets, 2):
        assert not (a & b)
    assert sorted(i for part in parts for i in part) == list(range(len(paths)))
    total = len(paths)
    summary = {}
    for name, part, members in zip(PARTS, parts, group_sets):
        summary[name] = {
            "images": len(part),
            "groups": len(members),
            "share": len(part) / total,
            "by_weather": dict(sorted(Counter(weathers[i] for i in part).items())),
            "paths": [paths[i] for i in part],
        }
    return {
        "protocol": "DAWN near-duplicate split v1",
        "seed": seed,
        "fractions": list(FRACTIONS),
        "stratum": "weather of the lowest-index member of each group",
        "hash": {
            "type": "dhash",
            "size": size,
            "bits": size * size,
            "threshold": threshold,
            "linkage": "single",
        },
        "images": total,
        "groups": len(ordered),
        "mixed_weather_groups": mixed,
        "parts": summary,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dawn-dir", default="data/DAWN")
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--hash-size", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out")
    parser.add_argument("--memory-limit-mb", type=float, default=3000)
    args = parser.parse_args()
    if not args.report_only and args.threshold is None:
        parser.error("--threshold is required unless --report-only is given")
    return args


def main():
    args = parse_args()
    config = load_config()
    seed = config.seed if args.seed is None else args.seed
    default_out = Path(config.paths.outputs_dir) / "dawn_split" / "split_v1.json"
    out = Path(args.out) if args.out else default_out
    if not args.report_only and out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    dawn_dir = Path(args.dawn_dir)
    paths = list_images(dawn_dir)
    if not paths:
        raise SystemExit("no .jpg files under %s" % dawn_dir)
    weathers = [Path(p).parts[0] for p in paths]
    guard = MemoryGuard(args.memory_limit_mb)
    start = time.perf_counter()
    hashes = compute_hashes(dawn_dir, paths, args.hash_size, guard)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("images %d, hash seconds %.1f, peak rss %.0f MB" % (len(paths), seconds, peak))
    if args.report_only:
        report(paths, weathers, hashes)
        return
    payload = build_split(paths, weathers, hashes, args.threshold, seed, args.hash_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("groups %d, mixed-weather groups %d" % (payload["groups"], payload["mixed_weather_groups"]))
    for name, part in payload["parts"].items():
        print(
            "%s: images %d (%.3f), groups %d, %s"
            % (name, part["images"], part["share"], part["groups"], part["by_weather"])
        )
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
