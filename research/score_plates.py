import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from weathercam.attributes.weather import wilson_interval
from weathercam.core.text_metrics import corpus_cer, exact_match_rate, position_accuracy
from weathercam.plates.ocr import decode_batch

EXCLUDED = (
    "images/022474.jpg", "images/022192.jpg", "images/024070.jpg",
    "images/022458.jpg", "images/022983.jpg", "images/023008.jpg", "images/023262.jpg",
    "images/023308.jpg", "images/023530.jpg", "images/023847.jpg", "images/024844.jpg",
    "images/026860.jpg", "images/027137.jpg",
)
RARE_MAX = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_logits(path, files):
    data = np.load(path)
    if [str(f) for f in data["files"]] != files:
        sys.exit(f"file order differs from the manifest: {path}")
    return data


def row(name, refs, hyps):
    n = len(refs)
    k = round(exact_match_rate(refs, hyps) * n)
    low, high = wilson_interval(k, n)
    pos = " ".join("%.3f" % v for v in position_accuracy(refs, hyps))
    return "%-15s n %5d  exact %5d/%5d = %.4f (wilson95 %.4f-%.4f)  CER %.4f  per-position %s" % (
        name, n, k, n, k / n, low, high, corpus_cer(refs, hyps), pos)


def first_char_lines(refs, hyps, train_counts):
    cats = collections.OrderedDict([("wan", []), ("non-wan common (>=11 train)", []), ("rare (<=10 train)", [])])
    for ref, hyp in zip(refs, hyps):
        c = ref[0]
        key = "wan" if c == "皖" else ("rare (<=10 train)" if train_counts[c] <= RARE_MAX else "non-wan common (>=11 train)")
        cats[key].append(hyp[:1] == c)
    lines = ["  first character:"]
    for key, hits in cats.items():
        if hits:
            lines.append("    %-28s n %5d  accuracy %.4f" % (key, len(hits), sum(hits) / len(hits)))
    lines.append("    always-wan baseline accuracy %.4f" % (len(cats["wan"]) / len(refs)))
    return lines


def section(title, refs, hyps, subsets, train_counts):
    lines = [title]
    for name in sorted(set(subsets)):
        ids = [k for k, s in enumerate(subsets) if s == name]
        lines.append(row(name, [refs[k] for k in ids], [hyps[k] for k in ids]))
    lines.append(row("ALL (pooled)", refs, hyps))
    lines += first_char_lines(refs, hyps, train_counts)
    return lines


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    run_dir = Path(args.run_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for name, expected in run["hashes"].items():
        if file_sha256(run_dir / name) != expected:
            sys.exit(f"hash differs from run.json: {name}")
    if run["split_sha256"] != manifest["split_sha256"]:
        sys.exit("split_sha256 differs between run.json and manifest")
    parts = {name: [i for i in manifest["items"] if i["part"] == name] for name in ("train", "val", "test")}
    train_counts = collections.Counter(i["plate"][0] for i in parts["train"])
    test = parts["test"]
    files = [i["file"] for i in test]
    refs = [i["plate"] for i in test]
    subsets = [i["subset"] for i in test]
    if len(set(EXCLUDED) & set(files)) != len(EXCLUDED):
        sys.exit("not all excluded images are in the test part")

    true = load_logits(run_dir / "logits_test_true.npz", files)
    det = load_logits(run_dir / "logits_test_detected.npz", files)
    true_text = [r.text for r in decode_batch(true["logits"])]
    det_text = ["" if m else r.text for r, m in zip(decode_batch(det["logits"]), det["missing"])]
    keep = [k for k, f in enumerate(files) if f not in set(EXCLUDED)]

    val_files = [i["file"] for i in parts["val"]]
    val = load_logits(run_dir / "logits_val.npz", val_files)
    val_refs = [i["plate"] for i in parts["val"]]
    val_text = [r.text for r in decode_batch(val["logits"])]

    excluded_hash = hashlib.sha256("\n".join(sorted(EXCLUDED)).encode()).hexdigest()
    lines = [
        "run %s | best epoch %d | split_sha256 %s" % (run_dir.name, run["best_epoch"], manifest["split_sha256"]),
        "exact match = all 7 characters right; CER over the 7-character plate string; missing detection = empty reading",
        "pooled rows weight subsets by their capped sizes, not by natural frequency; one seed, one run",
        "excluded (true-box section only): %d images, sha256 of sorted names %s" % (len(EXCLUDED), excluded_hash),
        "",
    ]
    lines += section("VAL (same distribution as train, ccpd_base)", val_refs, val_text, ["ccpd_base"] * len(val_refs), train_counts)
    lines.append("")
    lines += section("TEST, TRUE BOXES (%d images, %d excluded)" % (len(keep), len(EXCLUDED)),
                     [refs[k] for k in keep], [true_text[k] for k in keep], [subsets[k] for k in keep], train_counts)
    lines.append("")
    lines += section("TEST, DETECTED BOXES (%d images, %d without detection counted wrong)" % (len(files), int(det["missing"].sum())),
                     refs, det_text, subsets, train_counts)
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
