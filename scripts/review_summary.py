import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from weathercam.attributes.weather import wilson_interval
from weathercam.review.verdicts import elements_for, latest, read_log

CAVEATS = (
    "One reviewer: the labels are one person's judgement.",
    "Colour, type, brand, view and vehicle_visible were answered blind; plate text was judged with the model reading shown.",
    "Reviews of test-split images are never used to tune the OCR, detector, heads or fuzzy limit; headline test numbers stay as scored.",
    "CCPD 'fixed' rows are chosen from known misses and label problems, so they say nothing about overall rates; 'random' is about 14 images per subset.",
    "A model brand 'unknown' can never agree: the reviewer cannot answer 'unknown'.",
    "Sources are never pooled. 'agree' leaves out unsure, cannot judge and provided.",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="outputs/review/queue_v1.json")
    p.add_argument("--log", default="outputs/review/log_v1.jsonl")
    p.add_argument("--out", default=None)
    return p.parse_args()


def group_of(item):
    if item["source"] == "veri":
        return "sample"
    return "random" if item["reasons"] == ["random"] else "fixed"


def asked(item, lat):
    names = list(item["elements"])
    if item["source"] == "veri":
        r = lat.get(("veri", item["image"], "vehicle_visible"))
        if r is not None and r.verdict == "wrong":
            return names[:1]
    return names


def describe(pairs):
    counts = collections.Counter(r.verdict for _item, r in pairs)
    k = counts["correct"]
    n = k + counts["wrong"]
    if n:
        low, high = wilson_interval(k, n)
        agree = "agree %d/%d = %.3f (wilson95 %.3f-%.3f)" % (k, n, k / n, low, high)
    else:
        agree = "agree 0/0"
    reasons = collections.Counter(r.reason for _item, r in pairs if r.verdict == "not_determinable")
    detail = ", ".join("%s %d" % kv for kv in sorted(reasons.items()))
    return "reviewed %3d | %s | unsure %d | cannot judge %d%s | provided %d" % (
        len(pairs), agree, counts["unsure"], counts["not_determinable"], " (" + detail + ")" if detail else "", counts["provided"])


def dataset_line(pairs, name):
    n = same_human = same_model = 0
    for item, r in pairs:
        if r.verdict not in ("correct", "wrong"):
            continue
        data = item["elements"][name].get("dataset_value")
        if data is None:
            continue
        human = r.model_value if r.verdict == "correct" else r.corrected
        n += 1
        same_human += human == data
        same_model += r.model_value == data
    if n == 0:
        return None
    return "    your value vs dataset label: same %d/%d | model vs dataset label: same %d/%d (same images)" % (
        same_human, n, same_model, n)


def main():
    args = parse_args()
    out = Path(args.out) if args.out else None
    if out is not None and out.exists():
        sys.exit(f"EXISTS: {out}")
    queue_bytes = Path(args.queue).read_bytes()
    queue = json.loads(queue_bytes.decode("utf-8"))
    records = read_log(args.log)
    lat = latest(records)
    known = {(i["source"], i["image"]) for i in queue["items"]}
    stray = sum(1 for key in lat if (key[0], key[1]) not in known)

    lines = [
        "queue sha256 %s | log %s | %d lines, %d latest verdicts, %d for images not in the queue"
        % (hashlib.sha256(queue_bytes).hexdigest(), args.log, len(records), len(lat), stray),
        "",
    ]
    groups = collections.defaultdict(list)
    for source in ("veri", "ccpd"):
        items = [i for i in queue["items"] if i["source"] == source]
        touched = sum(any((source, i["image"], n) in lat for n in i["elements"]) for i in items)
        done = sum(all((source, i["image"], n) in lat for n in asked(i, lat)) for i in items)
        lines.append("%s: %d images in the queue, %d with at least one verdict, %d complete" % (source, len(items), touched, done))
        for item in items:
            for name in item["elements"]:
                r = lat.get((source, item["image"], name))
                if r is not None:
                    groups[(source, name, group_of(item))].append((item, r))
    lines.append("")
    for source in ("veri", "ccpd"):
        for name in elements_for(source):
            for group in sorted({g for (s, n, g) in groups if s == source and n == name}):
                pairs = groups[(source, name, group)]
                lines.append("%-5s %-16s [%-6s] %s" % (source, name, group, describe(pairs)))
                extra = dataset_line(pairs, name)
                if extra:
                    lines.append(extra)
    lines.append("")
    lines += ["note: " + c for c in CAVEATS]
    text = "\n".join(lines) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
