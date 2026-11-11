import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict_proba
from weathercam.attributes.zeroshot import class_vectors, image_probabilities
from weathercam.data.veri import read_names
from weathercam.plates.ocr import decode_batch
from weathercam.review.queue import select_ccpd, select_veri
from weathercam.review.verdicts import elements_for

# The same 13 images as research/score_plates.py: labelled box not on the labelled plate.
EXCLUDED = (
    "images/022474.jpg", "images/022192.jpg", "images/024070.jpg",
    "images/022458.jpg", "images/022983.jpg", "images/023008.jpg", "images/023262.jpg",
    "images/023308.jpg", "images/023530.jpg", "images/023847.jpg", "images/024844.jpg",
    "images/026860.jpg", "images/027137.jpg",
)
EXCLUDED_SHA256 = "2fdff74e937a4204759dba073a9829523c04c35fc2e3bdb067d6da7b57e0b474"
# Checked against the v26 handoff; the other inputs are only recorded with their hash.
EXPECTED = {
    "manifest": "80aafd729e0fdb9616fdd86a0262708af77c0363ad10bf789668c01b12601dab",
    "split": "fb3c5b296835c10dc1a0857db67263882bcba86b8fe59dbb2ec074b1b97fa435",
    "pred": "3199acae740baafdfe4b073eda02db62770972c0062f55a602237ad26b5ec156",
    "veri_b32_qg": "f896af50c15246e203cb999328dc6f4d449ddacc299b237dadeca54fdfa81208",
    "brand_head": "7da1109102f0f9189bb7b308d993cb13f8b563ef101155f2c508b562e5f913cc",
}
VIEW_SCALE = 100.0  # the default of research/zeroshot_view.py
TAG = "vitb16_laion2b"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--veri-zip", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ccpd-root", required=True)
    p.add_argument("--out", default="outputs/review/queue_v1.json")
    p.add_argument("--veri-b16", default="outputs/embeddings/veri_vitb16_laion2b.npz")
    p.add_argument("--veri-b32", default="outputs/embeddings/veri_vitb32_openai_qg.npz")
    p.add_argument("--view-text", default="outputs/view/view_text_vitb32_openai_qg.npz")
    p.add_argument("--heads-std", default="outputs/heads_std")
    p.add_argument("--heads-brand", default="outputs/heads_brand")
    p.add_argument("--run-dir", default="outputs/plate_ocr/ocr_full2")
    p.add_argument("--pred", default="outputs/plate_detector/pred_test_chunked.json")
    p.add_argument("--misses", default="outputs/plate_detector/misses_test.txt")
    p.add_argument("--veri-n", type=int, default=100)
    p.add_argument("--ccpd-per-subset", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(entries, label, path, expected=None):
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        sys.exit(f"hash differs from the expected value: {label} {path}\n  got      {digest}\n  expected {expected}")
    entries[label] = {"path": str(path), "sha256": digest, "checked": expected is not None}


def head_predict(head, raw):
    x = (raw - head["mean"]) / head["std"]
    probs = predict_proba(SoftmaxModel(head["weights"], head["bias"]), x)
    return probs.argmax(axis=1), probs.max(axis=1)


def conf(value):
    return round(float(value), 4)


def build_veri(args, entries):
    b16, b32 = Path(args.veri_b16), Path(args.veri_b32)
    colour_path = Path(args.heads_std) / f"{TAG}_color.npz"
    type_path = Path(args.heads_std) / f"{TAG}_type.npz"
    brand_path = Path(args.heads_brand) / f"{TAG}_brand.npz"
    record(entries, "veri_embeddings_b16", b16)
    record(entries, "veri_embeddings_b32_qg", b32, EXPECTED["veri_b32_qg"])
    record(entries, "view_text_b32_qg", args.view_text)
    record(entries, "head_colour_b16", colour_path)
    record(entries, "head_type_b16", type_path)
    record(entries, "head_brand_b16", brand_path, EXPECTED["brand_head"])

    d16, d32 = np.load(b16), np.load(b32)
    if not (d16["image_names"] == d32["image_names"]).all() or not (d16["split"] == d32["split"]).all():
        sys.exit("the B/16 and B/32 VeRi embedding files differ in image names or split")
    names = [str(x) for x in d16["image_names"]]
    vehicle = [str(x) for x in d16["vehicle_ids"]]
    camera = [str(x) for x in d16["camera_ids"]]
    split = [str(x) for x in d16["split"]]
    picks = select_veri(vehicle, camera, split, "test", args.veri_n, args.seed)
    if len(picks) != args.veri_n:
        sys.exit(f"only {len(picks)} VeRi test vehicles available, wanted {args.veri_n}")

    colors, types = read_names(args.veri_zip)
    emb16 = d16["embeddings"][picks]
    colour_head, type_head, brand_head = np.load(colour_path), np.load(type_path), np.load(brand_path)
    if not (brand_head["class_ids"] == np.arange(len(brand_head["class_ids"]))).all():
        sys.exit("brand class_ids are not 0..n-1")
    makes = [str(m) for m in brand_head["makes"]]
    c_idx, c_conf = head_predict(colour_head, emb16)
    t_idx, t_conf = head_predict(type_head, emb16)
    b_idx, b_conf = head_predict(brand_head, emb16)

    text = np.load(args.view_text)
    probs = image_probabilities(d32["embeddings"][picks], class_vectors(text["text"]), VIEW_SCALE)
    classes = [str(c) for c in text["classes"]]
    v_idx, v_conf = probs.argmax(axis=1), probs.max(axis=1)

    items = []
    for k, i in enumerate(picks):
        elements = {
            "vehicle_visible": {"model_value": "vehicle", "origin": "dataset"},
            "colour": {
                "model_value": colors[int(colour_head["class_ids"][c_idx[k]])], "confidence": conf(c_conf[k]),
                "dataset_value": colors[int(d16["color_ids"][i])], "origin": "model",
            },
            "type": {
                "model_value": types[int(type_head["class_ids"][t_idx[k]])], "confidence": conf(t_conf[k]),
                "dataset_value": types[int(d16["type_ids"][i])], "origin": "model",
            },
            "brand": {"model_value": makes[int(b_idx[k])], "confidence": conf(b_conf[k]), "origin": "model"},
            "view": {"model_value": classes[int(v_idx[k])], "confidence": conf(v_conf[k]), "origin": "model"},
            "plate_box": {"model_value": None, "origin": "none"},
            "plate_text": {"model_value": None, "origin": "none"},
        }
        if tuple(elements) != elements_for("veri"):
            sys.exit("VeRi element list differs from review.verdicts")
        items.append({
            "source": "veri", "image": names[i], "split": "test", "reasons": ["sample"],
            "vehicle_id": vehicle[i], "camera_id": camera[i], "elements": elements,
        })
    return items


def read_misses(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = re.match(r"misses \(top-1 IoU <= 0\.7 or no detection\): (\d+) \| no detection: (\d+) \| IoU <= 0\.5: (\d+)", lines[0])
    if head is None:
        sys.exit("unexpected first line in the misses report")
    total, n_none, n_low = (int(g) for g in head.groups())
    rows, none, low = 0, [], []
    for line in lines:
        if not line.startswith("images/"):
            continue
        fields = [f.strip() for f in line.split(" | ")]
        if len(fields) != 12:
            sys.exit(f"unexpected row in the misses report: {line[:80]}")
        rows += 1
        if fields[-1] == "0":
            none.append(fields[0])
        if float(fields[3]) <= 0.5:
            low.append(fields[0])
    if (rows, len(none), len(low)) != (total, n_none, n_low):
        sys.exit(f"misses report rows {(rows, len(none), len(low))} differ from its header {(total, n_none, n_low)}")
    return none, low


def build_ccpd(args, entries, manifest):
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for name, expected in run["hashes"].items():
        record(entries, "ocr_run/" + name, run_dir / name, expected)
    if "logits_test_detected.npz" not in run["hashes"]:
        record(entries, "ocr_run/logits_test_detected.npz", run_dir / "logits_test_detected.npz")
    if run["split_sha256"] != manifest["split_sha256"]:
        sys.exit("split_sha256 differs between run.json and the manifest")
    record(entries, "detector_predictions", args.pred, EXPECTED["pred"])
    record(entries, "detector_misses_report", args.misses)
    if hashlib.sha256("\n".join(sorted(EXCLUDED)).encode()).hexdigest() != EXCLUDED_SHA256:
        sys.exit("the excluded list differs from the stored hash")

    test = [i for i in manifest["items"] if i["part"] == "test"]
    files = [i["file"] for i in test]
    if len(files) != 13892:
        sys.exit(f"expected 13892 test images, found {len(files)}")
    det = np.load(run_dir / "logits_test_detected.npz")
    if [str(f) for f in det["files"]] != files:
        sys.exit("file order of the logits differs from the manifest")
    none, low = read_misses(args.misses)
    if set(none) != {files[k] for k in np.flatnonzero(det["missing"])}:
        sys.exit("no-detection files in the misses report differ from the OCR missing flags")

    by_subset = {}
    for item in test:
        by_subset.setdefault(item["subset"], []).append(item["file"])
    fixed = {"no_detection": none, "iou_le_0.5": low, "label_off_plate": list(EXCLUDED)}
    chosen = select_ccpd(by_subset, fixed, args.ccpd_per_subset, args.seed)
    row_of = {f: k for k, f in enumerate(files)}
    info = {i["file"]: i for i in test}
    rows = [row_of[name] for name, _reasons in chosen]
    readings = decode_batch(np.asarray(det["logits"][rows]))

    items = []
    for (name, reasons), row, reading in zip(chosen, rows, readings):
        meta = info[name]
        missing = bool(det["missing"][row])
        elements = {
            "detector_box": {"model_value": None if missing else [float(v) for v in det["boxes"][row]], "origin": "model"},
            "label_box": {"model_value": [float(v) for v in meta["box"]], "origin": "dataset"},
            "plate_text": {
                "model_value": None if missing else reading.text,
                "confidence": None if missing else conf(reading.confidence),
                "dataset_value": meta["plate"], "origin": "model",
            },
        }
        if tuple(elements) != elements_for("ccpd"):
            sys.exit("CCPD element list differs from review.verdicts")
        items.append({
            "source": "ccpd", "image": name, "split": "test", "subset": meta["subset"],
            "reasons": list(reasons), "elements": elements,
        })
    return items


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    if not Path(args.veri_zip).is_file():
        sys.exit(f"VeRi zip not found: {args.veri_zip}")
    if not (Path(args.ccpd_root) / "images").is_dir():
        sys.exit(f"no images folder under {args.ccpd_root}")
    entries = {}
    manifest_path = Path(args.manifest)
    record(entries, "ccpd_manifest", manifest_path, EXPECTED["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != EXPECTED["split"]:
        sys.exit("split_sha256 in the manifest differs from the expected value")

    veri_items = build_veri(args, entries)
    ccpd_items = build_ccpd(args, entries, manifest)
    queue = {
        "format": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "settings": {"veri_n": args.veri_n, "ccpd_per_subset": args.ccpd_per_subset},
        "paths": {"veri_zip": str(Path(args.veri_zip).resolve()), "ccpd_root": str(Path(args.ccpd_root).resolve())},
        "provenance": entries,
        "items": veri_items + ccpd_items,
    }
    text = json.dumps(queue, ensure_ascii=False, indent=1) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)

    reasons = {}
    for item in ccpd_items:
        for reason in item["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    print("written", out, "sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())
    print("veri items", len(veri_items), "ccpd items", len(ccpd_items))
    print("ccpd reasons (an image can have several)", reasons)
    print("inputs checked", sum(e["checked"] for e in entries.values()), "recorded only", sum(not e["checked"] for e in entries.values()))


if __name__ == "__main__":
    main()
