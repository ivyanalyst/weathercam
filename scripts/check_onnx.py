import argparse
import hashlib
import io
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.core.types import Box
from weathercam.data.veri import VeriReader
from weathercam.plates.crop import crop_array, expand_box, resize_rgb
from weathercam.plates.ocr import POSITION_CLASSES, decode_positions
from weathercam.retrieval.innersplit import veri_inner_split

VERI = "outputs/embeddings/veri_vitb16_laion2b.npz"
LOGITS = "outputs/plate_ocr/ocr_full2/logits_test_detected.npz"
CLIP_ONNX = "outputs/onnx/clip_vitb16_laion2b_visual.onnx"
PLATE_ONNX = "outputs/onnx/plate_ocr.onnx"
HASHES = {
    VERI: "b4d2e2458293cfed61c73e72e8ab1fc39872290fe42ec9e58d91fc5216f58640",
    LOGITS: "00f841cff6c481d04051f91f86041d317027cea462788ca08c90298c2fc914ea",
    CLIP_ONNX: "d69d460db439a2c9eac9edbb47a96c1d03c0e4df3c212a7c0c95ade43ebabf17",
    CLIP_ONNX + ".data": "910c3f75c0eda73990e39c7647735a44f9167a40ff79ae18a77086a2a5eeab30",
    PLATE_ONNX: "13b25203658a68dc3fc371ac60dcf9a0ffd87e94bd4dc79c80e2f6829342604a",
    PLATE_ONNX + ".data": "d593ef4bd171dc0944cd5173a1c18af8e00b09eab97971dc7db1e3b55b3111fa",
}
FIT_IMAGES = 29918
VAL_IMAGES = 7828
VAL_VEHICLES = 119
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
MARGIN = 0.15
OUT_H, OUT_W = 96, 192
TIMING_RUNS = 3
SEED = 42
LINES = []
MISSES = []


def emit(*parts):
    line = " ".join(str(part) for part in parts)
    LINES.append(line)
    print(line, flush=True)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_all(path, allow_pickle):
    with np.load(path, allow_pickle=allow_pickle) as archive:
        return {key: archive[key] for key in archive.files}


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/onnx/check_v1.txt")
    p.add_argument("--veri-zip", default="/path/to/VeRi.zip")
    p.add_argument("--ccpd-dir", default="../ccpd_subset")
    p.add_argument("--clip-images", type=int, default=64)
    p.add_argument("--per-subset", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--plate-batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


def inner_split(data, seed):
    fit_rows, val_rows = veri_inner_split(data, seed)
    count = len(set(data["vehicle_ids"][val_rows].tolist()))
    if (len(fit_rows), len(val_rows), count) != (FIT_IMAGES, VAL_IMAGES, VAL_VEHICLES):
        sys.exit("inner split sizes differ from the recorded 29918 / 7828 / 119")
    return fit_rows, val_rows


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def describe(label, sess):
    first, last = sess.get_inputs()[0], sess.get_outputs()[0]
    emit(label, first.name, first.shape, last.name, last.shape, sess.get_providers())


def run(sess, array):
    return sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: array})[0]


def run_batches(sess, inputs, batch):
    return np.concatenate([run(sess, inputs[start : start + batch]) for start in range(0, len(inputs), batch)])


def evenly(rows, count):
    if len(rows) <= count:
        return rows
    return rows[np.unique(np.linspace(0, len(rows) - 1, count).round().astype(int))]


def timing(sess, shape):
    data = np.random.default_rng(SEED).random(shape, dtype=np.float32)
    run(sess, data)
    runs = []
    for _ in range(TIMING_RUNS):
        started = time.perf_counter()
        run(sess, data)
        runs.append(round(time.perf_counter() - started, 3))
    return runs


def preprocess(blob):
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert("RGB")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    new_long = int(SIZE * long / short)
    new_width, new_height = (SIZE, new_long) if width <= height else (new_long, SIZE)
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    top = int(round((new_height - SIZE) / 2.0))
    left = int(round((new_width - SIZE) / 2.0))
    array = np.asarray(image.crop((left, top, left + SIZE, top + SIZE)), dtype=np.float32) / np.float32(255)
    return ((array - MEAN) / STD).transpose(2, 0, 1)


def unit(features):
    wide = features.astype(np.float64)
    return wide / np.linalg.norm(wide, axis=1, keepdims=True)


def check_clip(sess, veri, val_rows, veri_zip, args):
    rows = evenly(val_rows, args.clip_images)
    with VeriReader(veri_zip) as reader:
        blobs = [reader.read_image("train", str(veri["image_names"][row])) for row in rows]
    started = time.perf_counter()
    inputs = np.stack([preprocess(blob) for blob in blobs])
    prep_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = unit(run_batches(sess, inputs, args.batch))
    seconds = time.perf_counter() - started
    single = unit(run(sess, inputs[:1]))
    stored = veri["embeddings"][rows].astype(np.float64)
    cosines = np.sum(vectors * stored, axis=1)
    emit("F clip images", len(rows), "preprocess seconds", round(prep_seconds, 1), "onnx seconds", round(seconds, 1), "min cosine", float(cosines.min()), "max abs diff", float(np.abs(vectors - stored).max()), "batch 1 vs batch max abs diff", float(np.abs(single[0] - vectors[0]).max()))


def plate_inputs(ccpd_dir, stored, rows):
    inputs, kept = [], []
    for row in rows:
        with Image.open(Path(ccpd_dir) / str(stored["files"][row])) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        box = Box(*[float(v) for v in stored["boxes"][row]])
        try:
            crop = crop_array(rgb, expand_box(box, MARGIN, width, height))
        except ValueError:
            continue
        inputs.append(resize_rgb(crop.astype(np.float32) / 255.0, OUT_H, OUT_W).transpose(2, 0, 1))
        kept.append(row)
    return np.stack(inputs), np.array(kept)


def check_plate(sess, ccpd_dir, args):
    stored = load_all(LOGITS, False)
    usable = ~stored["missing"]
    subsets = stored["subsets"]
    provinces = set()
    for name in sorted(set(subsets[usable].tolist())):
        rows = evenly(np.flatnonzero(usable & (subsets == name)), args.per_subset)
        inputs, kept = plate_inputs(ccpd_dir, stored, rows)
        logits = run_batches(sess, inputs, args.plate_batch)
        wrong, prob_diffs, logit_diffs = [], [], []
        for row, got in zip(kept, logits):
            reference = stored["logits"][row]
            ours, theirs = decode_positions(got), decode_positions(reference)
            if ours.text != theirs.text:
                wrong.append(int(row))
                MISSES.append("plate row %d" % row)
            prob_diffs.append(max(abs(a - b) for a, b in zip(ours.positions, theirs.positions)))
            logit_diffs.append(float(np.abs(got - reference).max()))
            provinces.add(int(np.argmax(reference[0, : POSITION_CLASSES[0]])))
        emit("G", name, "rows", len(rows), "used", len(kept), "same text", len(kept) - len(wrong), "max logit diff", round(max(logit_diffs), 6), "max position-probability diff", round(max(prob_diffs), 6), "differing rows", wrong)
    single = run(sess, inputs[:1])
    emit("G batch 1 vs batch max abs diff (last subset)", float(np.abs(single[0] - logits[0]).max()))
    emit("G province classes in sample", len(provinces), "of", POSITION_CLASSES[0])


def main():
    args = parse_args()
    if min(args.clip_images, args.per_subset, args.batch, args.plate_batch, args.threads) < 1:
        sys.exit("counts, batch sizes and threads must be at least 1")
    out = Path(args.out)
    if out.exists():
        sys.exit("refusing to overwrite %s" % out)
    config = load_config()
    guard = MemoryGuard(config.memory_limit_mb)
    for path, expected in HASHES.items():
        if sha(path) != expected:
            sys.exit("hash differs for %s" % path)
        emit("hash ok", path)
    veri = load_all(VERI, True)
    _, val_rows = inner_split(veri, config.seed)
    guard.check()
    clip = session(CLIP_ONNX, args.threads)
    describe("F clip io", clip)
    guard.check()
    check_clip(clip, veri, val_rows, args.veri_zip, args)
    emit("F clip onnx timing seconds (batch 16, random input)", timing(clip, (16, 3, SIZE, SIZE)))
    guard.check()
    plate = session(PLATE_ONNX, args.threads)
    describe("G plate io", plate)
    check_plate(plate, args.ccpd_dir, args)
    emit("G plate onnx timing seconds (batch 64, random input)", timing(plate, (64, 3, OUT_H, OUT_W)))
    guard.check()
    emit("peak RSS main process MB", round(peak_mb()))
    emit("differences", len(MISSES), MISSES)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("wrote", out, "| sha256", sha(out))


if __name__ == "__main__":
    main()
