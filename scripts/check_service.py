import argparse
import hashlib
import resource
import sys
import time
from pathlib import Path

import numpy as np

from weathercam.attributes.weather import load_weather, part_arrays
from weathercam.core.boxes import iou
from weathercam.core.cls_metrics import macro_f1
from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.core.roc_metrics import roc_auc
from weathercam.core.types import Box
from weathercam.data.split import group_split
from weathercam.data.veri import VeriReader
from weathercam.detection.records import load_predictions
from weathercam.plates.detector import best_detection
from weathercam.plates.ocr import decode_positions
from weathercam.retrieval.evaluate import camera_aware_scores, overall_score, query_scores
from weathercam.retrieval.index import BruteForceIndex
from weathercam.service.handlers import ood_scores, weather
from weathercam.service.heads import load_linear_head, load_metric_head, standardise
from weathercam.service.plate_client import PlateClient
from weathercam.service.worker_client import WorkerClient

VERI = "outputs/embeddings/veri_vitb16_laion2b.npz"
DAWN = "outputs/embeddings/dawn_vitb16_laion2b.npz"
VMMR = "outputs/embeddings/vmmr_vitb16_laion2b.npz"
METRIC = "outputs/heads_metric/vitb16_laion2b_g2_metric.npz"
WEATHER = "outputs/heads_weather/vitb16_laion2b_weather.npz"
PRED_TEST = "outputs/plate_detector/pred_test_chunked.json"
LOGITS = "outputs/plate_ocr/ocr_full2/logits_test_detected.npz"
DETECTOR = "outputs/plate_detector/runs/plate_yolo11m/weights/best.pt"
OCR = "outputs/plate_ocr/ocr_full2/ocr_weights.pt"
HASHES = {
    VERI: "b4d2e2458293cfed61c73e72e8ab1fc39872290fe42ec9e58d91fc5216f58640",
    DAWN: "7578d8fca79ee6feb76d0716bf996e7e0f839c8d45514f85f61875c6634c89ce",
    VMMR: "5718b139c0524e67d39ff32b89a3b8d7c1445e146fc034c3855b632384da4af1",
    METRIC: "e1d95f5df72767cbbcf69ccb94ef6185bfe5f52e64c743ce452327f94b44fd8e",
    WEATHER: "198281d3df8cd0f9e7d5752def8128ed126733aee0142849704f81ee28a700d5",
    PRED_TEST: "3199acae740baafdfe4b073eda02db62770972c0062f55a602237ad26b5ec156",
}
FIT_IMAGES = 29918
VAL_IMAGES = 7828
VAL_VEHICLES = 119
LINES = []
MISSES = []


def emit(*parts):
    line = " ".join(str(part) for part in parts)
    LINES.append(line)
    print(line, flush=True)


def verdict(name, got, expected):
    ok = round(got, 4) == round(expected, 4)
    emit(name, "got", round(got, 6), "recorded", round(expected, 6), "MATCH" if ok else "DIFFERENT")
    if not ok:
        MISSES.append(name)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_all(path):
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/service/check_v1.txt")
    p.add_argument("--veri-zip", default="/path/to/VeRi.zip")
    p.add_argument("--ccpd-dir", default="../ccpd_subset")
    p.add_argument("--embed-worker-python")
    p.add_argument("--plate-worker-python")
    p.add_argument("--images", type=int, default=16)
    return p.parse_args()


def inner_split(data, seed):
    train_rows = np.flatnonzero(data["split"] == "train")
    vehicles = data["vehicle_ids"][train_rows]
    classes = list(zip(data["color_ids"][train_rows].tolist(), data["type_ids"][train_rows].tolist()))
    fit_part, val_part = group_split(vehicles.tolist(), classes, (0.8, 0.2), seed)
    fit_rows = train_rows[np.asarray(fit_part)]
    val_rows = train_rows[np.asarray(val_part)]
    count = len(set(data["vehicle_ids"][val_rows].tolist()))
    if (len(fit_rows), len(val_rows), count) != (FIT_IMAGES, VAL_IMAGES, VAL_VEHICLES):
        sys.exit("inner split sizes differ from the recorded 29918 / 7828 / 119")
    return fit_rows, val_rows


def check_weather():
    head = load_linear_head(WEATHER, names_key="weathers")
    stored = np.load(WEATHER, allow_pickle=False)
    if str(stored["embeddings_sha256"]) != HASHES[DAWN]:
        sys.exit("weather head was trained on a different embeddings file")
    data = load_weather(DAWN)
    x, y = part_arrays(data, "val")
    truth = y.tolist()
    predicted = [item.class_id for item in weather(head, x)]
    correct = sum(int(a == b) for a, b in zip(truth, predicted))
    emit("A weather validation images", len(truth), "correct", correct, "recorded 127", "MATCH" if correct == 127 else "DIFFERENT")
    if correct != 127:
        MISSES.append("A weather correct")
    verdict("A weather macro-F1", macro_f1(truth, predicted, list(range(len(head.class_ids)))), 0.7889)


def check_metric(veri, val_rows):
    stored = np.load(METRIC, allow_pickle=False)
    features = load_metric_head(METRIC).project(veri["embeddings"][val_rows])
    vehicles = veri["vehicle_ids"][val_rows]
    cameras = veri["camera_ids"][val_rows]
    index = BruteForceIndex(features.shape[1])
    index.add(features)
    plain = overall_score(query_scores(index, features, vehicles))
    camera = overall_score(camera_aware_scores(index, features, vehicles, cameras))
    emit("B images", len(val_rows), "vehicles", len(set(vehicles.tolist())))
    verdict("B plain P@1", plain.precision_at_1, float(stored["val_plain_p1"]))
    verdict("B plain mAP@R", plain.map_at_r, float(stored["val_plain_map"]))
    verdict("B camera-aware P@1", camera.precision_at_1, float(stored["val_camera_p1"]))
    verdict("B camera-aware mAP@R", camera.map_at_r, float(stored["val_camera_map"]))


def check_ood(veri, dawn, vmmr, fit_rows, val_rows):
    stored = np.load(METRIC, allow_pickle=False)
    mean, std = stored["mean"], stored["std"]
    bank = standardise(veri["embeddings"][fit_rows].astype(np.float64), mean, std)
    identity = ood_scores(bank, veri["embeddings"][val_rows], mean, std, 5)
    for name, data, recorded in (("DAWN", dawn, 0.9978), ("VMMR", vmmr, 0.9945)):
        rows = np.flatnonzero(np.isin(data["split"], ("train", "val")))
        other = ood_scores(bank, data["embeddings"][rows], mean, std, 5)
        scores = np.concatenate((identity, other))
        labels = np.concatenate((np.zeros(len(identity)), np.ones(len(other))))
        emit("C bank", len(bank), "id", len(identity), name, len(other))
        verdict("C OOD AUROC " + name, roc_auc(scores, labels), recorded)


def check_embed_worker(python, veri, val_rows, veri_zip, count):
    rows = val_rows[:count]
    command = ["nice", "-n", "10", python, "research/embed_worker.py"]
    started = time.perf_counter()
    with WorkerClient(command, 600.0, 600.0) as client:
        emit("D embed worker startup seconds", round(time.perf_counter() - started, 1), client.model, client.pretrained, "dim", client.dim)
        with VeriReader(veri_zip) as reader:
            blobs = [reader.read_image("train", str(veri["image_names"][row])) for row in rows]
        started = time.perf_counter()
        vectors = client.encode_images(blobs)
        seconds = time.perf_counter() - started
    stored = veri["embeddings"][rows].astype(np.float64)
    cosines = np.sum(vectors.astype(np.float64) * stored, axis=1)
    emit("D images", len(rows), "seconds", round(seconds, 1), "min cosine", float(cosines.min()), "max abs diff", float(np.abs(vectors - stored).max()))


def check_plate_worker(python, ccpd_dir, count):
    stored = np.load(LOGITS, allow_pickle=False)
    emit("E stored logits sha256", sha(LOGITS))
    usable = np.flatnonzero(~stored["missing"])
    rows = usable[:: max(len(usable) // count, 1)][:count]
    files = [str(stored["files"][row]) for row in rows]
    stored_scores = {item.name: best_detection(item.detections) for item in load_predictions(PRED_TEST).images}
    blobs = [(Path(ccpd_dir) / name).read_bytes() for name in files]
    command = ["env", "PYTHONPATH=src", "nice", "-n", "10", python, "research/plate_worker.py", "--detector", DETECTOR, "--ocr", OCR]
    started = time.perf_counter()
    with PlateClient(command, 600.0, 600.0) as client:
        emit("E plate worker startup seconds", round(time.perf_counter() - started, 1), "detector", client.detector_sha256[:12], "ocr", client.ocr_sha256[:12])
        started = time.perf_counter()
        results = client.read_plates(blobs)
        seconds = time.perf_counter() - started
    overlaps, diffs, same = [], [], 0
    for result, row, name in zip(results, rows, files):
        reference = decode_positions(stored["logits"][row])
        box = Box(*[float(v) for v in stored["boxes"][row]])
        overlap = 0.0 if result.box is None else iou(Box(*result.box), box)
        overlaps.append(overlap)
        if result.reading is None:
            emit("E", name, "no reading", "iou", round(overlap, 4))
            continue
        match = result.reading.text == reference.text
        same += int(match)
        diff = max(abs(a - b) for a, b in zip(result.reading.positions, reference.positions))
        diffs.append(diff)
        emit("E", name, "iou", round(overlap, 4), "score", round(result.score, 4), "stored score", round(stored_scores[name].score, 4), "text", result.reading.text, "stored text", reference.text, "same", match)
    emit("E images", len(rows), "seconds", round(seconds, 1), "min iou", round(min(overlaps), 4), "same text", same, "max position-probability diff", round(max(diffs), 6) if diffs else None)


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit("refusing to overwrite %s" % out)
    config = load_config()
    guard = MemoryGuard(config.memory_limit_mb)
    for path, expected in HASHES.items():
        if sha(path) != expected:
            sys.exit("hash differs for %s" % path)
        emit("hash ok", path)
    veri = load_all(VERI)
    fit_rows, val_rows = inner_split(veri, config.seed)
    guard.check()
    check_weather()
    check_metric(veri, val_rows)
    guard.check()
    check_ood(veri, load_all(DAWN), load_all(VMMR), fit_rows, val_rows)
    guard.check()
    if args.embed_worker_python:
        check_embed_worker(args.embed_worker_python, veri, val_rows, args.veri_zip, args.images)
    if args.plate_worker_python:
        check_plate_worker(args.plate_worker_python, args.ccpd_dir, args.images)
    emit("peak RSS main process MB", round(peak_mb()), "(workers not counted)")
    emit("differences", len(MISSES), MISSES)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("wrote", out, "| sha256", sha(out))


if __name__ == "__main__":
    main()
