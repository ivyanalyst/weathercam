import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import uvicorn

from weathercam.core.memory import MemoryGuard
from weathercam.retrieval.innersplit import veri_inner_split
from weathercam.service.app import OodBank, create_app
from weathercam.service.handlers import GalleryIndex
from weathercam.service.heads import COLOR_NAMES, TYPE_NAMES, load_linear_head, standardise
from weathercam.service.plate_client import PlateClient
from weathercam.service.worker_client import WorkerClient

VERI = "outputs/embeddings/veri_vitb16_laion2b.npz"
METRIC = "outputs/heads_metric/vitb16_laion2b_g2_metric.npz"
WEATHER = "outputs/heads_weather/vitb16_laion2b_weather.npz"
BRAND = "outputs/heads_brand/vitb16_laion2b_brand.npz"
COLOR = "outputs/heads_std/vitb16_laion2b_color.npz"
TYPE = "outputs/heads_std/vitb16_laion2b_type.npz"
DETECTOR = "outputs/plate_detector/runs/plate_yolo11m/weights/best.pt"
OCR = "outputs/plate_ocr/ocr_full2/ocr_weights.pt"
DETECTOR_SHA256 = "59b2521b4c0b239272373e4085d2ef854eb0a78dc4566dc5cb51e808e971666d"
OCR_SHA256 = "3dd0c9d2110c79a77e05ac521815449a35eb120481e02b242c60a1f2a58f7a57"
EMBED_DIM = 512
SEED = 42
FIT_IMAGES = 29918
VAL_IMAGES = 7828
OOD_K = 5
VERI_KEYS = ("embeddings", "split", "vehicle_ids", "color_ids", "type_ids")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-worker-python")
    p.add_argument("--plate-worker-python")
    p.add_argument("--gallery")
    p.add_argument("--image-zip")
    p.add_argument("--no-ood", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--memory-limit-mb", type=float, default=3000.0)
    return p.parse_args()


def load_heads():
    weather = load_linear_head(WEATHER, names_key="weathers")
    attributes = {
        "color": load_linear_head(COLOR, names=COLOR_NAMES),
        "type": load_linear_head(TYPE, names=TYPE_NAMES),
        "brand": load_linear_head(BRAND, names_key="makes"),
    }
    return weather, attributes


def load_ood():
    with np.load(VERI, allow_pickle=False) as archive:
        data = {key: archive[key] for key in VERI_KEYS}
    fit_rows, val_rows = veri_inner_split(data, SEED)
    if (len(fit_rows), len(val_rows)) != (FIT_IMAGES, VAL_IMAGES):
        sys.exit("inner split sizes differ from the recorded 29918 / 7828")
    with np.load(METRIC, allow_pickle=False) as stored:
        mean, std = stored["mean"], stored["std"]
    bank = standardise(data["embeddings"][fit_rows].astype(np.float64), mean, std)
    return OodBank(bank, mean, std, OOD_K)


def build_ood(args):
    if args.no_ood:
        print("ood disabled: --no-ood", flush=True)
        return None
    missing = [path for path in (VERI, METRIC) if not Path(path).exists()]
    if missing:
        print(f"ood disabled: missing {missing}", flush=True)
        return None
    return load_ood()


def load_gallery(path):
    with np.load(path, allow_pickle=False) as archive:
        for key in ("names", "embeddings"):
            if key not in archive.files:
                sys.exit(f"{path}: missing key {key!r}")
        names = archive["names"].tolist()
        embeddings = archive["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        sys.exit(f"gallery embeddings must have shape (n, {EMBED_DIM}), got {embeddings.shape}")
    return GalleryIndex(names, embeddings)


def start_embed(args, clients):
    command = ["nice", "-n", "10", args.embed_worker_python, "research/embed_worker.py"]
    client = WorkerClient(command, args.request_timeout, args.startup_timeout)
    clients.append(client)
    if client.dim != EMBED_DIM:
        sys.exit(f"embedding worker dim is {client.dim}, the heads need {EMBED_DIM}")
    print(f"embedding worker ready: {client.model} {client.pretrained} dim {client.dim}", flush=True)
    return client


def start_plates(args, clients):
    command = ["env", "PYTHONPATH=src", "YOLO_AUTOINSTALL=false", "nice", "-n", "10", args.plate_worker_python, "research/plate_worker.py", "--detector", DETECTOR, "--ocr", OCR]
    client = PlateClient(command, args.request_timeout, args.startup_timeout)
    clients.append(client)
    if (client.detector_sha256, client.ocr_sha256) != (DETECTOR_SHA256, OCR_SHA256):
        sys.exit(f"plate worker hashes differ: {client.detector_sha256} {client.ocr_sha256}")
    print(f"plate worker ready: detector {client.detector_sha256[:12]} ocr {client.ocr_sha256[:12]}", flush=True)
    return client


def open_images(path, gallery):
    import zipfile

    from weathercam.data.veri import ZIP_ROOT, VeriReader

    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as error:
        sys.exit(f"cannot read {path}: {error}")
    missing = [name for name in gallery.names if f"{ZIP_ROOT}/image_test/{name}" not in members]
    if missing:
        sys.exit(f"{len(missing)} gallery images are not in {path} image_test, first: {missing[0]}")
    return VeriReader(path)


def main():
    args = parse_args()
    if not args.embed_worker_python and not args.plate_worker_python:
        sys.exit("give --embed-worker-python, --plate-worker-python or both")
    if args.gallery and not args.embed_worker_python:
        sys.exit("--gallery needs --embed-worker-python")
    if args.image_zip and not args.gallery:
        sys.exit("--image-zip needs --gallery")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    guard = MemoryGuard(args.memory_limit_mb)
    clients = []
    encoder = reader = weather = attributes = ood = gallery = image_reader = None
    try:
        if args.embed_worker_python:
            encoder = start_embed(args, clients)
            weather, attributes = load_heads()
            ood = build_ood(args)
            if args.gallery:
                gallery = load_gallery(args.gallery)
                if args.image_zip:
                    image_reader = open_images(args.image_zip, gallery)
        if args.plate_worker_python:
            reader = start_plates(args, clients)
        print(f"main process RSS {guard.check():.0f} MB", flush=True)
        app = create_app(encoder=encoder, weather_head=weather, attribute_heads=attributes, ood=ood, gallery=gallery, plate_reader=reader, image_reader=image_reader)
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if image_reader is not None:
            image_reader.close()
        for client in clients:
            client.close()


if __name__ == "__main__":
    main()
