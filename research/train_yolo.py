import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

EXPECTED = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]
WEIGHTS_BYTES = 40684120
WEIGHTS_SHA256 = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="yolo11m_dawn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-weights-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        sys.exit(f"weights not found: {weights}")
    if not args.skip_weights_check:
        if weights.stat().st_size != WEIGHTS_BYTES:
            sys.exit(f"unexpected weights size: {weights.stat().st_size}")
        if file_sha256(weights) != WEIGHTS_SHA256:
            sys.exit("unexpected weights sha256")
    data_dir = Path(args.data_dir).resolve()
    source = yaml.safe_load((data_dir / "data.yaml").read_text())
    names = {int(key): value for key, value in source["names"].items()}
    if [names[i] for i in sorted(names)] != EXPECTED:
        sys.exit(f"unexpected class names: {names}")
    for part in ("train", "val"):
        if not (data_dir / "images" / part).is_dir():
            sys.exit(f"missing images/{part}")
    project = Path(args.project).resolve()
    if (project / args.name).exists():
        sys.exit(f"run folder exists: {project / args.name}")
    project.mkdir(parents=True, exist_ok=True)
    run_yaml = project / f"{args.name}_data.yaml"
    config = {"path": str(data_dir), "train": "images/train", "val": "images/val", "names": names}
    run_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    device = int(args.device) if args.device.isdigit() else args.device
    model = YOLO(str(weights))
    model.train(
        data=str(run_yaml),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        device=device,
        project=str(project),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        cache=False,
        amp=True,
        plots=args.plots,
    )
    save_dir = Path(model.trainer.save_dir)
    settings = {
        "args": vars(args),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "weights_sha256": file_sha256(weights),
        "best": str(model.trainer.best),
    }
    (save_dir / "train_settings.json").write_text(json.dumps(settings, indent=2))
    print("run folder", save_dir)
    print("best weights", model.trainer.best)


if __name__ == "__main__":
    main()
