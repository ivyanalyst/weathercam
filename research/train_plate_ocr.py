import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from weathercam.core.types import Box
from weathercam.detection.records import load_predictions
from weathercam.plates import ocr_augment
from weathercam.plates.crop import crop_array, expand_box, jitter_box, pixel_box, resize_rgb
from weathercam.plates.ocr import NUM_CLASSES, POSITION_CLASSES
from weathercam.plates.ocr_data import draw_probabilities, encode_plate, non_wan_share

WIDTH, HEIGHT = 720, 1160
OUT_H, OUT_W = 96, 192
WINDOW = 0.5
MARGIN = 0.15
SHIFT, SCALE = 0.1, 0.15
EXPECTED = {"train": 16001, "val": 3999, "test": 13892}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_rgb(path):
    with Image.open(path) as image:
        if image.size != (WIDTH, HEIGHT):
            raise ValueError(f"{path}: unexpected size {image.size}")
        return np.asarray(image.convert("RGB"))


def hull_of(item):
    return Box(*[float(v) for v in item["box"]])


def to_input(crop):
    return resize_rgb(crop.astype(np.float32) / 255.0, OUT_H, OUT_W)


class WindowCache:
    def __init__(self, src, items, threads, name):
        self.hulls = [hull_of(item) for item in items]
        windows = [pixel_box(expand_box(h, WINDOW, WIDTH, HEIGHT), WIDTH, HEIGHT) for h in self.hulls]
        self.origins = [(w[0], w[1]) for w in windows]
        self.shapes = [(w[3] - w[1], w[2] - w[0]) for w in windows]
        self.offsets = [0]
        for height, width in self.shapes:
            self.offsets.append(self.offsets[-1] + height * width * 3)
        self.data = np.empty(self.offsets[-1], dtype=np.uint8)
        print("%s cache: %d windows, %.0f MB" % (name, len(items), self.offsets[-1] / 1e6), flush=True)

        def fill(i):
            x1, y1, x2, y2 = windows[i]
            image = load_rgb(src / items[i]["file"])
            self.data[self.offsets[i] : self.offsets[i + 1]] = image[y1:y2, x1:x2].reshape(-1)

        started = time.time()
        with ThreadPoolExecutor(threads) as pool:
            for done, _ in enumerate(pool.map(fill, range(len(items))), start=1):
                if done % 2000 == 0 or done == len(items):
                    print("  %s decoded %d/%d | %.0f s" % (name, done, len(items), time.time() - started), flush=True)

    def window(self, i):
        height, width = self.shapes[i]
        return self.data[self.offsets[i] : self.offsets[i + 1]].reshape(height, width, 3)

    def crop(self, i, box):
        ox, oy = self.origins[i]
        return crop_array(self.window(i), Box(box.x1 - ox, box.y1 - oy, box.x2 - ox, box.y2 - oy))


def check_cache(cache, items, src, seed, count=20):
    n = len(items)
    for i in sorted({int(k) for k in np.linspace(0, n - 1, min(count, n))}):
        full = load_rgb(src / items[i]["file"])
        hull = cache.hulls[i]
        rng = np.random.default_rng([seed, 999, i])
        boxes = [
            expand_box(jitter_box(hull, rng, SHIFT, SCALE), MARGIN, WIDTH, HEIGHT),
            expand_box(hull, MARGIN, WIDTH, HEIGHT),
        ]
        cx, cy = (hull.x1 + hull.x2) / 2, (hull.y1 + hull.y2) / 2
        w2, h2 = hull.width * (1 + SCALE), hull.height * (1 + SCALE)
        for sx in (-1, 1):
            for sy in (-1, 1):
                mx, my = cx + sx * SHIFT * hull.width, cy + sy * SHIFT * hull.height
                boxes.append(expand_box(Box(mx - w2 / 2, my - h2 / 2, mx + w2 / 2, my + h2 / 2), MARGIN, WIDTH, HEIGHT))
        for box in boxes:
            if not np.array_equal(crop_array(full, box), cache.crop(i, box)):
                sys.exit(f"window crop differs from full-image crop: item {i} {items[i]['file']}")
    print("window self-check passed on %d items (random and worst-case boxes)" % min(count, n), flush=True)


class TrainSet(Dataset):
    def __init__(self, cache, labels, seed):
        self.cache, self.labels, self.seed, self.epoch = cache, labels, seed, 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        box = expand_box(jitter_box(self.cache.hulls[i], rng, SHIFT, SCALE), MARGIN, WIDTH, HEIGHT)
        image = ocr_augment.augment_crop(to_input(self.cache.crop(i, box)), rng)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), torch.from_numpy(self.labels[i])


class PlateNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (32, 64, 128, 256):
            for _ in range(2):
                layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
                c_in = c_out
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (OUT_H // 16) * (OUT_W // 16), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 7 * NUM_CLASSES),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.5
        return self.head(self.features(x)).view(-1, 7, NUM_CLASSES)


def ocr_loss(logits, labels):
    return sum(F.cross_entropy(logits[:, p, :valid], labels[:, p]) for p, valid in enumerate(POSITION_CLASSES))


def check_loss_mask():
    logits = torch.randn(4, 7, NUM_CLASSES, requires_grad=True)
    ocr_loss(logits, torch.zeros(4, 7, dtype=torch.long)).backward()
    for p, valid in enumerate(POSITION_CLASSES):
        if valid < NUM_CLASSES and float(logits.grad[:, p, valid:].abs().max()) != 0.0:
            sys.exit(f"masked classes received gradient at position {p}")
        if float(logits.grad[:, p, :valid].abs().max()) == 0.0:
            sys.exit(f"valid classes received no gradient at position {p}")


def predict(model, inputs, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            x = torch.from_numpy(inputs[start : start + batch]).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 7, NUM_CLASSES), np.float32)


def masked_argmax(logits):
    return np.stack([logits[:, p, :valid].argmax(axis=1) for p, valid in enumerate(POSITION_CLASSES)], axis=1)


def metrics(logits, labels):
    correct = masked_argmax(logits) == labels
    non_wan = labels[:, 0] != 0
    first = float(correct[non_wan, 0].mean()) if non_wan.any() else float("nan")
    return {
        "per_position": [float(v) for v in correct.mean(axis=0)],
        "pos_mean": float(correct.mean()),
        "exact": float(correct.all(axis=1).mean()),
        "first_non_wan": first,
    }


def run_test(model, device, src, items, det_boxes, threads):
    n = len(items)
    true_logits = np.zeros((n, 7, NUM_CLASSES), np.float32)
    det_logits = np.zeros((n, 7, NUM_CLASSES), np.float32)
    missing = np.zeros(n, dtype=bool)

    def job(k):
        image = load_rgb(src / items[k]["file"])
        true = to_input(crop_array(image, expand_box(hull_of(items[k]), MARGIN, WIDTH, HEIGHT)))
        det = None
        if det_boxes[k] is not None:
            try:
                det = to_input(crop_array(image, expand_box(det_boxes[k], MARGIN, WIDTH, HEIGHT)))
            except ValueError:
                det = None
        return true, det

    started = time.time()
    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, n, 128):
            results = list(pool.map(job, range(start, min(start + 128, n))))
            batch = np.stack([r[0].transpose(2, 0, 1) for r in results])
            true_logits[start : start + len(results)] = predict(model, batch, device)
            have = [j for j, r in enumerate(results) if r[1] is not None]
            for j, r in enumerate(results):
                missing[start + j] = r[1] is None
            if have:
                batch = np.stack([results[j][1].transpose(2, 0, 1) for j in have])
                logits = predict(model, batch, device)
                for row, j in enumerate(have):
                    det_logits[start + j] = logits[row]
            done = start + len(results)
            if done % 1280 == 0 or done == n:
                print("  test %d/%d | %.0f s" % (done, n, time.time() - started), flush=True)
    return true_logits, det_logits, missing


def main():
    args = parse_args()
    out = Path(args.out_dir)
    pack = out.with_name(out.name + ".zip")
    for path in (out, pack):
        if path.exists():
            sys.exit(f"EXISTS: {path}")
    if args.epochs < 1 or args.batch < 2:
        sys.exit("--epochs must be >= 1 and --batch >= 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("cuda is not available")
    device = torch.device(args.device)
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    check_loss_mask()

    src = Path(args.src)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    predictions = load_predictions(args.predictions)
    if predictions.settings.get("split_sha256") != manifest["split_sha256"]:
        sys.exit("split_sha256 differs between predictions and manifest")
    parts = {name: [i for i in manifest["items"] if i["part"] == name] for name in EXPECTED}
    if args.limit:
        parts = {name: items[: args.limit] for name, items in parts.items()}
    else:
        for name, count in EXPECTED.items():
            if len(parts[name]) != count:
                sys.exit(f"{name}: expected {count} items, got {len(parts[name])}")
    train_items, val_items, test_items = parts["train"], parts["val"], parts["test"]
    labels = {}
    for name, items in parts.items():
        try:
            labels[name] = np.array([encode_plate(i["plate"]) for i in items], dtype=np.int64)
        except ValueError as error:
            sys.exit(f"{name}: {error}")

    first_chars = [i["plate"][0] for i in train_items]
    probs = draw_probabilities(first_chars)
    share = non_wan_share(probs, first_chars)
    print("train %d | val %d | test %d | non-wan share of draws %.4f (raw %.4f) | max draws per image %.2f"
          % (len(train_items), len(val_items), len(test_items), share,
             sum(c != "皖" for c in first_chars) / len(first_chars), probs.max() * len(first_chars)), flush=True)

    by_name = {image.name: image for image in predictions.images}
    missing_names = [i["file"] for i in test_items if i["file"] not in by_name]
    if missing_names:
        sys.exit(f"{len(missing_names)} test images have no prediction record, first {missing_names[0]}")
    det_boxes = []
    for item in test_items:
        found = by_name[item["file"]].detections
        det_boxes.append(max(found, key=lambda d: d.score).box if found else None)

    cache_started = time.time()
    val_cache = WindowCache(src, val_items, args.threads, "val")
    check_cache(val_cache, val_items, src, args.seed)
    val_inputs = np.empty((len(val_items), 3, OUT_H, OUT_W), np.float32)
    for i in range(len(val_items)):
        box = expand_box(val_cache.hulls[i], MARGIN, WIDTH, HEIGHT)
        val_inputs[i] = to_input(val_cache.crop(i, box)).transpose(2, 0, 1)
    del val_cache
    train_cache = WindowCache(src, train_items, args.threads, "train")
    check_cache(train_cache, train_items, src, args.seed)
    cache_seconds = time.time() - cache_started
    print("cache done in %.0f s | peak RSS of this process %.1f GB" % (cache_seconds, peak_rss_gb()), flush=True)

    out.mkdir(parents=True)
    trainset = TrainSet(train_cache, labels["train"], args.seed)
    n_train = len(trainset)
    drop_last = n_train >= args.batch
    steps = n_train // args.batch if drop_last else math.ceil(n_train / args.batch)
    model = PlateNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * steps
    warmup = min(steps, max(1, total_steps // 2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )
    weights = torch.from_numpy(probs)
    best, best_epoch = -1.0, 0
    best_path = out / "ocr_weights.pt"
    train_started = time.time()
    with open(out / "epochs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_pos_mean", "val_exact", "val_first_non_wan", "seconds"])
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            trainset.epoch = epoch
            sampler = WeightedRandomSampler(weights, num_samples=n_train, replacement=True,
                                            generator=torch.Generator().manual_seed(args.seed + epoch))
            loader = DataLoader(trainset, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                pin_memory=device.type == "cuda", drop_last=drop_last)
            model.train()
            total, seen = 0.0, 0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = ocr_loss(model(x), y)
                if not torch.isfinite(loss):
                    sys.exit(f"non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(y)
                seen += len(y)
            val = metrics(predict(model, val_inputs, device), labels["val"])
            if val["pos_mean"] > best:
                best, best_epoch = val["pos_mean"], epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch}, str(best_path) + ".tmp")
                os.replace(str(best_path) + ".tmp", best_path)
            seconds = time.time() - epoch_started
            writer.writerow([epoch, "%.5f" % (total / seen), "%.5f" % val["pos_mean"], "%.5f" % val["exact"],
                             "%.5f" % val["first_non_wan"], "%.1f" % seconds])
            handle.flush()
            print("epoch %d/%d loss %.4f | val pos-mean %.4f exact %.4f first(non-wan) %.4f | %.0f s%s"
                  % (epoch, args.epochs, total / seen, val["pos_mean"], val["exact"], val["first_non_wan"], seconds,
                     " *best" if epoch == best_epoch else ""), flush=True)
    train_seconds = time.time() - train_started
    del train_cache, trainset

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    val_logits = predict(model, val_inputs, device)
    val_final = metrics(val_logits, labels["val"])
    np.savez(out / "logits_val.npz", logits=val_logits, labels=labels["val"],
             files=np.array([i["file"] for i in val_items]))
    print("best epoch %d | val pos-mean %.4f exact %.4f" % (best_epoch, val_final["pos_mean"], val_final["exact"]), flush=True)

    test_started = time.time()
    true_logits, det_logits, det_missing = run_test(model, device, src, test_items, det_boxes, args.threads)
    files = np.array([i["file"] for i in test_items])
    subsets = np.array([i["subset"] for i in test_items])
    boxes = np.array([b.as_tuple() if b is not None else (np.nan,) * 4 for b in det_boxes], dtype=np.float64)
    np.savez(out / "logits_test_true.npz", logits=true_logits, files=files, subsets=subsets,
             missing=np.zeros(len(test_items), dtype=bool))
    np.savez(out / "logits_test_detected.npz", logits=det_logits, files=files, subsets=subsets,
             missing=det_missing, boxes=boxes)
    no_detection = sum(b is None for b in det_boxes)
    print("test done in %.0f s | no detection %d | missing after clipping %d"
          % (time.time() - test_started, no_detection, int(det_missing.sum()) - no_detection), flush=True)

    run = {
        "args": vars(args),
        "split_sha256": manifest["split_sha256"],
        "predictions_sha256": file_sha256(args.predictions),
        "counts": {name: len(items) for name, items in parts.items()},
        "non_wan_share_of_draws": share,
        "augmentation": ocr_augment.settings(),
        "crop": {"window": WINDOW, "margin": MARGIN, "shift": SHIFT, "scale": SCALE, "size": [OUT_H, OUT_W]},
        "best_epoch": best_epoch,
        "val_final": val_final,
        "test_no_detection": no_detection,
        "test_missing_detected": int(det_missing.sum()),
        "seconds": {"cache": cache_seconds, "train": train_seconds, "total": time.time() - started},
        "peak_rss_gb_main_process": peak_rss_gb(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__,
                     "python": platform.python_version()},
        "hashes": {name: file_sha256(out / name) for name in
                   ("ocr_weights.pt", "logits_val.npz", "logits_test_true.npz", "logits_test_detected.npz", "epochs.csv")},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    shutil.make_archive(str(out), "zip", root_dir=str(out.parent), base_dir=out.name)
    print("wrote", pack, "| sha256", file_sha256(pack), flush=True)


if __name__ == "__main__":
    main()
