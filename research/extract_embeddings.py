import argparse
import hashlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--veri-root", required=True)
    p.add_argument("--src", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--prefetch-threads", type=int, default=32)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def start_prefetch(paths, threads):
    if threads <= 0:
        return None

    def read(path):
        try:
            with open(path, "rb") as handle:
                handle.read()
        except OSError:
            pass

    def run():
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read, paths))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def main():
    args = parse_args()
    if args.src:
        sys.path.insert(0, args.src)
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from weathercam.data.attrlabels import columns_of
    from weathercam.data.veri import parse_labels, unlabeled_count

    class Images(Dataset):
        def __init__(self, folder, names, transform):
            self.folder = folder
            self.names = names
            self.transform = transform

        def __len__(self):
            return len(self.names)

        def __getitem__(self, i):
            with Image.open(self.folder / self.names[i]) as img:
                return self.transform(img.convert("RGB"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    print("model", args.model, "pretrained", args.pretrained, "device", device)

    root = Path(args.veri_root)
    parts = {"embeddings": [], "image_names": [], "vehicle_ids": [], "camera_ids": [], "color_ids": [], "type_ids": [], "split": []}
    run_start = time.perf_counter()
    for split in args.splits:
        labels = parse_labels((root / f"{split}_label.xml").read_bytes())
        columns = columns_of(labels)
        names = list(columns.image_names)
        print(split, "embedded", len(names), "skipped_unlabelled", unlabeled_count(labels))
        folder = root / f"image_{split}"
        start_prefetch([folder / name for name in names], args.prefetch_threads)
        loader = DataLoader(Images(folder, names, transform), batch_size=args.batch_size, num_workers=args.workers)
        chunks = []
        done = 0
        split_start = time.perf_counter()
        with torch.no_grad():
            for number, batch in enumerate(loader, 1):
                features = model.encode_image(batch.to(device)).float()
                features = features / features.norm(dim=-1, keepdim=True)
                chunks.append(features.cpu().numpy())
                done += len(batch)
                if args.log_every and number % args.log_every == 0:
                    elapsed = time.perf_counter() - split_start
                    print("%s batch %d images %d/%d %.1f img/s elapsed %.0fs" % (split, number, done, len(names), done / elapsed, elapsed), flush=True)
        elapsed = time.perf_counter() - split_start
        print("%s done %d images in %.1fs (%.1f img/s)" % (split, done, elapsed, done / elapsed), flush=True)
        parts["embeddings"].append(np.concatenate(chunks).astype(np.float32))
        parts["image_names"] += names
        parts["vehicle_ids"] += list(columns.vehicle_ids)
        parts["camera_ids"] += list(columns.camera_ids)
        parts["color_ids"] += list(columns.color_ids)
        parts["type_ids"] += list(columns.type_ids)
        parts["split"] += [split] * len(names)
    print("total extraction %.1fs" % (time.perf_counter() - run_start))

    np.savez_compressed(
        args.out,
        embeddings=np.concatenate(parts["embeddings"]),
        image_names=np.array(parts["image_names"]),
        vehicle_ids=np.array(parts["vehicle_ids"]),
        camera_ids=np.array(parts["camera_ids"]),
        color_ids=np.array(parts["color_ids"], dtype=np.int64),
        type_ids=np.array(parts["type_ids"], dtype=np.int64),
        split=np.array(parts["split"]),
    )
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print("saved", args.out, "sha256", digest)


if __name__ == "__main__":
    main()
