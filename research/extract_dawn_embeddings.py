import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PARTS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dawn-dir", default="data/DAWN")
    p.add_argument("--split", default="outputs/dawn_split/split_v1.json")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--out", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    split_bytes = Path(args.split).read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    data = json.loads(split_bytes)
    names, split = [], []
    for part in PARTS:
        part_paths = data["parts"][part]["paths"]
        names += part_paths
        split += [part] * len(part_paths)
    if len(set(names)) != len(names):
        raise SystemExit("duplicate paths in split file")
    weathers = sorted({Path(n).parts[0] for n in names})
    weather_ids = [weathers.index(Path(n).parts[0]) for n in names]
    print("split sha256", split_sha)
    print("images", len(names), "classes", weathers)

    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class Images(Dataset):
        def __init__(self, folder, rels, transform):
            self.folder = folder
            self.rels = rels
            self.transform = transform

        def __len__(self):
            return len(self.rels)

        def __getitem__(self, i):
            with Image.open(self.folder / self.rels[i]) as img:
                return self.transform(img.convert("RGB"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    print("model", args.model, "pretrained", args.pretrained, "device", device)
    loader = DataLoader(
        Images(Path(args.dawn_dir), names, transform),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    chunks = []
    start = time.perf_counter()
    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            features = model.encode_image(batch.to(device)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy())
            if number % 5 == 0:
                done = sum(len(c) for c in chunks)
                print("images %d/%d elapsed %.0fs" % (done, len(names), time.perf_counter() - start), flush=True)
    embeddings = np.concatenate(chunks).astype(np.float32)
    print("done %d images in %.1fs" % (len(embeddings), time.perf_counter() - start))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        image_names=np.array(names),
        split=np.array(split),
        weather_ids=np.array(weather_ids, dtype=np.int64),
        weathers=np.array(weathers),
        model=np.array(args.model),
        pretrained=np.array(args.pretrained),
        split_sha256=np.array(split_sha),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
