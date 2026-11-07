import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

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

    root = Path(args.subset)
    manifest = json.loads((root / "manifest.json").read_text())
    items = manifest["items"]
    paths = [item["path"] for item in items]
    files = ["%06d%s" % (k, Path(path).suffix.lower()) for k, path in enumerate(paths)] if args.flat else paths
    print("items", len(items), dict(collections.Counter(item["split"] for item in items)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    print("model", args.model, "pretrained", args.pretrained, "device", device)

    loader = DataLoader(Images(root / "images", files, transform), batch_size=args.batch_size, num_workers=args.workers)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            features = model.encode_image(batch.to(device)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    assert embeddings.shape[0] == len(items), (embeddings.shape, len(items))

    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        paths=np.array(paths),
        make_ids=np.array([item["make_id"] for item in items], dtype=np.int64),
        groups=np.array([item["group"] for item in items], dtype=np.int64),
        split=np.array([item["split"] for item in items]),
        makes=np.array(manifest["makes"]),
    )
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print("saved", args.out, embeddings.shape, "sha256", digest)


if __name__ == "__main__":
    main()
