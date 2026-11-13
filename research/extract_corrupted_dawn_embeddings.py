import argparse
import hashlib
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image

from weathercam.attributes.weather import load_weather
from weathercam.data.augment import add_fog, add_haze, add_rain, add_snow
from weathercam.data.corrupt import darken, gaussian_blur, gaussian_noise, jpeg
from weathercam.data.imagecodec import load_rgb

SEED = 42
SEVERITIES = (0.3, 0.6)
CORRUPTIONS = ("blur", "noise", "darken", "jpeg", "fog", "haze", "rain", "snow")
FUNCTIONS = {
    "blur": gaussian_blur,
    "noise": gaussian_noise,
    "darken": darken,
    "jpeg": jpeg,
    "fog": add_fog,
    "haze": add_haze,
    "rain": add_rain,
    "snow": add_snow,
}
NEEDS_RNG = ("noise", "fog", "haze", "rain", "snow")
EVAL_PARTS = ("val", "test")
MEMORY_LIMIT_MB = 3000
MIN_COSINE = 0.9999
MAX_ABS_DIFF = 1e-4
TILE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dawn-dir", default="data/DAWN")
    p.add_argument("--split", default="outputs/dawn_split/split_v1.json")
    p.add_argument("--clean", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--expect-clean-sha256", required=True)
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--preview-dir", default="outputs/robustness/preview")
    return p.parse_args()


def conditions():
    items = [("clean", 0.0)]
    for name in CORRUPTIONS:
        for severity in SEVERITIES:
            items.append((name, severity))
    return items


def label(name, severity):
    return name if name == "clean" else "%s_%.1f" % (name, severity)


def apply(name, severity, image, seed_parts):
    if name == "clean":
        return image
    fn = FUNCTIONS[name]
    if name in NEEDS_RNG:
        return fn(image, severity, np.random.default_rng(seed_parts))
    return fn(image, severity)


def to_uint8(image):
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def select_rows(data, limit):
    rows = [i for i, part in enumerate(data.split) if str(part) in EVAL_PARTS]
    return rows[:limit] if limit else rows


def check_memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if peak > MEMORY_LIMIT_MB:
        raise SystemExit("peak memory %.0f MB exceeds the limit" % peak)
    return peak


def tile(pixels):
    image = Image.fromarray(pixels)
    width, height = image.size
    scale = TILE / min(width, height)
    image = image.resize((max(TILE, round(width * scale)), max(TILE, round(height * scale))))
    width, height = image.size
    left = (width - TILE) // 2
    top = (height - TILE) // 2
    return np.asarray(image.crop((left, top, left + TILE, top + TILE)))


def preview(data, rows, args):
    out_dir = Path(args.preview_dir)
    targets = []
    for weather_id in range(len(data.weathers)):
        for position, row in enumerate(rows):
            if str(data.split[row]) == "val" and int(data.labels[row]) == weather_id:
                targets.append((position, row))
                break
    if not targets:
        raise SystemExit("no validation rows to preview")
    paths = [out_dir / ("sheet_%.1f.png" % severity) for severity in SEVERITIES]
    if any(path.exists() for path in paths):
        raise SystemExit("refusing to overwrite preview sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_list = conditions()
    for severity, path in zip(SEVERITIES, paths):
        columns = [(i, n, s) for i, (n, s) in enumerate(cond_list) if n == "clean" or s == severity]
        grid_rows = []
        for position, row in targets:
            image = load_rgb(Path(args.dawn_dir) / data.image_names[row])
            tiles = [tile(to_uint8(apply(n, s, image, [SEED, position, i]))) for i, n, s in columns]
            grid_rows.append(np.concatenate(tiles, axis=1))
        Image.fromarray(np.concatenate(grid_rows, axis=0)).save(path)
        print("wrote", path, "columns", ", ".join(label(n, s) for _, n, s in columns))
    print("rows top to bottom:", ", ".join(data.image_names[row] for _, row in targets))


def verify_clean(embeddings, reference):
    a = embeddings.astype(np.float64)
    b = reference.astype(np.float64)
    cosine = float((a * b).sum(axis=1).min())
    diff = float(np.abs(a - b).max())
    print("clean vs stored: min cosine %.7f, max abs diff %.2e" % (cosine, diff), flush=True)
    if cosine < MIN_COSINE or diff > MAX_ABS_DIFF:
        raise SystemExit("clean path does not reproduce the stored embeddings; stopped")


def embed_all(args, data, rows, cond_list):
    import open_clip
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __init__(self, paths, name, severity, cond_index, transform):
            self.paths = paths
            self.name = name
            self.severity = severity
            self.cond_index = cond_index
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, position):
            image = load_rgb(self.paths[position])
            image = apply(self.name, self.severity, image, [SEED, position, self.cond_index])
            return self.transform(Image.fromarray(to_uint8(image)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    print("model", args.model, "pretrained", args.pretrained, "device", device, flush=True)
    paths = [Path(args.dawn_dir) / data.image_names[row] for row in rows]
    reference = data.embeddings[rows]
    outputs = []
    for index, (name, severity) in enumerate(cond_list):
        start = time.perf_counter()
        loader = DataLoader(
            Rows(paths, name, severity, index, transform),
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        parts = []
        with torch.no_grad():
            for batch in loader:
                features = model.encode_image(batch.to(device)).float()
                features = features / features.norm(dim=-1, keepdim=True)
                parts.append(features.cpu().numpy())
        embeddings = np.concatenate(parts).astype(np.float32)
        outputs.append(embeddings)
        print(
            "condition %d/%d %s: %d images in %.1fs, peak %.0f MB"
            % (index + 1, len(cond_list), label(name, severity), len(embeddings), time.perf_counter() - start, check_memory()),
            flush=True,
        )
        if index == 0:
            verify_clean(embeddings, reference)
    return np.stack(outputs)


def main():
    args = parse_args()
    if not args.preview and args.out is None:
        raise SystemExit("--out is required unless --preview")
    out = Path(args.out) if args.out else None
    if out is not None and out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    clean_sha = hashlib.sha256(Path(args.clean).read_bytes()).hexdigest()
    if clean_sha != args.expect_clean_sha256:
        raise SystemExit("clean embeddings hash differs from the expected value")
    split_sha = hashlib.sha256(Path(args.split).read_bytes()).hexdigest()
    with np.load(args.clean, allow_pickle=False) as stored:
        stored_split_sha = str(stored["split_sha256"])
    print("split file sha256 %s" % split_sha)
    print("stored split sha256 %s (%s)" % (stored_split_sha, "same" if stored_split_sha == split_sha else "DIFFERENT"))
    data = load_weather(args.clean)
    rows = select_rows(data, args.limit)
    cond_list = conditions()
    print("rows %d conditions %d" % (len(rows), len(cond_list)), flush=True)
    if args.preview:
        preview(data, rows, args)
        return
    start = time.perf_counter()
    embeddings = embed_all(args, data, rows, cond_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        rows=np.array(rows, dtype=np.int64),
        image_names=np.array([data.image_names[row] for row in rows]),
        split=data.split[rows],
        weather_ids=data.labels[rows],
        weathers=np.array(data.weathers),
        conditions=np.array([label(n, s) for n, s in cond_list]),
        corruptions=np.array([n for n, _ in cond_list]),
        severities=np.array([s for _, s in cond_list], dtype=np.float64),
        seed=np.array(SEED),
        model=np.array(args.model),
        pretrained=np.array(args.pretrained),
        clean_embeddings_sha256=np.array(clean_sha),
        split_sha256=np.array(stored_split_sha),
    )
    print("done in %.1fs, wrote %s" % (time.perf_counter() - start, out))


if __name__ == "__main__":
    main()
