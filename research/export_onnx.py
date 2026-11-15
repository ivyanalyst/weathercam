import argparse
import hashlib
import os
import resource
import time
from pathlib import Path

BYTES_PER_MB = 1024 * 1024
OUT_H, OUT_W = 96, 192
NUM_CLASSES = 35
OCR_SHA256 = "3dd0c9d2110c79a77e05ac521815449a35eb120481e02b242c60a1f2a58f7a57"
CLIP_FILE = "clip_vitb16_laion2b_visual.onnx"
PLATE_FILE = "plate_ocr.onnx"
TIMING_RUNS = 3
SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--ocr-sha256", default=OCR_SHA256)
    ap.add_argument("--out-dir", default="outputs/onnx")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--plate-batch", type=int, default=64)
    ap.add_argument("--memory-limit-mb", type=float, default=3000)
    return ap.parse_args()


def rss_mb():
    with open("/proc/self/statm") as handle:
        resident = int(handle.read().split()[1])
    return resident * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MB


def check_rss(limit):
    used = rss_mb()
    if used > limit:
        raise SystemExit("rss %.0f MB exceeds limit %.0f MB" % (used, limit))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    args = parse_args()
    if args.threads < 1 or args.clip_batch < 2 or args.plate_batch < 2 or args.memory_limit_mb <= 0:
        raise SystemExit("threads, memory limit and batch sizes (at least 2) must be valid")
    out_dir = Path(args.out_dir)
    targets = [out_dir / CLIP_FILE, out_dir / PLATE_FILE]
    for target in targets:
        if out_dir.exists() and any(out_dir.glob(target.name + "*")):
            raise SystemExit("EXISTS: %s" % target)
    if file_sha256(args.ocr) != args.ocr_sha256:
        raise SystemExit("OCR weights hash differs from the expected value")
    import open_clip
    import torch
    from torch import nn
    from torch.export import Dim

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)

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

    class ImageTower(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.encode_image(x)

    def export(module, dummy, path, input_name, output_name):
        batch = Dim("batch")
        torch.onnx.export(
            module,
            (dummy,),
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes={"x": {0: batch}},
            dynamo=True,
        )

    def timing(module, dummy):
        runs = []
        with torch.no_grad():
            module(dummy)
            for _ in range(TIMING_RUNS):
                started = time.perf_counter()
                module(dummy)
                runs.append(round(time.perf_counter() - started, 3))
        return runs

    out_dir.mkdir(parents=True, exist_ok=True)
    check_rss(args.memory_limit_mb)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tower = ImageTower(model.eval()).eval()
    clip_dummy = torch.randn(args.clip_batch, 3, *model.visual.image_size)
    print("clip torch seconds", timing(tower, clip_dummy), flush=True)
    export(tower, clip_dummy, targets[0], "image", "features")
    check_rss(args.memory_limit_mb)
    net = PlateNet()
    net.load_state_dict(torch.load(args.ocr, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    net.eval()
    plate_dummy = torch.rand(args.plate_batch, 3, OUT_H, OUT_W)
    print("plate torch seconds", timing(net, plate_dummy), flush=True)
    export(net, plate_dummy, targets[1], "crop", "logits")
    for target in targets:
        for path in sorted(out_dir.glob(target.name + "*")):
            print(path, path.stat().st_size, file_sha256(path), flush=True)
    print("peak RSS MB", round(peak_mb()), flush=True)


if __name__ == "__main__":
    main()
