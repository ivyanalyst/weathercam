import argparse
import base64
import io
import json
import os
import sys

import numpy as np

BYTES_PER_MB = 1024 * 1024


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-limit-mb", type=float, default=3000)
    return ap.parse_args()


def rss_mb():
    with open("/proc/self/statm") as handle:
        resident = int(handle.read().split()[1])
    return resident * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MB


def send(stream, payload):
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def pack(array):
    data = np.ascontiguousarray(array, dtype="<f4")
    encoded = base64.b64encode(data.tobytes()).decode("ascii")
    return {"shape": list(data.shape), "data": encoded}


def require_list(value):
    if not isinstance(value, list) or not value:
        raise ValueError("request needs a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("every entry must be a string")
    return value


def main():
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.memory_limit_mb <= 0:
        raise SystemExit("batch size, threads and memory limit must be positive")
    real = sys.stdout
    sys.stdout = sys.stderr
    import open_clip
    import torch
    from PIL import Image

    torch.set_num_threads(args.threads)
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    size = args.batch_size

    def embed(features):
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def encode_text(texts):
        parts = []
        with torch.no_grad():
            for start in range(0, len(texts), size):
                tokens = tokenizer(texts[start : start + size])
                parts.append(embed(model.encode_text(tokens)))
        return np.concatenate(parts)

    def encode_images(blobs):
        parts = []
        with torch.no_grad():
            for start in range(0, len(blobs), size):
                tensors = []
                for text in blobs[start : start + size]:
                    raw = base64.b64decode(text, validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        tensors.append(transform(image.convert("RGB")))
                parts.append(embed(model.encode_image(torch.stack(tensors))))
        return np.concatenate(parts)

    dim = encode_text(["a"]).shape[1]
    send(real, {"ready": True, "model": args.model, "pretrained": args.pretrained, "dim": dim})
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            used = rss_mb()
            if used > args.memory_limit_mb:
                raise MemoryError("rss %.0f MB exceeds limit %.0f MB" % (used, args.memory_limit_mb))
            op = request.get("op")
            if op == "text":
                array = encode_text(require_list(request.get("texts")))
            elif op == "image":
                array = encode_images(require_list(request.get("images")))
            else:
                raise ValueError("unknown op")
            reply = {"id": request_id, "ok": True}
            reply.update(pack(array))
        except Exception as error:
            reply = {"id": request_id, "ok": False, "error": "%s: %s" % (type(error).__name__, error)}
        send(real, reply)


if __name__ == "__main__":
    main()
