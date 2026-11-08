import argparse

import numpy as np
import open_clip
import torch

CLASSES = ["front", "rear", "side"]
PROMPTS = [
    [
        "a photo of the front of a car",
        "a photo of a car seen from the front, with headlights and grille visible",
        "a front view of a vehicle",
    ],
    [
        "a photo of the back of a car",
        "a photo of a car seen from behind, with taillights visible",
        "a rear view of a vehicle",
    ],
    [
        "a photo of the side of a car",
        "a photo of a car seen from the side, with wheels and doors visible",
        "a side view of a vehicle",
    ],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    flat = [p for group in PROMPTS for p in group]
    with torch.no_grad():
        feats = model.encode_text(tokenizer(flat)).float().numpy()
    feats = feats.reshape(len(PROMPTS), len(PROMPTS[0]), -1).astype(np.float32)
    np.savez(args.out, text=feats, classes=np.array(CLASSES), prompts=np.array(PROMPTS))
    print(feats.shape, args.model, args.pretrained)


if __name__ == "__main__":
    main()
