import argparse
from pathlib import Path

from weathercam.data.dawn import load_annotation
from weathercam.data.imagecodec import load_rgb, save_png
from weathercam.data.overlay import draw_boxes

RED = (1.0, 0.0, 0.0)


def image_for(xml_path):
    return xml_path.parent.parent / (xml_path.stem + ".jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/previews"))
    parser.add_argument("--thickness", type=int, default=1)
    args = parser.parse_args()
    for xml_path in args.xml:
        annotation = load_annotation(xml_path)
        image = load_rgb(image_for(xml_path))
        height, width = image.shape[:2]
        if (width, height) != (annotation.width, annotation.height):
            raise SystemExit(
                f"{xml_path.stem}: xml {annotation.width}x{annotation.height} != image {width}x{height}"
            )
        boxes = [item.box for item in annotation.objects]
        out_path = args.out / f"{xml_path.stem}_boxes.png"
        save_png(out_path, draw_boxes(image, boxes, RED, args.thickness))
        print(f"{xml_path.stem} {width}x{height} boxes={len(boxes)} -> {out_path}")
        for item in annotation.objects:
            print(f"  {item.name} {item.box.as_tuple()}")


if __name__ == "__main__":
    main()
