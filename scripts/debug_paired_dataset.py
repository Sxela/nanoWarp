from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from src.img2img.dataset import AugmentConfig, PairedImageAugment, PairedImageDataset


def tensor_to_pil(x):
    arr = (x.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy())
    return Image.fromarray(arr)


def add_label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 24), "white")
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), text, fill=(20, 20, 20))
    return canvas


def make_row(sample) -> Image.Image:
    imgs = [
        add_label(tensor_to_pil(sample["source_geom"]), "source_geom"),
        add_label(tensor_to_pil(sample["source"]), "source_aug"),
        add_label(tensor_to_pil(sample["target"]), "target"),
    ]
    w = sum(i.width for i in imgs)
    h = max(i.height for i in imgs)
    row = Image.new("RGB", (w, h), "#f4f4f4")
    x = 0
    for img in imgs:
        row.paste(img, (x, 0))
        x += img.width
    return row


def main():
    p = argparse.ArgumentParser(description="Visual debug for paired img2img dataset")
    p.add_argument("root")
    p.add_argument("--output", default="out/paired_dataset_debug.png")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args()

    ds = PairedImageDataset(args.root, augment=PairedImageAugment(AugmentConfig(image_size=args.image_size)))
    rows = [make_row(ds[i]) for i in range(min(args.limit, len(ds)))]
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"saved dataset debug grid to {out}")


if __name__ == "__main__":
    main()

