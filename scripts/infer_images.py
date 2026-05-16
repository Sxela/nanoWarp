"""Run inference on a list of image paths and save an input/output panel.

Reuses the arch auto-detection from infer_video._build_exp25 so any
single-frame checkpoint trained via train.py or train_exp32_prog512.py
(exp25/32/33+) works without per-experiment branching.

Output: one PNG panel per checkpoint, named after the checkpoint stem.
Layout: rows = one per input image, columns = [source | output].

Usage:
    PYTHONPATH=. python3 scripts/infer_images.py \\
        --checkpoint out/exp35_pyramid_at_exp37_recipe_*/exp35_model.pt \\
        --images img1.jpg img2.png img3.jpg \\
        --max-size 512 --sample-steps 20 \\
        --outdir out/infer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# Make src.* and infer_video importable.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "010_img2img_photo2comics"))

from infer_video import _build_exp25  # type: ignore
from src.img2img.render import save_val_panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a single-frame model checkpoint (.pt). Arch is auto-detected.")
    p.add_argument("--images", nargs="+", required=True,
                   help="One or more input image paths.")
    p.add_argument("--max-size", type=int, default=512,
                   help="Square resize size before inference (default 512).")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--outdir", default="out/infer")
    return p.parse_args()


def load_and_resize(paths, size, device):
    """Load PIL images, resize each to a `size x size` square, return (B, 3, size, size)."""
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
        out.append(TF.to_tensor(img))
    return torch.stack(out).to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return

    # Panel filename: ckpt-stem with spaces stripped.
    tag = ckpt_path.stem.replace(" ", "_")

    print(f"[load] {len(args.images)} images @ {args.max_size}px")
    source = load_and_resize(args.images, args.max_size, device)

    print(f"[model] building from {ckpt_path}")
    model, diffusion = _build_exp25(str(ckpt_path), device)

    with torch.no_grad():
        samples, _ = diffusion.sample(
            model, source,
            image_size=args.max_size,
            sample_steps=args.sample_steps,
        )

    # save_val_panel takes (source, target, sampled, sampled_alt). Pass source
    # twice on the "target" side so the panel reads as [src | src | out | out];
    # cleaner: write a minimal custom panel with just two columns.
    panel = _make_two_col_panel(source.cpu(), samples.cpu(), args.images)
    panel_path = outdir / f"{tag}.png"
    panel.save(panel_path)
    print(f"[panel] saved {panel_path}")


def _make_two_col_panel(source: torch.Tensor, samples: torch.Tensor, paths: list[str]):
    """Two-column [source | output] grid, one row per input image, labelled."""
    from PIL import ImageDraw, ImageFont
    n = source.shape[0]
    h = source.shape[-2]
    w = source.shape[-1]
    label_h = 20
    cell_h = h + label_h
    panel_w = w * 2
    panel_h = cell_h * n
    panel = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw = ImageDraw.Draw(panel)
    for i in range(n):
        y0 = i * cell_h
        # Top label row
        name = Path(paths[i]).name
        draw.rectangle([(0, y0), (panel_w, y0 + label_h)], fill=(220, 220, 220))
        if font is not None:
            draw.text((4, y0 + 4), f"{name}  (source | output)", fill=(20, 20, 20), font=font)
        # Source on left, output on right
        src_img = TF.to_pil_image(source[i].clamp(0, 1))
        out_img = TF.to_pil_image(samples[i].clamp(0, 1))
        panel.paste(src_img, (0, y0 + label_h))
        panel.paste(out_img, (w, y0 + label_h))
    return panel


if __name__ == "__main__":
    main()
