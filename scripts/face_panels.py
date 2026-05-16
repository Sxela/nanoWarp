"""Post-hoc face-panel generator.

Given one or more single-frame checkpoint paths and a set of val stems, run
inference on the same pinned val pairs and save a side-by-side panel
(source | target | output per row, one row per stem) for each checkpoint.

Reuses the arch auto-detection from infer_video._build_exp25 so it works for
any checkpoint trained via train.py or train_exp32_prog512.py (exp25/32/33+).

Usage:
    PYTHONPATH=. python3 scripts/face_panels.py \\
        --data-root data/photo2anime_1k \\
        --stems 000942 000943 000921 \\
        --image-size 256 \\
        --outdir out/face_panels_compare \\
        --checkpoints  PATH1.pt  PATH2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# Add repo root to sys.path so `src.*` resolves when invoked from anywhere.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Add experiments/.../ so `from infer_video import ...` works.
sys.path.insert(0, str(REPO / "experiments" / "010_img2img_photo2comics"))

from infer_video import _build_exp25  # type: ignore
from src.img2img.render import save_val_panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths.")
    p.add_argument("--data-root", default="data/photo2anime_1k")
    p.add_argument("--split", default="val")
    p.add_argument("--stems", nargs="+", default=["000942", "000943", "000921"],
                   help="Val image stems to draw into the panel.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--outdir", default="out/face_panels")
    return p.parse_args()


def load_val_pairs(data_root: Path, split: str, stems: list[str]):
    base = data_root / split
    src_dir, tgt_dir = base / "source", base / "target"
    out = {}
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            src = src_dir / f"{stem}{ext}"
            tgt = tgt_dir / f"{stem}{ext}"
            if src.exists() and tgt.exists():
                out[stem] = (Image.open(src).convert("RGB"), Image.open(tgt).convert("RGB"))
                break
        else:
            print(f"[warn] stem {stem} not found in {src_dir}")
    return out


def to_tensor_batch(pils, size, device):
    out = []
    for img in pils:
        r = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
        out.append(TF.to_tensor(r))
    return torch.stack(out).to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_val_pairs(Path(args.data_root), args.split, args.stems)
    if not pairs:
        print("[error] no val pairs found")
        return
    stems = list(pairs.keys())
    sources_pil = [pairs[s][0] for s in stems]
    targets_pil = [pairs[s][1] for s in stems]
    source = to_tensor_batch(sources_pil, args.image_size, device)
    target = to_tensor_batch(targets_pil, args.image_size, device)
    print(f"[load] {len(stems)} pairs: {stems}")

    for ckpt_path_s in args.checkpoints:
        ckpt_path = Path(ckpt_path_s)
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path} not found")
            continue

        # Name panel after parent dir's first underscore token, or file stem.
        parent = ckpt_path.parent.name
        tag = parent.split("_")[0] if parent else ckpt_path.stem.replace(" ", "_")
        # If parent is just "Downloads" or similar, use the file stem instead.
        if tag.lower() in {"downloads", "out", "tmp", "."}:
            tag = ckpt_path.stem.replace(" ", "_")
        print(f"[panel] {ckpt_path}  ->  {tag}_face_panel.png")

        model, diffusion = _build_exp25(str(ckpt_path), device)
        with torch.no_grad():
            samples, _ = diffusion.sample(
                model, source,
                image_size=args.image_size,
                sample_steps=args.sample_steps,
            )

        panel_path = outdir / f"{tag}_face_panel.png"
        save_val_panel(source.cpu(), target.cpu(), samples.cpu(), samples.cpu(), panel_path)
        del model, diffusion
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
