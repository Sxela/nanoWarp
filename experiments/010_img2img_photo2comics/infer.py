from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.render import save_inference_panel, save_progress_strip
from src.utils.config import apply_yaml_config


def build_method_from_ckpt(ckpt: dict, device: torch.device):
    method = ckpt.get("method", "diffusion")
    cfg_dict = ckpt.get("diffusion", {})
    if method == "flow":
        cfg = FlowConfig(**cfg_dict)
        return RectifiedImageFlow(cfg, device), cfg, method
    cfg = DiffusionConfig(**cfg_dict)
    return GaussianImageDiffusion(cfg, device), cfg, method


def parse_args():
    p = argparse.ArgumentParser(description="Run reverse-diffusion inference for img2img")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--outdir", default="out/img2img_v1_infer")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--limit-batches", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_cfg = ckpt.get("config", {})
    model = Img2ImgDiffusionUNet(
        pretrained_source_encoder=False,
        source_in_stem=train_cfg.get("source_in_stem", False),
    ).to(device)
    state_key = "ema_model" if args.use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    diffusion, method_cfg, method = build_method_from_ckpt(ckpt, device)
    print(f"loaded method={method} method_cfg={method_cfg.__dict__}")

    ds = PairedImageDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    image_size = train_cfg.get("image_size", 128)

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.limit_batches:
            break
        source = batch["source"].to(device)
        samples, frames = diffusion.sample(
            model,
            source,
            image_size=image_size,
            sample_steps=args.sample_steps,
            log_every=args.progress_every,
        )
        save_inference_panel(source, samples, outdir / f"infer_panel_{batch_idx:03d}.png")
        save_progress_strip(frames, outdir / f"infer_progress_{batch_idx:03d}.png")
        print(f"saved inference artifacts for batch {batch_idx} to {outdir} using {args.sample_steps} sampling steps")


if __name__ == "__main__":
    main()
