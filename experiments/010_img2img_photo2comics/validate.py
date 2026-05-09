from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import IdentityPairedAugment, Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.metrics import ValidationMetrics
from src.img2img.render import save_training_panel
from src.utils.config import apply_yaml_config


def parse_args():
    p = argparse.ArgumentParser(description="Validate pixel-space img2img diffusion")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--panel-count", type=int, default=3)
    p.add_argument("--outdir", default="out/img2img_v1_val")
    p.add_argument("--use-ema", action="store_true")
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

    diffusion_cfg = DiffusionConfig(**ckpt.get("diffusion", {}))
    diffusion = GaussianImageDiffusion(diffusion_cfg, device)
    metrics_fn = ValidationMetrics(device)

    ds = PairedImageDataset(args.data_root, augment=IdentityPairedAugment(image_size=args.image_size), split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    losses = []
    ssim_vals = []
    lpips_vals = []
    panels_written = 0
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.max_batches:
            break
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        loss, t, x_t, _noise, _eps_hat, x0_hat, _dloss, _ploss = diffusion.training_loss(model, source, target)
        losses.append(float(loss.item()))
        metric_vals = metrics_fn.compute(x0_hat, target)
        ssim_vals.append(metric_vals["ssim"])
        lpips_vals.append(metric_vals["lpips"])

        if panels_written < args.panel_count:
            save_training_panel(source, target, x_t.clamp(0, 1), x0_hat, outdir / f"val_panel_{batch_idx:03d}.png")
            panels_written += 1

    metrics = {
        "checkpoint": args.checkpoint,
        "num_batches": len(losses),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "mean_ssim": sum(ssim_vals) / max(len(ssim_vals), 1),
        "mean_lpips": sum(lpips_vals) / max(len(lpips_vals), 1),
        "train_config": train_cfg,
    }
    with open(outdir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
