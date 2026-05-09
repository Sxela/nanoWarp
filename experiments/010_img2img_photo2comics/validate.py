from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.render import save_training_panel


def parse_args():
    p = argparse.ArgumentParser(description="Validate pixel-space img2img diffusion")
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--panel-count", type=int, default=3)
    p.add_argument("--outdir", default="out/img2img_v1_val")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_cfg = ckpt.get("config", {})
    model = Img2ImgDiffusionUNet(pretrained_source_encoder=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion_cfg = DiffusionConfig(**ckpt.get("diffusion", {}))
    diffusion = GaussianImageDiffusion(diffusion_cfg, device)

    ds = PairedImageDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    losses = []
    panels_written = 0
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.max_batches:
            break
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        loss, t, x_t, _noise, _eps_hat, x0_hat = diffusion.training_loss(model, source, target)
        losses.append(float(loss.item()))

        if panels_written < args.panel_count:
            save_training_panel(source, target, x_t.clamp(0, 1), x0_hat, outdir / f"val_panel_{batch_idx:03d}.png")
            panels_written += 1

    metrics = {
        "checkpoint": args.checkpoint,
        "num_batches": len(losses),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "train_config": train_cfg,
    }
    with open(outdir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

