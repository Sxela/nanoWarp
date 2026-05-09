from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import EMA, Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.render import save_training_panel
from src.toy_diffusion.render import save_loss_plot


def parse_args():
    p = argparse.ArgumentParser(description="Train pixel-space img2img diffusion for photo->comics")
    p.add_argument("data_root")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--panel-every", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--outdir", default="out/img2img_v1")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = PairedImageDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    it = cycle(dl)

    model = Img2ImgDiffusionUNet(pretrained_source_encoder=not args.no_pretrained).to(device)
    ema = EMA(model, decay=args.ema_decay)
    diffusion = GaussianImageDiffusion(DiffusionConfig(), device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        batch = next(it)
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        loss, t, x_t, _noise, _eps_hat, x0_hat = diffusion.training_loss(model, source, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.6f}")
            save_loss_plot(losses, outdir / "loss.png")
        if step % args.panel_every == 0 or step == args.steps:
            save_training_panel(source, target, x_t.clamp(0, 1), x0_hat, outdir / f"panel_step_{step:06d}.png")

    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.model.state_dict(),
            "config": vars(args),
            "diffusion": diffusion.config.__dict__,
        },
        outdir / "model.pt",
    )
    with open(outdir / "metrics.json", "w") as f:
        json.dump({"final_loss": losses[-1], "mean_loss_last_50": sum(losses[-50:]) / min(50, len(losses))}, f, indent=2)
    print(f"saved checkpoint to {outdir / 'model.pt'}")


if __name__ == "__main__":
    main()
