from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.img2img import EMA, Img2ImgDiffusionUNet, build_train_val_datasets
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.render import save_training_panel
from src.toy_diffusion.render import save_loss_plot
from src.utils.config import apply_yaml_config


def parse_args():
    p = argparse.ArgumentParser(description="Train pixel-space img2img diffusion for photo->comics")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--val-root", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--panel-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--outdir", default="out/img2img_v1")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--source-in-stem", action="store_true")
    p.add_argument("--lpips-weight", type=float, default=0.0)
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

    train_ds, val_ds = build_train_val_datasets(
        train_root=args.data_root,
        val_root=args.val_root,
        image_size=args.image_size,
        train_split=args.train_split,
        val_split=args.val_split,
    )
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    it = cycle(dl)
    val_it = cycle(val_dl)

    model = Img2ImgDiffusionUNet(
        pretrained_source_encoder=not args.no_pretrained,
        source_in_stem=args.source_in_stem,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)
    diffusion = GaussianImageDiffusion(DiffusionConfig(), device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    aux_lpips = None
    if args.lpips_weight > 0:
        aux_lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)

    losses: list[float] = []
    val_history: list[float] = []
    for step in range(1, args.steps + 1):
        batch = next(it)
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        loss, t, x_t, _noise, _eps_hat, x0_hat, diffusion_loss, lpips_loss = diffusion.training_loss(
            model,
            source,
            target,
            aux_lpips=aux_lpips,
            aux_lpips_weight=args.lpips_weight,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:5d} | loss {loss.item():.6f} | diffusion {float(diffusion_loss):.6f} | lpips {float(lpips_loss):.6f}"
            )
            save_loss_plot(losses, outdir / "loss.png")
        if step % args.panel_every == 0 or step == args.steps:
            save_training_panel(source, target, x_t.clamp(0, 1), x0_hat, outdir / f"panel_step_{step:06d}.png")
        if step % args.val_every == 0 or step == args.steps:
            val_losses = []
            for _ in range(args.val_batches):
                vbatch = next(val_it)
                vsource = vbatch["source"].to(device)
                vtarget = vbatch["target"].to(device)
                vloss, *_ = diffusion.training_loss(model, vsource, vtarget)
                val_losses.append(float(vloss.item()))
            mean_val = sum(val_losses) / len(val_losses)
            val_history.append(mean_val)
            print(f"val step {step:5d} | mean_loss {mean_val:.6f}")

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
        json.dump({
            "final_loss": losses[-1],
            "mean_loss_last_50": sum(losses[-50:]) / min(50, len(losses)),
            "last_val_loss": val_history[-1] if val_history else None,
        }, f, indent=2)
    print(f"saved checkpoint to {outdir / 'model.pt'}")


if __name__ == "__main__":
    main()
