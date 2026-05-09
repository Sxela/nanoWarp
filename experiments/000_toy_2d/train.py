from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.toy_diffusion import DiffusionConfig, TinyDenoiser, ToyDiffusion, sample_dataset
from src.toy_diffusion.render import save_image_grid, save_loss_plot, save_scatter_png


def parse_args():
    p = argparse.ArgumentParser(description="Train toy 2D diffusion")
    p.add_argument("--dataset", default="moons", choices=["moons", "spiral", "circle"])
    p.add_argument("--dataset-size", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--noise", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--outdir", default="out/toy2d")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    points = sample_dataset(args.dataset, args.dataset_size, args.noise, args.seed)
    save_scatter_png(points, outdir / "dataset.png")

    x = torch.from_numpy(points).to(device)

    model = TinyDenoiser(hidden_dim=args.hidden_dim).to(device)
    diffusion = ToyDiffusion(DiffusionConfig(timesteps=args.timesteps), device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    sample_paths = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x.shape[0], (args.batch_size,), device=device)
        batch = x[idx]
        loss = diffusion.training_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        save_loss_plot(losses, outdir / "loss.png")
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.6f}")

        if step % args.sample_every == 0 or step == args.steps:
            samples = diffusion.sample(model, 2048).detach().cpu().numpy()
            sample_path = outdir / f"samples_step_{step:06d}.png"
            save_scatter_png(samples, sample_path)
            sample_paths.append(sample_path)
            save_image_grid(sample_paths, outdir / "progress_grid.png")

    ckpt = {
        "model": model.state_dict(),
        "config": vars(args),
        "diffusion": diffusion.config.__dict__,
    }
    torch.save(ckpt, outdir / "model.pt")
    with open(outdir / "metrics.json", "w") as f:
        json.dump({"final_loss": losses[-1], "mean_loss_last_100": sum(losses[-100:]) / min(len(losses), 100)}, f, indent=2)
    print(f"saved checkpoint to {outdir / 'model.pt'}")


if __name__ == "__main__":
    main()
