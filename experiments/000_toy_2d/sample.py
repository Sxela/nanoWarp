from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.toy_diffusion import DiffusionConfig, TinyDenoiser, ToyDiffusion
from src.toy_diffusion.render import save_scatter_png


def parse_args():
    p = argparse.ArgumentParser(description="Sample from toy 2D diffusion checkpoint")
    p.add_argument("--checkpoint", default="out/toy2d/model.pt")
    p.add_argument("--num-samples", type=int, default=4096)
    p.add_argument("--output", default="out/toy2d/samples_latest.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = TinyDenoiser(hidden_dim=ckpt["config"]["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = ToyDiffusion(DiffusionConfig(**ckpt["diffusion"]), device)
    samples = diffusion.sample(model, args.num_samples).cpu().numpy()
    save_scatter_png(samples, Path(args.output))
    print(f"saved samples to {args.output}")


if __name__ == "__main__":
    main()

