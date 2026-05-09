from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import IdentityPairedAugment, Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.metrics import ValidationMetrics
from src.img2img.render import save_progress_strip, save_val_panel
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
    p = argparse.ArgumentParser(description="Validate pixel-space img2img diffusion (full-inference panels)")
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
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--high-t-min", type=int, default=800)
    p.add_argument("--high-t-max", type=int, default=999)
    p.add_argument("--save-progress-strip", action="store_true")
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
        model_ch=train_cfg.get("model_ch", 64),
        pretrained_source_encoder=False,
        source_in_stem=train_cfg.get("source_in_stem", False),
        use_source_encoder=not train_cfg.get("no_source_encoder", False),
    ).to(device)
    state_key = "ema_model" if args.use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    diffusion, method_cfg, method = build_method_from_ckpt(ckpt, device)
    print(f"loaded method={method} method_cfg={method_cfg.__dict__}")
    metrics_fn = ValidationMetrics(device)

    ds = PairedImageDataset(args.data_root, augment=IdentityPairedAugment(image_size=args.image_size), split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    high_t_min = max(0, min(args.high_t_min, method_cfg.timesteps - 1))
    high_t_max = max(high_t_min, min(args.high_t_max, method_cfg.timesteps - 1))

    losses = []
    ssim_vals = []
    lpips_vals = []
    panels_written = 0
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.max_batches:
            break
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        loss, _t, _x_t, _noise, _eps_hat, _x0_rand_t, _dloss, _ploss = diffusion.training_loss(model, source, target)
        losses.append(float(loss.item()))

        if panels_written < args.panel_count:
            samples, frames = diffusion.sample(
                model,
                source,
                image_size=args.image_size,
                sample_steps=args.sample_steps,
                log_every=args.progress_every,
            )

            if method == "flow":
                T = method_cfg.timesteps
                fm_low_min = max(1e-3, (T - 1 - high_t_max) / T)
                fm_low_max = max(fm_low_min + 1e-3, (T - 1 - high_t_min) / T)
                t_diag_cont = torch.rand(target.shape[0], device=device) * (fm_low_max - fm_low_min) + fm_low_min
                x_t_diag, _ = diffusion.q_sample(source, target, t_diag_cont)
                t_diag_emb = diffusion._scale_t(t_diag_cont)
                v_hat_diag = model(source, x_t_diag, t_diag_emb)
                x0_hat_high = diffusion.predict_target_from_v(x_t_diag, t_diag_cont, v_hat_diag)
                diag_label = f"target_hat_t_cont={fm_low_min:.3f}-{fm_low_max:.3f}"
            else:
                t_high = torch.randint(high_t_min, high_t_max + 1, (target.shape[0],), device=device)
                x_t_high, _ = diffusion.q_sample(target, t_high)
                model_out_high = model(source, x_t_high, t_high)
                x0_hat_high, _ = diffusion.predict_pair(x_t_high, t_high, model_out_high)
                diag_label = f"x0_hat_t={int(t_high[0].item())}-{int(t_high[-1].item())}"

            metric_vals = metrics_fn.compute(samples, target)
            ssim_vals.append(metric_vals["ssim"])
            lpips_vals.append(metric_vals["lpips"])

            save_val_panel(
                source,
                target,
                samples,
                x0_hat_high,
                outdir / f"val_panel_{batch_idx:03d}.png",
                high_t_label=diag_label,
            )
            if args.save_progress_strip:
                save_progress_strip(frames, outdir / f"val_progress_{batch_idx:03d}.png")
            panels_written += 1
        else:
            samples, _frames = diffusion.sample(
                model,
                source,
                image_size=args.image_size,
                sample_steps=args.sample_steps,
                log_every=None,
            )
            metric_vals = metrics_fn.compute(samples, target)
            ssim_vals.append(metric_vals["ssim"])
            lpips_vals.append(metric_vals["lpips"])

    metrics = {
        "checkpoint": args.checkpoint,
        "method": method,
        "use_ema": args.use_ema,
        "sample_steps": args.sample_steps,
        "high_t_range": [high_t_min, high_t_max],
        "num_batches": len(losses),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "mean_ssim_sampled": sum(ssim_vals) / max(len(ssim_vals), 1),
        "mean_lpips_sampled": sum(lpips_vals) / max(len(lpips_vals), 1),
        "train_config": train_cfg,
    }
    with open(outdir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
