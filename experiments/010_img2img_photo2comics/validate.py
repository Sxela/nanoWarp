from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.img2img import IdentityPairedAugment, Img2ImgDiffusionUNet, PairedImageDataset
from src.img2img.colorspace import linear_to_srgb
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.metrics import ValidationMetrics, val_corrupt
from src.img2img.render import save_progress_strip, save_val_panel
from src.utils.config import apply_yaml_config


def build_method_from_ckpt(ckpt: dict, device: torch.device):
    # Older single-stage runs saved {"method": "...", "diffusion": cfg_dict}.
    # exp32+ saves the flow config under the "flow" key and omits "method",
    # so infer "flow" when that key is present and there's no explicit method.
    method = ckpt.get("method")
    if method is None:
        method = "flow" if "flow" in ckpt else "diffusion"
    if method == "flow":
        cfg_dict = ckpt.get("flow", ckpt.get("diffusion", {}))
        cfg = FlowConfig(**cfg_dict)
        return RectifiedImageFlow(cfg, device), cfg, method
    cfg_dict = ckpt.get("diffusion", {})
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
    state_key = "ema_model" if args.use_ema and "ema_model" in ckpt else "model"
    sd = ckpt[state_key]
    attn_res_str = train_cfg.get("attn_resolutions", "8")
    attn_res = tuple(int(x) for x in str(attn_res_str).split(",") if x.strip())
    color_space = train_cfg.get("color_space", "srgb")

    # Architecture metadata: prefer the saved config, but fall back to
    # detecting from state_dict shapes for older checkpoints (e.g. exp32+
    # runs whose save_checkpoint() didn't include these fields).
    if "source_in_stem" in train_cfg:
        source_in_stem = train_cfg["source_in_stem"]
    else:
        in_ch = sd["in_conv.weight"].shape[1]
        source_in_stem = (in_ch == 6)
    if "no_source_encoder" in train_cfg:
        use_source_encoder = not train_cfg["no_source_encoder"]
    else:
        use_source_encoder = any(k.startswith("source_encoder.") for k in sd.keys())
    upsample_type = train_cfg.get("upsample_type", "resize_conv")

    # image_size determines which attn modules get instantiated. For exp32+
    # the model is built at 512 regardless of training resolution. Default
    # for older single-resolution training scripts is the train image_size.
    # Resolve by checking which attn{1..4} keys are actually in the state_dict
    # — image_size must be the smallest value for which the present attn keys
    # match the configured attn_resolutions.
    if "image_size" in train_cfg:
        image_size = int(train_cfg["image_size"])
    else:
        present = {i for i in range(1, 5) if f"attn{i}.norm.weight" in sd}
        attn_set = set(attn_res)
        candidates = [128, 256, 512, 1024]
        image_size = next(
            (sz for sz in candidates
             if {i for i in range(1, 5) if (sz >> (i - 1)) in attn_set} == present),
            train_cfg.get("image_size", 128),
        )

    print(f"[arch] source_in_stem={source_in_stem}  use_source_encoder={use_source_encoder}  "
          f"image_size={image_size}  attn_res={attn_res}")
    model = Img2ImgDiffusionUNet(
        model_ch=train_cfg.get("model_ch", 64),
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=upsample_type,
        attn_resolutions=attn_res,
        image_size=image_size,
        color_space=color_space,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    diffusion, method_cfg, method = build_method_from_ckpt(ckpt, device)
    print(f"loaded method={method} method_cfg={method_cfg.__dict__}")
    metrics_fn = ValidationMetrics(device)

    ds = PairedImageDataset(
        args.data_root,
        augment=IdentityPairedAugment(image_size=args.image_size),
        split=args.split,
        color_space=color_space,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    high_t_min = max(0, min(args.high_t_min, method_cfg.timesteps - 1))
    high_t_max = max(high_t_min, min(args.high_t_max, method_cfg.timesteps - 1))

    def to_display(x: torch.Tensor) -> torch.Tensor:
        # Metrics + panels live in sRGB so numbers are comparable across runs
        # regardless of training color space.
        return linear_to_srgb(x).clamp(0, 1) if color_space == "linear_rgb" else x

    losses = []
    ssim_vals = []
    lpips_vals = []
    lpips_vgg_vals = []
    # Corruption-robustness pass: same val sources passed through
    # val_corrupt() before sampling. Robustness ≈ how small the gap is
    # between clean and corrupted lpips.
    ssim_vals_corr = []
    lpips_vals_corr = []
    lpips_vgg_vals_corr = []
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

            metric_vals = metrics_fn.compute(to_display(samples), to_display(target))
            ssim_vals.append(metric_vals["ssim"])
            lpips_vals.append(metric_vals["lpips_squeeze"])
            lpips_vgg_vals.append(metric_vals["lpips_vgg"])

            save_val_panel(
                to_display(source),
                to_display(target),
                to_display(samples),
                to_display(x0_hat_high),
                outdir / f"val_panel_{batch_idx:03d}.png",
                high_t_label=diag_label,
            )
            if args.save_progress_strip:
                disp_frames = [to_display(f) for f in frames]
                save_progress_strip(disp_frames, outdir / f"val_progress_{batch_idx:03d}.png")
            panels_written += 1
        else:
            samples, _frames = diffusion.sample(
                model,
                source,
                image_size=args.image_size,
                sample_steps=args.sample_steps,
                log_every=None,
            )
            metric_vals = metrics_fn.compute(to_display(samples), to_display(target))
            ssim_vals.append(metric_vals["ssim"])
            lpips_vals.append(metric_vals["lpips_squeeze"])
            lpips_vgg_vals.append(metric_vals["lpips_vgg"])

        # --- corruption-robustness pass on the same batch ---
        source_corr = val_corrupt(source)
        samples_corr, _ = diffusion.sample(
            model,
            source_corr,
            image_size=args.image_size,
            sample_steps=args.sample_steps,
            log_every=None,
        )
        m_corr = metrics_fn.compute(to_display(samples_corr), to_display(target))
        ssim_vals_corr.append(m_corr["ssim"])
        lpips_vals_corr.append(m_corr["lpips_squeeze"])
        lpips_vgg_vals_corr.append(m_corr["lpips_vgg"])

    def _mean(xs):
        return sum(xs) / max(len(xs), 1)

    metrics = {
        "checkpoint": args.checkpoint,
        "method": method,
        "use_ema": args.use_ema,
        "sample_steps": args.sample_steps,
        "high_t_range": [high_t_min, high_t_max],
        "num_batches": len(losses),
        "mean_loss": _mean(losses),
        # Clean-source metrics (primary; comparable to exp01-32 numbers)
        "mean_ssim_sampled": _mean(ssim_vals),
        "mean_lpips_sampled": _mean(lpips_vals),  # squeeze, kept for exp01-15 continuity
        "mean_lpips_squeeze_sampled": _mean(lpips_vals),
        "mean_lpips_vgg_sampled": _mean(lpips_vgg_vals),
        # Corrupted-source metrics — proxy for real-video robustness.
        # Same val pairs, source put through val_corrupt() before sampling.
        "mean_ssim_corrupted": _mean(ssim_vals_corr),
        "mean_lpips_squeeze_corrupted": _mean(lpips_vals_corr),
        "mean_lpips_vgg_corrupted": _mean(lpips_vgg_vals_corr),
        # Robustness deltas: smaller is more robust. exp23/25 (clean-only
        # training) will show big positive deltas; exp33/33b should show
        # smaller ones if the aug stack is paying off.
        "delta_lpips_vgg": _mean(lpips_vgg_vals_corr) - _mean(lpips_vgg_vals),
        "delta_lpips_squeeze": _mean(lpips_vals_corr) - _mean(lpips_vals),
        "train_config": train_cfg,
    }
    with open(outdir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
