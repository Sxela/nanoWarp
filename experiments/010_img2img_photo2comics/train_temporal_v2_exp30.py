"""Temporal fine-tune exp30: exp29b + source corruption robustness.

Same architecture as exp29 (temporal attn + decoder LoRA rank=8 + anchor conditioning).
Adds random source corruption during training to make the model robust to compression
artifacts and blur present in real video:

  1. Random Gaussian blur  (sigma uniform in [0.5, corrupt_blur_max])
  2. Random JPEG compression (quality uniform in [corrupt_jpeg_min, 95])

Each is applied independently per sample with probability corrupt_blur_prob /
corrupt_jpeg_prob. Corruption is applied to the SOURCE only — target stays clean.
Validation is always on clean sources (no corruption) for fair metric comparison.

Typical usage (resume from exp29b 20k):
    python3 experiments/010_img2img_photo2comics/train_temporal_v2_exp30.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --resume out/exp29b_temporal_declora_lpips/model_step_020000.pt \\
        --steps 5000 --lr 2e-5 --lr-min 1e-6 --lr-warmup-steps 100 \\
        --corrupt-blur-max 3.0 --corrupt-jpeg-min 30 \\
        --outdir out/exp30_corrupt_robust
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from src.img2img import Img2ImgDiffusionUNet
from src.img2img.decoder_lora import add_decoder_lora, decoder_lora_params
from src.img2img.ema import EMA
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.metrics import ValidationMetrics
from src.img2img.render import save_video_panel
from src.img2img.temporal_dataset_v2 import TemporalAugConfig, TemporalPairedDataset
from src.utils.config import apply_yaml_config

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as _LPIPS
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

try:
    import wandb as wandb_lib
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Source corruption
# ---------------------------------------------------------------------------

def _jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG round-trip on a (3, H, W) float [0,1] cpu tensor. Returns same shape."""
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf)).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).to(img.device)


def corrupt_source(x: torch.Tensor, T: int,
                   blur_max: float, jpeg_min: int,
                   blur_prob: float = 0.7, jpeg_prob: float = 0.7) -> torch.Tensor:
    """Randomly corrupt (B*T, 3, H, W) float [0,1] with blur and/or JPEG.

    Corruption parameters are sampled once per clip (B clips of T frames) so that
    all frames in the same clip share the same sigma / JPEG quality — matching
    real video where codec settings are consistent across frames.
    """
    if blur_max <= 0 and jpeg_min >= 95:
        return x
    N = x.shape[0]
    B = N // T
    out = x.clone()
    for b in range(B):
        # Sample clip-level corruption params once
        do_blur = blur_max > 0 and random.random() < blur_prob
        do_jpeg = jpeg_min < 95 and random.random() < jpeg_prob
        sigma = random.uniform(0.5, blur_max) if do_blur else None
        quality = random.randint(jpeg_min, 95) if do_jpeg else None
        k = max(3, int(2 * math.ceil(3 * sigma) + 1) | 1) if do_blur else None

        for t in range(T):
            img = out[b * T + t]
            if do_blur:
                img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
            if do_jpeg:
                img = _jpeg_compress(img, quality)
            out[b * T + t] = img
    return out


# ---------------------------------------------------------------------------
# Helpers (identical to exp29)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Temporal v2 + decoder LoRA + corruption robustness (exp30)")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True,
                   help="Spatial (exp25) checkpoint — used to read model config")
    p.add_argument("--resume", default=None,
                   help="Temporal checkpoint to resume from (exp29b weights)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--spatial-scale", type=float, default=2.0)
    p.add_argument("--anchor-prob", type=float, default=0.5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument("--lr-cosine", action="store_true", default=True)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--lpips-aux-net", default="vgg", choices=["squeeze", "vgg", "alex"])
    p.add_argument("--amp", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--panel-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=20)
    # corruption
    p.add_argument("--corrupt-blur-max", type=float, default=3.0,
                   help="Max Gaussian blur sigma (0 = disabled)")
    p.add_argument("--corrupt-jpeg-min", type=int, default=30,
                   help="Min JPEG quality (95 = disabled)")
    p.add_argument("--corrupt-blur-prob", type=float, default=0.7,
                   help="Per-sample probability of applying blur")
    p.add_argument("--corrupt-jpeg-prob", type=float, default=0.7,
                   help="Per-sample probability of applying JPEG compression")
    p.add_argument("--outdir", default="out/exp30_corrupt_robust")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="nanoWarp")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default="")
    return p.parse_args()


def cosine_lr(step, total_steps, warmup_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


@torch.no_grad()
def _sample(diffusion, model, source, frame_mask, anchor_frames, anc_mask, sample_steps=20):
    b = source.shape[0]
    x = source.clone()
    ts = torch.linspace(0.0, 1.0, sample_steps + 1, device=source.device)
    for i in range(sample_steps):
        t_cur = ts[i].expand(b)
        dt = float(ts[i + 1] - ts[i])
        v_hat = model(source, x, diffusion._scale_t(t_cur), frame_mask)
        x = x + dt * v_hat
        if anchor_frames is not None and anc_mask is not None and anc_mask.any():
            x[anc_mask] = anchor_frames[anc_mask]
    return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read model config from spatial checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_cfg = ckpt.get("config", {})
    attn_res = tuple(int(x) for x in str(train_cfg.get("attn_resolutions", "8")).split(",") if x.strip())
    color_space = train_cfg.get("color_space", "srgb")

    model = Img2ImgDiffusionUNet(
        model_ch=train_cfg.get("model_ch", 88),
        pretrained_source_encoder=False,
        source_in_stem=train_cfg.get("source_in_stem", False),
        use_source_encoder=not train_cfg.get("no_source_encoder", False),
        upsample_type=train_cfg.get("upsample_type", "resize_conv"),
        attn_resolutions=attn_res,
        image_size=args.image_size,
        color_space=color_space,
        use_temporal=True,
        mask_channels=1,
    ).to(device)

    resume_step = 0
    if args.resume:
        add_decoder_lora(model, rank=args.lora_rank)
        res_ckpt = torch.load(args.resume, map_location=device)
        state_key = "ema_model" if "ema_model" in res_ckpt else "model"
        missing, _ = model.load_state_dict(res_ckpt[state_key], strict=False)
        if missing:
            print(f"[warn] resume missing keys: {missing[:4]}")
        resume_step = res_ckpt.get("step", 0)
        print(f"resumed from {args.resume} ({state_key}) at step {resume_step}")
    else:
        state_key = "ema_model" if "ema_model" in ckpt else "model"
        missing, _ = model.load_state_dict(ckpt[state_key], strict=False)
        temporal_keys = {k for k in missing if "tattn" in k or "mask_proj" in k}
        other_missing = set(missing) - temporal_keys
        if other_missing:
            print(f"[warn] non-temporal missing keys: {other_missing}")
        print(f"loaded spatial weights from {args.checkpoint} ({state_key})")
        print(f"new temporal/mask modules: {len(temporal_keys)} keys")
        add_decoder_lora(model, rank=args.lora_rank)

    lora_p = decoder_lora_params(model)
    print(f"decoder LoRA rank={args.lora_rank}  lora_params={sum(p.numel() for p in lora_p):,}")

    for name, param in model.named_parameters():
        if "tattn" not in name and "mask_proj" not in name and "lora_" not in name:
            param.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"params total={total:,}  trainable={trainable:,}")
    print(f"corruption: blur_max={args.corrupt_blur_max} jpeg_min={args.corrupt_jpeg_min} "
          f"blur_prob={args.corrupt_blur_prob} jpeg_prob={args.corrupt_jpeg_prob}")

    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    flow_cfg_dict = ckpt.get("flow", ckpt.get("diffusion", {}))
    flow_cfg = FlowConfig(**flow_cfg_dict) if flow_cfg_dict else FlowConfig()
    diffusion = RectifiedImageFlow(flow_cfg, device)

    aux_lpips = None
    if _LPIPS_AVAILABLE and args.lpips_weight > 0:
        aux_lpips = _LPIPS(net_type=args.lpips_aux_net, normalize=True).to(device)
        for p in aux_lpips.parameters():
            p.requires_grad_(False)
        print(f"lpips_aux_net={args.lpips_aux_net} weight={args.lpips_weight}")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))

    aug_cfg = TemporalAugConfig(
        image_size=args.image_size,
        spatial_scale=args.spatial_scale,
        num_frames=args.num_frames,
        max_pan_frac=0.25,
        zoom_range=(0.90, 1.10),
        horizontal_flip_prob=0.5,
        anchor_prob=args.anchor_prob,
    )
    train_ds = TemporalPairedDataset(args.data_root, split="train", config=aug_cfg)
    val_aug_cfg = TemporalAugConfig(
        image_size=args.image_size,
        spatial_scale=args.spatial_scale,
        num_frames=args.num_frames,
        max_pan_frac=0.15,
        zoom_range=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        anchor_prob=0.0,
    )
    val_ds = TemporalPairedDataset(args.data_root, split="val", config=val_aug_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        import os
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb_lib.login(key=api_key)
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        try:
            wandb_lib.init(project=args.wandb_project, name=args.wandb_run_name, tags=tags,
                           config=vars(args))
        except Exception as e:
            print(f"[warn] wandb init failed ({e}), continuing without wandb")
            use_wandb = False

    metrics_fn = ValidationMetrics(device)

    model.train()
    step = resume_step
    T = args.num_frames
    data_iter = iter(train_loader)
    loss_log: list[float] = []

    target_step = resume_step + args.steps
    print(f"fine-tuning for {args.steps} steps  (steps {resume_step}→{target_step})  "
          f"T={T}  bs={args.batch_size}  anchor_prob={args.anchor_prob}")

    while step < target_step:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        source_t = batch["source"].to(device)       # (B, T, 3, H, W)
        target_t = batch["target"].to(device)
        anchor   = batch["anchor_mask"].to(device)  # (B, T) bool

        B, _, C, H, W = source_t.shape

        src_flat = source_t.reshape(B * T, C, H, W)
        tgt_flat = target_t.reshape(B * T, C, H, W)
        anc_flat = anchor.reshape(B * T)

        # Apply source corruption — same params for all T frames in each clip
        src_corrupt = corrupt_source(
            src_flat, T,
            blur_max=args.corrupt_blur_max,
            jpeg_min=args.corrupt_jpeg_min,
            blur_prob=args.corrupt_blur_prob,
            jpeg_prob=args.corrupt_jpeg_prob,
        )

        lr = cosine_lr(step - resume_step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            t_per_clip = torch.rand(B, device=device)
            t_cont = t_per_clip.unsqueeze(1).expand(B, T).reshape(B * T)

            noisy_free, _ = diffusion.q_sample(src_corrupt, tgt_flat, t_cont)
            noisy_flat = torch.where(anc_flat[:, None, None, None], tgt_flat, noisy_free)

            frame_mask = (~anc_flat).float().view(B * T, 1, 1, 1).expand(B * T, 1, H, W)

            t_emb = diffusion._scale_t(t_cont)
            model.set_temporal_frames(T)
            v_hat = model(src_corrupt, noisy_flat, t_emb, frame_mask)
            model.set_temporal_frames(1)

            v_tgt = tgt_flat - src_corrupt

            free_mask = ~anc_flat
            if free_mask.any():
                flow_loss = F.mse_loss(v_hat[free_mask], v_tgt[free_mask])
            else:
                flow_loss = torch.tensor(0.0, device=device)

            lpips_loss = torch.tensor(0.0, device=device)
            if aux_lpips is not None and args.lpips_weight > 0 and free_mask.any():
                x0_hat = diffusion.predict_target_from_v(noisy_flat, t_cont, v_hat)
                lpips_loss = aux_lpips(x0_hat[free_mask], tgt_flat[free_mask]).mean()

            loss = flow_loss + args.lpips_weight * lpips_loss

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        loss_log.append(float(loss.item()))
        step += 1

        if step % 100 == 0:
            avg_loss = sum(loss_log[-100:]) / len(loss_log[-100:])
            print(f"step={step:6d}  loss={avg_loss:.5f}  lr={lr:.2e}")
            if use_wandb:
                wandb_lib.log({"loss": avg_loss, "flow_loss": float(flow_loss),
                               "lpips_loss": float(lpips_loss), "lr": lr}, step=step)

        if step % args.checkpoint_every == 0 or step == target_step:
            ckpt_path = outdir / f"model_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "ema_model": ema.model.state_dict(),
                "config": vars(args),
                "flow": flow_cfg.__dict__,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

        if step % args.val_every == 0:
            _run_val(model, ema, diffusion, val_loader, metrics_fn, args, device, step, outdir, use_wandb)

        if step % args.panel_every == 0:
            _save_panel(ema, diffusion, val_loader, args, device, step, outdir)

    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "config": vars(args),
        "flow": flow_cfg.__dict__,
    }, outdir / "model.pt")
    print("[done] saved final model.pt")
    if use_wandb:
        wandb_lib.finish()


@torch.no_grad()
def _run_val(model, ema, diffusion, val_loader, metrics_fn, args, device, step, outdir, use_wandb):
    """Val on clean sources (no corruption) for fair comparison."""
    eval_model = ema.model
    T = args.num_frames
    lpips_vals, ssim_vals = [], []

    for i, batch in enumerate(val_loader):
        if i >= 10:
            break
        source_t = batch["source"].to(device)
        target_t = batch["target"].to(device)
        B, _, C, H, W = source_t.shape

        src_flat = source_t.reshape(B * T, C, H, W)
        tgt_flat = target_t.reshape(B * T, C, H, W)

        frame_mask = torch.ones(B * T, 1, H, W, device=device)
        eval_model.set_temporal_frames(T)
        samples = _sample(diffusion, eval_model, src_flat, frame_mask,
                          anchor_frames=None, anc_mask=None, sample_steps=args.sample_steps)
        eval_model.set_temporal_frames(1)

        m = metrics_fn.compute(samples, tgt_flat)
        lpips_vals.append(m["lpips_squeeze"])
        ssim_vals.append(m["ssim"])

    mean_lpips = sum(lpips_vals) / len(lpips_vals)
    mean_ssim  = sum(ssim_vals)  / len(ssim_vals)
    print(f"[val] step={step}  lpips_sq={mean_lpips:.4f}  ssim={mean_ssim:.4f}")

    with open(outdir / f"val_step{step:06d}.json", "w") as f:
        json.dump({"step": step, "lpips_sq": mean_lpips, "ssim": mean_ssim}, f, indent=2)

    if use_wandb:
        wandb_lib.log({"val/lpips_sq": mean_lpips, "val/ssim": mean_ssim}, step=step)


@torch.no_grad()
def _save_panel(ema, diffusion, val_loader, args, device, step, outdir):
    eval_model = ema.model
    T = args.num_frames

    batch = next(iter(val_loader))
    source_t = batch["source"].to(device)
    target_t = batch["target"].to(device)

    src = source_t[0]
    tgt = target_t[0]
    _, C, H, W = src.shape

    frame_mask = torch.ones(T, 1, H, W, device=device)
    eval_model.set_temporal_frames(T)
    samples = _sample(diffusion, eval_model, src, frame_mask,
                      anchor_frames=None, anc_mask=None, sample_steps=args.sample_steps)
    eval_model.set_temporal_frames(1)

    save_video_panel(
        src.cpu(), tgt.cpu(), samples.cpu(),
        outdir / f"video_step_{step:06d}",
        fps=8,
    )


if __name__ == "__main__":
    main()
