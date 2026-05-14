"""Temporal finetuning v2 — HotShot-XL style with WAN-style anchor conditioning.

Architecture (vs v1 / exp27):
  - Simple temporal self-attention (HotShot-XL): sinusoidal pos emb added to
    normalised sequence before QKV, no cross-chunk state.
  - mask_proj: zero-init 1×1 conv adds mask channel signal to stem features.
    mask=1 (free frame, model generates), mask=0 (anchor frame, copy given).
  - T=4 frames per training clip (default).

Training loop (per step):
  1. Sample T-frame clip; anchor_mask[0] may be True (WAN-style conditioning).
  2. For anchor frames: noisy_target = clean target (no flow noise), mask=0.
     For free frames:   noisy_target = flow-sampled x_t, mask=1.
  3. Single forward pass over all T frames (B*T batch).
  4. Loss computed on free frames only; anchor frames provide context.

Inference (long video, WAN-style chunking):
  - Run chunk 0 with all frames free (mask=1).
  - For chunk N>0: use last output frame of chunk N-1 as frame 0 (anchor,
    mask=0); remaining T-1 frames are free (mask=1).
  - The model has learned this conditioning during training.

Usage:
    python3 experiments/010_img2img_photo2comics/train_temporal_v2.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --steps 20000 --outdir out/exp28_temporal_v2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.img2img import Img2ImgDiffusionUNet
from src.img2img.ema import EMA
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.metrics import ValidationMetrics
from src.img2img.render import save_video_panel
from src.img2img.temporal_dataset_v2 import TemporalAugConfig, TemporalPairedDataset
from src.utils.config import apply_yaml_config

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

try:
    import wandb as wandb_lib
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Temporal finetuning v2 (HotShot-XL + WAN mask)")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4, help="Clips per batch")
    p.add_argument("--num-frames", type=int, default=4, help="T: frames per clip")
    p.add_argument("--spatial-scale", type=float, default=2.0)
    p.add_argument("--anchor-prob", type=float, default=0.5,
                   help="Probability that first frame of a clip is an anchor")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--lr-cosine", action="store_true", default=True)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--lpips-aux-net", default="vgg", choices=["squeeze", "vgg", "alex"])
    p.add_argument("--amp", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--panel-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--outdir", default="out/temporal_v2")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="nanoWarp")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default="")
    return p.parse_args()


def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _sample(diffusion, model, source: torch.Tensor, frame_mask: torch.Tensor,
            sample_steps: int = 20) -> torch.Tensor:
    """Euler integration with frame_mask passed at every denoising step."""
    b = source.shape[0]
    x = source.clone()
    ts = torch.linspace(0.0, 1.0, sample_steps + 1, device=source.device)
    for i in range(sample_steps):
        t_cur = ts[i].expand(b)
        dt = float(ts[i + 1] - ts[i])
        t_emb = diffusion._scale_t(t_cur)
        v_hat = model(source, x, t_emb, frame_mask)
        x = x + dt * v_hat
    return x.clamp(0, 1)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load pretrained checkpoint ---
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
        mask_channels=1,   # WAN-style mask channel
    ).to(device)

    # Load spatial weights (strict=False: tattn* and mask_proj are new)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    temporal_keys = {k for k in missing if "tattn" in k or "mask_proj" in k}
    other_missing = set(missing) - temporal_keys
    if other_missing:
        print(f"[warn] non-temporal missing keys: {other_missing}")
    print(f"loaded spatial weights from {args.checkpoint} ({state_key})")
    print(f"new temporal/mask modules: {sorted(temporal_keys)[:6]}{'...' if len(temporal_keys) > 6 else ''}")

    # Freeze all spatial weights; only train temporal attn + mask_proj
    for name, param in model.named_parameters():
        if "tattn" not in name and "mask_proj" not in name:
            param.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"params total={total:,}  trainable={trainable:,}")

    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    flow_cfg_dict = ckpt.get("flow", ckpt.get("diffusion", {}))
    flow_cfg = FlowConfig(**flow_cfg_dict) if flow_cfg_dict else FlowConfig()
    diffusion = RectifiedImageFlow(flow_cfg, device)

    aux_lpips = None
    if _LPIPS_AVAILABLE and args.lpips_weight > 0:
        aux_lpips = lpips_lib.LPIPS(net=args.lpips_aux_net).to(device)
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
        anchor_prob=0.0,   # no anchor during val sampling
        deterministic=True,
    )
    val_ds = TemporalPairedDataset(args.data_root, split="val", config=val_aug_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_lib.init(project=args.wandb_project, name=args.wandb_run_name, tags=tags,
                       config=vars(args))

    metrics_fn = ValidationMetrics(device)

    # --- training loop ---
    model.train()
    step = 0
    T = args.num_frames
    data_iter = iter(train_loader)
    loss_log: list[float] = []

    print(f"training for {args.steps} steps  T={T}  bs={args.batch_size}  anchor_prob={args.anchor_prob}")

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        source_t = batch["source"].to(device)       # (B, T, 3, H, W)
        target_t = batch["target"].to(device)
        anchor   = batch["anchor_mask"].to(device)  # (B, T) bool

        B, _, C, H, W = source_t.shape

        # Flatten to (B*T, C, H, W) for the UNet
        src_flat = source_t.reshape(B * T, C, H, W)
        tgt_flat = target_t.reshape(B * T, C, H, W)

        # Anchor mask flat: (B*T,) bool
        anc_flat = anchor.reshape(B * T)

        lr = cosine_lr(step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            # Sample a single noise level per clip, broadcast across frames
            t_per_clip = torch.rand(B, device=device)
            t_cont = t_per_clip.unsqueeze(1).expand(B, T).reshape(B * T)  # (B*T,)

            # Build noisy targets: anchor frames get clean target (t=0 effectively)
            noisy_free, _ = diffusion.q_sample(src_flat, tgt_flat, t_cont)
            noisy_flat = torch.where(anc_flat[:, None, None, None], tgt_flat, noisy_free)

            # frame_mask channel: 1=free (generate), 0=anchor (given)
            frame_mask = (~anc_flat).float().view(B * T, 1, 1, 1).expand(B * T, 1, H, W)

            t_emb = diffusion._scale_t(t_cont)
            model.set_temporal_frames(T)
            v_hat = model(src_flat, noisy_flat, t_emb, frame_mask)
            model.set_temporal_frames(1)

            v_tgt = tgt_flat - src_flat

            # Loss on free frames only
            free_mask = ~anc_flat  # (B*T,) bool
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

        if step % args.checkpoint_every == 0 or step == args.steps:
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

        # Val: all frames free, no anchor
        frame_mask = torch.ones(B * T, 1, H, W, device=device)
        eval_model.set_temporal_frames(T)
        samples = _sample(diffusion, eval_model, src_flat, frame_mask, args.sample_steps)
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

    # Use first sample in batch: (T, C, H, W)
    src = source_t[0]
    tgt = target_t[0]
    _, C, H, W = src.shape

    # Run all-free (no anchor) for the panel
    frame_mask = torch.ones(T, 1, H, W, device=device)
    eval_model.set_temporal_frames(T)
    samples = _sample(diffusion, eval_model, src, frame_mask, args.sample_steps)
    eval_model.set_temporal_frames(1)

    save_video_panel(
        src.cpu(), tgt.cpu(), samples.cpu(),
        outdir / f"video_step_{step:06d}",
        fps=8,
    )


if __name__ == "__main__":
    main()
