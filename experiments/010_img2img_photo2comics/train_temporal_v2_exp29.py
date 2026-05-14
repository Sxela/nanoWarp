"""Temporal finetuning v2 with decoder LoRA (exp29).

Extends train_temporal_v2.py (exp28c) with:
  1. Decoder LoRA (rank=8) on dec4/dec3/dec2/dec1 conv1+conv2.
     Zero-init B → frozen backbone behaviour at step 0.
  2. Anchor reinjection at every ODE step during inference sampling.
     After each Euler step x = x + dt*v, anchor frame is hard-reset:
       x[anchor_mask] = anchor_clean[anchor_mask]
     This gives strong first-frame conditioning without any architectural change.

Trainable parameters: temporal attention modules + mask_proj + decoder LoRA.
All other spatial weights remain frozen.

Usage:
    python3 experiments/010_img2img_photo2comics/train_temporal_v2_exp29.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --steps 20000 --outdir out/exp29_temporal_declora
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
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Temporal v2 + decoder LoRA (exp29)")
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
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for decoder convs")
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
    p.add_argument("--outdir", default="out/exp29_temporal_declora")
    p.add_argument("--resume", default=None,
                   help="Path to a temporal checkpoint to resume from (overrides --checkpoint for weights)")
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
            anchor_frames: torch.Tensor | None, anc_mask: torch.Tensor | None,
            sample_steps: int = 20) -> torch.Tensor:
    """Euler integration with frame_mask + optional anchor reinjection at every step.

    anchor_frames : (B*T, C, H, W) clean anchor targets — reinject after each step.
    anc_mask      : (B*T,) bool — which positions are anchors.
    """
    b = source.shape[0]
    x = source.clone()
    ts = torch.linspace(0.0, 1.0, sample_steps + 1, device=source.device)
    for i in range(sample_steps):
        t_cur = ts[i].expand(b)
        dt = float(ts[i + 1] - ts[i])
        t_emb = diffusion._scale_t(t_cur)
        v_hat = model(source, x, t_emb, frame_mask)
        x = x + dt * v_hat
        # Reinject clean anchor frames after every Euler step
        if anchor_frames is not None and anc_mask is not None and anc_mask.any():
            x[anc_mask] = anchor_frames[anc_mask]
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
        mask_channels=1,
    ).to(device)

    resume_step = 0
    if args.resume:
        # Resume from a temporal checkpoint (already has tattn + lora keys).
        # Inject LoRA first so the model's key names match the saved state_dict.
        add_decoder_lora(model, rank=args.lora_rank)
        res_ckpt = torch.load(args.resume, map_location=device)
        state_key = "ema_model" if "ema_model" in res_ckpt else "model"
        missing, unexpected = model.load_state_dict(res_ckpt[state_key], strict=False)
        if missing:
            print(f"[warn] resume missing keys: {missing[:4]}")
        resume_step = res_ckpt.get("step", 0)
        print(f"resumed from {args.resume} ({state_key}) at step {resume_step}")
    else:
        # Fresh start: load spatial weights first (keys are plain conv names),
        # then inject LoRA so it wraps the already-loaded frozen convs.
        state_key = "ema_model" if "ema_model" in ckpt else "model"
        missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
        temporal_keys = {k for k in missing if "tattn" in k or "mask_proj" in k}
        other_missing = set(missing) - temporal_keys
        if other_missing:
            print(f"[warn] non-temporal missing keys: {other_missing}")
        print(f"loaded spatial weights from {args.checkpoint} ({state_key})")
        print(f"new temporal/mask modules: {len(temporal_keys)} keys")
        add_decoder_lora(model, rank=args.lora_rank)
    lora_p = decoder_lora_params(model)
    print(f"decoder LoRA rank={args.lora_rank}  lora_params={sum(p.numel() for p in lora_p):,}")

    # Freeze all spatial weights; train temporal attn + mask_proj + decoder LoRA
    for name, param in model.named_parameters():
        if "tattn" not in name and "mask_proj" not in name and "lora_" not in name:
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

    # --- training loop ---
    model.train()
    step = resume_step
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

        src_flat = source_t.reshape(B * T, C, H, W)
        tgt_flat = target_t.reshape(B * T, C, H, W)
        anc_flat = anchor.reshape(B * T)

        lr = cosine_lr(step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            t_per_clip = torch.rand(B, device=device)
            t_cont = t_per_clip.unsqueeze(1).expand(B, T).reshape(B * T)

            noisy_free, _ = diffusion.q_sample(src_flat, tgt_flat, t_cont)
            noisy_flat = torch.where(anc_flat[:, None, None, None], tgt_flat, noisy_free)

            frame_mask = (~anc_flat).float().view(B * T, 1, 1, 1).expand(B * T, 1, H, W)

            t_emb = diffusion._scale_t(t_cont)
            model.set_temporal_frames(T)
            v_hat = model(src_flat, noisy_flat, t_emb, frame_mask)
            model.set_temporal_frames(1)

            v_tgt = tgt_flat - src_flat

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

        # Val: all frames free, no anchor reinjection
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
