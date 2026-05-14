"""Temporal finetuning of a pretrained img2img flow model for video consistency.

Architecture changes over train.py:
  - UNet gains TemporalAttn at the 32px encoder level and the 16px bottleneck.
  - All spatial weights are frozen; only temporal attention parameters are trained.
  - Dataset synthesizes 2T-frame clips from still pairs via pan/zoom trajectories.

Training loop (per step):
  1. Split the 2T-frame clip into chunk_a (frames 0..T-1) and chunk_b (frames T..2T-1).
  2. Forward chunk_a with no prev context → loss_a + warp_consistency_a.
  3. Detach stored KV (no backprop through the chunk boundary).
  4. Forward chunk_b using chunk_a's KV as context → loss_b + warp_consistency_b.
  5. Add cross-chunk boundary warp loss.
  6. Total loss = flow_loss + lpips_loss + temporal_weight * warp_loss.

Warp consistency: the model output at frame t, when translated by the known
analytic flow (dx, dy), should match the output at frame t+1.

Usage:
    python3 experiments/010_img2img_photo2comics/train_temporal.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --steps 20000 --temporal-weight 1.0 \\
        --outdir out/exp27_temporal_finetune
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
from src.img2img.render import save_val_panel, save_video_panel
from src.img2img.temporal_dataset import TemporalAugConfig, TemporalPairedDataset, warp_by_translation
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
    p = argparse.ArgumentParser(description="Temporal finetuning for video-consistent img2img")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--checkpoint", required=True, help="Pretrained spatial model (exp25/model.pt)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2, help="Clips per batch (each clip = 2T frames)")
    p.add_argument("--num-frames", type=int, default=8, help="T: frames per chunk")
    p.add_argument("--spatial-scale", type=float, default=2.0, help="Canvas = image_size * spatial_scale")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--lr-cosine", action="store_true", default=True)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--lpips-aux-net", default="vgg", choices=["squeeze", "vgg", "alex"])
    p.add_argument("--temporal-weight", type=float, default=1.0,
                   help="Weight for warp consistency loss relative to (flow + lpips)")
    p.add_argument("--amp", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--panel-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--outdir", default="out/temporal_finetune")
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


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def warp_consistency_loss(
    outputs: torch.Tensor,   # (B, T, C, H, W)
    flow: torch.Tensor,      # (B, T-1, 2)  — dx, dy in pixels
) -> torch.Tensor:
    """L1 warp consistency across T frames.

    For each t in [0, T-2]: warp output[t] by flow[t] → should match output[t+1].
    """
    B, T, C, H, W = outputs.shape
    total = torch.tensor(0.0, device=outputs.device)
    count = 0
    for t in range(T - 1):
        dx = flow[:, t, 0]   # (B,)
        dy = flow[:, t, 1]   # (B,)
        warped = warp_by_translation(outputs[:, t], dx, dy)
        total = total + F.l1_loss(warped, outputs[:, t + 1])
        count += 1
    return total / max(count, 1)


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
    attn_res_str = train_cfg.get("attn_resolutions", "8")
    attn_res = tuple(int(x) for x in str(attn_res_str).split(",") if x.strip())
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
        use_temporal=True,  # NEW: adds tattn4 + tattn_mid
    ).to(device)

    # load spatial weights (strict=False: temporal modules not in ckpt)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    temporal_keys = {k for k in missing if "tattn" in k}
    other_missing = set(missing) - temporal_keys
    if other_missing:
        print(f"[warn] non-temporal missing keys: {other_missing}")
    print(f"loaded spatial weights from {args.checkpoint} ({state_key})")
    print(f"new temporal modules: {sorted(temporal_keys)[:6]}{'...' if len(temporal_keys) > 6 else ''}")

    # freeze all spatial parameters
    for name, param in model.named_parameters():
        if "tattn" not in name:
            param.requires_grad_(False)

    temporal_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in temporal_params)
    print(f"params total={total_params:,}  trainable (temporal only)={trainable_params:,}")

    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.AdamW(temporal_params, lr=args.lr, weight_decay=1e-4)

    # --- flow method (same config as spatial model) ---
    flow_cfg_dict = ckpt.get("flow", ckpt.get("diffusion", {}))
    flow_cfg = FlowConfig(**flow_cfg_dict) if flow_cfg_dict else FlowConfig()
    diffusion = RectifiedImageFlow(flow_cfg, device)

    # --- LPIPS ---
    aux_lpips = None
    if _LPIPS_AVAILABLE and args.lpips_weight > 0:
        aux_lpips = lpips_lib.LPIPS(net=args.lpips_aux_net).to(device)
        for p in aux_lpips.parameters():
            p.requires_grad_(False)
        print(f"lpips_aux_net={args.lpips_aux_net} weight={args.lpips_weight}")

    # --- AMP ---
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))

    # --- dataset ---
    aug_cfg = TemporalAugConfig(
        image_size=args.image_size,
        spatial_scale=args.spatial_scale,
        num_frames=args.num_frames,
        max_pan_frac=0.25,
        zoom_range=(0.90, 1.10),
        horizontal_flip_prob=0.5,
    )
    train_ds = TemporalPairedDataset(args.data_root, split="train", config=aug_cfg)
    # val: use val split with same aug (we'll look at single-frame quality too)
    val_aug_cfg = TemporalAugConfig(
        image_size=args.image_size, spatial_scale=args.spatial_scale,
        num_frames=args.num_frames, max_pan_frac=0.15,
        zoom_range=(1.0, 1.0), horizontal_flip_prob=0.0,
        deterministic=True,
    )
    val_ds = TemporalPairedDataset(args.data_root, split="val", config=val_aug_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    # --- wandb ---
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_lib.init(project=args.wandb_project, name=args.wandb_run_name, tags=tags,
                       config=vars(args))

    metrics_fn = ValidationMetrics(device)

    # --- training loop ---
    model.train()
    step = 0
    data_iter = iter(train_loader)
    T = args.num_frames
    loss_log: list[float] = []
    warp_log: list[float] = []

    print(f"training for {args.steps} steps  T={T}  bs={args.batch_size}  "
          f"temporal_weight={args.temporal_weight}")

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # shapes: (B, 2T, 3, H, W) and (B, 2T-1, 2)
        source_2t = batch["source"].to(device)   # (B, 2T, C, H, W)
        target_2t = batch["target"].to(device)
        flow_2t   = batch["flow"].to(device)     # (B, 2T-1, 2)

        B = source_2t.shape[0]
        C, H, W = source_2t.shape[2], source_2t.shape[3], source_2t.shape[4]

        # split into two T-frame chunks
        src_a = source_2t[:, :T].reshape(B * T, C, H, W)
        tgt_a = target_2t[:, :T].reshape(B * T, C, H, W)
        src_b = source_2t[:, T:].reshape(B * T, C, H, W)
        tgt_b = target_2t[:, T:].reshape(B * T, C, H, W)
        flow_a = flow_2t[:, :T - 1]   # (B, T-1, 2) — within chunk A
        flow_b = flow_2t[:, T:]        # (B, T-1, 2) — within chunk B
        flow_boundary = flow_2t[:, T - 1]  # (B, 2) — boundary frame A→B[0]

        lr = cosine_lr(step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            # --- chunk A: no prev context ---
            model.reset_temporal()
            model.set_temporal_frames(T)

            t_cont_a = torch.rand(B * T, device=device)
            x_t_a, _ = diffusion.q_sample(src_a, tgt_a, t_cont_a)
            t_emb_a = diffusion._scale_t(t_cont_a)
            v_hat_a = model(src_a, x_t_a, t_emb_a)
            v_tgt_a = tgt_a - src_a
            flow_loss_a = F.mse_loss(v_hat_a, v_tgt_a)
            x0_hat_a = diffusion.predict_target_from_v(x_t_a, t_cont_a, v_hat_a)

            lpips_loss_a = torch.tensor(0.0, device=device)
            if aux_lpips is not None and args.lpips_weight > 0:
                lpips_loss_a = aux_lpips(x0_hat_a, tgt_a).mean()

            # warp consistency within chunk A
            x0_a_chunks = x0_hat_a.view(B, T, C, H, W)
            warp_a = warp_consistency_loss(x0_a_chunks, flow_a)

            # detach KV before chunk B
            model.detach_temporal_kv()

            # --- chunk B: with context from chunk A ---
            t_cont_b = torch.rand(B * T, device=device)
            x_t_b, _ = diffusion.q_sample(src_b, tgt_b, t_cont_b)
            t_emb_b = diffusion._scale_t(t_cont_b)
            v_hat_b = model(src_b, x_t_b, t_emb_b)
            v_tgt_b = tgt_b - src_b
            flow_loss_b = F.mse_loss(v_hat_b, v_tgt_b)
            x0_hat_b = diffusion.predict_target_from_v(x_t_b, t_cont_b, v_hat_b)

            lpips_loss_b = torch.tensor(0.0, device=device)
            if aux_lpips is not None and args.lpips_weight > 0:
                lpips_loss_b = aux_lpips(x0_hat_b, tgt_b).mean()

            x0_b_chunks = x0_hat_b.view(B, T, C, H, W)
            warp_b = warp_consistency_loss(x0_b_chunks, flow_b)

            # cross-chunk boundary warp: warp last frame of A → should match first frame of B
            dx_bnd = flow_boundary[:, 0]
            dy_bnd = flow_boundary[:, 1]
            warped_bnd = warp_by_translation(x0_a_chunks[:, -1], dx_bnd, dy_bnd)
            warp_cross = F.l1_loss(warped_bnd.detach(), x0_b_chunks[:, 0])

            # total loss
            frame_loss = (flow_loss_a + flow_loss_b) * 0.5
            lpips_loss = (lpips_loss_a + lpips_loss_b) * 0.5
            warp_loss = (warp_a + warp_b) * 0.5 + warp_cross
            loss = frame_loss + args.lpips_weight * lpips_loss + args.temporal_weight * warp_loss

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(temporal_params, args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        # reset to single-frame mode after step
        model.set_temporal_frames(1)

        loss_log.append(float(loss.item()))
        warp_log.append(float(warp_loss.item()))
        step += 1

        if step % 100 == 0:
            avg_loss = sum(loss_log[-100:]) / len(loss_log[-100:])
            avg_warp = sum(warp_log[-100:]) / len(warp_log[-100:])
            print(f"step={step:6d}  loss={avg_loss:.5f}  warp={avg_warp:.5f}  lr={lr:.2e}")
            if use_wandb:
                wandb_lib.log({"loss": avg_loss, "warp_loss": avg_warp,
                               "frame_loss": float(frame_loss), "lr": lr}, step=step)

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
            _save_panel(model, ema, diffusion, val_loader, args, device, step, outdir)

    # final model
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
    eval_model = ema.model  # separate EMA copy, always eval
    T = args.num_frames
    lpips_vals, ssim_vals, warp_vals = [], [], []

    for i, batch in enumerate(val_loader):
        if i >= 10:
            break
        source_2t = batch["source"].to(device)
        target_2t = batch["target"].to(device)
        flow_2t   = batch["flow"].to(device)
        B = source_2t.shape[0]
        C, H, W = source_2t.shape[2:]

        src = source_2t[:, :T].reshape(B * T, C, H, W)
        tgt = target_2t[:, :T].reshape(B * T, C, H, W)
        flow_a = flow_2t[:, :T - 1]

        eval_model.reset_temporal()
        eval_model.set_temporal_frames(T)
        samples, _ = diffusion.sample(eval_model, src, image_size=args.image_size,
                                       sample_steps=args.sample_steps)
        eval_model.reset_temporal()

        m = metrics_fn.compute(samples, tgt)
        lpips_vals.append(m["lpips_squeeze"])
        ssim_vals.append(m["ssim"])

        samples_chunks = samples.view(B, T, C, H, W)
        wc = warp_consistency_loss(samples_chunks, flow_a)
        warp_vals.append(float(wc.item()))

    mean_lpips = sum(lpips_vals) / len(lpips_vals)
    mean_ssim  = sum(ssim_vals)  / len(ssim_vals)
    mean_warp  = sum(warp_vals)  / len(warp_vals)
    print(f"[val] step={step}  lpips_sq={mean_lpips:.4f}  ssim={mean_ssim:.4f}  warp_cons={mean_warp:.5f}")

    metrics = {"step": step, "mean_lpips_squeeze": mean_lpips,
               "mean_ssim": mean_ssim, "mean_warp_consistency": mean_warp}
    with open(outdir / f"val_step{step:06d}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if use_wandb:
        import wandb as wandb_lib
        wandb_lib.log({"val/lpips_sq": mean_lpips, "val/ssim": mean_ssim,
                       "val/warp_consistency": mean_warp}, step=step)


@torch.no_grad()
def _save_panel(model, ema, diffusion, val_loader, args, device, step, outdir):
    eval_model = ema.model
    T = args.num_frames

    batch = next(iter(val_loader))
    source_2t = batch["source"].to(device)
    target_2t = batch["target"].to(device)

    src_all = source_2t[0]  # (2T, C, H, W) — first clip
    tgt_all = target_2t[0]

    outputs = []
    eval_model.reset_temporal()
    for chunk_idx in range(2):
        src_chunk = src_all[chunk_idx * T:(chunk_idx + 1) * T]
        eval_model.set_temporal_frames(T)
        samples, _ = diffusion.sample(eval_model, src_chunk, image_size=args.image_size,
                                       sample_steps=args.sample_steps)
        eval_model.detach_temporal_kv()
        outputs.append(samples)
    eval_model.reset_temporal()

    output_all = torch.cat(outputs, dim=0)  # (2T, C, H, W)
    save_video_panel(
        src_all.cpu(), tgt_all.cpu(), output_all.cpu(),
        outdir / f"video_step_{step:06d}",
        fps=8,
    )


if __name__ == "__main__":
    main()
