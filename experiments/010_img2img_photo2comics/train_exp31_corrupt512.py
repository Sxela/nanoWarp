"""exp31 — single-frame img2img fine-tune at 512x512 with source corruption robustness.

Fine-tunes the exp25 checkpoint (flow + LPIPS-VGG, 80k steps) at 512×512 with random
source corruption (Gaussian blur + JPEG artifacts) to improve robustness to real-video
compression artifacts.

Corruption is applied per image with (1 - clean_prob) probability. When applied, blur
and JPEG are each applied independently with their own probabilities. The target is
always clean. Validation uses clean sources.

Usage:
    OUTDIR=out/exp31_corrupt512_$(date +%Y%m%d_%H%M%S)
    mkdir -p $OUTDIR
    PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \\
    TORCH_HOME=/tmp/torch_home \\
    WANDB_API_KEY=wandb_v1_... \\
    WANDB_CACHE_DIR=/tmp/wandb_cache \\
    WANDB_CONFIG_DIR=/tmp/wandb_config \\
    python3 experiments/010_img2img_photo2comics/train_exp31_corrupt512.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --resume out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --steps 10000 --image-size 512 --aug-resize-scale 2.0 \\
        --lr 2e-5 --lr-min 1e-6 --lr-warmup-steps 200 \\
        --corrupt-blur-max 3.0 --corrupt-jpeg-min 30 --clean-prob 0.2 \\
        --wandb --wandb-run-name exp31_corrupt512 \\
        --outdir $OUTDIR \\
        2>&1 | tee $OUTDIR/train.log
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.img2img import EMA, Img2ImgDiffusionUNet, build_train_val_datasets
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.render import save_val_panel
from src.utils.config import apply_yaml_config


# ---------------------------------------------------------------------------
# Source corruption
# ---------------------------------------------------------------------------

def _jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG round-trip on a (3, H, W) float [0,1] cpu tensor."""
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf)).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).to(img.device)


def corrupt_source(x: torch.Tensor, blur_max: float, jpeg_min: int,
                   clean_prob: float = 0.2,
                   blur_prob: float = 0.7, jpeg_prob: float = 0.7) -> torch.Tensor:
    """Randomly corrupt (N, 3, H, W) float [0,1].

    Each image independently:
    - clean_prob chance: returned unchanged.
    - Otherwise: blur with blur_prob, JPEG with jpeg_prob.
    """
    if clean_prob >= 1.0:
        return x
    out = x.clone()
    for i in range(out.shape[0]):
        if random.random() < clean_prob:
            continue
        img = out[i]
        if blur_max > 0 and random.random() < blur_prob:
            sigma = random.uniform(0.5, blur_max)
            k = max(3, int(2 * math.ceil(3 * sigma) + 1) | 1)
            img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
        if jpeg_min < 95 and random.random() < jpeg_prob:
            quality = random.randint(jpeg_min, 95)
            img = _jpeg_compress(img, quality)
        out[i] = img
    return out


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="exp31: single-frame 512px corruption fine-tune")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--resume", required=True, help="Checkpoint to fine-tune from (exp25 model.pt)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--aug-resize-scale", type=float, default=2.0)
    p.add_argument("--aug-scale-jitter", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--lpips-aux-net", default="vgg", choices=["squeeze", "vgg", "alex"])
    p.add_argument("--amp", default="bf16", choices=["no", "bf16"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--panel-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--sample-panel-steps", type=int, default=20)
    # corruption
    p.add_argument("--corrupt-blur-max", type=float, default=3.0)
    p.add_argument("--corrupt-jpeg-min", type=int, default=30)
    p.add_argument("--clean-prob", type=float, default=0.2,
                   help="Probability of leaving source uncorrupted (0=always corrupt, 1=never)")
    p.add_argument("--corrupt-blur-prob", type=float, default=0.7)
    p.add_argument("--corrupt-jpeg-prob", type=float, default=0.7)
    p.add_argument("--outdir", default="out/exp31_corrupt512")
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


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- wandb ---
    # WANDB_CACHE_DIR + WANDB_CONFIG_DIR must be writable (~/.cache/wandb is not).
    # Set them in the launch env: WANDB_CACHE_DIR=/tmp/wandb_cache WANDB_CONFIG_DIR=/tmp/wandb_config
    wandb = None
    if args.wandb:
        import os
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key  # ensure subprocess inherits it
        import wandb as _wandb
        if api_key:
            _wandb.login(key=api_key, relogin=True)
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] or None
        run_name = args.wandb_run_name or outdir.name
        try:
            _wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=tags,
                config=vars(args),
                dir=str(outdir),
            )
            wandb = _wandb
            print(f"wandb run: {wandb.run.name}  ({wandb.run.url})")
        except Exception as e:
            print(f"[warn] wandb init failed: {type(e).__name__}: {e} — continuing without wandb")

    # --- load model config from checkpoint ---
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    train_cfg = ckpt.get("config", {})
    attn_res = tuple(int(x) for x in str(train_cfg.get("attn_resolutions", "16,32,64")).split(",") if x.strip())
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
    ).to(device)

    state_key = "ema_model" if "ema_model" in ckpt else "model"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:4]}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:4]}")
    resume_step = ckpt.get("step", 0)
    print(f"loaded {state_key} from {args.resume} (step {resume_step})")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total:,}  trainable={trainable:,}")
    print(f"image_size={args.image_size}  aug_resize_scale={args.aug_resize_scale}  attn_res={attn_res}")
    print(f"corruption: blur_max={args.corrupt_blur_max} jpeg_min={args.corrupt_jpeg_min} "
          f"clean_prob={args.clean_prob} blur_prob={args.corrupt_blur_prob} jpeg_prob={args.corrupt_jpeg_prob}")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )

    flow_cfg_dict = ckpt.get("flow", ckpt.get("diffusion", {}))
    flow_cfg = FlowConfig(**flow_cfg_dict) if flow_cfg_dict else FlowConfig()
    diffusion = RectifiedImageFlow(flow_cfg, device)
    print(f"flow_cfg={flow_cfg.__dict__}")

    aux_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=args.lpips_aux_net, normalize=True
    ).to(device)
    for p in aux_lpips.parameters():
        p.requires_grad_(False)
    print(f"lpips_aux_net={args.lpips_aux_net} weight={args.lpips_weight}")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    use_amp = amp_dtype is not None and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    print(f"amp={args.amp}  autocast={use_amp}")

    train_ds, val_ds = build_train_val_datasets(
        train_root=args.data_root,
        image_size=args.image_size,
        train_split="train",
        val_split="val",
        color_space=color_space,
        aug_resize_scale=args.aug_resize_scale,
        aug_scale_jitter=args.aug_scale_jitter,
    )
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=(args.num_workers > 0 and device.type == "cuda"),
                     persistent_workers=(args.num_workers > 0), prefetch_factor=2 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    train_iter = cycle(train_loader)
    val_iter   = cycle(val_loader)

    losses: list[float] = []
    target_step = resume_step + args.steps
    print(f"fine-tuning steps {resume_step}→{target_step}  bs={args.batch_size}")

    for step in range(resume_step + 1, target_step + 1):
        model.train()
        lr = cosine_lr(step - resume_step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = next(train_iter)
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        # Apply source corruption
        source_c = corrupt_source(
            source,
            blur_max=args.corrupt_blur_max,
            jpeg_min=args.corrupt_jpeg_min,
            clean_prob=args.clean_prob,
            blur_prob=args.corrupt_blur_prob,
            jpeg_prob=args.corrupt_jpeg_prob,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss, t, x_t, _noise, _model_out, x0_hat, flow_loss, lpips_loss = diffusion.training_loss(
                model,
                source_c,
                target,
                aux_lpips=aux_lpips,
                aux_lpips_weight=args.lpips_weight,
            )

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            print(f"step {step:6d} | SKIP non-finite loss={loss_val}")
            optimizer.zero_grad(set_to_none=True)
            continue
        if len(losses) >= 10:
            recent_mean = sum(losses[-50:]) / min(50, len(losses))
            if recent_mean > 0 and loss_val > 10.0 * recent_mean:
                print(f"step {step:6d} | SKIP spike {loss_val/recent_mean:.1f}x")
                optimizer.zero_grad(set_to_none=True)
                continue

        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip_norm
            )
        optimizer.step()
        ema.update(model)
        losses.append(loss_val)

        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
            print(f"step={step:6d}  loss={avg:.5f}  flow={float(flow_loss):.5f}"
                  f"  lpips={float(lpips_loss):.5f}  lr={lr:.2e}")
            if wandb is not None:
                wandb.log({"loss": avg, "flow_loss": float(flow_loss),
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
            _run_val(model, ema, diffusion, val_iter, aux_lpips, args, device, step, outdir, wandb)

        if step % args.panel_every == 0:
            _save_panel(ema, diffusion, val_iter, args, device, step, outdir)

    torch.save({
        "step": target_step,
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "config": vars(args),
        "flow": flow_cfg.__dict__,
    }, outdir / "model.pt")
    print("[done] saved final model.pt")
    if wandb is not None:
        wandb.finish()


@torch.no_grad()
def _run_val(model, ema, diffusion, val_iter, aux_lpips, args, device, step, outdir, wandb):
    """Validate on clean sources (no corruption)."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as _L
    from src.img2img.metrics import ValidationMetrics
    metrics_fn = ValidationMetrics(device)

    eval_model = ema.model
    eval_model.eval()
    lpips_vals, ssim_vals = [], []

    for i in range(args.val_batches):
        batch = next(val_iter)
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        ts = torch.linspace(0.0, 1.0, args.sample_panel_steps + 1, device=device)
        x = source.clone()
        for j in range(args.sample_panel_steps):
            t_cur = ts[j].expand(source.shape[0])
            v = eval_model(source, x, diffusion._scale_t(t_cur))
            x = x + float(ts[j + 1] - ts[j]) * v
        samples = x.clamp(0, 1)

        m = metrics_fn.compute(samples, target)
        lpips_vals.append(m["lpips_squeeze"])
        ssim_vals.append(m["ssim"])

    mean_lpips = sum(lpips_vals) / len(lpips_vals)
    mean_ssim  = sum(ssim_vals)  / len(ssim_vals)
    print(f"[val] step={step}  lpips_sq={mean_lpips:.4f}  ssim={mean_ssim:.4f}")

    with open(outdir / f"val_step{step:06d}.json", "w") as f:
        json.dump({"step": step, "lpips_sq": mean_lpips, "ssim": mean_ssim}, f, indent=2)

    if wandb is not None:
        wandb.log({"val/lpips_sq": mean_lpips, "val/ssim": mean_ssim}, step=step)

    # --- nat1 frame-0 inference (visual progress check, no metrics) ---
    _infer_nat1_frame(ema.model, diffusion, args, device, step, outdir)


@torch.no_grad()
def _infer_nat1_frame(eval_model, diffusion, args, device, step, outdir):
    """Run inference on frame 0 of nat1.mp4; save source|result as nat1_step_{step:06d}.png."""
    nat1_path = "/home/researcher/reference/nat1.mp4"
    try:
        import torchvision.io as tvio
        frames, _, _ = tvio.read_video(nat1_path, start_pts=0, end_pts=0.5, pts_unit="sec")
        if frames.shape[0] == 0:
            print("[warn] nat1: no frames decoded")
            return
        frame_np = frames[0].numpy()                                        # (H, W, C) uint8
        frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        frame_t = TF.resize(frame_t, [args.image_size, args.image_size], antialias=True)
        source = frame_t.unsqueeze(0).to(device)                            # (1, C, H, W)

        eval_model.eval()
        ts = torch.linspace(0.0, 1.0, args.sample_panel_steps + 1, device=device)
        x = source.clone()
        for j in range(args.sample_panel_steps):
            t_cur = ts[j].expand(1)
            v = eval_model(source, x, diffusion._scale_t(t_cur))
            x = x + float(ts[j + 1] - ts[j]) * v
        result = x.clamp(0, 1)

        grid = torch.cat([source.cpu(), result.cpu()], dim=3)               # side by side
        TF.to_pil_image(grid[0]).save(outdir / f"nat1_step_{step:06d}.png")
        print(f"[nat1] saved nat1_step_{step:06d}.png")
    except Exception as e:
        print(f"[warn] nat1 inference failed: {e}")


@torch.no_grad()
def _save_panel(ema, diffusion, val_iter, args, device, step, outdir):
    from src.img2img.render import save_val_panel
    eval_model = ema.model
    eval_model.eval()

    batch = next(val_iter)
    source = batch["source"].to(device)[:4]
    target = batch["target"].to(device)[:4]

    ts = torch.linspace(0.0, 1.0, args.sample_panel_steps + 1, device=device)
    x = source.clone()
    for j in range(args.sample_panel_steps):
        t_cur = ts[j].expand(source.shape[0])
        v = eval_model(source, x, diffusion._scale_t(t_cur))
        x = x + float(ts[j + 1] - ts[j]) * v
    samples = x.clamp(0, 1)

    save_val_panel(source.cpu(), target.cpu(), samples.cpu(), samples.cpu(),
                   outdir / f"panel_step_{step:06d}.png")


if __name__ == "__main__":
    main()
