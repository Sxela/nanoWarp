from __future__ import annotations

import argparse
import json
import math
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.img2img import EMA, Img2ImgDiffusionUNet, build_train_val_datasets
from src.img2img.colorspace import linear_to_srgb
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.render import save_val_panel
from src.toy_diffusion.render import save_loss_plot
from src.utils.config import apply_yaml_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def git_state(cwd: Path) -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except Exception:
            return ""
    commit = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short=12", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "commit_short": short or "unknown",
        "branch": branch or "unknown",
        "dirty": "yes" if porcelain else "no",
    }


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
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--lr-cosine", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes. 0 = synchronous (slow). 4 default works on most "
                        "machines; 8 saturates GPU at bs=4/128px on a 4090 in benchmarks. Pin_memory + "
                        "persistent_workers + prefetch are auto-enabled when >0; pin_memory is "
                        "auto-skipped at 0 to avoid a Windows-specific slowdown.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--panel-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--outdir", default="out/img2img_v1")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--source-in-stem", action="store_true")
    p.add_argument("--no-source-encoder", action="store_true",
                   help="Drop the ResNet18 source encoder + all FuseBlocks. Source enters via concat at the input "
                        "stem only (forces --source-in-stem). Tests whether multiscale source conditioning is "
                        "actually contributing on top of stem concat.")
    p.add_argument("--model-ch", type=int, default=64,
                   help="Base width of the UNet. All level widths are multiples (1x, 2x, 4x, 4x, 8x). "
                        "Default 64 = original architecture (64/128/256/256/512). 96 = 1.5x wider.")
    p.add_argument("--upsample-type", choices=["resize_conv", "pixel_shuffle"], default="resize_conv",
                   help="Decoder upsampler. resize_conv = nearest interp + 3x3 conv (DDPM-style, default). "
                        "pixel_shuffle = sub-pixel conv with ICNR init (sharper edges, fastai-style).")
    p.add_argument("--attn-resolutions", default="8",
                   help="Comma-separated list of UNet feature-map resolutions where self-attention is applied "
                        "(in addition to the always-on bottleneck attention). e.g. '8' (current default = "
                        "bottleneck only), '8,16', '8,16,32'. Applies after FuseBlock at each level.")
    p.add_argument("--amp", choices=["none", "bf16"], default="none",
                   help="Mixed-precision training. bf16 wraps forward+loss in autocast (fp32 master weights "
                        "preserved by AdamW). Unlocks FlashAttention on Ampere+/Ada GPUs and ~half activation "
                        "memory. Default off for fp32 baseline reproducibility.")
    p.add_argument("--color-space", choices=["srgb", "linear_rgb"], default="srgb",
                   help="Color space the model trains in. srgb (default) = standard PIL/PNG behavior. "
                        "linear_rgb = undo sRGB gamma at the dataset boundary so FM interpolation, noise, "
                        "and bilinear upsampling are physically correct. ImageNet encoder + LPIPS aux + "
                        "panel saves auto-convert linear -> sRGB at their boundaries.")
    p.add_argument("--freeze-source-encoder", choices=["none", "stem", "partial", "all"], default="partial",
                   help="Which ResNet stages of the source encoder to freeze. "
                        "stem=conv1+bn1; partial=stem+layer1 (current default); all=stem+layer1+layer2+layer3+layer4. "
                        "Frozen stages also have their BN running stats locked (eval mode).")
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--lpips-aux-net", choices=["squeeze", "alex", "vgg"], default="squeeze",
                   help="Backbone for the LPIPS auxiliary loss. squeeze=~0.7M params (fastest, default), "
                        "alex=~5M, vgg=~14M (best for style/texture per Gatys/Johnson literature). "
                        "VGG adds ~10ms/step at bs=4 128px. Validation metric stays on SqueezeNet for "
                        "continuity with exp01-exp11.")
    p.add_argument("--prediction-type", choices=["eps", "v"], default="eps",
                   help="Diffusion only. Ignored when --method flow.")
    p.add_argument("--source-dropout", type=float, default=0.0)
    p.add_argument("--high-t-warmup-steps", type=int, default=0,
                   help="Initial steps where t is sampled from [high_t_warmup_low, timesteps). "
                        "For diffusion this is the hard regime (lots of noise). "
                        "For flow matching the analogous hard regime is t near 0; the flag is honored "
                        "as-is and translates to the same integer-timestep range.")
    p.add_argument("--high-t-warmup-low", type=int, default=500)
    p.add_argument("--method", choices=["diffusion", "flow"], default="diffusion",
                   help="diffusion = GaussianImageDiffusion (eps or v). flow = direct img2img rectified flow matching.")
    p.add_argument("--flow-sigma-noise", type=float, default=0.0,
                   help="Flow only. Optional Gaussian noise added to the interpolant for off-path regularization.")
    p.add_argument("--sample-panel-steps", type=int, default=20,
                   help="Number of reverse-sampling steps used when rendering training panels. "
                        "Diffusion: DDIM stride; Flow: Euler steps. Bumps panel cost by ~steps forwards.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="If >0, save model.pt every N steps (in addition to the final save). "
                        "Lets validate.py run mid-training in a parallel process.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt to resume from. Loads model + EMA + optimizer state "
                        "and continues training from the saved step (or step 1 if --resume's checkpoint "
                        "doesn't record a step). New CLI args override the saved config; mismatched "
                        "architecture flags will cause a state_dict load error.")
    p.add_argument("--wandb", action="store_true",
                   help="Log metrics + panels to Weights & Biases. Requires `wandb login` already done.")
    p.add_argument("--wandb-project", default="nanoWarp")
    p.add_argument("--wandb-run-name", default=None,
                   help="Defaults to the basename of --outdir.")
    p.add_argument("--wandb-tags", default=None,
                   help="Comma-separated tag list, e.g. 'flow,no-encoder,exp08'.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="online (default) syncs to wandb cloud. offline writes to disk only (sync later with `wandb sync`).")
    return p.parse_args()


def build_method(args, device):
    if args.method == "flow":
        cfg = FlowConfig(sigma_noise=args.flow_sigma_noise)
        return RectifiedImageFlow(cfg, device), cfg
    cfg = DiffusionConfig(prediction_type=args.prediction_type)
    return GaussianImageDiffusion(cfg, device), cfg


FREEZE_STAGE_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "stem": ("stem",),
    "partial": ("stem", "layer1"),
    "all": ("stem", "layer1", "layer2", "layer3", "layer4"),
}


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def lr_at(step: int, args, total_steps: int) -> float:
    if args.lr_warmup_steps > 0 and step <= args.lr_warmup_steps:
        return args.lr * (step / max(1, args.lr_warmup_steps))
    if args.lr_cosine and total_steps > args.lr_warmup_steps:
        progress = (step - args.lr_warmup_steps) / max(1, total_steps - args.lr_warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr


def t_range_at(step: int, args, timesteps: int) -> tuple[int, int]:
    if args.high_t_warmup_steps > 0 and step <= args.high_t_warmup_steps:
        return max(0, min(args.high_t_warmup_low, timesteps - 1)), timesteps
    return 0, timesteps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    git = git_state(REPO_ROOT)
    print(f"git commit={git['commit_short']} branch={git['branch']} dirty={git['dirty']}")

    wandb = None
    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        run_name = args.wandb_run_name or outdir.name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=tags,
            mode=args.wandb_mode,
            config={**vars(args), **{f"git_{k}": v for k, v in git.items()}},
            dir=str(outdir),
        )
        print(f"wandb run: {wandb.run.name}  ({wandb.run.url})")

    train_ds, val_ds = build_train_val_datasets(
        train_root=args.data_root,
        val_root=args.val_root,
        image_size=args.image_size,
        train_split=args.train_split,
        val_split=args.val_split,
        color_space=args.color_space,
    )
    dl_kwargs: dict = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    if args.num_workers > 0:
        dl_kwargs.update(pin_memory=(device.type == "cuda"), persistent_workers=True, prefetch_factor=2)
    dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    it = cycle(dl)
    val_it = cycle(val_dl)

    freeze_stages = FREEZE_STAGE_PRESETS[args.freeze_source_encoder]
    use_source_encoder = not args.no_source_encoder
    attn_resolutions = tuple(int(x) for x in args.attn_resolutions.split(",") if x.strip())
    model = Img2ImgDiffusionUNet(
        model_ch=args.model_ch,
        pretrained_source_encoder=not args.no_pretrained,
        freeze_source_stages=freeze_stages,
        source_in_stem=args.source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=args.upsample_type,
        attn_resolutions=attn_resolutions,
        image_size=args.image_size,
        color_space=args.color_space,
    ).to(device)
    print(f"use_source_encoder={use_source_encoder} model_ch={args.model_ch} unet_channels={model.unet_channels} upsample={args.upsample_type} attn_resolutions={model.attn_resolutions} color_space={args.color_space}")
    print(f"freeze_source_encoder={args.freeze_source_encoder} stages={freeze_stages}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total_params:,} trainable={trainable_params:,} frozen={total_params - trainable_params:,}")
    ema = EMA(model, decay=args.ema_decay)
    diffusion, method_cfg = build_method(args, device)
    print(f"method={args.method} method_cfg={method_cfg.__dict__}")
    if wandb is not None:
        wandb.config.update({
            "params_total": total_params,
            "params_trainable": trainable_params,
            "params_frozen": total_params - trainable_params,
            "unet_channels": list(model.unet_channels),
            "method_cfg": method_cfg.__dict__,
        }, allow_val_change=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    aux_lpips = None
    if args.lpips_weight > 0:
        raw_lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_aux_net, normalize=True).to(device)
        print(f"lpips_aux_net={args.lpips_aux_net} (val metric stays on squeeze for continuity)")
        if args.color_space == "linear_rgb":
            class _LpipsLinearWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def forward(self, a, b):
                    return self.inner(linear_to_srgb(a).clamp(0, 1), linear_to_srgb(b).clamp(0, 1))
            aux_lpips = _LpipsLinearWrapper(raw_lpips)
        else:
            aux_lpips = raw_lpips

    start_step = 1
    if args.resume:
        ckpt_resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume["model"])
        ema.model.load_state_dict(ckpt_resume["ema_model"])
        if "optimizer" in ckpt_resume:
            opt.load_state_dict(ckpt_resume["optimizer"])
            print(f"resumed optimizer state from {args.resume}")
        else:
            print(f"warning: {args.resume} has no saved optimizer state; AdamW moments will start fresh")
        start_step = int(ckpt_resume.get("step", 0)) + 1
        print(f"resumed from {args.resume} -> starting at step {start_step}")
        if start_step > args.steps:
            raise SystemExit(f"resume step {start_step} > --steps {args.steps}; pass a larger --steps")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    use_amp = amp_dtype is not None and device.type == "cuda"
    print(f"amp={args.amp}  autocast_active={use_amp}")

    losses: list[float] = []
    val_history: list[float] = []
    grad_norms: list[float] = []
    for step in range(start_step, args.steps + 1):
        cur_lr = lr_at(step, args, args.steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        t_low, t_high = t_range_at(step, args, method_cfg.timesteps)

        batch = next(it)
        non_blocking = args.num_workers > 0 and device.type == "cuda"
        source = batch["source"].to(device, non_blocking=non_blocking)
        target = batch["target"].to(device, non_blocking=non_blocking)
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            loss, t, x_t, _noise, _model_out, x0_hat, diffusion_loss, lpips_loss = diffusion.training_loss(
                model,
                source,
                target,
                aux_lpips=aux_lpips,
                aux_lpips_weight=args.lpips_weight,
                t_low=t_low,
                t_high=t_high,
                source_dropout=args.source_dropout,
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip_norm
            )
            grad_norms.append(float(grad_norm))
        opt.step()
        ema.update(model)

        losses.append(float(loss.item()))
        if step % args.log_every == 0 or step == 1:
            extra = f" | grad_norm {grad_norms[-1]:.4f}" if grad_norms else ""
            print(
                f"step {step:5d} | lr {cur_lr:.6f} | t in [{t_low},{t_high}) | "
                f"loss {loss.item():.6f} | diffusion {float(diffusion_loss):.6f} | lpips {float(lpips_loss):.6f}{extra}"
            )
            save_loss_plot(losses, outdir / "loss.png")
            if wandb is not None:
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/method_loss": float(diffusion_loss),
                    "train/lpips_loss": float(lpips_loss),
                    "train/lr": cur_lr,
                    "train/grad_norm": grad_norms[-1] if grad_norms else None,
                    "train/t_low": t_low,
                    "train/t_high": t_high,
                }, step=step)
        if step % args.panel_every == 0 or step == args.steps:
            model.eval()
            sample_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with torch.no_grad(), sample_ctx:
                samples_panel, _ = diffusion.sample(
                    model,
                    source,
                    image_size=args.image_size,
                    sample_steps=args.sample_panel_steps,
                    log_every=None,
                )
            samples_panel = samples_panel.float()
            model.train()
            if args.color_space == "linear_rgb":
                src_disp = linear_to_srgb(source).clamp(0, 1)
                tgt_disp = linear_to_srgb(target).clamp(0, 1)
                samples_disp = linear_to_srgb(samples_panel).clamp(0, 1)
                x0_disp = linear_to_srgb(x0_hat.float()).clamp(0, 1)
            else:
                src_disp, tgt_disp, samples_disp, x0_disp = source, target, samples_panel, x0_hat
            panel_path = outdir / f"panel_step_{step:06d}.png"
            save_val_panel(
                src_disp,
                tgt_disp,
                samples_disp,
                x0_disp,
                panel_path,
                high_t_label="x0_hat_random_t",
            )
            if wandb is not None:
                wandb.log({"train/panel": wandb.Image(str(panel_path))}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            val_losses = []
            for _ in range(args.val_batches):
                vbatch = next(val_it)
                vsource = vbatch["source"].to(device, non_blocking=non_blocking)
                vtarget = vbatch["target"].to(device, non_blocking=non_blocking)
                vloss, *_ = diffusion.training_loss(model, vsource, vtarget)
                val_losses.append(float(vloss.item()))
            mean_val = sum(val_losses) / len(val_losses)
            val_history.append(mean_val)
            print(f"val step {step:5d} | mean_loss {mean_val:.6f}")
            if wandb is not None:
                wandb.log({"val/mean_loss_random_t": mean_val}, step=step)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and step != args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema.model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": vars(args),
                    "diffusion": method_cfg.__dict__,
                    "method": args.method,
                    "step": step,
                },
                outdir / f"model_step_{step:06d}.pt",
            )
            print(f"saved intermediate checkpoint at step {step}")

    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": vars(args),
            "diffusion": method_cfg.__dict__,
            "method": args.method,
            "step": args.steps,
        },
        outdir / "model.pt",
    )
    with open(outdir / "metrics.json", "w") as f:
        json.dump({
            "final_loss": losses[-1],
            "mean_loss_last_50": sum(losses[-50:]) / min(50, len(losses)),
            "last_val_loss": val_history[-1] if val_history else None,
            "mean_grad_norm_last_50": (sum(grad_norms[-50:]) / min(50, len(grad_norms))) if grad_norms else None,
        }, f, indent=2)
    print(f"saved checkpoint to {outdir / 'model.pt'}")
    if wandb is not None:
        wandb.summary["final_loss"] = losses[-1]
        wandb.summary["mean_loss_last_50"] = sum(losses[-50:]) / min(50, len(losses))
        wandb.summary["last_val_loss"] = val_history[-1] if val_history else None
        wandb.finish()


if __name__ == "__main__":
    main()
