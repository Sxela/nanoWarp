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
from src.img2img.discriminator import PatchDiscriminator
from src.img2img.feature_loss import VGGFeatureLoss
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.gan_loss import hinge_d_loss, hinge_g_loss
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
    p.add_argument("--max-loss-spike-ratio", type=float, default=10.0,
                   help="If current loss > N * mean(last 50 losses), skip backward/step/EMA for this batch. "
                        "Protects against single-bad-batch divergence. 0 = disabled. Default 10.")
    p.add_argument("--no-skip-nonfinite", action="store_true",
                   help="By default we skip backward/step when loss or grad-norm is NaN/Inf. "
                        "Pass this to disable the safety net (not recommended).")
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
    p.add_argument("--feature-content-weight", type=float, default=0.0,
                   help="VGG feature L1 loss weight (fastai 'feature loss' content term). "
                        "Computes L1 between predicted and target VGG features at chosen layers. "
                        "0 = disabled. Typical start: 1.0.")
    p.add_argument("--feature-style-weight", type=float, default=0.0,
                   help="VGG Gram-matrix style L1 loss weight (Gatys / Johnson / fastai style term). "
                        "Captures texture statistics independent of position. 0 = disabled. "
                        "Gram values are tiny so this needs a large multiplier — typical start: 5e3. "
                        "Useful specifically for stylization tasks like photo->anime.")
    p.add_argument("--feature-loss-layers", default="8,15,22",
                   help="Comma-separated VGG16 layer indices for content + style loss. "
                        "Defaults follow fastai/Johnson (after relu2_2, relu3_3, relu4_3). "
                        "Earlier layers capture low-level texture; later layers capture semantic style.")
    p.add_argument("--feature-content-layer-weights", default=None,
                   help="Comma-separated per-layer multipliers for the content L1 term, in the same "
                        "order as --feature-loss-layers. Default uniform 1/n (preserves the old "
                        "averaged behaviour). fastai used '5,15,2' to emphasize mid-level features.")
    p.add_argument("--feature-style-layer-weights", default=None,
                   help="Comma-separated per-layer multipliers for the Gram style L1 term. Default "
                        "uniform 1/n. Try matching fastai's content weights or your own ratio.")
    p.add_argument("--gan-weight", type=float, default=0.0,
                   help="Adversarial loss weight on the generator. 0 = no GAN. Typical pix2pix range "
                        "0.1-1.0. Start small (0.1) since GAN is a regulariser on top of LPIPS/feature "
                        "loss, not a primary signal.")
    p.add_argument("--gan-d-channels", type=int, default=64,
                   help="Base channels for the PatchGAN discriminator (~2-3M params at 64).")
    p.add_argument("--gan-d-layers", type=int, default=3,
                   help="Number of strided conv layers in the discriminator. 3 = pix2pix 70x70 PatchGAN.")
    p.add_argument("--gan-d-lr", type=float, default=1e-4,
                   help="Learning rate for the discriminator's AdamW. Typically lower than generator LR.")
    p.add_argument("--gan-d-beta1", type=float, default=0.5,
                   help="AdamW beta1 for the discriminator. 0.5 is pix2pix / GAN convention "
                        "(vs 0.9 for generator); makes D less momentum-driven and more responsive.")
    p.add_argument("--gan-pretrain-g-steps", type=int, default=0,
                   help="Phase 1 (fastai NoGAN): first N steps train G with LPIPS/feature only, "
                        "no GAN gradient on G, no D updates. Lets G reach a 'reasonable' baseline "
                        "before D enters the game. 0 = skip phase 1.")
    p.add_argument("--gan-pretrain-d-steps", type=int, default=0,
                   help="Phase 2 (fastai NoGAN): after G pretrain, freeze G for M steps while D "
                        "trains alone on (real, current-G-output) pairs. Calibrates D before "
                        "alternating GAN. 0 = skip phase 2 (D starts fresh in phase 3).")
    p.add_argument("--aug-resize-scale", type=float, default=1.10,
                   help="Intermediate resize ratio in PairedImageAugment before random crop. "
                        "1.10 = 10%% extra room (default, conservative). 1.5-2.0 = aggressive zoom-crop, "
                        "useful when source images are high-resolution (e.g. 1024px) since the default "
                        "throws away most source detail. Higher values give more crop variation per pair.")
    p.add_argument("--aug-scale-jitter", type=float, default=0.10,
                   help="Affine scale jitter range applied to the (already resized) intermediate. "
                        "Default 0.10 = ±10%% scale variation. Increase to 0.15-0.20 for more zoom variation "
                        "when training data is plentiful (e.g. 1k+ pairs).")
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


def gan_phase_at(step: int, args, discriminator) -> str:
    """fastai NoGAN three-phase schedule. Returns 'g_pretrain', 'd_pretrain', or 'full'."""
    if discriminator is None or args.gan_weight <= 0:
        return "off"
    if step <= args.gan_pretrain_g_steps:
        return "g_pretrain"
    if step <= args.gan_pretrain_g_steps + args.gan_pretrain_d_steps:
        return "d_pretrain"
    return "full"


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
        aug_resize_scale=args.aug_resize_scale,
        aug_scale_jitter=args.aug_scale_jitter,
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
        print(f"lpips_aux_net={args.lpips_aux_net} weight={args.lpips_weight} (val metric stays on squeeze for continuity)")

    discriminator = None
    opt_d = None
    if args.gan_weight > 0:
        discriminator = PatchDiscriminator(
            in_channels=6, base_channels=args.gan_d_channels, n_layers=args.gan_d_layers
        ).to(device)
        opt_d = torch.optim.AdamW(
            discriminator.parameters(), lr=args.gan_d_lr, betas=(args.gan_d_beta1, 0.999)
        )
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"discriminator PatchGAN ch={args.gan_d_channels} layers={args.gan_d_layers}  "
              f"params={d_params:,}  d_lr={args.gan_d_lr}  d_beta1={args.gan_d_beta1}  "
              f"gan_weight={args.gan_weight}")

    feature_loss_fn = None
    if args.feature_content_weight > 0 or args.feature_style_weight > 0:
        feat_layers = tuple(int(x) for x in args.feature_loss_layers.split(",") if x.strip())
        content_lw = tuple(float(x) for x in args.feature_content_layer_weights.split(",")) if args.feature_content_layer_weights else None
        style_lw = tuple(float(x) for x in args.feature_style_layer_weights.split(",")) if args.feature_style_layer_weights else None
        feature_loss_fn = VGGFeatureLoss(
            layers=feat_layers,
            content_weight=args.feature_content_weight,
            style_weight=args.feature_style_weight,
            content_layer_weights=content_lw,
            style_layer_weights=style_lw,
        ).to(device)
        print(
            f"feature_loss layers={feat_layers} content_w={args.feature_content_weight} "
            f"style_w={args.feature_style_weight} content_lw={feature_loss_fn.content_layer_weights} "
            f"style_lw={feature_loss_fn.style_layer_weights}"
        )

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
        if discriminator is not None and "discriminator" in ckpt_resume:
            discriminator.load_state_dict(ckpt_resume["discriminator"])
            if "optimizer_d" in ckpt_resume and opt_d is not None:
                opt_d.load_state_dict(ckpt_resume["optimizer_d"])
            print("resumed discriminator + opt_d state")
        elif discriminator is not None:
            print("warning: discriminator enabled but checkpoint has no discriminator state; D starts fresh")
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
            feature_total = torch.tensor(0.0, device=device)
            feature_content = torch.tensor(0.0, device=device)
            feature_style = torch.tensor(0.0, device=device)
            if feature_loss_fn is not None:
                if args.color_space == "linear_rgb":
                    fpred = linear_to_srgb(x0_hat).clamp(0, 1)
                    ftgt = linear_to_srgb(target).clamp(0, 1)
                else:
                    fpred = x0_hat.clamp(0, 1)
                    ftgt = target.clamp(0, 1)
                fl = feature_loss_fn(fpred, ftgt)
                feature_total = fl["total"]
                feature_content = fl["content"].detach()
                feature_style = fl["style"].detach()
                loss = loss + feature_total
            g_gan_loss = torch.tensor(0.0, device=device)
            phase = gan_phase_at(step, args, discriminator)
            # GAN term enters G's loss only in the "full" phase. During "g_pretrain"
            # we train G on LPIPS/feature only. During "d_pretrain" G is frozen below.
            if phase == "full":
                d_fake_for_g = discriminator(source, x0_hat)
                g_gan_loss = hinge_g_loss(d_fake_for_g)
                loss = loss + args.gan_weight * g_gan_loss

        # Safety: skip backward/step if loss is non-finite or a huge spike vs recent average.
        loss_val = float(loss.item())
        skip_reason = None
        if not args.no_skip_nonfinite and not math.isfinite(loss_val):
            skip_reason = "non-finite-loss"
        elif args.max_loss_spike_ratio > 0 and len(losses) >= 10:
            recent_mean = sum(losses[-50:]) / min(50, len(losses))
            if recent_mean > 0 and loss_val > args.max_loss_spike_ratio * recent_mean:
                skip_reason = f"spike-{loss_val / recent_mean:.1f}x"
        if skip_reason is not None:
            feat_part = (f" | feat_c {float(feature_content):.4f} feat_s {float(feature_style):.6f}"
                         if feature_loss_fn is not None else "")
            print(
                f"step {step:5d} | SKIP ({skip_reason}) | loss {loss_val:.4f} | "
                f"diffusion {float(diffusion_loss):.4f} | lpips {float(lpips_loss):.4f}{feat_part}"
            )
            if wandb is not None:
                wandb.log({
                    "train/skipped_step": 1,
                    "train/skipped_loss": loss_val if math.isfinite(loss_val) else 0.0,
                    "train/skip_reason_non_finite": int(skip_reason == "non-finite-loss"),
                }, step=step)
            # Record a finite placeholder so the spike detector doesn't get derailed
            # by a single corrupted value, but don't backward/step/ema.
            if math.isfinite(loss_val):
                losses.append(loss_val)
            opt.zero_grad(set_to_none=True)
            continue

        # G update: skipped entirely during phase 2 (D pretrain).
        if phase != "d_pretrain":
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip_norm
                )
                if not args.no_skip_nonfinite and not torch.isfinite(grad_norm).item():
                    print(f"step {step:5d} | SKIP (non-finite-grad-norm) | grad_norm {float(grad_norm)}")
                    if wandb is not None:
                        wandb.log({"train/skipped_step": 1, "train/skip_reason_non_finite_grad": 1}, step=step)
                    opt.zero_grad(set_to_none=True)
                    continue
                grad_norms.append(float(grad_norm))
            opt.step()
            ema.update(model)
        else:
            # During D pretrain: clear any G gradients that might have accumulated
            # from the autocast G forward (shouldn't be needed since we didn't
            # backward, but defensive).
            opt.zero_grad(set_to_none=True)

        # D update — happens in phases 2 (d_pretrain) and 3 (full). Skipped in
        # phase 1 (g_pretrain) since D doesn't exist conceptually yet.
        d_loss_val = 0.0
        d_real_score = 0.0
        d_fake_score = 0.0
        if discriminator is not None and phase in ("d_pretrain", "full"):
            with amp_ctx:
                d_real = discriminator(source, target)
                d_fake_for_d = discriminator(source, x0_hat.detach())
                d_loss = hinge_d_loss(d_real, d_fake_for_d)
                d_real_score = float(d_real.mean().item())
                d_fake_score = float(d_fake_for_d.mean().item())
            d_loss_val = float(d_loss.item())
            if math.isfinite(d_loss_val):
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.grad_clip_norm)
                opt_d.step()
            else:
                # Skip D step if its loss is NaN/Inf.
                opt_d.zero_grad(set_to_none=True)

        losses.append(loss_val)
        if step % args.log_every == 0 or step == 1:
            extra = f" | grad_norm {grad_norms[-1]:.4f}" if grad_norms else ""
            feat_extra = f" | feat_c {float(feature_content):.4f} feat_s {float(feature_style):.6f}" if feature_loss_fn is not None else ""
            gan_extra = (f" | phase {phase} g_gan {float(g_gan_loss):.4f} d_loss {d_loss_val:.4f} "
                         f"d_real {d_real_score:+.3f} d_fake {d_fake_score:+.3f}"
                         if discriminator is not None else "")
            print(
                f"step {step:5d} | lr {cur_lr:.6f} | t in [{t_low},{t_high}) | "
                f"loss {loss.item():.6f} | diffusion {float(diffusion_loss):.6f} | lpips {float(lpips_loss):.6f}{feat_extra}{gan_extra}{extra}"
            )
            save_loss_plot(losses, outdir / "loss.png")
            if wandb is not None:
                wandb_metrics = {
                    "train/loss": float(loss.item()),
                    "train/method_loss": float(diffusion_loss),
                    "train/lpips_loss": float(lpips_loss),
                    "train/lr": cur_lr,
                    "train/grad_norm": grad_norms[-1] if grad_norms else None,
                    "train/t_low": t_low,
                    "train/t_high": t_high,
                }
                if feature_loss_fn is not None:
                    wandb_metrics["train/feature_content"] = float(feature_content)
                    wandb_metrics["train/feature_style"] = float(feature_style)
                    wandb_metrics["train/feature_total"] = float(feature_total)
                if discriminator is not None:
                    wandb_metrics["train/g_gan_loss"] = float(g_gan_loss)
                    wandb_metrics["train/d_loss"] = d_loss_val
                    wandb_metrics["train/d_real_score"] = d_real_score
                    wandb_metrics["train/d_fake_score"] = d_fake_score
                wandb.log(wandb_metrics, step=step)
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
            ckpt_payload = {
                "model": model.state_dict(),
                "ema_model": ema.model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": vars(args),
                "diffusion": method_cfg.__dict__,
                "method": args.method,
                "step": step,
            }
            if discriminator is not None:
                ckpt_payload["discriminator"] = discriminator.state_dict()
                ckpt_payload["optimizer_d"] = opt_d.state_dict()
            torch.save(ckpt_payload, outdir / f"model_step_{step:06d}.pt")
            print(f"saved intermediate checkpoint at step {step}")

    final_payload = {
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": vars(args),
        "diffusion": method_cfg.__dict__,
        "method": args.method,
        "step": args.steps,
    }
    if discriminator is not None:
        final_payload["discriminator"] = discriminator.state_dict()
        final_payload["optimizer_d"] = opt_d.state_dict()
    torch.save(final_payload, outdir / "model.pt")
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
