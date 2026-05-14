"""Run a video clip through exp25, exp27e, exp28c, and exp29 and save as MP4s.

Usage:
    python3 experiments/010_img2img_photo2comics/infer_video.py \\
        /path/to/video.mp4 \\
        --start-frame 60 --end-frame 90 \\
        --outdir out/infer_nat1 \\
        --exp25  out/exp25_lpipsvgg_80k_from_exp23/model.pt \\
        --exp27e out/exp27e_temporal_relposs/model_step_015000.pt \\
        --exp28c out/exp28c_temporal_v2/model_step_020000.pt \\
        --exp29  out/exp29_temporal_declora/model_step_020000.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import resize, InterpolationMode

from src.img2img import Img2ImgDiffusionUNet
from src.img2img.decoder_lora import add_decoder_lora
from src.img2img.flow import FlowConfig, RectifiedImageFlow


# ---------------------------------------------------------------------------
# TemporalAttn V1 (inline for exp27e compatibility)
# Exact parameter names as in exp27e checkpoint.
# ---------------------------------------------------------------------------

def _sinusoidal_v1(T: int, dim: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    args = pos.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(T, 1, device=device)], dim=1)
    return emb.unsqueeze(0)


class TemporalAttnV1(nn.Module):
    """Exp27e architecture: self-attn with pos on Q/K + cross-attn to prev chunk."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.qk_proj = nn.Linear(channels, channels * 2, bias=False)
        self.v_proj  = nn.Linear(channels, channels,     bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.self_gate  = nn.Parameter(torch.zeros(1))

        self.cross_norm   = nn.LayerNorm(channels)
        self.cross_q_proj = nn.Linear(channels, channels, bias=False)
        self.cross_k_proj = nn.Linear(channels, channels, bias=False)
        self.cross_v_proj = nn.Linear(channels, channels, bias=False)
        self.cross_out_proj = nn.Linear(channels, channels, bias=False)
        self.cross_gate = nn.Parameter(torch.zeros(1))

        self._num_frames: int = 1
        self._prev_content: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_content = None

    def detach_kv(self) -> None:
        if self._prev_content is not None:
            self._prev_content = self._prev_content.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self._num_frames
        if T <= 1 and self._prev_content is None:
            return x

        BT, C, H, W = x.shape
        B = BT // T
        HW = H * W
        nh, hd = self.num_heads, self.head_dim

        h = x.view(B, T, C, HW).permute(0, 3, 1, 2).reshape(B * HW, T, C)
        h_norm = self.norm(h)

        pos = _sinusoidal_v1(T, C, x.device, offset=0)
        h_pos = h_norm + pos
        qk = self.qk_proj(h_pos).view(B * HW, T, 2, nh, hd).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v_proj(h_norm).view(B * HW, T, nh, hd).permute(0, 2, 1, 3)
        self_out = F.scaled_dot_product_attention(q, k, v)
        self_out = self_out.permute(0, 2, 1, 3).reshape(B * HW, T, C)
        h = h + self.self_gate.tanh() * self.out_proj(self_out)

        self._prev_content = h

        if self._prev_content is not None:
            prev = self._prev_content
            T_prev = prev.shape[1]
            prev_norm = self.cross_norm(prev)

            pos_k = _sinusoidal_v1(T_prev, C, x.device, offset=0)
            ck = self.cross_k_proj(prev_norm + pos_k)
            ck = ck.view(B * HW, T_prev, nh, hd).permute(0, 2, 1, 3)
            cv = self.cross_v_proj(prev_norm)
            cv = cv.view(B * HW, T_prev, nh, hd).permute(0, 2, 1, 3)

            pos_q = _sinusoidal_v1(T, C, x.device, offset=T)
            cq = self.cross_q_proj(self.cross_norm(h) + pos_q)
            cq = cq.view(B * HW, T, nh, hd).permute(0, 2, 1, 3)

            cross_out = F.scaled_dot_product_attention(cq, ck, cv)
            cross_out = cross_out.permute(0, 2, 1, 3).reshape(B * HW, T, C)
            h = h + self.cross_gate.tanh() * self.cross_out_proj(cross_out)

        return h.view(B, HW, T, C).permute(0, 2, 3, 1).reshape(B * T, C, H, W)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_exp25(ckpt_path: str, device: torch.device) -> tuple:
    """Build any single-frame (non-temporal) Img2ImgDiffusionUNet checkpoint.

    Despite the name, this handles exp25 *and* anything trained via
    train_exp32_prog512.py (exp33 / exp34 / exp35 / exp36) by inferring
    architecture flags from the state_dict when ckpt["config"] doesn't carry
    them explicitly (older exp33 checkpoints).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    sd = ckpt[state_key]
    attn_res = tuple(int(x) for x in str(cfg.get("attn_resolutions", "8")).split(",") if x.strip())

    # Detect arch flags missing from config by inspecting the state_dict.
    if "source_in_stem" in cfg:
        source_in_stem = bool(cfg["source_in_stem"])
    else:
        source_in_stem = sd["in_conv.weight"].shape[1] == 6
    if "no_source_encoder" in cfg:
        use_source_encoder = not cfg["no_source_encoder"]
    else:
        use_source_encoder = any(k.startswith("source_encoder.") for k in sd.keys())
    if "image_size" in cfg:
        image_size = int(cfg["image_size"])
    else:
        present = {i for i in range(1, 5) if f"attn{i}.norm.weight" in sd}
        attn_set = set(attn_res)
        image_size = next(
            (sz for sz in (128, 256, 512, 1024)
             if {i for i in range(1, 5) if (sz >> (i - 1)) in attn_set} == present),
            256,
        )

    # Detect new optional modules (exp34/35/36).
    use_decoder_attn = any(k.startswith("attn_dec") for k in sd.keys())
    use_source_pyramid = any(k.startswith("source_pyramid.") for k in sd.keys())
    use_dit_bottleneck = any(k.startswith("dit_bottleneck.") for k in sd.keys())
    num_dit_blocks = max(
        (int(k.split(".")[2]) + 1 for k in sd.keys()
         if k.startswith("dit_bottleneck.blocks.")),
        default=4,
    )

    print(f"[build] source_in_stem={source_in_stem}  use_source_encoder={use_source_encoder}  "
          f"image_size={image_size}  attn_res={attn_res}  "
          f"decoder_attn={use_decoder_attn}  pyramid={use_source_pyramid}  "
          f"dit={use_dit_bottleneck}({num_dit_blocks} blk)")

    model = Img2ImgDiffusionUNet(
        model_ch=cfg.get("model_ch", 88),
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=cfg.get("upsample_type", "resize_conv"),
        attn_resolutions=attn_res,
        image_size=image_size,
        color_space=cfg.get("color_space", "srgb"),
        use_temporal=False,
        use_decoder_attn=use_decoder_attn,
        use_source_pyramid=use_source_pyramid,
        use_dit_bottleneck=use_dit_bottleneck,
        num_dit_blocks=num_dit_blocks,
    ).to(device).eval()
    model.load_state_dict(sd, strict=True)
    flow_cfg = FlowConfig(**(ckpt.get("flow") or ckpt.get("diffusion") or {}))
    return model, RectifiedImageFlow(flow_cfg, device)


def _build_exp27e(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    attn_res = tuple(int(x) for x in str(cfg.get("attn_resolutions", "8")).split(",") if x.strip())
    mc = cfg.get("model_ch", 88)
    # Infer source_in_stem from in_conv shape (train_temporal.py doesn't store these flags)
    in_conv_in = ckpt["ema_model"]["in_conv.weight"].shape[1]
    source_in_stem = (in_conv_in == 6)
    use_source_encoder = not source_in_stem
    # Build with temporal disabled, then inject V1 modules at all 7 active levels
    model = Img2ImgDiffusionUNet(
        model_ch=mc,
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=cfg.get("upsample_type", "resize_conv"),
        attn_resolutions=attn_res,
        image_size=256,
        color_space=cfg.get("color_space", "srgb"),
        use_temporal=False,
    ).to(device)
    mc = cfg.get("model_ch", 88)
    c2, c3, c4, cm = mc*2, mc*4, mc*4, mc*8
    for name, ch in [("tattn2", c2), ("tattn3", c3), ("tattn4", c4), ("tattn_mid", cm),
                     ("tattn_dec4", c3), ("tattn_dec3", c3), ("tattn_dec2", c2)]:
        setattr(model, name, TemporalAttnV1(ch).to(device))
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    if missing:
        print(f"[exp27e] missing: {missing[:4]}")
    model.eval()
    flow_cfg = FlowConfig(**(ckpt.get("flow") or ckpt.get("diffusion") or {}))
    return model, RectifiedImageFlow(flow_cfg, device)


def _build_exp29(ckpt_path: str, device: torch.device, lora_rank: int = 8) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    attn_res = tuple(int(x) for x in str(cfg.get("attn_resolutions", "8")).split(",") if x.strip())
    in_conv_in = ckpt["ema_model"]["in_conv.weight"].shape[1]
    source_in_stem = (in_conv_in == 6)
    use_source_encoder = not source_in_stem
    # rank from saved config if available
    rank = cfg.get("lora_rank", lora_rank)
    model = Img2ImgDiffusionUNet(
        model_ch=cfg.get("model_ch", 88),
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=cfg.get("upsample_type", "resize_conv"),
        attn_resolutions=attn_res,
        image_size=256,
        color_space=cfg.get("color_space", "srgb"),
        use_temporal=True,
        mask_channels=1,
    ).to(device)
    add_decoder_lora(model, rank=rank)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    if missing:
        print(f"[exp29] missing keys: {missing[:4]}")
    model.eval()
    flow_cfg = FlowConfig(**(ckpt.get("flow") or ckpt.get("diffusion") or {}))
    return model, RectifiedImageFlow(flow_cfg, device)


def _build_exp28c(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    attn_res = tuple(int(x) for x in str(cfg.get("attn_resolutions", "8")).split(",") if x.strip())
    in_conv_in = ckpt["ema_model"]["in_conv.weight"].shape[1]
    source_in_stem = (in_conv_in == 6)
    use_source_encoder = not source_in_stem
    model = Img2ImgDiffusionUNet(
        model_ch=cfg.get("model_ch", 88),
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=cfg.get("upsample_type", "resize_conv"),
        attn_resolutions=attn_res,
        image_size=256,
        color_space=cfg.get("color_space", "srgb"),
        use_temporal=True,
        mask_channels=1,
    ).to(device).eval()
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key], strict=False)
    flow_cfg = FlowConfig(**(ckpt.get("flow") or ckpt.get("diffusion") or {}))
    return model, RectifiedImageFlow(flow_cfg, device)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_frames_exp25(model, diffusion, frames: torch.Tensor,
                        sample_steps: int = 20,
                        fixed_seed: int | None = None,
                        noise_strength: float = 0.3) -> torch.Tensor:
    """frames: (T, 3, H, W). Single-frame pass, no temporal.

    fixed_seed: if set, all frames share the same initial noise offset so the
    ODE starts from source + noise_strength * fixed_noise. This anchors style
    consistently across frames (no per-frame noise variation).
    """
    T, C, H, W = frames.shape
    fixed_noise = None
    if fixed_seed is not None:
        g = torch.Generator(device=frames.device)
        g.manual_seed(fixed_seed)
        fixed_noise = torch.randn(1, C, H, W, device=frames.device, generator=g)

    results = []
    for t in range(T):
        src = frames[t:t+1]
        x = src.clone()
        if fixed_noise is not None:
            x = x + noise_strength * fixed_noise
        ts = torch.linspace(0, 1, sample_steps + 1, device=frames.device)
        for i in range(sample_steps):
            t_cur = ts[i].expand(1)
            dt = float(ts[i+1] - ts[i])
            v = model(src, x, diffusion._scale_t(t_cur))
            x = x + dt * v
        results.append(x.clamp(0, 1))
    return torch.cat(results, dim=0)


@torch.no_grad()
def sample_frames_exp27e(model, diffusion, frames: torch.Tensor,
                         chunk_size: int = 8, sample_steps: int = 20) -> torch.Tensor:
    """frames: (T, 3, H, W). Chunked temporal inference with cross-chunk state."""
    T = frames.shape[0]
    results = []
    model.reset_temporal()
    for start in range(0, T, chunk_size):
        chunk = frames[start:start + chunk_size]  # (Tc, 3, H, W)
        Tc = chunk.shape[0]
        model.set_temporal_frames(Tc)
        x = chunk.clone()
        ts = torch.linspace(0, 1, sample_steps + 1, device=frames.device)
        for i in range(sample_steps):
            t_cur = ts[i].expand(Tc)
            dt = float(ts[i+1] - ts[i])
            v = model(chunk, x, diffusion._scale_t(t_cur))
            x = x + dt * v
        x = x.clamp(0, 1)
        results.append(x)
        model.detach_temporal_kv()
    model.set_temporal_frames(1)
    return torch.cat(results, dim=0)


@torch.no_grad()
def sample_frames_exp29(model, diffusion, frames: torch.Tensor,
                        chunk_size: int = 4, sample_steps: int = 20) -> torch.Tensor:
    """frames: (T, 3, H, W). WAN-style chunked inference with anchor reinjected at every ODE step.

    Same chunking as exp28c (last output frame → anchor of next chunk), but:
    - After each Euler step x = x + dt*v, anchor position is hard-reset to the
      clean anchor frame. This prevents ODE from drifting the anchor and gives
      stronger first-frame conditioning (WAN inpainting style).
    """
    T, C, H, W = frames.shape
    results = []
    prev_last: torch.Tensor | None = None

    for start in range(0, T, chunk_size):
        chunk = frames[start:start + chunk_size]
        Tc = chunk.shape[0]

        if prev_last is not None:
            chunk_in = torch.cat([prev_last, chunk[1:]], dim=0)
            anchor_first = True
        else:
            chunk_in = chunk
            anchor_first = False

        mask = torch.ones(Tc, 1, H, W, device=frames.device)
        if anchor_first:
            mask[0] = 0.0

        model.set_temporal_frames(Tc)
        x = chunk_in.clone()
        ts = torch.linspace(0, 1, sample_steps + 1, device=frames.device)
        for i in range(sample_steps):
            t_cur = ts[i].expand(Tc)
            dt = float(ts[i + 1] - ts[i])
            v = model(chunk_in, x, diffusion._scale_t(t_cur), mask)
            x = x + dt * v
            # Hard-reinject clean anchor after every step (WAN inpainting trick)
            if anchor_first:
                x[0] = chunk_in[0]
        x = x.clamp(0, 1)

        # Restore anchor from clean input (it was reinjected every step anyway)
        if anchor_first:
            x[0] = prev_last[0]

        results.append(x)
        prev_last = x[-1:]

    model.set_temporal_frames(1)
    return torch.cat(results, dim=0)


@torch.no_grad()
def sample_frames_exp28c(model, diffusion, frames: torch.Tensor,
                         chunk_size: int = 4, sample_steps: int = 20) -> torch.Tensor:
    """frames: (T, 3, H, W). WAN-style chunked inference with anchor reinjection."""
    T, C, H, W = frames.shape
    results = []
    prev_last: torch.Tensor | None = None  # last output frame of previous chunk

    for start in range(0, T, chunk_size):
        chunk = frames[start:start + chunk_size]  # (Tc, 3, H, W)
        Tc = chunk.shape[0]

        if prev_last is not None:
            # Replace first frame with previous chunk's last output (anchor)
            chunk_in = torch.cat([prev_last, chunk[1:]], dim=0)
            anchor_first = True
        else:
            chunk_in = chunk
            anchor_first = False

        # frame_mask: 0=anchor, 1=free
        mask = torch.ones(Tc, 1, H, W, device=frames.device)
        if anchor_first:
            mask[0] = 0.0

        model.set_temporal_frames(Tc)
        x = chunk_in.clone()
        ts = torch.linspace(0, 1, sample_steps + 1, device=frames.device)
        for i in range(sample_steps):
            t_cur = ts[i].expand(Tc)
            dt = float(ts[i+1] - ts[i])
            v = model(chunk_in, x, diffusion._scale_t(t_cur), mask)
            x = x + dt * v
        x = x.clamp(0, 1)

        # If anchor was used, keep the anchor frame from input (it's already styled)
        if anchor_first:
            x[0] = prev_last[0]

        results.append(x)
        prev_last = x[-1:]  # last frame → anchor for next chunk

    model.set_temporal_frames(1)
    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_frames(video_path: str, start: int, end: int) -> tuple[torch.Tensor, float]:
    """Read frames [start, end) from video. Returns (T, 3, H, W) uint8, fps."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i < start:
            continue
        if i >= end:
            break
        img = frame.to_image()  # PIL
        frames.append(img)
    container.close()
    import numpy as np
    tensors = [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames]
    return torch.stack(tensors), fps  # (T, 3, H, W) uint8


def write_mp4(frames: torch.Tensor, path: Path, fps: float) -> None:
    """frames: (T, 3, H, W) float32 [0,1] or uint8."""
    if frames.is_floating_point():
        frames_u8 = (frames * 255).clamp(0, 255).byte()
    else:
        frames_u8 = frames
    T, C, H, W = frames_u8.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "slow"}
    import numpy as np
    for t in range(T):
        arr = frames_u8[t].permute(1, 2, 0).numpy()
        av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"  saved {path} ({T} frames @ {fps:.2f}fps, {W}x{H})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--start-frame", type=int, default=60)
    p.add_argument("--end-frame",   type=int, default=90)
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--sample-steps", type=int, default=20)
    # Single-frame model checkpoints. Path(s) — any checkpoint trained via
    # train.py (exp25-era) or train_exp32_prog512.py (exp33+) works; arch is
    # auto-detected from the state_dict in _build_exp25.
    # `--single` accepts one or more paths and produces one MP4 per checkpoint
    # (named after each checkpoint's parent dir). Legacy `--exp25` aliases.
    p.add_argument("--exp25", "--single", dest="single_paths", nargs="+",
                   default=None,
                   help="One or more single-frame model checkpoints (.pt). "
                        "Space-separate paths to run side-by-side. Each writes "
                        "{outdir}/{name}.mp4 named from the checkpoint's parent dir.")
    p.add_argument("--exp27e", default=None, help="exp27e checkpoint (cross-chunk attn); skip if not provided")
    p.add_argument("--exp28c", default=None, help="exp28c checkpoint (WAN anchor); skip if not provided")
    p.add_argument("--exp29",  default=None, help="exp29/exp29b checkpoint (decoder LoRA + anchor reinjection)")
    p.add_argument("--outdir", default="out/infer_nat1")
    p.add_argument("--fixed-seed", type=int, default=None,
                   help="Seed for fixed per-video noise offset in exp25 (all frames same noise)")
    p.add_argument("--noise-strength", type=float, default=0.3,
                   help="Strength of fixed noise added to source before ODE (default 0.3)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"reading frames {args.start_frame}–{args.end_frame-1} from {args.video}")
    raw, fps = read_frames(args.video, args.start_frame, args.end_frame)
    print(f"  {raw.shape[0]} frames @ {fps}fps, native size {raw.shape[3]}x{raw.shape[2]}")

    # resize to model input size, keep float [0,1]
    frames = F.interpolate(raw.float() / 255.0,
                           size=(args.image_size, args.image_size),
                           mode="bilinear", align_corners=False).to(device)

    # save source reference
    write_mp4(frames.cpu(), outdir / "source.mp4", fps)

    # --- single-frame model(s) (flag: --single, legacy --exp25) ---
    # One MP4 per checkpoint, named after the parent dir's first underscore
    # token (so out/exp33_aug32stack_.../exp33_model.pt → exp33.mp4).
    single_paths = args.single_paths or ["out/exp25_lpipsvgg_80k_from_exp23/model.pt"]
    suffix = f"_seed{args.fixed_seed}_ns{args.noise_strength}" if args.fixed_seed is not None else ""
    for ckpt_str in single_paths:
        ckpt_path = Path(ckpt_str)
        out_stem = ckpt_path.parent.name.split("_")[0] if ckpt_path.parent.name else ckpt_path.stem
        print(f"running single-frame model from {ckpt_path} (output: {out_stem}.mp4)...")
        m25, d25 = _build_exp25(ckpt_str, device)
        out25 = sample_frames_exp25(m25, d25, frames, args.sample_steps,
                                    fixed_seed=args.fixed_seed,
                                    noise_strength=args.noise_strength)
        write_mp4(out25.cpu(), outdir / f"{out_stem}{suffix}.mp4", fps)
        del m25, d25

    # --- exp27e (optional) ---
    if args.exp27e:
        print("running exp27e (T=8 chunks, cross-chunk attn)...")
        m27, d27 = _build_exp27e(args.exp27e, device)
        out27 = sample_frames_exp27e(m27, d27, frames, chunk_size=8, sample_steps=args.sample_steps)
        write_mp4(out27.cpu(), outdir / "exp27e.mp4", fps)
        del m27, d27

    # --- exp28c (optional) ---
    if args.exp28c:
        print("running exp28c (T=4 chunks, WAN anchor)...")
        m28, d28 = _build_exp28c(args.exp28c, device)
        out28 = sample_frames_exp28c(m28, d28, frames, chunk_size=4, sample_steps=args.sample_steps)
        write_mp4(out28.cpu(), outdir / "exp28c.mp4", fps)
        del m28, d28

    # --- exp29 (optional) ---
    if args.exp29:
        print("running exp29 (T=4 chunks, WAN anchor + decoder LoRA + anchor every step)...")
        m29, d29 = _build_exp29(args.exp29, device)
        out29 = sample_frames_exp29(m29, d29, frames, chunk_size=4, sample_steps=args.sample_steps)
        write_mp4(out29.cpu(), outdir / "exp29.mp4", fps)
        del m29, d29

    print("done.")


if __name__ == "__main__":
    main()
