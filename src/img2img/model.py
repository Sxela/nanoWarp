from __future__ import annotations

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .colorspace import linear_to_srgb
from .dit import DiTBottleneck
from .source_pyramid import CrossAttnCond, FiLM, SourcePyramid
from .temporal import TemporalAttn


RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(t, self.dim))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class SourceEncoder(nn.Module):
    """ResNet18-compatible source encoder with optional ImageNet weights and freeze controls."""

    def __init__(self, in_ch: int = 3, pretrained: bool = True, freeze_stages: tuple[str, ...] = ("stem", "layer1")):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

        self._frozen_stage_names: tuple[str, ...] = tuple(freeze_stages) if freeze_stages else ()

        if pretrained:
            self.load_pretrained_weights()
        if freeze_stages:
            self.freeze_stages(freeze_stages)

    def _stage_modules(self, stage: str):
        if stage == "stem":
            return [self.conv1, self.bn1]
        return [getattr(self, stage)]

    def load_pretrained_weights(self):
        try:
            state = load_state_dict_from_url(RESNET18_URL, map_location="cpu", progress=True)
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing:
                warnings.warn(f"SourceEncoder missing pretrained keys: {missing}")
            if unexpected:
                warnings.warn(f"SourceEncoder unexpected pretrained keys: {unexpected}")
        except Exception as e:
            warnings.warn(f"Falling back to random SourceEncoder init; failed to load ResNet18 weights: {e}")

    def freeze_stages(self, stages: tuple[str, ...] = ("stem", "layer1")):
        self._frozen_stage_names = tuple(stages)
        for stage in stages:
            for module in self._stage_modules(stage):
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for stage in self._frozen_stage_names:
            for module in self._stage_modules(stage):
                module.eval()
        return self

    def trainable_summary(self) -> dict[str, bool]:
        return {
            "stem": any(p.requires_grad for p in list(self.conv1.parameters()) + list(self.bn1.parameters())),
            "layer1": any(p.requires_grad for p in self.layer1.parameters()),
            "layer2": any(p.requires_grad for p in self.layer2.parameters()),
            "layer3": any(p.requires_grad for p in self.layer3.parameters()),
            "layer4": any(p.requires_grad for p in self.layer4.parameters()),
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.relu(self.bn1(self.conv1(x)))  # 64x64 for 128 input
        x = self.maxpool(f0)                     # 32x32
        f1 = self.layer1(x)        # 32x32
        f2 = self.layer2(f1)       # 16x16
        f3 = self.layer3(f2)       # 8x8
        f4 = self.layer4(f3)       # 4x4
        return [f0, f1, f2, f3, f4]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class FuseBlock(nn.Module):
    def __init__(self, unet_ch: int, src_ch: int):
        super().__init__()
        self.src_proj = nn.Conv2d(src_ch, unet_ch, 1)
        self.fuse = nn.Conv2d(unet_ch * 2, unet_ch, 3, padding=1)

    def forward(self, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if s.shape[-2:] != u.shape[-2:]:
            s = F.interpolate(s, size=u.shape[-2:], mode="bilinear", align_corners=False)
        s = self.src_proj(s)
        return self.fuse(torch.cat([u, s], dim=1))


class BottleneckAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.heads

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, self.heads, head_dim, h * w).transpose(2, 3)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


def icnr_init(weight: torch.Tensor, scale: int = 2, init=nn.init.kaiming_normal_) -> None:
    """ICNR (Aitken et al. 2017): initialise a sub-pixel conv so that PixelShuffle
    produces the same output as nearest-neighbor upsampling at step 0. This is what
    keeps PixelShuffle from emitting a checkerboard pattern early in training.

    `weight` shape is (out_ch, in_ch, kH, kW) with out_ch = base_ch * scale**2.
    """
    out_ch, in_ch = weight.shape[:2]
    spatial = weight.shape[2:]
    base_ch = out_ch // (scale ** 2)
    sub = torch.empty(base_ch, in_ch, *spatial, device=weight.device, dtype=weight.dtype)
    init(sub)
    sub = sub.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        weight.copy_(sub)


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel conv upsampling with ICNR init.

    Drop-in replacement for `Upsample`: same in_ch == out_ch, 2x spatial.
    Tends to produce sharper edges than resize+conv in image-translation tasks
    (cf. fastai's UNet). The cost is one extra hyperparameter (the init).
    """

    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        icnr_init(self.conv.weight, scale=scale)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class Img2ImgDiffusionUNet(nn.Module):
    """Pixel-space conditional diffusion skeleton for photo -> comics."""

    def __init__(
        self,
        in_ch: int = 3,
        model_ch: int = 64,
        out_ch: int = 3,
        time_dim: int = 128,
        pretrained_source_encoder: bool = True,
        freeze_source_stages: tuple[str, ...] = ("stem", "layer1"),
        source_in_stem: bool = False,
        use_source_encoder: bool = True,
        upsample_type: str = "resize_conv",
        attn_resolutions: tuple[int, ...] = (8,),
        image_size: int = 128,
        color_space: str = "srgb",
        use_temporal: bool = False,
        mask_channels: int = 0,
        use_source_pyramid: bool = False,
        use_decoder_attn: bool = False,
        use_dit_bottleneck: bool = False,
        num_dit_blocks: int = 4,
        dit_mlp_ratio: float = 4.0,
        use_cross_attn_cond: bool = False,
    ):
        super().__init__()
        if upsample_type not in ("resize_conv", "pixel_shuffle"):
            raise ValueError(f"upsample_type must be 'resize_conv' or 'pixel_shuffle', got {upsample_type!r}")
        if color_space not in ("srgb", "linear_rgb"):
            raise ValueError(f"color_space must be 'srgb' or 'linear_rgb', got {color_space!r}")
        if not use_source_encoder:
            source_in_stem = True
        self.use_source_encoder = use_source_encoder
        self.source_in_stem = source_in_stem
        self.upsample_type = upsample_type
        self.attn_resolutions = tuple(sorted(set(int(r) for r in attn_resolutions)))
        self.image_size = image_size
        self.color_space = color_space
        self.use_temporal = use_temporal
        self.mask_channels = mask_channels
        self.use_source_pyramid = use_source_pyramid
        self.use_decoder_attn = use_decoder_attn
        self.use_dit_bottleneck = use_dit_bottleneck
        self._temporal_num_frames: int = 1  # set via set_temporal_frames()
        if use_source_encoder:
            self.source_encoder = SourceEncoder(
                in_ch=3,
                pretrained=pretrained_source_encoder,
                freeze_stages=freeze_source_stages,
            )
        else:
            self.source_encoder = None
        self.time_mlp = TimeMLP(time_dim)

        # WAN-style mask channel injection: a zero-init 1×1 conv adds to the
        # stem features. Keeps in_conv shape unchanged → loads cleanly from any
        # spatial checkpoint. mask=1 means "generate this frame", mask=0 means
        # "this frame is a given anchor (e.g. reinject from previous chunk)".
        c1_early = model_ch  # need c1 before the block below computes it
        if mask_channels > 0:
            self.mask_proj = nn.Conv2d(mask_channels, c1_early, 1, bias=False)
            nn.init.zeros_(self.mask_proj.weight)
        else:
            self.mask_proj = None

        # UNet widths are multiples of model_ch (base = 64 in the original design).
        # Scaling model_ch scales the whole UNet by the same factor.
        c1 = model_ch          # level 1 (full res)
        c2 = model_ch * 2      # level 2
        c3 = model_ch * 4      # level 3
        c4 = model_ch * 4      # level 4
        cm = model_ch * 8      # bottleneck
        self.unet_channels = (c1, c2, c3, c4, cm)

        stem_in_ch = in_ch + 3 if source_in_stem else in_ch
        self.in_conv = nn.Conv2d(stem_in_ch, c1, 3, padding=1)

        self.down1 = ResBlock(c1, c1, time_dim * 4)
        self.ds1 = Downsample(c1)

        self.down2 = ResBlock(c1, c2, time_dim * 4)
        self.ds2 = Downsample(c2)

        self.down3 = ResBlock(c2, c3, time_dim * 4)
        self.ds3 = Downsample(c3)

        self.down4 = ResBlock(c3, c4, time_dim * 4)
        self.ds4 = Downsample(c4)

        # Bottleneck. `mid1` always projects c4 → cm. `mid_attn` and `mid2` are
        # either the original conv-attn-conv stack OR replaced by a stack of
        # DiT blocks (use_dit_bottleneck=True). The DiT stack is identity at
        # init via adaLN-zero, so a non-DiT checkpoint loads cleanly via
        # strict=False when use_dit_bottleneck is flipped on.
        self.mid1 = ResBlock(c4, cm, time_dim * 4)
        if use_dit_bottleneck:
            self.mid_attn = None
            self.mid2 = None
            self.dit_bottleneck = DiTBottleneck(
                dim=cm,
                t_emb_dim=time_dim * 4,
                num_blocks=num_dit_blocks,
                mlp_ratio=dit_mlp_ratio,
            )
        else:
            self.mid_attn = BottleneckAttention(cm)
            self.mid2 = ResBlock(cm, cm, time_dim * 4)
            self.dit_bottleneck = None

        # Temporal attention — inserted after every encoder/bottleneck/decoder block.
        # AnimateDiff-style: all levels, zero-init gates → identity at init.
        # Only created when use_temporal=True; absent modules keep state_dict clean.
        # Temporal attention at all levels ≤128px. 256px skipped: B×HW=131072 exceeds
        # CUDA attention kernel limits, and fine structure is already source-anchored.
        ta = lambda ch: TemporalAttn(ch) if use_temporal else None
        # 256px and 128px levels skipped: B×H×W too large for CUDA attention
        # kernels at typical batch sizes. Temporal attention at 64px and below
        # provides good consistency at manageable memory cost.
        self.tattn1     = None       # 256px — skipped (B×HW > kernel limit)
        self.tattn2     = None       # 128px — skipped (B×HW > kernel limit)
        self.tattn3     = ta(c3)    # encoder 64px
        self.tattn4     = ta(c4)    # encoder 32px
        self.tattn_mid  = ta(cm)    # bottleneck 16px
        self.tattn_dec4 = ta(c3)    # decoder 32px
        self.tattn_dec3 = ta(c3)    # decoder 64px
        self.tattn_dec2 = None      # 128px — skipped
        self.tattn_dec1 = None      # 256px — skipped

        # Encoder-side resolutions for the four levels at this image_size.
        # h1 stays at image_size, then each ds halves.
        level_resolutions = (
            image_size,         # after down1 (level 1)
            image_size // 2,    # after down2 (level 2)
            image_size // 4,    # after down3 (level 3)
            image_size // 8,    # after down4 (level 4)
        )
        level_channels = (c1, c2, c3, c4)
        attn_set = set(self.attn_resolutions)
        # Conditionally create per-level self-attention. None when not requested,
        # which keeps the state_dict empty and old checkpoints loadable.
        # The bottleneck attention (8x8 by default) is mid_attn above and is
        # always present.
        self.attn1 = BottleneckAttention(level_channels[0]) if level_resolutions[0] in attn_set else None
        self.attn2 = BottleneckAttention(level_channels[1]) if level_resolutions[1] in attn_set else None
        self.attn3 = BottleneckAttention(level_channels[2]) if level_resolutions[2] in attn_set else None
        self.attn4 = BottleneckAttention(level_channels[3]) if level_resolutions[3] in attn_set else None

        # Optional decoder-side spatial self-attention, mirroring the encoder.
        # SD/SDXL UNets put attn symmetrically on encoder + decoder; ours
        # originally only had encoder attn + mid_attn, leaving the decoder
        # without spatial mixing at any resolution. With use_decoder_attn=True
        # we add BottleneckAttention at the same resolutions as encoder attn
        # but operating on decoder output channels.
        #   dec4 output → c3 @ H/8
        #   dec3 output → c3 @ H/4
        #   dec2 output → c2 @ H/2
        #   dec1 output → c1 @ H
        if use_decoder_attn:
            self.attn_dec4 = BottleneckAttention(c3) if level_resolutions[3] in attn_set else None
            self.attn_dec3 = BottleneckAttention(c3) if level_resolutions[2] in attn_set else None
            self.attn_dec2 = BottleneckAttention(c2) if level_resolutions[1] in attn_set else None
            self.attn_dec1 = BottleneckAttention(c1) if level_resolutions[0] in attn_set else None
        else:
            self.attn_dec4 = None
            self.attn_dec3 = None
            self.attn_dec2 = None
            self.attn_dec1 = None

        if use_source_encoder:
            # SourceEncoder (ResNet18) feature widths are fixed: 64, 64, 128, 256, 512.
            self.fuse1 = FuseBlock(c1, 64)
            self.fuse2 = FuseBlock(c2, 64)
            self.fuse3 = FuseBlock(c3, 128)
            self.fuse4 = FuseBlock(c4, 256)
            self.mid_fuse = FuseBlock(cm, 512)
        else:
            self.fuse1 = self.fuse2 = self.fuse3 = self.fuse4 = self.mid_fuse = None

        UpModule = PixelShuffleUpsample if upsample_type == "pixel_shuffle" else Upsample
        self.up4 = UpModule(cm)
        self.dec4 = ResBlock(cm + c4, c3, time_dim * 4)
        self.up3 = UpModule(c3)
        self.dec3 = ResBlock(c3 + c3, c3, time_dim * 4)
        self.up2 = UpModule(c3)
        self.dec2 = ResBlock(c3 + c2, c2, time_dim * 4)
        self.up1 = UpModule(c2)
        self.dec1 = ResBlock(c2 + c1, c1, time_dim * 4)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, out_ch, 3, padding=1)

        # Optional in-model source feature pyramid + FiLM modulation of the
        # decoder. Zero-init FiLM → identity at init; old (no-pyramid)
        # checkpoints load cleanly into a pyramid-enabled model via
        # strict=False (the pyramid + FiLM weights are then missing keys
        # initialised here, which is the desired warm-start behaviour).
        # Decoder block output channels after each dec*: dec4→c3 @ H/8,
        # dec3→c3 @ H/4, dec2→c2 @ H/2, dec1→c1 @ H. Pyramid stages emit
        # (c1, c2, c3, c4) at (H, H/2, H/4, H/8), so the channel widths line
        # up with the matching-resolution decoder activations.
        if use_source_pyramid:
            self.source_pyramid = SourcePyramid(channels=(c1, c2, c3, c4))
            self.film_dec4 = FiLM(cond_ch=c4, target_ch=c3)   # H/8
            self.film_dec3 = FiLM(cond_ch=c3, target_ch=c3)   # H/4
            self.film_dec2 = FiLM(cond_ch=c2, target_ch=c2)   # H/2
            self.film_dec1 = FiLM(cond_ch=c1, target_ch=c1)   # H
            # Optional cross-attn at H/8 decoder level (deepest non-bottleneck
            # level where pyramid feature is available). Token count at 256px
            # input = 32*32 = 1024 — practical. Identity-at-init via zero-init
            # output proj in CrossAttnCond, so this is safe to insert alongside
            # FiLM; older checkpoints (no cross-attn) auto-detect in ckpt.py.
            if use_cross_attn_cond:
                self.cross_attn_dec4 = CrossAttnCond(target_ch=c3, cond_ch=c4)
            else:
                self.cross_attn_dec4 = None
        else:
            self.source_pyramid = None
            self.cross_attn_dec4 = None
            self.film_dec4 = None
            self.film_dec3 = None
            self.film_dec2 = None
            self.film_dec1 = None

    # ------------------------------------------------------------------
    # Temporal state helpers
    # ------------------------------------------------------------------

    def _temporal_modules(self) -> list[TemporalAttn]:
        return [ta for ta in (
            self.tattn1, self.tattn2, self.tattn3, self.tattn4, self.tattn_mid,
            self.tattn_dec4, self.tattn_dec3, self.tattn_dec2, self.tattn_dec1,
        ) if ta is not None]

    def set_temporal_frames(self, n: int) -> None:
        """Set T before a forward pass over a sequence of n frames."""
        self._temporal_num_frames = n
        for ta in self._temporal_modules():
            ta._num_frames = n

    def reset_temporal(self) -> None:
        """Clear prev KV state — call at the start of each new video."""
        self._temporal_num_frames = 1
        for ta in self._temporal_modules():
            ta.reset()
            ta._num_frames = 1

    def detach_temporal_kv(self) -> None:
        """Detach stored KV after chunk A so chunk B doesn't backprop through it."""
        for ta in self._temporal_modules():
            ta.detach_kv()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source: torch.Tensor,
        noisy_target: torch.Tensor,
        t: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        stem_input = torch.cat([source, noisy_target], dim=1) if self.source_in_stem else noisy_target
        x0 = self.in_conv(stem_input)
        # inject mask channel (zero-init → no effect at init)
        if self.mask_proj is not None and frame_mask is not None:
            x0 = x0 + self.mask_proj(frame_mask)

        def _ta(feat, mod):
            return mod(feat) if mod is not None else feat

        # Source pyramid features (one set per source, independent of t).
        # pyr_feats = [f_H, f_H/2, f_H/4, f_H/8] with channels (c1, c2, c3, c4).
        pyr_feats = self.source_pyramid(source) if self.source_pyramid is not None else None

        def _film(feat, mod, idx):
            return mod(feat, pyr_feats[idx]) if (mod is not None and pyr_feats is not None) else feat

        if self.use_source_encoder:
            encoder_input = linear_to_srgb(source).clamp(0, 1) if self.color_space == "linear_rgb" else source
            src_feats = self.source_encoder(encoder_input)
            h1 = self.fuse1(self.down1(x0, t_emb), src_feats[0])
            if self.attn1 is not None: h1 = self.attn1(h1)
            h1 = _ta(h1, self.tattn1)
            h2 = self.fuse2(self.down2(self.ds1(h1), t_emb), src_feats[1])
            if self.attn2 is not None: h2 = self.attn2(h2)
            h2 = _ta(h2, self.tattn2)
            h3 = self.fuse3(self.down3(self.ds2(h2), t_emb), src_feats[2])
            if self.attn3 is not None: h3 = self.attn3(h3)
            h3 = _ta(h3, self.tattn3)
            h4 = self.fuse4(self.down4(self.ds3(h3), t_emb), src_feats[3])
            if self.attn4 is not None: h4 = self.attn4(h4)
            h4 = _ta(h4, self.tattn4)
            mid = self.mid1(self.ds4(h4), t_emb)
            mid = self.mid_fuse(mid, src_feats[4])
        else:
            h1 = self.down1(x0, t_emb)
            if self.attn1 is not None: h1 = self.attn1(h1)
            h1 = _ta(h1, self.tattn1)
            h2 = self.down2(self.ds1(h1), t_emb)
            if self.attn2 is not None: h2 = self.attn2(h2)
            h2 = _ta(h2, self.tattn2)
            h3 = self.down3(self.ds2(h2), t_emb)
            if self.attn3 is not None: h3 = self.attn3(h3)
            h3 = _ta(h3, self.tattn3)
            h4 = self.down4(self.ds3(h3), t_emb)
            if self.attn4 is not None: h4 = self.attn4(h4)
            h4 = _ta(h4, self.tattn4)
            mid = self.mid1(self.ds4(h4), t_emb)
        if self.dit_bottleneck is not None:
            # DiT replaces (mid_attn → mid2). tattn_mid still applied after
            # spatial mixing, before the decoder upsweep, for temporal runs.
            mid = self.dit_bottleneck(mid, t_emb)
            mid = _ta(mid, self.tattn_mid)
        else:
            mid = self.mid_attn(mid)
            mid = _ta(mid, self.tattn_mid)
            mid = self.mid2(mid, t_emb)

        x = self.up4(mid)
        x = self.dec4(torch.cat([x, h4], dim=1), t_emb)
        if self.attn_dec4 is not None: x = self.attn_dec4(x)
        x = _film(x, self.film_dec4, 3)   # H/8 source feat
        if self.cross_attn_dec4 is not None and pyr_feats is not None:
            x = self.cross_attn_dec4(x, pyr_feats[3])
        x = _ta(x, self.tattn_dec4)
        x = self.up3(x)
        x = self.dec3(torch.cat([x, h3], dim=1), t_emb)
        if self.attn_dec3 is not None: x = self.attn_dec3(x)
        x = _film(x, self.film_dec3, 2)   # H/4 source feat
        x = _ta(x, self.tattn_dec3)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, h2], dim=1), t_emb)
        if self.attn_dec2 is not None: x = self.attn_dec2(x)
        x = _film(x, self.film_dec2, 1)   # H/2 source feat
        x = _ta(x, self.tattn_dec2)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, h1], dim=1), t_emb)
        if self.attn_dec1 is not None: x = self.attn_dec1(x)
        x = _film(x, self.film_dec1, 0)   # H source feat
        x = _ta(x, self.tattn_dec1)

        return self.out_conv(F.silu(self.out_norm(x)))


if __name__ == "__main__":
    model = Img2ImgDiffusionUNet(pretrained_source_encoder=True)
    s = torch.randn(2, 3, 128, 128)
    y_t = torch.randn(2, 3, 128, 128)
    t = torch.randint(0, 1000, (2,))
    out = model(s, y_t, t)
    print("output shape:", tuple(out.shape))
    print("source encoder trainable:", model.source_encoder.trainable_summary())
