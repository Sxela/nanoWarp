from __future__ import annotations

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


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
    ):
        super().__init__()
        self.source_in_stem = source_in_stem
        self.source_encoder = SourceEncoder(
            in_ch=3,
            pretrained=pretrained_source_encoder,
            freeze_stages=freeze_source_stages,
        )
        self.time_mlp = TimeMLP(time_dim)

        stem_in_ch = in_ch + 3 if source_in_stem else in_ch
        self.in_conv = nn.Conv2d(stem_in_ch, model_ch, 3, padding=1)

        self.down1 = ResBlock(model_ch, 64, time_dim * 4)
        self.fuse1 = FuseBlock(64, 64)
        self.ds1 = Downsample(64)

        self.down2 = ResBlock(64, 128, time_dim * 4)
        self.fuse2 = FuseBlock(128, 64)
        self.ds2 = Downsample(128)

        self.down3 = ResBlock(128, 256, time_dim * 4)
        self.fuse3 = FuseBlock(256, 128)
        self.ds3 = Downsample(256)

        self.down4 = ResBlock(256, 256, time_dim * 4)
        self.fuse4 = FuseBlock(256, 256)
        self.ds4 = Downsample(256)

        self.mid1 = ResBlock(256, 512, time_dim * 4)
        self.mid_fuse = FuseBlock(512, 512)
        self.mid_attn = BottleneckAttention(512)
        self.mid2 = ResBlock(512, 512, time_dim * 4)

        self.up4 = Upsample(512)
        self.dec4 = ResBlock(512 + 256, 256, time_dim * 4)
        self.up3 = Upsample(256)
        self.dec3 = ResBlock(256 + 256, 256, time_dim * 4)
        self.up2 = Upsample(256)
        self.dec2 = ResBlock(256 + 128, 128, time_dim * 4)
        self.up1 = Upsample(128)
        self.dec1 = ResBlock(128 + 64, 64, time_dim * 4)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, source: torch.Tensor, noisy_target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        src_feats = self.source_encoder(source)
        t_emb = self.time_mlp(t)

        stem_input = torch.cat([source, noisy_target], dim=1) if self.source_in_stem else noisy_target
        x0 = self.in_conv(stem_input)

        h1 = self.fuse1(self.down1(x0, t_emb), src_feats[0])
        h2 = self.fuse2(self.down2(self.ds1(h1), t_emb), src_feats[1])
        h3 = self.fuse3(self.down3(self.ds2(h2), t_emb), src_feats[2])
        h4 = self.fuse4(self.down4(self.ds3(h3), t_emb), src_feats[3])

        mid = self.mid1(self.ds4(h4), t_emb)
        mid = self.mid_fuse(mid, src_feats[4])
        mid = self.mid_attn(mid)
        mid = self.mid2(mid, t_emb)

        x = self.up4(mid)
        x = self.dec4(torch.cat([x, h4], dim=1), t_emb)
        x = self.up3(x)
        x = self.dec3(torch.cat([x, h3], dim=1), t_emb)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, h2], dim=1), t_emb)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


if __name__ == "__main__":
    model = Img2ImgDiffusionUNet(pretrained_source_encoder=True)
    s = torch.randn(2, 3, 128, 128)
    y_t = torch.randn(2, 3, 128, 128)
    t = torch.randint(0, 1000, (2,))
    out = model(s, y_t, t)
    print("output shape:", tuple(out.shape))
    print("source encoder trainable:", model.source_encoder.trainable_summary())
