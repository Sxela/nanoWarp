"""PatchGAN discriminator (pix2pix-style) with spectral norm.

Used as an adversarial auxiliary signal alongside our existing flow/diffusion
+ LPIPS aux + (optional) VGG feature loss stack. Receives `(source, output)`
as concatenated 6-channel input and produces a feature map of per-patch
real/fake scores. Each scalar in the output map corresponds to ~70x70 pixels
of the input (the classic pix2pix receptive field at default depth).

Spectral norm on every conv keeps the Lipschitz constant bounded, which is
the standard stability tool for GAN training (Miyato et al. 2018). Hinge loss
is used in [src/img2img/gan_loss.py](gan_loss.py) — it pairs well with
spectral norm and avoids the saturation issues of vanilla BCE.

Notes:
- Small model: ~2-3M params at `base_channels=64`, vs our 24M+ generator.
- LeakyReLU(0.2), the pix2pix convention.
- No batch/group norm — spectral norm handles the stability role and we
  avoid the train/eval BN-stats problem entirely.
"""

from __future__ import annotations

import torch
from torch import nn


def _sn_conv(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Module:
    """Conv2d wrapped in spectral norm. Helper to keep the model cleaner."""
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding))


class PatchDiscriminator(nn.Module):
    """pix2pix-style 70x70 PatchGAN."""

    def __init__(self, in_channels: int = 6, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        ks = 4
        layers: list[nn.Module] = []

        # First conv: no spectral norm typically not used on first layer in pix2pix,
        # but we apply it anyway for stricter Lipschitz bound (matches Miyato 2018).
        layers.append(_sn_conv(in_channels, base_channels, ks, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        c_prev = base_channels
        for n in range(1, n_layers):
            c_cur = min(base_channels * (2 ** n), 512)
            layers.append(_sn_conv(c_prev, c_cur, ks, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c_prev = c_cur

        c_cur = min(base_channels * (2 ** n_layers), 512)
        layers.append(_sn_conv(c_prev, c_cur, ks, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(_sn_conv(c_cur, 1, ks, stride=1, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Returns a feature map of per-patch real/fake scores. Higher = more 'real'.

        Both source and output should be in the same value range (typically [0,1])
        and same spatial size.
        """
        x = torch.cat([source, output], dim=1)
        return self.net(x)
