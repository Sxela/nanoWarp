"""Hinge GAN losses (the SAGAN / spectral-norm convention).

Hinge loss is the empirically-preferred GAN objective for stylization /
image-translation tasks. Compared to vanilla BCE-GAN:

- No saturation issue: gradient flows cleanly even when D is confident.
- Pairs naturally with spectral norm on the discriminator.
- The "1" margins act as a soft hinge — D is rewarded for being correctly
  confident, not for being arbitrarily confident.

Definitions:
    L_D = mean(max(0, 1 - D(real))) + mean(max(0, 1 + D(fake)))
    L_G = -mean(D(fake))

D wants to push D(real) → +1 and D(fake) → -1 (with hinge tolerance).
G wants D(fake) → +inf (i.e. fool the discriminator).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss. d_real and d_fake are raw D outputs (no sigmoid)."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss. Maximises D's score on fake samples."""
    return -d_fake.mean()
