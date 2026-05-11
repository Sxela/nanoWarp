"""VGG feature loss + Gram-matrix style loss (Gatys / Johnson / fastai pattern).

The motivation: LPIPS measures perceptual similarity (does the image look like
the target?) but doesn't explicitly enforce texture statistics. For style
transfer tasks (like photo->anime), what we actually want is for the output
to share the *texture statistics* of the target style independent of pixel
alignment. Gram matrices of VGG features capture exactly that.

`VGGFeatureLoss` computes two terms simultaneously from a single VGG16
forward pass:

- **Content loss**: L1 between VGG features at the chosen layers.
  Similar in spirit to LPIPS but unweighted and unlearned. Cheap.
- **Style loss**: L1 between Gram matrices of those features. Pushes the
  output toward the target's channel-correlation statistics, i.e. its
  texture style.

Defaults follow fastai's `FeatureLoss` (Johnson et al.): three layers
after `relu2_2`, `relu3_3`, `relu4_3`. Layer indices in
torchvision's `vgg16().features` are 8, 15, 22.

Inputs expected in [0, 1] sRGB. ImageNet mean/std normalisation applied
internally before VGG. VGG weights are frozen and the module is held in
eval mode.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# Default VGG16 layer indices for content + style (fastai / Johnson defaults).
# Slice points after relu blocks 2_2, 3_3, 4_3 in vgg16.features.
DEFAULT_VGG_LAYERS = (8, 15, 22)


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Channel-wise feature covariance. Input (B, C, H, W) -> output (B, C, C).

    Normalised by (C * H * W) for magnitude stability across different
    feature sizes — same convention as fastai's `FeatureLoss`.
    """
    b, c, h, w = x.shape
    flat = x.view(b, c, h * w)
    gram = flat @ flat.transpose(1, 2)
    return gram / (c * h * w)


class VGGFeatureLoss(nn.Module):
    def __init__(
        self,
        layers: tuple[int, ...] = DEFAULT_VGG_LAYERS,
        content_weight: float = 1.0,
        style_weight: float = 5000.0,
        content_layer_weights: tuple[float, ...] | None = None,
        style_layer_weights: tuple[float, ...] | None = None,
    ):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        for p in vgg.parameters():
            p.requires_grad = False

        # ImageNet normalisation buffers (VGG was trained on these stats).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Build slices so we can compute features incrementally and read off
        # the activations at each chosen layer without re-running earlier layers.
        layers = tuple(sorted(set(int(l) for l in layers)))
        self.layer_indices = layers
        slices = []
        prev = 0
        for idx in layers:
            slices.append(vgg[prev:idx + 1])
            prev = idx + 1
        self.slices = nn.ModuleList(slices)
        # Keep frozen modules in eval permanently (BN running stats — VGG has
        # no BN but this is defensive in case torchvision changes).
        for m in self.slices:
            m.eval()

        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)

        # Per-layer weights. None -> uniform 1/n (preserves the old `.mean()`
        # behavior exactly). Custom values are raw multipliers; the user
        # picks the magnitude.
        n = len(layers)
        self.content_layer_weights = self._resolve_layer_weights(content_layer_weights, n)
        self.style_layer_weights = self._resolve_layer_weights(style_layer_weights, n)

    @staticmethod
    def _resolve_layer_weights(weights, n):
        if weights is None:
            return [1.0 / n] * n
        weights = [float(w) for w in weights]
        if len(weights) != n:
            raise ValueError(f"layer weights length {len(weights)} does not match num layers {n}")
        return weights

    def train(self, mode: bool = True):
        # Override: VGG slices must always be in eval to keep frozen weights
        # and statistics stable during training mode toggles.
        super().train(mode)
        for m in self.slices:
            m.eval()
        return self

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(0.0, 1.0) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a dict with `content`, `style`, and `total` losses. The
        per-term values use the per-layer weights so they are directly
        comparable across configs; `total` is then scaled by content/style
        weights and summed."""
        pred_h = self._normalize(pred)
        with torch.no_grad():
            target_h = self._normalize(target)

        content_terms: list[torch.Tensor] = []
        style_terms: list[torch.Tensor] = []

        for i, slice_module in enumerate(self.slices):
            pred_h = slice_module(pred_h)
            with torch.no_grad():
                target_h = slice_module(target_h)

            if self.content_weight > 0:
                content_terms.append(self.content_layer_weights[i] * F.l1_loss(pred_h, target_h))
            if self.style_weight > 0:
                pred_gram = _gram_matrix(pred_h)
                with torch.no_grad():
                    target_gram = _gram_matrix(target_h)
                style_terms.append(self.style_layer_weights[i] * F.l1_loss(pred_gram, target_gram))

        content = torch.stack(content_terms).sum() if content_terms else torch.zeros((), device=pred.device)
        style = torch.stack(style_terms).sum() if style_terms else torch.zeros((), device=pred.device)
        total = self.content_weight * content + self.style_weight * style
        return {"content": content, "style": style, "total": total}
