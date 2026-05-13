from __future__ import annotations

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class ValidationMetrics:
    """SSIM + LPIPS variants.

    LPIPS-squeeze stays the primary metric for continuity with exp01-15.
    LPIPS-VGG is reported alongside as a secondary check that we're not
    overfitting to the training-time LPIPS backbone. When a model trains
    with `--lpips-aux-net squeeze` (the default) and we validate against
    LPIPS-squeeze, we have a partial overfit-to-metric confound: the same
    network drives both gradient and report. LPIPS-VGG is an independent
    perceptual yardstick (different backbone, never in our training loop
    as of 2026-05-11) that we can compare across runs honestly.
    """

    def __init__(self, device: torch.device):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_squeeze = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
        self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return {
            "ssim": float(self.ssim(pred, target).item()),
            "lpips": float(self.lpips_squeeze(pred, target).mean().item()),  # alias for back-compat
            "lpips_squeeze": float(self.lpips_squeeze(pred, target).mean().item()),
            "lpips_vgg": float(self.lpips_vgg(pred, target).mean().item()),
        }

