from __future__ import annotations

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class ValidationMetrics:
    def __init__(self, device: torch.device):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return {
            "ssim": float(self.ssim(pred, target).item()),
            "lpips": float(self.lpips(pred, target).mean().item()),
        }

