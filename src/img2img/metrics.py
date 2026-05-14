from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _build_face_detector(min_confidence: float = 0.5):
    """Build an OpenCV Haar-cascade frontal face detector, or None if cv2 missing.

    Note: `min_confidence` is accepted for API parity with the previous
    MediaPipe-based detector but isn't applicable to Haar cascades; the
    detection threshold is governed by `minNeighbors` in the detect call.
    """
    del min_confidence
    if not _CV2_AVAILABLE:
        return None
    import os
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        return None
    return detector


def face_crops(
    source: torch.Tensor,
    target: torch.Tensor,
    sample: torch.Tensor,
    face_detector,
    crop_size: int = 128,
    min_face_px: int = 16,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Detect faces in `source`, crop the same bbox from all three tensors, resize.

    Args:
        source, target, sample: (B, 3, H, W) float in [0, 1] on any device.
        face_detector: a MediaPipe FaceDetection instance (or None → returns None,None,None).
        crop_size: square output size for each crop.
        min_face_px: discard detections whose bbox is smaller than this on either side.

    Returns:
        Three tensors of shape (N, 3, crop_size, crop_size) — face crops from
        source, target, sample respectively, stacked across all detected faces
        in the batch. N may be 0 (or all None) if no faces were detected.
    """
    if face_detector is None:
        return None, None, None
    B, _, H, W = source.shape
    src_crops, tgt_crops, smp_crops = [], [], []
    for i in range(B):
        src_np = (source[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
        # detectMultiScale → array of (x, y, w, h) per face in pixel coords.
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                               minSize=(min_face_px, min_face_px))
        for (x, y, w, h) in faces:
            x0, y0 = int(x), int(y)
            x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
            if (x1 - x0) < min_face_px or (y1 - y0) < min_face_px:
                continue
            for crops, src in ((src_crops, source), (tgt_crops, target), (smp_crops, sample)):
                patch = src[i:i+1, :, y0:y1, x0:x1]
                patch = F.interpolate(patch, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                crops.append(patch[0])
    if not src_crops:
        return None, None, None
    return (
        torch.stack(src_crops),
        torch.stack(tgt_crops),
        torch.stack(smp_crops),
    )


def val_corrupt(source: torch.Tensor, jpeg_q: int = 60,
                blur_sigma: float = 1.0, resize_factor: float = 0.5) -> torch.Tensor:
    """Deterministic mid-strength corruption of a source batch for the
    robustness-val pass. Fixed strength → numbers are comparable across
    training steps and runs.

    Pipeline (matches the in-training degrade aug at moderate values):
        1. Downsample → upsample (resize_factor, bilinear) — pixelation
        2. JPEG roundtrip at quality `jpeg_q`
        3. Gaussian blur σ=`blur_sigma`

    Args:
        source: (B, 3, H, W) float in [0, 1].
    Returns:
        Corrupted tensor on the same device/dtype.
    """
    B, _, H, W = source.shape
    down = F.interpolate(source, scale_factor=resize_factor, mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
    out_list = []
    for i in range(B):
        img = TF.to_pil_image(up[i].clamp(0, 1).cpu())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        out_list.append(TF.to_tensor(img))
    corrupted = torch.stack(out_list).to(source.device, dtype=source.dtype)
    kernel = max(3, int(2 * round(2 * blur_sigma) + 1))
    if kernel % 2 == 0:
        kernel += 1
    corrupted = TF.gaussian_blur(corrupted, kernel_size=kernel, sigma=[blur_sigma, blur_sigma])
    return corrupted


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
        # Reset before each call so the stateful torchmetrics accumulators only
        # see the current batch — not a running mean over the whole val loop.
        # Also avoids calling lpips_squeeze twice (which doubled the sample count
        # and returned the wrong mean for the "lpips" alias key).
        self.ssim.reset()
        self.lpips_squeeze.reset()
        self.lpips_vgg.reset()
        lpips_sq = float(self.lpips_squeeze(pred, target).mean().item())
        return {
            "ssim": float(self.ssim(pred, target).item()),
            "lpips": lpips_sq,  # alias for back-compat
            "lpips_squeeze": lpips_sq,
            "lpips_vgg": float(self.lpips_vgg(pred, target).mean().item()),
        }

