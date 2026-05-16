## exp24 — exp23 + native-res crops (aug-resize-scale 4.0, no zoom)

**Status: DONE 2026-05-11**

Hypothesis: downscaling 1024px→281px before cropping (default scale=1.10)
blurs stroke widths. Scale=4.0 keeps 1024px native, random crop 256px.

Results: lpips_sq=0.273, lpips_vgg=0.449, ssim=0.537 — much worse than exp23.

Root cause: 256 from 1024 = 1/16th image area per crop. High crop variance:
easy (background) crops dominate early training, then model hits hard (face)
crops at step ~4k and needs 6k steps to recover. Wasted half the training budget.

**Lesson**: native-res crops are too sparse for 256px training at 1k pairs. Need
larger effective receptive field per crop. scale=2.0 (512→256 crop, 1/4 image
area) is the planned fix (exp24b).

---
