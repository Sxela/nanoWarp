## exp23 — exp14v2 20k + LPIPS-VGG backbone (--lpips-aux-net vgg)

**Status: DONE 2026-05-11**

Single change from exp14v2: swap LPIPS training backbone from squeeze to vgg.
Val metric (squeeze) stays for continuity.

Results: lpips_sq=0.127, lpips_vgg=0.234, ssim=0.689

**Biggest single-run improvement of the project.** −30% lpips_sq vs exp14v2 20k,
−3% lpips_vgg vs exp14v2 40k, SSIM matches exp14v2 40k in only 20k steps.
VGG LPIPS as training loss forces the model to respect mid-level feature structure
(relu2_2/relu3_3/relu4_3) — exactly what matters for facial detail and edge quality.

**New baseline**: exp23 replaces exp14v2 as the reference recipe.

---
