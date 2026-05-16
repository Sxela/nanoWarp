## exp42 — exp35 + LPIPS weight cosine-annealed 0.2 → 0.0

**Status: DONE 2026-05-15** (metrics win, visual loss)

Hypothesis (incorrect, as it turned out): LPIPS-VGG's L2-in-feature-space
incentivises mode-averaging — the model converges toward a feature
centroid of plausible targets rather than committing to specific stylistic
patterns. Anneal to zero by end → keep LPIPS nudge early, let MSE sharpen
specifics late.

Result on metrics:
- **Best ssim yet** (0.700, first >0.69 crossing)
- **Best face_ssim** (0.744 vs exp35's 0.728)
- Best whole-image `lpips_vgg` (0.229 vs exp35's 0.240)
- mean_loss dropped 4× by end (mostly via flow_loss converging better when
  LPIPS isn't fighting it)

Result visually: **blurrier than exp35**, not sharper. The pixel-MSE
centroid IS the smooth output — that's what MSE rewards in pixel space
with deterministic `v_target = target − source`. LPIPS was actually pushing
*away* from pixel-MSE centroid into perceptually-sharper specifics, not
toward feature centroid.

Why FLUX/SD3 don't have this problem: they predict in **latent space**
with a sharp VAE decoder, *and* their `x_t` has stochastic noise so the
training target is a *distribution* not a deterministic map. Our setup
(pixel space + paired data + tiny `σ_noise=0.05`) is the worst case for
MSE blur. See exp43 below.

---
