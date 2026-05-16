## exp21b — GAN aux + NoGAN phases, gan-weight 0.005

**Status: DONE 2026-05-11**

Results: lpips_sq=0.187, lpips_vgg=0.284, ssim=0.640

Training stable (D scores near zero), but gan-weight too weak to improve over
exp14v2. GAN at 0.005 adds adversarial noise without enough signal.

---
