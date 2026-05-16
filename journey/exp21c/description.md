## exp21c — GAN aux + NoGAN phases + adaptive switching, gan-weight 0.1

**Status: DONE 2026-05-11**

Implemented `--gan-adaptive-switch`: in full phase, update G-adv if
`g_gan_ema >= d_loss_ema`, update D if `d_loss_ema >= g_gan_ema`. EMA alpha=0.1.

Results: lpips_sq=0.211, lpips_vgg=0.320, ssim=0.606

Training stable (ema_g≈0.68, ema_d≈0.89 at end — balanced). But val metrics
still worse than exp14v2. G learned to fool D on train distribution but doesn't
generalise to val. Observation: GAN helps texture/colour but loses facial detail.

**GAN conclusion**: All GAN variants (exp20/21/21b/21c) trail exp14v2 on honest
metrics. Balanced training (adaptive switch) is better than fixed phases, but
adversarial pressure at any tested weight hurts reconstruction quality vs pure
LPIPS. GAN may help qualitatively (texture, colour) even when metrics regress.
Parked pending further investigation.

---
