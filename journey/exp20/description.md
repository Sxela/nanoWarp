### exp20 — vanilla GAN aux (no NoGAN phasing) — DONE, grid-artifact collapse

First attempt at adding the GAN aux on top of the exp14v2 recipe, with
**both G and D updating from step 1** (no pretrain phasing). Used the
PatchGAN discriminator + hinge loss + spectral norm setup described
above, at `--gan-weight 0.1`.

**Result: training "succeeded" by loss curves but visually produced
grid-like artifacts** — periodic patterns roughly at the PatchGAN's
~70-pixel receptive-field scale. The G output composed locally
plausible 70×70 patches that didn't tile into coherent images. Classic
GAN-from-scratch failure mode: random D produces meaningless gradient
direction in the early steps, G wanders into a degenerate "fool the
discriminator per patch" mode before learning the basic photo→anime
mapping.

This is *exactly* what fastai's NoGAN three-phase approach was designed
to prevent. Implemented and queued as exp21.
