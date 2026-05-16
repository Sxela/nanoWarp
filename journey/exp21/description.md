### exp21 — GAN aux with fastai NoGAN three-phase scheduling — IMPLEMENTED, PLANNED to launch

**Status update (2026-05-11)**: now implemented and smoke-tested. The
exp15/exp16/exp17 sequence revealed that VGG content + Gram is NOT what
made fastai's image generation work — the **GAN adversarial loss** was the
load-bearing piece. We now have a clean test: add GAN to the proven
exp14v2 recipe (LPIPS only, no VGG) and see if it closes the shape-
simplicity gap.

**Implementation** (~250 lines across three new files):
- [src/img2img/discriminator.py](../src/img2img/discriminator.py):
  PatchGAN discriminator, pix2pix 70×70 receptive field at default depth.
  Spectral norm on every conv. ~2.8M params at `--gan-d-channels 64`.
- [src/img2img/gan_loss.py](../src/img2img/gan_loss.py): hinge GAN losses
  (`hinge_d_loss`, `hinge_g_loss`). Hinge pairs naturally with spectral
  norm and avoids BCE saturation.
- [experiments/010_img2img_photo2comics/train.py](../experiments/010_img2img_photo2comics/train.py):
  GAN flags, separate AdamW for D (lr=1e-4, β1=0.5 by pix2pix
  convention), G update with adversarial term, D update on detached
  fake. Discriminator state + opt_d state saved in checkpoints (preserved
  on `--resume`). wandb logs `train/g_gan_loss`, `train/d_loss`,
  `train/d_real_score`, `train/d_fake_score`.

Smoke-tested at `--gan-weight 0.1`: D loss settles to its 2.0 hinge
equilibrium, real and fake scores track each other (D learning to
discriminate fairly), no NaN or spike under bf16 autocast.

**NoGAN phase scheduling (added 2026-05-11)**: implementing fastai's three-
phase approach to avoid random-G-meets-random-D chaos. Two new flags:

- `--gan-pretrain-g-steps N`: phase 1, GAN inactive. G trains on
  LPIPS/feature only. Reaches a "reasonable" baseline.
- `--gan-pretrain-d-steps M`: phase 2, G frozen. D trains alone on
  `(real, current-G-output)` pairs. Calibrates D before adversarial play.
- After `N + M` steps, phase 3 (full GAN) starts.

The smoke test confirms all three phases activate correctly at their
step boundaries and the metric printout shows the active phase
(`phase g_pretrain / d_pretrain / full`).

```powershell
# Recommended exp21 launch with NoGAN phasing.
# Phase 1 (5k): pure LPIPS pretrain  — G learns the basics
# Phase 2 (2k): D calibration       — D catches up on current G's output
# Phase 3 (13k): full adversarial   — alternating G+D updates
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --gan-weight 0.1 --gan-d-channels 64 --gan-d-layers 3 --gan-d-lr 1e-4 --gan-d-beta1 0.5 `
    --gan-pretrain-g-steps 5000 --gan-pretrain-d-steps 2000 `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp21_nogan_lpips_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,gan,patch-gan,nogan-phased,bf16,1k-dataset,256px,exp21" `
    --outdir out/exp21_nogan_lpips_noenc_1k_256px_20k
```

**Alternative: resume from exp14v2's pretrained G.** If we want to skip
phase 1 (since exp14v2's checkpoint is already an LPIPS-pretrained G),
we can launch with `--resume out/exp14v2_*/model.pt --gan-pretrain-g-steps 0
--gan-pretrain-d-steps 2000 --steps 22000`. But: exp14v2's cosine LR is
already at lr_min (1e-5) at step 40k, so the resume must override the
LR schedule (drop `--lr-cosine`, set `--lr 1e-4 --lr-warmup-steps 100`)
to actually train. Worth trying as a follow-up if the from-scratch exp21
is promising.

**Why this is now the highest-priority experiment:**

After exp17/exp16 confirmed that VGG content + Gram alone doesn't beat
LPIPS-only, the "fastai had a richer aux loss" hypothesis collapses.
What's left is the **discriminator** — which directly attacks the
shape-simplicity failure mode by saying "this output looks too simple
to be real" in a learnable, per-image way that no static perceptual
loss can replicate. This is exactly what the pix2pix lineage (pix2pix
→ AnimeGAN → AnimeGANv3) relies on for crisp stylization quality.

**Tuning notes for the launch:**
- Start with `--gan-weight 0.1` (auxiliary regularizer, not primary
  signal). If outputs look unchanged from exp14v2, raise to 0.3 or 0.5.
  If outputs look unstable / artifact-laden, lower to 0.05 or kill.
- Watch `d_real_score` and `d_fake_score` in wandb. Healthy training:
  both drift toward small positives/negatives but stay within ~±2.
  If `d_real_score` → +1.0 and `d_fake_score` → -1.0 with no movement,
  D has won (no useful G gradient). If they oscillate wildly, λ_gan
  is too high.
- D updates happen every step (1:1 ratio with G). If D is winning too
  hard later, we could go 1:2 (D every other step) — currently not
  exposed as a flag, but easy to add.

Predictions:
- If exp21 visually shows crisper shapes / more detail than exp14v2 →
  the "GAN drives shape complexity" hypothesis holds, *and* the NoGAN
  phasing was what fixed the grid-artifact failure of vanilla exp20.
  exp21 becomes the new recipe.
- If exp21 ≈ exp14v2 visually + slight metric improvement → GAN helps
  but is not transformative. Worth keeping at low weight.
- If exp21 hurts metrics or shows training instability → either λ_gan
  is mis-tuned (drop to 0.05), or NoGAN phasing alone wasn't enough
  to stabilise (would need longer phase 2 or smaller D).
- If exp21 *also* shows grid artifacts → PatchGAN's receptive field
  itself is the culprit. Try `--gan-d-layers 2` (smaller receptive
  field, ~34×34 instead of 70×70) or revisit the GAN approach entirely.

#### Original vanilla-GAN exp20 spec preserved below (for diff vs the NoGAN-phased exp21 launch above)

Triggered by visual inspection finding **shape simplicity** in our outputs:
crisp lines where they exist, but fewer of them than the target. The model
draws "soft cartoon" — locally plausible but topologically simpler than the
target anime art.

Distinct from blur (positional uncertainty smearing pixels). Shape
simplicity is about the model finding a *low-complexity attractor*: the
common subset of shapes across pairs that are correct most often. This is
what GANs in the pix2pix lineage (pix2pix → pix2pixHD → AnimeGAN →
AnimeGANv3) explicitly fix: a discriminator can identify
"this is too smoothed/simple to be real anime" in ways static perceptual
losses can't.

**Why our other losses don't fully fix this:**
- LPIPS at weight 0.2 is a regression-to-feature-mean tilt that *encourages*
  simplification.
- Gram (exp15) matches statistical complexity but doesn't punish per-image
  simplification — two images with similar Gram can have different complexity.
- L1/MSE rewards average shapes when uncertain.

**Architecture sketch:**
- Small PatchGAN discriminator (à la pix2pix), ~3M params (vs our 24M
  generator). Takes `(source, output)` pair, outputs per-patch real/fake
  scores.
- Spectral norm on every discriminator conv (already shipped as
  `torch.nn.utils.spectral_norm`; the standard pix2pix recipe).
- **Hinge loss** preferred over BCE: more stable, avoids the
  discriminator-saturation issue.
- λ_gan = 0.1 → 1.0; start at 0.1 (auxiliary regularizer, not primary
  signal). pix2pix used 1.0 with bigger discriminators; AnimeGAN uses
  ~0.5-1.0.
- Alternating G/D updates each step (1:1) or D-every-other-step (1:2).
- Discriminator LR 1e-4 with AdamW(β1=0.5, β2=0.999) (pix2pix convention).

**Cost:**
- ~150 lines: discriminator module, GAN loss helper, alternating updates
  in trainer, separate D optimizer + EMA.
- ~30% slower per step (extra D forward+backward).
- 2-3 short runs to tune λ_gan and D update frequency.
- Stability risk: GANs can collapse. Mitigated by spectral norm + small λ.

**Why we didn't implement it yet:**
- exp12 (256px) and exp14 (1k pairs) are likely to address most of the
  shape-simplicity issue at lower complexity cost.
- GAN training is real engineering and we should defer it until we know
  the simpler levers can't get there.
- This spec exists so we know what "exp20" means if we need to escalate.

```powershell
# (sketch only — implementation TBD)
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.1 `
    --gan-weight 0.1 --gan-d-channels 64 --gan-d-lr 1e-4 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-tags "flow,no-encoder,lpips,gan,exp20" `
    --wandb-run-name exp20_gan_noenc_attn832_bf16_mc88_30k `
    --outdir out/exp20_gan_noenc_attn832_bf16_mc88_30k
```

Predictions:
- If exp12 + exp14 close the shape-simplicity gap → exp16 unnecessary,
  skip the implementation cost.
- If they don't → exp16 is the standard pix2pix-lineage fix and very
  likely (~80% confidence) to give visible complexity gains. Cost
  ~150 LoC + a stability-conscious tuning pass.
