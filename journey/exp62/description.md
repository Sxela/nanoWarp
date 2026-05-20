## exp62 — drop source-in-stem + add cross-attn at H/4

**Status: WIRED 2026-05-19**

Two-knob delta vs exp59 (cross-attn @ H/8, 20k):
1. `--no-source-in-stem`: in_conv goes 6→88 ch to 3→88 ch. Source no
   longer concatenated into the encoder input. Source signal now comes
   purely via SourcePyramid + FiLM + cross-attn.
2. `--use-cross-attn-cond-h4`: adds a second CrossAttnCond at the H/4
   decoder level. Multi-scale source conditioning: H/8 [1024 tokens] +
   H/4 [4096 tokens].

Net param delta vs exp60: +495k (essentially same budget at ~49M).

Hypothesis: in flow matching, `x_t = (1-t)·source + t·target`, so at
t=0 the model sees source via `x_t` itself — making source-in-stem
redundant with pyramid+cross-attn. Removing it eliminates double
conditioning and frees capacity. Multi-scale cross-attn (H/8 + H/4)
compensates by giving stronger pyramid-mediated source conditioning.

Code changes:
- `model.py`: relaxed the `use_source_encoder=False → source_in_stem=True`
  override when pyramid is enabled. Added `use_cross_attn_cond_h4` flag
  + `cross_attn_dec3` module at H/4 decoder level.
- `ckpt.py`: auto-detects both cross-attn levels from state_dict.
- `train_exp32_prog512.py`: `--no-source-in-stem` and
  `--use-cross-attn-cond-h4` flags; saved config records actual
  source_in_stem value (not the hardcoded True default).

Smokes confirmed: forward pass works without source concat, ckpt
roundtrip preserves both new flags via state_dict key auto-detection,
in_conv weight shape correctly reflects 3-channel input.

A/B target — exp59 (20k, cross-attn @ H/8 only, source_in_stem=True):
- face_lpips_sq portraits = 0.122
- face_lpips_vgg portraits = 0.282
- Δ_lpips_vgg portraits = 0.035

Script: `scripts/run_exp62_no_concat_plus_ca_h4_at_exp50_recipe.sh`

If wins → 80k promotion as exp62b vs exp60 (current quality canonical
at 0.0997).

---

## ⚠️ Benchmark caveat: Flux skin-tone bias in val_portraits (2026-05-19)

Discovered while reviewing exp58b outputs: Flux occasionally produced
lighter-skinned anime targets when given darker-skinned real-photo
sources. The val_portraits "ground truth" is therefore biased for those
pairs — a model that faithfully preserves source identity will score
*worse* on LPIPS/SSIM than one that drifts toward the lightened target.

**Implications for prior comparisons** in this log:
- Small wins (1-5% deltas like exp59's -1.6% face_lpips_sq) — probably
  robust to this bias since the magnitude is below what skin-tone
  drift could explain.
- Medium regressions (exp58b's +10%) — partially confounded; could be
  half-method, half-bias.
- Large regressions (exp58's +44%, exp54's +310%) — too big to be
  pure bias artifact; the method-level issues are real.

**Action items**:
- [ ] Build a skin-tone-stratified val_portraits subset (or use a face
      attribute classifier) so we can report metrics per-bin.
- [ ] Re-generate the affected target pairs with explicit skin-tone-
      preservation prompts, or filter them out of val_portraits.
- [ ] For high-stakes A/B's going forward, supplement metrics with
      visual side-by-sides on a curated set of darker-skinned sources.

**What this does NOT invalidate**:
- exp52 / exp56 / exp59's clean-val wins (small deltas, dominated by
  signal not bias).
- Robustness Δ comparisons (corrupt-val vs clean-val is bias-symmetric
  since both pass through the same biased target).

---

## Open follow-ups (3k era, updated 2026-05-19)
- **More diverse real-photo sources**: Unsplash people, Places365
  with people-filter, AFW/IJB-C in-the-wild faces. Currently FFHQ
  alone biases toward studio-lit Western 25-35yo portraits. This is
  now the highest-leverage open lever, since exp53 ruled out
  resize-filter and exp50→exp52 already extracted the training-
  duration gain.
- **Resolution scale-up**: every 3k-era run is at 256px; FFHQ source
  is 512. Train at 384 or 512 to test the resolution ceiling.
  Independent of exp53's negative — at 512 target, BILINEAR is
  near-identity on 512px sources so this is a "more pixels of
  capacity" experiment, not a "sharper input" one.
- **Curriculum option** (deferred): if portrait quality stalls below
  some threshold, start from exp51's FFHQ-only checkpoint and
  fine-tune on the 3k mixed set. Might give exp50-on-FFHQ quality
  *and* exp50-on-legacy capability simultaneously.
- **In-the-wild face val split**: a third val with small / off-center /
  partially-occluded faces, real photos. Currently no val covers this
  honestly; legacy val is group photos but skewed toward peripheral
  subjects.
