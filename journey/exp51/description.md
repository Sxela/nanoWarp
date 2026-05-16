## exp51 — FFHQ-only sanity test

**Status: DONE 2026-05-16**

Same recipe as exp35/50 but trained on `data/photo2anime_ffhq2k`
(2321 train + 200 val_portraits, no synth-source pairs at all).
Question: is the architecture+recipe even capable of learning faces
on a clean uniform dataset?

Final val on two splits:

| split | metric | exp51 (FFHQ-only) | exp50 (3k mixed) |
|---|---|---|---|
| val_portraits | face_lpips_sq | **0.122** | 0.124 |
| val_portraits | face_lpips_vgg | **0.280** | 0.285 |
| val_portraits | face_ssim | **0.550** | 0.544 |
| val_portraits | Δ lpips_vgg | **+0.031** | +0.037 |
| legacy val | face_lpips_sq | **0.290** (-90% vs exp35) | 0.201 |
| legacy val | face_ssim | 0.510 | 0.605 |

**Conclusions**:

1. FFHQ-only ≈ 3k-mixed on FFHQ portraits (1-2% improvement, noise floor).
   The synth-source 1k pairs in exp50 weren't actually hurting portrait
   quality — they just weren't helping either.
2. FFHQ-only **catastrophically regresses** on legacy val (small /
   peripheral / group faces). The model has no concept of off-center
   faces anymore.
3. The "data is the lever" thesis from exp50 was right but more
   specific: it was the **real-photo source domain** that mattered,
   not FFHQ-specifically. Adding ANY real-photo source domain (mixed
   or pure) closed the train-test gap.
4. **exp50 (3k mixed) stays as the canonical baseline**. FFHQ-only is
   not a useful endpoint — the curriculum option (FFHQ-only pretrain
   → fine-tune on mixed) is still open if face quality stalls again.

---

## Open follow-ups (3k era)

- **exp52 = exp50 × 80k steps** — natural next move now that data
  scale-up worked. exp25-80k extension was the legacy-era big lift;
  same pattern at the new scale should compound.
- **More diverse real-photo sources**: Unsplash people, Places365
  with people-filter, AFW/IJB-C in-the-wild faces. Currently FFHQ
  alone biases toward studio-lit Western 25-35yo portraits.
- **Curriculum option** (deferred): if portrait quality stalls below
  some threshold, start from exp51's FFHQ-only checkpoint and
  fine-tune on the 3k mixed set. Might give exp50-on-FFHQ quality
  *and* exp50-on-legacy capability simultaneously.
- **In-the-wild face val split**: a third val with small / off-center /
  partially-occluded faces, real photos. Currently no val covers this
  honestly; legacy val is group photos but skewed toward peripheral
  subjects.
