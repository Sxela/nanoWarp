## exp57 — source dropout 0.2 (regularization, NO CFG)

**Status: DONE 2026-05-19** — tie on quality + robustness win, candidate
for 80k promotion.

Single-flag delta vs exp50: `--source-dropout 0.2`. 20% of training
batch elements get their source channels zeroed → model must predict
target from noise + time only for those samples. **NOT CFG** — at
inference we keep `--cfg-scale 1.0` (single conditioned pass).
exp41's CFG-at-flow failure (ssim 0.36 at scale=2.0) is binding: in
flow, v is a true velocity, can't be amplified.

Hypothesis: at 3k pairs × 80k steps = 100+ epochs (exp52 regime), the
model may be over-memorizing source→target shortcuts. Dropout as
regularization forces a target-distribution prior, which should also
help robustness on weak-source inputs.

Recipe: exp50 base + `--source-dropout 0.2`, 20k @ 256px bs=4. If A/B
wins vs exp50, promote to 80k vs exp52.

Script: `scripts/run_exp57_source_dropout_at_exp50_recipe.sh`

**Results vs exp50 (20k baseline)**:

| split | metric | exp50 | exp57 | Δ |
|---|---|---|---|---|
| val_portraits | **face_lpips_sq** | **0.124** | **0.124** | 0% (exact tie) |
| val_portraits | face_lpips_vgg | 0.285 | 0.290 | +1.8% (tie) |
| val_portraits | face_ssim | 0.544 | 0.550 | +1.1% (tie) |
| val_portraits | whole ssim | 0.444 | **0.457** | **+2.9% (WIN)** |
| val_portraits | **Δ_lpips_vgg** | 0.037 | **0.034** | **-8.1% (WIN)** |
| legacy val | face_lpips_sq | 0.201 | 0.207 | +3.0% (mild loss) |
| legacy val | whole lpips_sq | 0.150 | 0.156 | +4.0% (mild loss) |

Read: essentially a tie on quality with mild robustness gain. At 20k
steps × 3k pairs = 26 epochs, the "over-memorization at long training"
hypothesis hasn't had a chance to differentiate.

**Important nuance on the robustness Δ improvement**: there are two
ways Δ_lpips_vgg can shrink. Either (a) corrupted-val genuinely
improves while clean stays flat — a real robustness gain (this is what
exp56 mid-aug shows in mid-training charts), or (b) clean degrades and
corrupted degrades by ~the same amount — Δ shrinks mechanically but
absolute corrupt-val is unchanged. exp57 is closer to (b): exp50
portraits corrupted ≈ 0.390, exp57 portraits corrupted ≈ 0.393 — the
absolute robustness barely moved. The Δ improvement is real arithmetic
but mechanically less compelling than the "model learned to invariance"
story exp56 is telling. Source dropout alone doesn't expose the model
to corruption — that's what training-time aug (clean_prob<1) actually
does.

**Recommendation**: promote to 80k vs exp52 as **exp57b**. Defer until
exp58 + exp59 land — if either of those is a clearer 20k win, prioritize
that promotion instead.

---
