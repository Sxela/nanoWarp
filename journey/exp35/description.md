## exp35 — exp34 (or exp33) recipe + in-model source feature pyramid + FiLM

**Status: WIRED 2026-05-14** (ready to launch after exp34 settles)

Clean A/B vs the previous best of {exp33, exp34} (whichever wins) to
isolate the contribution of a permanent in-model source feature pyramid.
Single architectural delta on top of that baseline: adds `SourcePyramid`
+ per-decoder-level `FiLM` modulation.

**Architecture**:
- `SourcePyramid` ([src/img2img/source_pyramid.py](../src/img2img/source_pyramid.py)):
  4-stage conv pyramid (`stem → s1 → s2 → s3`) run **once per source per
  forward pass**, producing features at the four decoder resolutions
  (H, H/2, H/4, H/8) with channels matching UNet widths
  (c1=88, c2=176, c3=352, c4=352). ~1.8M params at mc=88.
- `FiLM` (same file): per-decoder-level 1×1 conv produces (γ, β) from a
  pyramid feature; decoder activation becomes `x * (1 + γ) + β`. Both γ and
  β zero-init → identity at init. ~573k params total across the 4 levels.
- Net: ~2.4M extra params over the baseline (49M → ~51M).

**Why FiLM not cross-attention** (first try): half the params, no head-dim
hyperparams, identity-at-init for free. If FiLM shows even modest gain we
can iterate to cross-attn later.

**Why no inference-time external deps**: pyramid is part of the UNet and
ships in the same checkpoint as everything else. Permanent in-model
alternative to the optional ResNet18 source encoder.

**Recipe**: same as the baseline plus `--use-source-pyramid`. If exp34
wins, also pass `--use-decoder-attn` to stack on top.

```bash
WANDB_API_KEY=... bash scripts/run_exp35_pyramid_at_exp33_recipe.sh
```

Script: `scripts/run_exp35_pyramid_at_exp33_recipe.sh`
Outdir: `out/exp35_pyramid_film_noenc_attn163264_bf16_mc88_256px_20k`

**Compute redundancy**: pyramid output is independent of t, so it
recomputes wastefully across the 20 ODE steps at inference. Pyramid is
~1.8M cheap conv params → <1ms per call on a 4090 → ~600ms wasted per
30-frame video clip. Acceptable for now; can cache later if profiling
demands it.

Results: TBD.

**Architecture diagram**: [docs/model_architecture.html](model_architecture.html)
(self-contained HTML/SVG; shows the full UNet, the sub-block structure
within each level — ResBlock → attn? → FiLM? → tattn? — and the exp35
source pyramid + FiLM hooks).

---
