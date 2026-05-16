## exp38/39 — exp35 + source-contrastive loss

**Status: DONE 2026-05-15**

Margin-form contrastive: `loss += w * relu(margin − lpips(out, source))`.
Only contributes when output is too close to source; zero past the margin.

- exp38 (w=0.1, m=0.15): neutral on every metric vs exp35.
- exp39 (w=0.3, m=0.25): slight whole-image lpips_vgg improvement (0.240 →
  0.238), slight face regression (~1–2%), modest robustness gain
  (Δ 0.133 → 0.126). Pushed model marginally away from source-copy
  behaviour. Not a clear win.

---
