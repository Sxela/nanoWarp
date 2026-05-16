## exp48 — pixel DiT + multiscale + LPIPS warmup

**Status: DONE 2026-05-16** (still bad, multiscale didn't fix DiT)

Three fixes applied to exp47:
  1. Progressive 1k @ 128 (64 tokens) → 4k @ 256 (256 tokens) → 15k @ 512
     (1024 tokens). Start small to let attention learn cross-patch
     coherence quickly with limited data.
  2. LPIPS-weight warmup 0 → 0.2 over the 128 phase. Pure flow-MSE
     establishes structure first.
  3. Linear ramp (`--lpips-weight-warmup-steps` flag added).

Result: still bad at both 256 (lpips_sq 0.377, 3× exp35) and 512
(lpips_sq 0.451, 3× exp32-100k). The multiscale + warmup helped a bit
at 128/256 but the model couldn't learn at 1024 tokens (512px) with
1k pairs.

**Conclusion**: pure pixel DiT is dead at our data scale. Even with
multi-day MAE pretraining on a 13k Imagenette subset, the 0.3 lpips_sq
gap probably wouldn't close. UNet's conv inductive bias is doing real
work that DiT needs ~10–100× more data to replicate.

---
