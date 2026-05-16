## exp43 — exp35 + flow σ_noise bumped from 0.05 → 0.30

**Status: DONE 2026-05-15** (catastrophic failure)

σ=0.30 was way too aggressive. Every metric collapsed: ssim 0.689 → **0.134**,
lpips_sq 0.124 → **0.514**, face_ssim 0.728 → 0.209. Visually: grid
artifacts, oversaturation, high-pass-filter appearance.

Root cause: with σ=0.30 ambient noise at the inference start, the per-step
ODE displacement (~0.05 per `dt=1/20`) is **6× smaller than the noise
floor**. The integrator chases noise patterns instead of denoising →
grid-like garbage.

Lesson: σ_noise needs to be balanced against `dt = 1/sample_steps`. At 20
steps, σ must be ≪ 0.05 or much more sample-steps are needed. σ at the
0.05–0.10 range may still be worth a test, but σ=0.30 is far past the
useful band.

---
