## exp41 — exp35 + classifier-free guidance

**Status: DONE 2026-05-15** (failed at every CFG scale tested)

Trained with `--source-dropout 0.1`. Validate.py extended with `--cfg-scale`
that does two forwards per ODE step and combines: `v = v_u + s*(v_c − v_u)`.

CFG-scale sweep on the same checkpoint:

| cfg | lpips_vgg | face_lpips_vgg | face_ssim |
|---|---|---|---|
| 1.0 (no CFG) | 0.244 | 0.293 | 0.721 |
| 1.1 | 0.246 | 0.297 | **0.722** |
| 1.2 | 0.253 | 0.304 | 0.715 |
| 1.3 | 0.262 | 0.316 | 0.702 |
| 1.5 | 0.291 | 0.354 | 0.656 |
| 2.0 | **0.419** | 0.485 | 0.457 (garbage) |

Monotonic degradation past cfg=1.0. Root cause analysis: in flow matching,
`v` is a true velocity (target − source per unit time). CFG amplification
literally makes each ODE step `s×` too big → integrator overshoots the
target manifold. Diffusion CFG of 2–7 doesn't translate to FM directly.
Source-in-stem also makes guidance redundant — source conditioning is so
dense at every level there's no useful "amplify in this direction".

Lesson: CFG is the wrong tool for tight pixel-aligned img2img with strong
source conditioning. Don't repeat.

---
