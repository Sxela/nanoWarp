## exp33b — exp33 recipe with scale capped at 1.5

**Status: DONE 2026-05-14**

Tests whether scale crop variance is the dominant cost of exp33's regression
vs exp23. Result: yes — `scale=[1.0, 1.5]` recovered ~50% of the gap
(lpips_vgg 0.308 → 0.274 vs exp23's 0.234). The rest of the aug stack
(rotate/perspective/color/blur/JPEG) costs maybe 0.04 lpips_vgg combined,
not 0.07.

---
