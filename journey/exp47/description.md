## exp47 — pure-pixel DiT (HiDream-O1 style), exp35 recipe slot

**Status: DONE 2026-05-16** (block artifacts, worse than UNet)

Pure-pixel transformer architecture per the HiDream-O1 paper: patch=16
linear embedding → 11 DiT blocks at dim=384 → linear unpatchify (no
conv stem, no decoder). 48.5M params, matching the 49M UNet budget for
fair A/B.

Result: worse than UNet at every metric. **Visible block artifacts** in
panels — patch boundaries aren't smoothed by any conv stem/head, and at
1k pairs the model can't learn cross-patch coherence purely through
self-attention.

How HiDream avoids this at 8B params: **brute force scale**. Their
`FinalLayer` is literally just `self.linear(x)`. No spatial smoothing
trick — they just have enough capacity + data that attention learns
cross-patch consistency empirically. We're firmly in the opposite
regime (49M params, 1k pairs).

---
