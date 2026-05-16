## exp45 — exp35 + LPIPS anneal 0.2 → 0.1 (with floor)

**Status: WIRED 2026-05-16**

Same recipe as exp42 (20k single-phase, minimal aug, exp35 arch) except
the LPIPS anneal stops at 0.1 instead of going to 0. Targets the bottom
of exp44's U-curve as a steady-state weight.

If exp45 beats exp35 on faces with similar or lower whole-image LPIPS,
the floor is the right pattern going forward. If not, exp35's constant
0.2 stays canonical and we move on to **data scale-up** (FFHQ → Flux
pairs) as the next leverage axis.

---

## Dataset switch — `photo2anime_3k` (2026-05-16)

All experiments after exp49 use the new merged dataset and a new val
split. See **[captains_log_3k.md](captains_log_3k.md)** for the 3k-era
log; metrics there are not directly comparable to numbers below.

Legacy log entries continue here for runs trained on `photo2anime_1k`.

---
