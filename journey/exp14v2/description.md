### exp14v2 — exp10 architecture on the 1k-pair dataset, 256px — DONE

After the LPIPS-aux bug fix (see Known bugs), exp14 was re-launched as
exp14v2 with the same spec on the 1k-pair dataset at 256px. Validation on
the 1k val split:

| step | LPIPS ↓ | SSIM ↑ |
|---:|---:|---:|
|  5k | 0.1770 | 0.632 |
| **10k** | **0.1767 (best)** | 0.654 |
| 15k | 0.1798 | 0.666 |
| 20k | 0.1819 | 0.674 |
| 25k | 0.1805 | 0.679 |
| 30k | 0.1816 | 0.683 |
| 35k | 0.1830 | 0.685 |
| 40k | 0.1832 | **0.686 (final)** |

**Same LPIPS-regression-while-SSIM-keeps-climbing pattern** we saw in
exp10/exp12 — LPIPS bottoms at step 10k; training past 10k trades
perceptual quality for structural fidelity. Best-LPIPS checkpoint is
step 10k, not the 40k final.

Comparing on the same 1k val set (eval on identical val pairs even for
older checkpoints):

| model | LPIPS-squeeze | SSIM |
|---|---:|---:|
| exp12 (287 pairs, 20k, 256px) | 0.190 | 0.635 |
| exp14v2 (1k pairs, 40k, 256px) | 0.183 | **0.686** |
| Δ | -3.7% | **+8.0%** |

**Surprise**: 3.5× more paired data and 2× more steps mainly bought us
**SSIM (structural diversity)**, not LPIPS (perceptual quality). This is
the opposite of what I predicted earlier. The 1k dataset has more pose /
composition variety, which exp14v2 captures (SSIM up), but the perceptual
ceiling didn't move much from data alone.
