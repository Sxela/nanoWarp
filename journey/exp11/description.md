### exp11 — exp10 + linear RGB — DONE (essentially neutral)

Adds physically-correct linear-RGB training on top of the confirmed-best
exp10 architecture (no encoder, mc=88, attn 8,16,32, bf16, FM, LPIPS 0.2,
dropout 0.15).

The hypothesis: FM's interpolant `x_t = (1-t)·source + t·target + σ·noise`
and bilinear upsampling are physically meaningful operations on light
intensity, which sRGB does not represent linearly. Operating in linear RGB
should clean up off-path drift and slightly sharpen edges/colors. We expect
2-5% LPIPS improvement.

Wired via `--color-space linear_rgb`. Conversions happen at boundaries:
- **Dataset**: PIL loads sRGB, conversion to linear after PIL augmentation
  ([dataset.py L160-L166](../src/img2img/dataset.py#L160-L166)).
- **Source encoder** (when used): linear → sRGB before the ResNet18 forward
  ([model.py L344-L347](../src/img2img/model.py#L344-L347)). exp11 uses
  `--no-source-encoder` so this branch is inactive but kept correct.
- **LPIPS aux**: trainer wraps the LPIPS network so it always sees sRGB
  inputs.
- **Panels**: trainer / validate.py / infer.py convert linear → sRGB at
  the display boundary so panels are display-correct and metrics are
  comparable across runs.

Validation always reports SSIM/LPIPS in sRGB regardless of training color
space, so exp11 numbers compare directly to exp08-lpips/exp10/etc.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --color-space linear_rgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp11_linrgb_noenc_attn832_bf16_mc88_30k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,linear-rgb,exp11" `
    --outdir out/exp11_linrgb_noenc_attn832_bf16_mc88_30k
```

Notes:
- `--num-workers 8` enables the full dataloader speedup (pin_memory +
  persistent_workers + non_blocking auto-on, ~2.7× data-loading throughput).
- `--checkpoint-every 5000` saves 5 intermediate + 1 final = 6 checkpoints
  total (~2 GB disk) instead of 30 (~20 GB). Still enough granularity to
  identify best-LPIPS step from the val curve. If you need finer for some
  reason, set to 2500.

Predictions (made before the run):
- LPIPS: -2 to -5% vs exp10 (0.146-0.150 range vs exp10's 0.153)
- SSIM: tiny shift, probably +/-1%.
- Visual: cleaner palette in mid-trajectory frames.

**Actual val curve:**

| step | exp10 SSIM | exp11 SSIM | exp10 LPIPS | exp11 LPIPS |
|---:|---:|---:|---:|---:|
|  5k | — | 0.686 | — | 0.160 |
| 10k | — | 0.713 | — | 0.154 |
| 15k | — | 0.725 | — | 0.153 |
| 20k | 0.736 | 0.732 | 0.153 | 0.154 |
| 30k | 0.740 | 0.737 | 0.158 | 0.156 |

**Outcome: linear RGB is essentially neutral.** Within val-set noise (~1-2%
with n=50). The -2 to -5% LPIPS prediction was wrong.

**Why linear RGB didn't deliver:**
- LPIPS aux is computed in sRGB (we convert at the boundary), so the
  perceptual gradient signal is identical between sRGB and linear RGB
  trainings. Only the MSE/v_target and bilinear-upsample paths use
  physically correct math, and at 128px those don't dominate.
- bf16 numerical noise probably drowns out small precision gains from
  linear math.
- PIL augmentations still happen in sRGB before conversion, so part of
  the data path retains sRGB nonlinearity.
- Most importantly: **all three architectures (exp08-lpips, exp10, exp11)
  converge to the same ~0.152-0.153 best-LPIPS ceiling. The architectural
  ceiling for this dataset is real.**

### Architectural ceiling — strong evidence (at 128px specifically)

| model | best LPIPS | best SSIM | trainable params | external priors |
|---|---:|---:|---:|---|
| exp08-lpips | **0.152** | 0.719 | 31.6M | ImageNet ResNet18 |
| exp10 | 0.153 | **0.736** | 43.9M | none |
| exp11 (linear) | 0.153 | 0.732 | 43.9M | none |
| floor (128px) | 0.199 | 0.617 | 0 | — |

All three architectures land within ~0.001 LPIPS of each other at 128px.
**More architecture work at 128px produces ~zero perceptual gain.**

**Caveat (added 2026-05-10 after exp12):** this ceiling turned out to be
**a 128px artifact, not an absolute one.** exp12 at 256px hit LPIPS 0.142,
substantially below the 0.152 ceiling — and the LPIPS-vs-floor improvement
went from 23% to 53%, a 2.3× relative jump. The architectures weren't out
of expressiveness; they were out of pixels. Resolution turned out to be
the highest-leverage single axis we tested.

### Recommended next moves (revised after exp11)

| candidate | expected LPIPS gain | cost | priority |
|---|---|---|---|
| **Larger paired dataset (1k-10k pairs)** | **5-10%** | data generation | **🔴 high** |
| Higher resolution (256px) | unclear, possibly meaningful | 4x compute | 🟡 medium |
| VGG-backbone LPIPS aux | unclear, plausible | one flag | 🟡 medium |
| Longer training + best-LPIPS early-stop | 1-2% | trivial | 🟡 medium |
| `--lpips-weight 0.4` ablation | 1-2% | 1 flag | 🟡 medium |
| FiLM + 2 ResBlocks (Path C continuation) | <2% | hour | 🟢 low |
| linear RGB everywhere (incl. PIL aug rewrite) | <2% | hours | ⚫ skip |

---

## Planned next experiments

(Below the line: experiments specified but not yet run. Each is a
single-variable change vs the current best architecture (exp10) so
attribution stays clean. exp12 and exp13 are orthogonal axes and can
run in parallel on the two VMs.)
