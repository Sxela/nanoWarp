## exp31 — exp25 fine-tune at 512px with source corruption robustness

**Status: IN PROGRESS 2026-05-13** (steps 80k→90k)

Fine-tunes the exp25 checkpoint (best single-frame model) at 512×512 with source
corruption to improve robustness to real-video blur and compression artifacts.

Architecture: identical to exp25 (flow FM, mc=88, attn 16/32/64, no source encoder,
LPIPS-VGG 0.2). All weights trainable (no freeze).

**Key changes vs exp25**:
- Resolution: 256px → 512px
- Augmentation: `aug_resize_scale=2.0` (crops 512 from ~1024px images)
- Source corruption per image (target always clean):
  - 20% chance: no corruption (clean source)
  - 80% chance: independently apply blur σ∼U[0.5,3.0] (70% prob) and/or
    JPEG quality∼U[30,95] (70% prob)
- LR: 2e-5 → 1e-6 cosine (vs 2e-4 for original exp25 — fine-tune rate)
- 10k steps (resume step 80k → target 90k)

```bash
OUTDIR=out/exp31_corrupt512_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR
PYTHONPATH=. \
TORCH_HOME=/tmp/torch_home \
MPLCONFIGDIR=/tmp/mplconfig \
WANDB_API_KEY=$WANDB_API_KEY \
WANDB_CACHE_DIR=/tmp/wandb_cache \
WANDB_CONFIG_DIR=/tmp/wandb_config \
python3 experiments/010_img2img_photo2comics/train_exp31_corrupt512.py \
    data/photo2anime_1k/photo2anime_1k \
    --resume out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --steps 10000 --image-size 512 --aug-resize-scale 2.0 \
    --lr 2e-5 --lr-min 1e-6 --lr-warmup-steps 200 \
    --corrupt-blur-max 3.0 --corrupt-jpeg-min 30 --clean-prob 0.2 \
    --wandb --wandb-run-name exp31_corrupt512 \
    --outdir $OUTDIR \
    2>&1 | tee $OUTDIR/train.log
```

Outdir: `out/exp31_corrupt512_20260513_214306/`
Wandb: https://wandb.ai/alx-spirin/nanoWarp/runs/4k1iquss

**Wandb debug**: Earlier runs failed with `CommError: user is not logged in` despite
`WANDB_API_KEY` set. Root cause: the stored API key had expired — `wandb.login()`
returns `True` without verifying the key; the Go subprocess (`wandb-core`) is what
actually calls the API and fails. Secondary issues: `~/.cache/wandb` and
`~/.config/wandb` not writable → fixed with `WANDB_CACHE_DIR` and `WANDB_CONFIG_DIR`.
Full debug notes in [captains_log_video.md#wandb-auth-failures](captains_log_video.md).

**Val curve** (clean sources, ↓ better):

| step  | lpips_sq | ssim   |
|-------|----------|--------|
| 81000 | 0.1447   | 0.6347 | ← same as exp25 (1k into fine-tune)
| 82000 | 0.1622   | 0.6161 |
| 84000 | 0.1767   | 0.6032 |
| 86000 | 0.1794   | 0.6031 |
| 88000 | 0.1853   | 0.5964 |
| 90000 | 0.1824   | 0.5973 |

**Conclusion**: clean-val degraded 0.1447 → 0.1824 (+26% LPIPS regression).
Expected — same pattern as exp30 (temporal corruption). The model learns to
de-corrupt sources, which changes its response to clean sources. For clean-source
inference use exp25 (step 80k). For real-video compressed inputs, use exp31 final
(step 90k) — nat1 nat1_step_09*.png frames will show whether it improved visually.

Nat1 frame-0 panels saved every 1k steps in the outdir.

---
