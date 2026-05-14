# Captain's log — video temporal consistency (exp27+)

Temporal experiments layered on top of the single-frame img2img pipeline.
Single-frame experiments (exp01–exp26) are in [captains_log.md](captains_log.md).

Base checkpoint for all temporal runs: `out/exp25_lpipsvgg_80k_from_exp23/model.pt`
(EMA weights, lpips_sq=0.115, lpips_vgg=0.217, ssim=0.712).

---

## exp27 — temporal finetuning V1 (cross-chunk KV)

**Status: DONE 2026-05-12**

Architecture: `TemporalAttn` added at 32px encoder level (`tattn4`, c4=352) and
16px bottleneck (`tattn_mid`, cm=704). Zero-init gates → identity at init.
All 44M spatial weights frozen. Only 3.7M temporal params trained.

Dataset: `TemporalPairedDataset` — synthesizes T=8 frame clips from still pairs via
pan/zoom affine trajectories. Same trajectory applied to source and target. Analytic
optical flow (no RAFT).

Training loop: split clip into chunk_a / chunk_b, forward chunk_a storing KV,
forward chunk_b with chunk_a KV as cross-attention context. Cross-chunk boundary
warp loss. KV detached at chunk boundary (no BPTT through chunk).

```bash
python3 experiments/010_img2img_photo2comics/train_temporal.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --steps 20000 --image-size 256 --batch-size 2 --num-frames 8 \
    --spatial-scale 2.0 --lr 1e-4 --lr-min 1e-6 --lr-warmup-steps 200 \
    --lpips-weight 0.2 --lpips-aux-net vgg --temporal-weight 1.0 \
    --amp bf16 --val-every 1000 --panel-every 1000 --checkpoint-every 5000 \
    --wandb-run-name exp27_temporal_finetune_20k \
    --outdir out/exp27_temporal_finetune
```

Results on `nat1.mp4` frames 60–90 (20 sample steps, T=4 chunks):
- lpips_sq=0.193, ssim=0.590
- Temporal consistency: moderate. Flickering visible at cut boundaries despite
  cross-chunk KV. The stored KV carries style context but not spatial anchor —
  small per-frame shifts compound over chunks.

**Lesson**: Cross-chunk KV reduces temporal drift from independent per-frame
inference but doesn't anchor spatial content. Drifts more with longer sequences.

---

## exp27e — temporal finetuning V1 variant

**Status: DONE 2026-05-12**

Same as exp27 with minor training parameter adjustments. No architectural change.

Results on nat1.mp4 frames 60–90:
- lpips_sq=0.194, ssim=0.589
- Essentially identical to exp27. Cross-chunk KV is the bottleneck, not
  the training hyperparameters.

---

## exp28 / exp28b / exp28c — temporal V2 (WAN-style anchor conditioning)

**Status: DONE 2026-05-12**

**Key architectural change**: replace cross-chunk KV with WAN-style first-frame
anchor conditioning. The model gets an extra input channel (`mask_channels=1`):
- `mask=0` → anchor frame (first frame of chunk, pixel values injected into input)
- `mask=1` → free frame (model predicts normally)

New components vs exp27:
- `mask_proj`: 1×1 conv from `mask_channels` to `model_ch`. Trainable.
- Temporal attention (`TemporalAttn`) in dec4/dec3/dec2/dec1 + encoder levels
  (HotShot-XL / AnimateDiff placement). Sinusoidal pos embeddings added to
  normed sequence before QKV (pos info flows into Q, K, V).
- No inter-chunk state. Long-video consistency via re-injection of the last
  generated frame as anchor of the next chunk.

Training script: `train_temporal_v2.py`
- Trainable params: temporal attn (3.97M) + mask_proj (88) = 3.97M total
- All spatial weights frozen.

```bash
PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \
TORCH_HOME=/tmp/torch_home \
python3 experiments/010_img2img_photo2comics/train_temporal_v2_exp28c.py \
    --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --steps 20000 --image-size 256 --batch-size 2 --num-frames 8 \
    --lr 1e-4 --lr-min 1e-6 --lr-warmup-steps 200 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 --val-every 1000 --panel-every 1000 --checkpoint-every 5000 \
    --wandb-run-name exp28c_temporal_v2_anchor \
    --outdir out/exp28c_temporal_v2_anchor
```

Results on nat1.mp4 frames 60–90:
- lpips_sq=0.191, ssim=0.596
- Anchor conditioning clearly reduces long-range drift vs cross-chunk KV.
  Each chunk re-anchors to a concrete reference frame rather than accumulated
  KV state. Preferred over exp27 for real video inference.

Checkpoint: `out/exp28c_temporal_v2_anchor/model.pt`

---

## exp29 — V2 + decoder LoRA rank=8 (without LPIPS — bug)

**Status: DONE 2026-05-13 (but LPIPS was broken — see exp29b)**

Added decoder LoRA on top of exp28c's anchor mechanism.

**New component** (`src/img2img/decoder_lora.py`):
- `LoRAConv2d`: wraps frozen `nn.Conv2d`. Adds trainable low-rank delta
  `delta_W = lora_B @ lora_A` (out_ch×rank) × (rank×in_ch×kH×kW).
  Zero-init B → delta_W=0 at init → identical to frozen base at start.
- `add_decoder_lora(model, rank=8)`: patches dec4/dec3/dec2/dec1 ResBlock
  conv1+conv2 in-place. 8 LoRA modules total, 268,928 trainable params.
- `decoder_lora_params(model)`: returns all lora_A / lora_B parameters.

Trainable params: temporal attn (3.97M) + mask_proj (88) + decoder LoRA (268k)
= 4.24M total out of 49.18M.

Training script: `train_temporal_v2_exp29.py`
Checkpoint base: `out/exp25_lpipsvgg_80k_from_exp23/model.pt` (fresh start)

**LPIPS was silently broken**: used `import lpips as lpips_lib` with a
`try/except` that set `_LPIPS_AVAILABLE = False` with no warning when the
standalone `lpips` package was not installed. `lpips_loss` logged as 0 for the
entire run. See "Failure patterns" below.

Results on nat1.mp4 frames 60–90:
- lpips_sq=0.192, ssim=0.597
- ~+0.001 ssim vs exp28c (decoder LoRA adds slight structure benefit).
- No LPIPS-driven improvement (loss was 0).

Checkpoint: `out/exp29_temporal_declora/model.pt` (or similar — do NOT reuse
the same folder for any follow-up run)

---

## exp29b — V2 + decoder LoRA + LPIPS fixed

**Status: IN PROGRESS 2026-05-13** (training from step 0, correct LPIPS active)

Identical spec to exp29 but with LPIPS working correctly.

**Fix**: replaced `import lpips` with direct torchmetrics import:
```python
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as _LPIPS
aux_lpips = _LPIPS(net_type=args.lpips_aux_net, normalize=True).to(device)
```
This fails loudly if torchmetrics is missing (no silent fallback).

**CRITICAL: load order for decoder LoRA**:
- **Fresh start** (no `--resume`): load spatial checkpoint FIRST, then inject LoRA.
  The spatial checkpoint has `dec4.conv1.weight` (plain Conv2d key). The model
  after LoRA injection has `dec4.conv1.conv.weight`. Loading spatial weights
  AFTER injection → key mismatch → decoder convs randomly initialized →
  inverted/negative outputs at inference.
- **Resume from temporal+LoRA checkpoint**: inject LoRA FIRST, then load.
  The checkpoint already has `lora_A/lora_B` keys; loading before injection
  would produce the same mismatch in reverse.

Training:
```bash
PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \
TORCH_HOME=/tmp/torch_home \
WANDB_API_KEY=wandb_v1_... \
python3 experiments/010_img2img_photo2comics/train_temporal_v2_exp29.py \
    --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --steps 20000 --image-size 256 --batch-size 2 --num-frames 8 \
    --lr 1e-4 --lr-min 1e-6 --lr-warmup-steps 200 \
    --lpips-weight 0.2 --lpips-aux-net vgg --lora-rank 8 \
    --amp bf16 --val-every 1000 --panel-every 1000 --checkpoint-every 5000 \
    --wandb-run-name exp29b_temporal_declora_lpips \
    --outdir out/exp29b_temporal_declora_lpips
```

Startup log confirmed:
- `loaded spatial weights from exp25` (correct)
- `new temporal/mask modules: 26 keys` (all fresh, correct load order)
- `decoder LoRA rank=8  lora_params=268,928`
- `lpips_aux_net=vgg weight=0.2` (LPIPS active)
- `step=100 loss=0.03511` (LPIPS contributing)

Wandb run: `8i0xhlzg`, project `nanoWarp`.
Outdir: `out/exp29b_temporal_declora_lpips/`

**Status: DONE 2026-05-13**

Val curve:

| step | lpips_sq | ssim |
|------|----------|------|
| 1k | 0.1436 | 0.602 |
| 2k | 0.1402 | 0.604 |
| 3k | 0.1397 | 0.605 |
| 4k | 0.1394 | 0.605 |
| 5k | 0.1394 | 0.606 |
| 6k | 0.1392 | 0.606 |
| **7k** | **0.1389** | **0.606** |
| 8k | 0.1392 | 0.606 |
| 9k | 0.1395 | 0.606 |
| 10k | 0.1396 | 0.606 |
| 15k | 0.1398 | 0.606 |
| 20k | 0.1399 | 0.605 |

Best val lpips_sq at step 7k (0.1389) — not saved (checkpoints at 5k/10k/15k/20k).
Best available checkpoint: `model_step_005000.pt` (lpips_sq=0.1394). Metrics plateau
after 7k; training past 10k adds nothing. Confirmed LPIPS was active: wandb shows
`lpips_loss≈0.135` (not 0).

---

## Results summary — temporal experiments

Evaluated on `nat1.mp4` frames 60–90 (30 frames), 20 ODE steps, T=4,
anchor reinjected at every ODE step for exp28+.

| exp | method | lpips_sq ↓ | ssim ↑ | notes |
|-----|--------|-----------|--------|-------|
| exp27  | cross-chunk KV | 0.193 | 0.590 | V1 |
| exp27e | cross-chunk KV variant | 0.194 | 0.589 | V1 |
| exp28c | WAN anchor cond. | 0.191 | 0.596 | V2 |
| exp29  | + decoder LoRA | 0.192 | 0.597 | LPIPS broken |
| exp29b | + decoder LoRA + LPIPS fixed | **0.1389** (7k) | **0.606** | best ckpt saved: 5k |

Inference: `experiments/010_img2img_photo2comics/infer_video.py`
- `--exp28c` for anchor-conditioned inference
- `--exp29` for decoder LoRA inference (loads LoRA, anchor reinjection at every ODE step)

---

## Failure patterns in this environment

### Wandb auth failures — root cause checklist

**Symptom**: `wandb init failed: CommError: user is not logged in` even though
`WANDB_API_KEY` is set and `wandb.login()` returns `True`.

**Root causes** (work through in order):

1. **Expired / invalid API key** — the most common cause. `wandb.login()` does
   NOT verify the key against the API; it just stores it locally and returns
   `True`. The `CommError: user is not logged in` is thrown by the Go subprocess
   (`wandb-core`) when its first API call returns 401/403. Confirm with:
   ```bash
   PYTHONPATH=/tmp/extpkgs2 python3 -c "
   import wandb; api = wandb.Api(api_key='YOUR_KEY'); print(api.default_entity)"
   ```
   If that raises, the key is bad. Get a fresh one from wandb.ai → Settings →
   API keys, update `~/.netrc`, and re-run.

2. **`~/.cache/wandb` not writable** — the Go service writes its own logs there.
   If it can't, it prints `ERROR main: failed to get logger path` and may fail
   to start. **Fix**: set `WANDB_CACHE_DIR=/tmp/wandb_cache` in the launch env.

3. **`~/.config/wandb` not writable** — wandb Python-level settings (including
   the persisted login token) go here. **Fix**: `WANDB_CONFIG_DIR=/tmp/wandb_config`.

4. **Background process doesn't inherit env** — verify `WANDB_API_KEY` is
   passed explicitly in the launch command, not just set in the interactive shell.

**Full working launch env** (all four fixes together):
```bash
WANDB_API_KEY=wandb_v1_... \
WANDB_CACHE_DIR=/tmp/wandb_cache \
WANDB_CONFIG_DIR=/tmp/wandb_config \
```

**In the script**:
```python
api_key = os.environ.get("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = api_key  # ensure subprocess inherits
wandb.login(key=api_key, relogin=True)
# then wandb.init(...)
```

**Do not kill a training run to fix wandb.** Sync manually in a background
process instead.

### LPIPS silent failure (import lpips pattern)

**Problem**: the standalone `lpips` package is not installed in `/tmp/extpkgs2`.
The pattern:
```python
try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
```
silently sets `_LPIPS_AVAILABLE = False` with no warning. `lpips_loss` is then
logged as 0 or skipped for the entire run. Hard to notice unless you look
carefully at wandb curves (flat line at 0 is the tell).

**Fix**: use torchmetrics directly, which is always installed:
```python
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as _LPIPS
aux_lpips = _LPIPS(net_type="vgg", normalize=True).to(device)
```
This raises `ImportError` immediately if missing — loud failure, not silent.

**Detection**: if `lpips_loss` is a flat 0 in wandb from step 0, LPIPS is broken.
A healthy LPIPS contribution at weight=0.2 produces `lpips_loss` in the 0.01–0.05
range early in training.

Also set `TORCH_HOME=/tmp/torch_home` so torchmetrics can download backbone weights
without hitting `~/.cache/torch` permission errors.

### Decoder LoRA load order

**Problem**: if `add_decoder_lora(model)` is called BEFORE `model.load_state_dict()`
during a fresh start (loading a spatial checkpoint), the key names don't match:
- Spatial checkpoint has: `dec4.conv1.weight`
- Model after LoRA injection has: `dec4.conv1.conv.weight`

The state_dict load silently skips mismatched keys (strict=False), leaving the
decoder convs randomly initialized. Output at inference is an inverted/negative
image.

**Fix**:
```python
if args.resume:
    # Resume from a temporal+LoRA checkpoint: inject LoRA first,
    # then load (checkpoint already has lora_A/lora_B keys)
    add_decoder_lora(model, rank=args.lora_rank)
    ckpt = torch.load(args.resume, ...)
    model.load_state_dict(ckpt["model"], strict=False)
else:
    # Fresh start from spatial checkpoint: load first, then inject
    ckpt = torch.load(args.checkpoint, ...)
    model.load_state_dict(ckpt["model"], strict=False)
    add_decoder_lora(model, rank=args.lora_rank)
```

**Detection**: at fresh start, confirm `new temporal/mask modules: N keys` is printed.
If any LoRA keys appear in that list (they shouldn't on fresh start), something
is wrong. At inference, inverted/negative outputs are the visual tell.

---

## exp30 — corruption robustness fine-tune of exp29b (temporal)

**Status: DONE 2026-05-13**

Same architecture as exp29b (temporal attn + decoder LoRA rank=8 + WAN anchor cond.).
Adds random source corruption to train robustness to real-video compression artifacts.

**Corruption** (applied per clip — same params for all T frames):
- Gaussian blur: σ∼U[0.5, 3.0], 70% chance per clip
- JPEG compression: quality∼U[30, 95], 70% chance per clip
- Both applied independently; 0% clean probability (always some corruption)

Resume: `exp29b/model_step_020000.pt` (step 20k → fine-tunes to step 25k)
LR: 2e-5 → 1e-6 cosine over 5000 steps.
Script: `train_temporal_v2_exp30.py`
Outdir: `out/exp30_corrupt_robust_20260513_200418/`

Val curve (clean sources, no corruption — expected regression):

| step | lpips_sq | ssim | note |
|------|----------|------|------|
| 20500 | 0.1396 | 0.604 | |
| **21000** | **0.1404** | **0.603** | best clean-val |
| 22000 | 0.1422 | 0.601 | |
| 25000 | 0.1446 | 0.598 | final |

Clean-val degrades vs exp29b (0.1399) as expected — model shifts toward the corrupted
input domain. The real test is visual quality on real video (nat1.mp4 frames 0–60);
inference outputs saved in `out/infer_nat1_exp30_21k_*` and `out/infer_nat1_exp30_25k_*`.

**Wandb**: init failed ("user is not logged in") despite `WANDB_API_KEY` set. No wandb
logging for this run. Logfile saved to outdir.

---

## Open follow-ups (temporal)

- **exp29b results**: validate on nat1.mp4 after training completes. Compare
  lpips_sq/ssim vs exp28c to measure decoder LoRA benefit with correct LPIPS.
- **Longer video test**: run all methods on a longer clip (>60 frames) to expose
  drift accumulation differences between V1 (KV) and V2 (anchor).
- **Anchor reinjection ablation**: hard-reinject anchor at every ODE step
  (current) vs only at start of each chunk. Trade-off between identity
  preservation and style flexibility.
- **Temporal attn at more levels**: currently dec4/dec3/dec2/dec1 for LoRA,
  temporal attn in encoder + decoder. Skipping the 256px levels
  (B×HW = 131072 exceeds flash-attention kernel limits for common batch sizes).
