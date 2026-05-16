## exp34 — exp33 recipe + symmetric decoder spatial self-attention

**Status: WIRED 2026-05-14** (decoder-attn on the full-aug recipe; pair with
exp37 for the same architecture change on minimal aug)

Clean A/B vs exp33 (same data, same aug, same compute budget) to isolate
the contribution of putting `BottleneckAttention` on the decoder side at
the same resolutions as the encoder. Closes the long-standing asymmetry
in nanoWarp's UNet, which had spatial self-attn only in encoder + bottleneck
and never in the decoder (vs SD/SDXL convention which puts it symmetrically).

**Architecture delta vs exp33** — with `attn_resolutions = (16, 32, 64)`
and `image_size = 256`:
- `attn_dec4` at H/8 (32px), channels c3=352 — mirrors `attn4`
- `attn_dec3` at H/4 (64px), channels c3=352 — mirrors `attn3`
- `attn_dec2` at H/2 (128px) — None (128 ∉ attn_set)
- `attn_dec1` at H (256px) — None (256 ∉ attn_set)
- 16px bottleneck `mid_attn` already always on, unchanged.

Each `attn_dec*` is applied after the corresponding `dec*` ResBlock,
before any FiLM/tattn hooks.

**Gating**: `--use-decoder-attn` flag (default off → exp33 behaviour). When
on, the decoder attn modules are instantiated at exactly the same
resolutions as the encoder attn — single `attn_resolutions` knob controls
both.

**Recipe**: identical to exp33 except `--use-decoder-attn` is passed. Reads
the architectural delta on top of the full corruption-robustness aug stack;
exp37 below runs the same architecture change on the minimal-aug
(exp23-like) recipe so the delta can be read without aug confound.

```bash
WANDB_API_KEY=... bash scripts/run_exp34_decoder_attn_at_exp33_recipe.sh
```

Script: `scripts/run_exp34_decoder_attn_at_exp33_recipe.sh`
Outdir: `out/exp34_decoder_attn_noenc_attn163264_bf16_mc88_256px_20k`

Results: TBD.

---
