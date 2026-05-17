"""Emit per-experiment architecture.html into journey/expNN/ — but ONLY
when the architecture changed vs the previous experiment.

Walks the experiments in chronological order, computes an `arch signature`
for each (set of active modules, model widths, key flags), and writes
a self-contained HTML page with an inline SVG diagram when the signature
differs from the immediate predecessor. Otherwise no file is written —
that experiment inherits the most recent architecture.

Run: `python scripts/build_journey_arch.py`
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# Windows console defaults to cp1252; force UTF-8 so unicode in labels doesn't
# blow up the print statements.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

JOURNEY = Path(__file__).resolve().parent.parent / "journey"


# ---------------------------------------------------------------------------
# Arch timeline — hand-curated from captain's log + run scripts.
# Each entry: (first_exp_introducing_it, arch_dict). Exps between two
# entries inherit the earlier arch's signature.
# ---------------------------------------------------------------------------

TIMELINE: list[tuple[str, dict]] = [
    ("exp01", {
        "label": "Baseline: ε-diffusion UNet with ImageNet-ResNet18 encoder",
        "kind": "unet_with_encoder",
        "method": "diffusion (ε-prediction)",
        "encoder": "ResNet18 (partial freeze, ImageNet pretrained)",
        "source_in_stem": False,
        "model_ch": 64,
        "attn_resolutions": (8,),
        "upsample": "resize_conv",
        "extras": [],
        "notes": "Random-t x0_hat panels looked clean, but full DDIM reverse sampling collapsed to grey scribbles.",
    }),
    ("exp02", {
        "label": "+ source-in-stem (concat source channels into input conv)",
        "kind": "unet_with_encoder",
        "method": "diffusion (ε-prediction)",
        "encoder": "ResNet18 (partial freeze)",
        "source_in_stem": True,
        "model_ch": 64,
        "attn_resolutions": (8,),
        "upsample": "resize_conv",
        "extras": [],
        "notes": "Loss dropped, structural similarity hurt — source at stem trades random-t reconstruction for SSIM.",
    }),
    ("exp07", {
        "label": "+ flow matching + freeze entire source encoder (stability fix)",
        "kind": "unet_with_encoder",
        "method": "rectified flow matching",
        "encoder": "ResNet18 (fully frozen, eval mode locked)",
        "source_in_stem": True,
        "model_ch": 64,
        "attn_resolutions": (8,),
        "upsample": "resize_conv",
        "extras": [],
        "notes": "Earlier 20k runs collapsed at ~step 5k; encoder freeze + flow eliminated the spike. LPIPS aux added next (exp08-lpips).",
    }),
    ("exp08", {
        "label": "Drop ResNet18 encoder entirely; widen UNet to mc=88",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux 0.2",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (8,),
        "upsample": "resize_conv",
        "extras": [],
        "notes": "Removed encoder priors. 'Encoder helps' was 90% amortized ImageNet pretraining, not architecture.",
    }),
    ("exp09", {
        "label": "+ pixel_shuffle upsample (replaces resize_conv)",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (8,),
        "upsample": "pixel_shuffle",
        "extras": [],
        "notes": "Pixel-shuffle didn't beat resize_conv at this scale.",
    }),
    ("exp10", {
        "label": "+ multi-scale attention at H/2, H/4, H/8 (encoder + bottleneck)",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (16, 32, 64),
        "upsample": "resize_conv",
        "extras": [],
        "notes": "exp10/exp14v2 canonical baseline. Held through exp33 with only training/data changes.",
    }),
    ("exp34", {
        "label": "+ symmetric decoder spatial self-attention",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (16, 32, 64),
        "upsample": "resize_conv",
        "extras": ["decoder_attn"],
        "notes": "Mirrors encoder attn on the decoder side at the same resolutions. SD/SDXL convention.",
    }),
    ("exp35", {
        "label": "+ source pyramid + FiLM modulation of the decoder",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (16, 32, 64),
        "upsample": "resize_conv",
        "extras": ["decoder_attn", "source_pyramid"],
        "notes": "Tiny in-model conv pyramid (~1.8M) computes source features at 4 resolutions; FiLM γ,β modulate decoder activations. Canonical baseline from here forward.",
    }),
    ("exp36", {
        "label": "+ DiT bottleneck (replaces mid_attn + mid2 with 4 DiT blocks)",
        "kind": "unet_noenc",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "model_ch": 88,
        "attn_resolutions": (16, 32, 64),
        "upsample": "resize_conv",
        "extras": ["decoder_attn", "source_pyramid", "dit_bottleneck"],
        "notes": "adaLN-zero modulated DiT blocks at the 16×16 token grid. +28M params for marginal lift.",
    }),
    ("exp47", {
        "label": "Pure-pixel DiT (HiDream-O1 style, no UNet)",
        "kind": "pixel_dit",
        "method": "rectified flow matching + LPIPS aux",
        "encoder": None,
        "source_in_stem": True,
        "dit_dim": 384,
        "dit_layers": 11,
        "dit_patch": 16,
        "extras": [],
        "notes": "patch=16 patchify → 11 DiT blocks → unpatchify. 48.5M params. Block artifacts at this data scale.",
    }),
    # No more arch changes after exp47/48. exp49 onwards revert to exp35-arch
    # (or just the canonical decoder_attn + pyramid combo) for the data-scale
    # experiments. We re-emit the exp35 signature at exp49 so the data
    # narrative is clear, but no new architecture file is generated since the
    # signature matches the previous exp35 entry.
]


def signature(arch: dict) -> tuple:
    """Hashable fingerprint of an architecture for change detection."""
    keys = ["kind", "method", "encoder", "source_in_stem", "model_ch",
            "attn_resolutions", "upsample", "dit_dim", "dit_layers",
            "dit_patch"]
    sig = tuple(arch.get(k) for k in keys)
    extras = tuple(sorted(arch.get("extras", [])))
    return sig + extras


def list_exps() -> list[str]:
    """Sorted exp tags from journey/ folders."""
    def sort_key(name: str):
        m = re.match(r"exp(\d+)([a-z]*)", name.lower())
        if not m:
            return (9999, name)
        return (int(m.group(1)), m.group(2))
    return sorted([p.name for p in JOURNEY.iterdir() if p.is_dir() and p.name.startswith("exp")],
                  key=sort_key)


def arch_for(exp: str) -> dict:
    """Look up the most recent arch entry ≤ this exp number (by chronological order)."""
    def num(name):
        m = re.match(r"exp(\d+)", name.lower())
        return int(m.group(1)) if m else 0
    target = num(exp)
    chosen = TIMELINE[0][1]
    for tag, arch in TIMELINE:
        if num(tag) <= target:
            chosen = arch
    return chosen


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def render_unet_svg(arch: dict, title: str) -> str:
    """Render the conv-UNet variant (with or without source encoder)."""
    has_enc = arch["kind"] == "unet_with_encoder"
    extras = arch.get("extras", [])
    has_dec_attn = "decoder_attn" in extras
    has_pyramid = "source_pyramid" in extras
    has_dit = "dit_bottleneck" in extras
    mc = arch.get("model_ch", 64)
    attn_set = set(arch.get("attn_resolutions") or ())
    upsample = arch.get("upsample", "resize_conv")
    method = arch.get("method", "")

    # Compute channel widths from mc.
    c1, c2, c3, c4, cm = mc, mc * 2, mc * 4, mc * 4, mc * 8

    # Level resolutions assume image_size=256 for the diagram.
    res = [256, 128, 64, 32]
    bottleneck_res = 16

    # Module labels with optional state.
    def attn_label(level_res: int) -> str:
        on = level_res in attn_set
        return f'attn @ {level_res}' if on else f'attn @ {level_res} (off)'

    if has_enc:
        enc_desc = arch.get("encoder", "") or ""
        # Split "ModelName (status, status2)" → ["ModelName", "status,", "status2"]
        # so the description wraps across 2-3 lines inside the box.
        if " (" in enc_desc and enc_desc.endswith(")"):
            model_name, _, status = enc_desc.partition(" (")
            status = status.rstrip(")")
            status_parts = [s.strip() for s in status.split(",")]
            desc_lines = [model_name] + status_parts
        else:
            desc_lines = [enc_desc]
        line_h = 12
        text_y0 = 138
        desc_svg = "".join(
            f'<text x="130" y="{text_y0 + i*line_h}" text-anchor="middle" font-size="10">'
            f'{html.escape(line)}</text>'
            for i, line in enumerate(desc_lines)
        )
        rect_h = 30 + line_h * len(desc_lines) + 6
        enc_block = (
            f'<rect x="20" y="100" width="220" height="{rect_h}" rx="6" '
            f'fill="#fef6e4" stroke="#c89b3c"/>'
            f'<text x="130" y="120" text-anchor="middle" font-weight="600">SourceEncoder</text>'
            f'{desc_svg}'
        )
    else:
        enc_block = ""

    pyramid_block = (
        '<rect x="40" y="180" width="160" height="50" rx="6" fill="#fde4d4" stroke="#d94e2a" stroke-width="1.5"/>'
        '<text x="120" y="200" text-anchor="middle" font-weight="600" fill="#a8391e">SourcePyramid</text>'
        '<text x="120" y="218" text-anchor="middle" font-size="10">'
        f'4 stages -> (c1..c4) features</text>'
    ) if has_pyramid else ""

    # Encoder/decoder mini-grid.
    def lvl_row(y: int, label: str, ch: int, lvl_res: int, side: str) -> str:
        attn_color = "#e0d7f5" if lvl_res in attn_set else "#fff"
        attn_stroke = "#5a4d99" if lvl_res in attn_set else "#bbb"
        attn_dash = "" if lvl_res in attn_set else 'stroke-dasharray="4 3"'
        rb_x = 260 if side == "enc" else 540
        attn_x = rb_x + 100
        if side == "dec":
            # decoder also has FiLM and decoder_attn possibilities
            return (
                f'<rect x="{rb_x}" y="{y}" width="90" height="36" rx="4" fill="#dcf3df" stroke="#3f8a4f"/>'
                f'<text x="{rb_x+45}" y="{y+22}" text-anchor="middle" font-weight="600">dec ({ch})</text>'
                + (
                    f'<rect x="{attn_x}" y="{y}" width="80" height="36" rx="4" fill="{attn_color}" stroke="{attn_stroke}" {attn_dash}/>'
                    f'<text x="{attn_x+40}" y="{y+22}" text-anchor="middle" font-size="11">attn_dec</text>'
                    if has_dec_attn and lvl_res in attn_set else ""
                )
                + (
                    f'<rect x="{attn_x + (90 if (has_dec_attn and lvl_res in attn_set) else 0)}" y="{y}" width="80" height="36" rx="4" fill="#fff" stroke="#d94e2a" stroke-dasharray="4 3"/>'
                    f'<text x="{attn_x + (90 if (has_dec_attn and lvl_res in attn_set) else 0)+40}" y="{y+22}" text-anchor="middle" font-size="11" fill="#a8391e">FiLM</text>'
                    if has_pyramid else ""
                )
            )
        # encoder
        return (
            f'<rect x="{rb_x}" y="{y}" width="90" height="36" rx="4" fill="#dbe9ff" stroke="#3a6fbc"/>'
            f'<text x="{rb_x+45}" y="{y+22}" text-anchor="middle" font-weight="600">down ({ch})</text>'
            + (
                f'<rect x="{attn_x}" y="{y}" width="80" height="36" rx="4" fill="{attn_color}" stroke="{attn_stroke}" {attn_dash}/>'
                f'<text x="{attn_x+40}" y="{y+22}" text-anchor="middle" font-size="11">attn</text>'
                if lvl_res in attn_set else ""
            )
        )

    rows = ""
    for i, lvl_res in enumerate(res):
        y = 90 + i * 60
        ch = (c1, c2, c3, c4)[i]
        rows += lvl_row(y, str(lvl_res), ch, lvl_res, "enc")
        rows += lvl_row(y, str(lvl_res), ch, lvl_res, "dec")
        # Skip arrow.
        rows += (f'<path d="M 350 {y+18} L 540 {y+18}" stroke="#9aa3b2" stroke-dasharray="5 4" stroke-width="1.4"/>')

    # Bottleneck.
    bot_y = 90 + len(res) * 60
    if has_dit:
        bot = (
            f'<rect x="260" y="{bot_y}" width="370" height="46" rx="6" fill="#fff8f4" stroke="#d94e2a" stroke-width="1.6"/>'
            f'<text x="445" y="{bot_y+20}" text-anchor="middle" font-weight="700" fill="#a8391e">DiT bottleneck (4× adaLN-zero blocks @ {bottleneck_res}px, dim={cm})</text>'
            f'<text x="445" y="{bot_y+36}" text-anchor="middle" font-size="10" fill="#555">replaces mid_attn + mid2</text>'
        )
    else:
        bot = (
            f'<rect x="260" y="{bot_y}" width="370" height="46" rx="6" fill="#fff3cf" stroke="#b88a1d"/>'
            f'<text x="445" y="{bot_y+20}" text-anchor="middle" font-weight="600">mid1 -> mid_attn -> mid2 (cm={cm}, {bottleneck_res}px)</text>'
        )

    # Compose.
    svg_h = bot_y + 120
    svg = f"""
<svg viewBox="0 0 850 {svg_h}" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" font-size="11">
  <text x="20" y="30" font-size="14" font-weight="700">{html.escape(title)}</text>
  <text x="20" y="50" font-size="11" fill="#555">{html.escape(method)} · mc={mc} · attn_res={sorted(attn_set) or 'none'} · upsample={upsample}</text>

  <text x="40" y="80" font-size="11" fill="#3a6fbc" font-weight="700">Encoder</text>
  <text x="540" y="80" font-size="11" fill="#3f8a4f" font-weight="700">Decoder</text>

  {enc_block}
  {pyramid_block}
  {rows}
  {bot}
</svg>
"""
    return svg


def render_pixel_dit_svg(arch: dict, title: str) -> str:
    dim = arch.get("dit_dim", 384)
    layers = arch.get("dit_layers", 11)
    patch = arch.get("dit_patch", 16)
    method = arch.get("method", "")
    return f"""
<svg viewBox="0 0 900 360" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" font-size="11">
  <text x="20" y="30" font-size="14" font-weight="700">{html.escape(title)}</text>
  <text x="20" y="50" font-size="11" fill="#555">{html.escape(method)} · dim={dim} · layers={layers} · patch={patch}</text>

  <rect x="40" y="80" width="120" height="50" rx="6" fill="#fef6e4" stroke="#c89b3c"/>
  <text x="100" y="100" text-anchor="middle" font-weight="600">source</text>
  <text x="100" y="118" text-anchor="middle" font-size="10">(B, 3, H, W)</text>

  <rect x="180" y="80" width="120" height="50" rx="6" fill="#fef6e4" stroke="#c89b3c"/>
  <text x="240" y="100" text-anchor="middle" font-weight="600">noisy_target</text>
  <text x="240" y="118" text-anchor="middle" font-size="10">(B, 3, H, W)</text>

  <text x="240" y="150" text-anchor="middle" font-size="10" fill="#666">concat -> 6 ch</text>

  <rect x="100" y="170" width="200" height="46" rx="6" fill="#dbe9ff" stroke="#3a6fbc"/>
  <text x="200" y="190" text-anchor="middle" font-weight="600">PatchEmbed (Conv {patch}×{patch}, stride {patch})</text>
  <text x="200" y="206" text-anchor="middle" font-size="10">-> ((H/{patch})·(W/{patch}), {dim})</text>

  <rect x="350" y="170" width="240" height="46" rx="6" fill="#e0d7f5" stroke="#5a4d99"/>
  <text x="470" y="190" text-anchor="middle" font-weight="600">DiT blocks × {layers}</text>
  <text x="470" y="206" text-anchor="middle" font-size="10">adaLN-zero → MHSA → adaLN-zero → MLP</text>

  <rect x="640" y="170" width="200" height="46" rx="6" fill="#dcf3df" stroke="#3f8a4f"/>
  <text x="740" y="190" text-anchor="middle" font-weight="600">Head (Linear → Unpatchify)</text>
  <text x="740" y="206" text-anchor="middle" font-size="10">-> velocity v (B, 3, H, W)</text>

  <line x1="300" y1="193" x2="350" y2="193" stroke="#666" stroke-width="1.4"/>
  <line x1="590" y1="193" x2="640" y2="193" stroke="#666" stroke-width="1.4"/>
</svg>
"""


def render_html(exp: str, arch: dict, previous_exp: str | None) -> str:
    title = f"{exp} architecture — {arch['label']}"
    if arch["kind"] == "pixel_dit":
        svg = render_pixel_dit_svg(arch, title)
    else:
        svg = render_unet_svg(arch, title)

    inherits = ""
    if previous_exp:
        inherits = (
            f'<p style="font-size:12px;color:#555;">Inherits from previous architecture; '
            f'see <a href="../{previous_exp}/architecture.html">{previous_exp}/architecture.html</a> '
            f'for the prior state.</p>'
        )

    extras = arch.get("extras") or []
    extras_html = ", ".join(extras) if extras else "none"
    notes = arch.get("notes", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 980px; margin: 32px auto; padding: 0 16px; color: #1d1d1f; }}
  h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
  .subtitle {{ color: #555; font-size: 13px; }}
  .arch-box {{ background: white; border: 1px solid #e5e5e5; border-radius: 8px;
               padding: 14px; margin: 16px 0; }}
  svg {{ max-width: 100%; height: auto; display: block; }}
  .meta dt {{ font-weight: 600; margin-top: 6px; font-size: 13px; }}
  .meta dd {{ margin: 2px 0 6px 0; font-family: ui-monospace, monospace; font-size: 12px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">{html.escape(arch.get('method',''))} · this file is emitted only when arch changed vs the previous experiment</div>

{inherits}

<div class="arch-box">{svg}</div>

<h2 style="font-size:14px;margin:18px 0 6px 0;">What changed in {exp}</h2>
<p style="font-size:13px;">{html.escape(notes)}</p>

<dl class="meta">
  <dt>kind</dt><dd>{html.escape(arch['kind'])}</dd>
  <dt>method</dt><dd>{html.escape(arch.get('method',''))}</dd>
  <dt>source encoder</dt><dd>{html.escape(str(arch.get('encoder')))}</dd>
  <dt>source-in-stem</dt><dd>{arch.get('source_in_stem')}</dd>
  <dt>model_ch</dt><dd>{arch.get('model_ch','—')}</dd>
  <dt>attn_resolutions</dt><dd>{arch.get('attn_resolutions','—')}</dd>
  <dt>upsample</dt><dd>{arch.get('upsample','—')}</dd>
  <dt>extras</dt><dd>{html.escape(extras_html)}</dd>
</dl>
</body>
</html>
"""


def main():
    exps = list_exps()
    prev_sig = None
    prev_exp_with_arch = None
    written = 0
    skipped = 0
    for exp in exps:
        arch = arch_for(exp)
        sig = signature(arch)
        if sig == prev_sig:
            skipped += 1
            # Remove any stale architecture.html from a prior run if the
            # arch hasn't changed (idempotency).
            stale = JOURNEY / exp / "architecture.html"
            if stale.exists():
                stale.unlink()
                print(f"[remove-stale] {exp}/architecture.html")
            continue
        out = JOURNEY / exp / "architecture.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(exp, arch, prev_exp_with_arch), encoding="utf-8")
        print(f"[write] {exp}/architecture.html -- {arch['label']}")
        written += 1
        prev_sig = sig
        prev_exp_with_arch = exp
    print(f"\n[done] {written} architecture.html written, {skipped} inherited")


if __name__ == "__main__":
    main()
