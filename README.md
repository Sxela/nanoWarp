# nanoWarp

A living journal for building **small, fast image-to-image and video-to-video models**.

Not a giant foundation-model playground.
Not a paper zoo.
A focused repo for figuring out how far we can push **structure-preserving visual transformation** with compact models and visual-first tooling.

---

## Current thesis

The interesting problem for us is **not** random-noise-to-world generation.
It is:

- **photo -> comics**
- **image -> stylized image**
- **video -> stylized video**
- strong preservation of layout / pose / identity / timing
- small enough to train and iterate on quickly

The core bet:

> these tasks should allow much smaller trainable systems than fully generative text-to-image / text-to-video, because the input already gives us structure.

That does **not** mean zero pretrained priors.
It means the trainable part can be much smaller and more targeted.

---

## Why this repo exists

There are great small repos for LLM training.
Vision still feels split between:

- educational notebooks that stop before things get useful
- giant frameworks that hide the core ideas
- brittle paper repos that barely run outside the original lab

nanoWarp should be the middle path:

- small enough to read
- visual enough to teach
- practical enough to matter
- opinionated enough to avoid sludge

---

## Design rules

1. **Visual-first logging is mandatory by default.**
   People should see progress, not just loss numbers.
2. **Local artifacts first, hosted logging optional.**
   PNGs, grids, plots, maybe GIFs. W&B later if useful.
3. **Small trainable systems over giant infrastructure.**
4. **Image-first, then video.** If single-frame translation is weak, video will just flicker more beautifully.
5. **This repo is a journal, not just a code dump.** Important decisions and pivots should be reflected here.

---

## Current direction

### Old idea
Start from diffusion fundamentals and grow toward image/video generation.

### Updated idea
Keep toy diffusion as foundation, but make the repo center of gravity:

- **img2img** first
- then **v2v**
- especially **photo -> comics** / stylization tasks

Why this pivot makes sense:

- narrower target distribution
- easier than full generative modeling
- more relevant to Mago-style product work
- better fit for compact architectures
- easier to evaluate visually

---

## Architecture direction we're interested in

Not starting from a huge 1.3B+ model unless we're only training tiny deltas.

More interesting starting points:

- compact conditional UNet
- ResNet-initialized encoder + trainable decoder/UNet-ish body
- latent conditional diffusion
- optional small attention blocks instead of full heavy attention everywhere
- later: temporal conditioning for V2V

In spirit, this is closer to evolving a **fastai-era small image translation model** into a more stable diffusion-based system than to building a giant foundation model clone.

---

## Why diffusion here

Previous-style image translation stacks often needed awkward balancing:

- frozen pretrained encoder
- trainable UNet/decoder
- perceptual loss
- GAN loss
- pretrained discriminator
- careful pretraining so the adversarial stage doesn't explode

Diffusion looks promising here because it gives a cleaner optimization target for stylized translation.
Still not trivial, but likely simpler and more stable than juggling GAN training just right.

---

## Primary task candidate

## photo -> comics

This is a strong first target because it reduces to a smaller and more stylized output distribution.
The model mostly needs to learn:

- what structure to preserve
- what texture/detail to simplify
- how to replace realism with linework, stylized shading, and shape-driven color

That feels much more compatible with small, fast models than unconstrained generation.

---

## Roadmap (current)

### Stage 0 — toy diffusion foundation
Purpose: learn and show the denoising mechanics clearly.

Status: **done / working**

Implemented:
- toy 2D diffusion training
- toy 2D sampling
- visual-first local logging

### Stage 1 — minimal img2img
Goal:
- pick a compact conditional architecture
- train on a paired or pseudo-paired image translation task
- probably photo -> comics or adjacent stylization

Questions:
- pixel-space or latent-space first?
- how small can the model be before quality falls apart?
- what conditioning interface is simplest and strongest?

### Stage 2 — better visual training UX
Goal:
- richer progress artifacts
- comparison grids
- maybe GIF/MP4 progression
- optional hosted logging for pros

### Stage 3 — minimal V2V
Goal:
- extend img2img to short clips
- keep temporal coherence with the smallest possible extra machinery
- likely previous-frame conditioning first, then more advanced temporal tricks

---

## Stage 0 details

Stage 0 is live: **toy 2D diffusion**.

### Train

```bash
python3 scripts/train.py toy2d --dataset moons --steps 2000 --sample-every 200
```

### Sample from checkpoint

```bash
python3 scripts/sample.py toy2d --checkpoint out/toy2d/model.pt --output out/toy2d/samples_latest.png
```

### Supported toy datasets

- `moons`
- `spiral`
- `circle`

### Outputs

Training writes to `out/toy2d/` by default:
- `dataset.png`
- `samples_step_*.png`
- `progress_grid.png`
- `loss.png`
- `model.pt`
- `metrics.json`

---

## Journal notes

### 2026-05-09
- Repo name set to **nanoWarp**.
- Stage 0 implemented as a working toy diffusion baseline.
- Important repo principle established: **progress must be visible locally by default**.
- Direction pivoted away from generic unconditional image/video generation toward:
  - **img2img**
  - **v2v**
  - especially **photo -> comics**
- Reason for pivot: more aligned with what we actually care about, smaller trainable systems, and more practical for stylization/video work.

---

## Repo shape

```text
nanoWarp/
  README.md
  roadmap.md
  notes/
    why.md
  data/
  src/
  experiments/
    000_toy_2d/
  scripts/
    train.py
    sample.py
    prepare_data.py
```

This shape will change as the real img2img path crystallizes.

---

## Setup from scratch

```bash
./install.sh
source .venv/bin/activate
```

Main dependencies currently live in `requirements.txt`.

## Validation

For img2img, we want validation to be visible too, not just one scalar.

Current validation flow:

```bash
python3 scripts/sample.py img2img-v1-val <val_data_root> --checkpoint <checkpoint_path> --outdir out/img2img_v1_val
```

This writes:
- `val_metrics.json`
- `val_panel_*.png`

The validation panels show:
- source
- target
- noisy target
- reconstructed `x0_hat`
