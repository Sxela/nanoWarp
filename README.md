# nanoWarp

A Karpathy-style repo for learning visual generation from first principles, then growing that same codepath into practical image and video diffusion.

## Goal

Build the `nanoGPT` of visual generation:
- small enough to read in a day
- simple enough to train on a single GPU
- real enough to extend into image, latent, and video diffusion

## Philosophy

- minimal code over framework sprawl
- explicit tensors over magic abstractions
- one concept per file when possible
- image-first, video-second
- educational, but not toy-only

## The learning arc

### Stage 0 — toy diffusion
Learn the algorithm on tiny 2D datasets.
- forward noising
- epsilon prediction
- timestep embeddings
- reverse sampling
- schedules and ablations

### Stage 1 — tiny image diffusion
Train a small model on 64x64 images.
- tiny UNet or tiny DiT
- MNIST / CIFAR / folder dataset
- unconditional generation first

### Stage 2 — latent image diffusion
Show how diffusion gets practical.
- tiny VAE
- latent diffusion training
- 256x256 path
- optional LoRA finetuning

### Stage 3 — toy video diffusion
Extend the same abstractions to short clips.
- moving-MNIST / synthetic videos
- temporal blocks or temporal attention
- short fixed-length clips

### Stage 4 — practical conditioning
Start bridging toward real product work.
- image-to-video
- video-to-video
- conditioning on reference frames, flow, depth, masks
- segment-wise generation ideas

## Repo shape

```text
nanoWarp/
  README.md
  roadmap.md
  notes/
    why.md
  data/
  src/
    datasets/
    models/
    trainers/
    samplers/
    utils/
  experiments/
    000_toy_2d/
    001_image_64/
    002_latent_image/
    100_video_toy/
  scripts/
    train.py
    sample.py
    prepare_data.py
```

## What makes this worth building

Most vision repos fall into one of three buckets:
- educational but too toy to grow from
- practical but bloated and unreadable
- paper code that barely runs outside the original lab

nanoWarp should be the bridge.

## Visual-first logging

Learning should be visible, not just numeric.

By default, experiments should log locally in a way that makes progress obvious:
- sample images
- progress grids
- loss plots
- simple artifacts you can inspect from the filesystem

Optional integrations like Weights & Biases are welcome, but they should stay optional.

## Recommended MVP

1. toy 2D diffusion
2. 64x64 image diffusion
3. latent image diffusion
4. toy video diffusion

Do **not** start with full text-to-video. That's how the repo becomes sludge.

## Current status

Stage 0 is now live: **toy 2D diffusion**.

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
- intermediate sample PNGs
- `progress_grid.png`
- `loss.png`
- `model.pt`
- `metrics.json`
