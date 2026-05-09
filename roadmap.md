# nanoWarp roadmap

## Core rules

1. Every stage gets one clear training command.
2. Every stage gets one clear sampling command.
3. Every stage gets a short doc explaining the idea.
4. Shared abstractions should stay visible and small.
5. Video should reuse image concepts instead of splitting into a second framework.
6. Local visual logging is mandatory by default; hosted logging is optional.

## MVP phases

### Phase A — toy diffusion fundamentals
- `experiments/000_toy_2d`
- DDPM on 2D point clouds
- visualize forward corruption and reverse denoising
- tiny MLP with timestep embedding

### Phase B — pixel-space img2img
- small conditional diffusion UNet
- source photo -> target comic/stylized image
- multiscale source conditioning
- one bottleneck attention block with SDPA
- visual comparison logging by default

### Phase C — practical img2img training
- paired or pseudo-paired dataset pipeline
- EMA, better progress artifacts, optional hosted logging
- study structure preservation vs style strength

### Phase D — minimal V2V
- extend img2img to short clips
- previous-frame conditioning first
- keep temporal machinery as small as possible

## Stretch goals
- class conditioning
- text conditioning
- LoRA finetuning
- image conditioning
- segment-wise video generation
- V2V conditioning

## Non-goals for v1
- SOTA benchmarks
- giant multimodal stack
- distributed training labyrinth
- 2K video generation
- polished UI before working code

## First implementation order

1. scaffolding
2. toy 2D diffusion end-to-end
3. shared training utilities
4. 64x64 image diffusion
5. sample logging
6. toy video diffusion
