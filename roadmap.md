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

### Phase B — tiny image diffusion
- `experiments/001_image_64`
- small UNet baseline
- CIFAR-10 or folder dataset
- image samples + EMA checkpoint + simple metrics

### Phase C — latent image diffusion
- `experiments/002_latent_image`
- tiny VAE encoder/decoder or pluggable pretrained one
- train diffusion in latent space
- explain why latent beats pixel-space for scaling

### Phase D — toy video diffusion
- `experiments/100_video_toy`
- moving-MNIST or bouncing shapes
- temporal conv or temporal attention
- short low-res clips only

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
