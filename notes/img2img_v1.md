# img2img v1 — photo -> comics architecture

## Goal

Build a **small pixel-space conditional diffusion model** for paired or pseudo-paired
photo -> comics translation.

The constraint is important:
- total system should stay relatively small
- no giant VAE hiding half the complexity
- training progress should be visible locally
- architecture should extend naturally to V2V later

## Core training setup

For each pair `(source_photo, target_comic)`:

1. sample timestep `t`
2. sample Gaussian noise `eps`
3. corrupt the target comic to obtain `y_t`
4. run the model conditioned on the source photo
5. predict the target noise `eps`

So the model learns:

`model(source, noisy_target, t) -> predicted_noise`

## Architecture

### Source branch
A small **ImageNet-pretrained ResNet18-compatible encoder** extracts multiscale structure features from the input photo.
These features act as the conditioning signal.

Default plan:
- use pretrained ResNet18 weights
- freeze `stem` and `layer1` first
- leave `layer2-4` trainable by default

### Diffusion branch
A compact UNet denoises the noisy comic target.
The diffusion branch gets timestep conditioning in every residual block.

### Fusion
At each matching scale, source features are projected and fused into the diffusion path.
Default fusion for v1:
- concat features
- 3x3 conv

### Attention
Use **one bottleneck attention block only**.
Implementation should use **PyTorch SDPA**, not brute-force attention.

## Why pixel-space first

We are explicitly avoiding a large VAE-first setup because:
- some VAEs are larger than the whole model we actually want
- pixel-space keeps the system honest
- easier to inspect failure modes visually
- better continuity with the old fastai-style image translation setup

## Proposed first resolution

- start at **128x128**
- move to **256x256** only after the architecture is stable

## Open questions

- should source also be concatenated with noisy target at the input stem, in addition to multiscale conditioning?
- should early source encoder layers be frozen at first?
- do we want plain epsilon prediction only, or add an x0 reconstruction preview/logging path?
- paired data first, or pseudo-paired synthetic comic transforms first?

## Intended next step

Turn this note into:
1. a minimal model skeleton
2. a training loop stub
3. visual-first logging for paired translation experiments
