# ViT Temporal Gradient Rank Analysis

This repository provides a modular PyTorch research codebase to measure temporal gradient effective rank and intrinsic dimensionality for Vision Transformer training dynamics. The implementation uses the Gram matrix trick and supports streaming approximations to avoid large-memory $O(P^2)$ operations.

## Key features

- ViT model support via torchvision and timm, with optional LoRA adapters.
- Gradient collection with subsampling, random projection, mixed precision storage, and CPU offload.
- Temporal Gram matrix construction and streaming accumulation.
- Spectral analysis: effective rank, participation ratio, stable rank, spectral entropy, energy thresholds, power-law fits.
- Layerwise analysis for attention and MLP blocks.
- Streaming PCA: Oja, incremental PCA, and Frequent Directions.
- Isotropy analysis for weights and covariance.
- Publication-ready plots.

## Quick start

1. Install dependencies (see requirements).
2. Prepare a dataset path.
3. Run training with analysis:

```bash
bash run.sh
```

## Repository layout

- models: ViT factories and LoRA.
- training: data loaders, optimizers, training loop.
- analysis: gradient collection, Gram matrix, spectral metrics.
- visualization: plotting utilities.
- utils: configuration, seeding, logging, distributed helpers.
- configs: YAML experiment configs.

## Notes

- The code saves intermediate projections, Gram matrices, eigenvalues, and plots in the output directory.
- For large models, prefer random projection and CPU offload for gradients.
