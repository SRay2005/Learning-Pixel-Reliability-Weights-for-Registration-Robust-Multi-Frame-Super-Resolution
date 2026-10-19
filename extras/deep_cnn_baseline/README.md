# Archived: Deep CNN Baseline (not the paper's method)

This folder contains the original deep-learning SR implementation that was built
before switching to the paper's classical method. It uses:

- `RobustSRNetwork`: CNN + ResBlocks + temporal attention fusion
- `RobustSRLoss`: Charbonnier + Edge + optional VGG perceptual loss
- Adam optimiser with cosine annealing

## Known bugs (not fixed here)
1. SSIM > 1.0 in validation (accumulation bug in train.py)
2. Train/val data leakage (same images, random patches)
3. No fixed random seeds
4. Parabolic interpolation bug in registration.py

## Status
Archived 2026-10-19. Not part of the core classifier comparison.
May be resurrected later as an optional "modern baseline" if bugs are fixed.
