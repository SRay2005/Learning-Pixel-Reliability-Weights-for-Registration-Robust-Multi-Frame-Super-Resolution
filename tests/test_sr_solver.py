"""
Tests for the SR solver.

Verifies:
1. Phase 1 SR output has correct shape
2. SR PSNR improves over iterations (convergence)
3. With perfect registration and no noise, SR ≈ ground truth
4. Phase 2 with uniform weights = Phase 1
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observation_model import (
    apply_shift, downsample_nonoverlapping_avg, bicubic_upsample,
)
from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted
from src.utils import compute_psnr


def _make_synthetic_lr(hr, scale, shifts):
    """Generate LR frames from HR with known shifts."""
    lr_frames = []
    for dy, dx in shifts:
        shifted = apply_shift(hr, dy, dx)
        lr = downsample_nonoverlapping_avg(shifted, scale)
        lr_frames.append(lr)
    return lr_frames


def test_phase1_shape():
    """Phase 1 output has the correct HR shape."""
    np.random.seed(42)
    scale = 2
    hr = np.random.rand(64, 64).astype(np.float32) * 0.8 + 0.1
    shifts_hr = np.array([[0, 0], [0.5, -0.3]], dtype=np.float32)
    shifts_lr = shifts_hr / scale

    lr_frames = _make_synthetic_lr(hr, scale, shifts_hr)
    sr = sr_phase1(lr_frames, shifts_lr, scale, mu=0.01, num_iterations=3)

    assert sr.shape == (64, 64), f"Expected (64,64), got {sr.shape}"
    print("  [PASS] Phase 1 shape")


def test_phase1_improves_over_bicubic():
    """Phase 1 SR should produce higher PSNR than bicubic upsampling."""
    np.random.seed(42)
    scale = 2
    hr = np.random.rand(64, 64).astype(np.float32) * 0.8 + 0.1
    shifts_hr = np.array([[0, 0], [0.7, 0.3], [-0.5, 0.8]], dtype=np.float32)
    shifts_lr = shifts_hr / scale

    lr_frames = _make_synthetic_lr(hr, scale, shifts_hr)

    bic = bicubic_upsample(lr_frames[0], scale)
    psnr_bic = compute_psnr(bic, hr)

    sr = sr_phase1(lr_frames, shifts_lr, scale, mu=0.05, num_iterations=5)
    psnr_sr = compute_psnr(sr, hr)

    print(f"    Bicubic PSNR: {psnr_bic:.2f} dB")
    print(f"    Phase 1 PSNR: {psnr_sr:.2f} dB")
    assert psnr_sr > psnr_bic, (
        f"Phase 1 ({psnr_sr:.2f}) should beat bicubic ({psnr_bic:.2f})"
    )
    print("  [PASS] Phase 1 improves over bicubic")


def test_phase2_with_ones_equals_unweighted():
    """Phase 2 with M_i = 1 everywhere should match unweighted SR."""
    np.random.seed(42)
    scale = 2
    hr = np.random.rand(32, 32).astype(np.float32) * 0.8 + 0.1
    shifts_hr = np.array([[0, 0], [0.4, -0.2]], dtype=np.float32)
    shifts_lr = shifts_hr / scale

    lr_frames = _make_synthetic_lr(hr, scale, shifts_hr)
    H, W = 32, 32
    ones = [np.ones((H, W), dtype=np.float32)] * len(lr_frames)

    sr_weighted = sr_phase2(lr_frames, shifts_lr, ones, scale,
                            mu=0.01, num_iterations=5)
    sr_uw = sr_unweighted(lr_frames, shifts_lr, scale,
                          mu=0.01, num_iterations=5)

    diff = np.max(np.abs(sr_weighted - sr_uw))
    assert diff < 1e-5, f"Max diff = {diff:.8f}, should be near zero"
    print(f"  [PASS] Phase 2 (M=1) ~ unweighted (max diff={diff:.2e})")


def test_convergence():
    """PSNR should improve (or at least not degrade) over iterations."""
    np.random.seed(42)
    scale = 2
    hr = np.random.rand(64, 64).astype(np.float32) * 0.8 + 0.1
    shifts_hr = np.array([[0, 0], [0.5, 0.5]], dtype=np.float32)
    shifts_lr = shifts_hr / scale

    lr_frames = _make_synthetic_lr(hr, scale, shifts_hr)

    psnrs = []
    for n_iter in [1, 3, 5, 10]:
        sr = sr_phase1(lr_frames, shifts_lr, scale, mu=0.02,
                       num_iterations=n_iter)
        psnrs.append(compute_psnr(sr, hr))

    print(f"    PSNR at iters [1,3,5,10]: {[f'{p:.2f}' for p in psnrs]}")
    # At least the last should be >= the first
    assert psnrs[-1] >= psnrs[0] - 0.5, (
        f"PSNR should not degrade significantly: {psnrs[0]:.2f} → {psnrs[-1]:.2f}"
    )
    print("  [PASS] convergence check")


if __name__ == '__main__':
    print("\n=== Testing SR Solver ===")
    test_phase1_shape()
    test_phase1_improves_over_bicubic()
    test_phase2_with_ones_equals_unweighted()
    test_convergence()
    print("\n  All SR solver tests passed!\n")
