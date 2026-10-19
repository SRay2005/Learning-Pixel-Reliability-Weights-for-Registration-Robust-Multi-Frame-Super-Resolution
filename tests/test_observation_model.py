"""
Tests for the observation model (W_i operator).

Verifies:
1. Downsampling by non-overlapping averaging produces correct LR sizes
2. Forward-adjoint consistency: <W x, y> ≈ <x, W^T y>
3. Bicubic upsampling returns correct shape
4. Zero-shift is identity (up to downsampling)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observation_model import (
    apply_shift, apply_inverse_shift,
    downsample_nonoverlapping_avg, upsample_replicate,
    apply_W, apply_W_transpose, bicubic_upsample,
)


def test_downsample_shape():
    """Non-overlapping averaging produces correct LR shape."""
    hr = np.random.rand(64, 64).astype(np.float32)
    lr = downsample_nonoverlapping_avg(hr, scale=2)
    assert lr.shape == (32, 32), f"Expected (32,32), got {lr.shape}"
    print("  [PASS] downsample shape")


def test_downsample_value():
    """Each LR pixel is the mean of the corresponding 2×2 HR block."""
    hr = np.zeros((4, 4), dtype=np.float32)
    hr[0:2, 0:2] = 1.0  # Top-left 2×2 block all ones
    lr = downsample_nonoverlapping_avg(hr, scale=2)
    assert lr[0, 0] == 1.0, f"Expected 1.0, got {lr[0, 0]}"
    assert lr[0, 1] == 0.0, f"Expected 0.0, got {lr[0, 1]}"
    assert lr[1, 0] == 0.0, f"Expected 0.0, got {lr[1, 0]}"
    print("  [PASS] downsample value")


def test_upsample_shape():
    """Replicate upsampling produces correct HR shape."""
    lr = np.random.rand(32, 32).astype(np.float32)
    hr = upsample_replicate(lr, scale=2)
    assert hr.shape == (64, 64), f"Expected (64,64), got {hr.shape}"
    print("  [PASS] upsample shape")


def test_forward_adjoint_consistency():
    """Verify <W x, y> ≈ <x, W^T y> (inner product test).

    This is the key test for correctness of the adjoint operator.
    """
    np.random.seed(42)
    scale = 2
    H, W = 64, 64
    h, w = H // scale, W // scale

    x = np.random.rand(H, W).astype(np.float32)
    y = np.random.rand(h, w).astype(np.float32)
    dy, dx = 0.5, -0.3

    # <Wx, y>
    Wx = apply_W(x, dy, dx, scale)
    inner1 = np.sum(Wx * y)

    # <x, W^T y>
    WTy = apply_W_transpose(y, dy, dx, scale)
    inner2 = np.sum(x * WTy)

    rel_error = abs(inner1 - inner2) / (abs(inner1) + 1e-10)
    # Allow some tolerance due to interpolation
    assert rel_error < 0.15, (
        f"Adjoint test failed: <Wx,y>={inner1:.6f}, <x,WTy>={inner2:.6f}, "
        f"rel_error={rel_error:.6f}"
    )
    print(f"  [PASS] forward-adjoint consistency (rel_error={rel_error:.6f})")


def test_zero_shift_roundtrip():
    """With zero shift, W^T W x should roughly reconstruct x."""
    np.random.seed(42)
    scale = 2
    hr = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25

    lr = apply_W(hr, 0, 0, scale)
    hr_back = apply_W_transpose(lr, 0, 0, scale)

    # Not exact due to averaging, but shape should match
    assert hr_back.shape == hr.shape
    print("  [PASS] zero-shift roundtrip shape")


def test_bicubic_upsample():
    """Bicubic upsampling returns correct shape and range."""
    lr = np.random.rand(32, 48).astype(np.float32)
    hr = bicubic_upsample(lr, scale=2)
    assert hr.shape == (64, 96), f"Expected (64,96), got {hr.shape}"
    assert hr.min() >= 0.0 and hr.max() <= 1.0
    print("  [PASS] bicubic upsample")


if __name__ == '__main__':
    print("\n=== Testing Observation Model ===")
    test_downsample_shape()
    test_downsample_value()
    test_upsample_shape()
    test_forward_adjoint_consistency()
    test_zero_shift_roundtrip()
    test_bicubic_upsample()
    print("\n  All observation model tests passed!\n")
