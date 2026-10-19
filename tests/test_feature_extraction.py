"""
Tests for feature extraction.

Verifies:
1. Feature vector has correct dimensionality (50)
2. Gaussian window is symmetric and peaks at centre
3. Canny + grid sampling produces valid positions
4. Features are finite and not all-zero
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import (
    extract_pixel_features,
    extract_features_at_positions,
    get_canny_edge_positions,
    _gaussian_window_2d,
    _half_amplitude_spectrum,
)


def test_gaussian_window():
    """Gaussian window is correct shape and symmetric."""
    w = _gaussian_window_2d(7, 0.4)
    assert w.shape == (7, 7)
    # Centre should be the maximum
    assert w[3, 3] == w.max()
    # Symmetric
    assert abs(w[2, 3] - w[4, 3]) < 1e-6
    assert abs(w[3, 2] - w[3, 4]) < 1e-6
    print("  [PASS] Gaussian window shape and symmetry")


def test_half_spectrum_length():
    """Half amplitude spectrum has 25 coefficients for a 7×7 block."""
    block = np.random.rand(7, 7).astype(np.float32)
    spectrum = _half_amplitude_spectrum(block)
    assert len(spectrum) == 25, f"Expected 25, got {len(spectrum)}"
    assert np.all(np.isfinite(spectrum))
    print("  [PASS] half spectrum length = 25")


def test_feature_vector_dim():
    """Feature vector per pixel is 50-dimensional."""
    np.random.seed(42)
    sr = np.random.rand(64, 64).astype(np.float32)
    bic = np.random.rand(64, 64).astype(np.float32)

    feat = extract_pixel_features(sr, bic, row=32, col=32)
    assert feat.shape == (50,), f"Expected (50,), got {feat.shape}"
    assert np.all(np.isfinite(feat))
    print("  [PASS] feature vector dimensionality = 50")


def test_batch_features():
    """Batch extraction matches single-pixel extraction."""
    np.random.seed(42)
    sr = np.random.rand(64, 64).astype(np.float32)
    bic = np.random.rand(64, 64).astype(np.float32)

    positions = np.array([[10, 20], [30, 40], [50, 50]])
    batch = extract_features_at_positions(sr, bic, positions)
    assert batch.shape == (3, 50)

    single = extract_pixel_features(sr, bic, 10, 20)
    diff = np.max(np.abs(batch[0] - single))
    assert diff < 1e-5, f"Batch vs single mismatch: max diff = {diff}"
    print("  [PASS] batch features match single-pixel")


def test_canny_positions():
    """Canny + grid sampling produces valid positions within image bounds."""
    np.random.seed(42)
    # Create an image with clear edges
    img = np.zeros((128, 128), dtype=np.float32)
    img[30:90, 30:90] = 1.0  # Square → edges

    positions = get_canny_edge_positions(img, grid_spacing=15)
    assert len(positions) > 0, "Expected some edge pixels"

    # All positions within bounds
    for r, c in positions:
        assert 0 <= r < 128 and 0 <= c < 128
    print(f"  [PASS] Canny positions: {len(positions)} edge pixels found")


def test_features_not_all_zero():
    """Features from a real-looking image should not be trivially zero."""
    np.random.seed(42)
    sr = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25
    bic = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25

    feat = extract_pixel_features(sr, bic, 32, 32)
    assert np.any(feat != 0), "Features should not be all zero"
    print("  [PASS] features are non-trivial")


if __name__ == '__main__':
    print("\n=== Testing Feature Extraction ===")
    test_gaussian_window()
    test_half_spectrum_length()
    test_feature_vector_dim()
    test_batch_features()
    test_canny_positions()
    test_features_not_all_zero()
    print("\n  All feature extraction tests passed!\n")
