"""
Training Data Generation for Artifact Detection.

Builds the training dataset following the paper's protocol (Section IV.C):

For each of 42 training images:
  1. Generate two LR images: zero motion (g_c) and arbitrary motion (g_i)
  2. Run Phase 1 SR from both → SR image
  3. Compute bicubic upscaling of g_c
  4. Run Canny edge detection on bicubic LR image
  5. Sample edge pixels on a 15-pixel grid
  6. Extract 50-dim features at those positions

Error-free data: use correct (true) registration
Erroneous data:
  - First 31 images: spatially variant error, uniform in [0.5, 5] px
  - Remaining 11: spatially invariant error, uniform in [0.4, 1.5] px

Outlier simulation:
  - 5 SR images: change DC level (add/subtract a constant)
  - 5 feature vectors: use a DIFFERENT bicubic-upscaled LR image

Labels:
  y = 1 for artifact-free pixels (correct registration)
  y = 0 for artifact pixels (erroneous registration / outliers)
"""

import numpy as np
import os
from src.observation_model import (
    apply_shift, bicubic_upsample,
    downsample_nonoverlapping_avg,
)
from src.sr_solver import sr_phase1
from src.feature_extraction import (
    extract_features_at_positions,
    get_canny_edge_positions,
)
from src.utils import load_image, ensure_dir


def _generate_lr_pair(hr_image, scale, rng):
    """Generate a pair of LR images: one reference (zero motion), one shifted.

    Args:
        hr_image: (H, W) float32 grayscale HR image.
        scale: Downsampling factor.
        rng: numpy RandomState for reproducibility.

    Returns:
        lr_ref: (h, w) reference LR image (zero motion).
        lr_shifted: (h, w) LR image with arbitrary motion.
        true_shift: (dy, dx) true shift in LR domain.
    """
    H, W = hr_image.shape
    h, w = H // scale, W // scale

    # Reference LR: no shift
    lr_ref = downsample_nonoverlapping_avg(hr_image, scale)

    # Shifted LR: random sub-pixel shift
    dy = rng.uniform(-1.5, 1.5)
    dx = rng.uniform(-1.5, 1.5)
    shifted_hr = apply_shift(hr_image, dy, dx)
    lr_shifted = downsample_nonoverlapping_avg(shifted_hr, scale)

    true_shift_lr = np.array([dy / scale, dx / scale], dtype=np.float32)
    return lr_ref, lr_shifted, true_shift_lr


def _add_spatially_variant_error(true_shift, H, W, error_range, rng):
    """Add spatially variant registration error.

    Each pixel gets a different error, uniform in [error_range[0], error_range[1]].
    Returns an (H, W, 2) error field and the noisy shift is true_shift + error.

    For Phase 1 SR, we use a single shift per frame (not per-pixel), so we
    sample one representative error from the range.
    """
    error_mag = rng.uniform(error_range[0], error_range[1])
    error_angle = rng.uniform(0, 2 * np.pi)
    error = np.array([
        error_mag * np.sin(error_angle),
        error_mag * np.cos(error_angle),
    ], dtype=np.float32)
    return true_shift + error


def _add_spatially_invariant_error(true_shift, error_range, rng):
    """Add spatially invariant registration error."""
    error_mag = rng.uniform(error_range[0], error_range[1])
    error_angle = rng.uniform(0, 2 * np.pi)
    error = np.array([
        error_mag * np.sin(error_angle),
        error_mag * np.cos(error_angle),
    ], dtype=np.float32)
    return true_shift + error


def build_training_data(image_paths, scale=2, mu=0.01, phase1_iters=3,
                        seed=42):
    """Build the full training dataset for the artifact detector.

    Args:
        image_paths: List of paths to grayscale training images (42 expected).
        scale: Downsampling factor.
        mu: Step size for Phase 1 SR.
        phase1_iters: Number of Phase 1 iterations (paper: 3).
        seed: Random seed for reproducibility.

    Returns:
        X_clean:  (N1, 50) features from error-free SR.
        X_error:  (N2, 50) features from erroneous SR.
        y_clean:  (N1,) all ones (artifact-free).
        y_error:  (N2,) all zeros (artifact).
        image_ids_clean: (N1,) source image index per sample.
        image_ids_error: (N2,) source image index per sample.
    """
    rng = np.random.RandomState(seed)
    n_images = len(image_paths)

    # Paper's split: first 31 with SV error, remaining 11 with SI error
    n_sv = min(31, n_images)

    all_X_clean = []
    all_X_error = []
    all_ids_clean = []
    all_ids_error = []

    for img_idx, path in enumerate(image_paths):
        print(f"  Processing image {img_idx+1}/{n_images}: {os.path.basename(path)}")

        hr = load_image(path, color='gray')
        H, W = hr.shape

        # Ensure divisible by scale
        H = (H // scale) * scale
        W = (W // scale) * scale
        hr = hr[:H, :W]

        # Generate LR pair
        lr_ref, lr_shifted, true_shift = _generate_lr_pair(hr, scale, rng)

        # Bicubic upscale of reference LR
        bic_up = bicubic_upsample(lr_ref, scale)

        # --- Error-free SR (correct registration) ---
        shifts_correct = np.array([[0.0, 0.0], true_shift], dtype=np.float32)
        sr_clean = sr_phase1(
            [lr_ref, lr_shifted], shifts_correct, scale,
            mu=mu, num_iterations=phase1_iters, lambda_reg=0.0,
        )

        # --- Erroneous SR (wrong registration) ---
        if img_idx < n_sv:
            noisy_shift = _add_spatially_variant_error(
                true_shift, H, W, [0.5, 5.0], rng
            )
        else:
            noisy_shift = _add_spatially_invariant_error(
                true_shift, [0.4, 1.5], rng
            )

        shifts_noisy = np.array([[0.0, 0.0], noisy_shift], dtype=np.float32)
        sr_error = sr_phase1(
            [lr_ref, lr_shifted], shifts_noisy, scale,
            mu=mu, num_iterations=phase1_iters, lambda_reg=0.0,
        )

        # --- Outlier simulation ---
        # DC level change for 5 images (apply to erroneous SR)
        if img_idx < 5:
            dc_change = rng.uniform(-0.1, 0.1)
            sr_error = np.clip(sr_error + dc_change, 0, 1)

        # Wrong bicubic for 5 images (use a shifted version)
        if 5 <= img_idx < 10:
            wrong_bic_up = bicubic_upsample(lr_shifted, scale)
        else:
            wrong_bic_up = bic_up

        # --- Canny edge detection + 15-pixel grid sampling ---
        positions = get_canny_edge_positions(bic_up, grid_spacing=15)

        if len(positions) == 0:
            print(f"    Warning: no edge pixels found for image {img_idx}")
            continue

        # --- Feature extraction ---
        # Clean features (error-free SR, correct bicubic)
        feat_clean = extract_features_at_positions(
            sr_clean, bic_up, positions
        )
        all_X_clean.append(feat_clean)
        all_ids_clean.append(np.full(len(feat_clean), img_idx, dtype=np.int32))

        # Error features (erroneous SR, possibly wrong bicubic)
        feat_error = extract_features_at_positions(
            sr_error, wrong_bic_up, positions
        )
        all_X_error.append(feat_error)
        all_ids_error.append(np.full(len(feat_error), img_idx, dtype=np.int32))

    # Concatenate
    X_clean = np.concatenate(all_X_clean, axis=0)
    X_error = np.concatenate(all_X_error, axis=0)
    y_clean = np.ones(len(X_clean), dtype=np.int32)
    y_error = np.zeros(len(X_error), dtype=np.int32)
    ids_clean = np.concatenate(all_ids_clean)
    ids_error = np.concatenate(all_ids_error)

    print(f"\n  Training data summary:")
    print(f"    Clean (y=1): {len(X_clean)} samples")
    print(f"    Error (y=0): {len(X_error)} samples")
    print(f"    Total:       {len(X_clean) + len(X_error)} samples")

    return X_clean, X_error, y_clean, y_error, ids_clean, ids_error
