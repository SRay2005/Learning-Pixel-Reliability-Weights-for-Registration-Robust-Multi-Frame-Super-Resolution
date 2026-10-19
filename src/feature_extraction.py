"""
Feature Extraction for Artifact Detection.

Extracts a 50-dimensional feature vector per HR pixel for the artifact
detector, following Section IV of the paper:

Part 1 (25 coeffs):
    - Take 7×7 block centred on pixel (r,c) from the Phase-1 SR image
    - Subtract 7×7 block from bicubic-upscaled current LR image
    - Multiply by Gaussian window (σ = 0.4)
    - 2D DFT, take amplitude spectrum
    - Exploit conjugate symmetry → keep half, raster-scan, first 25 coefficients

Part 2 (25 coeffs):
    - Take 7×7 block from bicubic-upscaled LR image, filtered with 3×3 Laplacian
    - Multiply by Gaussian window (σ = 0.4)
    - 2D DFT, amplitude spectrum, half, 25 coefficients

Concatenation → 50-dimensional raw feature vector per pixel.

Paper reference: Section IV.A.
"""

import numpy as np
import cv2


def _gaussian_window_2d(size, sigma):
    """Create a 2D Gaussian window.

    Args:
        size: Window size (assumes square).
        sigma: Standard deviation.

    Returns:
        (size, size) float32 array, normalised so max=1.
    """
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    window = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return window  # Not normalised to sum=1; max=1 at centre


def _half_amplitude_spectrum(block):
    """Compute 2D DFT amplitude spectrum and return half (conjugate symmetry).

    For a 7×7 block, the full DFT has 49 coefficients.  Due to conjugate
    symmetry of the DFT of a real signal, we keep roughly half: the first
    25 coefficients when raster-scanning the amplitude spectrum.

    Args:
        block: (size, size) float32.

    Returns:
        1D array of the first 25 amplitude coefficients.
    """
    spectrum = np.fft.fft2(block)
    amplitude = np.abs(spectrum)
    # Raster-scan the amplitude spectrum (row-major order)
    flat = amplitude.ravel()
    # Keep first 25 coefficients (half of 49, plus the DC)
    return flat[:25].astype(np.float32)


def _laplacian_filter(image):
    """Apply 3×3 Laplacian filter to a grayscale image.

    Kernel:
        [0  1  0]
        [1 -4  1]
        [0  1  0]
    """
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def extract_pixel_features(sr_image, bicubic_lr_up, row, col,
                            block_size=7, sigma=0.4):
    """Extract the 50-dimensional feature vector at a single HR pixel.

    Args:
        sr_image:      (H, W) Phase-1 SR image.
        bicubic_lr_up: (H, W) bicubic-upscaled current LR image.
        row, col:      Pixel coordinates in the HR image.
        block_size:    Block size (default 7, as in paper).
        sigma:         Gaussian window sigma (default 0.4, as in paper).

    Returns:
        (50,) float32 feature vector.
    """
    half = block_size // 2
    H, W = sr_image.shape[:2]

    # Boundary-safe block extraction with reflect padding
    # Pad images so we can always extract a full block
    padded_sr = np.pad(sr_image, half, mode='reflect')
    padded_bic = np.pad(bicubic_lr_up, half, mode='reflect')

    # Adjust coordinates for padding
    r = row + half
    c = col + half

    # Extract blocks
    sr_block = padded_sr[r - half:r + half + 1, c - half:c + half + 1].copy()
    bic_block = padded_bic[r - half:r + half + 1, c - half:c + half + 1].copy()

    # Gaussian window
    gauss_win = _gaussian_window_2d(block_size, sigma)

    # --- Part 1: difference block ---
    diff_block = (sr_block - bic_block) * gauss_win
    part1 = _half_amplitude_spectrum(diff_block)

    # --- Part 2: Laplacian of bicubic, windowed ---
    # Apply Laplacian to the full bicubic image first, then extract block
    padded_bic_lap = np.pad(_laplacian_filter(bicubic_lr_up), half, mode='reflect')
    bic_lap_block = padded_bic_lap[r - half:r + half + 1, c - half:c + half + 1].copy()
    bic_lap_windowed = bic_lap_block * gauss_win
    part2 = _half_amplitude_spectrum(bic_lap_windowed)

    return np.concatenate([part1, part2])


def extract_features_at_positions(sr_image, bicubic_lr_up, positions,
                                   block_size=7, sigma=0.4):
    """Extract features at multiple pixel positions.

    Args:
        sr_image:      (H, W) Phase-1 SR image.
        bicubic_lr_up: (H, W) bicubic-upscaled current LR image.
        positions:     (N, 2) array of (row, col) positions.
        block_size:    Block size.
        sigma:         Gaussian window sigma.

    Returns:
        (N, 50) float32 feature matrix.
    """
    n = len(positions)
    features = np.zeros((n, 50), dtype=np.float32)

    # Pre-compute Laplacian of bicubic image once
    bic_laplacian = _laplacian_filter(bicubic_lr_up)
    half = block_size // 2
    gauss_win = _gaussian_window_2d(block_size, sigma)

    # Pad all images once
    padded_sr = np.pad(sr_image, half, mode='reflect')
    padded_bic = np.pad(bicubic_lr_up, half, mode='reflect')
    padded_bic_lap = np.pad(bic_laplacian, half, mode='reflect')

    for idx, (row, col) in enumerate(positions):
        r = row + half
        c = col + half

        # Part 1: SR - bicubic difference
        sr_block = padded_sr[r - half:r + half + 1, c - half:c + half + 1]
        bic_block = padded_bic[r - half:r + half + 1, c - half:c + half + 1]
        diff_block = (sr_block - bic_block) * gauss_win
        part1 = _half_amplitude_spectrum(diff_block)

        # Part 2: Laplacian of bicubic
        bic_lap_block = padded_bic_lap[r - half:r + half + 1, c - half:c + half + 1]
        bic_lap_windowed = bic_lap_block * gauss_win
        part2 = _half_amplitude_spectrum(bic_lap_windowed)

        features[idx] = np.concatenate([part1, part2])

    return features


def get_canny_edge_positions(image, grid_spacing=15):
    """Detect edges using Canny and sample on a regular grid.

    Paper: "Edge pixels from Canny on the bicubic LR image,
    kept only on a 15-pixel grid."

    Args:
        image: (H, W) float32 grayscale image in [0, 1].
        grid_spacing: Spacing of the sampling grid (default 15).

    Returns:
        (N, 2) int array of (row, col) positions of selected edge pixels.
    """
    # Canny expects uint8
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, 50, 150)

    # Keep only pixels on the grid
    H, W = edges.shape
    mask = np.zeros_like(edges)
    for r in range(0, H, grid_spacing):
        for c in range(0, W, grid_spacing):
            mask[r, c] = 1

    selected = edges & mask
    positions = np.argwhere(selected > 0)  # (N, 2) with (row, col)

    return positions


def extract_full_weight_map(sr_image, bicubic_lr_up, block_size=7, sigma=0.4):
    """Extract features for every pixel in the HR image.

    Used at inference time to produce a full weight map.

    Args:
        sr_image:      (H, W) Phase-1 SR image.
        bicubic_lr_up: (H, W) bicubic-upscaled current LR image.
        block_size:    Block size.
        sigma:         Gaussian window sigma.

    Returns:
        (H*W, 50) float32 feature matrix.
        (H, W) shape tuple for reshaping.
    """
    H, W = sr_image.shape[:2]
    all_positions = np.array([(r, c) for r in range(H) for c in range(W)])
    features = extract_features_at_positions(
        sr_image, bicubic_lr_up, all_positions, block_size, sigma
    )
    return features, (H, W)
