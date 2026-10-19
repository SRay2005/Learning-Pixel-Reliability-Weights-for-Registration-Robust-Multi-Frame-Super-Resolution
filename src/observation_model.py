"""
Observation Model W_i for Multi-Frame Super-Resolution.

Implements the forward model relating HR images to LR observations:
    g_i = W_i f + noise
where W_i = D * S_i:
    S_i = sub-pixel shift operator (translation by motion vector)
    D   = downsampling by non-overlapping averaging (box filter, stride=scale)

Also implements the adjoint (transpose) operator W_i^T needed for
the gradient-descent SR update.

Paper reference: Section II, equation (1).
"""

import numpy as np
import cv2


def apply_shift(image, dy, dx):
    """Apply sub-pixel translational shift to a 2D image.

    Args:
        image: (H, W) float32 grayscale image.
        dy: Vertical shift (positive = down).
        dx: Horizontal shift (positive = right).

    Returns:
        Shifted image, same shape.
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return shifted


def apply_inverse_shift(image, dy, dx):
    """Apply the inverse (negative) shift."""
    return apply_shift(image, -dy, -dx)


def downsample_nonoverlapping_avg(image, scale):
    """Downsample by non-overlapping block averaging.

    This is the paper's downsampling model: partition the HR image into
    non-overlapping scale×scale blocks and average each block to produce
    one LR pixel.  This is equivalent to a box filter with stride=scale.

    Args:
        image: (H, W) float32, H and W must be divisible by scale.
        scale: Integer downsampling factor.

    Returns:
        (H//scale, W//scale) float32 image.
    """
    h, w = image.shape[:2]
    assert h % scale == 0 and w % scale == 0, (
        f"Image size ({h},{w}) not divisible by scale {scale}"
    )
    # Reshape into blocks and average
    lr = image.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))
    return lr.astype(np.float32)


def upsample_replicate(lr_image, scale):
    """Adjoint of non-overlapping averaging: replicate each LR pixel
    into a scale×scale block (and divide by scale^2 for proper adjoint).

    This is W_i^T applied to an LR-domain signal.  The division by scale^2
    accounts for the averaging in the forward operator.

    Args:
        lr_image: (h, w) float32.
        scale: Upsampling factor.

    Returns:
        (h*scale, w*scale) float32 image.
    """
    h, w = lr_image.shape[:2]
    # np.repeat along both axes
    hr = np.repeat(np.repeat(lr_image, scale, axis=0), scale, axis=1)
    # Proper adjoint scaling: each LR pixel contributed to scale^2 HR pixels
    # in the forward pass via averaging, so adjoint divides.
    return hr.astype(np.float32) / (scale ** 2)


def apply_W(hr_image, dy, dx, scale):
    """Forward observation operator: W_i f = D * S_i(f).

    Shift HR image by (dy, dx), then downsample by non-overlapping averaging.

    Args:
        hr_image: (H, W) float32.
        dy, dx: Sub-pixel shift of LR frame i relative to reference.
        scale: Downsampling factor.

    Returns:
        LR image (H//scale, W//scale) float32.
    """
    shifted = apply_shift(hr_image, dy, dx)
    lr = downsample_nonoverlapping_avg(shifted, scale)
    return lr


def apply_W_transpose(lr_residual, dy, dx, scale):
    """Adjoint (transpose) operator: W_i^T e.

    Upsample LR residual by replication, then inverse-shift.

    Args:
        lr_residual: (h, w) float32 residual in LR domain.
        dy, dx: Shift parameters of W_i.
        scale: Upsampling factor.

    Returns:
        HR-domain image (h*scale, w*scale) float32.
    """
    # Adjoint of downsampling: replicate
    hr_residual = upsample_replicate(lr_residual, scale)
    # Adjoint of shift S_i is inverse shift S_i^{-1}
    hr_aligned = apply_inverse_shift(hr_residual, dy, dx)
    return hr_aligned


def bicubic_upsample(lr_image, scale):
    """Bicubic interpolation upsampling (for initial SR estimate).

    Args:
        lr_image: (h, w) float32.
        scale: Upsampling factor.

    Returns:
        (h*scale, w*scale) float32 image.
    """
    h, w = lr_image.shape[:2]
    upsampled = cv2.resize(
        lr_image, (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(upsampled, 0, 1).astype(np.float32)
