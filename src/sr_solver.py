"""
Iterative Gradient-Descent Super-Resolution Solver.

Implements the two-phase SR algorithm from the paper:

Phase 1 (unweighted — for artifact detection training):
    f_{j+1} = f_j - mu * [ sum_i W_i^T (W_i f_j - g_i) + lambda Q^T Q f_j ]
    Paper settings: lambda=0, 3 iterations, init = bicubic(g_c)

Phase 2 (weighted — final SR with per-pixel reliability):
    f_{j+1} = f_j - mu * [ sum_i M_i . W_i^T (W_i f_j - g_i) + lambda Q^T Q f_j ]
    where M_i is the weight map from the classifier.

NOTE on sign convention:
    The paper writes the update with +mu (gradient ascent on the correlation).
    We use the standard minimisation form f - mu*gradient, which is equivalent
    when the data-fidelity term is ||W_i f - g_i||^2.  We verified that PSNR
    improves over iterations with this sign.

Paper reference: Equations (2)-(3) and Section III.
"""

import numpy as np
from src.observation_model import (
    apply_W, apply_W_transpose, bicubic_upsample,
    downsample_nonoverlapping_avg,
)


def _laplacian_operator(image):
    """Apply discrete Laplacian Q to an image (for regularisation Q^T Q f).

    Uses the standard 3x3 Laplacian kernel:
        [0  1  0]
        [1 -4  1]
        [0  1  0]

    Returns Q*f (same shape as input).
    """
    import cv2
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _regularisation_term(f, lambda_reg):
    """Compute lambda * Q^T Q f.

    For the Laplacian, Q^T = Q (symmetric), so Q^T Q = Q^2.
    We apply Q twice.
    """
    if lambda_reg == 0:
        return np.zeros_like(f)
    Qf = _laplacian_operator(f)
    QTQf = _laplacian_operator(Qf)
    return lambda_reg * QTQf


def sr_phase1(lr_images, shifts, scale, mu=0.01, num_iterations=3,
              lambda_reg=0.0):
    """Phase 1 SR: unweighted iterative gradient descent.

    Produces one SR image per neighbouring LR frame pair (g_c, g_i).
    In the paper, for each neighbour g_i, Phase 1 super-resolves using
    the current frame g_c and g_i jointly.

    For simplicity and following the paper's approach:
    - g_c is lr_images[0] (the reference / current LR frame)
    - We produce a single SR estimate using ALL LR frames simultaneously

    Args:
        lr_images: list of (h, w) float32 LR frames.  lr_images[0] is g_c.
        shifts:    (N, 2) array of (dy, dx) shifts for each frame in LR domain.
                   shifts[0] should be (0, 0) for the reference frame.
        scale:     Upsampling factor.
        mu:        Gradient descent step size.  OUR CHOICE (paper unspecified).
        num_iterations: Number of iterations.  Paper says 3 for Phase 1.
        lambda_reg: Regularisation weight.  Paper says 0 for Phase 1.

    Returns:
        sr_image: (H, W) float32 super-resolved image.
    """
    n_frames = len(lr_images)
    h_lr, w_lr = lr_images[0].shape[:2]
    H, W = h_lr * scale, w_lr * scale

    # Initial estimate: bicubic upsampling of the reference frame g_c
    f = bicubic_upsample(lr_images[0], scale)

    for j in range(num_iterations):
        gradient = np.zeros((H, W), dtype=np.float32)

        for i in range(n_frames):
            dy_i, dx_i = shifts[i]
            # Forward: simulate LR observation from current HR estimate
            simulated_lr = apply_W(f, dy_i * scale, dx_i * scale, scale)
            # Residual in LR domain
            residual_lr = simulated_lr - lr_images[i]
            # Back-project to HR domain
            gradient_i = apply_W_transpose(residual_lr, dy_i * scale, dx_i * scale, scale)
            gradient += gradient_i

        # Regularisation
        gradient += _regularisation_term(f, lambda_reg)

        # Update
        f = f - mu * gradient
        f = np.clip(f, 0, 1)

    return f.astype(np.float32)


def sr_phase2(lr_images, shifts, weight_maps, scale, mu=0.01,
              num_iterations=10, lambda_reg=0.0):
    """Phase 2 SR: weighted iterative gradient descent using reliability maps.

    Same as Phase 1, but each frame's contribution is weighted by the
    per-pixel weight map M_i from the artifact detector.

    Args:
        lr_images:   list of (h, w) float32 LR frames.
        shifts:      (N, 2) array of (dy, dx) shifts in LR domain.
        weight_maps: list of (H, W) float32 weight maps M_i in [0, 1].
                     One per frame.  M_i(r,c) = classifier probability that
                     pixel (r,c) is artifact-free when using frame i.
        scale:       Upsampling factor.
        mu:          Step size.
        num_iterations: Number of iterations.
        lambda_reg:  Regularisation weight.

    Returns:
        sr_image: (H, W) float32 super-resolved image.
    """
    n_frames = len(lr_images)
    h_lr, w_lr = lr_images[0].shape[:2]
    H, W = h_lr * scale, w_lr * scale

    # Initial estimate: bicubic upsampling of the reference frame
    f = bicubic_upsample(lr_images[0], scale)

    for j in range(num_iterations):
        gradient = np.zeros((H, W), dtype=np.float32)

        for i in range(n_frames):
            dy_i, dx_i = shifts[i]
            simulated_lr = apply_W(f, dy_i * scale, dx_i * scale, scale)
            residual_lr = simulated_lr - lr_images[i]
            gradient_i = apply_W_transpose(residual_lr, dy_i * scale, dx_i * scale, scale)

            # Apply per-pixel weight map M_i
            gradient_i = weight_maps[i] * gradient_i
            gradient += gradient_i

        # Regularisation
        gradient += _regularisation_term(f, lambda_reg)

        # Update
        f = f - mu * gradient
        f = np.clip(f, 0, 1)

    return f.astype(np.float32)


def sr_unweighted(lr_images, shifts, scale, mu=0.01, num_iterations=10,
                  lambda_reg=0.0):
    """Unweighted L2 SR (all M_i = 1).  Used as a baseline.

    This is equivalent to Phase 1 but with more iterations and
    optional regularisation, for fair comparison against weighted Phase 2.
    """
    H = lr_images[0].shape[0] * scale
    W = lr_images[0].shape[1] * scale
    ones = [np.ones((H, W), dtype=np.float32)] * len(lr_images)
    return sr_phase2(lr_images, shifts, ones, scale, mu, num_iterations,
                     lambda_reg)
