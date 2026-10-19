"""
Utility functions for the Super-Resolution project.
Provides image I/O, color space conversion, metrics computation,
seed management, and visualization helpers.

Fixes applied (2026-10-19):
  - SSIM now correctly bounded in [-1, 1]
  - Global seed management via set_global_seed()
  - Synthetic image generator uses local RNG (no global seed pollution)
  - Grayscale loading as default (paper operates on grayscale)
"""

import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# ============================================================
# Seed management
# ============================================================

def set_global_seed(seed: int = 42):
    """Set random seed for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch (CPU+CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# File / directory helpers
# ============================================================

def ensure_dir(path):
    """Create directory if it doesn't exist. Returns the path."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# Image I/O
# ============================================================

def load_image(path, color='gray'):
    """Load an image from disk.

    Args:
        path: Path to image file.
        color: 'gray' (default, paper uses grayscale), 'rgb', or 'bgr'.

    Returns:
        numpy array float32 in [0, 1].
        Shape (H, W) for grayscale, (H, W, C) for colour.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    if color == 'gray':
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == 'rgb':
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 'bgr' leaves as-is

    return img.astype(np.float32) / 255.0


def save_image(img, path):
    """Save a float32 [0,1] image to disk."""
    ensure_dir(os.path.dirname(path))
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # For grayscale 2D arrays, save directly
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_uint8)


# ============================================================
# Tensor / numpy conversions
# ============================================================

def numpy_to_tensor(img):
    """Convert numpy image to torch tensor.

    (H,W) → (1,1,H,W)
    (H,W,C) → (1,C,H,W)
    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def tensor_to_numpy(tensor):
    """Convert torch tensor (1,C,H,W) or (C,H,W) to numpy image float32 [0,1]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = img.squeeze(2)
    return np.clip(img, 0, 1)


# ============================================================
# Metrics
# ============================================================

def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio between two images.

    Both images must be float arrays with the same shape.
    Returns PSNR in dB.  Returns inf if images are identical.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(img1, img2):
    """Compute Structural Similarity Index between two images.

    Returns a single scalar in [-1, 1].
    For multi-channel images, SSIM is computed per-channel and averaged.
    """
    from skimage.metrics import structural_similarity as ssim

    # Ensure float64 for precision
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if len(img1.shape) == 3:
        val = ssim(img1, img2, channel_axis=2, data_range=1.0)
    elif len(img1.shape) == 2:
        val = ssim(img1, img2, data_range=1.0)
    else:
        raise ValueError(f"Unexpected image shape: {img1.shape}")

    # Sanity check — SSIM must be in [-1, 1]
    assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, (
        f"SSIM out of range: {val:.6f}. Check inputs."
    )
    return float(np.clip(val, -1.0, 1.0))


# ============================================================
# Visualization helpers
# ============================================================

def create_comparison_figure(images, titles, save_path=None, figsize=(16, 4)):
    """Create a side-by-side comparison figure of multiple images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def create_metrics_plot(results, save_path=None):
    """Create a bar chart comparing PSNR/SSIM across methods."""
    methods = list(results.keys())
    psnr_values = [results[m]['psnr'] for m in methods]
    ssim_values = [results[m]['ssim'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96e6a1', '#dda0dd']

    bars1 = ax1.bar(methods, psnr_values, color=colors[:len(methods)],
                     edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Peak Signal-to-Noise Ratio', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, psnr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(methods, ssim_values, color=colors[:len(methods)],
                     edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('Structural Similarity Index', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
