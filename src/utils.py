"""
Utility functions for the Super-Resolution project.
Provides image I/O, color space conversion, metrics computation,
and visualization helpers.
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import requests
from io import BytesIO


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def load_image(path, color='rgb'):
    """Load an image from disk.
    
    Args:
        path: Path to image file.
        color: 'rgb', 'bgr', or 'gray'.
    
    Returns:
        numpy array (H, W, C) or (H, W) for grayscale, float32 in [0, 1].
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    
    if color == 'rgb':
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif color == 'gray':
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    """Save a float32 [0,1] image to disk."""
    ensure_dir(os.path.dirname(path))
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)
    
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_uint8)


def numpy_to_tensor(img):
    """Convert numpy image (H,W,C) or (H,W) float32 [0,1] to torch tensor (1,C,H,W)."""
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def tensor_to_numpy(tensor):
    """Convert torch tensor (1,C,H,W) or (C,H,W) to numpy image (H,W,C) float32 [0,1]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = img.squeeze(2)
    return np.clip(img, 0, 1)


def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(img1, img2):
    """Compute Structural Similarity Index between two images."""
    from skimage.metrics import structural_similarity as ssim
    
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    if len(img1.shape) == 3:
        return ssim(img1, img2, channel_axis=2, data_range=1.0)
    return ssim(img1, img2, data_range=1.0)


def download_sample_images(save_dir, num_images=8):
    """Generate diverse sample images for training/testing.
    
    Creates synthetic images with varied textures, edges, and patterns
    that provide good training signal for the SR network.
    """
    ensure_dir(save_dir)
    
    names = ["butterfly", "flower", "building", "landscape",
             "bird", "texture", "portrait", "city"]
    
    downloaded = []
    for i, name in enumerate(names[:num_images]):
        filepath = os.path.join(save_dir, f"{name}.png")
        if os.path.exists(filepath):
            downloaded.append(filepath)
            continue
        
        print(f"  Generating {name}.png...")
        img = create_synthetic_image(256, 256, seed=i * 42 + 7)
        save_image(img, filepath)
        downloaded.append(filepath)
    
    return downloaded


def create_synthetic_image(height, width, seed=None):
    """Create a rich synthetic image with varied edges, textures, and gradients.
    
    Each seed produces a visually distinct image with different combinations
    of shapes, color palettes, and texture patterns — ensuring diverse training.
    """
    if seed is not None:
        np.random.seed(seed)
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Random color palette
    base_hues = np.random.uniform(0, 1, size=3)
    
    # Gradient background (random direction)
    angle = np.random.uniform(0, 2 * np.pi)
    grad = np.cos(angle) * xx + np.sin(angle) * yy
    for c in range(3):
        img[:, :, c] = base_hues[c] * 0.3 + 0.15 * grad
    
    # Random geometric shapes (5-12)
    num_shapes = np.random.randint(5, 13)
    for _ in range(num_shapes):
        color = tuple(np.random.uniform(0.2, 0.95, size=3).tolist())
        shape_type = np.random.choice(['circle', 'rect', 'ellipse', 'line'])
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        if shape_type == 'circle':
            r = np.random.randint(min(height, width) // 15, min(height, width) // 4)
            thickness = np.random.choice([-1, 1, 2, 3])
            cv2.circle(img, (cx, cy), r, color, thickness)
        elif shape_type == 'rect':
            w2 = np.random.randint(10, width // 3)
            h2 = np.random.randint(10, height // 3)
            thickness = np.random.choice([-1, 1, 2])
            cv2.rectangle(img, (cx, cy), (cx + w2, cy + h2), color, thickness)
        elif shape_type == 'ellipse':
            axes = (np.random.randint(10, width // 4), np.random.randint(10, height // 4))
            angle_deg = np.random.uniform(0, 360)
            cv2.ellipse(img, (cx, cy), axes, angle_deg, 0, 360, color, -1)
        elif shape_type == 'line':
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)
            cv2.line(img, (cx, cy), (x2, y2), color, np.random.randint(1, 4))
    
    # Sinusoidal texture patterns (multi-frequency)
    for _ in range(np.random.randint(3, 8)):
        freq = np.random.uniform(2, 15)
        phase = np.random.uniform(0, 2 * np.pi)
        direction = np.random.uniform(0, 2 * np.pi)
        pattern = np.cos(direction) * xx + np.sin(direction) * yy
        channel = np.random.randint(0, 3)
        amplitude = np.random.uniform(0.02, 0.08)
        img[:, :, channel] += amplitude * np.sin(2 * np.pi * freq * pattern + phase)
    
    # Concentric rings (adds high-frequency detail)
    if np.random.random() > 0.4:
        cx_r = np.random.uniform(0.2, 0.8)
        cy_r = np.random.uniform(0.2, 0.8)
        dist = np.sqrt((xx - cx_r)**2 + (yy - cy_r)**2)
        ring_freq = np.random.uniform(8, 20)
        for c in range(3):
            img[:, :, c] += 0.04 * np.sin(2 * np.pi * ring_freq * dist)
    
    # Checkerboard region (tests aliasing behavior)
    if np.random.random() > 0.5:
        check_size = np.random.randint(4, 16)
        cx_s = np.random.randint(0, width // 2)
        cy_s = np.random.randint(0, height // 2)
        cw = np.random.randint(width // 4, width // 2)
        ch = np.random.randint(height // 4, height // 2)
        for cy_p in range(cy_s, min(cy_s + ch, height)):
            for cx_p in range(cx_s, min(cx_s + cw, width)):
                if ((cy_p // check_size) + (cx_p // check_size)) % 2 == 0:
                    img[cy_p, cx_p] += 0.05
    
    # Gaussian noise for realism
    noise = np.random.randn(height, width, 3).astype(np.float32) * 0.02
    img += noise
    
    return np.clip(img, 0, 1)


def create_comparison_figure(images, titles, save_path=None, figsize=(16, 4)):
    """Create a side-by-side comparison figure of multiple images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = tensor_to_numpy(img)
        ax.imshow(img if len(img.shape) == 3 else img, cmap='gray')
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
