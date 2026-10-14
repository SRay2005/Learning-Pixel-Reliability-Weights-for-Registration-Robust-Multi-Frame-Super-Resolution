"""
Data Pipeline for Multi-Frame Super-Resolution.

Generates synthetic low-resolution (LR) frame sequences from high-resolution (HR)
images by simulating:
  1. Sub-pixel translational motion between frames
  2. Downsampling (decimation) with anti-aliasing
  3. Additive Gaussian noise
  4. Optional registration error injection

This pipeline is the foundation of the SR system — it creates the training
data that teaches the network to be robust to registration inaccuracies.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import load_image, ensure_dir, numpy_to_tensor


class SRDataPipeline:
    """
    Multi-frame super-resolution data generation pipeline.
    
    Given an HR image, generates N LR frames with known sub-pixel shifts,
    simulating the real-world scenario of capturing multiple slightly
    displaced observations of the same scene.
    """
    
    def __init__(self, scale_factor=2, num_frames=4, noise_std=0.01):
        """
        Args:
            scale_factor: Downsampling factor (e.g., 2 means HR is 2x LR).
            num_frames: Number of LR frames to generate per HR image.
            noise_std: Standard deviation of additive Gaussian noise.
        """
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        self.noise_std = noise_std
    
    def generate_shifts(self, max_shift=1.0):
        """
        Generate random sub-pixel shifts for each LR frame.
        
        In real multi-frame SR, these shifts come from camera motion or
        object motion. Here we simulate them synthetically.
        
        Args:
            max_shift: Maximum shift in pixels (in HR domain).
        
        Returns:
            shifts: (num_frames, 2) array of (dy, dx) sub-pixel shifts.
        """
        # First frame is reference (no shift)
        shifts = np.zeros((self.num_frames, 2))
        for i in range(1, self.num_frames):
            shifts[i] = np.random.uniform(-max_shift, max_shift, size=2)
        return shifts
    
    def apply_shift(self, img, dy, dx):
        """
        Apply sub-pixel translational shift to an image using affine transform.
        
        Args:
            img: Input image (H, W, C) or (H, W), float32.
            dy, dx: Sub-pixel shift amounts.
        
        Returns:
            Shifted image.
        """
        h, w = img.shape[:2]
        # Affine transformation matrix for translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), 
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    def downsample(self, img):
        """
        Downsample image with anti-aliasing (simulates sensor integration).
        
        Uses Gaussian blur before decimation to prevent aliasing,
        mimicking the point spread function (PSF) of a real camera.
        """
        s = self.scale_factor
        # Anti-aliasing Gaussian blur (sigma proportional to scale)
        sigma = 0.5 * s
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        
        h, w = img.shape[:2]
        new_h, new_w = h // s, w // s
        
        # Decimate
        downsampled = cv2.resize(blurred, (new_w, new_h), 
                                  interpolation=cv2.INTER_AREA)
        return downsampled
    
    def add_noise(self, img):
        """Add Gaussian noise to simulate sensor noise."""
        noise = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
        return np.clip(img + noise, 0, 1)
    
    def generate_lr_frames(self, hr_image, max_shift=1.5):
        """
        Generate multiple LR frames from a single HR image.
        
        Pipeline:  HR → shift → blur → downsample → noise → LR
        
        Args:
            hr_image: High-resolution image (H, W, C), float32 [0,1].
            max_shift: Maximum sub-pixel shift in HR domain.
        
        Returns:
            lr_frames: List of LR images.
            shifts: The true sub-pixel shifts used.
        """
        shifts = self.generate_shifts(max_shift)
        lr_frames = []
        
        for i in range(self.num_frames):
            # Apply sub-pixel shift in HR domain
            shifted = self.apply_shift(hr_image, shifts[i][0], shifts[i][1])
            # Downsample
            lr = self.downsample(shifted)
            # Add noise
            lr = self.add_noise(lr)
            lr_frames.append(lr)
        
        return lr_frames, shifts
    
    def add_registration_error(self, shifts, error_std=0.3):
        """
        Add registration errors to the true shifts.
        
        This simulates the inaccuracy of real motion estimation algorithms.
        The error_std controls how much the estimated shifts deviate from
        the true shifts — this is the KEY parameter for robustness testing.
        
        Args:
            shifts: True sub-pixel shifts (N, 2).
            error_std: Standard deviation of registration error (in LR pixels).
        
        Returns:
            noisy_shifts: Shifts with added registration errors.
        """
        errors = np.random.randn(*shifts.shape).astype(np.float32) * error_std
        # Reference frame remains unchanged
        errors[0] = 0
        return shifts + errors


class MultiFrameSRDataset(Dataset):
    """
    PyTorch Dataset for multi-frame super-resolution training.
    
    Loads HR images, generates LR frame sequences on-the-fly with
    random sub-pixel shifts, and optionally injects registration errors
    during training to build robustness.
    """
    
    def __init__(self, hr_dir, scale_factor=2, num_frames=4, patch_size=64,
                 noise_std=0.01, reg_error_std=0.0, augment=True, patches_per_image=16):
        """
        Args:
            hr_dir: Directory containing HR images.
            scale_factor: SR scale factor.
            num_frames: Number of LR frames per sample.
            patch_size: Size of HR patches for training.
            noise_std: Sensor noise std.
            reg_error_std: Registration error std (0 = perfect registration).
            augment: Whether to apply data augmentation.
            patches_per_image: Number of patches extracted per image per epoch.
        """
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.reg_error_std = reg_error_std
        self.augment = augment
        self.patches_per_image = patches_per_image
        
        self.pipeline = SRDataPipeline(scale_factor, num_frames, noise_std)
        
        # Collect image paths
        self.hr_paths = []
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for f in sorted(os.listdir(hr_dir)):
            if os.path.splitext(f)[1].lower() in valid_ext:
                self.hr_paths.append(os.path.join(hr_dir, f))
        
        if len(self.hr_paths) == 0:
            raise ValueError(f"No images found in {hr_dir}")
    
    def __len__(self):
        return len(self.hr_paths) * self.patches_per_image
    
    def _extract_patch(self, img):
        """Extract a random patch from the image."""
        h, w = img.shape[:2]
        ps = self.patch_size
        
        if h < ps or w < ps:
            # Resize up if image is too small
            scale = max(ps / h, ps / w) * 1.1
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        
        y = np.random.randint(0, h - ps + 1)
        x = np.random.randint(0, w - ps + 1)
        return img[y:y+ps, x:x+ps]
    
    def _augment(self, hr_patch, lr_patches):
        """Apply random flip/rotation augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            hr_patch = hr_patch[:, ::-1].copy()
            lr_patches = [p[:, ::-1].copy() for p in lr_patches]
        
        # Random vertical flip
        if np.random.random() > 0.5:
            hr_patch = hr_patch[::-1].copy()
            lr_patches = [p[::-1].copy() for p in lr_patches]
        
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            hr_patch = np.rot90(hr_patch, k).copy()
            lr_patches = [np.rot90(p, k).copy() for p in lr_patches]
        
        return hr_patch, lr_patches
    
    def __getitem__(self, idx):
        img_idx = idx % len(self.hr_paths)
        hr_image = load_image(self.hr_paths[img_idx], color='rgb')
        if len(hr_image.shape) == 2:
            hr_image = np.stack([hr_image] * 3, axis=2)
        elif hr_image.shape[2] == 4:
            hr_image = hr_image[:, :, :3]
        
        # Extract random HR patch
        hr_patch = self._extract_patch(hr_image)
        
        # Generate LR frames
        lr_frames, true_shifts = self.pipeline.generate_lr_frames(hr_patch)
        
        # Augmentation
        if self.augment:
            hr_patch, lr_frames = self._augment(hr_patch, lr_frames)
        
        # Add registration errors to shifts (for robustness training)
        if self.reg_error_std > 0:
            estimated_shifts = self.pipeline.add_registration_error(
                true_shifts / self.scale_factor,  # Convert to LR domain
                self.reg_error_std
            )
        else:
            estimated_shifts = true_shifts / self.scale_factor
        
        # Upsample LR frames to HR size (pre-processing for network input)
        lr_upsampled = []
        for lr in lr_frames:
            if len(lr.shape) == 2:
                lr = np.stack([lr] * 3, axis=2)
            up = cv2.resize(lr, (self.patch_size, self.patch_size),
                          interpolation=cv2.INTER_CUBIC)
            if len(up.shape) == 2:
                up = np.stack([up] * 3, axis=2)
            lr_upsampled.append(up)
        
        # Stack LR frames along channel dimension: (N*C, H, W)
        lr_stack = np.concatenate(lr_upsampled, axis=2)  # (H, W, N*C)
        lr_tensor = torch.from_numpy(lr_stack.transpose(2, 0, 1)).float()
        hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float()
        shifts_tensor = torch.from_numpy(estimated_shifts).float()
        
        return lr_tensor, hr_tensor, shifts_tensor


def create_data_loaders(hr_dir, scale_factor=2, num_frames=4, patch_size=64,
                        batch_size=8, noise_std=0.01, reg_error_std=0.3,
                        num_workers=0):
    """
    Create training and validation data loaders.
    
    Training uses registration error injection for robustness.
    Validation uses perfect registration for fair evaluation.
    """
    # Training: with registration error augmentation
    train_dataset = MultiFrameSRDataset(
        hr_dir=hr_dir,
        scale_factor=scale_factor,
        num_frames=num_frames,
        patch_size=patch_size,
        noise_std=noise_std,
        reg_error_std=reg_error_std,  # KEY: inject reg errors during training
        augment=True
    )
    
    # Validation: perfect registration (test ideal conditions)
    val_dataset = MultiFrameSRDataset(
        hr_dir=hr_dir,
        scale_factor=scale_factor,
        num_frames=num_frames,
        patch_size=patch_size,
        noise_std=noise_std,
        reg_error_std=0.0,  # No registration error for validation
        augment=False,
        patches_per_image=2,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
