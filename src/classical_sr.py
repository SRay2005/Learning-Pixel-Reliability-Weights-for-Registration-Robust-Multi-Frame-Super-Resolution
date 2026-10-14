"""
Classical Super-Resolution Baselines.

Implements traditional (non-ML) super-resolution methods for comparison:

  1. Bicubic Interpolation — simplest upsampling baseline
  2. Iterative Back-Projection (IBP) — classic multi-frame SR method
  3. Mean Fusion — simple frame averaging after registration

These classical methods show WHY ML-based approaches are needed:
they degrade severely under registration errors because they have
no mechanism to adaptively weight or reject misaligned data.
"""

import numpy as np
import cv2


def bicubic_upsample(lr_image, scale_factor):
    """
    Bicubic interpolation upsampling.
    
    The simplest SR baseline — just interpolates a single LR frame.
    Produces smooth but blurry results. No multi-frame information used.
    
    Args:
        lr_image: Low-resolution input (H, W, C), float32 [0,1].
        scale_factor: Upsampling factor.
    
    Returns:
        Upsampled image (sH, sW, C).
    """
    h, w = lr_image.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    upsampled = cv2.resize(lr_image, (new_w, new_h),
                           interpolation=cv2.INTER_CUBIC)
    return np.clip(upsampled, 0, 1)


def mean_fusion_sr(lr_frames, scale_factor, shifts=None):
    """
    Simple mean fusion super-resolution.
    
    Aligns frames using estimated shifts, upsamples each, and averages.
    This is the simplest multi-frame approach. Averaging reduces noise
    but also blurs registration errors across the output.
    
    Args:
        lr_frames: List of LR frames [(H,W,C), ...].
        scale_factor: Upsampling factor.
        shifts: Optional (N, 2) shift estimates. If None, no alignment.
    
    Returns:
        Mean-fused HR image.
    """
    h, w = lr_frames[0].shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    
    upsampled_frames = []
    for i, frame in enumerate(lr_frames):
        # Compensate shift if provided
        if shifts is not None and i > 0:
            dy, dx = shifts[i]
            M = np.float32([[1, 0, -dx * scale_factor], 
                           [0, 1, -dy * scale_factor]])
            frame_shifted = cv2.warpAffine(frame, M, (w, h),
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REFLECT)
        else:
            frame_shifted = frame
        
        upsampled = cv2.resize(frame_shifted, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
        upsampled_frames.append(upsampled)
    
    # Average all aligned, upsampled frames
    result = np.mean(upsampled_frames, axis=0)
    return np.clip(result, 0, 1).astype(np.float32)


def iterative_back_projection(lr_frames, scale_factor, shifts=None,
                               num_iterations=20, step_size=0.5,
                               blur_sigma=0.6):
    """
    Iterative Back-Projection (IBP) super-resolution.
    
    A classical multi-frame SR algorithm that iteratively refines an
    HR estimate by:
    1. Simulating LR observations from current HR estimate
    2. Computing the error between simulated and actual LR frames
    3. Back-projecting the error to update the HR estimate
    
    This is a fundamental SR algorithm from Irani & Peleg (1991).
    It works well with perfect registration but degrades significantly
    with registration errors — making it a good baseline to compare
    against our ML-based robust approach.
    
    Args:
        lr_frames: List of LR frames [(H,W,C), ...].
        scale_factor: Upsampling factor.
        shifts: (N, 2) array of estimated shifts in LR domain.
        num_iterations: Number of IBP iterations.
        step_size: Learning rate for the update.
        blur_sigma: Sigma of the PSF (point spread function) for simulation.
    
    Returns:
        HR reconstruction.
    """
    s = scale_factor
    h, w = lr_frames[0].shape[:2]
    new_h, new_w = h * s, w * s
    n_frames = len(lr_frames)
    
    if shifts is None:
        shifts = np.zeros((n_frames, 2), dtype=np.float32)
    
    # Initialize HR estimate with mean fusion
    hr_estimate = mean_fusion_sr(lr_frames, s, shifts)
    
    for iteration in range(num_iterations):
        # Accumulate updates from all frames
        update = np.zeros_like(hr_estimate)
        
        for i in range(n_frames):
            # Step 1: Simulate LR observation from current HR estimate
            # Apply the shift to HR estimate
            dy_hr = shifts[i][0] * s
            dx_hr = shifts[i][1] * s
            M_shift = np.float32([[1, 0, dx_hr], [0, 1, dy_hr]])
            hr_shifted = cv2.warpAffine(hr_estimate, M_shift, 
                                         (new_w, new_h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REFLECT)
            
            # Apply PSF blur (simulating camera optics)
            hr_blurred = cv2.GaussianBlur(hr_shifted, (0, 0), blur_sigma * s)
            
            # Downsample to LR
            simulated_lr = cv2.resize(hr_blurred, (w, h),
                                       interpolation=cv2.INTER_AREA)
            
            # Step 2: Compute error
            error = lr_frames[i] - simulated_lr
            
            # Step 3: Back-project error to HR domain
            # Upsample error
            error_hr = cv2.resize(error, (new_w, new_h),
                                   interpolation=cv2.INTER_CUBIC)
            
            # Apply inverse shift
            M_inv = np.float32([[1, 0, -dx_hr], [0, 1, -dy_hr]])
            error_aligned = cv2.warpAffine(error_hr, M_inv,
                                            (new_w, new_h),
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REFLECT)
            
            # Apply back-projection filter (mild blur for stability)
            bp_filter = cv2.GaussianBlur(error_aligned, (0, 0), 0.5)
            
            update += bp_filter
        
        # Average update across frames
        update /= n_frames
        
        # Update HR estimate
        hr_estimate += step_size * update
        hr_estimate = np.clip(hr_estimate, 0, 1)
    
    return hr_estimate.astype(np.float32)


def run_all_classical(lr_frames, scale_factor, shifts=None):
    """
    Run all classical SR methods and return results.
    
    Args:
        lr_frames: List of LR frames.
        scale_factor: Upsampling factor.
        shifts: Estimated shifts.
    
    Returns:
        Dictionary of {method_name: hr_result}.
    """
    results = {}
    
    # Bicubic (single frame)
    results['Bicubic'] = bicubic_upsample(lr_frames[0], scale_factor)
    
    # Mean Fusion
    results['Mean Fusion'] = mean_fusion_sr(lr_frames, scale_factor, shifts)
    
    # IBP
    results['IBP'] = iterative_back_projection(
        lr_frames, scale_factor, shifts,
        num_iterations=20, step_size=0.5
    )
    
    return results
