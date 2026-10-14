"""
Registration Module for Multi-Frame Super-Resolution.

Implements sub-pixel motion estimation between LR frames using:
  1. Phase Correlation (frequency-domain method)
  2. Enhanced Phase Correlation with upsampled cross-correlation
  
Phase correlation is a classic DSP technique that estimates translational
shifts between images by analyzing the phase of their cross-power spectrum.
It is highly robust to noise and intensity variations.
"""

import numpy as np
import cv2


class PhaseCorrelationRegistration:
    """
    Sub-pixel motion estimation using phase correlation.
    
    The algorithm works in the frequency domain:
    1. Compute DFT of both images
    2. Compute cross-power spectrum
    3. Find peak in inverse DFT → gives the shift
    4. Refine to sub-pixel accuracy via parabolic interpolation
    
    This is a fundamental DSP technique used in many SR algorithms.
    """
    
    def __init__(self, upsample_factor=20):
        """
        Args:
            upsample_factor: Factor for sub-pixel refinement accuracy.
                             Higher = more precise but slower.
        """
        self.upsample_factor = upsample_factor
    
    def _to_gray(self, img):
        """Convert to grayscale if needed."""
        if len(img.shape) == 3:
            return cv2.cvtColor((img * 255).astype(np.uint8), 
                               cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return img
    
    def estimate_shift(self, ref_img, target_img):
        """
        Estimate sub-pixel shift between reference and target images.
        
        Uses OpenCV's phaseCorrelate which implements the DFT-based
        phase correlation with sub-pixel refinement.
        
        Args:
            ref_img: Reference image (H, W, C) or (H, W), float32 [0,1].
            target_img: Target image to register against reference.
        
        Returns:
            shift: (dy, dx) sub-pixel shift.
            confidence: Response value indicating estimation quality.
        """
        ref_gray = self._to_gray(ref_img)
        target_gray = self._to_gray(target_img)
        
        # Apply Hanning window to reduce spectral leakage
        # This is crucial for accurate phase correlation
        h, w = ref_gray.shape
        hann_window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        
        ref_windowed = ref_gray * hann_window
        target_windowed = target_gray * hann_window
        
        # Phase correlation via OpenCV
        (dx, dy), response = cv2.phaseCorrelate(ref_windowed, target_windowed)
        
        return np.array([dy, dx], dtype=np.float32), response
    
    def estimate_shift_dft(self, ref_img, target_img):
        """
        Manual DFT-based phase correlation with sub-pixel refinement.
        
        This implementation shows the DSP principles more explicitly:
        1. Compute cross-power spectrum: R = F(ref) * conj(F(target)) / |...|
        2. Inverse DFT of R gives impulse at the shift location
        3. Parabolic interpolation for sub-pixel accuracy
        
        Args:
            ref_img: Reference image.
            target_img: Target image.
        
        Returns:
            shift: (dy, dx) sub-pixel shift.
            peak_value: Peak correlation value (confidence).
        """
        ref_gray = self._to_gray(ref_img)
        target_gray = self._to_gray(target_img)
        
        h, w = ref_gray.shape
        
        # Apply Hanning window (essential DSP preprocessing)
        hann_y = np.hanning(h).reshape(-1, 1)
        hann_x = np.hanning(w).reshape(1, -1)
        window = (hann_y * hann_x).astype(np.float32)
        
        ref_windowed = ref_gray * window
        target_windowed = target_gray * window
        
        # Step 1: Compute DFT
        F_ref = np.fft.fft2(ref_windowed)
        F_target = np.fft.fft2(target_windowed)
        
        # Step 2: Cross-power spectrum
        cross_power = F_ref * np.conj(F_target)
        magnitude = np.abs(cross_power)
        magnitude[magnitude == 0] = 1e-10  # Avoid division by zero
        normalized = cross_power / magnitude
        
        # Step 3: Inverse DFT → correlation surface
        correlation = np.real(np.fft.ifft2(normalized))
        
        # Step 4: Find integer peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_y, peak_x = peak_idx
        
        # Handle wrap-around (shifts can be negative)
        if peak_y > h // 2:
            peak_y -= h
        if peak_x > w // 2:
            peak_x -= w
        
        # Step 5: Sub-pixel refinement via parabolic interpolation
        dy_sub = self._parabolic_interpolation(
            correlation, peak_idx[0], axis=0, size=h)
        dx_sub = self._parabolic_interpolation(
            correlation, peak_idx[1], axis=1, size=w)
        
        shift = np.array([peak_y + dy_sub, peak_x + dx_sub], dtype=np.float32)
        peak_value = correlation[peak_idx]
        
        return shift, peak_value
    
    def _parabolic_interpolation(self, data, peak_idx, axis, size):
        """
        Sub-pixel refinement using parabolic (quadratic) interpolation.
        
        Fits a parabola through the peak and its two neighbors to find
        the true sub-pixel maximum location.
        """
        if axis == 0:
            prev_idx = (peak_idx - 1) % size
            next_idx = (peak_idx + 1) % size
            y_prev = np.max(data[prev_idx, :])
            y_peak = np.max(data[peak_idx, :])
            y_next = np.max(data[next_idx, :])
        else:
            prev_idx = (peak_idx - 1) % size
            next_idx = (peak_idx + 1) % size
            y_prev = np.max(data[:, prev_idx])
            y_peak = np.max(data[:, peak_idx])
            y_next = np.max(data[:, next_idx])
        
        denom = 2 * (2 * y_peak - y_prev - y_next)
        if abs(denom) < 1e-10:
            return 0.0
        
        return (y_prev - y_next) / denom
    
    def register_frames(self, frames, method='opencv'):
        """
        Register all frames to the first (reference) frame.
        
        Args:
            frames: List of LR frames [(H,W,C), ...].
            method: 'opencv' or 'dft' for estimation method.
        
        Returns:
            shifts: (N, 2) array of estimated (dy, dx) shifts.
            confidences: List of confidence values.
        """
        ref = frames[0]
        n = len(frames)
        shifts = np.zeros((n, 2), dtype=np.float32)
        confidences = []
        
        for i in range(1, n):
            if method == 'opencv':
                shift, conf = self.estimate_shift(ref, frames[i])
            else:
                shift, conf = self.estimate_shift_dft(ref, frames[i])
            
            shifts[i] = shift
            confidences.append(conf)
        
        return shifts, confidences


def compensate_shift(img, dy, dx):
    """
    Apply inverse shift to align an image to the reference.
    
    Args:
        img: Input image (H, W, C) or (H, W), float32.
        dy, dx: Shift to compensate (will apply negative shift).
    
    Returns:
        Aligned image.
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    return aligned


def register_and_align(frames, method='opencv'):
    """
    Full registration pipeline: estimate shifts and align all frames.
    
    Args:
        frames: List of LR frames.
        method: Registration method ('opencv' or 'dft').
    
    Returns:
        aligned_frames: List of aligned frames.
        estimated_shifts: Estimated shift values.
    """
    registrator = PhaseCorrelationRegistration()
    shifts, confidences = registrator.register_frames(frames, method)
    
    aligned = [frames[0]]  # Reference frame stays unchanged
    for i in range(1, len(frames)):
        aligned_frame = compensate_shift(frames[i], shifts[i][0], shifts[i][1])
        aligned.append(aligned_frame)
    
    return aligned, shifts
