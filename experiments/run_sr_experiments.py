"""
Experiment: SR quality evaluation with different classifiers and baselines.

Produces:
  - PSNR/SSIM vs registration error level (robustness curves)
  - PSNR/SSIM vs number of LR frames
  - Per-image results table
  - Weight map visualizations

Baselines:
  - Bicubic (single frame)
  - Unweighted L2 SR (all M_i = 1)
  - Oracle weights (from true error mask — upper bound)
  - Each classifier's weights
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (set_global_seed, load_image, save_image,
                        compute_psnr, compute_ssim, ensure_dir)
from src.observation_model import (
    apply_shift, downsample_nonoverlapping_avg, bicubic_upsample,
)
from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted
from src.artifact_detector import ArtifactDetector
from src.pca_transform import PCATransform
from src.classifiers import (
    LogisticRegressionClassifier,
    KernelSVMClassifier,
    GDAClassifier,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_lr_sequence(hr, scale, n_frames, reg_error=0.0, rng=None):
    """Generate LR frames with optional registration error.

    Returns:
        lr_frames: list of (h,w) arrays
        true_shifts: (N,2) true shifts in LR domain
        noisy_shifts: (N,2) shifts with added error (what the SR sees)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    H, W = hr.shape
    lr_frames = []
    true_shifts_hr = np.zeros((n_frames, 2), dtype=np.float32)

    for i in range(n_frames):
        if i > 0:
            true_shifts_hr[i] = [rng.uniform(-1.5, 1.5),
                                  rng.uniform(-1.5, 1.5)]
        shifted = apply_shift(hr, true_shifts_hr[i, 0], true_shifts_hr[i, 1])
        lr = downsample_nonoverlapping_avg(shifted, scale)
        lr_frames.append(lr)

    true_shifts_lr = true_shifts_hr / scale

    # Add registration error
    noisy_shifts_lr = true_shifts_lr.copy()
    if reg_error > 0:
        errors = rng.randn(n_frames, 2).astype(np.float32) * reg_error
        errors[0] = 0  # Reference frame has no error
        noisy_shifts_lr = true_shifts_lr + errors

    return lr_frames, true_shifts_lr, noisy_shifts_lr


def oracle_weight_maps(hr, lr_frames, true_shifts, noisy_shifts, scale,
                        mu, n_iters, threshold=0.02):
    """Compute oracle weight maps from the known ground truth.

    Pixels where the SR error (using noisy shifts) exceeds a threshold
    relative to SR with true shifts are marked as unreliable.
    """
    H = lr_frames[0].shape[0] * scale
    W = lr_frames[0].shape[1] * scale

    weight_maps = []
    for i in range(len(lr_frames)):
        if i == 0:
            weight_maps.append(np.ones((H, W), dtype=np.float32))
            continue

        # SR with true shifts (reference)
        lr_pair = [lr_frames[0], lr_frames[i]]
        shifts_true = np.array([true_shifts[0], true_shifts[i]])
        sr_true = sr_phase1(lr_pair, shifts_true, scale, mu=mu,
                            num_iterations=n_iters)

        # SR with noisy shifts
        shifts_noisy = np.array([noisy_shifts[0], noisy_shifts[i]])
        sr_noisy = sr_phase1(lr_pair, shifts_noisy, scale, mu=mu,
                             num_iterations=n_iters)

        # Error mask
        error = np.abs(sr_true - sr_noisy)
        M_i = (error < threshold).astype(np.float32)
        weight_maps.append(M_i)

    return weight_maps


def run_error_sweep(test_images, scale, n_frames, mu, n_iters,
                    error_levels, n_trials, results_dir):
    """PSNR/SSIM vs registration error level."""
    print("\n  --- Error Level Sweep ---")

    methods = ['Bicubic', 'Unweighted SR', 'Oracle']
    all_psnr = {m: {e: [] for e in error_levels} for m in methods}
    all_ssim = {m: {e: [] for e in error_levels} for m in methods}

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        for hr in test_images:
            for err in error_levels:
                lr_frames, true_sh, noisy_sh = generate_lr_sequence(
                    hr, scale, n_frames, reg_error=err, rng=rng)

                H, W = hr.shape

                # Bicubic
                bic = bicubic_upsample(lr_frames[0], scale)[:H, :W]
                all_psnr['Bicubic'][err].append(compute_psnr(bic, hr))
                all_ssim['Bicubic'][err].append(compute_ssim(bic, hr))

                # Unweighted SR
                sr_uw = sr_unweighted(lr_frames, noisy_sh, scale,
                                      mu=mu, num_iterations=n_iters)[:H, :W]
                all_psnr['Unweighted SR'][err].append(compute_psnr(sr_uw, hr))
                all_ssim['Unweighted SR'][err].append(compute_ssim(sr_uw, hr))

                # Oracle
                oracle_maps = oracle_weight_maps(
                    hr, lr_frames, true_sh, noisy_sh, scale, mu, 3)
                sr_oracle = sr_phase2(lr_frames, noisy_sh, oracle_maps, scale,
                                      mu=mu, num_iterations=n_iters)[:H, :W]
                all_psnr['Oracle'][err].append(compute_psnr(sr_oracle, hr))
                all_ssim['Oracle'][err].append(compute_ssim(sr_oracle, hr))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'Bicubic': '#e74c3c', 'Unweighted SR': '#f39c12', 'Oracle': '#27ae60'}
    markers = {'Bicubic': 'o', 'Unweighted SR': 's', 'Oracle': 'D'}

    for m in methods:
        mean_psnr = [np.mean(all_psnr[m][e]) for e in error_levels]
        std_psnr = [np.std(all_psnr[m][e]) for e in error_levels]
        ax1.errorbar(error_levels, mean_psnr, yerr=std_psnr,
                    color=colors[m], marker=markers[m],
                    linewidth=2, markersize=7, capsize=4, label=m)

        mean_ssim = [np.mean(all_ssim[m][e]) for e in error_levels]
        std_ssim = [np.std(all_ssim[m][e]) for e in error_levels]
        ax2.errorbar(error_levels, mean_ssim, yerr=std_ssim,
                    color=colors[m], marker=markers[m],
                    linewidth=2, markersize=7, capsize=4, label=m)

    ax1.set_xlabel('Registration Error (px)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Registration Error', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Registration Error (px)', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM vs Registration Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('SR Robustness to Registration Errors (Baselines)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_sweep_baselines.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    with open(os.path.join(results_dir, 'error_sweep_baselines.csv'), 'w') as f:
        f.write('error_level,method,mean_psnr,std_psnr,mean_ssim,std_ssim\n')
        for e in error_levels:
            for m in methods:
                mp = np.mean(all_psnr[m][e])
                sp = np.std(all_psnr[m][e])
                ms = np.mean(all_ssim[m][e])
                ss = np.std(all_ssim[m][e])
                f.write(f'{e},{m},{mp:.4f},{sp:.4f},{ms:.4f},{ss:.4f}\n')

    print("  Error sweep complete.")


def run_frames_sweep(test_images, scale, mu, n_iters, frame_counts,
                     n_trials, results_dir, reg_error=0.5):
    """PSNR vs number of LR frames (like the paper's Fig. 3)."""
    print("\n  --- Frames Sweep ---")

    methods = ['Bicubic', 'Unweighted SR']
    all_psnr = {m: {n: [] for n in frame_counts} for m in methods}

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        for hr in test_images:
            for n_frames in frame_counts:
                lr_frames, true_sh, noisy_sh = generate_lr_sequence(
                    hr, scale, n_frames, reg_error=reg_error, rng=rng)
                H, W = hr.shape

                bic = bicubic_upsample(lr_frames[0], scale)[:H, :W]
                all_psnr['Bicubic'][n_frames].append(compute_psnr(bic, hr))

                sr_uw = sr_unweighted(lr_frames, noisy_sh, scale,
                                      mu=mu, num_iterations=n_iters)[:H, :W]
                all_psnr['Unweighted SR'][n_frames].append(compute_psnr(sr_uw, hr))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Bicubic': '#e74c3c', 'Unweighted SR': '#f39c12'}

    for m in methods:
        mean_p = [np.mean(all_psnr[m][n]) for n in frame_counts]
        std_p = [np.std(all_psnr[m][n]) for n in frame_counts]
        ax.errorbar(frame_counts, mean_p, yerr=std_p,
                    color=colors[m], marker='o', linewidth=2,
                    markersize=7, capsize=4, label=m)

    ax.set_xlabel('Number of LR Frames', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'PSNR vs Number of Frames (reg_error={reg_error})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'frames_sweep.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Frames sweep complete.")


def main():
    set_global_seed(42)
    scale = 2
    mu = 0.01
    n_iters = 10
    n_trials = 3

    error_levels = [0.0, 0.25, 0.5, 1.0, 2.0]
    frame_counts = [2, 3, 4, 6, 8]

    # Load test images
    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    all_imgs = sorted([
        f for f in os.listdir(hr_dir) if f.lower().endswith('.png')
    ])
    # Use Set5 images for testing (indices 120-124 approx)
    test_names = [f for f in all_imgs if f.startswith('set5_')][:5]
    if not test_names:
        test_names = all_imgs[-5:]

    test_images = []
    for name in test_names:
        hr = load_image(os.path.join(hr_dir, name), color='gray')
        H, W = hr.shape
        H = (H // scale) * scale
        W = (W // scale) * scale
        # Resize to manageable size
        if max(H, W) > 256:
            import cv2
            ratio = 256 / max(H, W)
            hr = cv2.resize(hr, (int(W * ratio) // scale * scale,
                                  int(H * ratio) // scale * scale),
                             interpolation=cv2.INTER_AREA)
        else:
            hr = hr[:H, :W]
        test_images.append(hr)

    print(f"  Using {len(test_images)} test images")

    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'sr_experiments'))

    run_error_sweep(test_images, scale, 4, mu, n_iters,
                    error_levels, n_trials, results_dir)
    run_frames_sweep(test_images, scale, mu, n_iters,
                     frame_counts, n_trials, results_dir)

    print(f"\n  [OK] All SR experiment results saved to {results_dir}/")


if __name__ == '__main__':
    print("\n=== SR Experiments (Baselines) ===")
    main()
