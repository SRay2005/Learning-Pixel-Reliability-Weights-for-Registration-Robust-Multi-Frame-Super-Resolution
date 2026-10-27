"""
Experiment: Lambda > 0 regularisation for Phase 2 SR.

The paper uses lambda=0 (no Tikhonov regularisation).
This experiment tests whether adding Laplacian regularisation (lambda > 0)
improves SR quality under registration errors.

Tests: lambda in [0, 0.001, 0.01, 0.1, 1.0] at several error levels.
Uses the best classifier (SVM) for weight maps.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (set_global_seed, load_image, compute_psnr,
                        compute_ssim, ensure_dir)
from src.observation_model import (
    apply_shift, downsample_nonoverlapping_avg, bicubic_upsample,
)
from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_lr_sequence(hr, scale, n_frames, reg_error, rng):
    """Generate LR frames with registration error."""
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
    noisy_shifts_lr = true_shifts_lr.copy()
    if reg_error > 0:
        errors = rng.randn(n_frames, 2).astype(np.float32) * reg_error
        errors[0] = 0
        noisy_shifts_lr = true_shifts_lr + errors

    return lr_frames, true_shifts_lr, noisy_shifts_lr


def main():
    set_global_seed(42)

    scale = 2
    n_frames = 4
    mu = 0.1
    phase2_iters = 30
    n_trials = 5

    lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    error_levels = [0.0, 0.5, 1.0, 2.0]

    # Load test images
    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    all_imgs = sorted([
        f for f in os.listdir(hr_dir) if f.lower().endswith('.png')
    ])
    test_names = [f for f in all_imgs if f.startswith('set5_')][:3]
    if not test_names:
        test_names = all_imgs[-3:]

    import cv2
    test_images = []
    for name in test_names:
        hr = load_image(os.path.join(hr_dir, name), color='gray')
        H, W = hr.shape
        H = (H // scale) * scale
        W = (W // scale) * scale
        if max(H, W) > 128:
            ratio = 128 / max(H, W)
            new_H = int(H * ratio) // scale * scale
            new_W = int(W * ratio) // scale * scale
            hr = cv2.resize(hr, (new_W, new_H), interpolation=cv2.INTER_AREA)
        else:
            hr = hr[:H, :W]
        test_images.append(hr)

    print(f"  Test images: {len(test_images)}")

    # Results: lambda -> error -> list of PSNR
    all_psnr = {lam: {e: [] for e in error_levels} for lam in lambda_values}

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        for hr in test_images:
            for err in error_levels:
                H, W = hr.shape
                lr_frames, _, noisy_sh = generate_lr_sequence(
                    hr, scale, n_frames, reg_error=err, rng=rng)

                for lam in lambda_values:
                    sr = sr_unweighted(lr_frames, noisy_sh, scale,
                                       mu=mu, num_iterations=phase2_iters,
                                       lambda_reg=lam)
                    psnr = compute_psnr(sr[:H, :W], hr)
                    all_psnr[lam][err].append(psnr)

        print(f"  Trial {trial+1}/{n_trials} done")

    # Plot: PSNR vs lambda for each error level
    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'lambda_reg'))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#27ae60', '#3498db', '#e74c3c', '#9b59b6']

    for idx, err in enumerate(error_levels):
        mean_p = [np.mean(all_psnr[lam][err]) for lam in lambda_values]
        std_p = [np.std(all_psnr[lam][err]) for lam in lambda_values]
        # Use log scale for x, but handle lambda=0
        x_pos = list(range(len(lambda_values)))
        ax.errorbar(x_pos, mean_p, yerr=std_p,
                    color=colors[idx], marker='o', linewidth=2,
                    markersize=7, capsize=4, label=f'err={err} px')

    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels([str(l) for l in lambda_values])
    ax.set_xlabel('Lambda (regularisation)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Effect of Tikhonov Regularisation (lambda > 0)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lambda_regularisation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    with open(os.path.join(results_dir, 'lambda_results.csv'), 'w') as f:
        f.write('lambda,error_level,mean_psnr,std_psnr\n')
        for lam in lambda_values:
            for err in error_levels:
                mp = np.mean(all_psnr[lam][err])
                sp = np.std(all_psnr[lam][err])
                f.write(f'{lam},{err},{mp:.4f},{sp:.4f}\n')

    # Summary
    print("\n  === Lambda Regularisation Results ===")
    print(f"  {'Lambda':<10}", end='')
    for err in error_levels:
        print(f"  err={err:<5}", end='')
    print()
    print(f"  {'-'*50}")
    for lam in lambda_values:
        print(f"  {lam:<10}", end='')
        for err in error_levels:
            mp = np.mean(all_psnr[lam][err])
            print(f"  {mp:<6.2f}", end='')
        print()

    print(f"\n  Results saved to {results_dir}/")


if __name__ == '__main__':
    print("\n=== Lambda Regularisation Experiment ===")
    main()
