"""
Experiment: Tune mu (step size) and Phase 2 iterations on the dev set.

Finds stable convergence parameters for the SR solver using a grid search
over mu and num_iterations, evaluated by PSNR on synthetic dev images.

These values are then frozen and used identically for all three classifiers.

Output: best (mu, num_iterations) and a convergence plot.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_global_seed, load_image, compute_psnr, ensure_dir
from src.observation_model import (
    apply_shift, downsample_nonoverlapping_avg, bicubic_upsample,
)
from src.sr_solver import sr_phase1, sr_unweighted


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_synthetic_test(hr, scale, n_frames, rng):
    """Generate LR frames with known shifts from an HR image."""
    H, W = hr.shape
    lr_frames = []
    shifts_hr = np.zeros((n_frames, 2), dtype=np.float32)

    for i in range(n_frames):
        if i > 0:
            shifts_hr[i] = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
        shifted = apply_shift(hr, shifts_hr[i, 0], shifts_hr[i, 1])
        lr = downsample_nonoverlapping_avg(shifted, scale)
        lr_frames.append(lr)

    shifts_lr = shifts_hr / scale
    return lr_frames, shifts_lr


def tune_mu(config_path=None):
    """Grid-search mu and iterations on dev images."""
    set_global_seed(42)

    scale = 2
    n_frames = 4

    # Load a few dev images
    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    all_imgs = sorted([
        f for f in os.listdir(hr_dir) if f.lower().endswith('.png')
    ])
    # Use 5 images from the middle of the list (avoid first 42 = training)
    dev_imgs = all_imgs[42:47] if len(all_imgs) > 47 else all_imgs[-5:]
    print(f"  Using {len(dev_imgs)} dev images for tuning")

    mu_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    iter_values = [1, 3, 5, 10, 15, 20, 30]

    results = {}  # (mu, n_iter) -> mean PSNR

    for mu in mu_values:
        for n_iter in iter_values:
            psnrs = []
            for img_name in dev_imgs:
                hr = load_image(os.path.join(hr_dir, img_name), color='gray')
                H, W = hr.shape
                H = (H // scale) * scale
                W = (W // scale) * scale
                hr = hr[:H, :W]

                rng = np.random.RandomState(42)
                lr_frames, shifts_lr = make_synthetic_test(hr, scale, n_frames, rng)
                bic = bicubic_upsample(lr_frames[0], scale)

                sr = sr_unweighted(
                    lr_frames, shifts_lr, scale,
                    mu=mu, num_iterations=n_iter, lambda_reg=0.0,
                )
                psnr = compute_psnr(sr[:H, :W], hr)
                psnrs.append(psnr)

            mean_psnr = np.mean(psnrs)
            results[(mu, n_iter)] = mean_psnr

    # Find best
    best_key = max(results, key=results.get)
    best_mu, best_iter = best_key
    best_psnr = results[best_key]

    print(f"\n  Best: mu={best_mu}, iters={best_iter}, PSNR={best_psnr:.2f} dB")

    # Also get bicubic baseline for reference
    bic_psnrs = []
    for img_name in dev_imgs:
        hr = load_image(os.path.join(hr_dir, img_name), color='gray')
        H, W = hr.shape
        H = (H // scale) * scale
        W = (W // scale) * scale
        hr = hr[:H, :W]
        rng = np.random.RandomState(42)
        lr_frames, _ = make_synthetic_test(hr, scale, n_frames, rng)
        bic = bicubic_upsample(lr_frames[0], scale)
        bic_psnrs.append(compute_psnr(bic[:H, :W], hr))
    print(f"  Bicubic baseline: {np.mean(bic_psnrs):.2f} dB")

    # Plot: PSNR vs iterations for each mu
    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'tuning'))
    fig, ax = plt.subplots(figsize=(10, 6))
    for mu in mu_values:
        psnrs_line = [results[(mu, n)] for n in iter_values]
        ax.plot(iter_values, psnrs_line, 'o-', label=f'mu={mu}', linewidth=2)

    ax.axhline(y=np.mean(bic_psnrs), color='gray', linestyle='--',
               label='Bicubic', linewidth=1.5)
    ax.set_xlabel('Number of Iterations', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('SR Convergence: mu and Iteration Tuning', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mu_tuning.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {results_dir}/mu_tuning.png")

    # Save results as CSV
    with open(os.path.join(results_dir, 'mu_tuning.csv'), 'w') as f:
        f.write('mu,iterations,mean_psnr\n')
        for (mu, n_iter), psnr in sorted(results.items()):
            f.write(f'{mu},{n_iter},{psnr:.4f}\n')

    return best_mu, best_iter


if __name__ == '__main__':
    print("\n=== Tuning mu and iterations ===")
    best_mu, best_iter = tune_mu()
    print(f"\n  RESULT: Use mu={best_mu}, num_iterations={best_iter}")
    print("  Update configs/default.yaml with these values.\n")
