"""
Experiment: Weighted SR with trained classifiers.

THE CORE EXPERIMENT: Does the classifier's weight map actually improve SR
compared to unweighted SR, especially under registration errors?

For each test image and error level:
  1. Generate LR frames with known shifts + registration error
  2. Run Phase 1 SR (unweighted, 3 iters) for feature extraction
  3. Extract features, PCA transform, run each classifier -> weight maps M_i
  4. Run Phase 2 SR (weighted, N iters) with those weight maps
  5. Compare PSNR/SSIM against baselines

Baselines:
  - Bicubic (single frame)
  - Unweighted SR (all M_i = 1)
  - Oracle weights (from ground truth)

Methods:
  - Weighted SR with Logistic Regression
  - Weighted SR with Kernel SVM
  - Weighted SR with GDA
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (set_global_seed, load_image, save_image,
                        compute_psnr, compute_ssim, ensure_dir,
                        create_comparison_figure)
from src.observation_model import (
    apply_shift, downsample_nonoverlapping_avg, bicubic_upsample,
)
from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted
from src.feature_extraction import extract_features_at_positions
from src.pca_transform import PCATransform
from src.classifiers import (
    LogisticRegressionClassifier,
    KernelSVMClassifier,
    GDAClassifier,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_lr_sequence(hr, scale, n_frames, reg_error, rng):
    """Generate LR frames with registration error."""
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

    noisy_shifts_lr = true_shifts_lr.copy()
    if reg_error > 0:
        errors = rng.randn(n_frames, 2).astype(np.float32) * reg_error
        errors[0] = 0
        noisy_shifts_lr = true_shifts_lr + errors

    return lr_frames, true_shifts_lr, noisy_shifts_lr


def compute_oracle_weights(hr, lr_frames, true_shifts, noisy_shifts,
                           scale, mu, n_iters):
    """Oracle: use GT to compute ideal per-pixel weights.

    For each frame, compare the per-pixel error from using noisy vs true shifts.
    Pixels where noisy registration causes large SR error get low weights.
    Uses a soft sigmoid-based weighting instead of hard threshold.
    """
    H = lr_frames[0].shape[0] * scale
    W = lr_frames[0].shape[1] * scale

    weight_maps = []
    for i in range(len(lr_frames)):
        if i == 0:
            weight_maps.append(np.ones((H, W), dtype=np.float32))
            continue

        # SR with true shifts
        lr_pair = [lr_frames[0], lr_frames[i]]
        shifts_true = np.array([true_shifts[0], true_shifts[i]])
        sr_true = sr_phase1(lr_pair, shifts_true, scale, mu=mu,
                            num_iterations=n_iters)

        # SR with noisy shifts
        shifts_noisy = np.array([noisy_shifts[0], noisy_shifts[i]])
        sr_noisy = sr_phase1(lr_pair, shifts_noisy, scale, mu=mu,
                             num_iterations=n_iters)

        # Soft oracle: sigmoid of negative error magnitude
        error = np.abs(sr_true[:H, :W] - sr_noisy[:H, :W])
        # Scale: when error > median, weight approaches 0
        median_err = np.median(error[error > 0]) if np.any(error > 0) else 0.01
        k = 5.0 / max(median_err, 1e-6)  # Steepness
        M_i = 1.0 / (1.0 + np.exp(k * (error - median_err)))
        M_i = M_i.astype(np.float32)

        # Ensure full HR size
        if M_i.shape != (H, W):
            M_i = M_i[:H, :W]

        weight_maps.append(M_i)

    return weight_maps


def compute_classifier_weights(lr_frames, noisy_shifts, scale, mu,
                                phase1_iters, classifier, pca):
    """Compute weight maps using a trained classifier.

    For each neighbouring frame:
    1. Run Phase 1 SR with ref + frame i
    2. Extract features at subsampled positions (every 4 pixels for speed)
    3. PCA transform
    4. Classifier predict_proba -> weight map
    """
    import cv2

    n_frames = len(lr_frames)
    H = lr_frames[0].shape[0] * scale
    W = lr_frames[0].shape[1] * scale
    bic_up = bicubic_upsample(lr_frames[0], scale)

    weight_maps = []
    subsample = 4  # Compute every 4th pixel, interpolate

    for i in range(n_frames):
        if i == 0:
            weight_maps.append(np.ones((H, W), dtype=np.float32))
            continue

        # Phase 1 SR for feature extraction
        lr_pair = [lr_frames[0], lr_frames[i]]
        shifts_pair = np.array([noisy_shifts[0], noisy_shifts[i]])
        sr_image = sr_phase1(lr_pair, shifts_pair, scale, mu=mu,
                             num_iterations=phase1_iters)

        # Subsample positions for speed
        rows = np.arange(0, H, subsample)
        cols = np.arange(0, W, subsample)
        positions = np.array([(r, c) for r in rows for c in cols])

        # Extract features
        features_raw = extract_features_at_positions(
            sr_image[:H, :W], bic_up[:H, :W], positions
        )

        # PCA transform
        features_pca = pca.transform(features_raw)

        # Classifier prediction
        probs = classifier.predict_proba(features_pca)

        # Reshape to subsampled grid and upscale
        h_sub = len(rows)
        w_sub = len(cols)
        M_sub = probs.reshape(h_sub, w_sub)
        M_i = cv2.resize(M_sub, (W, H), interpolation=cv2.INTER_LINEAR)
        M_i = np.clip(M_i, 0, 1).astype(np.float32)
        weight_maps.append(M_i)

    return weight_maps


def run_weighted_sr_experiment():
    """Main experiment: weighted SR with all classifiers."""
    set_global_seed(42)

    scale = 2
    n_frames = 4
    mu = 0.1           # TUNED on dev set
    phase1_iters = 3
    phase2_iters = 30  # TUNED on dev set
    n_trials = 3
    error_levels = [0.0, 0.25, 0.5, 1.0, 2.0]

    # Load PCA
    pca = PCATransform()
    pca_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'pca.npz')
    pca.load(pca_path)

    # Load classifiers
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'classifiers')
    classifiers = []

    lr_clf = LogisticRegressionClassifier()
    lr_clf.load(os.path.join(ckpt_dir, 'LogisticRegression.npz'))
    classifiers.append(('LogReg', lr_clf))

    try:
        svm_clf = KernelSVMClassifier()
        svm_clf.load(os.path.join(ckpt_dir,
                     'KernelSVM(C=1.0, gamma=scale).joblib'))
        classifiers.append(('SVM', svm_clf))
    except Exception as e:
        print(f"  Warning: Could not load SVM: {e}")

    gda_clf = GDAClassifier()
    gda_clf.load(os.path.join(ckpt_dir, 'GDA(shared_cov).npz'))
    classifiers.append(('GDA', gda_clf))

    # Load test images
    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    all_imgs = sorted([
        f for f in os.listdir(hr_dir) if f.lower().endswith('.png')
    ])
    # Use Set5 images or last 5 images as test
    test_names = [f for f in all_imgs if f.startswith('set5_')][:3]
    if not test_names:
        test_names = all_imgs[-3:]

    import cv2
    test_images = []
    test_image_names = []
    for name in test_names:
        hr = load_image(os.path.join(hr_dir, name), color='gray')
        H, W = hr.shape
        H = (H // scale) * scale
        W = (W // scale) * scale
        # Resize large images for tractability
        if max(H, W) > 128:
            ratio = 128 / max(H, W)
            new_H = int(H * ratio) // scale * scale
            new_W = int(W * ratio) // scale * scale
            hr = cv2.resize(hr, (new_W, new_H), interpolation=cv2.INTER_AREA)
        else:
            hr = hr[:H, :W]
        test_images.append(hr)
        test_image_names.append(name)

    print(f"  Test images: {len(test_images)} ({[n for n in test_image_names]})")
    print(f"  Image sizes: {[img.shape for img in test_images]}")

    # Method names
    method_names = ['Bicubic', 'Unweighted SR', 'Oracle']
    method_names += [f'Weighted-{name}' for name, _ in classifiers]

    # Results storage
    all_psnr = {m: {e: [] for e in error_levels} for m in method_names}
    all_ssim = {m: {e: [] for e in error_levels} for m in method_names}

    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results',
                                          'weighted_sr'))

    for trial in range(n_trials):
        print(f"\n  Trial {trial+1}/{n_trials}")
        rng = np.random.RandomState(42 + trial)

        for img_idx, hr in enumerate(test_images):
            for err in error_levels:
                H, W = hr.shape
                lr_frames, true_sh, noisy_sh = generate_lr_sequence(
                    hr, scale, n_frames, reg_error=err, rng=rng)

                # --- Bicubic ---
                bic = bicubic_upsample(lr_frames[0], scale)[:H, :W]
                all_psnr['Bicubic'][err].append(compute_psnr(bic, hr))
                all_ssim['Bicubic'][err].append(compute_ssim(bic, hr))

                # --- Unweighted SR ---
                sr_uw = sr_unweighted(lr_frames, noisy_sh, scale,
                                      mu=mu, num_iterations=phase2_iters)
                sr_uw = sr_uw[:H, :W]
                all_psnr['Unweighted SR'][err].append(compute_psnr(sr_uw, hr))
                all_ssim['Unweighted SR'][err].append(compute_ssim(sr_uw, hr))

                # --- Oracle ---
                oracle_maps = compute_oracle_weights(
                    hr, lr_frames, true_sh, noisy_sh, scale, mu, phase1_iters)
                sr_oracle = sr_phase2(lr_frames, noisy_sh, oracle_maps, scale,
                                      mu=mu, num_iterations=phase2_iters)
                sr_oracle = sr_oracle[:H, :W]
                all_psnr['Oracle'][err].append(compute_psnr(sr_oracle, hr))
                all_ssim['Oracle'][err].append(compute_ssim(sr_oracle, hr))

                # --- Classifier-weighted SR ---
                for clf_name, clf in classifiers:
                    method = f'Weighted-{clf_name}'
                    clf_maps = compute_classifier_weights(
                        lr_frames, noisy_sh, scale, mu, phase1_iters,
                        clf, pca)
                    sr_clf = sr_phase2(lr_frames, noisy_sh, clf_maps, scale,
                                       mu=mu, num_iterations=phase2_iters)
                    sr_clf = sr_clf[:H, :W]
                    all_psnr[method][err].append(compute_psnr(sr_clf, hr))
                    all_ssim[method][err].append(compute_ssim(sr_clf, hr))

                print(f"    img={img_idx} err={err:.2f}: "
                      f"bic={all_psnr['Bicubic'][err][-1]:.2f} "
                      f"uw={all_psnr['Unweighted SR'][err][-1]:.2f} "
                      f"oracle={all_psnr['Oracle'][err][-1]:.2f} "
                      + " ".join(f"{n}={all_psnr[f'Weighted-{n}'][err][-1]:.2f}"
                                 for n, _ in classifiers))

    # === PLOTS ===
    colors = {
        'Bicubic': '#95a5a6', 'Unweighted SR': '#e74c3c',
        'Oracle': '#27ae60',
        'Weighted-LogReg': '#3498db',
        'Weighted-SVM': '#9b59b6',
        'Weighted-GDA': '#f39c12',
    }
    markers = {
        'Bicubic': 'o', 'Unweighted SR': 's', 'Oracle': 'D',
        'Weighted-LogReg': '^', 'Weighted-SVM': 'v', 'Weighted-GDA': 'P',
    }

    # PSNR vs error level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for m in method_names:
        mean_p = [np.mean(all_psnr[m][e]) for e in error_levels]
        std_p = [np.std(all_psnr[m][e]) for e in error_levels]
        ax1.errorbar(error_levels, mean_p, yerr=std_p,
                    color=colors.get(m, '#333333'),
                    marker=markers.get(m, 'o'),
                    linewidth=2, markersize=7, capsize=4, label=m)

        mean_s = [np.mean(all_ssim[m][e]) for e in error_levels]
        std_s = [np.std(all_ssim[m][e]) for e in error_levels]
        ax2.errorbar(error_levels, mean_s, yerr=std_s,
                    color=colors.get(m, '#333333'),
                    marker=markers.get(m, 'o'),
                    linewidth=2, markersize=7, capsize=4, label=m)

    ax1.set_xlabel('Registration Error (px)', fontsize=13)
    ax1.set_ylabel('PSNR (dB)', fontsize=13)
    ax1.set_title('PSNR vs Registration Error', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Registration Error (px)', fontsize=13)
    ax2.set_ylabel('SSIM', fontsize=13)
    ax2.set_title('SSIM vs Registration Error', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('SR Robustness: Classifier-Weighted vs Baselines',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'weighted_sr_robustness.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    csv_path = os.path.join(results_dir, 'weighted_sr_results.csv')
    with open(csv_path, 'w') as f:
        f.write('error_level,method,mean_psnr,std_psnr,mean_ssim,std_ssim\n')
        for e in error_levels:
            for m in method_names:
                mp = np.mean(all_psnr[m][e])
                sp = np.std(all_psnr[m][e])
                ms = np.mean(all_ssim[m][e])
                ss = np.std(all_ssim[m][e])
                f.write(f'{e},{m},{mp:.4f},{sp:.4f},{ms:.6f},{ss:.6f}\n')

    # Print summary table
    print("\n  === WEIGHTED SR RESULTS (mean +/- std) ===")
    print(f"  {'Method':<20}", end='')
    for e in error_levels:
        print(f"  err={e:<5}", end='')
    print()
    print(f"  {'-'*80}")
    for m in method_names:
        print(f"  {m:<20}", end='')
        for e in error_levels:
            mp = np.mean(all_psnr[m][e])
            print(f"  {mp:<6.2f}", end='')
        print()

    print(f"\n  Results saved to {results_dir}/")


if __name__ == '__main__':
    print("\n=== Weighted SR Experiment (Core Result) ===")
    run_weighted_sr_experiment()
