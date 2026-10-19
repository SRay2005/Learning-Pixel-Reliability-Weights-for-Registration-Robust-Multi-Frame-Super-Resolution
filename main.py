"""
Main entry point for the SR pipeline.

Usage:
    python main.py build_data     -- Generate training features from 42 images
    python main.py train          -- Train all classifiers on the training data
    python main.py evaluate       -- Run SR experiments and produce results
    python main.py demo           -- Run a single SR demo on a test image
    python main.py all            -- Run the full pipeline end-to-end
"""

import os
import sys
import argparse
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_global_seed, load_image, save_image, ensure_dir
from src.utils import compute_psnr, compute_ssim


def load_config(path=None):
    """Load configuration from YAML file."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_training_images(data_dir, n_images=42):
    """Get paths to training images."""
    hr_dir = os.path.join(data_dir, 'hr')
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"Data directory not found: {hr_dir}")

    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    paths = sorted([
        os.path.join(hr_dir, f)
        for f in os.listdir(hr_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if len(paths) < n_images:
        print(f"  Warning: requested {n_images} images but found {len(paths)}")

    return paths[:n_images]


def cmd_build_data(config):
    """Build training data: generate features from images."""
    print("\n" + "=" * 60)
    print("  STAGE 1: Building Training Data")
    print("=" * 60)

    from src.training_data import build_training_data
    from src.pca_transform import PCATransform
    from src.data_split import split_by_image

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    image_paths = get_training_images(data_dir, config['training']['num_images'])

    scale = config['scale_factor']
    mu = config['phase1']['mu']
    iters = config['phase1']['num_iterations']

    X_clean, X_error, y_clean, y_error, ids_clean, ids_error = build_training_data(
        image_paths, scale=scale, mu=mu, phase1_iters=iters,
        seed=config['seed'],
    )

    # Combine clean and erroneous data
    X = np.concatenate([X_clean, X_error], axis=0)
    y = np.concatenate([y_clean, y_error], axis=0)
    image_ids = np.concatenate([ids_clean, ids_error], axis=0)

    # Fit PCA on ALL features (clean + erroneous), as per paper
    print("\n  Fitting PCA...")
    pca = PCATransform(
        n_components=config['pca']['total_raw_features'],
        drop_last_n=config['pca']['drop_last_n'],
    )
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {X.shape[1]}D → {X_pca.shape[1]}D")
    print(f"  Variance explained by kept components: "
          f"{pca.explained_variance_ratio[:pca.final_dim].sum():.4f}")

    # Save PCA
    ckpt_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'checkpoints'))
    pca.save(os.path.join(ckpt_dir, 'pca.npz'))

    # Split by image
    splits = split_by_image(
        X_pca, y, image_ids,
        n_train_images=config['split']['num_train'],
        n_dev_images=config['split']['num_dev'],
        seed=config['seed'],
    )

    # Save splits
    data_out = ensure_dir(os.path.join(PROJECT_ROOT, 'checkpoints', 'data'))
    for split_name, split_data in splits.items():
        np.savez(
            os.path.join(data_out, f'{split_name}.npz'),
            X=split_data['X'],
            y=split_data['y'],
            image_ids=split_data['image_ids'],
        )

    print(f"\n  [OK] Training data saved to {data_out}/")
    return splits


def cmd_train(config):
    """Train all three classifiers."""
    print("\n" + "=" * 60)
    print("  STAGE 2: Training Classifiers")
    print("=" * 60)

    from src.classifiers import (
        LogisticRegressionClassifier,
        KernelSVMClassifier,
        GDAClassifier,
    )

    # Load training data
    data_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'data')
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    X_train, y_train = train_data['X'], train_data['y']

    dev_data = np.load(os.path.join(data_dir, 'dev.npz'))
    X_dev, y_dev = dev_data['X'], dev_data['y']

    ckpt_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'checkpoints', 'classifiers'))

    classifiers = [
        LogisticRegressionClassifier(
            learning_rate=config['logistic']['learning_rate'],
            max_iterations=config['logistic']['max_iterations'],
            l2_lambda=config['logistic']['l2_lambda'],
        ),
        KernelSVMClassifier(
            C=config['svm']['C'],
            gamma=config['svm']['gamma'],
        ),
        GDAClassifier(
            cov_reg_eps=config['gda']['cov_reg_eps'],
        ),
    ]

    for clf in classifiers:
        print(f"\n  Training {clf.name}...")
        train_time = clf.timed_fit(X_train, y_train)

        # Dev set evaluation
        probs_dev, pred_time = clf.timed_predict_proba(X_dev)
        preds_dev = (probs_dev >= 0.5).astype(int)
        acc = (preds_dev == y_dev).mean()

        print(f"    Train time:  {train_time:.2f}s")
        print(f"    Pred time:   {pred_time:.4f}s ({len(X_dev)} samples)")
        print(f"    Dev accuracy: {acc:.4f}")
        print(f"    Param count: {clf.param_count}")

        # Save
        ext = '.npz' if not isinstance(clf, KernelSVMClassifier) else '.joblib'
        clf.save(os.path.join(ckpt_dir, f'{clf.name}{ext}'))

    print(f"\n  [OK] Classifiers saved to {ckpt_dir}/")


def cmd_demo(config):
    """Run a simple SR demo on a test image."""
    print("\n" + "=" * 60)
    print("  DEMO: Single Image SR")
    print("=" * 60)

    from src.observation_model import bicubic_upsample, downsample_nonoverlapping_avg, apply_shift
    from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted
    from src.artifact_detector import ArtifactDetector
    from src.pca_transform import PCATransform
    from src.classifiers import LogisticRegressionClassifier

    # Load a test image
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    test_images = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith('.png')
    ])

    if not test_images:
        print("  No test images found!")
        return

    hr = load_image(test_images[0], color='gray')
    scale = config['scale_factor']
    H, W = hr.shape
    H = (H // scale) * scale
    W = (W // scale) * scale
    hr = hr[:H, :W]

    print(f"  Test image: {test_images[0]}")
    print(f"  HR size: {H}x{W}")

    # Generate synthetic LR frames
    rng = np.random.RandomState(config['seed'])
    n_frames = 4
    lr_frames = []
    shifts = np.zeros((n_frames, 2), dtype=np.float32)

    for i in range(n_frames):
        if i > 0:
            dy = rng.uniform(-1.0, 1.0)
            dx = rng.uniform(-1.0, 1.0)
            shifts[i] = [dy / scale, dx / scale]
            shifted_hr = apply_shift(hr, dy, dx)
        else:
            shifted_hr = hr
        lr = downsample_nonoverlapping_avg(shifted_hr, scale)
        lr_frames.append(lr)

    # Bicubic baseline
    bic = bicubic_upsample(lr_frames[0], scale)
    psnr_bic = compute_psnr(bic[:H, :W], hr)
    print(f"  Bicubic PSNR: {psnr_bic:.2f} dB")

    # Unweighted SR
    sr_uw = sr_unweighted(lr_frames, shifts, scale,
                          mu=config['phase2']['mu'],
                          num_iterations=config['phase2']['num_iterations'])
    psnr_uw = compute_psnr(sr_uw[:H, :W], hr)
    print(f"  Unweighted SR PSNR: {psnr_uw:.2f} dB")

    # Save results
    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results'))
    save_image(bic[:H, :W], os.path.join(results_dir, 'demo_bicubic.png'))
    save_image(sr_uw[:H, :W], os.path.join(results_dir, 'demo_unweighted_sr.png'))
    save_image(hr, os.path.join(results_dir, 'demo_groundtruth.png'))

    print(f"  [OK] Demo results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='ML-Based SR Robust to Registration Errors'
    )
    parser.add_argument(
        'command', type=str, nargs='?', default='demo',
        choices=['build_data', 'train', 'evaluate', 'demo', 'all'],
        help='Command to run',
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file',
    )

    args = parser.parse_args()
    config = load_config(args.config)

    set_global_seed(config['seed'])

    print("\n" + "+" + "=" * 58 + "+")
    print("|  Learning Pixel-Reliability Weights for                  |")
    print("|  Registration-Robust Multi-Frame Super-Resolution        |")
    print("+" + "=" * 58 + "+")
    print(f"  Command: {args.command}")
    print(f"  Seed: {config['seed']}")

    if args.command in ('build_data', 'all'):
        cmd_build_data(config)

    if args.command in ('train', 'all'):
        cmd_train(config)

    if args.command in ('demo', 'all'):
        cmd_demo(config)

    if args.command == 'evaluate':
        print("  [TODO] Full evaluation pipeline — Stage 4")

    print("\n  [DONE]")


if __name__ == '__main__':
    main()
