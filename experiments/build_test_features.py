"""
Extract test set features from held-out images (beyond the 42 training images).

The 42 training images were all consumed by train+dev.
This script extracts features from the remaining images in data/hr/
to create a proper held-out test set for classifier evaluation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_global_seed, ensure_dir
from src.training_data import build_training_data
from src.pca_transform import PCATransform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    set_global_seed(42)

    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    all_imgs = sorted([
        f for f in os.listdir(hr_dir) if f.lower().endswith('.png')
    ])

    # Test images: everything after the first 42 training images
    test_names = all_imgs[42:]
    if not test_names:
        print("  No test images found beyond the 42 training images!")
        print("  Add more images to data/hr/ (e.g., from BSD100 or Set5/Set14).")
        return

    # Take up to 14 images for test
    test_names = test_names[:14]
    test_paths = [os.path.join(hr_dir, n) for n in test_names]
    print(f"  Test images ({len(test_names)}): {test_names}")

    # Build features using the same protocol as training
    X_clean, X_error, y_clean, y_error, ids_clean, ids_error = \
        build_training_data(test_paths, scale=2, mu=0.1, phase1_iters=3,
                            seed=99)  # Different seed from training

    # Combine
    X_raw = np.concatenate([X_clean, X_error], axis=0)
    y = np.concatenate([y_clean, y_error], axis=0)
    ids = np.concatenate([ids_clean, ids_error], axis=0)

    # PCA transform (using pre-fitted PCA from training)
    pca = PCATransform()
    pca.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'pca.npz'))
    X = pca.transform(X_raw)

    # Save
    data_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'checkpoints', 'data'))
    np.savez_compressed(
        os.path.join(data_dir, 'test.npz'),
        X=X, y=y, image_ids=ids
    )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\n  Test set saved:")
    print(f"    Clean (y=1): {n_pos}")
    print(f"    Error (y=0): {n_neg}")
    print(f"    Total: {len(y)}")
    print(f"    Features: {X.shape[1]}D (PCA-transformed)")


if __name__ == '__main__':
    print("\n=== Extracting Test Features ===")
    main()
