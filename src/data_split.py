"""
Data Splitting for Artifact Detection.

Splits training data BY IMAGE (never by pixel) to avoid leakage
between neighbouring pixels from the same image.

Provides:
  - Fixed train / dev / test split
  - Grouped k-fold CV on train+dev for hyperparameter selection
  - The test set is touched exactly once at the end

Paper note: the paper does not describe a dev/test protocol.
This is our methodological contribution for rigorous evaluation.
"""

import numpy as np
from sklearn.model_selection import GroupKFold


def split_by_image(X, y, image_ids, n_train_images=30, n_dev_images=12,
                   seed=42):
    """Split data into train / dev / test by source image ID.

    Args:
        X: (N, D) feature matrix.
        y: (N,) labels.
        image_ids: (N,) source image index per sample.
        n_train_images: Number of images for training.
        n_dev_images: Number of images for dev (validation).
        seed: Random seed for the image-level shuffle.

    Returns:
        Dictionary with keys 'train', 'dev', 'test', each containing:
            'X': feature matrix
            'y': labels
            'image_ids': image indices
    """
    rng = np.random.RandomState(seed)

    unique_images = np.unique(image_ids)
    n_total = len(unique_images)
    assert n_train_images + n_dev_images <= n_total, (
        f"Not enough images: {n_total} < {n_train_images} + {n_dev_images}"
    )

    # Shuffle images deterministically
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    train_images = set(shuffled[:n_train_images])
    dev_images = set(shuffled[n_train_images:n_train_images + n_dev_images])
    test_images = set(shuffled[n_train_images + n_dev_images:])

    def _mask(ids, image_set):
        return np.array([i in image_set for i in ids])

    train_mask = _mask(image_ids, train_images)
    dev_mask = _mask(image_ids, dev_images)
    test_mask = _mask(image_ids, test_images)

    splits = {
        'train': {
            'X': X[train_mask],
            'y': y[train_mask],
            'image_ids': image_ids[train_mask],
        },
        'dev': {
            'X': X[dev_mask],
            'y': y[dev_mask],
            'image_ids': image_ids[dev_mask],
        },
        'test': {
            'X': X[test_mask],
            'y': y[test_mask],
            'image_ids': image_ids[test_mask],
        },
    }

    print(f"  Data split (by image):")
    print(f"    Train: {train_mask.sum()} samples from {len(train_images)} images")
    print(f"    Dev:   {dev_mask.sum()} samples from {len(dev_images)} images")
    print(f"    Test:  {test_mask.sum()} samples from {len(test_images)} images")

    return splits


def grouped_kfold_cv(X, y, image_ids, n_folds=5):
    """Generate grouped k-fold CV splits for hyperparameter tuning.

    Groups are defined by image index — all pixels from the same image
    stay in the same fold.

    Args:
        X: (N, D) feature matrix (train + dev combined).
        y: (N,) labels.
        image_ids: (N,) source image index.
        n_folds: Number of folds.

    Yields:
        (X_train, y_train, X_val, y_val) for each fold.
    """
    gkf = GroupKFold(n_splits=n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=image_ids)):
        yield (
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
        )
