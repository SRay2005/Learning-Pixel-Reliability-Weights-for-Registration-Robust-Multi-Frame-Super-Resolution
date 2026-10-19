"""
Artifact Detector — orchestrates the full weight-map pipeline.

Given LR frames and their estimated shifts, produces per-pixel
weight maps M_i ∈ [0,1] for each frame by:

1. Running Phase 1 SR (unweighted) for each frame pair
2. Computing bicubic upscaling of the reference LR frame
3. Extracting 50-dim features per pixel
4. Applying PCA → 44 features
5. Running the trained classifier to get P(artifact-free | pixel)
6. Reshaping to produce weight maps M_i

Paper reference: Section III (Algorithm Overview) and Section IV.
"""

import numpy as np
from src.observation_model import bicubic_upsample
from src.sr_solver import sr_phase1
from src.feature_extraction import extract_full_weight_map
from src.pca_transform import PCATransform
from src.classifiers.base import ClassifierBase


class ArtifactDetector:
    """Full artifact detection pipeline: features → PCA → classifier → weight map."""

    def __init__(self, classifier, pca_transform, scale=2,
                 mu=0.01, phase1_iters=3):
        """
        Args:
            classifier: A fitted ClassifierBase instance.
            pca_transform: A fitted PCATransform instance.
            scale: SR scale factor.
            mu: Phase 1 step size.
            phase1_iters: Phase 1 iterations (paper: 3).
        """
        self.classifier = classifier
        self.pca = pca_transform
        self.scale = scale
        self.mu = mu
        self.phase1_iters = phase1_iters

    def compute_weight_maps(self, lr_images, shifts):
        """Compute per-pixel reliability weight maps for all frames.

        For each neighbouring frame g_i (i != 0), we compute:
        1. Phase 1 SR using g_c (ref) and g_i
        2. Features at every HR pixel
        3. PCA transform
        4. Classifier probability → M_i

        The reference frame (i=0) gets weight map = 1 everywhere
        (it is always "correctly registered" to itself).

        Args:
            lr_images: list of (h, w) float32 LR frames.
            shifts: (N, 2) array of shifts in LR domain.

        Returns:
            weight_maps: list of (H, W) float32 arrays, one per frame.
        """
        n_frames = len(lr_images)
        H = lr_images[0].shape[0] * self.scale
        W = lr_images[0].shape[1] * self.scale

        # Bicubic upscale of reference frame
        bic_up = bicubic_upsample(lr_images[0], self.scale)

        weight_maps = []

        for i in range(n_frames):
            if i == 0:
                # Reference frame: fully reliable
                weight_maps.append(np.ones((H, W), dtype=np.float32))
                continue

            # Phase 1 SR using ref and frame i
            lr_pair = [lr_images[0], lr_images[i]]
            shifts_pair = np.array([shifts[0], shifts[i]], dtype=np.float32)

            sr_image = sr_phase1(
                lr_pair, shifts_pair, self.scale,
                mu=self.mu, num_iterations=self.phase1_iters,
            )

            # Extract features for all pixels
            features_raw, shape = extract_full_weight_map(sr_image, bic_up)

            # PCA transform
            features_pca = self.pca.transform(features_raw)

            # Classifier prediction
            probs = self.classifier.predict_proba(features_pca)

            # Reshape to weight map
            M_i = probs.reshape(shape)
            weight_maps.append(M_i)

        return weight_maps

    def compute_weight_maps_fast(self, lr_images, shifts, subsample=4):
        """Faster weight map computation with spatial subsampling + interpolation.

        For large images, computing features at every pixel is expensive.
        This version computes on a subsampled grid and interpolates.

        Args:
            lr_images: list of LR frames.
            shifts: (N, 2) shifts.
            subsample: Compute features every `subsample` pixels.

        Returns:
            weight_maps: list of (H, W) float32 arrays.
        """
        import cv2

        n_frames = len(lr_images)
        H = lr_images[0].shape[0] * self.scale
        W = lr_images[0].shape[1] * self.scale

        bic_up = bicubic_upsample(lr_images[0], self.scale)

        weight_maps = []

        for i in range(n_frames):
            if i == 0:
                weight_maps.append(np.ones((H, W), dtype=np.float32))
                continue

            lr_pair = [lr_images[0], lr_images[i]]
            shifts_pair = np.array([shifts[0], shifts[i]], dtype=np.float32)

            sr_image = sr_phase1(
                lr_pair, shifts_pair, self.scale,
                mu=self.mu, num_iterations=self.phase1_iters,
            )

            # Subsample positions
            rows = np.arange(0, H, subsample)
            cols = np.arange(0, W, subsample)
            positions = np.array([(r, c) for r in rows for c in cols])

            from src.feature_extraction import extract_features_at_positions
            features_raw = extract_features_at_positions(
                sr_image, bic_up, positions
            )

            features_pca = self.pca.transform(features_raw)
            probs = self.classifier.predict_proba(features_pca)

            # Reshape to subsampled grid
            h_sub = len(rows)
            w_sub = len(cols)
            M_sub = probs.reshape(h_sub, w_sub)

            # Upsample to full resolution via bicubic interpolation
            M_i = cv2.resize(M_sub, (W, H), interpolation=cv2.INTER_LINEAR)
            M_i = np.clip(M_i, 0, 1).astype(np.float32)
            weight_maps.append(M_i)

        return weight_maps
