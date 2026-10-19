"""
PCA Transform for Artifact Detection Features.

Fits PCA on the union of error-free and erroneous feature vectors
from training images, then transforms features and drops the last
6 components to give 44-dimensional feature vectors.

Paper reference: Section IV.A — "PCA transform P (fit on error-free
and erroneous feature vectors), drop the last 6 coefficients,
giving 44 features."
"""

import numpy as np
import os
import json


class PCATransform:
    """PCA dimensionality reduction for artifact detection features.

    Fits on raw 50-dim features, projects to 50 PCA components,
    then drops the last `drop_last_n` components.
    """

    def __init__(self, n_components=50, drop_last_n=6):
        """
        Args:
            n_components: Number of PCA components to compute (should match
                          raw feature dimensionality, i.e. 50).
            drop_last_n:  Number of trailing components to drop (paper: 6).
        """
        self.n_components = n_components
        self.drop_last_n = drop_last_n
        self.final_dim = n_components - drop_last_n  # 44

        # Fitted parameters
        self.mean_ = None        # (n_components,) mean of training features
        self.components_ = None  # (n_components, n_components) eigenvectors
        self.eigenvalues_ = None  # (n_components,) eigenvalues (descending)
        self.is_fitted = False

    def fit(self, X):
        """Fit PCA on training features.

        Args:
            X: (N, 50) float32/64 feature matrix.
                Should be the union of error-free and erroneous features.
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        assert D == self.n_components, (
            f"Expected {self.n_components} features, got {D}"
        )

        # Centre the data
        self.mean_ = X.mean(axis=0)
        X_centred = X - self.mean_

        # Covariance matrix
        cov = np.cov(X_centred, rowvar=False)  # (D, D)

        # Eigendecomposition (sorted descending)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order, flip to descending
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx].astype(np.float64)
        self.components_ = eigenvectors[:, idx].T.astype(np.float64)  # (D, D)

        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform features using fitted PCA, dropping the last n components.

        Args:
            X: (N, 50) feature matrix.

        Returns:
            (N, 44) transformed feature matrix.
        """
        assert self.is_fitted, "PCA not fitted yet. Call fit() first."
        X = np.asarray(X, dtype=np.float64)
        X_centred = X - self.mean_

        # Project onto all components
        X_pca = X_centred @ self.components_.T  # (N, 50)

        # Drop the last `drop_last_n` components
        X_reduced = X_pca[:, :self.final_dim]  # (N, 44)

        return X_reduced.astype(np.float32)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        """Save PCA parameters to disk."""
        assert self.is_fitted, "PCA not fitted yet."
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(
            path,
            mean=self.mean_,
            components=self.components_,
            eigenvalues=self.eigenvalues_,
            n_components=self.n_components,
            drop_last_n=self.drop_last_n,
        )

    def load(self, path):
        """Load PCA parameters from disk."""
        data = np.load(path)
        self.mean_ = data['mean']
        self.components_ = data['components']
        self.eigenvalues_ = data['eigenvalues']
        self.n_components = int(data['n_components'])
        self.drop_last_n = int(data['drop_last_n'])
        self.final_dim = self.n_components - self.drop_last_n
        self.is_fitted = True
        return self

    @property
    def explained_variance_ratio(self):
        """Fraction of variance explained by each component."""
        if not self.is_fitted:
            return None
        total = self.eigenvalues_.sum()
        if total == 0:
            return np.zeros_like(self.eigenvalues_)
        return self.eigenvalues_ / total
