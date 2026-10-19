"""
Abstract base class for pluggable artifact-detection classifiers.

All classifiers must implement fit, predict_proba, save, and load
so they can be swapped interchangeably in the SR pipeline.
Features, training data, and splits are identical for all classifiers.
"""

from abc import ABC, abstractmethod
import numpy as np
import time


class ClassifierBase(ABC):
    """Base interface for artifact-detection classifiers.

    Convention:
        y = 1  → artifact-free pixel (reliable)
        y = 0  → artifact pixel (unreliable)
        predict_proba returns P(y=1 | x) for each sample.
    """

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the classifier.

        Args:
            X_train: (N, D) float array of feature vectors.
            y_train: (N,) int array of labels (0 or 1).
        """
        ...

    @abstractmethod
    def predict_proba(self, X):
        """Return probability of the positive class (artifact-free).

        Args:
            X: (N, D) float array of feature vectors.

        Returns:
            (N,) float array of P(y=1 | x) values in [0, 1].
        """
        ...

    def predict(self, X, threshold=0.5):
        """Hard classification at a given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @abstractmethod
    def save(self, path):
        """Persist trained model to disk."""
        ...

    @abstractmethod
    def load(self, path):
        """Load trained model from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable classifier name."""
        ...

    @property
    def param_count(self) -> int:
        """Number of trainable parameters (for cost reporting)."""
        return 0

    def timed_fit(self, X_train, y_train):
        """Fit with timing.  Returns training time in seconds."""
        t0 = time.time()
        self.fit(X_train, y_train)
        return time.time() - t0

    def timed_predict_proba(self, X):
        """Predict with timing.  Returns (probabilities, time_seconds)."""
        t0 = time.time()
        probs = self.predict_proba(X)
        return probs, time.time() - t0
