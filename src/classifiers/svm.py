"""
Kernel SVM Classifier for Artifact Detection.

Uses a Gaussian (RBF) kernel SVM with Platt scaling to obtain
probability outputs in [0, 1] for the weight maps.

This is one of our project contributions — comparing against
the paper's logistic regression baseline.

Hyperparameters C and gamma are tuned via grouped k-fold CV.
"""

import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.classifiers.base import ClassifierBase


class KernelSVMClassifier(ClassifierBase):
    """RBF Kernel SVM with Platt scaling for probability output.

    Wraps sklearn's SVC with probability=True.
    """

    def __init__(self, C=1.0, gamma='scale', kernel='rbf'):
        """
        Args:
            C: Regularisation parameter (tuned via CV).
            gamma: RBF kernel coefficient ('scale', 'auto', or float).
            kernel: Kernel type (default 'rbf' = Gaussian).
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

        self.scaler_ = StandardScaler()
        self.svm_ = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return f"KernelSVM(C={self.C}, gamma={self.gamma})"

    @property
    def param_count(self) -> int:
        """Number of support vectors (analogous to parameter count)."""
        if self.svm_ is not None and hasattr(self.svm_, 'n_support_'):
            return int(np.sum(self.svm_.n_support_))
        return 0

    def fit(self, X_train, y_train):
        """Train the SVM.

        Features are standardised (zero mean, unit variance) before
        training for numerical stability.

        Args:
            X_train: (N, D) float features.
            y_train: (N,) int labels (0 or 1).
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int32)

        # Standardise features
        X_scaled = self.scaler_.fit_transform(X)

        # Train SVM with Platt scaling (probability=True)
        self.svm_ = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            probability=True,   # Enables Platt scaling
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            max_iter=10000,
        )
        self.svm_.fit(X_scaled, y)
        self._is_fitted = True

    def predict_proba(self, X):
        """Return P(y=1 | x) via Platt scaling."""
        assert self._is_fitted, "Classifier not fitted. Call fit() first."
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X)
        # svm.predict_proba returns (N, 2): [P(y=0), P(y=1)]
        proba = self.svm_.predict_proba(X_scaled)
        # Column index for class 1
        idx = list(self.svm_.classes_).index(1)
        return proba[:, idx].astype(np.float32)

    def save(self, path):
        """Save SVM model and scaler."""
        assert self._is_fitted, "Classifier not fitted."
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'svm': self.svm_,
            'scaler': self.scaler_,
            'C': self.C,
            'gamma': self.gamma,
            'kernel': self.kernel,
        }, path)

    def load(self, path):
        """Load SVM model from disk."""
        data = joblib.load(path)
        self.svm_ = data['svm']
        self.scaler_ = data['scaler']
        self.C = data['C']
        self.gamma = data['gamma']
        self.kernel = data['kernel']
        self._is_fitted = True
        return self
