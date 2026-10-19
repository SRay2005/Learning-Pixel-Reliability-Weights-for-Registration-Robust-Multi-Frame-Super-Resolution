"""
Logistic Regression Classifier for Artifact Detection.

Implements the paper's baseline classifier:
    h(x) = 1 / (1 + exp(-theta^T x))
    trained by gradient ascent on the log-likelihood.

Also includes an L2-regularised variant (lambda tuned via CV).

Paper reference: Section IV.B.
Label convention: y=1 → artifact-free, y=0 → artifact.
"""

import numpy as np
import os
import json
from src.classifiers.base import ClassifierBase


class LogisticRegressionClassifier(ClassifierBase):
    """Paper's logistic regression with gradient-ascent training.

    This is a from-scratch implementation (not sklearn) to faithfully
    reproduce the paper's method.
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000,
                 convergence_tol=1e-6, l2_lambda=0.0):
        """
        Args:
            learning_rate: Gradient ascent step size.  OUR CHOICE.
            max_iterations: Maximum training iterations.
            convergence_tol: Stop when change in log-likelihood < tol.
            l2_lambda: L2 regularisation strength (0 = paper baseline).
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.l2_lambda = l2_lambda

        # Trained parameters
        self.theta_ = None  # (D+1,) including bias
        self.train_history_ = []  # log-likelihood per iteration
        self._is_fitted = False

    @property
    def name(self) -> str:
        if self.l2_lambda > 0:
            return f"LogisticRegression(L2={self.l2_lambda:.1e})"
        return "LogisticRegression"

    @property
    def param_count(self) -> int:
        return len(self.theta_) if self.theta_ is not None else 0

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X):
        """Prepend a column of ones for the bias term."""
        N = X.shape[0]
        return np.hstack([np.ones((N, 1), dtype=X.dtype), X])

    def fit(self, X_train, y_train):
        """Train by gradient ascent on the log-likelihood.

        Log-likelihood:
            L(theta) = sum_i [ y_i log h(x_i) + (1-y_i) log(1-h(x_i)) ]

        Gradient:
            dL/d(theta) = sum_i [ (y_i - h(x_i)) x_i ]

        With L2 regularisation:
            L_reg = L - (l2_lambda/2) ||theta||^2
            gradient_reg = gradient - l2_lambda * theta

        Args:
            X_train: (N, D) float features.
            y_train: (N,) int labels (0 or 1).
        """
        X = self._add_bias(np.asarray(X_train, dtype=np.float64))
        y = np.asarray(y_train, dtype=np.float64)
        N, D = X.shape

        # Initialise theta to zeros
        self.theta_ = np.zeros(D, dtype=np.float64)
        self.train_history_ = []

        prev_ll = -np.inf

        for iteration in range(self.max_iterations):
            # Forward pass
            z = X @ self.theta_
            h = self._sigmoid(z)

            # Log-likelihood
            eps = 1e-15
            ll = np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))

            # L2 regularisation penalty (don't regularise bias)
            if self.l2_lambda > 0:
                ll -= 0.5 * self.l2_lambda * np.sum(self.theta_[1:]**2)

            self.train_history_.append(float(ll))

            # Gradient ascent
            gradient = X.T @ (y - h)  # (D,)

            # L2 regularisation gradient (don't regularise bias)
            if self.l2_lambda > 0:
                reg_grad = self.l2_lambda * self.theta_.copy()
                reg_grad[0] = 0  # No regularisation on bias
                gradient -= reg_grad

            self.theta_ += self.learning_rate * gradient / N

            # Convergence check
            if abs(ll - prev_ll) < self.convergence_tol:
                break
            prev_ll = ll

        self._is_fitted = True

    def predict_proba(self, X):
        """Return P(y=1 | x) = sigmoid(theta^T x)."""
        assert self._is_fitted, "Classifier not fitted. Call fit() first."
        X = self._add_bias(np.asarray(X, dtype=np.float64))
        z = X @ self.theta_
        return self._sigmoid(z).astype(np.float32)

    def save(self, path):
        """Save theta and config to disk."""
        assert self._is_fitted, "Classifier not fitted."
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(
            path,
            theta=self.theta_,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            convergence_tol=self.convergence_tol,
            l2_lambda=self.l2_lambda,
        )

    def load(self, path):
        """Load trained parameters from disk."""
        data = np.load(path)
        self.theta_ = data['theta']
        self.learning_rate = float(data['learning_rate'])
        self.max_iterations = int(data['max_iterations'])
        self.convergence_tol = float(data['convergence_tol'])
        self.l2_lambda = float(data['l2_lambda'])
        self._is_fitted = True
        return self
