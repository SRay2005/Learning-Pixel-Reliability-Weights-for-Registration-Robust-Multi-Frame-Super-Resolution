"""
Gaussian Discriminant Analysis (GDA) Classifier for Artifact Detection.

Implements GDA with shared covariance (Linear Discriminant Analysis model):
    P(x | y=k) = N(x; mu_k, Sigma)  with shared Sigma across classes.

The posterior P(y=1 | x) is used as the weight map value.

This is one of our project contributions — a generative classifier
compared against the paper's discriminative logistic regression.

Also includes Gaussian Naive Bayes (diagonal covariance) as an
optional simpler variant.
"""

import numpy as np
import os
from src.classifiers.base import ClassifierBase


class GDAClassifier(ClassifierBase):
    """Gaussian Discriminant Analysis with shared covariance.

    Fits class-conditional Gaussians with a shared covariance matrix.
    The posterior is computed in closed form via Bayes' rule.

    If the covariance matrix is singular, regularisation (epsilon * I)
    is applied.
    """

    def __init__(self, cov_reg_eps=1e-6, naive_bayes=False):
        """
        Args:
            cov_reg_eps: Regularisation added to covariance diagonal
                         if it is singular or ill-conditioned.
            naive_bayes: If True, use diagonal covariance (Naive Bayes).
        """
        self.cov_reg_eps = cov_reg_eps
        self.naive_bayes = naive_bayes

        # Fitted parameters
        self.mu_0_ = None      # (D,) mean of class 0
        self.mu_1_ = None      # (D,) mean of class 1
        self.sigma_ = None     # (D, D) shared covariance
        self.sigma_inv_ = None  # (D, D) inverse covariance
        self.prior_1_ = None   # P(y=1)
        self.prior_0_ = None   # P(y=0)
        self.log_det_ = None   # log determinant of sigma
        self._is_fitted = False

    @property
    def name(self) -> str:
        if self.naive_bayes:
            return "GaussianNaiveBayes"
        return "GDA(shared_cov)"

    @property
    def param_count(self) -> int:
        if not self._is_fitted:
            return 0
        D = len(self.mu_0_)
        if self.naive_bayes:
            return 2 * D + D + 1  # two means + diagonal + prior
        return 2 * D + D * (D + 1) // 2 + 1  # two means + upper-tri cov + prior

    def fit(self, X_train, y_train):
        """Fit GDA by computing class means, shared covariance, and priors.

        Args:
            X_train: (N, D) float features.
            y_train: (N,) int labels (0 or 1).
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int32)
        N, D = X.shape

        # Class masks
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        n_0 = mask_0.sum()
        n_1 = mask_1.sum()

        assert n_0 > 0 and n_1 > 0, (
            f"Both classes must be present. Got {n_0} class-0, {n_1} class-1."
        )

        # Class means
        self.mu_0_ = X[mask_0].mean(axis=0)
        self.mu_1_ = X[mask_1].mean(axis=0)

        # Priors
        self.prior_0_ = n_0 / N
        self.prior_1_ = n_1 / N

        # Shared covariance
        X0_centred = X[mask_0] - self.mu_0_
        X1_centred = X[mask_1] - self.mu_1_

        if self.naive_bayes:
            # Diagonal covariance (Naive Bayes assumption)
            var_0 = np.var(X0_centred, axis=0)
            var_1 = np.var(X1_centred, axis=0)
            # Pooled diagonal variance
            pooled_var = (n_0 * var_0 + n_1 * var_1) / N
            pooled_var = np.maximum(pooled_var, self.cov_reg_eps)
            self.sigma_ = np.diag(pooled_var)
            self.sigma_inv_ = np.diag(1.0 / pooled_var)
            self.log_det_ = np.sum(np.log(pooled_var))
        else:
            # Full shared covariance
            S_0 = X0_centred.T @ X0_centred
            S_1 = X1_centred.T @ X1_centred
            self.sigma_ = (S_0 + S_1) / N

            # Regularise if ill-conditioned
            try:
                np.linalg.cholesky(self.sigma_)
            except np.linalg.LinAlgError:
                self.sigma_ += self.cov_reg_eps * np.eye(D)

            self.sigma_inv_ = np.linalg.inv(self.sigma_)
            sign, self.log_det_ = np.linalg.slogdet(self.sigma_)
            if sign <= 0:
                # Fallback: add more regularisation
                self.sigma_ += self.cov_reg_eps * 10 * np.eye(D)
                self.sigma_inv_ = np.linalg.inv(self.sigma_)
                sign, self.log_det_ = np.linalg.slogdet(self.sigma_)

        self._is_fitted = True

    def _log_class_conditional(self, X, mu):
        """Compute log P(x | y=k) = log N(x; mu_k, Sigma).

        Returns (N,) log-likelihood for each sample.
        """
        D = X.shape[1]
        diff = X - mu  # (N, D)
        mahal = np.sum(diff @ self.sigma_inv_ * diff, axis=1)  # (N,)
        log_p = -0.5 * (D * np.log(2 * np.pi) + self.log_det_ + mahal)
        return log_p

    def predict_proba(self, X):
        """Return P(y=1 | x) via Bayes' rule.

        P(y=1|x) = P(x|y=1) P(y=1) / [P(x|y=0) P(y=0) + P(x|y=1) P(y=1)]
        """
        assert self._is_fitted, "Classifier not fitted. Call fit() first."
        X = np.asarray(X, dtype=np.float64)

        log_p0 = self._log_class_conditional(X, self.mu_0_) + np.log(self.prior_0_)
        log_p1 = self._log_class_conditional(X, self.mu_1_) + np.log(self.prior_1_)

        # Numerically stable softmax-style computation
        max_log = np.maximum(log_p0, log_p1)
        log_sum = max_log + np.log(
            np.exp(log_p0 - max_log) + np.exp(log_p1 - max_log)
        )

        log_posterior_1 = log_p1 - log_sum
        return np.exp(log_posterior_1).astype(np.float32)

    def save(self, path):
        """Save GDA parameters to disk."""
        assert self._is_fitted, "Classifier not fitted."
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(
            path,
            mu_0=self.mu_0_,
            mu_1=self.mu_1_,
            sigma=self.sigma_,
            sigma_inv=self.sigma_inv_,
            prior_0=self.prior_0_,
            prior_1=self.prior_1_,
            log_det=self.log_det_,
            cov_reg_eps=self.cov_reg_eps,
            naive_bayes=self.naive_bayes,
        )

    def load(self, path):
        """Load GDA parameters from disk."""
        data = np.load(path)
        self.mu_0_ = data['mu_0']
        self.mu_1_ = data['mu_1']
        self.sigma_ = data['sigma']
        self.sigma_inv_ = data['sigma_inv']
        self.prior_0_ = float(data['prior_0'])
        self.prior_1_ = float(data['prior_1'])
        self.log_det_ = float(data['log_det'])
        self.cov_reg_eps = float(data['cov_reg_eps'])
        self.naive_bayes = bool(data['naive_bayes'])
        self._is_fitted = True
        return self
