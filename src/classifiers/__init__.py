"""Classifier package — pluggable artifact-detection classifiers."""

from src.classifiers.base import ClassifierBase
from src.classifiers.logistic import LogisticRegressionClassifier
from src.classifiers.svm import KernelSVMClassifier
from src.classifiers.gda import GDAClassifier

__all__ = [
    'ClassifierBase',
    'LogisticRegressionClassifier',
    'KernelSVMClassifier',
    'GDAClassifier',
]
