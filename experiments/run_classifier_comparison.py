"""
Experiment: Train all classifiers, evaluate on dev and test sets.

Produces:
  - Per-classifier: accuracy, precision, recall, F1, ROC-AUC, log-loss
  - Calibration curves
  - ROC curves
  - Training/inference time, parameter count
  - Summary CSV table

All classifiers use identical features, PCA, and data splits.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, classification_report,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_global_seed, ensure_dir
from src.classifiers import (
    LogisticRegressionClassifier,
    KernelSVMClassifier,
    GDAClassifier,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_splits():
    """Load the pre-computed train/dev/test splits."""
    data_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'data')
    splits = {}
    for name in ['train', 'dev', 'test']:
        data = np.load(os.path.join(data_dir, f'{name}.npz'))
        splits[name] = {'X': data['X'], 'y': data['y'],
                        'image_ids': data['image_ids']}
    return splits


def evaluate_classifier(clf, X, y, name=''):
    """Compute all classification metrics."""
    probs = clf.predict_proba(X)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0,
        'log_loss': log_loss(y, np.clip(probs, 1e-15, 1 - 1e-15)),
    }
    return metrics, probs


def run_comparison():
    """Train and evaluate all classifiers."""
    set_global_seed(42)

    splits = load_splits()
    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_dev, y_dev = splits['dev']['X'], splits['dev']['y']
    X_test, y_test = splits['test']['X'], splits['test']['y']

    print(f"  Train: {len(X_train)} samples")
    print(f"  Dev:   {len(X_dev)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # If no separate test set, use dev for evaluation (report as dev)
    if len(X_test) == 0:
        print("  NOTE: No test split available. Using dev set for evaluation.")
        X_test, y_test = X_dev, y_dev

    classifiers = [
        LogisticRegressionClassifier(learning_rate=0.01, max_iterations=1000,
                                      l2_lambda=0.0),
        LogisticRegressionClassifier(learning_rate=0.01, max_iterations=1000,
                                      l2_lambda=0.01),
        KernelSVMClassifier(C=1.0, gamma='scale'),
        GDAClassifier(cov_reg_eps=1e-6),
    ]

    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'classifiers'))
    all_results = []

    # For ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    # For calibration curves
    fig_cal, ax_cal = plt.subplots(figsize=(8, 8))

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']

    for idx, clf in enumerate(classifiers):
        print(f"\n  --- {clf.name} ---")

        # Train
        train_time = clf.timed_fit(X_train, y_train)

        # Dev metrics
        dev_metrics, dev_probs = evaluate_classifier(clf, X_dev, y_dev, 'dev')

        # Inference timing
        _, inf_time = clf.timed_predict_proba(X_dev)
        inf_time_per_sample = inf_time / max(len(X_dev), 1)

        result = {
            'name': clf.name,
            'train_time': train_time,
            'inf_time_per_sample': inf_time_per_sample,
            'param_count': clf.param_count,
            **{f'dev_{k}': v for k, v in dev_metrics.items()},
        }

        # Test metrics (TOUCH ONCE)
        test_metrics, test_probs = evaluate_classifier(clf, X_test, y_test, 'test')
        result.update({f'test_{k}': v for k, v in test_metrics.items()})

        all_results.append(result)

        print(f"    Train time:      {train_time:.2f}s")
        print(f"    Param count:     {clf.param_count}")
        print(f"    Dev accuracy:    {dev_metrics['accuracy']:.4f}")
        print(f"    Dev ROC-AUC:     {dev_metrics['roc_auc']:.4f}")
        print(f"    Test accuracy:   {test_metrics['accuracy']:.4f}")
        print(f"    Test ROC-AUC:    {test_metrics['roc_auc']:.4f}")

        # ROC curve (on test set)
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            ax_roc.plot(fpr, tpr, color=colors[idx], linewidth=2,
                       label=f'{clf.name} (AUC={test_metrics["roc_auc"]:.3f})')

        # Calibration curve (on test set)
        if len(np.unique(y_test)) > 1:
            prob_true, prob_pred = calibration_curve(y_test, test_probs,
                                                      n_bins=10, strategy='uniform')
            ax_cal.plot(prob_pred, prob_true, 's-', color=colors[idx],
                       linewidth=2, label=clf.name)

        # Save classifier
        ckpt_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'checkpoints', 'classifiers'))
        ext = '.joblib' if isinstance(clf, KernelSVMClassifier) else '.npz'
        clf.save(os.path.join(ckpt_dir, f'{clf.name}{ext}'))

    # Finalize ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves — Artifact Detection', fontsize=14, fontweight='bold')
    ax_roc.legend(fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig_roc)

    # Finalize calibration plot
    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax_cal.set_ylabel('Fraction of Positives', fontsize=12)
    ax_cal.set_title('Calibration Curves', fontsize=14, fontweight='bold')
    ax_cal.legend(fontsize=10)
    ax_cal.grid(True, alpha=0.3)
    fig_cal.tight_layout()
    fig_cal.savefig(os.path.join(results_dir, 'calibration_curves.png'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig_cal)

    # Summary CSV
    csv_path = os.path.join(results_dir, 'classifier_comparison.csv')
    with open(csv_path, 'w') as f:
        keys = all_results[0].keys()
        f.write(','.join(keys) + '\n')
        for r in all_results:
            f.write(','.join(str(r[k]) for k in keys) + '\n')

    # Print summary table
    print("\n  === CLASSIFIER COMPARISON (Test Set) ===")
    print(f"  {'Classifier':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} "
          f"{'LogLoss':>8} {'Params':>8} {'Train(s)':>8}")
    print(f"  {'-'*78}")
    for r in all_results:
        print(f"  {r['name']:<30} {r['test_accuracy']:>6.4f} "
              f"{r['test_f1']:>6.4f} {r['test_roc_auc']:>6.4f} "
              f"{r['test_log_loss']:>8.4f} {r['param_count']:>8} "
              f"{r['train_time']:>8.2f}")

    print(f"\n  Results saved to {results_dir}/")
    return all_results


if __name__ == '__main__':
    print("\n=== Classifier Comparison ===")
    run_comparison()
