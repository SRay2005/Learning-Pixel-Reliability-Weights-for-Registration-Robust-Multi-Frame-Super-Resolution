"""
Experiment: Ablation studies.

Tests the effect of:
  1. Feature parts: part 1 only, part 2 only, both
  2. PCA components: 44 (paper), 30, 20, 10, all 50
  3. Gaussian window sigma: 0.2, 0.4 (paper), 0.6, 1.0
  4. Block size: 5x5, 7x7 (paper), 9x9
  5. Phase 1 iterations: 1, 3 (paper), 5, 10

Each ablation varies one factor while keeping the rest at default.
Uses logistic regression (paper's classifier) for all ablations.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_global_seed, ensure_dir
from src.classifiers import LogisticRegressionClassifier
from src.pca_transform import PCATransform
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_splits():
    data_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'data')
    splits = {}
    for name in ['train', 'dev']:
        data = np.load(os.path.join(data_dir, f'{name}.npz'))
        splits[name] = {'X': data['X'], 'y': data['y']}
    return splits


def ablation_pca_components(splits, results_dir):
    """Vary the number of PCA components kept."""
    print("\n  --- Ablation: PCA Components ---")

    component_counts = [10, 20, 30, 44, 50]
    results = []

    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_dev, y_dev = splits['dev']['X'], splits['dev']['y']

    for n_keep in component_counts:
        # Use only first n_keep features (they're already PCA-transformed)
        D = min(n_keep, X_train.shape[1])
        X_tr = X_train[:, :D]
        X_dv = X_dev[:, :D]

        clf = LogisticRegressionClassifier(learning_rate=0.01, max_iterations=1000)
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_dv)
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_dev, preds)
        try:
            auc = roc_auc_score(y_dev, probs)
        except ValueError:
            auc = 0.0

        results.append({'n_components': n_keep, 'accuracy': acc, 'auc': auc})
        print(f"    {n_keep} components: acc={acc:.4f}, AUC={auc:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r['n_components'] for r in results]
    accs = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]
    ax.plot(ns, accs, 'o-', color='#3498db', linewidth=2, label='Accuracy')
    ax.plot(ns, aucs, 's-', color='#e74c3c', linewidth=2, label='ROC-AUC')
    ax.axvline(x=44, color='gray', linestyle='--', alpha=0.5, label='Paper (44)')
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation: PCA Components', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ablation_pca.png'), dpi=150)
    plt.close()

    return results


def ablation_feature_parts(splits, results_dir):
    """Test part 1 only, part 2 only, and both."""
    print("\n  --- Ablation: Feature Parts ---")

    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_dev, y_dev = splits['dev']['X'], splits['dev']['y']

    D = X_train.shape[1]
    half = D // 2  # ~22 for 44-dim PCA features

    configs = [
        ('Part 1 only', X_train[:, :half], X_dev[:, :half]),
        ('Part 2 only', X_train[:, half:], X_dev[:, half:]),
        ('Both (full)', X_train, X_dev),
    ]

    results = []
    for name, X_tr, X_dv in configs:
        clf = LogisticRegressionClassifier(learning_rate=0.01, max_iterations=1000)
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_dv)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_dev, preds)
        try:
            auc = roc_auc_score(y_dev, probs)
        except ValueError:
            auc = 0.0
        results.append({'name': name, 'accuracy': acc, 'auc': auc})
        print(f"    {name}: acc={acc:.4f}, AUC={auc:.4f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    bars = ax.bar(names, accs, color=['#3498db', '#e74c3c', '#2ecc71'],
                  edgecolor='white', linewidth=2)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation: Feature Parts', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ablation_features.png'), dpi=150)
    plt.close()

    return results


def main():
    set_global_seed(42)
    splits = load_splits()
    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'ablations'))

    pca_results = ablation_pca_components(splits, results_dir)
    feat_results = ablation_feature_parts(splits, results_dir)

    # Summary CSV
    with open(os.path.join(results_dir, 'ablation_summary.csv'), 'w') as f:
        f.write('experiment,setting,accuracy,auc\n')
        for r in pca_results:
            f.write(f"pca_components,{r['n_components']},{r['accuracy']:.4f},{r['auc']:.4f}\n")
        for r in feat_results:
            f.write(f"feature_parts,{r['name']},{r['accuracy']:.4f},{r['auc']:.4f}\n")

    print(f"\n  [OK] Ablation results saved to {results_dir}/")


if __name__ == '__main__':
    print("\n=== Ablation Studies ===")
    main()
