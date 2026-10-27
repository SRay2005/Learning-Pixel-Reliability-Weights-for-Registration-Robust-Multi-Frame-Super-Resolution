"""
Experiment: k-fold CV for SVM hyperparameter selection.

Grid search over C and gamma for the RBF kernel SVM using
grouped k-fold cross-validation (groups = source image).

Also searches L2 lambda for logistic regression.

Output: best hyperparameters, CV results CSV.
"""

import os
import sys
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_global_seed, ensure_dir
from src.data_split import grouped_kfold_cv
from src.classifiers import (
    LogisticRegressionClassifier,
    KernelSVMClassifier,
    GDAClassifier,
)
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cv_svm(X, y, image_ids, n_folds=5):
    """Grid search C and gamma for SVM via grouped k-fold CV."""
    print("\n  --- SVM Hyperparameter Search (k-fold CV) ---")

    C_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = ['scale', 0.01, 0.1, 1.0]

    results = []

    for C, gamma in itertools.product(C_values, gamma_values):
        fold_aucs = []
        fold_accs = []

        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(
                grouped_kfold_cv(X, y, image_ids, n_folds)):
            try:
                clf = KernelSVMClassifier(C=C, gamma=gamma)
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_val)
                preds = (probs >= 0.5).astype(int)

                auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
                acc = accuracy_score(y_val, preds)
                fold_aucs.append(auc)
                fold_accs.append(acc)
            except Exception as e:
                print(f"    Warning: C={C}, gamma={gamma}, fold {fold_idx}: {e}")
                fold_aucs.append(0.5)
                fold_accs.append(0.5)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_acc = np.mean(fold_accs)

        results.append({
            'C': C, 'gamma': gamma,
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'mean_acc': mean_acc,
        })
        print(f"    C={C:<6} gamma={str(gamma):<8} "
              f"AUC={mean_auc:.4f}+/-{std_auc:.4f}  acc={mean_acc:.4f}")

    # Best by AUC
    best = max(results, key=lambda r: r['mean_auc'])
    print(f"\n  Best SVM: C={best['C']}, gamma={best['gamma']}, "
          f"AUC={best['mean_auc']:.4f}")

    return results, best


def cv_logistic_l2(X, y, image_ids, n_folds=5):
    """Search L2 regularisation strength for logistic regression."""
    print("\n  --- LogReg L2 Lambda Search (k-fold CV) ---")

    lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    results = []

    for lam in lambda_values:
        fold_aucs = []
        fold_accs = []

        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(
                grouped_kfold_cv(X, y, image_ids, n_folds)):
            clf = LogisticRegressionClassifier(
                learning_rate=0.01, max_iterations=1000, l2_lambda=lam)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)
            preds = (probs >= 0.5).astype(int)

            auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
            acc = accuracy_score(y_val, preds)
            fold_aucs.append(auc)
            fold_accs.append(acc)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_acc = np.mean(fold_accs)

        results.append({
            'lambda': lam,
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'mean_acc': mean_acc,
        })
        print(f"    lambda={lam:<8} AUC={mean_auc:.4f}+/-{std_auc:.4f}  "
              f"acc={mean_acc:.4f}")

    best = max(results, key=lambda r: r['mean_auc'])
    print(f"\n  Best LogReg: lambda={best['lambda']}, "
          f"AUC={best['mean_auc']:.4f}")

    return results, best


def main():
    set_global_seed(42)

    # Load train + dev combined for CV
    data_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'data')
    train = np.load(os.path.join(data_dir, 'train.npz'))
    dev = np.load(os.path.join(data_dir, 'dev.npz'))

    X = np.concatenate([train['X'], dev['X']], axis=0)
    y = np.concatenate([train['y'], dev['y']], axis=0)
    ids = np.concatenate([train['image_ids'], dev['image_ids']], axis=0)

    print(f"  Total samples for CV: {len(X)}")

    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'cv'))

    # SVM CV
    svm_results, svm_best = cv_svm(X, y, ids, n_folds=5)

    # LogReg L2 CV
    lr_results, lr_best = cv_logistic_l2(X, y, ids, n_folds=5)

    # Save CSV
    with open(os.path.join(results_dir, 'cv_svm.csv'), 'w') as f:
        f.write('C,gamma,mean_auc,std_auc,mean_acc\n')
        for r in svm_results:
            f.write(f"{r['C']},{r['gamma']},{r['mean_auc']:.4f},"
                    f"{r['std_auc']:.4f},{r['mean_acc']:.4f}\n")

    with open(os.path.join(results_dir, 'cv_logistic_l2.csv'), 'w') as f:
        f.write('lambda,mean_auc,std_auc,mean_acc\n')
        for r in lr_results:
            f.write(f"{r['lambda']},{r['mean_auc']:.4f},"
                    f"{r['std_auc']:.4f},{r['mean_acc']:.4f}\n")

    # Heatmap for SVM
    C_vals = sorted(set(r['C'] for r in svm_results))
    gamma_vals = sorted(set(str(r['gamma']) for r in svm_results))
    auc_grid = np.zeros((len(C_vals), len(gamma_vals)))

    for r in svm_results:
        i = C_vals.index(r['C'])
        j = gamma_vals.index(str(r['gamma']))
        auc_grid[i, j] = r['mean_auc']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(auc_grid, cmap='RdYlGn', aspect='auto',
                   vmin=0.5, vmax=max(0.85, auc_grid.max()))
    ax.set_xticks(range(len(gamma_vals)))
    ax.set_xticklabels(gamma_vals)
    ax.set_yticks(range(len(C_vals)))
    ax.set_yticklabels([str(c) for c in C_vals])
    ax.set_xlabel('gamma', fontsize=12)
    ax.set_ylabel('C', fontsize=12)
    ax.set_title('SVM Hyperparameter CV (AUC)', fontsize=14, fontweight='bold')

    for i in range(len(C_vals)):
        for j in range(len(gamma_vals)):
            ax.text(j, i, f'{auc_grid[i,j]:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if auc_grid[i,j] < 0.65 else 'black')

    plt.colorbar(im, label='Mean AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'svm_cv_heatmap.png'), dpi=150)
    plt.close()

    print(f"\n  [OK] CV results saved to {results_dir}/")
    print(f"\n  RECOMMENDED CONFIG UPDATES:")
    print(f"    SVM: C={svm_best['C']}, gamma={svm_best['gamma']}")
    print(f"    LogReg L2: lambda={lr_best['lambda']}")


if __name__ == '__main__':
    print("\n=== k-Fold Cross-Validation ===")
    main()
