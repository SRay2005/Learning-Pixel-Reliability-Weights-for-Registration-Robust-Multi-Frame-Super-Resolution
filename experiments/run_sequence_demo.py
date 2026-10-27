"""
Experiment: Qualitative SR on Football/Bus video sequences.

Extracts consecutive frames, runs Phase 1 + Phase 2 (weighted) SR,
and produces side-by-side comparison figures.

These are standard test sequences from the video SR literature.
Results are purely qualitative (no ground truth at the SR resolution).
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (set_global_seed, load_image, save_image, ensure_dir)
from src.observation_model import bicubic_upsample
from src.sr_solver import sr_phase1, sr_phase2, sr_unweighted
from src.feature_extraction import extract_features_at_positions
from src.pca_transform import PCATransform
from src.classifiers import KernelSVMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sequence_frames(seq_dir, start=0, count=4):
    """Load consecutive grayscale frames from a directory."""
    frame_files = sorted([
        f for f in os.listdir(seq_dir) if f.endswith('.png')
    ])
    frame_files = frame_files[start:start + count]
    frames = []
    for f in frame_files:
        img = load_image(os.path.join(seq_dir, f), color='gray')
        frames.append(img)
    return frames


def register_frames(frames):
    """Estimate sub-pixel shifts using phase correlation."""
    from src.registration import PhaseCorrelationRegistration
    reg = PhaseCorrelationRegistration()
    shifts, confidences = reg.register_frames(frames, method='opencv')
    return shifts.astype(np.float32)


def compute_weight_maps(frames, shifts, scale, mu, phase1_iters,
                         classifier, pca):
    """Compute SVM weight maps for each frame."""
    n_frames = len(frames)
    H, W = frames[0].shape
    bic_up = bicubic_upsample(frames[0], scale)

    weight_maps = []
    subsample = 4

    for i in range(n_frames):
        if i == 0:
            weight_maps.append(np.ones((H * scale, W * scale),
                                       dtype=np.float32))
            continue

        # Phase 1 SR for feature extraction
        lr_pair = [frames[0], frames[i]]
        shifts_pair = np.array([shifts[0], shifts[i]])
        sr_img = sr_phase1(lr_pair, shifts_pair, scale, mu=mu,
                           num_iterations=phase1_iters)

        H_hr, W_hr = H * scale, W * scale
        rows = np.arange(0, H_hr, subsample)
        cols = np.arange(0, W_hr, subsample)
        positions = np.array([(r, c) for r in rows for c in cols])

        features_raw = extract_features_at_positions(
            sr_img[:H_hr, :W_hr], bic_up[:H_hr, :W_hr], positions)
        features_pca = pca.transform(features_raw)
        probs = classifier.predict_proba(features_pca)

        h_sub, w_sub = len(rows), len(cols)
        M_sub = probs.reshape(h_sub, w_sub)
        M_i = cv2.resize(M_sub, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)
        M_i = np.clip(M_i, 0, 1).astype(np.float32)
        weight_maps.append(M_i)

    return weight_maps


def run_sequence(seq_name, seq_dir, results_dir, scale=2, n_frames=4):
    """Process one video sequence."""
    print(f"\n  --- Sequence: {seq_name} ---")

    frames = load_sequence_frames(seq_dir, start=0, count=n_frames)
    if len(frames) < 2:
        print(f"    Not enough frames in {seq_dir}")
        return

    print(f"    Loaded {len(frames)} frames, size {frames[0].shape}")

    mu = 0.1
    phase1_iters = 3
    phase2_iters = 30

    # Register
    shifts = register_frames(frames)
    shifts_lr = shifts  # Already in LR domain
    print(f"    Estimated shifts: {shifts}")

    # Bicubic
    bic = bicubic_upsample(frames[0], scale)

    # Unweighted SR
    sr_uw = sr_unweighted(frames, shifts_lr, scale, mu=mu,
                           num_iterations=phase2_iters)

    # Weighted SR (SVM)
    pca = PCATransform()
    pca_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'pca.npz')
    pca.load(pca_path)

    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'classifiers')
    svm = KernelSVMClassifier()
    svm.load(os.path.join(ckpt_dir, 'KernelSVM(C=100.0, gamma=0.01).joblib'))

    weight_maps = compute_weight_maps(frames, shifts_lr, scale, mu,
                                       phase1_iters, svm, pca)
    sr_weighted = sr_phase2(frames, shifts_lr, weight_maps, scale,
                             mu=mu, num_iterations=phase2_iters)

    # Save individual images
    seq_results = ensure_dir(os.path.join(results_dir, seq_name))
    H, W = frames[0].shape
    H_hr, W_hr = H * scale, W * scale

    save_image(os.path.join(seq_results, 'lr_frame0.png'), frames[0])
    save_image(os.path.join(seq_results, 'bicubic.png'),
               np.clip(bic[:H_hr, :W_hr], 0, 1))
    save_image(os.path.join(seq_results, 'sr_unweighted.png'),
               np.clip(sr_uw[:H_hr, :W_hr], 0, 1))
    save_image(os.path.join(seq_results, 'sr_weighted_svm.png'),
               np.clip(sr_weighted[:H_hr, :W_hr], 0, 1))

    # Weight map visualization
    if len(weight_maps) > 1:
        fig, axes = plt.subplots(1, len(weight_maps) - 1,
                                  figsize=(5 * (len(weight_maps) - 1), 5))
        if len(weight_maps) - 1 == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(weight_maps[i + 1], cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f'Weight Map (Frame {i+1})', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.suptitle(f'{seq_name}: SVM Weight Maps', fontsize=14,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(seq_results, 'weight_maps.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes,
                               [bic[:H_hr, :W_hr],
                                sr_uw[:H_hr, :W_hr],
                                sr_weighted[:H_hr, :W_hr]],
                               ['Bicubic', 'Unweighted SR', 'Weighted SR (SVM)']):
        ax.imshow(np.clip(img, 0, 1), cmap='gray')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'{seq_name} Sequence: SR Comparison (x{scale})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(seq_results, 'comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Results saved to {seq_results}/")


def main():
    set_global_seed(42)

    seq_base = os.path.join(PROJECT_ROOT, 'data', 'sequences')
    results_dir = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'sequences'))

    for seq_name in ['bus', 'football']:
        seq_dir = os.path.join(seq_base, seq_name)
        if os.path.exists(seq_dir) and len(os.listdir(seq_dir)) > 0:
            run_sequence(seq_name, seq_dir, results_dir)
        else:
            print(f"\n  Skipping {seq_name}: no frames found at {seq_dir}")
            print(f"  Run 'python experiments/download_sequences.py' first.")


if __name__ == '__main__':
    print("\n=== Qualitative Video Sequence SR ===")
    main()
