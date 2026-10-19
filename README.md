# Learning Pixel-Reliability Weights for Registration-Robust Multi-Frame Super-Resolution

A Comparison of Logistic Regression, Kernel SVM and Generative Classifiers

---

## Paper

Implementation of: *"Machine Learning Based Super-Resolution Algorithm Robust to
Registration Errors"* (Milchevski, Ivanovski, Mustafa, DSP/SPE 2011).

Extended with Kernel SVM and Gaussian Discriminant Analysis classifiers
for the MLEE (Machine Learning, EEE G513) course project at BITS Pilani.

## Method

Multi-frame SR reconstructs a high-resolution (HR) image from multiple
low-resolution (LR) frames. Registration errors between frames cause
severe artifacts. The paper addresses this by:

1. **Phase 1 SR** (unweighted): iterative gradient-descent SR from LR frames
2. **Artifact detection**: per-pixel feature extraction (7x7 blocks, DFT,
   Gaussian window, PCA) followed by a classifier that outputs a reliability
   weight M(r,c) in [0,1]
3. **Phase 2 SR** (weighted): the same SR update, but each frame's contribution
   is weighted by its reliability map

### Classifiers compared

| Classifier | Type | Source |
|-----------|------|--------|
| Logistic Regression | Discriminative | Paper baseline |
| Logistic Regression (L2) | Discriminative | Our extension |
| Kernel SVM (RBF) + Platt | Discriminative | Our contribution |
| GDA (shared covariance) | Generative | Our contribution |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Training Data (features from 42 images)
```bash
python main.py build_data
```

### 3. Train All Classifiers
```bash
python main.py train
```

### 4. Run SR Demo
```bash
python main.py demo
```

### 5. Full Pipeline
```bash
python main.py all
```

## Project Structure

```
MLEE/
+-- main.py                      # Entry point
+-- configs/
|   +-- default.yaml             # All hyperparameters
+-- src/
|   +-- observation_model.py     # W_i operator (shift + downsample)
|   +-- sr_solver.py             # Phase 1 & Phase 2 iterative SR
|   +-- feature_extraction.py    # 7x7 blocks, DFT, amplitude spectrum
|   +-- pca_transform.py         # PCA dimensionality reduction (50->44)
|   +-- artifact_detector.py     # Features -> PCA -> classifier -> weight map
|   +-- training_data.py         # Build training set per paper protocol
|   +-- data_split.py            # Image-level train/dev/test split
|   +-- registration.py          # Phase correlation motion estimation
|   +-- classical_sr.py          # Bicubic baseline
|   +-- utils.py                 # Image I/O, metrics, seeds
|   +-- classifiers/
|       +-- base.py              # Abstract classifier interface
|       +-- logistic.py          # Paper's logistic regression
|       +-- svm.py               # Kernel SVM + Platt scaling
|       +-- gda.py               # Gaussian Discriminant Analysis
+-- experiments/                  # One script per experiment
+-- tests/                       # Unit tests
+-- extras/
|   +-- deep_cnn_baseline/       # Archived CNN approach (not the paper)
+-- data/hr/                     # Training/test images
+-- checkpoints/                 # Saved models, PCA, data splits
+-- results/                     # Tables, figures, CSV
+-- requirements.txt
+-- README.md
```

## Evaluation

- **Classifier metrics**: accuracy, precision, recall, F1, ROC-AUC, log-loss,
  calibration curve
- **SR metrics**: PSNR and SSIM
- **Cost**: training time, inference time, parameter count / support vectors
- **Controlled synthetic tests**: known ground truth, error levels 0-2 px
- **Ablations**: feature parts, PCA components, block size, Phase-1 iterations

## Key Design Decisions

| Decision | Value | Source |
|----------|-------|--------|
| Downsampling | Non-overlapping averaging | Paper |
| Phase 1 iterations | 3 | Paper |
| Phase 1 lambda | 0 | Paper |
| Step size mu | Tuned on dev set | Our choice |
| Phase 2 iterations | Tuned on dev set | Our choice |
| Feature block size | 7x7 | Paper |
| Gaussian window sigma | 0.4 | Paper |
| PCA components kept | 44 (drop last 6) | Paper |
| Data split | By image, never by pixel | Our choice |
| Random seeds | Fixed (42) | Our choice |

## Reference

Milchevski, Ivanovski, Mustafa. *"Machine Learning Based Super-Resolution
Algorithm Robust to Registration Errors."* DSP/SPE 2011.
