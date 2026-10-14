# ML-Based Super-Resolution Robust to Registration Errors


A Python implementation of multi-frame image super-resolution using deep learning,
specifically designed to be robust against registration (motion estimation) errors.

---

## Problem

Multi-frame super-resolution reconstructs a high-resolution (HR) image from multiple
low-resolution (LR) frames. Traditional methods require **precise sub-pixel registration**
between frames, but real-world motion estimation is imprecise. These registration errors
cause severe artifacts in the final SR output.

## Our Approach

We train a neural network with **temporal attention fusion** that:
1. **Learns to ignore misaligned frames** via attention weights
2. **Trains with simulated registration errors** so the network becomes invariant to them
3. **Uses robust loss functions** (Charbonnier) that don't amplify outlier errors

## Architecture

```
Input: N upsampled LR frames
    │
    ├─ Per-frame Feature Extraction (shared CNN + ResBlocks)
    │
    ├─ Temporal Attention Fusion (learns frame reliability)
    │       ↓ Poorly registered frames get low weights
    ├─ Deep Reconstruction (ResBlocks + Channel Attention)
    │
    └─ Residual Connection → HR Output
```

## Methods Compared

| Method | Type | Multi-Frame | Robust to Reg Errors |
|--------|------|-------------|----------------------|
| Bicubic | Interpolation | ❌ | N/A |
| Mean Fusion | Classical | ✅ | ❌ |
| IBP (Irani & Peleg) | Classical | ✅ | ❌ |
| **Ours (Robust ML)** | **Deep Learning** | ✅ | ✅ |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Everything (Train + Evaluate + Demo)
```bash
python main.py --mode all --epochs 50
```

### 3. Quick Test (10 epochs)
```bash
python main.py --mode all --quick
```

### 4. Individual Steps
```bash
# Download training data only
python main.py --mode data

# Train only
python main.py --mode train --epochs 50

# Evaluate (requires trained model)
python main.py --mode evaluate

# Launch interactive demo
python main.py --mode demo
```

## Project Structure

```
MLEE/
├── main.py                  # 🚀 Main entry point
├── requirements.txt         # Dependencies
├── README.md                # This file
├── src/
│   ├── data_pipeline.py     # LR frame generation & augmentation
│   ├── registration.py      # Phase correlation motion estimation
│   ├── sr_model.py          # Neural network architecture
│   ├── robust_loss.py       # Charbonnier, Huber, Edge, Perceptual losses
│   ├── classical_sr.py      # Bicubic, Mean Fusion, IBP baselines
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation & robustness analysis
│   └── utils.py             # Image I/O, metrics, visualization
├── demo/
│   └── app.py               # Gradio interactive demo
├── data/hr/                  # Training images
├── checkpoints/              # Saved model weights
└── results/                  # Output figures and metrics
```

## 📈 Expected Results

The robustness curves show that as registration error increases:
- Classical methods (IBP, Mean Fusion) degrade rapidly
- Our ML method maintains significantly higher PSNR/SSIM

## 📖 Reference

Based on: *"Machine Learning Based Super-Resolution Algorithm Robust to Registration Errors"*
