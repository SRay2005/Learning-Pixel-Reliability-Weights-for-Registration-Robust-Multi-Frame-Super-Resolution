"""
Main Runner Script for DSP Super-Resolution Project.

This is the entry point that orchestrates the entire pipeline:
  1. Download/prepare training data
  2. Train the robust SR model
  3. Train a baseline model (without robustness) for comparison
  4. Run comprehensive evaluation
  5. Generate all result figures
  6. Launch the interactive demo

Usage:
    python main.py --mode all        # Run everything
    python main.py --mode train      # Train only
    python main.py --mode evaluate   # Evaluate only (needs trained model)
    python main.py --mode demo       # Launch demo only
"""

import os
import sys
import argparse
import torch
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (download_sample_images, load_image, ensure_dir,
                        create_comparison_figure, create_metrics_plot)
from src.data_pipeline import SRDataPipeline
from src.train import train_model
from src.evaluate import SREvaluator
from src.sr_model import get_model


def setup_data():
    """Download sample images for training and testing."""
    print("\n" + "=" * 60)
    print("  [STEP 1] Preparing Training Data")
    print("=" * 60)
    
    hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    ensure_dir(hr_dir)
    
    # Check if images already exist
    existing = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(existing) >= 4:
        print(f"  Found {len(existing)} existing images in {hr_dir}")
        return hr_dir
    
    print("  Downloading sample images...")
    paths = download_sample_images(hr_dir, num_images=8)
    print(f"  [OK] Prepared {len(paths)} training images")
    
    return hr_dir


def train_robust_model(hr_dir, num_epochs=50, force=False, batch_size=8):
    """Train the robust SR model (with registration error augmentation)."""
    print("\n" + "=" * 60)
    print("  [STEP 2] Training Robust SR Model")
    print("=" * 60)
    
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'robust')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'robust')
    
    # Check if already trained
    best_path = os.path.join(save_dir, 'best.pth')
    if not force and os.path.exists(best_path):
        print(f"  Found existing checkpoint: {best_path}")
        print("  Loading pre-trained model...")
        model = get_model('robust', num_frames=4, num_channels=64, num_res_blocks=8)
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    model, history = train_model(
        hr_dir=hr_dir,
        save_dir=save_dir,
        results_dir=results_dir,
        model_type='robust',
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_frames=4,
        scale_factor=2,
        reg_error_std=0.3,       # KEY: train with registration errors
        learning_rate=1e-3,
        num_channels=64,
        num_res_blocks=8,
        patch_size=64,
    )
    
    # Also copy best checkpoint to the default location
    default_ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    ensure_dir(default_ckpt_dir)
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, os.path.join(default_ckpt_dir, 'best.pth'))
    latest_path = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest_path):
        import shutil
        shutil.copy2(latest_path, os.path.join(default_ckpt_dir, 'latest.pth'))
    
    return model


def train_baseline_model(hr_dir, num_epochs=50):
    """Train baseline model WITHOUT registration error augmentation."""
    print("\n" + "=" * 60)
    print("  [STEP 2b] Training Baseline Model (no robustness)")
    print("=" * 60)
    
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'baseline')
    
    best_path = os.path.join(save_dir, 'best.pth')
    if os.path.exists(best_path):
        print(f"  Found existing baseline checkpoint: {best_path}")
        model = get_model('simple', num_frames=4, num_channels=64, num_res_blocks=8)
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    model, history = train_model(
        hr_dir=hr_dir,
        save_dir=save_dir,
        results_dir=results_dir,
        model_type='simple',        # No attention fusion
        num_epochs=num_epochs,
        batch_size=4,
        num_frames=4,
        scale_factor=2,
        reg_error_std=0.0,          # NO registration error augmentation
        learning_rate=1e-3,
        num_channels=64,
        num_res_blocks=8,
        patch_size=64,
    )
    
    return model


def evaluate_models(hr_dir, robust_model):
    """Run comprehensive evaluation."""
    print("\n" + "=" * 60)
    print("  [STEP 3] Comprehensive Evaluation")
    print("=" * 60)
    
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    ensure_dir(results_dir)
    
    # Get test images
    hr_paths = []
    for f in sorted(os.listdir(hr_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            hr_paths.append(os.path.join(hr_dir, f))
    
    if not hr_paths:
        print("  Error: No test images found!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    robust_model = robust_model.to(device)
    
    evaluator = SREvaluator(scale_factor=2, num_frames=4, device=device)
    
    # Full evaluation
    eval_results = evaluator.full_evaluation(
        hr_paths, robust_model, results_dir
    )
    
    print(f"\n  [OK] Results saved to {results_dir}/")
    return eval_results


def launch_demo():
    """Launch the interactive Gradio demo."""
    print("\n" + "=" * 60)
    print("  [STEP 4] Launching Interactive Demo")
    print("=" * 60)
    
    from demo.app import create_demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='ML-Based Super-Resolution Robust to Registration Errors'
    )
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['all', 'train', 'evaluate', 'demo', 'data'],
        help='What to run: all, train, evaluate, demo, or data'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer epochs for fast testing'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force retraining even if checkpoint exists'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
    
    print("\n" + "+" + "=" * 58 + "+")
    print("|  ML-Based Super-Resolution Robust to Registration Errors  |")
    print("|  DSP Project - BITS Pilani                                |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Mode: {args.mode}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Force retrain: {args.force}")
    
    if args.mode in ['all', 'data', 'train']:
        hr_dir = setup_data()
    else:
        hr_dir = os.path.join(PROJECT_ROOT, 'data', 'hr')
    
    if args.mode in ['all', 'train']:
        robust_model = train_robust_model(hr_dir, num_epochs=args.epochs, force=args.force)
    
    if args.mode in ['all', 'evaluate']:
        if args.mode == 'evaluate':
            # Load pre-trained model
            ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'robust', 'best.pth')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best.pth')
            
            if os.path.exists(ckpt_path):
                robust_model = get_model('robust', num_frames=4, 
                                        num_channels=64, num_res_blocks=8)
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                robust_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("  Error: No trained model found! Run training first.")
                return
        
        evaluate_models(hr_dir, robust_model)
    
    if args.mode in ['all', 'demo']:
        launch_demo()
    
    print("\n  [DONE] All tasks completed!")


if __name__ == "__main__":
    main()
