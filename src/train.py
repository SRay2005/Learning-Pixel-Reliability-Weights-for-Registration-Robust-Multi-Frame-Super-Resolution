"""
Training Pipeline for Multi-Frame Super-Resolution.

Handles the complete training workflow:
  1. Data loading with registration error augmentation
  2. Model training with robust loss
  3. Validation with metrics tracking
  4. Checkpoint saving and learning rate scheduling
  5. Training visualization (loss curves)

Two training modes are supported:
  - "robust": Trains WITH registration error injection (our method)
  - "standard": Trains WITHOUT error injection (baseline for comparison)
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.sr_model import get_model
from src.robust_loss import RobustSRLoss, CharbonnierLoss
from src.data_pipeline import create_data_loaders
from src.utils import ensure_dir, compute_psnr, compute_ssim, tensor_to_numpy


class Trainer:
    """
    Super-Resolution model trainer.
    
    Manages the training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary with training configuration.
        """
        self.config = config
        if torch.cuda.is_available():
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Using device: {self.device}")
        
        # Create model
        self.model = get_model(
            model_type=config.get('model_type', 'robust'),
            num_frames=config.get('num_frames', 4),
            num_channels=config.get('num_channels', 64),
            num_res_blocks=config.get('num_res_blocks', 8),
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {total_params:,}")
        
        # Loss function
        use_perceptual = config.get('use_perceptual', False)
        self.criterion = RobustSRLoss(
            pixel_weight=1.0,
            edge_weight=config.get('edge_weight', 0.1),
            perceptual_weight=config.get('perceptual_weight', 0.0),
            use_perceptual=use_perceptual,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5),
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=config.get('min_lr', 1e-6),
        )
        
        # Training state
        self.start_epoch = 0
        self.best_psnr = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr': [],
        }
        
        # Directories
        self.save_dir = ensure_dir(config.get('save_dir', 'checkpoints'))
        self.results_dir = ensure_dir(config.get('results_dir', 'results'))
    
    def train_epoch(self, train_loader, epoch):
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for batch_idx, (lr_stack, hr_target, shifts) in enumerate(pbar):
            lr_stack = lr_stack.to(self.device)
            hr_target = hr_target.to(self.device)
            
            # Forward pass
            hr_pred = self.model(lr_stack)
            loss = self.criterion(hr_pred, hr_target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{epoch_loss/num_batches:.4f}'
            })
        
        return epoch_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        for lr_stack, hr_target, shifts in val_loader:
            lr_stack = lr_stack.to(self.device)
            hr_target = hr_target.to(self.device)
            
            hr_pred = self.model(lr_stack)
            loss = self.criterion(hr_pred, hr_target)
            
            total_loss += loss.item()
            
            # Compute metrics per image in batch
            for i in range(hr_pred.shape[0]):
                pred = tensor_to_numpy(hr_pred[i])
                target = tensor_to_numpy(hr_target[i])
                total_psnr += compute_psnr(pred, target)
                total_ssim += compute_ssim(pred, target)
            
            num_batches += 1
        
        n_samples = max(num_batches * hr_pred.shape[0], 1)
        return {
            'loss': total_loss / max(num_batches, 1),
            'psnr': total_psnr / n_samples,
            'ssim': total_ssim / n_samples,
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config,
            'history': self.history,
        }
        
        path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.history = checkpoint.get('history', self.history)
        print(f"  Resumed from epoch {self.start_epoch}, best PSNR: {self.best_psnr:.2f}")
    
    def plot_history(self, save_path=None):
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', 
                        label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', 
                        label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PSNR
        axes[0, 1].plot(epochs, self.history['val_psnr'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Validation PSNR', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # SSIM
        axes[1, 0].plot(epochs, self.history['val_ssim'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['lr'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, train_loader, val_loader):
        """
        Full training loop.
        
        Returns:
            Training history dictionary.
        """
        num_epochs = self.config.get('num_epochs', 50)
        
        print(f"\n{'='*60}")
        print(f"  Starting training for {num_epochs} epochs")
        print(f"  Model: {self.config.get('model_type', 'robust')}")
        print(f"  Reg Error Std: {self.config.get('reg_error_std', 0.0)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['lr'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.4f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
                  f"Val SSIM: {val_metrics['ssim']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{'* Best' if is_best else ''} | "
                  f"Time: {elapsed:.0f}s")
        
        # Plot final history
        self.plot_history(os.path.join(self.results_dir, 'training_history.png'))
        
        # Save final metrics
        with open(os.path.join(self.results_dir, 'training_metrics.json'), 'w') as f:
            json.dump({
                'best_psnr': float(self.best_psnr),
                'final_train_loss': float(self.history['train_loss'][-1]),
                'final_val_psnr': float(self.history['val_psnr'][-1]),
                'final_val_ssim': float(self.history['val_ssim'][-1]),
                'total_time': float(time.time() - start_time),
                'config': self.config,
            }, f, indent=2)
        
        print(f"\n  Training complete! Best PSNR: {self.best_psnr:.2f} dB")
        return self.history


def train_model(hr_dir, save_dir='checkpoints', results_dir='results',
                model_type='robust', num_epochs=50, batch_size=8,
                num_frames=4, scale_factor=2, reg_error_std=0.3,
                learning_rate=1e-3, num_channels=64, num_res_blocks=8,
                patch_size=64, resume=None):
    """
    Convenience function to train a super-resolution model.
    
    Args:
        hr_dir: Directory with HR training images.
        save_dir: Where to save checkpoints.
        results_dir: Where to save results/plots.
        model_type: 'robust' or 'simple'.
        num_epochs: Training epochs.
        batch_size: Batch size.
        num_frames: Number of LR frames.
        scale_factor: SR scale factor.
        reg_error_std: Registration error std for training augmentation.
        learning_rate: Initial learning rate.
        num_channels: Network width.
        num_res_blocks: Network depth.
        patch_size: Training patch size.
        resume: Path to checkpoint to resume from.
    
    Returns:
        Trained model and training history.
    """
    config = {
        'model_type': model_type,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_frames': num_frames,
        'scale_factor': scale_factor,
        'reg_error_std': reg_error_std,
        'learning_rate': learning_rate,
        'num_channels': num_channels,
        'num_res_blocks': num_res_blocks,
        'patch_size': patch_size,
        'save_dir': save_dir,
        'results_dir': results_dir,
        'edge_weight': 0.1,
        'use_perceptual': False,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
    }
    
    # Create data loaders
    print("  Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        hr_dir=hr_dir,
        scale_factor=scale_factor,
        num_frames=num_frames,
        patch_size=patch_size,
        batch_size=batch_size,
        noise_std=0.01,
        reg_error_std=reg_error_std,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if checkpoint provided
    if resume and os.path.exists(resume):
        trainer.load_checkpoint(resume)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    return trainer.model, history
