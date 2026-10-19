"""
Evaluation Module for Super-Resolution.

Provides comprehensive evaluation of SR methods:
  1. Quantitative metrics (PSNR, SSIM) under varying registration errors
  2. Visual comparison generation
  3. Robustness analysis plots
  4. Per-image detailed results

The key experiment is the ROBUSTNESS CURVE: how does each method's
performance degrade as registration error increases?
"""

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from src.utils import (load_image, save_image, compute_psnr, compute_ssim,
                       numpy_to_tensor, tensor_to_numpy, ensure_dir,
                       create_comparison_figure)
from src.data_pipeline import SRDataPipeline
from src.classical_sr import (bicubic_upsample, mean_fusion_sr,
                               iterative_back_projection)
from src.registration import PhaseCorrelationRegistration


class SREvaluator:
    """
    Comprehensive evaluation of super-resolution methods.
    """
    
    def __init__(self, scale_factor=2, num_frames=4, device=None):
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.pipeline = SRDataPipeline(scale_factor, num_frames, noise_std=0.01)
        self.registrator = PhaseCorrelationRegistration()
    
    def evaluate_single_image(self, hr_image, model, reg_error_std=0.0):
        """
        Evaluate all SR methods on a single HR image.
        
        Args:
            hr_image: Ground truth HR image (H, W, C), float32 [0,1].
            model: Trained PyTorch SR model.
            reg_error_std: Registration error std to simulate.
        
        Returns:
            Dictionary of {method: {'psnr': float, 'ssim': float, 'image': array}}.
        """
        # Ensure image dimensions are divisible by scale factor
        s = self.scale_factor
        h, w = hr_image.shape[:2]
        h = (h // s) * s
        w = (w // s) * s
        hr_image = hr_image[:h, :w]
        
        # Generate LR frames
        lr_frames, true_shifts = self.pipeline.generate_lr_frames(hr_image)
        
        # Add registration errors to shifts
        if reg_error_std > 0:
            estimated_shifts = self.pipeline.add_registration_error(
                true_shifts / s, reg_error_std
            )
        else:
            estimated_shifts = true_shifts / s
        
        results = {}
        
        # --- Classical Methods ---
        # Bicubic (single frame)
        bicubic_result = bicubic_upsample(lr_frames[0], s)
        bicubic_result = bicubic_result[:h, :w]
        results['Bicubic'] = {
            'psnr': compute_psnr(bicubic_result, hr_image),
            'ssim': compute_ssim(bicubic_result, hr_image),
            'image': bicubic_result,
        }
        
        # Mean Fusion
        mean_result = mean_fusion_sr(lr_frames, s, estimated_shifts)
        mean_result = mean_result[:h, :w]
        results['Mean Fusion'] = {
            'psnr': compute_psnr(mean_result, hr_image),
            'ssim': compute_ssim(mean_result, hr_image),
            'image': mean_result,
        }
        
        # IBP
        ibp_result = iterative_back_projection(
            lr_frames, s, estimated_shifts, num_iterations=20
        )
        ibp_result = ibp_result[:h, :w]
        results['IBP'] = {
            'psnr': compute_psnr(ibp_result, hr_image),
            'ssim': compute_ssim(ibp_result, hr_image),
            'image': ibp_result,
        }
        
        # --- ML-Based Method ---
        model.eval()
        with torch.no_grad():
            # Prepare input: upsample LR frames and stack
            lr_upsampled = []
            for lr in lr_frames:
                up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
                lr_upsampled.append(up)
            
            lr_stack = np.concatenate(lr_upsampled, axis=2)  # (H, W, N*C)
            lr_tensor = torch.from_numpy(
                lr_stack.transpose(2, 0, 1)
            ).float().unsqueeze(0).to(self.device)
            
            ml_result = model(lr_tensor)
            ml_result = tensor_to_numpy(ml_result)
            ml_result = ml_result[:h, :w]
        
        results['Ours (Robust ML)'] = {
            'psnr': compute_psnr(ml_result, hr_image),
            'ssim': compute_ssim(ml_result, hr_image),
            'image': ml_result,
        }
        
        return results, hr_image
    
    def robustness_analysis(self, hr_images, model, 
                            error_levels=None, num_trials=3):
        """
        Analyze how each method degrades under increasing registration errors.
        
        This is the KEY experiment: sweep registration error from 0 to large
        values and plot the performance curve for each method.
        
        Args:
            hr_images: List of HR images.
            model: Trained SR model.
            error_levels: List of registration error std values to test.
            num_trials: Number of random trials per error level.
        
        Returns:
            Robustness results dictionary.
        """
        if error_levels is None:
            error_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        
        methods = ['Bicubic', 'Mean Fusion', 'IBP', 'Ours (Robust ML)']
        robustness = {m: {'psnr': [], 'ssim': []} for m in methods}
        
        print("\n  Running robustness analysis...")
        for err_std in tqdm(error_levels, desc='  Error levels'):
            method_psnrs = {m: [] for m in methods}
            method_ssims = {m: [] for m in methods}
            
            for hr_img in hr_images:
                for trial in range(num_trials):
                    results, _ = self.evaluate_single_image(
                        hr_img, model, reg_error_std=err_std
                    )
                    for m in methods:
                        if m in results:
                            method_psnrs[m].append(results[m]['psnr'])
                            method_ssims[m].append(results[m]['ssim'])
            
            for m in methods:
                robustness[m]['psnr'].append(
                    np.mean(method_psnrs[m]) if method_psnrs[m] else 0
                )
                robustness[m]['ssim'].append(
                    np.mean(method_ssims[m]) if method_ssims[m] else 0
                )
        
        robustness['error_levels'] = error_levels
        return robustness
    
    def plot_robustness_curves(self, robustness, save_path=None):
        """
        Plot the robustness curves showing performance vs registration error.
        
        This is the CENTRAL FIGURE of the project — it directly demonstrates
        that our ML method maintains higher quality under registration errors.
        """
        error_levels = robustness['error_levels']
        methods = [k for k in robustness.keys() if k != 'error_levels']
        
        colors = {
            'Bicubic': '#ff6b6b',
            'Mean Fusion': '#ffd93d',
            'IBP': '#6bcb77',
            'Ours (Robust ML)': '#4d96ff',
        }
        markers = {
            'Bicubic': 'o',
            'Mean Fusion': 's',
            'IBP': '^',
            'Ours (Robust ML)': 'D',
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PSNR vs Registration Error
        for m in methods:
            ax1.plot(error_levels, robustness[m]['psnr'],
                    color=colors.get(m, 'gray'),
                    marker=markers.get(m, 'o'),
                    linewidth=2.5, markersize=8,
                    label=m)
        
        ax1.set_xlabel('Registration Error Std (pixels)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('PSNR vs Registration Error', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([error_levels[0] - 0.05, error_levels[-1] + 0.05])
        
        # SSIM vs Registration Error
        for m in methods:
            ax2.plot(error_levels, robustness[m]['ssim'],
                    color=colors.get(m, 'gray'),
                    marker=markers.get(m, 'o'),
                    linewidth=2.5, markersize=8,
                    label=m)
        
        ax2.set_xlabel('Registration Error Std (pixels)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.set_title('SSIM vs Registration Error', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([error_levels[0] - 0.05, error_levels[-1] + 0.05])
        
        plt.suptitle('Super-Resolution Robustness to Registration Errors',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_visual_comparison(self, hr_image, model, 
                                    reg_error_std=0.5, save_path=None):
        """
        Generate a detailed visual comparison figure.
        
        Shows ground truth, all methods' results, and error maps
        to visualize where each method fails.
        """
        results, gt = self.evaluate_single_image(
            hr_image, model, reg_error_std=reg_error_std
        )
        
        methods = ['Bicubic', 'Mean Fusion', 'IBP', 'Ours (Robust ML)']
        n_methods = len(methods)
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, n_methods + 1, figure=fig, hspace=0.3, wspace=0.2)
        
        # Ground truth
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_gt.imshow(gt)
        ax_gt.set_title('Ground Truth', fontsize=11, fontweight='bold')
        ax_gt.axis('off')
        
        # Error map legend
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.text(0.5, 0.5, f'Reg Error\nStd = {reg_error_std}',
                      transform=ax_legend.transAxes, fontsize=14,
                      va='center', ha='center', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_legend.axis('off')
        
        for i, m in enumerate(methods):
            if m not in results:
                continue
            
            result = results[m]
            
            # Result image
            ax_img = fig.add_subplot(gs[0, i + 1])
            ax_img.imshow(result['image'])
            ax_img.set_title(f'{m}\nPSNR: {result["psnr"]:.2f} dB | '
                           f'SSIM: {result["ssim"]:.4f}',
                           fontsize=10, fontweight='bold')
            ax_img.axis('off')
            
            # Error map
            ax_err = fig.add_subplot(gs[1, i + 1])
            error = np.mean(np.abs(result['image'] - gt), axis=2)
            im = ax_err.imshow(error, cmap='hot', vmin=0, vmax=0.15)
            ax_err.set_title('Error Map', fontsize=10)
            ax_err.axis('off')
            plt.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04)
        
        plt.suptitle('Visual Comparison of Super-Resolution Methods',
                     fontsize=14, fontweight='bold')
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return results
    
    def full_evaluation(self, hr_image_paths, model, results_dir='results'):
        """
        Run the complete evaluation pipeline.
        
        Args:
            hr_image_paths: List of paths to HR test images.
            model: Trained SR model.
            results_dir: Directory to save results.
        
        Returns:
            All evaluation results.
        """
        ensure_dir(results_dir)
        
        # Load images
        hr_images = []
        for path in hr_image_paths:
            try:
                img = load_image(path, color='rgb')
                # Ensure reasonable size
                h, w = img.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)),
                                   interpolation=cv2.INTER_AREA)
                hr_images.append(img)
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
        
        if not hr_images:
            print("  Error: No valid images for evaluation!")
            return None
        
        print(f"\n  Evaluating on {len(hr_images)} images...")
        
        # 1. Visual comparisons for each image
        print("  Generating visual comparisons...")
        for i, img in enumerate(hr_images[:4]):  # First 4 images
            self.generate_visual_comparison(
                img, model, reg_error_std=0.5,
                save_path=os.path.join(results_dir, f'comparison_{i}.png')
            )
        
        # 2. Robustness analysis
        print("  Running robustness analysis...")
        robustness = self.robustness_analysis(
            hr_images[:4],  # Use subset for speed
            model,
            error_levels=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
            num_trials=2
        )
        
        self.plot_robustness_curves(
            robustness,
            save_path=os.path.join(results_dir, 'robustness_curves.png')
        )
        
        # 3. Summary metrics table
        print("\n  === Evaluation Results ===")
        print(f"  {'Method':<20} {'PSNR (dB)':<12} {'SSIM':<12}")
        print(f"  {'-'*44}")
        
        # Average across images at moderate error level (0.3)
        avg_results = {}
        for img in hr_images:
            results, _ = self.evaluate_single_image(
                img, model, reg_error_std=0.3
            )
            for m, r in results.items():
                if m not in avg_results:
                    avg_results[m] = {'psnr': [], 'ssim': []}
                avg_results[m]['psnr'].append(r['psnr'])
                avg_results[m]['ssim'].append(r['ssim'])
        
        for m in ['Bicubic', 'Mean Fusion', 'IBP', 'Ours (Robust ML)']:
            if m in avg_results:
                avg_psnr = np.mean(avg_results[m]['psnr'])
                avg_ssim = np.mean(avg_results[m]['ssim'])
                print(f"  {m:<20} {avg_psnr:<12.2f} {avg_ssim:<12.4f}")
        
        return {
            'robustness': robustness,
            'avg_results': avg_results,
        }
