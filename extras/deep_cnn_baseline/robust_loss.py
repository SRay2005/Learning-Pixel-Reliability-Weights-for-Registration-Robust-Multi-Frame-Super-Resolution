"""
Robust Loss Functions for Super-Resolution.

Standard MSE loss amplifies the effect of registration errors because
misaligned pixels create large squared errors. These robust alternatives
are less sensitive to outliers (misaligned regions), which is KEY to
building SR systems that tolerate registration inaccuracies.

Implements:
  1. Charbonnier Loss (smooth L1) — robust to outliers
  2. Huber Loss — quadratic for small errors, linear for large
  3. Perceptual Loss — feature-space similarity for visual quality
  4. Combined Multi-Term Loss — balances pixel accuracy and visual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier penalty function (differentiable variant of L1).
    
    L(x) = sqrt(x^2 + eps^2)
    
    This is a smooth approximation to the L1 norm that:
    - Behaves like L2 for small errors (good gradient near zero)
    - Behaves like L1 for large errors (robust to outliers/misalignment)
    
    The epsilon parameter controls the transition point.
    This is widely used in robust SR because registration errors
    cause large localized pixel differences that would dominate MSE.
    """
    
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps_sq = eps ** 2
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber loss — piecewise quadratic/linear.
    
    L(x) = 0.5 * x^2           if |x| <= delta
           delta * |x| - 0.5*delta^2  otherwise
    
    Provides explicit control over the outlier threshold via delta.
    Good for SR when you know the approximate magnitude of registration errors.
    """
    
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        quadratic = 0.5 * diff ** 2
        linear = self.delta * diff - 0.5 * self.delta ** 2
        loss = torch.where(diff <= self.delta, quadratic, linear)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature extraction.
    
    Instead of comparing pixels directly (which is sensitive to
    small shifts), this compares high-level features that are more
    invariant to small spatial displacements.
    
    Uses features from early VGG layers (relu2_2) which capture
    textures and edges — important for perceptual quality.
    """
    
    def __init__(self, layer_idx=8):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:layer_idx+1]
        ).eval()
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG normalization constants
        self.register_buffer('mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # Normalize to VGG input range
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        return F.mse_loss(pred_features, target_features)


class EdgeLoss(nn.Module):
    """
    Edge-preserving loss using Sobel operator.
    
    Compares the gradient (edge) structure between predicted and target
    images. This encourages sharp edges in the SR output, which are
    often lost due to registration error blurring.
    """
    
    def __init__(self):
        super().__init__()
        # 3-channel grouped Sobel kernels for fast, robust edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_edges(self, img):
        """Compute edge magnitude using vectorized Sobel operator."""
        gx = F.conv2d(img, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(img, self.sobel_y, padding=1, groups=3)
        edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-4)
        return edge_mag
    
    def forward(self, pred, target):
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        return F.l1_loss(pred_edges, target_edges)


class RobustSRLoss(nn.Module):
    """
    Combined loss function for robust super-resolution.
    
    Balances:
    - Charbonnier loss: Pixel-level accuracy, robust to outliers
    - Edge loss: Preserves sharp edges
    - Perceptual loss: High-level visual quality (optional)
    
    The weighting is designed so that the model focuses on getting
    the overall structure right (Charbonnier) while maintaining
    sharpness (Edge) and visual quality (Perceptual).
    """
    
    def __init__(self, pixel_weight=1.0, edge_weight=0.1, 
                 perceptual_weight=0.0, use_perceptual=False):
        super().__init__()
        self.pixel_loss = CharbonnierLoss(eps=1e-3)
        self.edge_loss = EdgeLoss()
        self.pixel_weight = pixel_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        loss = self.pixel_weight * self.pixel_loss(pred, target)
        loss += self.edge_weight * self.edge_loss(pred, target)
        
        if self.use_perceptual and self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual_loss(pred, target)
        
        return loss
