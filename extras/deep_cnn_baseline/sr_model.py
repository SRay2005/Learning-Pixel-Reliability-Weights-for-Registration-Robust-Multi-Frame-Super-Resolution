"""
Super-Resolution Neural Network Model.

Implements a multi-frame SR architecture that is designed to be robust
to registration errors. The architecture has three main stages:

  1. Feature Extraction: ResBlock-based backbone extracts rich features
     from each upsampled LR frame.
  
  2. Attention-Based Fusion: A temporal attention mechanism learns to
     weight frames based on their quality/alignment, automatically
     down-weighting poorly registered frames.
  
  3. Reconstruction: Residual learning produces the HR output as a
     refinement over bicubic upsampling.

Key Design Choices for Robustness:
  - Attention fusion naturally handles registration errors by learning
    to ignore misaligned features
  - Residual learning makes training stable even with noisy inputs
  - Charbonnier loss (external) doesn't amplify outlier errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection.
    
    The skip connection enables training very deep networks and
    helps preserve fine details from the input to the output.
    """
    
    def __init__(self, channels, use_bn=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        
        self.body = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.body(x)  # Skip connection


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    
    Learns to re-weight feature channels based on their importance,
    helping the network focus on the most informative features
    across the fused multi-frame representation.
    """
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class TemporalAttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple frame features.
    
    This is the KEY module for registration error robustness.
    Instead of simple averaging (which spreads registration errors),
    it learns attention weights for each frame at each spatial location.
    
    Poorly registered frames get low attention weights, so their
    misaligned information doesn't corrupt the final reconstruction.
    
    Architecture:
    - Each frame's features are projected to an attention score map
    - Softmax across frames gives per-pixel weights
    - Weighted sum produces the fused features
    """
    
    def __init__(self, channels, num_frames):
        super().__init__()
        self.num_frames = num_frames
        
        # Attention score computation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_frames, 1),  # One score per frame
        )
    
    def forward(self, frame_features):
        """
        Args:
            frame_features: List of tensors, each (B, C, H, W).
        
        Returns:
            Fused features (B, C, H, W).
        """
        # Concatenate all frame features
        concat = torch.cat(frame_features, dim=1)  # (B, N*C, H, W)
        
        # Compute attention scores
        scores = self.attention_conv(concat)  # (B, N, H, W)
        weights = F.softmax(scores, dim=1)    # Normalize across frames
        
        # Weighted fusion
        fused = torch.zeros_like(frame_features[0])
        for i in range(self.num_frames):
            w = weights[:, i:i+1, :, :]  # (B, 1, H, W)
            fused += w * frame_features[i]
        
        return fused


class RobustSRNetwork(nn.Module):
    """
    Multi-Frame Super-Resolution Network Robust to Registration Errors.
    
    Full architecture:
    
    Input: N upsampled LR frames stacked along channels → (B, N*3, H, W)
           ↓
    Feature Extraction: Per-frame feature extraction with shared weights
           ↓
    Temporal Attention Fusion: Learns to weight frames by quality
           ↓
    Deep Reconstruction: Residual blocks refine the fused features
           ↓
    Output: Residual image added to mean of inputs → HR output (B, 3, H, W)
    
    The residual learning (output = input + learned_residual) makes
    training much more stable, especially with noisy registration.
    """
    
    def __init__(self, num_frames=4, num_channels=64, num_res_blocks=8,
                 in_channels=3):
        """
        Args:
            num_frames: Number of input LR frames.
            num_channels: Feature channels in the network.
            num_res_blocks: Number of residual blocks in reconstruction.
            in_channels: Channels per frame (3 for RGB).
        """
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        
        # === Feature Extraction (shared across frames) ===
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ChannelAttention(num_channels),
        )
        
        # === Temporal Attention Fusion ===
        self.fusion = TemporalAttentionFusion(num_channels, num_frames)
        
        # === Deep Reconstruction ===
        reconstruction_blocks = []
        for _ in range(num_res_blocks):
            reconstruction_blocks.append(ResidualBlock(num_channels))
        reconstruction_blocks.append(ChannelAttention(num_channels))
        self.reconstruction = nn.Sequential(*reconstruction_blocks)
        
        # === Output Projection ===
        self.output_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Stacked upsampled LR frames (B, N*C, H, W).
        
        Returns:
            HR reconstruction (B, C, H, W).
        """
        B, _, H, W = x.shape
        C = self.in_channels
        N = self.num_frames
        
        # Split into individual frames
        frames = []
        for i in range(N):
            frame = x[:, i*C:(i+1)*C, :, :]  # (B, C, H, W)
            frames.append(frame)
        
        # Compute mean of inputs for residual learning
        input_mean = torch.stack(frames, dim=0).mean(dim=0)  # (B, C, H, W)
        
        # Extract features from each frame (weight sharing)
        frame_features = []
        for frame in frames:
            feat = self.feature_extract(frame)
            frame_features.append(feat)
        
        # Attention-based fusion
        fused = self.fusion(frame_features)
        
        # Deep reconstruction
        reconstructed = self.reconstruction(fused)
        
        # Output projection + residual connection
        output = self.output_conv(reconstructed)
        hr = output + input_mean  # Residual learning
        
        return torch.clamp(hr, 0, 1)


class SimpleSRNetwork(nn.Module):
    """
    Simplified SR network WITHOUT attention fusion (for comparison).
    
    This uses simple concatenation + convolution fusion, which is
    more susceptible to registration errors. Used as a baseline
    to demonstrate the benefit of attention-based fusion.
    """
    
    def __init__(self, num_frames=4, num_channels=64, num_res_blocks=8,
                 in_channels=3):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        
        # Direct concatenation fusion
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_frames, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Reconstruction
        blocks = [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        self.reconstruction = nn.Sequential(*blocks)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x):
        B, _, H, W = x.shape
        C = self.in_channels
        N = self.num_frames
        
        frames = [x[:, i*C:(i+1)*C, :, :] for i in range(N)]
        input_mean = torch.stack(frames, dim=0).mean(dim=0)
        
        feat = self.input_conv(x)
        reconstructed = self.reconstruction(feat)
        output = self.output_conv(reconstructed)
        
        return torch.clamp(output + input_mean, 0, 1)


def get_model(model_type='robust', num_frames=4, num_channels=64,
              num_res_blocks=8):
    """
    Factory function to create SR models.
    
    Args:
        model_type: 'robust' (with attention) or 'simple' (without).
        num_frames: Number of input LR frames.
        num_channels: Feature channel count.
        num_res_blocks: Depth of reconstruction network.
    
    Returns:
        PyTorch model.
    """
    if model_type == 'robust':
        return RobustSRNetwork(num_frames, num_channels, num_res_blocks)
    elif model_type == 'simple':
        return SimpleSRNetwork(num_frames, num_channels, num_res_blocks)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
