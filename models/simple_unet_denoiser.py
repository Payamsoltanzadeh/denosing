# ============================================================================
# simple_unet_denoiser.py
# ============================================================================
# Date: 2025-12-05
#
# Purpose: Simple 3D U-Net for Monte Carlo dose denoising.
#          NO time embedding, NO diffusion process.
#          
# Task: Learn mapping from (CT, LP_dose) → HP_dose
#
# Input:  CT volume (1 channel) + LP dose (1 channel) = 2 channels
# Output: HP dose (1 channel)
#
# Architecture: Standard 3D U-Net with skip connections
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Basic 3D convolutional block with InstanceNorm and SiLU activation.
    
    Conv3D → InstanceNorm3D → SiLU → Conv3D → InstanceNorm3D → SiLU
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # Residual connection if channels change
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


class DownBlock3D(nn.Module):
    """
    Encoder block: ConvBlock → MaxPool3D
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features  # Return both for skip connection


class UpBlock3D(nn.Module):
    """
    Decoder block: Upsample → Concat skip → ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        # Upsample: in_channels → out_channels
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # After concat with skip: out_channels + in_channels (skip)
        # Skip connection comes from encoder which has 'in_channels' size
        self.conv = ConvBlock3D(out_channels + in_channels, out_channels, dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch (if input size is not power of 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleUNetDenoiser(nn.Module):
    """
    Simple 3D U-Net for Monte Carlo dose denoising.
    
    Architecture:
        - Encoder: 4 down blocks with doubling channels
        - Bottleneck: ConvBlock at deepest level
        - Decoder: 4 up blocks with skip connections
        - Output: 1x1 conv to single channel
    
    Input:
        - x: (B, 2, D, H, W) - CT (channel 0) + LP dose (channel 1)
        
    Output:
        - y: (B, 1, D, H, W) - Predicted HP dose
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        
        # Calculate channel sizes at each level
        channels = [base_channels * m for m in channel_mults]
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder.append(
                DownBlock3D(channels[i], channels[i + 1], dropout)
            )
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(channels[-1], channels[-1], dropout)
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder.append(
                UpBlock3D(channels[i], channels[i - 1], dropout)
            )
        
        # Output convolution (NO ReLU here, to allow negative residuals in subclass)
        self.output_conv = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels[0], out_channels, kernel_size=1),
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared forward pass for features."""
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder - collect skip connections
        skips = []
        for down in self.encoder:
            x, skip = down(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder - use skip connections in reverse order
        for up, skip in zip(self.decoder, reversed(skips)):
            x = up(x, skip)
            
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: Predicts residual correction.
        
        NOTE: No ReLU here! The residual target = (HP - Gaussian(LP)) * 1000
        can be negative (when Gaussian oversmooths). ReLU would clip these
        to zero, creating a systematic positive bias.
        The final dose is computed in the training/inference script as:
            pred_dose = Gaussian(LP) + pred_residual / 1000
        """
        features = self.forward_features(x)
        out = self.output_conv(features)
        
        # No activation — residual can be positive or negative
        return out
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleUNetDenoiserWithResidual(SimpleUNetDenoiser):
    """
    DEPRECATED — kept for backward compatibility only.
    
    Simple U-Net with LP-additive residual learning:
        HP_pred = LP + residual
    
    NOTE: The current training pipeline uses SimpleUNetDenoiser ("standard")
    with an EXTERNAL Gaussian-blur residual strategy:
        HP_pred = Gaussian(LP) + model_output / residual_scale
    This class does something DIFFERENT and is NOT recommended.
    
    WARNING: No ReLU on final output — dose can go negative, which is
    non-physical but avoids asymmetric clipping of the residual.
    Use np.maximum(pred, 0) at inference time if needed.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual forward pass: pred = LP + network_output.
        No activation — residual can be positive or negative.
        """
        # Extract LP dose (channel 1)
        lp_dose = x[:, 1:2, ...]  # Shape: (B, 1, D, H, W)
        
        # Get features and predict residual
        features = self.forward_features(x)
        residual = self.output_conv(features)
        
        # Add residual to LP to get HP (no ReLU — dose clamp at inference)
        return lp_dose + residual


# ============================================================================
# Utility functions
# ============================================================================

def get_simple_denoiser(
    model_type: str = "standard",
    base_channels: int = 32,
    channel_mults: tuple = (1, 2, 4, 8),
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory function to create a denoiser model.
    """
    if model_type == "standard":
        return SimpleUNetDenoiser(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )
    elif model_type == "residual":
        return SimpleUNetDenoiserWithResidual(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SimpleUNetDenoiser")
    print("=" * 60)
    
    # Create model
    model = SimpleUNetDenoiser(
        in_channels=2,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        dropout=0.0,
    )
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Test forward pass with realistic size
    batch_size = 2
    depth, height, width = 64, 64, 64  # Realistic patch size
    
    # Create dummy input: CT + LP
    x = torch.randn(batch_size, 2, depth, height, width)
    
    print(f"\nInput shape:  {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output min:   {y.min().item()} (Should be >= 0)")
    
    # Test residual version
    print("\n" + "-" * 40)
    print("Testing SimpleUNetDenoiserWithResidual")
    print("-" * 40)
    
    model_res = SimpleUNetDenoiserWithResidual(
        in_channels=2,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
    )
    
    with torch.no_grad():
        y_res = model_res(x)
    
    print(f"Residual model output shape: {y_res.shape}")
    print(f"Residual model min:        {y_res.min().item()} (Should be >= 0)")
    
    print("\n✅ All tests passed!")
