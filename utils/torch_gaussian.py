import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianBlur3D(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1.0):
        super(GaussianBlur3D, self).__init__()
        self.padding = kernel_size // 2
        
        # Create 1D Gaussian kernel
        x_coord = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.
        
        # 1D kernel
        gaussian_1d = torch.exp(-((x_coord - mean) ** 2) / (2 * variance))
        gaussian_1d = gaussian_1d / torch.sum(gaussian_1d)  # Normalize
        
        # Create 3D kernel: outer product of 1D kernels
        kernel_3d = gaussian_1d.view(-1, 1, 1) * gaussian_1d.view(1, -1, 1) * gaussian_1d.view(1, 1, -1)
        
        # Expand for depthwise convolution: [channels, 1, k, k, k]
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k, k]
        kernel_3d = kernel_3d.repeat(channels, 1, 1, 1, 1)  # [channels, 1, k, k, k]
        
        self.register_buffer('weight', kernel_3d)
        self.groups = channels

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=self.padding, groups=self.groups)

def get_gaussian_layer(channels=1, sigma=1.0, device='cuda'):
    layer = GaussianBlur3D(channels=channels, kernel_size=7, sigma=sigma)
    return layer.to(device)
