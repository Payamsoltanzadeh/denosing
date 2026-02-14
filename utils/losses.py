import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    """
    Dose-dependent weighted MSE.
    
    Upweights high-dose regions so the model focuses on clinically
    important voxels rather than background/air.
    
    IMPORTANT: `dose_weight` should be the actual dose tensor
    (e.g. Gaussian-blurred LP), NOT the residual target. The residual
    target is in ×1000 space and its magnitude does not correspond to
    physical dose, so weighting by it would be semantically wrong.
    """
    def __init__(self, alpha=5.0, beta=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target, dose_weight=None):
        """
        Args:
            pred: predicted residual (×residual_scale space)
            target: target residual (×residual_scale space)
            dose_weight: ACTUAL dose tensor (e.g. lp_gaussian in [0,1])
                         used to compute spatially-varying weights.
                         If None, falls back to absolute target (legacy).
        """
        if dose_weight is not None:
            # Weight by actual dose level — physically correct
            bmax = dose_weight.view(dose_weight.size(0), -1).max(dim=1)[0]
            bmax = bmax.view(-1, 1, 1, 1, 1) + 1e-8
            norm_t = dose_weight / bmax
        else:
            # Legacy fallback: weight by target magnitude (not recommended)
            bmax = target.view(target.size(0), -1).max(dim=1)[0]
            bmax = bmax.view(-1, 1, 1, 1, 1) + 1e-8
            norm_t = target / bmax
        w = 1.0 + self.alpha * (norm_t ** self.beta)
        return (w * (pred - target) ** 2).mean()

class L4Loss(nn.Module):
    """
    L4 (fourth-power) loss for penalizing large errors.
    
    IMPORTANT: If inputs are in scaled residual space (e.g. ×1000),
    set residual_scale so that L4 is computed in dose space.
    Otherwise the 4th power on ×1000 values produces astronomically
    large gradients that dominate over WMSE.
    """
    def __init__(self, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
    def forward(self, pred, target):
        # Convert back to dose space before computing 4th power
        diff = (pred - target) / self.residual_scale
        return (diff ** 4).mean()

class HesserLoss(nn.Module):
    def __init__(self, l4_weight=0.1, dose_weight_alpha=5.0, residual_scale=1.0):
        super().__init__()
        self.l4_weight = l4_weight
        self.wmse = WeightedMSELoss(alpha=dose_weight_alpha)
        self.l4 = L4Loss(residual_scale=residual_scale)

    def forward(self, pred, target, dose_weight=None):
        """
        Args:
            pred: predicted residual
            target: target residual
            dose_weight: actual dose tensor for WMSE spatial weighting.
                         Pass lp_gaussian (Gaussian-blurred LP) here.
        """
        return self.wmse(pred, target, dose_weight=dose_weight) + \
               self.l4_weight * self.l4(pred, target)

class SobelGradient3DLoss(nn.Module):
    """
    Calculates the 1st order gradient difference (Sobel) between pred and target.
    Helper to sharpen edges which is critical for Gamma 1%/1mm.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # 3D Sobel kernels
        # x-direction
        sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                                [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32)
        # y-direction (transpose x)
        sobel_y = sobel_x.transpose(1, 2)
        # z-direction (transpose x to depth)
        sobel_z = sobel_x.transpose(0, 2)
        
        # Reshape to (Out, In, D, H, W) -> (1, 1, 3, 3, 3)
        self.kx = sobel_x.view(1, 1, 3, 3, 3).to(device)
        self.ky = sobel_y.view(1, 1, 3, 3, 3).to(device)
        self.kz = sobel_z.view(1, 1, 3, 3, 3).to(device)

    def forward(self, pred, target):
        # We assume pred/target are on self.device. Kernels must be too.
        # If input came from DataParallel, device might differ, so ensure kernels match input
        if pred.device != self.kx.device:
            self.kx = self.kx.to(pred.device)
            self.ky = self.ky.to(pred.device)
            self.kz = self.kz.to(pred.device)
            
        # Padding=1 keeps dimensions same
        pred_grad_x = F.conv3d(pred, self.kx, padding=1)
        pred_grad_y = F.conv3d(pred, self.ky, padding=1)
        pred_grad_z = F.conv3d(pred, self.kz, padding=1)

        target_grad_x = F.conv3d(target, self.kx, padding=1)
        target_grad_y = F.conv3d(target, self.ky, padding=1)
        target_grad_z = F.conv3d(target, self.kz, padding=1)

        # L1 loss on gradients often sharper than L2
        loss_dx = F.l1_loss(pred_grad_x, target_grad_x)
        loss_dy = F.l1_loss(pred_grad_y, target_grad_y)
        loss_dz = F.l1_loss(pred_grad_z, target_grad_z)

        return (loss_dx + loss_dy + loss_dz) / 3.0
