"""
Self-Supervised Non-Reference Loss Functions for Zero-DCE / Zero-DCE++.
Implemented from analytical mathematical formulations:
1. Spatial Consistency Loss (L_spa)
2. Exposure Control Loss (L_exp)
3. Color Constancy Loss (L_color)
4. Illumination Smoothness Loss (L_TV_A)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L_spa(nn.Module):
    """
    Spatial Consistency Loss (L_spa):
    Preserves spatial gradient differences between adjacent local regions (4 directions:
    up, down, left, right) between input image I and enhanced image Y.
    Prevents loss of fine textures, edge distortion, and halo artifacts.
    """
    def __init__(self, patch_size: int = 4):
        super(L_spa, self).__init__()
        self.patch_size = patch_size

        # Directional gradient kernels: Left, Right, Up, Down
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            org: Original input image tensor (B, 3, H, W)
            enhance: Enhanced image tensor (B, 3, H, W)
        """
        # Average pooling across local patch
        org_mean = torch.mean(org, dim=1, keepdim=True)
        enhance_mean = torch.mean(enhance, dim=1, keepdim=True)

        org_pool = F.avg_pool2d(org_mean, kernel_size=self.patch_size, stride=self.patch_size)
        enhance_pool = F.avg_pool2d(enhance_mean, kernel_size=self.patch_size, stride=self.patch_size)

        # Compute directional differences on original image
        d_org_l = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_r = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_u = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_d = F.conv2d(org_pool, self.weight_down, padding=1)

        # Compute directional differences on enhanced image
        d_enh_l = F.conv2d(enhance_pool, self.weight_left, padding=1)
        d_enh_r = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enh_u = F.conv2d(enhance_pool, self.weight_up, padding=1)
        d_enh_d = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # Loss is sum of squared differences of directional gradients
        diff_l = torch.pow(d_org_l - d_enh_l, 2)
        diff_r = torch.pow(d_org_r - d_enh_r, 2)
        diff_u = torch.pow(d_org_u - d_enh_u, 2)
        diff_d = torch.pow(d_org_d - d_enh_d, 2)

        return torch.mean(diff_l + diff_r + diff_u + diff_d)


class L_exp(nn.Module):
    """
    Exposure Control Loss (L_exp):
    Measures the distance between the local average intensity of local patches
    (default: 16x16) and a predefined target exposure level E (default: 0.6).
    Restrains over-exposure in bright areas while boosting under-exposed dark areas.
    """
    def __init__(self, patch_size: int = 16, target_exposure: float = 0.6):
        super(L_exp, self).__init__()
        self.patch_size = patch_size
        self.target_exposure = target_exposure

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Enhanced image tensor (B, 3, H, W)
        """
        # Convert RGB to grayscale/luminance mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        # Average intensity in local patches
        patch_mean = F.avg_pool2d(x_mean, kernel_size=self.patch_size, stride=self.patch_size)
        loss = torch.mean(torch.abs(patch_mean - self.target_exposure))
        return loss


class L_color(nn.Module):
    """
    Color Constancy Loss (L_color):
    Based on the Gray-World Hypothesis, which assumes that the average reflectance
    in a natural scene is achromatic (gray). Penalizes deviations between the mean
    values of R, G, and B color channels to eliminate color casts.
    """
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Enhanced image tensor (B, 3, H, W)
        """
        # Mean across spatial dimensions for each channel
        mean_rgb = torch.mean(x, dim=[2, 3])  # (B, 3)
        r_mean = mean_rgb[:, 0]
        g_mean = mean_rgb[:, 1]
        b_mean = mean_rgb[:, 2]

        diff_rg = torch.pow(r_mean - g_mean, 2)
        diff_rb = torch.pow(r_mean - b_mean, 2)
        diff_gb = torch.pow(g_mean - b_mean, 2)

        loss = torch.mean(torch.sqrt(diff_rg + diff_rb + diff_gb + 1e-8))
        return loss


class L_TV_A(nn.Module):
    """
    Illumination Smoothness Loss (Total Variation Loss on Parameter Maps A):
    Imposes smoothness constraints on the predicted curve parameter maps A.
    Penalizes sharp first-order spatial gradients to preserve monotonic light transitions
    and prevent stripe or block artifacts.
    """
    def __init__(self):
        super(L_TV_A, self).__init__()

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: Predicted curve parameter maps (B, C, H, W)
        """
        b, c, h, w = A.shape
        # Total Variation along vertical (H) and horizontal (W) dimensions
        tv_h = torch.pow(A[:, :, 1:, :] - A[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(A[:, :, :, 1:] - A[:, :, :, :-1], 2).sum()
        count_h = (h - 1) * w
        count_w = h * (w - 1)
        loss = (tv_h / count_h + tv_w / count_w) / (b * c)
        return loss


class ZeroDCELoss(nn.Module):
    """
    Unified Non-Reference Zero-DCE Loss Engine:
    L_total = w_spa * L_spa + w_exp * L_exp + w_col * L_col + w_tv * L_tv_A
    """
    def __init__(
        self,
        w_spa: float = 1.0,
        w_exp: float = 10.0,
        w_col: float = 5.0,
        w_tv: float = 200.0,
        target_exposure: float = 0.6,
    ):
        super(ZeroDCELoss, self).__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tv = w_tv

        self.loss_spa = L_spa()
        self.loss_exp = L_exp(target_exposure=target_exposure)
        self.loss_col = L_color()
        self.loss_tv = L_TV_A()

    def forward(
        self,
        org: torch.Tensor,
        enhance: torch.Tensor,
        A: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes total loss and returns breakdown for logging.
        """
        l_spa = self.loss_spa(org, enhance)
        l_exp = self.loss_exp(enhance)
        l_col = self.loss_col(enhance)
        l_tv = self.loss_tv(A)

        total_loss = (
            self.w_spa * l_spa +
            self.w_exp * l_exp +
            self.w_col * l_col +
            self.w_tv * l_tv
        )

        breakdown = {
            "loss_total": float(total_loss.item()),
            "loss_spa": float(l_spa.item()),
            "loss_exp": float(l_exp.item()),
            "loss_col": float(l_col.item()),
            "loss_tv": float(l_tv.item()),
        }
        return total_loss, breakdown
