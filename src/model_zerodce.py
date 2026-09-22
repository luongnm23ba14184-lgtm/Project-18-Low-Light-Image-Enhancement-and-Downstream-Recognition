"""
Zero-DCE and Zero-DCE++ Model Architectures and LE-Curve Implementation.
References:
- Zero-DCE: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" (CVPR 2020)
- Zero-DCE++: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation" (TPAMI 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def enhance_image(x: torch.Tensor, A: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """
    Applies the higher-order Light-Enhancement Curve (LE-Curve) iteratively.
    
    Formula:
        LE_n(x) = LE_{n-1}(x) + A_n(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
    
    Args:
        x: Input normalized image tensor of shape (B, 3, H, W), range [0, 1].
        A: Predicted curve parameter maps of shape (B, 3 * iterations, H, W) or (B, iterations, 1, H, W).
           Values in range [-1, 1].
        iterations: Number of curve iterations (default: 8).
        
    Returns:
        Enhanced image tensor of shape (B, 3, H, W), guaranteed range [0, 1].
    """
    enhanced = x
    # If A has 24 channels (3 channels * 8 iterations)
    if A.shape[1] == iterations * 3:
        for i in range(iterations):
            A_i = A[:, i * 3 : (i + 1) * 3, :, :]
            enhanced = enhanced + A_i * enhanced * (1.0 - enhanced)
    elif A.shape[1] == iterations:
        # Each iteration predicts a 1-channel adjustment broadcast across 3 channels
        for i in range(iterations):
            A_i = A[:, i : i + 1, :, :]
            enhanced = enhanced + A_i * enhanced * (1.0 - enhanced)
    else:
        raise ValueError(f"Shape mismatch for parameter map A: expected {iterations*3} or {iterations} channels, got {A.shape[1]}")
    
    return torch.clamp(enhanced, 0.0, 1.0)


class DCENet(nn.Module):
    """
    DCE-Net Architecture (CVPR 2020):
    A 7-layer symmetrical convolutional network with skip connections to preserve
    multi-scale spatial information. Estimates 24 curve parameter maps for 8 iterations.
    Total parameters: ~79K.
    """
    def __init__(self, num_channels: int = 32, num_iterations: int = 8):
        super(DCENet, self).__init__()
        self.num_iterations = num_iterations
        out_channels = num_iterations * 3  # 8 * 3 = 24

        self.relu = nn.ReLU(inplace=True)

        # Layers 1 to 4: Feature extraction & progressive deepening
        self.e_conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # Layers 5 to 7: Reconstruction with symmetric skip connections
        # conv5 inputs: conv4 + conv3 (32 + 32 = 64)
        self.e_conv5 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # conv6 inputs: conv5 + conv2 (32 + 32 = 64)
        self.e_conv6 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # conv7 inputs: conv6 + conv1 (32 + 32 = 64)
        self.e_conv7 = nn.Conv2d(num_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Low-light input image tensor (B, 3, H, W) in [0, 1].
            
        Returns:
            enhanced_image: (B, 3, H, W) enhanced image.
            A: (B, 24, H, W) curve parameter maps.
        """
        # Feature extraction
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # Skip connections
        x5 = self.relu(self.e_conv5(torch.cat([x4, x3], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x5, x2], dim=1)))
        A = self.tanh(self.e_conv7(torch.cat([x6, x1], dim=1)))

        enhanced = enhance_image(x, A, self.num_iterations)
        return enhanced, A


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block:
    Separates spatial filtering (Depthwise Conv) from channel mixing (Pointwise Conv 1x1).
    Drastically cuts computational complexity and parameters.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ZeroDCEpp(nn.Module):
    """
    Zero-DCE++ Architecture (TPAMI 2021):
    Ultra-lightweight variant employing Depthwise Separable Convolutions.
    Parameters reduced to ~10K, achieving >100 FPS on edge/consumer GPUs.
    """
    def __init__(self, num_channels: int = 32, num_iterations: int = 8):
        super(ZeroDCEpp, self).__init__()
        self.num_iterations = num_iterations
        out_channels = num_iterations * 3

        self.relu = nn.ReLU(inplace=True)

        # Depthwise Separable conv layers
        self.d_conv1 = DepthwiseSeparableConv(3, num_channels)
        self.d_conv2 = DepthwiseSeparableConv(num_channels, num_channels)
        self.d_conv3 = DepthwiseSeparableConv(num_channels, num_channels)
        self.d_conv4 = DepthwiseSeparableConv(num_channels, num_channels)
        self.d_conv5 = DepthwiseSeparableConv(num_channels * 2, num_channels)
        self.d_conv6 = DepthwiseSeparableConv(num_channels * 2, num_channels)
        self.d_conv7 = DepthwiseSeparableConv(num_channels * 2, out_channels)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.d_conv1(x))
        x2 = self.relu(self.d_conv2(x1))
        x3 = self.relu(self.d_conv3(x2))
        x4 = self.relu(self.d_conv4(x3))

        x5 = self.relu(self.d_conv5(torch.cat([x4, x3], dim=1)))
        x6 = self.relu(self.d_conv6(torch.cat([x5, x2], dim=1)))
        A = self.tanh(self.d_conv7(torch.cat([x6, x1], dim=1)))

        enhanced = enhance_image(x, A, self.num_iterations)
        return enhanced, A
