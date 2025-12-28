"""
Custom layers for Physicist agents.

Implements BRST-constrained layers from the Hypostructure framework.
"""

import torch
import torch.nn as nn
from typing import Optional


class BRSTLinear(nn.Module):
    """
    Linear layer with near-orthogonal weight constraint (BRST defect).

    The BRST constraint forces W^T W ≈ I (or W W^T ≈ I for rectangular),
    ensuring the transformation preserves geometric structure.

    This is useful for:
    - Encoder layers: Preserves distances in latent space
    - Physics engines: Ensures volume-preserving dynamics (Liouville's theorem)

    Usage:
        layer = BRSTLinear(64, 32)
        output = layer(input)
        defect = layer.brst_defect()  # Add to loss with weight λ_brst
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def brst_defect(self) -> torch.Tensor:
        """
        Compute the BRST defect: ||W^T W - I||² or ||W W^T - I||².

        For rectangular W (m x n):
        - If m >= n: compute W^T W (n x n) vs I_n
        - If m < n: compute W W^T (m x m) vs I_m

        Returns:
            Scalar tensor with the orthogonality defect
        """
        W = self.linear.weight  # Shape: [out_features, in_features]

        if W.shape[0] >= W.shape[1]:
            # More outputs than inputs: W^T W should be identity
            gram = torch.matmul(W.t(), W)
            target = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        else:
            # More inputs than outputs: W W^T should be identity
            gram = torch.matmul(W, W.t())
            target = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)

        return torch.norm(gram - target) ** 2

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias


class BRSTConv2d(nn.Module):
    """
    Conv2d layer with near-orthogonal weight constraint (BRST defect).

    Treats the conv kernel as a matrix and applies BRST constraint.
    For kernel W of shape [out_channels, in_channels, kH, kW]:
    - Reshape to [out_channels, in_channels * kH * kW]
    - Apply BRST constraint on this reshaped matrix
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def brst_defect(self) -> torch.Tensor:
        """
        Compute BRST defect for conv weights.

        Reshapes [out_channels, in_channels, kH, kW] to [out_channels, -1]
        then computes orthogonality defect.
        """
        W = self.conv.weight  # [out, in, kH, kW]
        W_flat = W.view(W.shape[0], -1)  # [out, in * kH * kW]

        if W_flat.shape[0] >= W_flat.shape[1]:
            gram = torch.matmul(W_flat.t(), W_flat)
            target = torch.eye(W_flat.shape[1], device=W.device, dtype=W.dtype)
        else:
            gram = torch.matmul(W_flat, W_flat.t())
            target = torch.eye(W_flat.shape[0], device=W.device, dtype=W.dtype)

        return torch.norm(gram - target) ** 2

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.conv.bias


def compute_total_brst_defect(module: nn.Module) -> torch.Tensor:
    """
    Compute total BRST defect across all BRST layers in a module.

    Args:
        module: PyTorch module containing BRSTLinear/BRSTConv2d layers

    Returns:
        Sum of all BRST defects
    """
    total = torch.tensor(0.0)
    device = None

    for m in module.modules():
        if isinstance(m, (BRSTLinear, BRSTConv2d)):
            defect = m.brst_defect()
            if device is None:
                device = defect.device
                total = total.to(device)
            total = total + defect

    return total


def test_brst_layers():
    """Test BRST layer implementations."""
    print("Testing BRST layers...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test BRSTLinear
    layer = BRSTLinear(64, 32).to(device)
    x = torch.randn(8, 64, device=device)
    y = layer(x)
    defect = layer.brst_defect()
    print(f"  BRSTLinear: input={x.shape}, output={y.shape}, defect={defect.item():.4f}")

    # Test square layer (should have lower defect with orthogonal init)
    layer_sq = BRSTLinear(32, 32).to(device)
    nn.init.orthogonal_(layer_sq.linear.weight)
    defect_sq = layer_sq.brst_defect()
    print(f"  BRSTLinear (orthogonal init): defect={defect_sq.item():.6f}")

    # Test BRSTConv2d
    conv = BRSTConv2d(1, 32, 4, stride=2, padding=1).to(device)
    x_img = torch.randn(8, 1, 64, 64, device=device)
    y_img = conv(x_img)
    defect_conv = conv.brst_defect()
    print(f"  BRSTConv2d: input={x_img.shape}, output={y_img.shape}, defect={defect_conv.item():.4f}")

    # Test total defect computation
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = BRSTLinear(64, 32)
            self.l2 = BRSTLinear(32, 16)
            self.regular = nn.Linear(16, 8)  # Not BRST

        def forward(self, x):
            return self.regular(self.l2(self.l1(x)))

    model = TestModule().to(device)
    total_defect = compute_total_brst_defect(model)
    print(f"  Total defect (2 BRST layers): {total_defect.item():.4f}")

    print("All BRST tests passed!")


if __name__ == "__main__":
    test_brst_layers()
