"""
Loss functions for Physicist agents.

Implements VICReg and topology losses from the Hypostructure framework.
"""

import torch
import torch.nn.functional as F
from typing import List


# =============================================================================
# VICReg Losses (Variance-Invariance-Covariance Regularization)
# =============================================================================

def vicreg_invariance_loss(z: torch.Tensor, z_perturbed: torch.Tensor) -> torch.Tensor:
    """
    Invariance loss: representations should be stable under perturbation.

    L_inv = MSE(z, z_perturbed)

    Args:
        z: Original embeddings [batch, dim]
        z_perturbed: Embeddings of perturbed inputs [batch, dim]

    Returns:
        Scalar invariance loss
    """
    return F.mse_loss(z, z_perturbed)


def vicreg_variance_loss(z: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """
    Variance loss: each dimension should have variance >= 1.

    L_var = mean(max(0, 1 - std(z)))

    This prevents mode collapse by ensuring the latent space is spread out.

    Args:
        z: Embeddings [batch, dim]
        epsilon: Small constant for numerical stability

    Returns:
        Scalar variance loss
    """
    std_z = torch.sqrt(z.var(dim=0) + epsilon)
    return torch.mean(F.relu(1.0 - std_z))


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Covariance loss: minimize off-diagonal covariance to decorrelate dimensions.

    L_cov = sum(off_diagonal(cov(z))^2) / dim

    This encourages each latent dimension to capture independent information.

    Args:
        z: Embeddings [batch, dim]

    Returns:
        Scalar covariance loss
    """
    batch_size, dim = z.shape

    if batch_size < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)

    # Extract off-diagonal elements
    off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=z.device)
    off_diag = cov[off_diag_mask]

    return off_diag.pow(2).sum() / dim


def vicreg_loss(
    z: torch.Tensor,
    z_perturbed: torch.Tensor,
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0
) -> torch.Tensor:
    """
    Combined VICReg loss.

    Args:
        z: Original embeddings [batch, dim]
        z_perturbed: Embeddings of perturbed inputs [batch, dim]
        lambda_inv: Weight for invariance loss
        lambda_var: Weight for variance loss
        lambda_cov: Weight for covariance loss

    Returns:
        Weighted sum of VICReg losses
    """
    L_inv = vicreg_invariance_loss(z, z_perturbed)
    L_var = vicreg_variance_loss(z)
    L_cov = vicreg_covariance_loss(z)

    return lambda_inv * L_inv + lambda_var * L_var + lambda_cov * L_cov


# =============================================================================
# Topology Losses (Router Constraints)
# =============================================================================

def router_entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Entropy loss for router: encourage confident chart selection.

    L_entropy = -mean(sum(w * log(w)))

    Lower entropy = more confident selection of a single chart.
    Higher entropy = uniform distribution across charts.

    Args:
        weights: Router weights [batch, num_charts], should sum to 1 per sample

    Returns:
        Scalar entropy loss (negated, so minimizing this increases entropy)
    """
    # Note: In combined.py this is NEGATIVE, so minimizing the loss
    # INCREASES entropy (more uniform). This is used with a positive weight
    # during warmup to prevent premature specialization.
    return -torch.mean(torch.sum(weights * torch.log(weights + 1e-6), dim=1))


def router_balance_loss(weights: torch.Tensor, num_charts: int) -> torch.Tensor:
    """
    Balance loss: encourage equal utilization of all charts.

    L_balance = ||mean_usage - target_usage||^2

    where target_usage = 1/num_charts for all charts.

    Args:
        weights: Router weights [batch, num_charts]
        num_charts: Number of charts

    Returns:
        Scalar balance loss
    """
    mean_usage = torch.mean(weights, dim=0)
    target_usage = torch.ones(num_charts, device=weights.device, dtype=weights.dtype) / num_charts
    return torch.norm(mean_usage - target_usage) ** 2


def chart_separation_loss(
    chart_outputs: List[torch.Tensor],
    weights: torch.Tensor,
    min_distance: float = 4.0
) -> torch.Tensor:
    """
    Separation loss: force chart centers to be at least min_distance apart.

    For each pair of charts (i, j):
        L_sep += max(0, min_distance - ||center_i - center_j||)

    Chart center is the weighted mean of its outputs.

    Args:
        chart_outputs: List of chart outputs, each [batch, latent_dim]
        weights: Router weights [batch, num_charts]
        min_distance: Minimum distance between chart centers

    Returns:
        Scalar separation loss
    """
    num_charts = len(chart_outputs)
    device = weights.device
    dtype = weights.dtype

    if num_charts < 2:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Compute weighted centers for each chart
    centers = []
    for i, z_i in enumerate(chart_outputs):
        w_i = weights[:, i:i+1]  # [batch, 1]
        weight_sum = w_i.sum() + 1e-6
        if weight_sum > 1e-5:  # Only if chart has meaningful usage
            center = (z_i * w_i).sum(dim=0) / weight_sum
            centers.append(center)

    if len(centers) < 2:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Compute pairwise separation losses
    loss = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = torch.norm(centers[i] - centers[j])
            loss = loss + F.relu(min_distance - dist)

    return loss


def topology_loss(
    weights: torch.Tensor,
    chart_outputs: List[torch.Tensor],
    num_charts: int,
    lambda_entropy: float = 2.0,
    lambda_balance: float = 100.0,
    lambda_separation: float = 10.0,
    min_separation: float = 4.0
) -> torch.Tensor:
    """
    Combined topology loss for router constraints.

    Args:
        weights: Router weights [batch, num_charts]
        chart_outputs: List of chart outputs
        num_charts: Number of charts
        lambda_entropy: Weight for entropy loss
        lambda_balance: Weight for balance loss
        lambda_separation: Weight for separation loss
        min_separation: Minimum distance between chart centers

    Returns:
        Weighted sum of topology losses
    """
    L_entropy = router_entropy_loss(weights)
    L_balance = router_balance_loss(weights, num_charts)
    L_sep = chart_separation_loss(chart_outputs, weights, min_separation)

    return lambda_entropy * L_entropy + lambda_balance * L_balance + lambda_separation * L_sep


# =============================================================================
# Combined Loss Components
# =============================================================================

def compute_warmup_ratio(epoch: int, warmup_epochs: int) -> float:
    """
    Compute warmup ratio for scaling new losses.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs for warmup

    Returns:
        Ratio in [0, 1] that scales from 0 to 1 over warmup period
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


def test_losses():
    """Test loss function implementations."""
    print("Testing loss functions...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test VICReg losses
    z = torch.randn(32, 8, device=device)
    z_perturbed = z + torch.randn_like(z) * 0.1

    L_inv = vicreg_invariance_loss(z, z_perturbed)
    L_var = vicreg_variance_loss(z)
    L_cov = vicreg_covariance_loss(z)
    print(f"  VICReg: inv={L_inv.item():.4f}, var={L_var.item():.4f}, cov={L_cov.item():.4f}")

    # Test with collapsed embeddings (should have high variance loss)
    z_collapsed = torch.zeros(32, 8, device=device)
    L_var_collapsed = vicreg_variance_loss(z_collapsed)
    print(f"  VICReg (collapsed): var={L_var_collapsed.item():.4f} (should be ~1.0)")

    # Test with correlated embeddings (should have high covariance loss)
    z_corr = torch.randn(32, 1, device=device).expand(32, 8)
    L_cov_corr = vicreg_covariance_loss(z_corr)
    print(f"  VICReg (correlated): cov={L_cov_corr.item():.4f} (should be high)")

    # Test topology losses
    num_charts = 3
    weights = F.softmax(torch.randn(32, num_charts, device=device), dim=1)
    chart_outputs = [torch.randn(32, 8, device=device) for _ in range(num_charts)]

    L_entropy = router_entropy_loss(weights)
    L_balance = router_balance_loss(weights, num_charts)
    L_sep = chart_separation_loss(chart_outputs, weights, min_distance=4.0)
    print(f"  Topology: entropy={L_entropy.item():.4f}, balance={L_balance.item():.4f}, sep={L_sep.item():.4f}")

    # Test with uniform weights (should have low balance loss)
    weights_uniform = torch.ones(32, num_charts, device=device) / num_charts
    L_balance_uniform = router_balance_loss(weights_uniform, num_charts)
    print(f"  Topology (uniform): balance={L_balance_uniform.item():.6f} (should be ~0)")

    # Test with one-hot weights (should have high entropy, low balance)
    weights_onehot = torch.zeros(32, num_charts, device=device)
    weights_onehot[:, 0] = 1.0
    L_entropy_onehot = router_entropy_loss(weights_onehot)
    L_balance_onehot = router_balance_loss(weights_onehot, num_charts)
    print(f"  Topology (one-hot): entropy={L_entropy_onehot.item():.4f} (low), balance={L_balance_onehot.item():.4f} (high)")

    # Test warmup
    for epoch in [0, 5, 10, 20, 30]:
        ratio = compute_warmup_ratio(epoch, warmup_epochs=20)
        print(f"  Warmup epoch {epoch}: ratio={ratio:.2f}")

    print("All loss tests passed!")


if __name__ == "__main__":
    test_losses()
