"""
Lindblad-Atlas Projection (LAP) for Dimensionality Reduction

This implements the LAP algorithm using the theoretical machinery from fragile-index.md.
Instead of using smooth transition functions between atlas charts (which cause distortion),
LAP uses Lindbladian Jump Operators to "tunnel" between charts.

Key concepts:
- Macro Register (K): Each discrete code k is the ID of a topological chart
- Structured residual (z_n): The 2D coordinates within that specific chart (used by jumps/holonomy)
- Texture residual (z_tex): Reconstruction-only residual (does not affect routing/jumps)
- Jump Operators (L_{k→k'}): Affine transforms bridging charts
- Holonomy Constraint: Closed loops should return to identity

Reference: fragile-index.md Section 7.7 (Atlas-Based Architecture)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors

from datasets import (
    find_boundary_pairs,
    get_mnist_data,
    get_nightmare_data,
)

# Optional visualization dependencies (not required for training).
try:
    import pandas as pd  # type: ignore
    import holoviews as hv  # type: ignore

    hv.extension("plotly")
    _HAS_HOLOVIEWS = True
except ModuleNotFoundError:
    pd = None
    hv = None
    _HAS_HOLOVIEWS = False


# --- 1. OrthogonalLinear Layer (Section 7.7.2) ---
class OrthogonalLinear(nn.Module):
    """Linear layer with orthogonality regularizer for approximate isometry.

    Constraint: W^T W ≈ I (semi-orthogonality for rectangular W).
    Effect: Better conditioning and approximate distance preservation in the chart.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def orth_defect(self) -> torch.Tensor:
        """Compute ||W^T W - I||²_F to encourage orthogonality."""
        W = self.linear.weight  # [out_features, in_features]
        if W.shape[0] >= W.shape[1]:
            gram = torch.matmul(W.t(), W)
            target = torch.eye(W.shape[1], device=W.device)
        else:
            gram = torch.matmul(W, W.t())
            target = torch.eye(W.shape[0], device=W.device)
        return torch.norm(gram - target) ** 2


# --- 2. JumpCodebook: Affine Transforms Between Charts ---
class JumpCodebook(nn.Module):
    """Stores learned affine transformations between charts.

    L_{k→k'} transforms coordinates from chart k to chart k'.
    Each jump is parameterized as: z' = W_{k→k'} @ z + b_{k→k'}

    The holonomy constraint ensures that composing jumps around a closed
    loop approximately returns to the identity transformation.
    """

    def __init__(self, num_charts: int, latent_dim: int):
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim

        # Initialize transforms as identity matrices
        transforms = torch.eye(latent_dim).unsqueeze(0).unsqueeze(0)
        transforms = transforms.expand(num_charts, num_charts, -1, -1).clone()
        self.transforms = nn.Parameter(transforms)

        # Initialize translations as zeros
        self.translations = nn.Parameter(
            torch.zeros(num_charts, num_charts, latent_dim)
        )

    def apply_jump(self, z: torch.Tensor, k_from: int, k_to: int) -> torch.Tensor:
        """Apply affine transform: z' = W_{k→k'} @ z + b_{k→k'}"""
        W = self.transforms[k_from, k_to]  # [d, d]
        b = self.translations[k_from, k_to]  # [d]
        return z @ W.T + b

    def apply_jump_batched(
        self, z: torch.Tensor, k_from: torch.Tensor, k_to: torch.Tensor
    ) -> torch.Tensor:
        """Apply jumps for a batch with different chart pairs."""
        B = z.shape[0]
        result = torch.zeros_like(z)
        for i in range(B):
            kf, kt = int(k_from[i].item()), int(k_to[i].item())
            result[i] = self.apply_jump(z[i : i + 1], kf, kt).squeeze(0)
        return result

    def compose_path(self, path: list[int]) -> torch.Tensor:
        """Compose transforms along a path [k0, k1, k2, ...]."""
        composed = torch.eye(self.latent_dim, device=self.transforms.device)
        for i in range(len(path) - 1):
            k_from, k_to = path[i], path[i + 1]
            composed = self.transforms[k_from, k_to] @ composed
        return composed

    def holonomy_defect(self, cycle: list[int]) -> torch.Tensor:
        """Compute ||composed_cycle - I||²_F for a closed cycle."""
        composed = self.compose_path(cycle)
        identity = torch.eye(self.latent_dim, device=composed.device)
        return torch.norm(composed - identity) ** 2

    def apply_jump_thermal(
        self,
        z: torch.Tensor,
        k_from: int,
        k_to: int,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Apply jump with thermal noise (Lindbladian dissipator).

        Section 2.11.6: Jumps allow points to "diffuse" between charts,
        preventing the "Glassy Freeze" (Mode T.D) where points get stuck.
        """
        # Deterministic affine transform
        z_jumped = self.apply_jump(z, k_from, k_to)

        # Add thermal noise during training
        if self.training and temperature > 0:
            noise = torch.randn_like(z_jumped) * temperature
            z_jumped = z_jumped + noise

        return z_jumped


# --- 3. LAPEncoder: Atlas-Based Encoder with Gumbel-Softmax ---
class LAPEncoder(nn.Module):
    """Lindblad-Atlas Projection Encoder.

    Architecture:
    - Router: MLP that outputs chart assignment logits
    - Charts: ModuleList of orthogonality-constrained encoders
    - JumpCodebook: Learns inter-chart transitions

    Uses Gumbel-Softmax for differentiable discrete assignment.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_charts: int = 4,
        hidden_dim: int = 128,
        tau_init: float = 1.0,
        tex_dim: int = 8,
        struct_noise_std: float = 0.05,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.tau = tau_init  # Temperature for Gumbel-Softmax
        self.tex_dim = tex_dim
        self.struct_noise_std = struct_noise_std

        # Router: Learns chart assignments
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_charts),
        )

        # Charts: Each is an orthogonality-constrained encoder
        self.charts = nn.ModuleList()
        for _ in range(num_charts):
            expert = nn.Sequential(
                OrthogonalLinear(input_dim, hidden_dim),
                nn.ReLU(),
                OrthogonalLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                OrthogonalLinear(hidden_dim, latent_dim),
            )
            self.charts.append(expert)

        # Jump Codebook: Inter-chart transitions
        self.jump_codebook = JumpCodebook(num_charts, latent_dim)

        # --- Texture channel (reconstruction-only) ---
        # This implements the theory split: z_n (here: z) is structured/nuisance coordinates used
        # by jumps and atlas geometry; z_tex captures reconstruction detail and is NOT used for
        # routing, chart transitions, or holonomy.
        self.tex_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * tex_dim),  # [mu, logvar]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + tex_dim + num_charts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # EMA-smoothed diagonal metric proxy (Section 2.9.1).
        # In the full Fragile Agent, this corresponds to an EMA of the state-space Fisher/Hessian
        # diagonal; in this demo we use a cheap proxy derived from z-variance.
        self.register_buffer('G_ema', torch.ones(latent_dim))
        self.ema_decay = 0.95  # η = 0.05 → slow smoothing

    def update_fisher_ema(self, z: torch.Tensor, router_logits: torch.Tensor, eps: float = 1e-6) -> None:
        """Update the EMA-smoothed diagonal metric proxy.

        We treat the metric estimate as a *dissipative* state variable: it should not track
        minibatch noise instantaneously. This filters stochastic jitter while preserving slow drift.
        """
        with torch.no_grad():
            # Proxy diagonal metric: inverse variance per latent dimension.
            z_var = z.var(dim=0, unbiased=False) + eps
            g_new = 1.0 / z_var

            # Macro "temperature": uncertain routing (high entropy) → more damping downstream.
            probs = F.softmax(router_logits, dim=-1)
            assign_entropy = -(probs * torch.log(probs + eps)).sum(dim=1).mean()
            temperature_factor = 1.0 + 0.5 * assign_entropy
            g_new = g_new * temperature_factor

            eta = 1.0 - float(self.ema_decay)
            self.G_ema.mul_(float(self.ema_decay)).add_(g_new * eta)

    def encode_texture(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode reconstruction-only texture latent z_tex ~ N(mu, diag(exp(logvar)))."""
        stats = self.tex_encoder(x)
        mu, logvar = stats[..., : self.tex_dim], stats[..., self.tex_dim :]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        z_tex = mu + eps * std
        return z_tex, mu, logvar

    def decode(
        self, z: torch.Tensor, router_logits: torch.Tensor, z_tex: torch.Tensor
    ) -> torch.Tensor:
        """Decode x from (z_n, K, z_tex) where K is provided as routing probabilities."""
        probs = F.softmax(router_logits, dim=-1)
        dec_inp = torch.cat([z, z_tex, probs], dim=-1)
        return self.decoder(dec_inp)

    def forward(
        self, x: torch.Tensor, hard: bool = True, return_texture: bool = False
    ):
        """Forward pass through the atlas.

        If return_texture=True, also returns:
          (z_tex, x_recon, tex_mu, tex_logvar)
        """
        # Structural path sees a perturbed view to encourage invariance to "texture-like" jitter.
        # (For images you'd use a proper low/high-frequency split; here we use additive noise.)
        x_struct = x
        if self.training and self.struct_noise_std > 0:
            x_struct = x + torch.randn_like(x) * float(self.struct_noise_std)

        logits = self.router(x_struct)  # [B, num_charts]

        # Gumbel-Softmax for differentiable discrete assignment
        if self.training:
            assignment = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        else:
            assignment = F.one_hot(
                logits.argmax(dim=-1), num_classes=self.num_charts
            ).float()

        # Compute each chart's embedding
        z_per_chart = []
        for i in range(self.num_charts):
            z_i = self.charts[i](x_struct)
            z_per_chart.append(z_i)

        # Stack and blend using assignment weights
        z_stacked = torch.stack(z_per_chart, dim=1)  # [B, K, d]
        z = (assignment.unsqueeze(-1) * z_stacked).sum(dim=1)  # [B, d]

        if not return_texture:
            return z, assignment, z_per_chart, logits

        z_tex, tex_mu, tex_logvar = self.encode_texture(x)
        x_recon = self.decode(z, logits, z_tex)
        return z, assignment, z_per_chart, logits, z_tex, x_recon, tex_mu, tex_logvar

    def compute_orth_loss(self) -> torch.Tensor:
        """Compute total orthogonality defect across all charts."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for chart in self.charts:
            for layer in chart.children():
                if isinstance(layer, OrthogonalLinear):
                    total = total + layer.orth_defect()
        return total

    def get_hard_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard chart assignments (discrete indices)."""
        logits = self.router(x)
        return logits.argmax(dim=-1)


# --- 4. Loss Functions ---


def compute_vicreg_loss(
    z: torch.Tensor, z_prime: torch.Tensor, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg loss components for collapse prevention.

    Includes NaN-safe guards for degenerate cases.
    """
    device = z.device
    B = z.shape[0]

    # Invariance: augmented views should map similarly
    loss_inv = F.mse_loss(z, z_prime)

    # Variance: each dimension should have std >= 1
    # Guard against NaN from very small/zero variance
    var_z = z.var(dim=0)
    var_z = torch.clamp(var_z, min=eps)  # Prevent sqrt(0)
    std_z = torch.sqrt(var_z + eps)
    loss_var = torch.mean(F.relu(1 - std_z))

    # Covariance: off-diagonal should be zero (decorrelation)
    z_centered = z - z.mean(dim=0)
    if B > 1:
        cov = (z_centered.T @ z_centered) / (B - 1)
        d = z.shape[1]
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        loss_cov = off_diag.pow(2).sum() / d
    else:
        loss_cov = torch.tensor(0.0, device=device)

    # NaN guard: if any loss is NaN, return zeros to prevent propagation
    if torch.isnan(loss_inv) or torch.isnan(loss_var) or torch.isnan(loss_cov):
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
        )

    return loss_inv, loss_var, loss_cov


def compute_gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu,diag(exp(logvar))) || N(0,I) ), averaged over batch."""
    kl_per = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).sum(dim=-1)
    return kl_per.mean()


# Global cache for kNN indices (tensor on GPU)
_KNN_CACHE = {}
# Global cache for holonomy cycles (list of cycles per K)
_CYCLE_CACHE = {}


def _should_compute(step: int, every: int) -> bool:
    """Return True if a loss should be computed at this step."""
    return every <= 1 or (step % every) == 0


def compute_chart_centers(
    z_per_chart: list[torch.Tensor],
    assignment: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chart centers and usage weights.

    Returns:
        centers: [K, d]
        usage: [K] (sum of soft assignments per chart)
    """
    num_charts = len(z_per_chart)
    device = z_per_chart[0].device
    latent_dim = z_per_chart[0].shape[1]

    centers = torch.zeros(num_charts, latent_dim, device=device)
    usage = torch.zeros(num_charts, device=device)

    for i in range(num_charts):
        weights = assignment[:, i:i + 1]
        weight_sum = weights.sum() + eps
        centers[i] = (z_per_chart[i] * weights).sum(dim=0) / weight_sum
        usage[i] = weight_sum

    return centers, usage


def compute_jump_consistency_loss(
    z_per_chart: list[torch.Tensor],
    assignment: torch.Tensor,
    X: torch.Tensor,
    jump_codebook: JumpCodebook,
    k_neighbors: int = 10,
    sample_rate: float = 0.1,
    knn_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute jump consistency loss for inter-chart transitions (vectorized, kNN-based).

    Args:
        z_per_chart: List of embeddings from each chart
        assignment: Soft assignment weights [B, K]
        X: Input data [B, D] (used for kNN if not cached)
        jump_codebook: Jump operator codebook
        k_neighbors: Number of neighbors to consider
        sample_rate: Fraction of points to sample for efficiency
        knn_indices: Pre-computed kNN indices [N, k+1] (optional, for minibatch)
    """
    global _KNN_CACHE
    device = X.device
    B = X.shape[0]
    num_charts = len(z_per_chart)
    latent_dim = z_per_chart[0].shape[1]

    hard_assign = assignment.argmax(dim=-1)  # [B]

    # Use provided indices or compute/cache them
    if knn_indices is not None:
        # Minibatch mode: indices provided
        indices_tensor = knn_indices
    else:
        # Full batch mode: cache globally
        cache_key = (B, k_neighbors)
        if cache_key not in _KNN_CACHE:
            X_np = X.detach().cpu().numpy()
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="ball_tree")
            nn.fit(X_np)
            _, indices = nn.kneighbors(X_np)
            _KNN_CACHE[cache_key] = torch.from_numpy(indices).to(device)
        indices_tensor = _KNN_CACHE[cache_key]

    # Sample a subset for efficiency
    n_sample = max(10, int(B * sample_rate))
    sample_idx = torch.randperm(B, device=device)[:n_sample]

    # Stack z_per_chart: [K, B, d]
    z_stacked = torch.stack(z_per_chart, dim=0)

    # Get neighbors for sampled points: [n_sample, k]
    neighbor_indices = indices_tensor[sample_idx, 1:k_neighbors+1]  # exclude self

    # Get chart assignments: [n_sample] and [n_sample, k]
    k_i = hard_assign[sample_idx]  # [n_sample]
    k_j = hard_assign[neighbor_indices.flatten()].view(n_sample, k_neighbors)  # [n_sample, k]

    # Mask: only consider cross-chart pairs
    cross_chart_mask = (k_i.unsqueeze(1) != k_j)  # [n_sample, k]

    if not cross_chart_mask.any():
        return torch.tensor(0.0, device=device)

    # Gather z_i for each sampled point: [n_sample, d]
    # z_i = z_stacked[k_i, sample_idx]
    z_i = z_stacked[k_i, sample_idx, :]  # [n_sample, d]

    # Gather z_j for all neighbors: [n_sample, k, d]
    flat_neighbors = neighbor_indices.flatten()  # [n_sample * k]
    k_j_flat = k_j.flatten()  # [n_sample * k]
    z_j_flat = z_stacked[k_j_flat, flat_neighbors, :]  # [n_sample * k, d]
    z_j = z_j_flat.view(n_sample, k_neighbors, latent_dim)  # [n_sample, k, d]

    # Apply jumps: for each (i, j) pair, compute L_{k_i -> k_j}(z_i)
    # This is the slow part - we need to handle different jump matrices per pair
    # Vectorize by grouping by (k_from, k_to) pairs

    loss = torch.tensor(0.0, device=device)
    count = 0

    for kf in range(num_charts):
        for kt in range(num_charts):
            if kf == kt:
                continue
            # Find pairs where k_i == kf and k_j == kt
            mask_kf = (k_i == kf)  # [n_sample]
            mask_kt = (k_j == kt)  # [n_sample, k]
            pair_mask = mask_kf.unsqueeze(1) & mask_kt & cross_chart_mask  # [n_sample, k]

            if not pair_mask.any():
                continue

            # Get indices where mask is true
            i_indices, j_indices = pair_mask.nonzero(as_tuple=True)
            n_pairs = len(i_indices)

            # Get z_i values for these pairs: [n_pairs, d]
            z_i_batch = z_i[i_indices]

            # Get z_j values for these pairs: [n_pairs, d]
            z_j_batch = z_j[i_indices, j_indices]

            # Apply jump transform (batched): z' = z @ W.T + b
            W = jump_codebook.transforms[kf, kt]  # [d, d]
            b = jump_codebook.translations[kf, kt]  # [d]
            z_i_jumped = z_i_batch @ W.T + b  # [n_pairs, d]

            # Compute MSE loss
            loss = loss + F.mse_loss(z_i_jumped, z_j_batch, reduction='sum')
            count += n_pairs

    if count > 0:
        loss = loss / count
    return loss


def compute_jump_consistency_loss_sampled(
    z_per_chart: list[torch.Tensor],
    assignment: torch.Tensor,
    jump_codebook: JumpCodebook,
    num_pairs: int = 512,
) -> torch.Tensor:
    """Fast proxy for jump consistency using random cross-chart pairs."""
    device = assignment.device
    B = assignment.shape[0]

    if B < 2:
        return torch.tensor(0.0, device=device)

    hard_assign = assignment.argmax(dim=-1)  # [B]
    z_stacked = torch.stack(z_per_chart, dim=0)  # [K, B, d]

    # Oversample to increase chance of cross-chart pairs, then slice.
    oversample = max(2, 2 * num_pairs)
    i_idx = torch.randint(0, B, (oversample,), device=device)
    j_idx = torch.randint(0, B, (oversample,), device=device)
    mask = hard_assign[i_idx] != hard_assign[j_idx]

    i_idx = i_idx[mask][:num_pairs]
    j_idx = j_idx[mask][:num_pairs]

    if i_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    kf = hard_assign[i_idx]
    kt = hard_assign[j_idx]

    z_i = z_stacked[kf, i_idx, :]  # [P, d]
    z_j = z_stacked[kt, j_idx, :]  # [P, d]

    W = jump_codebook.transforms[kf, kt]  # [P, d, d]
    b = jump_codebook.translations[kf, kt]  # [P, d]
    z_i_jumped = torch.bmm(z_i.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b

    return F.mse_loss(z_i_jumped, z_j)


def compute_jump_consistency_loss_proto(
    centers: torch.Tensor,
    usage: torch.Tensor,
    jump_codebook: JumpCodebook,
    max_pairs: int = 0,
) -> torch.Tensor:
    """Fast proxy for jump consistency using chart prototypes."""
    device = centers.device
    num_charts = centers.shape[0]

    if num_charts < 2:
        return torch.tensor(0.0, device=device)

    if max_pairs <= 0:
        kf = torch.arange(num_charts, device=device).repeat_interleave(num_charts)
        kt = torch.arange(num_charts, device=device).repeat(num_charts)
        mask = kf != kt
        kf = kf[mask]
        kt = kt[mask]
    else:
        probs = usage / (usage.sum() + 1e-8)
        kf = torch.multinomial(probs, max_pairs, replacement=True)
        kt = torch.multinomial(probs, max_pairs, replacement=True)
        mask = kf != kt
        kf = kf[mask]
        kt = kt[mask]

    if kf.numel() == 0:
        return torch.tensor(0.0, device=device)

    z_from = centers[kf]
    z_to = centers[kt]
    W = jump_codebook.transforms[kf, kt]  # [P, d, d]
    b = jump_codebook.translations[kf, kt]  # [P, d]
    z_jumped = torch.bmm(z_from.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b

    return F.mse_loss(z_jumped, z_to)


def compute_holonomy_loss(
    jump_codebook: JumpCodebook, num_charts: int, n_cycles: int | None = None
) -> torch.Tensor:
    """Compute holonomy constraint: closed loops should be identity.

    Instead of computing all O(K³) 3-cycles, we sample n_cycles random cycles.
    For K=4: 24 cycles → sample 6; for K=10: 720 cycles → sample 30.

    If n_cycles is None, scales as max(6, 3*K) to maintain coverage.
    """
    # Scale n_cycles with K if not provided
    if n_cycles is None:
        n_cycles = max(6, num_charts * 3)  # K=10 → 30 cycles
    device = jump_codebook.transforms.device

    # Build all possible 3-cycles once per K
    if num_charts in _CYCLE_CACHE:
        all_cycles = _CYCLE_CACHE[num_charts]
    else:
        all_cycles = []
        for k0 in range(num_charts):
            for k1 in range(num_charts):
                if k1 == k0:
                    continue
                for k2 in range(num_charts):
                    if k2 == k0 or k2 == k1:
                        continue
                    all_cycles.append([k0, k1, k2, k0])
        _CYCLE_CACHE[num_charts] = all_cycles

    if not all_cycles:
        return torch.tensor(0.0, device=device)

    # Sample n_cycles random cycles (or all if fewer available)
    n_sample = min(n_cycles, len(all_cycles))
    sampled_cycles = random.sample(all_cycles, n_sample)

    loss = torch.tensor(0.0, device=device)
    for cycle in sampled_cycles:
        loss = loss + jump_codebook.holonomy_defect(cycle)

    return loss / n_sample


def compute_topology_loss(
    router_logits: torch.Tensor, num_charts: int, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Topology loss for atlas structure.

    Uses the *soft* routing distribution q(K|x) (from logits) to measure:
    - per-sample routing entropy (router uncertainty),
    - mean chart usage (collapse / dead charts / excessive imbalance).
    - diversity: H(K) entropy of mean usage (encourage using all charts)

    Returns:
        loss_entropy: Mean per-sample entropy (penalize uncertainty)
        loss_balance: L2 distance to uniform (legacy)
        loss_diversity: -H(K) + log(K) so that uniform = 0, collapsed = log(K)
        is_collapsed: True if max chart usage > 0.8 (emergency flag)
    """
    probs = F.softmax(router_logits, dim=-1)

    # Per-sample entropy (router confidence)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    loss_entropy = entropy.mean()

    # Mean chart usage across batch
    mean_usage = probs.mean(dim=0)

    # Legacy balance loss (L2 to uniform) - scales poorly with K
    target_usage = torch.ones(num_charts, device=router_logits.device) / num_charts
    loss_balance = torch.norm(mean_usage - target_usage) ** 2

    # NEW: Diversity loss = -H(K) + log(K)
    # H(K) = entropy of mean_usage; max when uniform, min=0 when collapsed
    # We want to maximize H(K), so loss = log(K) - H(K)
    # This ranges from 0 (uniform) to log(K) (collapsed)
    H_K = -torch.sum(mean_usage * torch.log(mean_usage + eps))
    log_K = float(np.log(num_charts))
    loss_diversity = log_K - H_K  # 0 when uniform, positive when collapsed

    # Collapse detection: if any chart has > 50% of points, flag emergency (stricter threshold)
    is_collapsed = (mean_usage.max().item() > 0.5)

    return loss_entropy, loss_balance, loss_diversity, is_collapsed


def compute_separation_loss(
    z_per_chart: list[torch.Tensor],
    assignment: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Force chart 'islands' to stay apart unless connected by Jump.

    Implements Topological Surgery: prevents Ontological Mixing (Mode S.D).
    From fragile-index.md: Different datasets should NOT share the same chart.
    The chart ID (K) should be a sufficient statistic for data identity.

    Args:
        z_per_chart: List of latent embeddings per chart [K x (B, d)]
        assignment: Soft assignment probabilities [B, K]
        margin: Minimum distance between chart centers (default 2.0)

    Returns:
        Hinge loss penalizing chart centers closer than margin.
    """
    num_charts = len(z_per_chart)
    device = z_per_chart[0].device

    # Compute weighted center for each chart
    centers, _ = compute_chart_centers(z_per_chart, assignment)

    # Hinge loss: force centers at least 'margin' apart
    loss_sep = torch.tensor(0.0, device=device)
    n_pairs = 0
    for i in range(num_charts):
        for j in range(i + 1, num_charts):
            dist = torch.norm(centers[i] - centers[j])
            loss_sep = loss_sep + torch.relu(margin - dist)
            n_pairs += 1

    return loss_sep / max(n_pairs, 1)


def compute_orbit_invariance_loss(
    model: LAPEncoder,
    X: torch.Tensor,
    n_transforms: int = 4,
) -> torch.Tensor:
    """Gauge coherence: chart assignment invariant to SO(3) rotations.

    L_orbit = E_g[KL(q(K|x) || q(K|g·x))]

    From fragile-index.md Section 3.3: Operationalizes the quotient intent
    "K approximates x/G_spatial" and prevents symmetry-blind representations.

    Note: Only applies to 3D input data. Returns 0 for high-dimensional data.
    """
    device = X.device

    # SO(3) orbit invariance only applies to 3D input
    if X.shape[1] != 3 or n_transforms <= 0:
        return torch.tensor(0.0, device=device)

    # Get original assignment probabilities
    original_logits = model.router(X)
    original_probs = F.softmax(original_logits, dim=-1)

    loss = torch.tensor(0.0, device=device)

    for _ in range(n_transforms):
        # Sample random SO(3) rotation via Rodrigues formula
        axis = torch.randn(3, device=device)
        axis = axis / (axis.norm() + 1e-8)
        angle = torch.rand(1, device=device) * 2 * np.pi

        # Skew-symmetric matrix for axis
        K_mat = torch.tensor([
            [0, -axis[2].item(), axis[1].item()],
            [axis[2].item(), 0, -axis[0].item()],
            [-axis[1].item(), axis[0].item(), 0]
        ], device=device)

        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = (torch.eye(3, device=device)
             + torch.sin(angle) * K_mat
             + (1 - torch.cos(angle)) * (K_mat @ K_mat))

        # Apply rotation to input
        X_rotated = X @ R.T

        # Get rotated assignment
        rotated_logits = model.router(X_rotated)

        # KL divergence: KL(original || rotated)
        kl = F.kl_div(
            F.log_softmax(rotated_logits, dim=-1),
            original_probs,
            reduction='batchmean'
        )
        loss = loss + kl

    return loss / n_transforms


def compute_macro_micro_disentangle_loss(
    z: torch.Tensor,
    assignment_probs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gauge coherence: macro–nuisance independence (in this demo).

    L_{K⊥n} = ||Cov(q(K|x), z_n)||²_F

    From fragile-index.md Section 3.3: Chart ID (macro) shouldn't predict
    position within chart (structured residual). Texture is handled separately
    via the reconstruction-only channel and is not used for routing/jumps.
    """
    device = z.device
    B = z.shape[0]

    # Guard: need at least 2 samples for covariance
    if B < 2:
        return torch.tensor(0.0, device=device)

    # Center both representations (with numerical stability)
    z_centered = z - z.mean(dim=0, keepdim=True)
    assign_centered = assignment_probs - assignment_probs.mean(dim=0, keepdim=True)

    # Clamp to prevent extreme values
    z_centered = torch.clamp(z_centered, -100, 100)
    assign_centered = torch.clamp(assign_centered, -1, 1)

    # Cross-covariance matrix [num_charts, latent_dim]
    cross_cov = (assign_centered.T @ z_centered) / max(B - 1, 1)

    # Frobenius norm squared with stability
    result = torch.sum(cross_cov ** 2)

    # NaN guard
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return result


def compute_window_loss(
    router_logits: torch.Tensor,
    num_charts: int,
    eps_ground: float = 0.1,
    eps_mix: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Information-Stability Window (Theorem 15.1.3).

    Implements the operational window as a computable penalty on the routing distribution:
    - Grounding: I(X;K) must not collapse.
    - Optional anti-dispersion: H(K) should not saturate log|K| (set eps_mix > 0 to enable).

    All quantities are measured in nats (natural log).
    """
    probs = F.softmax(router_logits, dim=-1)  # q(K|x)
    mean_usage = probs.mean(dim=0)            # q(K)

    H_K = -torch.sum(mean_usage * torch.log(mean_usage + eps))
    H_K_given_X = -torch.sum(probs * torch.log(probs + eps), dim=1).mean()
    I_XK = H_K - H_K_given_X

    logK = float(np.log(num_charts))
    loss_ground = torch.relu(eps_ground - I_XK).pow(2)
    loss_mix = torch.relu(H_K - (logK - eps_mix)).pow(2)

    window_loss = loss_ground + loss_mix
    metrics = {
        'H_K': H_K.item(),
        'H_K_given_X': H_K_given_X.item(),
        'I_XK': I_XK.item(),
    }
    return window_loss, metrics


def compute_closure_loss(
    model: LAPEncoder,
    X: torch.Tensor,
    delta_scale: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Causal Enclosure: K(x') predictable from K(x) for small δ.

    Section 2.8: The Macro symbol K must be a sufficient statistic for
    its own future. Small input perturbations → smooth assignment changes.
    """
    device = X.device

    # Perturb X slightly (simulating a step in time/space)
    delta = torch.randn_like(X) * delta_scale
    X_plus = X + delta

    # Get current and perturbed Macro states
    logits = model.router(X)
    logits_plus = model.router(X_plus)

    # Clamp logits to prevent overflow in softmax
    logits = torch.clamp(logits, -50, 50)
    logits_plus = torch.clamp(logits_plus, -50, 50)

    # For small delta, assignments should be similar. Use Jensen-Shannon divergence
    # (more numerically stable than symmetric KL).
    probs = F.softmax(logits, dim=-1)
    probs_plus = F.softmax(logits_plus, dim=-1)

    # Clamp probabilities to avoid log(0)
    probs = torch.clamp(probs, eps, 1 - eps)
    probs_plus = torch.clamp(probs_plus, eps, 1 - eps)

    # Jensen-Shannon divergence: JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    # where m = 0.5 * (p + q)
    m = 0.5 * (probs + probs_plus)
    m = torch.clamp(m, eps, 1 - eps)

    kl_pm = (probs * (torch.log(probs) - torch.log(m))).sum(dim=-1)
    kl_qm = (probs_plus * (torch.log(probs_plus) - torch.log(m))).sum(dim=-1)
    js_div = 0.5 * (kl_pm + kl_qm)

    # Average over batch
    result = js_div.mean()

    # NaN guard
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Normalize by perturbation size (Lipschitz-like audit: bounded KL per step).
    return result / (delta_scale + eps)


class SieveMonitor:
    """21-Node Sieve for auditing LAP representation quality.

    Section 3: Gate Nodes that audit the representation and warn
    if critical invariants are violated.
    """

    def __init__(self, num_charts: int, latent_dim: int):
        self.num_charts = num_charts
        self.latent_dim = latent_dim

    def run_checks(
        self,
        z: torch.Tensor,
        router_logits: torch.Tensor,
        losses: dict,
    ) -> dict[str, tuple[bool, float]]:
        """Run sieve checks and return (passed, value) for each."""
        checks = {}

        with torch.no_grad():
            probs = F.softmax(router_logits, dim=-1)
            logK = float(np.log(self.num_charts))

            # Node 3: CompactCheck - Are chart boundaries sharp?
            H_K_given_X = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean().item()
            compactness = 1.0 - (H_K_given_X / (logK + 1e-6))
            checks['compact'] = (compactness > 0.5, compactness)

            # Node 7: StiffnessCheck - Is gradient strong enough?
            z_var = z.var(dim=0).mean().item()
            stiffness = 1.0 / (z_var + 1e-6)
            checks['stiffness'] = (0.1 < stiffness < 100, stiffness)

            # Node 11: ComplexCheck - Using all charts?
            mean_usage = probs.mean(dim=0)
            H_K = -(mean_usage * torch.log(mean_usage + 1e-6)).sum().item()
            complexity = H_K / (logK + 1e-6)
            checks['complexity'] = (complexity > 0.3, complexity)

            # Node 13: BoundaryCheck - Grounded in input?
            # I(X;K) = H(K) - H(K|X) for discrete K.
            I_XK = H_K - H_K_given_X
            checks['grounding'] = (I_XK > 0.1, I_XK)

            # Node 15: HolonomyCheck - Cycles close?
            holo_loss = losses.get('holonomy', 0.0)
            checks['holonomy'] = (holo_loss < 0.1, holo_loss)

        return checks

    def format_report(self, checks: dict) -> str:
        """Format checks as a string report."""
        lines = []
        for name, (passed, value) in checks.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {name}: {value:.4f}")
        return "\n".join(lines)


class AdaptiveLambdas:
    """Self-tuning loss weights based on running statistics.

    Section 3.5: Auto-adjust λ weights so all losses contribute equally.
    """

    def __init__(self, loss_names: list[str], ema_decay: float = 0.99):
        self.loss_names = loss_names
        self.ema_losses = {name: 1.0 for name in loss_names}
        self.lambdas = {name: 1.0 for name in loss_names}
        self.ema_decay = ema_decay

    def update(self, losses: dict[str, float]) -> dict[str, float]:
        """Update EMAs and compute adaptive lambdas."""
        for name in self.loss_names:
            active_key = f"_active_{name}"
            if active_key in losses and not losses[active_key]:
                continue
            if name in losses:
                self.ema_losses[name] = (
                    self.ema_decay * self.ema_losses[name] +
                    (1 - self.ema_decay) * abs(losses[name])
                )

        # Normalize lambdas so all losses contribute equally
        ema_values = [v for v in self.ema_losses.values() if v > 1e-8]
        if ema_values:
            mean_loss = float(np.mean(ema_values))
            for name in self.loss_names:
                if self.ema_losses[name] > 1e-8:
                    self.lambdas[name] = mean_loss / self.ema_losses[name]
                else:
                    self.lambdas[name] = 1.0

        return self.lambdas


def lindblad_atlas_loss(
    z: torch.Tensor,
    assignment: torch.Tensor,
    router_logits: torch.Tensor,
    z_per_chart: list[torch.Tensor],
    X: torch.Tensor,
    model: LAPEncoder,
    x_recon: torch.Tensor | None = None,
    tex_mu: torch.Tensor | None = None,
    tex_logvar: torch.Tensor | None = None,
    step: int = 0,
    jump_mode: str = "full",
    jump_every: int = 1,
    jump_sample_rate: float = 0.1,
    jump_num_pairs: int = 512,
    jump_proto_pairs: int = 0,
    holo_every: int = 1,
    orbit_every: int = 1,
    orbit_transforms: int = 4,
    closure_every: int = 1,
    closure_delta: float = 0.05,
    lambda_inv: float = 10.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 10.0,
    lambda_entropy: float = 2.0,
    lambda_balance: float = 10.0,
    lambda_diversity: float = 50.0,  # Strong diversity loss to prevent collapse
    lambda_jump: float = 5.0,
    lambda_holo: float = 10.0,
    lambda_orth: float = 0.01,
    lambda_orbit: float = 1.0,
    lambda_disentangle: float = 0.5,
    lambda_window: float = 1.0,
    lambda_closure: float = 0.5,
    lambda_recon: float = 1.0,
    lambda_tex_kl: float = 0.1,
    lambda_sep: float = 1.0,
    sep_margin: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Unified LAP loss combining all components.

    Level 5 Fragile Agent loss includes:
    - VICReg: collapse prevention
    - Topology: chart entropy and balance
    - Jump consistency: inter-chart coherence
    - Holonomy: closed loops → identity
    - Orthogonality: approximate isometry
    - Orbit invariance: SO(3) gauge coherence
    - Macro–nuisance disentanglement: discourage trivial coupling between chart ID and chart coords
    - Information Window: H(K) in Goldilocks zone (Theorem 15.1.3)
    - Causal Enclosure: predictive closure for Macro state
    - Separation: force chart islands apart (prevent Ontological Mixing)
    - Texture channel (reconstruction-only): z_tex affects reconstruction but not routing/jumps

    Fast alternatives:
    - jump_mode: "full" (kNN), "sampled" (random pairs), "proto" (chart centers)
    - *_every: compute heavy losses every N steps (else zeroed)
    """
    losses = {}

    routing_probs = F.softmax(router_logits, dim=-1)

    # 1. VICReg
    X_aug = X + torch.randn_like(X) * 0.05
    z_prime, _, _, _ = model(X_aug)
    loss_inv, loss_var, loss_cov = compute_vicreg_loss(z, z_prime)
    losses["invariance"] = loss_inv.item()
    losses["variance"] = loss_var.item()
    losses["covariance"] = loss_cov.item()

    # 2. Topology (with diversity and collapse detection)
    loss_entropy, loss_balance, loss_diversity, is_collapsed = compute_topology_loss(
        router_logits, model.num_charts
    )
    losses["entropy"] = loss_entropy.item()
    losses["balance"] = loss_balance.item()
    losses["diversity"] = loss_diversity.item()
    losses["_is_collapsed"] = is_collapsed  # Internal flag for training loop

    # 3. Jump Consistency
    jump_active = _should_compute(step, jump_every)
    losses["_active_jump"] = jump_active
    if jump_active:
        if jump_mode == "full":
            loss_jump = compute_jump_consistency_loss(
                z_per_chart,
                assignment,
                X,
                model.jump_codebook,
                k_neighbors=5,
                sample_rate=jump_sample_rate,
            )
        elif jump_mode == "sampled":
            loss_jump = compute_jump_consistency_loss_sampled(
                z_per_chart,
                assignment,
                model.jump_codebook,
                num_pairs=jump_num_pairs,
            )
        elif jump_mode == "proto":
            centers, usage = compute_chart_centers(z_per_chart, assignment)
            loss_jump = compute_jump_consistency_loss_proto(
                centers,
                usage,
                model.jump_codebook,
                max_pairs=jump_proto_pairs,
            )
        else:
            raise ValueError(f"Unknown jump_mode: {jump_mode}")
    else:
        loss_jump = torch.tensor(0.0, device=X.device)
    losses["jump"] = loss_jump.item()

    # 4. Holonomy
    holo_active = _should_compute(step, holo_every)
    losses["_active_holonomy"] = holo_active
    if holo_active:
        loss_holo = compute_holonomy_loss(model.jump_codebook, model.num_charts)
    else:
        loss_holo = torch.tensor(0.0, device=X.device)
    losses["holonomy"] = loss_holo.item()

    # 5. Orthogonality
    loss_orth = model.compute_orth_loss()
    losses["orthogonality"] = loss_orth.item()

    # 6. Orbit Invariance (Gauge Coherence)
    orbit_active = _should_compute(step, orbit_every)
    losses["_active_orbit"] = orbit_active
    if orbit_active:
        loss_orbit = compute_orbit_invariance_loss(
            model, X, n_transforms=orbit_transforms
        )
    else:
        loss_orbit = torch.tensor(0.0, device=X.device)
    losses["orbit"] = loss_orbit.item()

    # 7. Macro-Micro Disentanglement (Gauge Coherence)
    loss_disentangle = compute_macro_micro_disentangle_loss(z, routing_probs)
    losses["disentangle"] = loss_disentangle.item()

    # 8. Information-Stability Window (Theorem 15.1.3)
    loss_window, window_metrics = compute_window_loss(router_logits, model.num_charts)
    losses["window"] = loss_window.item()
    losses["H_K"] = window_metrics["H_K"]
    losses["H_K_given_X"] = window_metrics["H_K_given_X"]
    losses["I_XK"] = window_metrics["I_XK"]

    # 9. Causal Enclosure (Section 2.8)
    closure_active = _should_compute(step, closure_every)
    losses["_active_closure"] = closure_active
    if closure_active:
        loss_closure = compute_closure_loss(model, X, delta_scale=closure_delta)
    else:
        loss_closure = torch.tensor(0.0, device=X.device)
    losses["closure"] = loss_closure.item()

    # 10. Chart Separation (Topological Surgery)
    loss_sep = compute_separation_loss(z_per_chart, assignment, margin=sep_margin)
    losses["separation"] = loss_sep.item()

    # 11. Texture channel (reconstruction-only)
    loss_recon = torch.tensor(0.0, device=X.device)
    loss_tex_kl = torch.tensor(0.0, device=X.device)
    if x_recon is not None:
        loss_recon = F.mse_loss(x_recon, X)
    if tex_mu is not None and tex_logvar is not None:
        loss_tex_kl = compute_gaussian_kl(tex_mu, tex_logvar)

    losses["recon"] = float(loss_recon.item())
    losses["tex_kl"] = float(loss_tex_kl.item())

    total = (
        lambda_inv * loss_inv
        + lambda_var * loss_var
        + lambda_cov * loss_cov
        + lambda_entropy * loss_entropy
        + lambda_balance * loss_balance
        + lambda_diversity * loss_diversity  # Strong anti-collapse
        + lambda_jump * loss_jump
        + lambda_holo * loss_holo
        + lambda_orth * loss_orth
        + lambda_orbit * loss_orbit
        + lambda_disentangle * loss_disentangle
        + lambda_window * loss_window
        + lambda_closure * loss_closure
        + lambda_sep * loss_sep
        + lambda_recon * loss_recon
        + lambda_tex_kl * loss_tex_kl
    )

    return total, losses


# --- 5. Visualization ---
# Dataset functions (get_nightmare_data, get_mnist_data, find_boundary_pairs)
# are now imported from datasets.py


def visualize_lap(
    model: LAPEncoder,
    X: torch.Tensor,
    colors: np.ndarray,
    save_path: str = "lap_projection.html",
    is_high_dim: bool = False,
):
    """Visualize LAP results.

    - .html: Interactive HoloViews/Plotly figure
    - .png: Static matplotlib figure

    For 3D input (nightmare): 4-panel layout with 3D input scatter
    For high-dim input (MNIST): 3-panel layout without 3D scatter

    Args:
        model: Trained LAPEncoder model
        X: Input data tensor
        colors: Continuous color values [0,1] for rainbow colormap
        save_path: Path to save (.html for interactive, .png for static)
        is_high_dim: If True, skip 3D input panel (for MNIST/high-dim data)
    """
    model.eval()
    with torch.no_grad():
        z, assignment, _, _ = model(X, hard=False)
        z = z.cpu().numpy()
        X_np = X.cpu().numpy()
        assignment_np = assignment.cpu().numpy()
        hard_assign = assignment_np.argmax(axis=1)

    # Static PNG: use matplotlib
    if save_path.endswith(".png"):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
        except ImportError:
            print("Visualization skipped: matplotlib not installed.")
            return

        if is_high_dim:
            # 3-panel layout for high-dimensional data (no 3D input scatter)
            fig = plt.figure(figsize=(15, 5))

            # Panel 1: Latent by structure (rainbow)
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.scatter(z[:, 0], z[:, 1], c=colors, cmap="rainbow", s=2, alpha=0.7)
            ax1.set_title("Latent Space\n(Colored by Class)")
            ax1.set_xlabel("z₁")
            ax1.set_ylabel("z₂")

            # Panel 2: Latent by chart
            ax2 = fig.add_subplot(1, 3, 2)
            scatter2 = ax2.scatter(z[:, 0], z[:, 1], c=hard_assign, cmap="tab10", s=2, alpha=0.7)
            ax2.set_title("Chart Assignment\n(Topological Surgery)")
            ax2.set_xlabel("z₁")
            ax2.set_ylabel("z₂")
            plt.colorbar(scatter2, ax=ax2, ticks=range(model.num_charts), label="Chart")

            # Panel 3: Portal view with soft blending + boundary lines
            ax3 = fig.add_subplot(1, 3, 3)
            cmap_tab10 = plt.get_cmap("tab10")
            chart_colors = cmap_tab10(np.linspace(0, 1, model.num_charts))[:, :3]
            blended = np.zeros((len(z), 3))
            for i in range(model.num_charts):
                blended += assignment_np[:, i:i+1] * chart_colors[i:i+1]
            blended = np.clip(blended, 0, 1)
            ax3.scatter(z[:, 0], z[:, 1], c=blended, s=2, alpha=0.7)

            boundary_pairs = find_boundary_pairs(z, hard_assign, X_np, k=3)
            if len(boundary_pairs) > 500:
                indices = np.random.choice(len(boundary_pairs), 500, replace=False)
                boundary_pairs = [boundary_pairs[i] for i in indices]
            if boundary_pairs:
                segments = [(z[i], z[j]) for i, j in boundary_pairs]
                lc = LineCollection(segments, colors="gray", alpha=0.2, linewidths=0.5)
                ax3.add_collection(lc)

            ax3.set_title("Portal View\n(Jump Operators)")
            ax3.set_xlabel("z₁")
            ax3.set_ylabel("z₂")
        else:
            # 4-panel layout for 3D input data
            fig = plt.figure(figsize=(20, 5))

            # Panel 1: 3D Input with rainbow coloring
            ax1 = fig.add_subplot(1, 4, 1, projection="3d")
            ax1.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=colors, cmap="rainbow", s=2, alpha=0.7)
            ax1.set_title("Input: The Nightmare\n(Roll, Sphere, Moons)")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # Panel 2: Latent by structure (rainbow)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.scatter(z[:, 0], z[:, 1], c=colors, cmap="rainbow", s=2, alpha=0.7)
            ax2.set_title("Latent Space\n(Colored by Structure)")
            ax2.set_xlabel("z₁")
            ax2.set_ylabel("z₂")

            # Panel 3: Latent by chart
            ax3 = fig.add_subplot(1, 4, 3)
            scatter3 = ax3.scatter(z[:, 0], z[:, 1], c=hard_assign, cmap="tab10", s=2, alpha=0.7)
            ax3.set_title("Chart Assignment\n(Topological Surgery)")
            ax3.set_xlabel("z₁")
            ax3.set_ylabel("z₂")
            plt.colorbar(scatter3, ax=ax3, ticks=range(model.num_charts), label="Chart")

            # Panel 4: Portal view with soft blending + boundary lines
            ax4 = fig.add_subplot(1, 4, 4)
            cmap_tab10 = plt.get_cmap("tab10")
            chart_colors = cmap_tab10(np.linspace(0, 1, model.num_charts))[:, :3]
            blended = np.zeros((len(z), 3))
            for i in range(model.num_charts):
                blended += assignment_np[:, i:i+1] * chart_colors[i:i+1]
            blended = np.clip(blended, 0, 1)
            ax4.scatter(z[:, 0], z[:, 1], c=blended, s=2, alpha=0.7)

            boundary_pairs = find_boundary_pairs(z, hard_assign, X_np, k=3)
            if len(boundary_pairs) > 500:
                indices = np.random.choice(len(boundary_pairs), 500, replace=False)
                boundary_pairs = [boundary_pairs[i] for i in indices]
            if boundary_pairs:
                segments = [(z[i], z[j]) for i, j in boundary_pairs]
                lc = LineCollection(segments, colors="gray", alpha=0.2, linewidths=0.5)
                ax4.add_collection(lc)

            ax4.set_title("Portal View\n(Jump Operators)")
            ax4.set_xlabel("z₁")
            ax4.set_ylabel("z₂")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    # Interactive HTML: use HoloViews/Plotly
    if not _HAS_HOLOVIEWS:
        # Fallback to matplotlib PNG if holoviews not available
        png_path = save_path.replace(".html", ".png")
        print(f"holoviews not installed. Falling back to matplotlib: {png_path}")
        visualize_lap(model, X, colors, save_path=png_path, is_high_dim=is_high_dim)
        return

    # Panel 1: Latent by structure (rainbow)
    df_latent = pd.DataFrame({"z1": z[:, 0], "z2": z[:, 1], "structure": colors})
    panel_structure = hv.Points(df_latent, kdims=["z1", "z2"], vdims=["structure"]).opts(
        color="structure", cmap="rainbow", size=3, alpha=0.7,
        title="Latent Space (by Class)" if is_high_dim else "Latent Space (by Structure)",
        width=400, height=400,
    )

    # Panel 2: Latent by chart (discrete)
    df_charts = pd.DataFrame({"z1": z[:, 0], "z2": z[:, 1], "chart": hard_assign})
    panel_charts = hv.Points(df_charts, kdims=["z1", "z2"], vdims=["chart"]).opts(
        color="chart", cmap="Category10", size=3, alpha=0.7,
        colorbar=True, title="Chart Assignment", width=400, height=400,
    )

    # Panel 3: Portal view with boundary lines
    uncertainty = 1.0 - assignment_np.max(axis=1)
    marker_size = 3 + uncertainty * 8
    df_portal = pd.DataFrame({
        "z1": z[:, 0], "z2": z[:, 1], "chart": hard_assign,
        "uncertainty": uncertainty, "size": marker_size,
    })
    scatter_portal = hv.Points(
        df_portal, kdims=["z1", "z2"], vdims=["chart", "uncertainty", "size"]
    ).opts(
        color="chart", cmap="Category10", size="size", alpha=0.7,
        title="Portal View (size = uncertainty)", width=400, height=400,
    )

    # Boundary segments
    boundary_pairs = find_boundary_pairs(z, hard_assign, X_np, k=3)
    if len(boundary_pairs) > 500:
        indices = np.random.choice(len(boundary_pairs), 500, replace=False)
        boundary_pairs = [boundary_pairs[i] for i in indices]

    if boundary_pairs:
        seg_data = [(z[i, 0], z[i, 1], z[j, 0], z[j, 1]) for i, j in boundary_pairs]
        df_segs = pd.DataFrame(seg_data, columns=["x0", "y0", "x1", "y1"])
        segments = hv.Segments(df_segs).opts(line_color="lightgray", line_width=0.5)
        panel_portal = scatter_portal * segments
    else:
        panel_portal = scatter_portal

    if is_high_dim:
        # 3-panel layout for high-dimensional data
        layout = (panel_structure + panel_charts + panel_portal).cols(3)
    else:
        # 4-panel layout with 3D input for low-dimensional data
        df_input = pd.DataFrame({
            "X": X_np[:, 0], "Y": X_np[:, 1], "Z": X_np[:, 2], "structure": colors,
        })
        panel_input = hv.Scatter3D(df_input, kdims=["X", "Y", "Z"], vdims=["structure"]).opts(
            color="structure", cmap="rainbow", size=3, alpha=0.7,
            title="Input: The Nightmare", width=400, height=400,
        )
        layout = (panel_input + panel_structure + panel_charts + panel_portal).cols(2)

    hv.save(layout, save_path)


# --- 7. Training ---


def train_lap(
    epochs: int = 4000,
    lr: float = 2e-3,
    batch_size: int = 256,
    tau_start: float = 1.0,
    tau_end: float = 0.5,
    save_every: int = 100,
    output_dir: str = "lap_training",
    grad_clip: float = 1.0,
    use_scheduler: bool = True,
    use_riemannian: bool = True,
    # Core loss weights (VICReg + Topology)
    lambda_inv: float = 10.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 10.0,
    lambda_balance: float = 10.0,
    lambda_jump: float = 5.0,
    lambda_holo: float = 10.0,
    lambda_diversity: float = 50.0,  # Anti-collapse diversity loss
    # Level 5 loss weights
    lambda_orbit: float = 1.0,
    lambda_disentangle: float = 0.5,
    lambda_window: float = 1.0,
    lambda_closure: float = 0.5,
    run_sieve: bool = True,
    use_adaptive_lambdas: bool = False,
    ema_decay: float = 0.95,
    metric_eps_min: float = 1e-6,
    metric_eps_scale: float = 1e-3,
    static_plots: bool = False,
    # Loss scheduling / proxies
    jump_mode: str = "full",
    jump_every: int = 1,
    jump_sample_rate: float = 0.1,
    jump_num_pairs: int = 512,
    jump_proto_pairs: int = 0,
    holo_every: int = 1,
    orbit_every: int = 1,
    orbit_transforms: int = 4,
    closure_every: int = 1,
    closure_delta: float = 0.05,
    # Dataset selection
    use_mnist: bool = False,
    n_samples: int = 3000,
    # Model architecture
    num_charts: int = 4,
    hidden_dim: int = 128,
    # Chart separation
    lambda_sep: float = 1.0,
    sep_margin: float = 2.0,
) -> tuple[LAPEncoder, torch.Tensor, np.ndarray]:
    """Train the LAP model - Level 5 Fragile Agent with Cybernetic Damping.

    Features:
    - Minibatching for O(B²) instead of O(N²) VICReg covariance
    - OneCycleLR scheduler with warmup for faster convergence
    - EMA-smoothed Fisher metric (Lindblad Diagonal) for gradient scaling
    - Gauge coherence: orbit invariance + macro-micro disentanglement
    - Information-Stability Window (Theorem 15.1.3)
    - Causal Enclosure (Section 2.8)
    - 21-Node Sieve Monitor for self-auditing

    Args:
        epochs: Number of training epochs
        lr: Max learning rate for OneCycleLR
        batch_size: Minibatch size (256 optimal for N=3000)
        tau_start: Initial Gumbel-Softmax temperature
        tau_end: Final Gumbel-Softmax temperature
        save_every: Save visualization every N epochs (0 to disable)
        output_dir: Directory to save training visualizations
        grad_clip: Gradient norm clipping threshold
        use_scheduler: Whether to use OneCycleLR
        use_riemannian: Whether to use EMA Fisher metric for gradient scaling
        lambda_orbit: Weight for orbit invariance loss
        lambda_disentangle: Weight for macro-micro disentanglement loss
        lambda_window: Weight for information window loss
        lambda_closure: Weight for causal enclosure loss
        run_sieve: Whether to run sieve checks (warn only, no halt)
        use_adaptive_lambdas: Whether to auto-tune loss weights
        ema_decay: EMA decay for Fisher metric smoothing
        metric_eps_min: Base stabilizer ε for metric inversion
        metric_eps_scale: Extra ε added when routing is weakly grounded
        use_mnist: If True, use MNIST dataset instead of nightmare
        n_samples: Number of samples to use from the dataset
        jump_mode: Jump loss mode: full, sampled, or proto
        jump_every: Compute jump loss every N steps
        jump_sample_rate: Sample rate for kNN jump loss
        jump_num_pairs: Number of random pairs for sampled jump loss
        jump_proto_pairs: Number of chart pairs for proto jump loss (0 = all)
        holo_every: Compute holonomy loss every N steps
        orbit_every: Compute orbit loss every N steps
        orbit_transforms: Number of random rotations for orbit loss
        closure_every: Compute closure loss every N steps
        closure_delta: Delta scale for closure loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if save_every > 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving training progress to: {output_dir}/")

    # Load dataset based on selection
    if use_mnist:
        X_full, labels, colors = get_mnist_data(n_samples)
        input_dim = 784
        dataset_name = "MNIST"
    else:
        X_full, labels, colors = get_nightmare_data(n_samples)
        input_dim = 3
        dataset_name = "Nightmare"

    X_full = X_full.to(device)
    labels_tensor = torch.from_numpy(labels).to(device)
    colors_tensor = torch.from_numpy(colors).to(device)

    # Create DataLoader for minibatching (include colors for visualization)
    dataset = TensorDataset(X_full, labels_tensor, colors_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)

    print(f"Dataset: {dataset_name} ({len(X_full)} samples, {steps_per_epoch} batches/epoch)")

    # Scale hidden_dim for high-dimensional input (MNIST)
    effective_hidden_dim = max(hidden_dim, 256) if use_mnist else hidden_dim

    model = LAPEncoder(
        input_dim=input_dim, latent_dim=2, num_charts=num_charts,
        hidden_dim=effective_hidden_dim
    ).to(device)
    model.ema_decay = ema_decay  # Set EMA decay for Fisher metric
    optimizer = optim.Adam(model.parameters(), lr=lr if not use_scheduler else lr / 10)

    # Initialize Level 5 components
    sieve = SieveMonitor(num_charts=num_charts, latent_dim=2) if run_sieve else None
    adaptive_lambdas = AdaptiveLambdas([
        'invariance', 'variance', 'covariance', 'entropy', 'balance',
        'jump', 'holonomy', 'orthogonality', 'orbit', 'disentangle',
        'window', 'closure', 'recon', 'tex_kl'
    ]) if use_adaptive_lambdas else None

    # Keep user-provided weights as anchors; adaptive scaling is applied relative to these.
    lambda_orbit_base = float(lambda_orbit)
    lambda_disentangle_base = float(lambda_disentangle)
    lambda_window_base = float(lambda_window)
    lambda_closure_base = float(lambda_closure)

    # OneCycleLR: warmup 10% → peak lr → cosine decay
    if use_scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=10,  # initial_lr = max_lr / 10
            final_div_factor=10,  # final_lr = max_lr / 100
        )
    else:
        scheduler = None

    print("Training Lindblad-Atlas Projection (Level 5 Fragile Agent)...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, Max LR: {lr}")
    print(f"  Gradient clipping: {grad_clip}, Scheduler: {'OneCycleLR' if use_scheduler else 'None'}")
    print(f"  EMA Riemannian: {use_riemannian} (decay={ema_decay})")
    print(f"  λ: orbit={lambda_orbit}, disentangle={lambda_disentangle}, window={lambda_window}, closure={lambda_closure}")
    print(
        "  Loss schedule: "
        f"jump={jump_mode} (every {jump_every}), "
        f"holo every {holo_every}, "
        f"orbit every {orbit_every} (n={orbit_transforms}), "
        f"closure every {closure_every}"
    )
    print(f"  Sieve Monitor: {run_sieve}, Adaptive λ: {use_adaptive_lambdas}")
    print("=" * 60)

    global_step = 0
    for epoch in range(epochs):
        progress = epoch / epochs
        model.tau = tau_start + (tau_end - tau_start) * progress

        epoch_loss = 0.0
        epoch_losses = {}
        n_batches = 0

        for batch_X, _, _ in train_loader:
            optimizer.zero_grad()

            # Forward pass on batch (return texture channel for reconstruction-only losses)
            z, assignment, z_per_chart, logits, _, x_recon, tex_mu, tex_logvar = model(
                batch_X, return_texture=True
            )

            # Update EMA Fisher metric (Lindblad Diagonal)
            model.update_fisher_ema(z, logits)

            # Compute loss with all Level 5 components
            loss, losses = lindblad_atlas_loss(
                z,
                assignment,
                logits,
                z_per_chart,
                batch_X,
                model,
                x_recon=x_recon,
                tex_mu=tex_mu,
                tex_logvar=tex_logvar,
                step=global_step,
                jump_mode=jump_mode,
                jump_every=jump_every,
                jump_sample_rate=jump_sample_rate,
                jump_num_pairs=jump_num_pairs,
                jump_proto_pairs=jump_proto_pairs,
                holo_every=holo_every,
                orbit_every=orbit_every,
                orbit_transforms=orbit_transforms,
                closure_every=closure_every,
                closure_delta=closure_delta,
                lambda_inv=lambda_inv,
                lambda_var=lambda_var,
                lambda_cov=lambda_cov,
                lambda_balance=lambda_balance,
                lambda_diversity=lambda_diversity,
                lambda_jump=lambda_jump,
                lambda_holo=lambda_holo,
                lambda_orbit=lambda_orbit,
                lambda_disentangle=lambda_disentangle,
                lambda_window=lambda_window,
                lambda_closure=lambda_closure,
                lambda_sep=lambda_sep,
                sep_margin=sep_margin,
            )

            loss.backward()

            # EMA-smoothed Riemannian gradient scaling (Section 2.5)
            # Uses G_ema instead of per-batch variance for stability
            if use_riemannian:
                with torch.no_grad():
                    # Macro damping: when routing is weakly grounded (low I(X;K)),
                    # damp router updates to avoid thrashing.
                    logK = float(np.log(model.num_charts))
                    I_ratio = 0.0 if logK <= 0 else float(losses.get("I_XK", 0.0)) / logK
                    macro_scale = float(np.clip(I_ratio, 0.1, 1.0))

                    # Adaptive ε stabilizer: noisier/less-grounded routing → more damping.
                    eps_metric = float(metric_eps_min + metric_eps_scale * (1.0 - I_ratio))

                    # Micro preconditioner (diagonal, EMA-smoothed, stabilized).
                    micro_scale = (1.0 / (model.G_ema + eps_metric)).mean()
                    micro_scale = torch.clamp(micro_scale, 0.1, 10.0)

                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if name.startswith("router."):
                            p.grad *= macro_scale
                        else:
                            p.grad *= micro_scale

            # NaN detection BEFORE optimizer step
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️ NaN/Inf detected at epoch {epoch}! Skipping batch.")
                optimizer.zero_grad()
                continue

            # Gradient clipping for stability with high LR
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Collapse detection and emergency correction
            if losses.get("_is_collapsed", False):
                with torch.no_grad():
                    # Add stronger noise to router to break symmetry
                    for name, p in model.named_parameters():
                        if name.startswith("router.") and p.requires_grad:
                            noise = torch.randn_like(p) * 0.05  # 5x stronger noise
                            p.add_(noise)

            epoch_loss += loss.item()
            for k, v in losses.items():
                if not k.startswith("_"):  # Skip internal flags
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1
            global_step += 1

        # Average losses over batches (handle all-NaN epochs)
        if n_batches == 0:
            print(f"  ⚠️ All batches NaN at epoch {epoch}! Reinitializing router...")
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name.startswith("router."):
                        nn.init.xavier_uniform_(p) if p.dim() >= 2 else nn.init.zeros_(p)
            continue

        epoch_loss /= n_batches
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Optional: adaptive λ weights (Section 3.5 style "loss balancing").
        # This is not a full constrained dual-ascent scheme; it is a conservative way to keep
        # auxiliary terms numerically relevant without manual retuning.
        if adaptive_lambdas is not None:
            scales = adaptive_lambdas.update(epoch_losses)
            lambda_orbit = float(np.clip(lambda_orbit_base * scales.get('orbit', 1.0), 1e-4, 1e4))
            lambda_disentangle = float(np.clip(lambda_disentangle_base * scales.get('disentangle', 1.0), 1e-4, 1e4))
            lambda_window = float(np.clip(lambda_window_base * scales.get('window', 1.0), 1e-4, 1e4))
            lambda_closure = float(np.clip(lambda_closure_base * scales.get('closure', 1.0), 1e-4, 1e4))

        # Visualization and logging (on full dataset)
        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs - 1):
            ext = ".png" if static_plots else ".html"
            save_path = f"{output_dir}/lap_epoch_{epoch:05d}{ext}"
            visualize_lap(model, X_full, colors, save_path=save_path, is_high_dim=use_mnist)

            if epoch % save_every == 0:
                # Compute full-dataset assignment for logging
                with torch.no_grad():
                    z_full, assignment_full, _, logits_full = model(X_full)
                    usage = assignment_full.mean(dim=0).cpu().numpy()

            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"Epoch {epoch:5d} | Loss: {epoch_loss:.4f} | τ: {model.tau:.3f} | LR: {current_lr:.2e}")
            print(f"  Usage: {usage}")
            print(
                f"  Components: inv={epoch_losses['invariance']:.3f} "
                f"var={epoch_losses['variance']:.3f} jump={epoch_losses['jump']:.3f} "
                f"holo={epoch_losses['holonomy']:.3f} div={epoch_losses.get('diversity', 0.0):.3f}"
            )
            print(
                f"  Gauge: orbit={epoch_losses['orbit']:.3f} "
                f"disentangle={epoch_losses['disentangle']:.3f}"
            )
            print(
                f"  Level5: window={epoch_losses['window']:.3f} "
                f"closure={epoch_losses['closure']:.3f} "
                f"recon={epoch_losses.get('recon', 0.0):.3f} "
                f"texKL={epoch_losses.get('tex_kl', 0.0):.3f} "
                f"I_XK={epoch_losses.get('I_XK', 0.0):.3f}"
            )

            if adaptive_lambdas is not None:
                print(
                    f"  λ(adaptive): orbit={lambda_orbit:.3g} "
                    f"disentangle={lambda_disentangle:.3g} "
                    f"window={lambda_window:.3g} "
                    f"closure={lambda_closure:.3g}"
                )

            # Run sieve checks (warn only)
            if sieve is not None:
                checks = sieve.run_checks(z_full, logits_full, epoch_losses)
                failed = [n for n, (p, _) in checks.items() if not p]
                if failed:
                    print(f"  ⚠️ Sieve warnings: {', '.join(failed)}")
                    print(sieve.format_report(checks))

            print("-" * 60)

    print("Training complete!")

    if save_every > 0:
        ext = ".png" if static_plots else ".html"
        visualize_lap(model, X_full, colors, save_path=f"{output_dir}/lap_final{ext}", is_high_dim=use_mnist)
        print(f"Final visualization saved to: {output_dir}/lap_final{ext}")

    return model, X_full, colors


# --- Main ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Lindblad-Atlas Projection - Level 5 Fragile Agent Training"
    )
    # Basic training
    parser.add_argument("--epochs", type=int, default=4000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Max learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable OneCycleLR")
    parser.add_argument("--tau-start", type=float, default=1.0, help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--tau-end", type=float, default=0.5, help="Final Gumbel-Softmax temperature")

    # Riemannian / EMA Fisher
    parser.add_argument("--no-riemannian", action="store_true", help="Disable EMA Riemannian scaling")
    parser.add_argument("--ema-decay", type=float, default=0.95, help="EMA decay for Fisher metric")
    parser.add_argument("--metric-eps-min", type=float, default=1e-6, help="Base ε for metric stabilization")
    parser.add_argument("--metric-eps-scale", type=float, default=1e-3, help="Extra ε when routing is weakly grounded")

    # Core loss weights (VICReg + Topology)
    parser.add_argument("--lambda-inv", type=float, default=10.0, help="VICReg invariance weight")
    parser.add_argument("--lambda-var", type=float, default=25.0, help="VICReg variance weight")
    parser.add_argument("--lambda-cov", type=float, default=10.0, help="VICReg covariance weight")
    parser.add_argument("--lambda-balance", type=float, default=10.0, help="Chart balance weight")
    parser.add_argument("--lambda-diversity", type=float, default=50.0, help="Diversity loss weight (anti-collapse)")
    parser.add_argument("--lambda-jump", type=float, default=5.0, help="Jump consistency weight")
    parser.add_argument("--lambda-holo", type=float, default=10.0, help="Holonomy defect weight")

    # Level 5 loss weights
    parser.add_argument("--lambda-orbit", type=float, default=1.0, help="Orbit invariance weight")
    parser.add_argument("--lambda-disentangle", type=float, default=0.5, help="Macro-micro disentangle weight")
    parser.add_argument("--lambda-window", type=float, default=1.0, help="Information window weight")
    parser.add_argument("--lambda-closure", type=float, default=0.5, help="Causal enclosure weight")

    # Level 5 features
    parser.add_argument("--no-sieve", action="store_true", help="Disable sieve monitoring")
    parser.add_argument("--adaptive-lambdas", action="store_true", help="Enable adaptive loss weights")

    # Loss scheduling / proxies
    parser.add_argument(
        "--jump-mode",
        type=str,
        default="full",
        choices=["full", "sampled", "proto"],
        help="Jump loss mode: full kNN, sampled pairs, or prototypes",
    )
    parser.add_argument("--jump-every", type=int, default=1, help="Compute jump loss every N steps")
    parser.add_argument("--jump-sample-rate", type=float, default=0.1, help="Sample rate for kNN jump loss")
    parser.add_argument("--jump-num-pairs", type=int, default=512, help="Pairs for sampled jump loss")
    parser.add_argument("--jump-proto-pairs", type=int, default=0, help="Chart pairs for proto jump loss (0=all)")
    parser.add_argument("--holo-every", type=int, default=1, help="Compute holonomy loss every N steps")
    parser.add_argument("--orbit-every", type=int, default=1, help="Compute orbit loss every N steps")
    parser.add_argument("--orbit-transforms", type=int, default=4, help="SO(3) rotations per orbit loss")
    parser.add_argument("--closure-every", type=int, default=1, help="Compute closure loss every N steps")
    parser.add_argument("--closure-delta", type=float, default=0.05, help="Delta scale for closure loss")

    # Output
    parser.add_argument("--save-every", type=int, default=10, help="Save viz every N epochs")
    parser.add_argument("--output-dir", type=str, default="lap_training", help="Output directory")
    parser.add_argument("--static-plots", action="store_true", help="Save static PNG instead of interactive HTML")

    # Dataset selection
    parser.add_argument("--mnist", action="store_true", help="Use MNIST dataset instead of nightmare")
    parser.add_argument("--n-samples", type=int, default=3000, help="Number of samples to use")

    # Model architecture
    parser.add_argument("--num-charts", type=int, default=4, help="Number of charts (atlas pages)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")

    # Chart separation (Topological Surgery)
    parser.add_argument("--lambda-sep", type=float, default=1.0, help="Chart separation weight")
    parser.add_argument("--sep-margin", type=float, default=2.0, help="Minimum distance between chart centers")

    args = parser.parse_args()

    model, X, colors = train_lap(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        save_every=args.save_every,
        output_dir=args.output_dir,
        grad_clip=args.grad_clip,
        use_scheduler=not args.no_scheduler,
        use_riemannian=not args.no_riemannian,
        # Core loss weights
        lambda_inv=args.lambda_inv,
        lambda_var=args.lambda_var,
        lambda_cov=args.lambda_cov,
        lambda_balance=args.lambda_balance,
        lambda_diversity=args.lambda_diversity,
        lambda_jump=args.lambda_jump,
        lambda_holo=args.lambda_holo,
        # Level 5 loss weights
        lambda_orbit=args.lambda_orbit,
        lambda_disentangle=args.lambda_disentangle,
        lambda_window=args.lambda_window,
        lambda_closure=args.lambda_closure,
        run_sieve=not args.no_sieve,
        use_adaptive_lambdas=args.adaptive_lambdas,
        ema_decay=args.ema_decay,
        metric_eps_min=args.metric_eps_min,
        metric_eps_scale=args.metric_eps_scale,
        static_plots=args.static_plots,
        jump_mode=args.jump_mode,
        jump_every=args.jump_every,
        jump_sample_rate=args.jump_sample_rate,
        jump_num_pairs=args.jump_num_pairs,
        jump_proto_pairs=args.jump_proto_pairs,
        holo_every=args.holo_every,
        orbit_every=args.orbit_every,
        orbit_transforms=args.orbit_transforms,
        closure_every=args.closure_every,
        closure_delta=args.closure_delta,
        # Dataset selection
        use_mnist=args.mnist,
        n_samples=args.n_samples,
        # Model architecture
        num_charts=args.num_charts,
        hidden_dim=args.hidden_dim,
        # Chart separation
        lambda_sep=args.lambda_sep,
        sep_margin=args.sep_margin,
    )
    ext = ".png" if args.static_plots else ".html"
    print(f"Training complete! Visualizations saved to {args.output_dir}/ ({ext} format)")
