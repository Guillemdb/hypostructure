"""
TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE

This script benchmarks the Attentive Atlas architecture (from fragile-index.md Section 7.8)
against a standard VQ-VAE on the "Manifold Mixture" problem.

The Manifold Mixture consists of three distinct geometric shapes:
1. Swiss Roll (flat curvature, rolled up)
2. Circles (topological loop)
3. Moons (discontinuous clusters)

Key architectural components (from mermaid diagram):
- Cross-attention router with learnable chart query bank
- Local VQ codebooks per chart
- Recursive decomposition: delta_total → (z_n, z_tex)
- TopologicalDecoder (inverse atlas) from Section 7.10

Metrics reported:
- Convergence speed (MSE vs epoch)
- Topological accuracy (AMI between ground truth and learned charts)
- Codebook usage (perplexity)

Usage:
    python topoencoder.py [--epochs 1000] [--n_samples 3000]

Reference: fragile-index.md Sections 7.8, 7.10
"""

import argparse
import math
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from datasets import (
    compute_chart_colors,
    find_boundary_pairs,
    get_nightmare_data,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class TopoEncoderConfig:
    """Configuration for the TopoEncoder benchmark."""

    # Data (using 3D nightmare dataset: Swiss Roll + Sphere + Moons)
    n_samples: int = 3000  # Total samples (divided by 3 per manifold)
    input_dim: int = 3  # 3D input for nightmare dataset

    # Model architecture
    hidden_dim: int = 32
    latent_dim: int = 2  # For 2D visualization
    num_charts: int = 3  # Match number of manifolds
    codes_per_chart: int = 32  # Better coverage (was 21)
    num_codes_standard: int = 64

    # Training
    epochs: int = 1000
    batch_size: int = 256  # Batch size for training (0 = full batch)
    lr: float = 1e-3
    vq_commitment_cost: float = 0.25
    entropy_weight: float = 0.1  # Encourage sharp routing (was 0.01)
    consistency_weight: float = 0.1  # Align encoder/decoder routing

    # Tier 1 losses (low overhead ~5%)
    variance_weight: float = 0.1  # Prevent latent collapse
    diversity_weight: float = 0.1  # Prevent chart collapse (was 1.0)
    separation_weight: float = 0.1  # Force chart centers apart (was 0.5)
    separation_margin: float = 2.0  # Minimum distance between chart centers

    # Tier 2 losses (medium overhead ~5%)
    window_weight: float = 0.5  # Information-stability (Theorem 15.1.3)
    window_eps_ground: float = 0.1  # Minimum I(X;K) threshold
    disentangle_weight: float = 0.1  # Gauge coherence (K ⊥ z_n)

    # Tier 3 losses (geometry/codebook health)
    orthogonality_weight: float = 0.01  # Metric isometry (W^T W ≈ I)
    code_entropy_weight: float = 0.1  # Prevent local index collapse

    # Tier 4 losses (invariance - expensive when enabled, disabled by default)
    kl_prior_weight: float = 0.01  # Residual KL prior on z_n, z_tex
    orbit_weight: float = 0.0  # Chart invariance under augmentation (2x slowdown)
    vicreg_inv_weight: float = 0.0  # Latent invariance (shares augmentation pass)
    augment_noise_std: float = 0.1  # Augmentation noise level
    augment_rotation_max: float = 0.3  # Max rotation in radians

    # Learning rate scheduling
    use_scheduler: bool = True  # Use cosine annealing LR scheduler
    min_lr: float = 1e-5  # Minimum LR at end of schedule

    # Gradient clipping
    grad_clip: float = 1.0  # Max gradient norm (0 to disable)

    # Benchmark control
    disable_ae: bool = False  # Skip VanillaAE baseline
    disable_vq: bool = False  # Skip StandardVQ baseline

    # Logging and output
    log_every: int = 100
    save_every: int = 100  # Save visualization every N epochs (0 to disable)
    output_dir: str = "outputs/topoencoder"

    # Device (CUDA if available, else CPU)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compute_matching_hidden_dim(
    target_params: int,
    input_dim: int = 3,
    latent_dim: int = 2,
    num_codes: int = 64,
) -> int:
    """Compute hidden_dim for StandardVQ to match target parameter count.

    StandardVQ params = 2h² + (4 + 2 + 3 + 3)h + (2 + num_codes*latent_dim + 3)
                      = 2h² + 12h + (5 + num_codes*latent_dim)

    Using quadratic formula: h = (-12 + sqrt(144 + 8*(target - offset))) / 4
    """
    offset = 5 + num_codes * latent_dim
    # Adjust for input_dim: encoder.0 has input_dim*h, decoder.4 has h*input_dim
    # Full formula: 2h² + (input_dim + 2 + 2 + input_dim + 2 + 2)h + ...
    #             = 2h² + (2*input_dim + 8)h + offset
    coef_h = 2 * input_dim + 8
    # 2h² + coef_h*h + offset = target
    # h = (-coef_h + sqrt(coef_h² + 8*(target - offset))) / 4
    discriminant = coef_h**2 + 8 * (target_params - offset)
    if discriminant < 0:
        return 32  # fallback
    h = (-coef_h + math.sqrt(discriminant)) / 4
    return max(16, int(h))


# ==========================================
# 2. STANDARD VQ-VAE BASELINE
# ==========================================
class StandardVQ(nn.Module):
    """Standard Vector-Quantized VAE baseline.

    Uses a single global codebook with Euclidean distance quantization.
    This represents the typical VQ-VAE without topological awareness.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_codes: int = 64,
    ):
        super().__init__()

        # Encoder: x → z_e
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Codebook: learnable embeddings
        self.embeddings = nn.Embedding(num_codes, latent_dim)
        # Initialize uniformly
        self.embeddings.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        # Decoder: z_q → x_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with VQ.

        Returns:
            x_recon: Reconstructed input
            vq_loss: Commitment + codebook loss
            indices: Quantized code indices
        """
        # Encode
        z_e = self.encoder(x)  # [B, latent_dim]

        # Vector quantization (Euclidean distance)
        dists = torch.cdist(z_e, self.embeddings.weight)  # [B, num_codes]
        indices = torch.argmin(dists, dim=1)  # [B]
        z_q = self.embeddings(indices)  # [B, latent_dim]

        # VQ losses
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # Straight-through estimator
        z_out = z_e + (z_q - z_e).detach()

        # Decode
        x_recon = self.decoder(z_out)

        return x_recon, vq_loss, indices

    def compute_perplexity(self, indices: torch.Tensor) -> float:
        """Compute codebook usage perplexity."""
        num_codes = self.embeddings.num_embeddings
        counts = torch.bincount(indices, minlength=num_codes).float()
        probs = counts / counts.sum()
        # Filter zeros for log
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())


# ==========================================
# 3. VANILLA AUTOENCODER (Reconstruction Baseline)
# ==========================================
class VanillaAE(nn.Module):
    """Continuous Autoencoder baseline (reconstruction upper bound).

    No discrete bottleneck - should reconstruct perfectly but
    fails to capture topology/clustering explicitly. Serves as
    the "gold standard" for reconstruction quality.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        latent_dim: int = 2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            x_recon: Reconstructed input
            z: Latent representation
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# ==========================================
# 4. ATTENTIVE ATLAS ENCODER
# ==========================================
class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing.

    Architecture (from mermaid diagram):
    - Feature extractor: x → features [B, D]
    - Key/Value projections: features → k(x), v(x)
    - Chart Query Bank: learnable q_i [N_c, D]
    - Cross-attention Router: softmax(k @ q.T / sqrt(d)) → w_i(x)
    - Local VQ codebooks: per-chart quantization
    - Recursive decomposition: delta → (z_n, z_tex)

    Output: (K_chart, K_code, z_n, z_tex, router_weights, z_geo)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart

        # --- Shared Backbone (Feature Extractor) ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # Added - was missing
        )

        # --- Routing (Topology) ---
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        # Chart Query Bank: learnable prototypes for each manifold
        self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim))
        # Scale for attention
        self.scale = math.sqrt(hidden_dim)

        # --- Value (Geometry) ---
        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        # --- Local VQ Codebooks (one per chart) ---
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codes_per_chart, latent_dim) for _ in range(num_charts)]
        )
        # Initialize codebooks
        for cb in self.codebooks:
            if hasattr(cb, "weight"):
                cb.weight.data.uniform_(-1.0 / codes_per_chart, 1.0 / codes_per_chart)

        # --- Recursive Decomposition ---
        # Structure filter: extracts structured nuisance z_n from residual
        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # K_chart [B]
        torch.Tensor,  # K_code [B]
        torch.Tensor,  # z_n [B, D]
        torch.Tensor,  # z_tex [B, D]
        torch.Tensor,  # router_weights [B, N_c]
        torch.Tensor,  # z_geo [B, D] (for decoder)
        torch.Tensor,  # vq_loss
        torch.Tensor,  # indices_stack [B, N_c] (for code entropy loss)
    ]:
        """Forward pass through the Attentive Atlas.

        Returns:
            K_chart: Hard chart assignment (argmax of router)
            K_code: VQ code index within selected chart
            z_n: Structured nuisance (from structure filter)
            z_tex: Texture residual (reconstruction-only)
            router_weights: Soft routing weights [B, N_c]
            z_geo: Geometric latent (e_K + z_n) for decoder
            vq_loss: Combined VQ loss
            indices_stack: Code indices per chart [B, N_c] (for entropy loss)
        """
        B = x.shape[0]
        device = x.device

        # 1. Feature extraction
        features = self.feature_extractor(x)  # [B, hidden_dim]

        # 2. Cross-attention routing
        k = self.key_proj(features)  # [B, hidden_dim]
        # Attention: k @ q.T / sqrt(d)
        scores = torch.matmul(k, self.chart_queries.T) / self.scale  # [B, N_c]
        router_weights = F.softmax(scores, dim=-1)  # [B, N_c]

        # Hard chart assignment
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # 3. Value projection
        v = self.val_proj(features)  # [B, latent_dim]

        # 4. Local VQ per chart
        # For each chart, find nearest code
        z_q_list = []
        indices_list = []
        vq_loss = torch.tensor(0.0, device=device)

        for i in range(self.num_charts):
            # Distance to codes in this chart
            codebook_i = self.codebooks[i]
            assert isinstance(codebook_i, nn.Embedding)
            dists = torch.cdist(v, codebook_i.weight)  # [B, codes_per_chart]
            inds = torch.argmin(dists, dim=1)  # [B]
            z_q_local = codebook_i(inds)  # [B, latent_dim]

            z_q_list.append(z_q_local)
            indices_list.append(inds)

            # Weighted VQ loss (only train active chart)
            w = router_weights[:, i].unsqueeze(1).detach()  # [B, 1]
            commitment = ((v - z_q_local.detach()) ** 2 * w).mean()
            codebook = ((z_q_local - v.detach()) ** 2 * w).mean()
            vq_loss = vq_loss + codebook + 0.25 * commitment

        # Stack quantized codes
        z_stack = torch.stack(z_q_list, dim=1)  # [B, N_c, latent_dim]
        indices_stack = torch.stack(indices_list, dim=1)  # [B, N_c]

        # 5. Soft blending for differentiability
        z_q_blended = (z_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Get hard K_code from selected chart
        K_code = indices_stack[torch.arange(B, device=device), K_chart]  # [B]

        # 6. Recursive Decomposition
        # Residual: delta_total = v - z_q
        delta_total = v - z_q_blended.detach()  # Stop gradient for clean decomposition

        # Structure filter extracts z_n
        z_n = self.structure_filter(delta_total)  # [B, latent_dim]

        # Texture residual: z_tex = delta_total - z_n
        z_tex = delta_total - z_n  # [B, latent_dim]

        # 7. Geometric latent for decoder: z_geo = e_K + z_n
        # Use straight-through for z_q
        z_q_st = v + (z_q_blended - v).detach()
        z_geo = z_q_st + z_n  # [B, latent_dim]

        return K_chart, K_code, z_n, z_tex, router_weights, z_geo, vq_loss, indices_stack


# ==========================================
# 5. TOPOLOGICAL DECODER (Inverse Atlas)
# ==========================================
class TopologicalDecoder(nn.Module):
    """Topological Decoder (Inverse Atlas) from Section 7.10.

    The inverse atlas - decodes chart-local geometry back to observation space.
    **Autonomous**: Can infer routing from geometry alone during dreaming,
    or accept a discrete chart index during planning.

    Architecture:
    - Chart projectors: z_geo → h_i (one per chart)
    - Inverse router: z_geo → w_soft (infers routing from geometry)
    - One-hot: K_chart → w_hard (optional hard routing)
    - Chart blend: h_global = sum(w_i * h_i)
    - Texture projector: z_tex → h_tex
    - Add: h_total = h_global + h_tex
    - Renderer: h_total → x_hat

    Routing modes:
    - Discrete planning: provide chart_index, use one-hot hard routing
    - Continuous generation: omit chart_index, infer weights from z_geo
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim

        # Chart-specific projectors (one per chart)
        self.chart_projectors = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim) for _ in range(num_charts)]
        )

        # Inverse router (dreaming mode) - infers routing from geometry alone
        self.latent_router = nn.Linear(latent_dim, num_charts)

        # Texture projector (global)
        self.tex_projector = nn.Linear(latent_dim, hidden_dim)

        # Shared renderer
        self.renderer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor,
        chart_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent components.

        Args:
            z_geo: [B, D] geometric content (e_K + z_n)
            z_tex: [B, D] texture residual
            chart_index: [B] optional chart IDs for hard routing.
                         If None, infers routing from z_geo (dreaming mode).

        Returns:
            x_hat: [B, output_dim] reconstructed observation
            router_weights: [B, N_c] the routing weights used (for consistency loss)
        """
        # Determine routing weights
        if chart_index is not None:
            # Discrete planning mode: hard one-hot routing
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()
        else:
            # Continuous generation / dreaming mode: infer from geometry
            logits = self.latent_router(z_geo)
            router_weights = F.softmax(logits, dim=-1)

        # Project through each chart
        projected = []
        for proj in self.chart_projectors:
            projected.append(proj(z_geo))

        h_stack = torch.stack(projected, dim=1)  # [B, N_c, hidden_dim]

        # Blend using router weights
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden]

        # Add texture
        h_tex = self.tex_projector(z_tex)
        h_total = h_global + h_tex

        # Render to output
        x_hat = self.renderer(h_total)

        return x_hat, router_weights


# ==========================================
# 6. TOPO ENCODER (Full Model)
# ==========================================
class TopoEncoder(nn.Module):
    """Complete TopoEncoder: Encoder + Autonomous Decoder.

    Combines AttentiveAtlasEncoder and TopologicalDecoder.

    The decoder is autonomous: it can infer routing from geometry alone
    (dreaming mode) or use explicit chart indices (planning mode).

    Training modes:
    - use_hard_routing=True: Decoder uses K_chart from encoder (planning)
    - use_hard_routing=False: Decoder infers routing from z_geo (dreaming)

    Consistency loss aligns encoder and decoder routing distributions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
    ):
        super().__init__()
        self.num_charts = num_charts

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
        )

        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
    ) -> tuple[
        torch.Tensor,  # x_recon
        torch.Tensor,  # vq_loss
        torch.Tensor,  # enc_router_weights (from encoder)
        torch.Tensor,  # dec_router_weights (from decoder)
        torch.Tensor,  # K_chart
    ]:
        """Full forward pass.

        Args:
            x: Input tensor [B, D_in]
            use_hard_routing: If True, decoder uses K_chart (planning mode).
                              If False, decoder infers routing from z_geo (dreaming).

        Returns:
            x_recon: Reconstructed input
            vq_loss: VQ commitment + codebook loss
            enc_router_weights: Encoder routing weights (for entropy loss)
            dec_router_weights: Decoder routing weights (for consistency loss)
            K_chart: Hard chart assignments from encoder
        """
        K_chart, _K_code, _z_n, z_tex, enc_router_weights, z_geo, vq_loss, _indices = (
            self.encoder(x)
        )

        # Decoder can use hard routing (planning) or infer from geometry (dreaming)
        chart_index = K_chart if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(z_geo, z_tex, chart_index)

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart

    def compute_consistency_loss(
        self,
        enc_weights: torch.Tensor,
        dec_weights: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute KL divergence between encoder and decoder routing.

        Keeps the inverse router aligned with the encoder routing.
        L_consistency = KL(w_enc || w_dec)
        """
        # KL(P || Q) = sum(P * log(P/Q))
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(
            dim=-1
        )
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        """Compute chart usage perplexity."""
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())


# ==========================================
# 7. METRICS
# ==========================================
def compute_routing_entropy(router_weights: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute mean routing entropy (lower = sharper decisions)."""
    entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    return entropy.mean().item()


def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


# ==========================================
# 8. ADDITIONAL LOSS FUNCTIONS (from embed_fragile.py)
# ==========================================


def compute_variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Prevent latent collapse by ensuring std >= 1 per dimension.

    From VICReg (variance component only). Skips invariance which requires augmentation.
    Overhead: ~1-2% (just variance computation).
    """
    var_z = z.var(dim=0).clamp(min=eps)
    std_z = torch.sqrt(var_z + eps)
    return F.relu(1 - std_z).mean()


def compute_diversity_loss(
    router_weights: torch.Tensor, num_charts: int, eps: float = 1e-6
) -> torch.Tensor:
    """Prevent chart collapse by maximizing entropy of mean usage.

    loss_diversity = log(K) - H(K)
    - Returns 0 when uniform (all charts equally used)
    - Returns positive when collapsed (one chart dominates)

    Overhead: ~1% (simple statistics).
    """
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    log_K = float(np.log(num_charts))
    return log_K - H_K


def compute_separation_loss(
    z_geo: torch.Tensor,
    router_weights: torch.Tensor,
    num_charts: int,
    margin: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Force chart centers apart in latent space.

    Implements Topological Surgery: prevents Ontological Mixing.
    Uses hinge loss: penalize if centers closer than margin.

    Overhead: ~1% (O(K²) pairwise distances).
    """
    device = z_geo.device

    # Compute weighted center for each chart
    centers = []
    for i in range(num_charts):
        weights = router_weights[:, i : i + 1]
        weight_sum = weights.sum() + eps
        center = (z_geo * weights).sum(dim=0) / weight_sum
        centers.append(center)
    centers_tensor = torch.stack(centers)  # [K, D]

    # Hinge loss: force centers at least 'margin' apart
    loss_sep = torch.tensor(0.0, device=device)
    n_pairs = 0
    for i in range(num_charts):
        for j in range(i + 1, num_charts):
            dist = torch.norm(centers_tensor[i] - centers_tensor[j])
            loss_sep = loss_sep + F.relu(margin - dist)
            n_pairs += 1

    return loss_sep / max(n_pairs, 1)


def compute_window_loss(
    router_weights: torch.Tensor,
    num_charts: int,
    eps_ground: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Information-Stability Window (Theorem 15.1.3).

    Ensures chart assignment carries information about input:
    I(X;K) = H(K) - H(K|X) >= eps_ground

    Returns:
        loss: Penalty for insufficient grounding
        metrics: Dictionary with H(K), H(K|X), I(X;K)

    Overhead: ~2% (entropy statistics).
    """
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    H_K_given_X = -(router_weights * torch.log(router_weights + eps)).sum(dim=1).mean()
    I_XK = H_K - H_K_given_X

    # Penalize if I(X;K) < eps_ground (not enough information)
    loss_ground = F.relu(eps_ground - I_XK).pow(2)

    metrics = {
        "H_K": H_K.item(),
        "H_K_given_X": H_K_given_X.item(),
        "I_XK": I_XK.item(),
    }
    return loss_ground, metrics


def compute_disentangle_loss(
    z_geo: torch.Tensor,
    router_weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gauge coherence: macro-nuisance independence.

    L_{K⊥n} = ||Cov(q(K|x), z_geo)||²_F

    Chart ID (macro) shouldn't predict position within chart (micro).
    Encourages clean separation of routing and geometry.

    Overhead: ~3% (cross-covariance computation).
    """
    device = z_geo.device
    B = z_geo.shape[0]

    if B < 2:
        return torch.tensor(0.0, device=device)

    # Center both representations
    z_centered = z_geo - z_geo.mean(dim=0, keepdim=True)
    w_centered = router_weights - router_weights.mean(dim=0, keepdim=True)

    # Clamp to prevent extreme values
    z_centered = torch.clamp(z_centered, -100, 100)
    w_centered = torch.clamp(w_centered, -1, 1)

    # Cross-covariance matrix [K, D]
    cross_cov = (w_centered.T @ z_centered) / max(B - 1, 1)

    # Frobenius norm squared
    result = (cross_cov**2).sum()

    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return result


def compute_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    """Enforce W^T W ≈ I for linear layers (metric isometry).

    Keeps the latent space well-conditioned by ensuring projections
    preserve distances (Section 7.7.2 - Action Metric).

    Without this, linear layers can stretch/squash space arbitrarily,
    breaking the geometric assumptions of the atlas.

    Overhead: ~1% (O(d³) matrix multiplication, tiny vs data forward pass).
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_layers = 0

    for name, param in model.named_parameters():
        # Apply to linear weights (2D tensors)
        if "weight" in name and param.dim() == 2:
            W = param
            rows, cols = W.shape
            if rows > cols:  # Semi-orthogonal: W^T W ≈ I
                gram = torch.mm(W.t(), W)
                I = torch.eye(cols, device=W.device)
            else:  # W W^T ≈ I
                gram = torch.mm(W, W.t())
                I = torch.eye(rows, device=W.device)

            # Frobenius norm of deviation from identity
            loss = loss + torch.norm(gram - I) ** 2
            n_layers += 1

    # Average over layers for stable gradient
    return loss / max(n_layers, 1)


def compute_code_entropy_loss(
    indices_stack: torch.Tensor,
    num_codes: int,
) -> torch.Tensor:
    """Maximize entropy of code usage within batch (micro-diversity).

    Prevents "index collapse" where a chart routes perfectly but
    maps every point to a single code index.

    Reference: Node 11 (ComplexCheck), Section 15.1 (Mixing Rate).

    Args:
        indices_stack: [B, N_charts] - code indices chosen per chart
        num_codes: Number of codes per chart

    Returns:
        loss: (max_entropy - H) where H is empirical code entropy

    Overhead: ~1% (just counting indices in batch).
    """
    device = indices_stack.device

    # Flatten all indices from all charts
    flat_indices = indices_stack.flatten()

    # Calculate empirical probabilities
    counts = torch.bincount(flat_indices, minlength=num_codes).float()
    probs = counts / (counts.sum() + 1e-6)

    # Filter zeros for log stability
    probs_nonzero = probs[probs > 0]

    # Entropy H(K_code)
    entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero + 1e-6))

    # Maximize entropy → minimize (max_entropy - H)
    max_entropy = math.log(num_codes)
    return torch.tensor(max_entropy, device=device) - entropy


# ==========================================
# 8b. TIER 4 LOSSES (Invariance - Expensive)
# ==========================================


def augment_nightmare(
    x: torch.Tensor,
    noise_std: float = 0.1,
    rotation_max: float = 0.3,
) -> torch.Tensor:
    """Apply random rotation + noise to 3D nightmare data.

    Augmentations preserve manifold identity but change local position.
    Used for orbit invariance and VICReg invariance losses.

    Args:
        x: Input tensor [B, 3] (3D points)
        noise_std: Standard deviation of additive noise
        rotation_max: Maximum rotation angle in radians (±)

    Returns:
        Augmented tensor [B, 3]
    """
    B = x.shape[0]
    device = x.device

    # Random rotation around Z-axis (preserves manifold structure)
    theta = torch.rand(B, device=device) * 2 * rotation_max - rotation_max
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Apply rotation (Z-axis): only affects X and Y
    x_rot = x.clone()
    x_rot[:, 0] = cos_t * x[:, 0] - sin_t * x[:, 1]
    x_rot[:, 1] = sin_t * x[:, 0] + cos_t * x[:, 1]
    # x_rot[:, 2] unchanged (Z-axis)

    # Add small noise
    x_aug = x_rot + torch.randn_like(x) * noise_std

    return x_aug


def compute_kl_prior_loss(z_n: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
    """KL divergence from standard normal prior for residual channels.

    Regularizes z_n and z_tex toward N(0,1). Uses simplified form assuming
    unit variance: KL(N(mu, 1) || N(0, 1)) = 0.5 * mu^2.

    Reference: fragile-index.md L_nuis-KL and L_tex-KL.

    Overhead: ~0.01% (element-wise operations on small vectors).
    """
    kl_n = 0.5 * (z_n**2).mean()
    kl_tex = 0.5 * (z_tex**2).mean()
    return kl_n + kl_tex


def compute_orbit_loss(
    enc_w: torch.Tensor,
    enc_w_aug: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Chart assignment should be invariant to augmentation (Node 18).

    Uses symmetric KL divergence between original and augmented routing
    distributions for stability.

    Reference: fragile-index.md L_orbit.

    Args:
        enc_w: Router weights for original input [B, K]
        enc_w_aug: Router weights for augmented input [B, K]

    Returns:
        Symmetric KL divergence (scalar)
    """
    # Symmetric KL for stability: 0.5 * (KL(P||Q) + KL(Q||P))
    kl_forward = (enc_w * torch.log((enc_w + eps) / (enc_w_aug + eps))).sum(dim=-1)
    kl_backward = (enc_w_aug * torch.log((enc_w_aug + eps) / (enc_w + eps))).sum(dim=-1)
    return 0.5 * (kl_forward + kl_backward).mean()


def compute_vicreg_invariance_loss(
    z_geo: torch.Tensor,
    z_geo_aug: torch.Tensor,
) -> torch.Tensor:
    """Latent geometry should be stable under augmentation (Section 7.7.3).

    Simple MSE between original and augmented geometric latents.
    Encourages the encoder to learn transformation-invariant representations.

    Reference: VICReg invariance component, fragile-index.md L_inv.

    Args:
        z_geo: Geometric latent for original input [B, D]
        z_geo_aug: Geometric latent for augmented input [B, D]

    Returns:
        MSE loss (scalar)
    """
    return F.mse_loss(z_geo, z_geo_aug)


# ==========================================
# 9. TRAINING
# ==========================================
def train_benchmark(config: TopoEncoderConfig) -> dict:
    """Train both models and return results.

    Returns dictionary with:
        - std_losses: List of StandardVQ losses per epoch
        - atlas_losses: List of TopoEncoder losses per epoch
        - ami_score: Final AMI for TopoEncoder
        - std_perplexity: Final perplexity for StandardVQ
        - atlas_perplexity: Final perplexity for TopoEncoder
        - X: Input data
        - labels: Ground truth labels
        - chart_assignments: Learned chart assignments
    """
    # Create output directory
    if config.save_every > 0:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Saving training progress to: {config.output_dir}/")

    # Generate data (3D nightmare dataset with rainbow colors)
    X, labels, colors = get_nightmare_data(config.n_samples)
    print(f"Generated {len(X)} points from 3 manifolds (Swiss Roll, Sphere, Moons)")

    # Create TopoEncoder first to get its parameter count
    model_atlas = TopoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
    )
    topo_params = count_parameters(model_atlas)

    # Create StandardVQ with matching parameter count (fair comparison)
    model_std = None
    opt_std = None
    std_params = 0
    std_hidden_dim = 0
    if not config.disable_vq:
        std_hidden_dim = compute_matching_hidden_dim(
            target_params=topo_params,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes_standard,
        )
        model_std = StandardVQ(
            input_dim=config.input_dim,
            hidden_dim=std_hidden_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes_standard,
        )
        std_params = count_parameters(model_std)

    # Create VanillaAE with similar parameter count (reconstruction baseline)
    model_ae = None
    opt_ae = None
    ae_params = 0
    ae_hidden_dim = 0
    if not config.disable_ae:
        ae_hidden_dim = compute_matching_hidden_dim(
            target_params=topo_params,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_codes=0,  # No codebook in AE
        )
        model_ae = VanillaAE(
            input_dim=config.input_dim,
            hidden_dim=ae_hidden_dim,
            latent_dim=config.latent_dim,
        )
        ae_params = count_parameters(model_ae)

    print(f"\nModel Parameters (fair comparison):")
    print(f"  TopoEncoder: {topo_params:,} params (hidden_dim={config.hidden_dim})")
    if not config.disable_vq:
        print(f"  StandardVQ:  {std_params:,} params (hidden_dim={std_hidden_dim})")
    else:
        print(f"  StandardVQ:  DISABLED")
    if not config.disable_ae:
        print(f"  VanillaAE:   {ae_params:,} params (hidden_dim={ae_hidden_dim})")
    else:
        print(f"  VanillaAE:   DISABLED")

    # Move models and data to device
    device = torch.device(config.device)
    model_atlas = model_atlas.to(device)
    if model_std is not None:
        model_std = model_std.to(device)
    if model_ae is not None:
        model_ae = model_ae.to(device)
    X = X.to(device)
    print(f"  Device: {device}")

    # Optimizers
    if model_std is not None:
        opt_std = optim.Adam(model_std.parameters(), lr=config.lr)
    opt_atlas = optim.Adam(model_atlas.parameters(), lr=config.lr)
    if model_ae is not None:
        opt_ae = optim.Adam(model_ae.parameters(), lr=config.lr)

    # Learning rate scheduler (cosine annealing)
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt_atlas, T_max=config.epochs, eta_min=config.min_lr
        )

    # Create data loader for minibatching (data already on device)
    from torch.utils.data import DataLoader, TensorDataset
    labels_t = torch.from_numpy(labels).float().to(device)
    colors_t = torch.from_numpy(colors).float().to(device)
    dataset = TensorDataset(X, labels_t, colors_t)
    batch_size = config.batch_size if config.batch_size > 0 else len(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training history
    std_losses = []
    atlas_losses = []
    ae_losses = []  # VanillaAE baseline
    loss_components: dict[str, list[float]] = {
        "recon": [],
        "vq": [],
        "entropy": [],
        "consistency": [],
        # Tier 1 losses
        "variance": [],
        "diversity": [],
        "separation": [],
        # Tier 2 losses
        "window": [],
        "disentangle": [],
        # Tier 3 losses
        "orthogonality": [],
        "code_entropy": [],
        # Tier 4 losses (conditional)
        "kl_prior": [],
        "orbit": [],
        "vicreg_inv": [],
    }
    info_metrics: dict[str, list[float]] = {
        "I_XK": [],
        "H_K": [],
    }

    print("=" * 60)
    print("Training TopoEncoder (Attentive Atlas)")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch size: {batch_size}")
    print(f"  Charts: {config.num_charts}, Codes/chart: {config.codes_per_chart}")
    print(f"  λ: entropy={config.entropy_weight}, consistency={config.consistency_weight}")
    print("=" * 60)

    for epoch in range(config.epochs + 1):
        # Accumulate batch losses for epoch average
        epoch_std_loss = 0.0
        epoch_atlas_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_losses = {k: 0.0 for k in loss_components.keys()}
        epoch_info = {"I_XK": 0.0, "H_K": 0.0}
        n_batches = 0

        for batch_X, _batch_labels, _batch_colors in dataloader:
            n_batches += 1

            # --- Standard VQ Step ---
            loss_s = torch.tensor(0.0, device=device)
            if model_std is not None:
                recon_s, vq_loss_s, _ = model_std(batch_X)
                loss_s = F.mse_loss(recon_s, batch_X) + vq_loss_s
                opt_std.zero_grad()
                loss_s.backward()
                opt_std.step()

            # --- Vanilla AE Step (reconstruction baseline) ---
            loss_ae = torch.tensor(0.0, device=device)
            if model_ae is not None:
                recon_ae, _ = model_ae(batch_X)
                loss_ae = F.mse_loss(recon_ae, batch_X)
                opt_ae.zero_grad()
                loss_ae.backward()
                opt_ae.step()

            # --- Atlas Step (dreaming mode: decoder infers routing from z_geo) ---
            # Get encoder outputs (need z_geo for regularization losses)
            _, _, z_n, z_tex, enc_w, z_geo, vq_loss_a, indices_stack = model_atlas.encoder(batch_X)

            # Decoder forward (dreaming mode - infers routing from z_geo)
            recon_a, dec_w = model_atlas.decoder(z_geo, z_tex, chart_index=None)

            # Core losses
            recon_loss_a = F.mse_loss(recon_a, batch_X)
            entropy = compute_routing_entropy(enc_w)
            consistency = model_atlas.compute_consistency_loss(enc_w, dec_w)

            # Tier 1 losses (low overhead)
            var_loss = compute_variance_loss(z_geo)
            div_loss = compute_diversity_loss(enc_w, config.num_charts)
            sep_loss = compute_separation_loss(
                z_geo, enc_w, config.num_charts, config.separation_margin
            )

            # Tier 2 losses (medium overhead)
            window_loss, window_info = compute_window_loss(
                enc_w, config.num_charts, config.window_eps_ground
            )
            dis_loss = compute_disentangle_loss(z_geo, enc_w)

            # Tier 3 losses (geometry/codebook health)
            orth_loss = compute_orthogonality_loss(model_atlas)
            code_ent_loss = compute_code_entropy_loss(indices_stack, config.codes_per_chart)

            # Tier 4 losses (invariance - expensive, conditional computation)
            # KL prior (cheap, compute if enabled)
            if config.kl_prior_weight > 0:
                kl_loss = compute_kl_prior_loss(z_n, z_tex)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            # Orbit and VICReg invariance (expensive - share augmented forward pass)
            orbit_loss = torch.tensor(0.0, device=device)
            vicreg_loss = torch.tensor(0.0, device=device)

            if config.orbit_weight > 0 or config.vicreg_inv_weight > 0:
                # Single augmented forward pass (shared between both losses)
                x_aug = augment_nightmare(
                    batch_X, config.augment_noise_std, config.augment_rotation_max
                )
                _, _, _, _, enc_w_aug, z_geo_aug, _, _ = model_atlas.encoder(x_aug)

                if config.orbit_weight > 0:
                    orbit_loss = compute_orbit_loss(enc_w, enc_w_aug)
                if config.vicreg_inv_weight > 0:
                    vicreg_loss = compute_vicreg_invariance_loss(z_geo, z_geo_aug)

            # Total loss
            loss_a = (
                recon_loss_a
                + vq_loss_a
                + config.entropy_weight * entropy
                + config.consistency_weight * consistency
                # Tier 1
                + config.variance_weight * var_loss
                + config.diversity_weight * div_loss
                + config.separation_weight * sep_loss
                # Tier 2
                + config.window_weight * window_loss
                + config.disentangle_weight * dis_loss
                # Tier 3
                + config.orthogonality_weight * orth_loss
                + config.code_entropy_weight * code_ent_loss
                # Tier 4 (conditional - 0 if disabled)
                + config.kl_prior_weight * kl_loss
                + config.orbit_weight * orbit_loss
                + config.vicreg_inv_weight * vicreg_loss
            )

            opt_atlas.zero_grad()
            loss_a.backward()
            # Gradient clipping (prevents instability from competing losses)
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_atlas.parameters(), config.grad_clip)
            opt_atlas.step()

            # Accumulate batch losses
            epoch_std_loss += loss_s.item()
            epoch_atlas_loss += loss_a.item()
            epoch_ae_loss += loss_ae.item()
            epoch_losses["recon"] += recon_loss_a.item()
            epoch_losses["vq"] += vq_loss_a.item()
            epoch_losses["entropy"] += entropy
            epoch_losses["consistency"] += consistency.item()
            epoch_losses["variance"] += var_loss.item()
            epoch_losses["diversity"] += div_loss.item()
            epoch_losses["separation"] += sep_loss.item()
            epoch_losses["window"] += window_loss.item()
            epoch_losses["disentangle"] += dis_loss.item()
            epoch_losses["orthogonality"] += orth_loss.item()
            epoch_losses["code_entropy"] += code_ent_loss.item()
            epoch_losses["kl_prior"] += kl_loss.item()
            epoch_losses["orbit"] += orbit_loss.item()
            epoch_losses["vicreg_inv"] += vicreg_loss.item()
            epoch_info["I_XK"] += window_info["I_XK"]
            epoch_info["H_K"] += window_info["H_K"]

        # Average over batches
        std_losses.append(epoch_std_loss / n_batches)
        atlas_losses.append(epoch_atlas_loss / n_batches)
        ae_losses.append(epoch_ae_loss / n_batches)
        for k in loss_components.keys():
            loss_components[k].append(epoch_losses[k] / n_batches)
        info_metrics["I_XK"].append(epoch_info["I_XK"] / n_batches)
        info_metrics["H_K"].append(epoch_info["H_K"] / n_batches)

        # Step LR scheduler at end of each epoch
        if scheduler is not None:
            scheduler.step()

        # Logging and visualization (matching embed_fragile.py style)
        if config.save_every > 0 and (
            epoch % config.save_every == 0 or epoch == config.epochs
        ):
            # Compute metrics on full dataset
            with torch.no_grad():
                K_chart_full, _, _, _, enc_w_full, _, _, _ = model_atlas.encoder(X)
                usage = enc_w_full.mean(dim=0).cpu().numpy()
                chart_assignments = K_chart_full.cpu().numpy()
                ami = compute_ami(labels, chart_assignments)
                perplexity = model_atlas.compute_perplexity(K_chart_full)

            # Get epoch-averaged losses
            avg_loss = atlas_losses[-1]
            avg_recon = loss_components["recon"][-1]
            avg_vq = loss_components["vq"][-1]
            avg_entropy = loss_components["entropy"][-1]
            avg_consistency = loss_components["consistency"][-1]
            avg_var = loss_components["variance"][-1]
            avg_div = loss_components["diversity"][-1]
            avg_sep = loss_components["separation"][-1]
            avg_window = loss_components["window"][-1]
            avg_disent = loss_components["disentangle"][-1]
            avg_orth = loss_components["orthogonality"][-1]
            avg_code_ent = loss_components["code_entropy"][-1]
            avg_kl = loss_components["kl_prior"][-1]
            avg_orbit = loss_components["orbit"][-1]
            avg_vicreg = loss_components["vicreg_inv"][-1]
            avg_ixk = info_metrics["I_XK"][-1]
            avg_hk = info_metrics["H_K"][-1]

            # Print in embed_fragile.py style
            current_lr = scheduler.get_last_lr()[0] if scheduler else config.lr
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            print(f"  Usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={avg_recon:.3f} "
                f"vq={avg_vq:.3f} "
                f"entropy={avg_entropy:.3f} "
                f"consistency={avg_consistency:.3f}"
            )
            print(
                f"  Tier1: var={avg_var:.3f} "
                f"div={avg_div:.3f} "
                f"sep={avg_sep:.3f}"
            )
            print(
                f"  Tier2: window={avg_window:.3f} "
                f"disent={avg_disent:.3f}"
            )
            print(
                f"  Tier3: orth={avg_orth:.3f} "
                f"code_ent={avg_code_ent:.3f}"
            )
            print(
                f"  Tier4: kl={avg_kl:.3f} "
                f"orbit={avg_orbit:.3f} "
                f"vicreg={avg_vicreg:.3f}"
            )
            print(
                f"  Info: I(X;K)={avg_ixk:.3f} "
                f"H(K)={avg_hk:.3f}"
            )
            print(f"  Metrics: AMI={ami:.4f} perplexity={perplexity:.2f}/{config.num_charts}")
            print("-" * 60)

            # Save visualization
            save_path = f"{config.output_dir}/topo_epoch_{epoch:05d}.png"
            visualize_latent(model_atlas, X, colors, labels, save_path, epoch)

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    with torch.no_grad():
        # VanillaAE metrics (reconstruction baseline)
        mse_ae = 0.0
        ami_ae = 0.0
        recon_ae_final = None
        if model_ae is not None:
            recon_ae_final, z_ae = model_ae(X)
            mse_ae = F.mse_loss(recon_ae_final, X).item()
            # Use K-Means on latent space for clustering (K=num_charts)
            z_ae_np = z_ae.cpu().numpy()
            kmeans = KMeans(n_clusters=config.num_charts, random_state=42, n_init=10)
            ae_clusters = kmeans.fit_predict(z_ae_np)
            ami_ae = compute_ami(labels, ae_clusters)

        # Standard VQ metrics
        mse_std = 0.0
        ami_std = 0.0
        std_perplexity = 0.0
        recon_std_final = None
        if model_std is not None:
            recon_std_final, _, indices_s = model_std(X)
            std_perplexity = model_std.compute_perplexity(indices_s)
            mse_std = F.mse_loss(recon_std_final, X).item()
            # Cluster VQ codes to get comparable AMI
            vq_clusters = indices_s.cpu().numpy() % config.num_charts  # Simple modulo clustering
            ami_std = compute_ami(labels, vq_clusters)

        # Atlas metrics (use dreaming mode to test autonomous routing)
        recon_atlas_final, _, enc_w, dec_w, K_chart = model_atlas(X, use_hard_routing=False)
        chart_assignments = K_chart.cpu().numpy()
        atlas_perplexity = model_atlas.compute_perplexity(K_chart)
        ami_atlas = compute_ami(labels, chart_assignments)
        mse_atlas = F.mse_loss(recon_atlas_final, X).item()
        final_consistency = model_atlas.compute_consistency_loss(enc_w, dec_w).item()

    # Results table
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'MSE':>10} {'AMI':>10} {'Perplexity':>15}")
    print("-" * 70)
    if model_ae is not None:
        print(f"{'Vanilla AE':<20} {mse_ae:>10.5f} {ami_ae:>10.4f} {'N/A (K-Means)':<15}")
    if model_std is not None:
        print(f"{'Standard VQ':<20} {mse_std:>10.5f} {ami_std:>10.4f} {std_perplexity:>6.1f}/{config.num_codes_standard:<8}")
    print(f"{'TopoEncoder':<20} {mse_atlas:>10.5f} {ami_atlas:>10.4f} {atlas_perplexity:>6.1f}/{config.num_charts:<8}")
    print("-" * 70)

    # Interpretation (only if baselines enabled)
    if model_ae is not None and model_std is not None:
        print("\nInterpretation:")
        if mse_ae < mse_atlas < mse_std:
            print("  AE has best reconstruction (expected - no bottleneck)")
            print("  TopoEncoder beats VQ on reconstruction (atlas routing helps)")
        if ami_atlas > ami_ae and ami_atlas > ami_std:
            print("  TopoEncoder has best topology discovery (charts match manifolds)")
        if ami_ae < ami_atlas:
            print("  AE fails at topology despite good reconstruction (entangled latent)")

    print(f"\nRouting Consistency (KL): {final_consistency:.4f}")

    # Save final visualization
    if config.save_every > 0:
        final_path = f"{config.output_dir}/topo_final.png"
        visualize_latent(model_atlas, X, colors, labels, final_path, epoch=None)
        print(f"\nFinal visualization saved to: {final_path}")

    # Results dict uses already-computed reconstructions from final evaluation
    return {
        "std_losses": std_losses,
        "atlas_losses": atlas_losses,
        "ae_losses": ae_losses,
        "loss_components": loss_components,
        # AMI scores
        "ami_ae": ami_ae,
        "ami_std": ami_std,
        "ami_atlas": ami_atlas,
        # MSE scores
        "mse_ae": mse_ae,
        "mse_std": mse_std,
        "mse_atlas": mse_atlas,
        # Perplexity
        "std_perplexity": std_perplexity,
        "atlas_perplexity": atlas_perplexity,
        # Data
        "X": X,
        "labels": labels,
        "colors": colors,
        "chart_assignments": chart_assignments,
        # For reconstruction comparison
        "recon_ae": recon_ae_final,
        "recon_std": recon_std_final,
        "recon_atlas": recon_atlas_final,
        # Models (for further analysis)
        "model_ae": model_ae,
        "model_std": model_std,
        "model_atlas": model_atlas,
        "config": config,
    }


# ==========================================
# 9. VISUALIZATION
# ==========================================
def visualize_latent(
    model: TopoEncoder,
    X: torch.Tensor,
    colors: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    epoch: int | None = None,
) -> None:
    """Visualize latent space with 4-panel layout (matching embed_fragile.py style).

    Panels:
    1. 3D Input space colored by structure (rainbow)
    2. Latent space colored by structure (rainbow)
    3. Latent space colored by chart assignment
    4. Portal view with soft blending + boundary lines

    Args:
        model: TopoEncoder model
        X: Input data [N, 3] (3D nightmare dataset)
        colors: Continuous colors for rainbow [N]
        labels: Ground truth manifold labels [N]
        save_path: Path to save visualization
        epoch: Current epoch (for title), None for final
    """
    model.eval()
    with torch.no_grad():
        # Get encoder outputs
        K_chart, _, _, _z_tex, enc_w, z_geo, _, _ = model.encoder(X)

        z = z_geo.cpu().numpy()
        X_np = X.cpu().numpy()
        enc_w_np = enc_w.cpu().numpy()
        hard_assign = K_chart.cpu().numpy()

    fig = plt.figure(figsize=(20, 5))
    title_suffix = f" (Epoch {epoch})" if epoch is not None else " (Final)"

    # Panel 1: 3D Input space colored by structure (rainbow)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.scatter(
        X_np[:, 0], X_np[:, 1], X_np[:, 2],
        c=colors, cmap="rainbow", s=2, alpha=0.7
    )
    ax1.set_title(f"Input: The Nightmare{title_suffix}\n(Roll, Sphere, Moons)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Panel 2: Latent by structure (rainbow colormap)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(z[:, 0], z[:, 1], c=colors, cmap="rainbow", s=3, alpha=0.7)
    ax2.set_title("Latent Space\n(Colored by Structure)")
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")

    # Panel 3: Latent by chart assignment
    ax3 = fig.add_subplot(1, 4, 3)
    scatter3 = ax3.scatter(
        z[:, 0], z[:, 1], c=hard_assign, cmap="tab10", s=3, alpha=0.7
    )
    ax3.set_title("Chart Assignment\n(Topological Surgery)")
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    plt.colorbar(scatter3, ax=ax3, ticks=range(model.num_charts), label="Chart")

    # Panel 4: Portal view with soft blending + boundary lines
    ax4 = fig.add_subplot(1, 4, 4)
    blended_colors = compute_chart_colors(enc_w_np, model.num_charts)
    ax4.scatter(z[:, 0], z[:, 1], c=blended_colors, s=3, alpha=0.7)

    # Add boundary lines
    boundary_pairs = find_boundary_pairs(z, hard_assign, X_np, k=3)
    if len(boundary_pairs) > 500:
        indices = np.random.choice(len(boundary_pairs), 500, replace=False)
        boundary_pairs = [boundary_pairs[i] for i in indices]
    if boundary_pairs:
        segments = [(z[i], z[j]) for i, j in boundary_pairs]
        lc = LineCollection(segments, colors="gray", alpha=0.2, linewidths=0.5)
        ax4.add_collection(lc)

    ax4.set_title("Portal View\n(Soft Blending + Boundaries)")
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_results(results: dict, save_path: str = "benchmark_result.png") -> None:
    """Create final visualization comparing ground truth, charts, and reconstructions.

    Layout (2 rows x 4 cols):
    Row 1: Input (3D rainbow) | Chart Assignments | Loss Curves | AMI Comparison
    Row 2: VanillaAE Recon | Standard VQ Recon | TopoEncoder Recon | Error Histogram
    """
    X = results["X"].cpu().numpy()
    colors = results["colors"]
    chart_assignments = results["chart_assignments"]
    recon_ae = results["recon_ae"].cpu().numpy()
    recon_std = results["recon_std"].cpu().numpy()
    recon_atlas = results["recon_atlas"].cpu().numpy()

    fig = plt.figure(figsize=(24, 10))

    # ========== Row 1 ==========

    # Panel 1: 3D Input with rainbow coloring
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    ax1.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors, cmap="rainbow", s=2, alpha=0.7
    )
    ax1.set_title("Input: The Nightmare\n(Roll, Sphere, Moons)", fontsize=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Panel 2: Atlas Chart Assignments (3D)
    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    scatter2 = ax2.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=chart_assignments, cmap="tab10", s=2, alpha=0.7
    )
    ax2.set_title("Atlas Chart Assignments\n(Learned Topology)", fontsize=12)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label="Chart")

    # Panel 3: Loss Curves (3-way)
    ax3 = fig.add_subplot(2, 4, 3)
    epochs = range(len(results["std_losses"]))
    ax3.plot(epochs, results["ae_losses"], label="VanillaAE", alpha=0.8, linewidth=1.5, color="C2")
    ax3.plot(epochs, results["std_losses"], label="Standard VQ", alpha=0.8, linewidth=1.5, color="C0")
    ax3.plot(epochs, results["atlas_losses"], label="TopoEncoder", alpha=0.8, linewidth=1.5, color="C1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Convergence", fontsize=12)
    ax3.legend()
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Panel 4: AMI Comparison (Bar Chart)
    ax4 = fig.add_subplot(2, 4, 4)
    models = ["VanillaAE", "Standard VQ", "TopoEncoder"]
    ami_scores = [results["ami_ae"], results["ami_std"], results["ami_atlas"]]
    bar_colors = ["C2", "C0", "C1"]
    bars = ax4.bar(models, ami_scores, color=bar_colors, alpha=0.8)
    ax4.set_ylabel("AMI Score")
    ax4.set_title("Topology Discovery\n(Adjusted Mutual Information)", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Excellent threshold")
    ax4.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Good threshold")
    # Add value labels on bars
    for bar, score in zip(bars, ami_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # ========== Row 2: Reconstructions ==========

    # Panel 5: VanillaAE Reconstruction (3D)
    ax5 = fig.add_subplot(2, 4, 5, projection="3d")
    ax5.scatter(
        recon_ae[:, 0], recon_ae[:, 1], recon_ae[:, 2],
        c=colors, cmap="rainbow", s=2, alpha=0.7
    )
    mse_ae = results["mse_ae"]
    ax5.set_title(f"VanillaAE Reconstruction\nMSE: {mse_ae:.5f}", fontsize=12)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")

    # Panel 6: Standard VQ Reconstruction (3D)
    ax6 = fig.add_subplot(2, 4, 6, projection="3d")
    ax6.scatter(
        recon_std[:, 0], recon_std[:, 1], recon_std[:, 2],
        c=colors, cmap="rainbow", s=2, alpha=0.7
    )
    mse_std = results["mse_std"]
    ax6.set_title(f"Standard VQ Reconstruction\nMSE: {mse_std:.5f}", fontsize=12)
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_zlabel("Z")

    # Panel 7: TopoEncoder Reconstruction (3D)
    ax7 = fig.add_subplot(2, 4, 7, projection="3d")
    ax7.scatter(
        recon_atlas[:, 0], recon_atlas[:, 1], recon_atlas[:, 2],
        c=colors, cmap="rainbow", s=2, alpha=0.7
    )
    mse_atlas = results["mse_atlas"]
    ax7.set_title(f"TopoEncoder Reconstruction\nMSE: {mse_atlas:.5f}", fontsize=12)
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_zlabel("Z")

    # Panel 8: Reconstruction Error Histogram (3-way)
    ax8 = fig.add_subplot(2, 4, 8)
    error_ae = np.linalg.norm(X - recon_ae, axis=1)
    error_std = np.linalg.norm(X - recon_std, axis=1)
    error_atlas = np.linalg.norm(X - recon_atlas, axis=1)

    ax8.hist(error_ae, bins=50, alpha=0.5, label=f"AE (μ={error_ae.mean():.3f})", color="C2")
    ax8.hist(error_std, bins=50, alpha=0.5, label=f"VQ (μ={error_std.mean():.3f})", color="C0")
    ax8.hist(error_atlas, bins=50, alpha=0.5, label=f"Topo (μ={error_atlas.mean():.3f})", color="C1")
    ax8.set_xlabel("Reconstruction Error (L2)")
    ax8.set_ylabel("Count")
    ax8.set_title("Error Distribution", fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Add summary text with 3-way comparison
    ami_ae = results["ami_ae"]
    ami_std = results["ami_std"]
    ami_atlas = results["ami_atlas"]
    perp_std = results["std_perplexity"]
    perp_atlas = results["atlas_perplexity"]
    fig.suptitle(
        f"3-Way Benchmark | AMI: AE={ami_ae:.3f}, VQ={ami_std:.3f}, Topo={ami_atlas:.3f} | "
        f"Perplexity: VQ={perp_std:.1f}, Topo={perp_atlas:.1f}",
        fontsize=14, y=0.98
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFinal visualization saved to: {save_path}")


# ==========================================
# 10. MAIN
# ==========================================
def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="TopoEncoder Benchmark: Attentive Atlas vs Standard VQ-VAE"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (0 = full batch)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3000,
        help="Total samples (divided by 3 per manifold)",
    )
    parser.add_argument(
        "--num_charts",
        type=int,
        default=3,
        help="Number of atlas charts",
    )
    parser.add_argument(
        "--codes_per_chart",
        type=int,
        default=21,
        help="VQ codes per chart",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for TopoEncoder",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save visualization every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/topoencoder",
        help="Output directory for visualizations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Tier 4 losses (invariance)
    parser.add_argument(
        "--kl_prior_weight",
        type=float,
        default=0.01,
        help="KL prior weight on z_n, z_tex (default: 0.01)",
    )
    parser.add_argument(
        "--orbit_weight",
        type=float,
        default=0.0,
        help="Orbit invariance weight (default: 0.0, enables 2x slowdown)",
    )
    parser.add_argument(
        "--vicreg_inv_weight",
        type=float,
        default=0.0,
        help="VICReg invariance weight (default: 0.0, shares augmentation pass)",
    )
    parser.add_argument(
        "--augment_noise_std",
        type=float,
        default=0.1,
        help="Augmentation noise std (default: 0.1)",
    )
    parser.add_argument(
        "--augment_rotation_max",
        type=float,
        default=0.3,
        help="Max rotation in radians for augmentation (default: 0.3)",
    )

    # Training dynamics
    parser.add_argument(
        "--use_scheduler",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use cosine annealing LR scheduler (default: True)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum LR for scheduler (default: 1e-5)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 to disable, default: 1.0)",
    )

    # Benchmark control
    parser.add_argument(
        "--disable_ae",
        action="store_true",
        help="Disable VanillaAE baseline (faster training)",
    )
    parser.add_argument(
        "--disable_vq",
        action="store_true",
        help="Disable StandardVQ baseline (faster training)",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = TopoEncoderConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
        hidden_dim=args.hidden_dim,
        save_every=args.save_every,
        output_dir=args.output_dir,
        device=args.device,
        # Tier 4 losses
        kl_prior_weight=args.kl_prior_weight,
        orbit_weight=args.orbit_weight,
        vicreg_inv_weight=args.vicreg_inv_weight,
        augment_noise_std=args.augment_noise_std,
        augment_rotation_max=args.augment_rotation_max,
        # Training dynamics
        use_scheduler=args.use_scheduler,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        # Benchmark control
        disable_ae=args.disable_ae,
        disable_vq=args.disable_vq,
    )

    print("=" * 50)
    print("TopoEncoder Benchmark")
    print("Attentive Atlas vs Standard VQ-VAE")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"  Total samples: {config.n_samples}")
    print(f"  Num charts: {config.num_charts}")
    print(f"  Codes per chart: {config.codes_per_chart}")
    print(f"  Total atlas codes: {config.num_charts * config.codes_per_chart}")
    print(f"  Standard VQ codes: {config.num_codes_standard}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Save every: {config.save_every} epochs")

    # Run benchmark
    results = train_benchmark(config)

    # Save final comparison visualization
    os.makedirs(config.output_dir, exist_ok=True)
    final_path = f"{config.output_dir}/benchmark_result.png"
    visualize_results(results, save_path=final_path)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ami_atlas = results["ami_atlas"]
    ami_ae = results["ami_ae"]
    if ami_atlas > 0.8:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Excellent! Atlas discovered the topology.")
    elif ami_atlas > 0.5:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Good. Atlas partially learned the topology.")
    else:
        print(f"TopoEncoder AMI = {ami_atlas:.4f} - Poor. Atlas did not learn the topology well.")

    if ami_atlas > ami_ae:
        print(f"TopoEncoder beats VanillaAE ({ami_atlas:.3f} > {ami_ae:.3f}) - better topology!")
    else:
        print(f"VanillaAE beats TopoEncoder ({ami_ae:.3f} > {ami_atlas:.3f}) - K-Means works well here")
    print(f"\nOutput saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
