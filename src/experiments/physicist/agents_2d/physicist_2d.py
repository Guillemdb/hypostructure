"""
Physicist Agent for 2D environments with CNN encoder.

Split-Brain VAE architecture adapted for 2D observations:
- z_macro (4D): x, y, vx, vy - the physics state
- z_micro (32D): noise/texture encoding

The physics engine predicts z_macro_{t+1} from z_macro_t,
blind to z_micro. This enforces causal enclosure.

Extended with techniques from combined.py:
- BRST layers for near-orthogonal constraints
- Multi-chart routing for specialized encoders
- VICReg losses for robust representations
- Topology losses for chart separation
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BRSTLinear, BRSTConv2d, compute_total_brst_defect
from ..losses import (
    vicreg_invariance_loss,
    vicreg_variance_loss,
    vicreg_covariance_loss,
    router_entropy_loss,
    router_balance_loss,
    chart_separation_loss,
    compute_warmup_ratio,
)


@dataclass
class PhysicistConfig2D:
    """Configuration for the 2D Physicist Agent."""

    # Architecture
    image_size: int = 64               # Input image size
    hidden_channels: int = 32          # CNN hidden channels
    macro_dim: int = 4                 # z_macro dimension (x, y, vx, vy)
    micro_dim: int = 32                # z_micro dimension
    physics_hidden: int = 64           # Physics engine hidden size

    # Training - balanced loss weights
    info_dropout_prob: float = 0.3     # Probability of dropping z_micro (reduced)
    lambda_closure: float = 0.1        # Causal enclosure (reduced to let recon dominate early)
    lambda_slowness: float = 0.01      # Temporal smoothness (reduced)
    lambda_dispersion: float = 0.001   # KL weight on z_micro
    lambda_kl_macro: float = 0.0001    # KL weight on z_macro

    # Multi-chart routing (1 = original behavior)
    num_charts: int = 1
    use_brst: bool = False

    # VICReg weights (0 = disabled)
    lambda_vicreg_inv: float = 0.0
    lambda_vicreg_var: float = 0.0
    lambda_vicreg_cov: float = 0.0
    vicreg_noise_std: float = 0.05

    # Topology weights (0 = disabled)
    lambda_entropy: float = 0.0
    lambda_balance: float = 0.0
    lambda_separation: float = 0.0
    chart_separation_dist: float = 4.0

    # BRST weight
    lambda_brst: float = 0.0

    # Warmup for new losses
    new_loss_warmup_epochs: int = 20

    # Device
    device: str = "cuda"


class CNNEncoder(nn.Module):
    """CNN encoder for 2D observations."""

    def __init__(self, config: PhysicistConfig2D):
        super().__init__()
        self.config = config

        # CNN layers: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # Added BatchNorm for training stability
        self.conv = nn.Sequential(
            nn.Conv2d(1, config.hidden_channels, 4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels * 2, 4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(config.hidden_channels * 2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels * 2, config.hidden_channels * 4, 4, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(config.hidden_channels * 4),
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels * 4, config.hidden_channels * 4, 4, stride=2, padding=1),  # -> 4x4
            nn.BatchNorm2d(config.hidden_channels * 4),
            nn.ReLU(),
        )

        # Flatten dimension: 4*4*128 = 2048
        self.flat_dim = 4 * 4 * config.hidden_channels * 4

        # Split heads for macro/micro
        self.fc = nn.Linear(self.flat_dim, 256)

        self.macro_mean = nn.Linear(256, config.macro_dim)
        self.macro_logvar = nn.Linear(256, config.macro_dim)
        self.micro_mean = nn.Linear(256, config.micro_dim)
        self.micro_logvar = nn.Linear(256, config.micro_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode 2D observation to split latent.

        Args:
            x: Observation [batch, size, size] or [batch, 1, size, size]

        Returns:
            macro_mean, macro_logvar, micro_mean, micro_logvar
        """
        # Add channel dim if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, H, W]

        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc(h))

        # Clamp logvar to prevent NaN (exp(20) is already huge)
        macro_logvar = torch.clamp(self.macro_logvar(h), -20, 5)
        micro_logvar = torch.clamp(self.micro_logvar(h), -20, 5)

        return (
            self.macro_mean(h),
            macro_logvar,
            self.micro_mean(h),
            micro_logvar
        )


class ChartedCNNEncoder(nn.Module):
    """
    Multi-chart CNN encoder with router.

    Routes input to specialized chart encoders for z_macro,
    while using a shared encoder for z_micro.

    This allows different regions of the state space to have
    specialized encoders while maintaining universal noise encoding.
    """

    def __init__(self, config: PhysicistConfig2D):
        super().__init__()
        self.config = config
        self.num_charts = config.num_charts

        # Shared CNN backbone (same for all charts)
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        if config.use_brst:
            self.conv = nn.Sequential(
                BRSTConv2d(1, config.hidden_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels),
                nn.ReLU(),
                BRSTConv2d(config.hidden_channels, config.hidden_channels * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 2),
                nn.ReLU(),
                BRSTConv2d(config.hidden_channels * 2, config.hidden_channels * 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 4),
                nn.ReLU(),
                BRSTConv2d(config.hidden_channels * 4, config.hidden_channels * 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 4),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, config.hidden_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels),
                nn.ReLU(),
                nn.Conv2d(config.hidden_channels, config.hidden_channels * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 2),
                nn.ReLU(),
                nn.Conv2d(config.hidden_channels * 2, config.hidden_channels * 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 4),
                nn.ReLU(),
                nn.Conv2d(config.hidden_channels * 4, config.hidden_channels * 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_channels * 4),
                nn.ReLU(),
            )

        self.flat_dim = 4 * 4 * config.hidden_channels * 4

        # Router: decides which chart(s) to use
        self.router = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_charts),
        )

        # Chart-specific macro heads
        LinearLayer = BRSTLinear if config.use_brst else nn.Linear
        self.chart_fc = nn.ModuleList([
            LinearLayer(self.flat_dim, 256) for _ in range(config.num_charts)
        ])
        self.chart_macro_mean = nn.ModuleList([
            nn.Linear(256, config.macro_dim) for _ in range(config.num_charts)
        ])
        self.chart_macro_logvar = nn.ModuleList([
            nn.Linear(256, config.macro_dim) for _ in range(config.num_charts)
        ])

        # Shared micro head
        self.shared_fc = nn.Linear(self.flat_dim, 256)
        self.micro_mean = nn.Linear(256, config.micro_dim)
        self.micro_logvar = nn.Linear(256, config.micro_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with chart routing.

        Returns:
            macro_mean, macro_logvar, micro_mean, micro_logvar
            Also stores router_weights and chart_outputs for loss computation.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        h = self.conv(x)
        h_flat = h.view(h.size(0), -1)

        # Router weights
        router_logits = self.router(h_flat)
        self.router_weights = F.softmax(router_logits, dim=-1)

        # Chart-specific macro encodings
        self.chart_outputs = []
        chart_means = []
        chart_logvars = []

        for i in range(self.num_charts):
            h_chart = F.relu(self.chart_fc[i](h_flat))
            mean_i = self.chart_macro_mean[i](h_chart)
            logvar_i = torch.clamp(self.chart_macro_logvar[i](h_chart), -20, 5)
            chart_means.append(mean_i)
            chart_logvars.append(logvar_i)
            self.chart_outputs.append(mean_i)

        # Weighted combination of chart outputs
        chart_means_stack = torch.stack(chart_means, dim=1)  # [batch, num_charts, macro_dim]
        chart_logvars_stack = torch.stack(chart_logvars, dim=1)
        weights = self.router_weights.unsqueeze(-1)  # [batch, num_charts, 1]

        macro_mean = (chart_means_stack * weights).sum(dim=1)
        macro_logvar = (chart_logvars_stack * weights).sum(dim=1)

        # Shared micro encoding
        h_shared = F.relu(self.shared_fc(h_flat))
        micro_mean = self.micro_mean(h_shared)
        micro_logvar = torch.clamp(self.micro_logvar(h_shared), -20, 5)

        return macro_mean, macro_logvar, micro_mean, micro_logvar


class CNNDecoder(nn.Module):
    """CNN decoder for 2D observations."""

    def __init__(self, config: PhysicistConfig2D):
        super().__init__()
        self.config = config

        latent_dim = config.macro_dim + config.micro_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * 4 * config.hidden_channels * 4),
            nn.ReLU(),
        )

        # Transposed CNN: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_channels * 4, config.hidden_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels * 4, config.hidden_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels * 2, config.hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, 1, 4, stride=2, padding=1),
        )

    def forward(
        self,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        drop_micro: bool = False
    ) -> torch.Tensor:
        """
        Decode latent to 2D observation.

        Args:
            z_macro: [batch, macro_dim]
            z_micro: [batch, micro_dim]
            drop_micro: If True, zero out z_micro

        Returns:
            x_recon: [batch, size, size]
        """
        if drop_micro:
            z_micro = torch.zeros_like(z_micro)

        z = torch.cat([z_macro, z_micro], dim=-1)
        h = self.fc(z)
        h = h.view(h.size(0), self.config.hidden_channels * 4, 4, 4)
        x = self.deconv(h)
        return x.squeeze(1)  # Remove channel dim


class PhysicsEngine2D(nn.Module):
    """
    Physics engine for 2D bouncing ball.

    Predicts z_macro_{t+1} from z_macro_t using RESIDUAL prediction.
    BLIND to z_micro - enforces causal enclosure.

    Residual prediction: z_{t+1} = z_t + delta(z_t)
    This is easier to learn because identity is just delta=0.

    Optionally uses BRST layers for near-volume-preserving dynamics
    (Liouville's theorem in phase space).
    """

    def __init__(self, config: PhysicistConfig2D):
        super().__init__()
        self.config = config

        LinearLayer = BRSTLinear if config.use_brst else nn.Linear

        self.dynamics = nn.Sequential(
            LinearLayer(config.macro_dim, config.physics_hidden),
            nn.ReLU(),
            LinearLayer(config.physics_hidden, config.physics_hidden),
            nn.ReLU(),
            nn.Linear(config.physics_hidden, config.macro_dim),  # Final layer not BRST
        )

        # Initialize last layer to near-zero for residual learning
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)

    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        """Predict next z_macro using residual."""
        delta = self.dynamics(z_macro)
        return z_macro + delta  # Residual connection


class PhysicistAgent2D(nn.Module):
    """
    Physicist Agent for 2D environments.

    Architecture:
        Encoder (CNN): x → (z_macro, z_micro)
        Physics: z_macro_t → z_macro_{t+1} (blind to z_micro)
        Decoder (CNN): (z_macro, z_micro) → x_recon

    Key properties:
    - z_macro encodes physics (position, velocity)
    - z_micro encodes noise/texture
    - Physics engine enforces causal enclosure
    - Information dropout during training

    Extended features (when num_charts > 1):
    - Multi-chart routing for specialized macro encoders
    - VICReg losses for robust representations
    - Topology losses for chart separation
    - BRST constraints for near-orthogonal transformations
    """

    def __init__(self, config: Optional[PhysicistConfig2D] = None):
        super().__init__()
        self.config = config or PhysicistConfig2D()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        # Use charted encoder if num_charts > 1
        if self.config.num_charts > 1:
            self.encoder = ChartedCNNEncoder(self.config)
            self.is_charted = True
        else:
            self.encoder = CNNEncoder(self.config)
            self.is_charted = False

        self.physics = PhysicsEngine2D(self.config)
        self.decoder = CNNDecoder(self.config)

        # Apply proper weight initialization
        self._init_weights()

        self.to(self.device)

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming for better training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observation to latent."""
        macro_mean, macro_logvar, micro_mean, micro_logvar = self.encoder(x)

        z_macro = self.reparameterize(macro_mean, macro_logvar)
        z_micro = self.reparameterize(micro_mean, micro_logvar)

        result = {
            "z_macro": z_macro,
            "z_micro": z_micro,
            "macro_mean": macro_mean,
            "macro_logvar": macro_logvar,
            "micro_mean": micro_mean,
            "micro_logvar": micro_logvar,
        }

        # Include router info if using charted encoder
        if self.is_charted:
            result["router_weights"] = self.encoder.router_weights
            result["chart_outputs"] = self.encoder.chart_outputs

        return result

    def compute_brst_loss(self) -> torch.Tensor:
        """Compute total BRST defect across all BRST layers."""
        return compute_total_brst_defect(self)

    def decode(
        self,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        drop_micro: bool = False
    ) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(z_macro, z_micro, drop_micro)

    def predict_next_macro(self, z_macro: torch.Tensor) -> torch.Tensor:
        """Predict next z_macro using physics engine."""
        return self.physics(z_macro)

    def forward(
        self,
        x: torch.Tensor,
        drop_micro: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass."""
        latents = self.encode(x)
        x_recon = self.decode(latents["z_macro"], latents["z_micro"], drop_micro)
        return x_recon, latents

    def compute_loss(
        self,
        x_seq: torch.Tensor,
        training: bool = True,
        occlusion_mask: Optional[torch.Tensor] = None,
        warmup_ratio: float = 1.0,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a sequence.

        Args:
            x_seq: Observation sequence [batch, seq_len, H, W]
            training: Whether in training mode
            occlusion_mask: Optional [batch, seq_len, H, W] mask (1=visible, 0=occluded)
            warmup_ratio: 0-1, scales auxiliary losses (0=recon only, 1=full)
            epoch: Current epoch for new loss warmup

        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, H, W = x_seq.shape
        cfg = self.config

        # Flatten for encoding
        x_flat = x_seq.reshape(-1, H, W)

        # Encode all timesteps
        latents = self.encode(x_flat)

        # Reshape back
        z_macro = latents["z_macro"].reshape(batch_size, seq_len, -1)
        z_micro = latents["z_micro"].reshape(batch_size, seq_len, -1)
        macro_mean = latents["macro_mean"].reshape(batch_size, seq_len, -1)
        macro_logvar = latents["macro_logvar"].reshape(batch_size, seq_len, -1)
        micro_mean = latents["micro_mean"].reshape(batch_size, seq_len, -1)
        micro_logvar = latents["micro_logvar"].reshape(batch_size, seq_len, -1)

        # Information dropout - only if past warmup
        drop_micro = training and warmup_ratio > 0.5 and torch.rand(1).item() < cfg.info_dropout_prob

        # Reconstruction
        z_macro_flat = z_macro.reshape(-1, cfg.macro_dim)
        z_micro_flat = z_micro.reshape(-1, cfg.micro_dim)
        x_recon = self.decode(z_macro_flat, z_micro_flat, drop_micro)

        # Reconstruction loss (optionally masked for occlusion)
        if occlusion_mask is not None:
            mask_flat = occlusion_mask.reshape(-1, H, W)
            L_recon = (((x_recon - x_flat) ** 2) * mask_flat).sum() / mask_flat.sum()
        else:
            L_recon = F.mse_loss(x_recon, x_flat)

        # Causal enclosure loss
        z_macro_pred = self.predict_next_macro(z_macro[:, :-1].reshape(-1, cfg.macro_dim))
        z_macro_target = macro_mean[:, 1:].reshape(-1, cfg.macro_dim).detach()
        L_closure = F.mse_loss(z_macro_pred, z_macro_target)

        # Slowness loss
        L_slowness = F.mse_loss(z_macro[:, 1:], z_macro[:, :-1])

        # KL losses
        L_kl_micro = -0.5 * torch.mean(
            1 + micro_logvar - micro_mean.pow(2) - micro_logvar.exp()
        )
        L_kl_macro = -0.5 * torch.mean(
            1 + macro_logvar - macro_mean.pow(2) - macro_logvar.exp()
        )

        # Apply warmup scaling to auxiliary losses
        total = (
            L_recon +
            warmup_ratio * cfg.lambda_closure * L_closure +
            warmup_ratio * cfg.lambda_slowness * L_slowness +
            warmup_ratio * cfg.lambda_dispersion * L_kl_micro +
            warmup_ratio * cfg.lambda_kl_macro * L_kl_macro
        )

        losses = {
            "total": total,
            "recon": L_recon,
            "closure": L_closure,
            "slowness": L_slowness,
            "kl_micro": L_kl_micro,
            "kl_macro": L_kl_macro,
        }

        # =====================================================================
        # New losses (VICReg, Topology, BRST) with epoch-based warmup
        # =====================================================================
        new_loss_warmup = compute_warmup_ratio(epoch, cfg.new_loss_warmup_epochs)

        # VICReg losses on z_macro (physics state should be robust/decorrelated)
        if cfg.lambda_vicreg_inv > 0 or cfg.lambda_vicreg_var > 0 or cfg.lambda_vicreg_cov > 0:
            z_macro_flat_for_vicreg = macro_mean.reshape(-1, cfg.macro_dim)

            # Invariance: stability under perturbation
            if cfg.lambda_vicreg_inv > 0:
                z_perturbed = z_macro_flat_for_vicreg + torch.randn_like(z_macro_flat_for_vicreg) * cfg.vicreg_noise_std
                L_vicreg_inv = vicreg_invariance_loss(z_macro_flat_for_vicreg, z_perturbed)
                losses["vicreg_inv"] = L_vicreg_inv
                total = total + new_loss_warmup * cfg.lambda_vicreg_inv * L_vicreg_inv

            # Variance: prevent mode collapse
            if cfg.lambda_vicreg_var > 0:
                L_vicreg_var = vicreg_variance_loss(z_macro_flat_for_vicreg)
                losses["vicreg_var"] = L_vicreg_var
                total = total + new_loss_warmup * cfg.lambda_vicreg_var * L_vicreg_var

            # Covariance: decorrelate dimensions
            if cfg.lambda_vicreg_cov > 0:
                L_vicreg_cov = vicreg_covariance_loss(z_macro_flat_for_vicreg)
                losses["vicreg_cov"] = L_vicreg_cov
                total = total + new_loss_warmup * cfg.lambda_vicreg_cov * L_vicreg_cov

        # Topology losses (only if using charted encoder)
        if self.is_charted and (cfg.lambda_entropy > 0 or cfg.lambda_balance > 0 or cfg.lambda_separation > 0):
            router_weights = latents["router_weights"]
            chart_outputs = latents["chart_outputs"]

            # Entropy: encourage confident chart selection
            if cfg.lambda_entropy > 0:
                L_entropy = router_entropy_loss(router_weights)
                losses["entropy"] = L_entropy
                total = total + new_loss_warmup * cfg.lambda_entropy * L_entropy

            # Balance: encourage equal chart utilization
            if cfg.lambda_balance > 0:
                L_balance = router_balance_loss(router_weights, cfg.num_charts)
                losses["balance"] = L_balance
                total = total + new_loss_warmup * cfg.lambda_balance * L_balance

            # Separation: force chart centers apart
            if cfg.lambda_separation > 0:
                L_separation = chart_separation_loss(chart_outputs, router_weights, cfg.chart_separation_dist)
                losses["separation"] = L_separation
                total = total + new_loss_warmup * cfg.lambda_separation * L_separation

        # BRST loss
        if cfg.lambda_brst > 0 and cfg.use_brst:
            L_brst = self.compute_brst_loss()
            losses["brst"] = L_brst
            total = total + new_loss_warmup * cfg.lambda_brst * L_brst

        losses["total"] = total
        return losses

    def compute_closure_ratio(self, x_seq: torch.Tensor) -> float:
        """Compute closure ratio diagnostic."""
        batch_size, seq_len, H, W = x_seq.shape

        with torch.no_grad():
            x_flat = x_seq.reshape(-1, H, W)
            latents = self.encode(x_flat)
            z_macro = latents["macro_mean"].reshape(batch_size, seq_len, -1)

            z_macro_pred = self.predict_next_macro(z_macro[:, :-1].reshape(-1, self.config.macro_dim))
            z_macro_target = z_macro[:, 1:].reshape(-1, self.config.macro_dim)
            closure_error = F.mse_loss(z_macro_pred, z_macro_target).item()

            baseline_error = F.mse_loss(z_macro[:, :-1], z_macro[:, 1:]).item()

            if baseline_error < 1e-8:
                return 1.0
            return 1.0 - (closure_error / baseline_error)

    def extract_position(self, x: torch.Tensor) -> torch.Tensor:
        """Extract estimated position (first 2 components of z_macro)."""
        with torch.no_grad():
            latents = self.encode(x)
            return latents["macro_mean"][:, :2]


def test_physicist_2d():
    """Test the 2D Physicist agent."""
    print("Testing PhysicistAgent2D...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Test 1: Default config (backward compatibility)
    # =========================================================================
    print("\n  Test 1: Default config (backward compatibility)")
    config = PhysicistConfig2D(device=device)
    agent = PhysicistAgent2D(config)

    # Count parameters
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"    Parameters: {num_params:,}")
    print(f"    is_charted: {agent.is_charted}")

    # Test single observation
    x = torch.randn(4, 64, 64, device=device)
    x_recon, latents = agent(x)
    assert x_recon.shape == (4, 64, 64), f"Expected (4, 64, 64), got {x_recon.shape}"
    assert latents["z_macro"].shape == (4, 4)
    assert latents["z_micro"].shape == (4, 32)
    assert "router_weights" not in latents  # Not charted
    print(f"    Single: x_recon={x_recon.shape}, z_macro={latents['z_macro'].shape}")

    # Test sequence loss
    x_seq = torch.randn(8, 32, 64, 64, device=device)
    losses = agent.compute_loss(x_seq)
    print(f"    Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    # Verify new losses are zero with default config
    assert "vicreg_inv" not in losses
    assert "entropy" not in losses
    assert "brst" not in losses
    print("    Default config: New losses correctly disabled")

    # Test closure ratio
    ratio = agent.compute_closure_ratio(x_seq)
    print(f"    Closure ratio: {ratio:.4f}")

    # Test position extraction
    pos = agent.extract_position(x)
    assert pos.shape == (4, 2)
    print(f"    Position extraction: {pos.shape}")

    # =========================================================================
    # Test 2: Charted encoder with new features
    # =========================================================================
    print("\n  Test 2: Charted encoder with new features")
    config_charted = PhysicistConfig2D(
        device=device,
        num_charts=3,
        use_brst=True,
        lambda_vicreg_inv=1.0,
        lambda_vicreg_var=1.0,
        lambda_vicreg_cov=1.0,
        lambda_entropy=1.0,
        lambda_balance=1.0,
        lambda_separation=1.0,
        lambda_brst=0.1,
    )
    agent_charted = PhysicistAgent2D(config_charted)

    num_params_charted = sum(p.numel() for p in agent_charted.parameters())
    print(f"    Parameters: {num_params_charted:,} (vs {num_params:,} default)")
    print(f"    is_charted: {agent_charted.is_charted}")

    # Test single observation
    x_recon_c, latents_c = agent_charted(x)
    assert latents_c["z_macro"].shape == (4, 4)
    assert "router_weights" in latents_c
    assert latents_c["router_weights"].shape == (4, 3)  # 3 charts
    print(f"    Router weights shape: {latents_c['router_weights'].shape}")

    # Test sequence loss with all new losses
    losses_c = agent_charted.compute_loss(x_seq, epoch=10)
    print(f"    Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses_c.items())}")

    # Verify new losses are present
    assert "vicreg_inv" in losses_c
    assert "vicreg_var" in losses_c
    assert "vicreg_cov" in losses_c
    assert "entropy" in losses_c
    assert "balance" in losses_c
    assert "separation" in losses_c
    assert "brst" in losses_c
    print("    Charted config: All new losses present")

    # Test BRST defect
    brst_defect = agent_charted.compute_brst_loss()
    print(f"    BRST defect: {brst_defect.item():.4f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_physicist_2d()
