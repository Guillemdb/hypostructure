"""
Physicist Agent for 1D Noisy Bouncing Pulse.

Split-Brain VAE-RNN architecture that separates:
- z_macro (dim=2): Position and velocity - the "physics"
- z_micro (dim=16): Noise encoding - the "entropy"

Key architectural features:
1. Causal Enclosure: Physics engine only sees z_macro
2. Information Dropout: Randomly zero z_micro during training
3. Slowness Constraint: z_macro changes slowly across timesteps
4. Multi-Chart Routing: Multiple encoder experts with soft selection (optional)
5. BRST Constraints: Near-orthogonal weights for geometric preservation (optional)
6. VICReg Losses: Variance-Invariance-Covariance regularization (optional)

See fragile-index.md Section 9 for theoretical background.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BRSTLinear, compute_total_brst_defect
from .losses import (
    vicreg_invariance_loss,
    vicreg_variance_loss,
    vicreg_covariance_loss,
    router_entropy_loss,
    router_balance_loss,
    chart_separation_loss,
    compute_warmup_ratio,
)


@dataclass
class PhysicistConfig1D:
    """Configuration for the 1D Physicist Agent."""

    # Architecture
    obs_dim: int = 64                  # Input width
    hidden_dim: int = 64               # Hidden layer size
    macro_dim: int = 2                 # z_macro dimension (position, velocity)
    micro_dim: int = 16                # z_micro dimension (noise encoding)
    rnn_hidden: int = 32               # Physics engine RNN hidden size

    # Training - original losses
    info_dropout_prob: float = 0.5     # Probability of dropping z_micro
    lambda_closure: float = 1.0        # Causal enclosure loss weight
    lambda_slowness: float = 0.1       # Temporal smoothness weight
    lambda_dispersion: float = 0.01    # KL weight on z_micro
    lambda_kl_macro: float = 0.001     # KL weight on z_macro (light)

    # Multi-chart routing (1 = original behavior)
    num_charts: int = 1
    use_brst: bool = False

    # VICReg loss weights (0 = disabled)
    lambda_vicreg_inv: float = 0.0
    lambda_vicreg_var: float = 0.0
    lambda_vicreg_cov: float = 0.0
    vicreg_noise_std: float = 0.05

    # Topology loss weights (0 = disabled)
    lambda_entropy: float = 0.0
    lambda_balance: float = 0.0
    lambda_separation: float = 0.0
    chart_separation_dist: float = 4.0

    # BRST loss weight (0 = disabled)
    lambda_brst: float = 0.0

    # Warmup for new losses
    new_loss_warmup_epochs: int = 20

    # Device
    device: str = "cuda"


class PhysicistEncoder(nn.Module):
    """
    Encoder that produces split latent: z_macro + z_micro.

    The encoder learns to separate physics-relevant features (macro)
    from noise features (micro) via the causal enclosure loss.
    """

    def __init__(self, config: PhysicistConfig1D):
        super().__init__()
        self.config = config

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # z_macro branch (position, velocity)
        self.macro_mean = nn.Linear(config.hidden_dim, config.macro_dim)
        self.macro_logvar = nn.Linear(config.hidden_dim, config.macro_dim)

        # z_micro branch (noise encoding)
        self.micro_mean = nn.Linear(config.hidden_dim, config.micro_dim)
        self.micro_logvar = nn.Linear(config.hidden_dim, config.micro_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to split latent.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            macro_mean: [batch, macro_dim]
            macro_logvar: [batch, macro_dim]
            micro_mean: [batch, micro_dim]
            micro_logvar: [batch, micro_dim]
        """
        h = self.shared(x)

        macro_mean = self.macro_mean(h)
        macro_logvar = self.macro_logvar(h)

        micro_mean = self.micro_mean(h)
        micro_logvar = self.micro_logvar(h)

        return macro_mean, macro_logvar, micro_mean, micro_logvar


class ChartedPhysicistEncoder(nn.Module):
    """
    Multi-chart encoder with BRST constraints and router.

    Routes input to multiple expert charts (macro encoders) using a soft router.
    Each chart specializes on different regions of the input space.
    Uses BRST layers for near-orthogonal weight constraints.
    """

    def __init__(self, config: PhysicistConfig1D):
        super().__init__()
        self.config = config
        self.num_charts = config.num_charts

        # Router: input → chart weights
        self.router = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_charts),
            nn.Softmax(dim=1)
        )

        # Chart experts for macro encoding (using BRST if enabled)
        LinearLayer = BRSTLinear if config.use_brst else nn.Linear
        self.macro_charts = nn.ModuleList()
        for _ in range(config.num_charts):
            chart = nn.Sequential(
                LinearLayer(config.obs_dim, config.hidden_dim),
                nn.ReLU(),
                LinearLayer(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.macro_charts.append(chart)

        # Per-chart macro heads
        self.macro_mean_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.macro_dim)
            for _ in range(config.num_charts)
        ])
        self.macro_logvar_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.macro_dim)
            for _ in range(config.num_charts)
        ])

        # Shared micro encoder (no charts needed - noise is uniform)
        self.micro_shared = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.micro_mean = nn.Linear(config.hidden_dim, config.micro_dim)
        self.micro_logvar = nn.Linear(config.hidden_dim, config.micro_dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, List[torch.Tensor]]:
        """
        Encode observation using multi-chart routing.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            macro_mean: Weighted macro mean [batch, macro_dim]
            macro_logvar: Weighted macro logvar [batch, macro_dim]
            micro_mean: Micro mean [batch, micro_dim]
            micro_logvar: Micro logvar [batch, micro_dim]
            chart_weights: Router weights [batch, num_charts]
            chart_outputs: List of per-chart macro means for separation loss
        """
        batch_size = x.shape[0]

        # Get router weights
        chart_weights = self.router(x)  # [batch, num_charts]

        # Compute per-chart outputs and weighted sum
        macro_mean = torch.zeros(batch_size, self.config.macro_dim, device=x.device)
        macro_logvar = torch.zeros(batch_size, self.config.macro_dim, device=x.device)
        chart_outputs = []

        for i in range(self.num_charts):
            h_i = self.macro_charts[i](x)
            mean_i = self.macro_mean_heads[i](h_i)
            logvar_i = self.macro_logvar_heads[i](h_i)
            chart_outputs.append(mean_i)

            # Weighted contribution
            w_i = chart_weights[:, i:i+1]  # [batch, 1]
            macro_mean = macro_mean + w_i * mean_i
            macro_logvar = macro_logvar + w_i * logvar_i

        # Micro encoding (shared, no charts)
        h_micro = self.micro_shared(x)
        micro_mean = self.micro_mean(h_micro)
        micro_logvar = self.micro_logvar(h_micro)

        return macro_mean, macro_logvar, micro_mean, micro_logvar, chart_weights, chart_outputs


class PhysicsEngine(nn.Module):
    """
    Physics engine that predicts z_macro_{t+1} from z_macro_t.

    CRITICAL: This module is BLIND to z_micro.
    The causal enclosure loss ensures this is sufficient for prediction.

    Optionally uses BRST constraints for near-volume-preserving dynamics
    (related to Liouville's theorem for Hamiltonian systems).
    """

    def __init__(self, config: PhysicistConfig1D):
        super().__init__()
        self.config = config

        # Simple MLP for deterministic physics prediction
        # For bouncing ball, this should learn: pos' = pos + vel, vel' = bounce(vel)
        LinearLayer = BRSTLinear if config.use_brst else nn.Linear
        self.dynamics = nn.Sequential(
            LinearLayer(config.macro_dim, config.rnn_hidden),
            nn.ReLU(),
            LinearLayer(config.rnn_hidden, config.rnn_hidden),
            nn.ReLU(),
            LinearLayer(config.rnn_hidden, config.macro_dim),
        )

    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        """
        Predict next z_macro.

        Args:
            z_macro: Current macro latent [batch, macro_dim]

        Returns:
            z_macro_pred: Predicted next macro latent [batch, macro_dim]
        """
        return self.dynamics(z_macro)


class PhysicistDecoder(nn.Module):
    """
    Decoder that reconstructs observation from z_macro + z_micro.

    During training, z_micro may be dropped (information dropout),
    forcing the decoder to rely on z_macro for structure.
    """

    def __init__(self, config: PhysicistConfig1D):
        super().__init__()
        self.config = config

        total_latent = config.macro_dim + config.micro_dim

        self.decoder = nn.Sequential(
            nn.Linear(total_latent, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.obs_dim),
        )

    def forward(
        self,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        drop_micro: bool = False
    ) -> torch.Tensor:
        """
        Decode to observation.

        Args:
            z_macro: Macro latent [batch, macro_dim]
            z_micro: Micro latent [batch, micro_dim]
            drop_micro: If True, zero out z_micro (information dropout)

        Returns:
            x_recon: Reconstructed observation [batch, obs_dim]
        """
        if drop_micro:
            z_micro = torch.zeros_like(z_micro)

        z = torch.cat([z_macro, z_micro], dim=-1)
        return self.decoder(z)


class PhysicistAgent1D(nn.Module):
    """
    Complete Physicist Agent for 1D Noisy Bouncing Pulse.

    Architecture:
        Encoder: x → (z_macro, z_micro) [optionally with multi-chart routing]
        Physics: z_macro_t → z_macro_{t+1} (blind to z_micro)
        Decoder: (z_macro, z_micro) → x_recon

    Training Losses (Original):
        L_recon: ||x - decoder(z)||^2
        L_closure: ||z_macro_{t+1,enc} - physics(z_macro_t)||^2
        L_slowness: ||z_macro_t - z_macro_{t-1}||^2
        L_dispersion: KL(q(z_micro) || N(0,I))

    Training Losses (New - optional):
        L_brst: BRST orthogonality defect
        L_vicreg: Variance-Invariance-Covariance regularization
        L_topology: Router entropy, balance, and chart separation
    """

    def __init__(self, config: Optional[PhysicistConfig1D] = None):
        super().__init__()
        self.config = config or PhysicistConfig1D()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        # Use charted encoder if num_charts > 1
        self.use_charted = self.config.num_charts > 1
        if self.use_charted:
            self.encoder = ChartedPhysicistEncoder(self.config)
        else:
            self.encoder = PhysicistEncoder(self.config)

        self.physics = PhysicsEngine(self.config)
        self.decoder = PhysicistDecoder(self.config)

        self.to(self.device)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode observation to latent.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            Dictionary with z_macro, z_micro, their parameters,
            and optionally chart_weights and chart_outputs
        """
        if self.use_charted:
            macro_mean, macro_logvar, micro_mean, micro_logvar, chart_weights, chart_outputs = self.encoder(x)
        else:
            macro_mean, macro_logvar, micro_mean, micro_logvar = self.encoder(x)
            chart_weights = None
            chart_outputs = None

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

        if chart_weights is not None:
            result["chart_weights"] = chart_weights
            result["chart_outputs"] = chart_outputs

        return result

    def compute_brst_loss(self) -> torch.Tensor:
        """
        Compute total BRST defect across encoder and physics engine.

        Returns:
            Scalar BRST loss (0 if BRST not enabled)
        """
        if not self.config.use_brst:
            return torch.tensor(0.0, device=self.device)

        total = compute_total_brst_defect(self.encoder)
        total = total + compute_total_brst_defect(self.physics)
        return total

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
        """
        Full forward pass: encode → decode.

        Args:
            x: Observation [batch, obs_dim]
            drop_micro: Whether to drop z_micro in decoder

        Returns:
            x_recon: Reconstruction [batch, obs_dim]
            latents: Dictionary of latent variables
        """
        latents = self.encode(x)
        x_recon = self.decode(latents["z_macro"], latents["z_micro"], drop_micro)
        return x_recon, latents

    def compute_loss(
        self,
        x_seq: torch.Tensor,
        training: bool = True,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a sequence.

        Args:
            x_seq: Observation sequence [batch, seq_len, obs_dim]
            training: Whether in training mode (enables dropout)
            epoch: Current training epoch (for warmup of new losses)

        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, obs_dim = x_seq.shape
        device = x_seq.device

        # Compute warmup ratio for new losses
        new_loss_warmup = compute_warmup_ratio(epoch, self.config.new_loss_warmup_epochs)

        # Flatten for encoding
        x_flat = x_seq.reshape(-1, obs_dim)

        # Encode all timesteps
        latents = self.encode(x_flat)

        # Reshape back to sequence
        z_macro = latents["z_macro"].reshape(batch_size, seq_len, -1)
        z_micro = latents["z_micro"].reshape(batch_size, seq_len, -1)
        macro_mean = latents["macro_mean"].reshape(batch_size, seq_len, -1)
        macro_logvar = latents["macro_logvar"].reshape(batch_size, seq_len, -1)
        micro_mean = latents["micro_mean"].reshape(batch_size, seq_len, -1)
        micro_logvar = latents["micro_logvar"].reshape(batch_size, seq_len, -1)

        # Get chart weights/outputs if using charted encoder
        chart_weights = latents.get("chart_weights")
        chart_outputs = latents.get("chart_outputs")

        # Information dropout during training
        drop_micro = training and torch.rand(1).item() < self.config.info_dropout_prob

        # Reconstruction loss
        z_macro_flat = z_macro.reshape(-1, self.config.macro_dim)
        z_micro_flat = z_micro.reshape(-1, self.config.micro_dim)
        x_recon = self.decode(z_macro_flat, z_micro_flat, drop_micro)
        L_recon = F.mse_loss(x_recon, x_flat)

        # Causal enclosure loss: physics(z_macro_t) should predict z_macro_{t+1}
        z_macro_pred = self.predict_next_macro(z_macro[:, :-1].reshape(-1, self.config.macro_dim))
        z_macro_target = macro_mean[:, 1:].reshape(-1, self.config.macro_dim).detach()
        L_closure = F.mse_loss(z_macro_pred, z_macro_target)

        # Slowness loss: z_macro should change slowly
        L_slowness = F.mse_loss(z_macro[:, 1:], z_macro[:, :-1])

        # KL divergence for z_micro (push toward N(0,I))
        L_kl_micro = -0.5 * torch.mean(
            1 + micro_logvar - micro_mean.pow(2) - micro_logvar.exp()
        )

        # Light KL on z_macro (don't over-constrain physics)
        L_kl_macro = -0.5 * torch.mean(
            1 + macro_logvar - macro_mean.pow(2) - macro_logvar.exp()
        )

        # --- NEW LOSSES (with warmup) ---

        # BRST loss (orthogonality constraint)
        L_brst = self.compute_brst_loss()

        # VICReg losses on z_macro (only during training with warmup)
        if training and new_loss_warmup > 0 and (
            self.config.lambda_vicreg_inv > 0 or
            self.config.lambda_vicreg_var > 0 or
            self.config.lambda_vicreg_cov > 0
        ):
            # Perturb input for invariance loss
            x_perturbed = x_flat + torch.randn_like(x_flat) * self.config.vicreg_noise_std
            latents_perturbed = self.encode(x_perturbed)
            z_macro_perturbed = latents_perturbed["z_macro"]

            L_vicreg_inv = vicreg_invariance_loss(z_macro_flat, z_macro_perturbed)
            L_vicreg_var = vicreg_variance_loss(z_macro_flat)
            L_vicreg_cov = vicreg_covariance_loss(z_macro_flat)
        else:
            L_vicreg_inv = torch.tensor(0.0, device=device)
            L_vicreg_var = torch.tensor(0.0, device=device)
            L_vicreg_cov = torch.tensor(0.0, device=device)

        # Topology losses (only if using charted encoder)
        if chart_weights is not None and chart_outputs is not None:
            L_entropy = router_entropy_loss(chart_weights)
            L_balance = router_balance_loss(chart_weights, self.config.num_charts)
            L_separation = chart_separation_loss(
                chart_outputs, chart_weights, self.config.chart_separation_dist
            )
        else:
            L_entropy = torch.tensor(0.0, device=device)
            L_balance = torch.tensor(0.0, device=device)
            L_separation = torch.tensor(0.0, device=device)

        # Total loss with new losses scaled by warmup
        total_loss = (
            # Original losses
            L_recon +
            self.config.lambda_closure * L_closure +
            self.config.lambda_slowness * L_slowness +
            self.config.lambda_dispersion * L_kl_micro +
            self.config.lambda_kl_macro * L_kl_macro +
            # New losses with warmup
            new_loss_warmup * (
                self.config.lambda_brst * L_brst +
                self.config.lambda_vicreg_inv * L_vicreg_inv +
                self.config.lambda_vicreg_var * L_vicreg_var +
                self.config.lambda_vicreg_cov * L_vicreg_cov +
                self.config.lambda_entropy * L_entropy +
                self.config.lambda_balance * L_balance +
                self.config.lambda_separation * L_separation
            )
        )

        return {
            "total": total_loss,
            "recon": L_recon,
            "closure": L_closure,
            "slowness": L_slowness,
            "kl_micro": L_kl_micro,
            "kl_macro": L_kl_macro,
            # New losses
            "brst": L_brst,
            "vicreg_inv": L_vicreg_inv,
            "vicreg_var": L_vicreg_var,
            "vicreg_cov": L_vicreg_cov,
            "entropy": L_entropy,
            "balance": L_balance,
            "separation": L_separation,
        }

    def compute_closure_ratio(
        self,
        x_seq: torch.Tensor
    ) -> float:
        """
        Compute the closure ratio diagnostic.

        Closure Ratio = 1 - (closure_loss / baseline_loss)

        Where baseline_loss is what you'd get predicting z_macro_{t+1}
        from z_macro_t using just the mean (i.e., no prediction).

        A ratio close to 1 means physics explains the variance well.
        A ratio close to 0 means physics doesn't help.
        """
        batch_size, seq_len, obs_dim = x_seq.shape

        with torch.no_grad():
            # Encode
            x_flat = x_seq.reshape(-1, obs_dim)
            latents = self.encode(x_flat)
            z_macro = latents["macro_mean"].reshape(batch_size, seq_len, -1)

            # Physics prediction error
            z_macro_pred = self.predict_next_macro(z_macro[:, :-1].reshape(-1, self.config.macro_dim))
            z_macro_target = z_macro[:, 1:].reshape(-1, self.config.macro_dim)
            closure_error = F.mse_loss(z_macro_pred, z_macro_target).item()

            # Baseline: predict with running mean (essentially zero delta)
            baseline_error = F.mse_loss(z_macro[:, :-1], z_macro[:, 1:]).item()

            # Avoid division by zero
            if baseline_error < 1e-8:
                return 1.0

            return 1.0 - (closure_error / baseline_error)

    def extract_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract estimated position from observation.

        Uses z_macro[0] as position estimate (should correlate with true pos).

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            position: Estimated position [batch]
        """
        with torch.no_grad():
            latents = self.encode(x)
            # First component of z_macro should encode position
            return latents["macro_mean"][:, 0]


def test_physicist_agent():
    """Test the Physicist agent."""
    print("Testing PhysicistAgent1D...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PhysicistConfig1D(device=device)
    agent = PhysicistAgent1D(config)

    # Test single observation
    x = torch.randn(4, 64, device=device)
    x_recon, latents = agent(x)
    assert x_recon.shape == (4, 64), f"Expected (4, 64), got {x_recon.shape}"
    assert latents["z_macro"].shape == (4, 2)
    assert latents["z_micro"].shape == (4, 16)
    print(f"  Single: x_recon={x_recon.shape}, z_macro={latents['z_macro'].shape}")

    # Test sequence loss
    x_seq = torch.randn(8, 32, 64, device=device)  # [batch, seq_len, obs_dim]
    losses = agent.compute_loss(x_seq)
    print(f"  Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    # Test closure ratio
    ratio = agent.compute_closure_ratio(x_seq)
    print(f"  Closure ratio: {ratio:.4f}")

    # Test position extraction
    pos = agent.extract_position(x)
    assert pos.shape == (4,)
    print(f"  Position extraction: {pos.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_physicist_agent()
