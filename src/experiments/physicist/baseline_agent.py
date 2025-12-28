"""
Baseline Agent for 1D Noisy Bouncing Pulse.

Standard VAE-LSTM architecture WITHOUT the Physicist split-brain structure.
This serves as a control to demonstrate that the Physicist architecture
provides genuine benefits for physics extraction.

Key differences from Physicist:
- Single unified latent z (no macro/micro split)
- LSTM for temporal modeling (no causal enclosure constraint)
- Standard VAE loss (no slowness or closure losses)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig1D:
    """Configuration for the 1D Baseline Agent."""

    # Architecture - matched capacity to Physicist
    obs_dim: int = 64                  # Input width
    hidden_dim: int = 64               # Hidden layer size
    latent_dim: int = 18               # Total latent (= macro_dim + micro_dim)
    rnn_hidden: int = 32               # LSTM hidden size

    # Training
    lambda_kl: float = 0.01            # KL weight

    # Device
    device: str = "cuda"


class BaselineEncoder(nn.Module):
    """Standard VAE encoder - no split latent."""

    def __init__(self, config: BaselineConfig1D):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        h = self.encoder(x)
        return self.mean(h), self.logvar(h)


class BaselineDecoder(nn.Module):
    """Standard VAE decoder."""

    def __init__(self, config: BaselineConfig1D):
        super().__init__()
        self.config = config

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(z)


class BaselineDynamics(nn.Module):
    """LSTM for temporal modeling - sees full latent."""

    def __init__(self, config: BaselineConfig1D):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.latent_dim,
            hidden_size=config.rnn_hidden,
            batch_first=True,
        )
        self.output = nn.Linear(config.rnn_hidden, config.latent_dim)

    def forward(
        self,
        z_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next latents from sequence.

        Args:
            z_seq: Latent sequence [batch, seq_len, latent_dim]
            hidden: Optional LSTM hidden state

        Returns:
            z_pred: Predicted next latents [batch, seq_len, latent_dim]
            hidden: Updated LSTM hidden state
        """
        output, hidden = self.lstm(z_seq, hidden)
        z_pred = self.output(output)
        return z_pred, hidden


class BaselineAgent1D(nn.Module):
    """
    Standard VAE-LSTM baseline for comparison.

    Architecture:
        Encoder: x → z (single latent)
        LSTM: z_seq → z_pred_seq
        Decoder: z → x_recon

    Training Losses:
        L_recon: ||x - decoder(z)||^2
        L_kl: KL(q(z|x) || N(0,I))
        L_pred: ||z_{t+1} - lstm(z_t)||^2 (optional)
    """

    def __init__(self, config: Optional[BaselineConfig1D] = None):
        super().__init__()
        self.config = config or BaselineConfig1D()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self.encoder = BaselineEncoder(self.config)
        self.decoder = BaselineDecoder(self.config)
        self.dynamics = BaselineDynamics(self.config)

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
            Dictionary with z and parameters
        """
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)

        return {
            "z": z,
            "mean": mean,
            "logvar": logvar,
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(z)

    def predict_next(
        self,
        z_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict next latent using LSTM."""
        return self.dynamics(z_seq, hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode → decode.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            x_recon: Reconstruction [batch, obs_dim]
            latents: Dictionary of latent variables
        """
        latents = self.encode(x)
        x_recon = self.decode(latents["z"])
        return x_recon, latents

    def compute_loss(
        self,
        x_seq: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a sequence.

        Args:
            x_seq: Observation sequence [batch, seq_len, obs_dim]
            training: Whether in training mode

        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, obs_dim = x_seq.shape

        # Flatten for encoding
        x_flat = x_seq.reshape(-1, obs_dim)

        # Encode all timesteps
        latents = self.encode(x_flat)

        # Reshape back to sequence
        z = latents["z"].reshape(batch_size, seq_len, -1)
        mean = latents["mean"].reshape(batch_size, seq_len, -1)
        logvar = latents["logvar"].reshape(batch_size, seq_len, -1)

        # Reconstruction loss
        x_recon = self.decode(latents["z"])
        L_recon = F.mse_loss(x_recon, x_flat)

        # KL divergence
        L_kl = -0.5 * torch.mean(
            1 + logvar - mean.pow(2) - logvar.exp()
        )

        # Prediction loss (LSTM predicts next z)
        z_pred, _ = self.predict_next(z[:, :-1])
        z_target = mean[:, 1:].detach()  # Use mean as target
        L_pred = F.mse_loss(z_pred, z_target)

        # Total loss
        total_loss = L_recon + self.config.lambda_kl * L_kl + L_pred

        return {
            "total": total_loss,
            "recon": L_recon,
            "kl": L_kl,
            "pred": L_pred,
        }

    def compute_closure_ratio(
        self,
        x_seq: torch.Tensor
    ) -> float:
        """
        Compute closure ratio equivalent for baseline.

        For baseline, we measure how well LSTM predicts the full latent.
        """
        batch_size, seq_len, obs_dim = x_seq.shape

        with torch.no_grad():
            # Encode
            x_flat = x_seq.reshape(-1, obs_dim)
            latents = self.encode(x_flat)
            z = latents["mean"].reshape(batch_size, seq_len, -1)

            # LSTM prediction error
            z_pred, _ = self.predict_next(z[:, :-1])
            z_target = z[:, 1:]
            pred_error = F.mse_loss(z_pred, z_target).item()

            # Baseline: predict with constant (just use previous)
            baseline_error = F.mse_loss(z[:, :-1], z[:, 1:]).item()

            if baseline_error < 1e-8:
                return 1.0

            return 1.0 - (pred_error / baseline_error)

    def extract_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract estimated position from observation.

        For baseline, we use z[0] and hope it correlates with position.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            position: Estimated position [batch]
        """
        with torch.no_grad():
            latents = self.encode(x)
            return latents["mean"][:, 0]


def test_baseline_agent():
    """Test the Baseline agent."""
    print("Testing BaselineAgent1D...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = BaselineConfig1D(device=device)
    agent = BaselineAgent1D(config)

    # Test single observation
    x = torch.randn(4, 64, device=device)
    x_recon, latents = agent(x)
    assert x_recon.shape == (4, 64), f"Expected (4, 64), got {x_recon.shape}"
    assert latents["z"].shape == (4, 18)
    print(f"  Single: x_recon={x_recon.shape}, z={latents['z'].shape}")

    # Test sequence loss
    x_seq = torch.randn(8, 32, 64, device=device)
    losses = agent.compute_loss(x_seq)
    print(f"  Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    # Test closure ratio
    ratio = agent.compute_closure_ratio(x_seq)
    print(f"  Closure ratio: {ratio:.4f}")

    # Test position extraction
    pos = agent.extract_position(x)
    assert pos.shape == (4,)
    print(f"  Position extraction: {pos.shape}")

    # Compare parameter counts
    physicist_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {physicist_params:,}")

    print("All tests passed!")


if __name__ == "__main__":
    test_baseline_agent()
