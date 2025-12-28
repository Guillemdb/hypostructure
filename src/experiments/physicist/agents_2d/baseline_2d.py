"""
Baseline Agent for 2D environments with CNN encoder.

Standard VAE-LSTM architecture WITHOUT the Physicist split-brain structure.
Uses a single unified latent space with LSTM for temporal modeling.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig2D:
    """Configuration for the 2D Baseline Agent."""

    # Architecture - matched capacity to Physicist
    image_size: int = 64
    hidden_channels: int = 32
    latent_dim: int = 36               # = macro_dim + micro_dim
    rnn_hidden: int = 64

    # Training
    lambda_kl: float = 0.01

    # Device
    device: str = "cuda"


class BaselineCNNEncoder(nn.Module):
    """Standard CNN encoder - no split latent."""

    def __init__(self, config: BaselineConfig2D):
        super().__init__()

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
        self.fc = nn.Linear(self.flat_dim, 256)
        self.mean = nn.Linear(256, config.latent_dim)
        self.logvar = nn.Linear(256, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc(h))
        # Clamp logvar for numerical stability
        logvar = torch.clamp(self.logvar(h), -20, 5)
        return self.mean(h), logvar


class BaselineCNNDecoder(nn.Module):
    """Standard CNN decoder."""

    def __init__(self, config: BaselineConfig2D):
        super().__init__()
        self.config = config

        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * 4 * config.hidden_channels * 4),
            nn.ReLU(),
        )

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self.config.hidden_channels * 4, 4, 4)
        x = self.deconv(h)
        return x.squeeze(1)


class BaselineDynamics(nn.Module):
    """LSTM for temporal modeling."""

    def __init__(self, config: BaselineConfig2D):
        super().__init__()
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
        output, hidden = self.lstm(z_seq, hidden)
        z_pred = self.output(output)
        return z_pred, hidden


class BaselineAgent2D(nn.Module):
    """
    Standard VAE-LSTM baseline for 2D environments.

    No split latent, no causal enclosure - just standard VAE + LSTM.
    """

    def __init__(self, config: Optional[BaselineConfig2D] = None):
        super().__init__()
        self.config = config or BaselineConfig2D()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self.encoder = BaselineCNNEncoder(self.config)
        self.decoder = BaselineCNNDecoder(self.config)
        self.dynamics = BaselineDynamics(self.config)

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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return {"z": z, "mean": mean, "logvar": logvar}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict_next(
        self,
        z_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.dynamics(z_seq, hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latents = self.encode(x)
        x_recon = self.decode(latents["z"])
        return x_recon, latents

    def compute_loss(
        self,
        x_seq: torch.Tensor,
        training: bool = True,
        occlusion_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, H, W = x_seq.shape

        x_flat = x_seq.reshape(-1, H, W)
        latents = self.encode(x_flat)

        z = latents["z"].reshape(batch_size, seq_len, -1)
        mean = latents["mean"].reshape(batch_size, seq_len, -1)
        logvar = latents["logvar"].reshape(batch_size, seq_len, -1)

        # Reconstruction
        x_recon = self.decode(latents["z"])

        if occlusion_mask is not None:
            mask_flat = occlusion_mask.reshape(-1, H, W)
            L_recon = (((x_recon - x_flat) ** 2) * mask_flat).sum() / mask_flat.sum()
        else:
            L_recon = F.mse_loss(x_recon, x_flat)

        # KL
        L_kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Prediction loss
        z_pred, _ = self.predict_next(z[:, :-1])
        z_target = mean[:, 1:].detach()
        L_pred = F.mse_loss(z_pred, z_target)

        total = L_recon + self.config.lambda_kl * L_kl + L_pred

        return {
            "total": total,
            "recon": L_recon,
            "kl": L_kl,
            "pred": L_pred,
        }

    def compute_closure_ratio(self, x_seq: torch.Tensor) -> float:
        batch_size, seq_len, H, W = x_seq.shape

        with torch.no_grad():
            x_flat = x_seq.reshape(-1, H, W)
            latents = self.encode(x_flat)
            z = latents["mean"].reshape(batch_size, seq_len, -1)

            z_pred, _ = self.predict_next(z[:, :-1])
            z_target = z[:, 1:]
            pred_error = F.mse_loss(z_pred, z_target).item()

            baseline_error = F.mse_loss(z[:, :-1], z[:, 1:]).item()

            if baseline_error < 1e-8:
                return 1.0
            return 1.0 - (pred_error / baseline_error)

    def extract_position(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.encode(x)
            return latents["mean"][:, :2]


def test_baseline_2d():
    """Test the 2D Baseline agent."""
    print("Testing BaselineAgent2D...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = BaselineConfig2D(device=device)
    agent = BaselineAgent2D(config)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"  Parameters: {num_params:,}")

    x = torch.randn(4, 64, 64, device=device)
    x_recon, latents = agent(x)
    assert x_recon.shape == (4, 64, 64)
    assert latents["z"].shape == (4, 36)
    print(f"  Single: x_recon={x_recon.shape}, z={latents['z'].shape}")

    x_seq = torch.randn(8, 32, 64, 64, device=device)
    losses = agent.compute_loss(x_seq)
    print(f"  Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    ratio = agent.compute_closure_ratio(x_seq)
    print(f"  Closure ratio: {ratio:.4f}")

    pos = agent.extract_position(x)
    assert pos.shape == (4, 2)
    print(f"  Position extraction: {pos.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_baseline_2d()
