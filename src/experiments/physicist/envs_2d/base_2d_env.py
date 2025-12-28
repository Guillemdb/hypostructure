"""
Base class for 2D Physicist environments.

Provides common functionality:
- 2D bouncing ball physics
- Noise generation
- Trajectory generation
- GPU acceleration via PyTorch
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


@dataclass
class Base2DConfig:
    """Base configuration for 2D environments."""

    size: int = 64                     # Image size (size x size)
    ball_radius: float = 3.0           # Ball radius in pixels
    speed: float = 2.0                 # Ball speed (pixels per step)
    noise_sigma: float = 0.2           # Noise standard deviation
    signal_intensity: float = 1.0      # Ball brightness (1.0 = white)
    device: str = "cuda"               # Device to use


class Base2DEnv(ABC):
    """
    Base class for 2D bouncing ball environments.

    Provides:
    - 2D physics simulation (bouncing ball)
    - Noise generation
    - Batch trajectory generation
    - GPU acceleration

    Subclasses implement specific modifications (occlusion, low SNR, etc.)
    """

    def __init__(self, config: Base2DConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # State: position [batch, 2], velocity [batch, 2]
        self.pos: Optional[torch.Tensor] = None
        self.vel: Optional[torch.Tensor] = None
        self.batch_size: int = 0

        # Precompute coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(config.size, device=self.device, dtype=torch.float32),
            torch.arange(config.size, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self.x_coords = x_coords  # [size, size]
        self.y_coords = y_coords  # [size, size]

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """
        Reset the environment.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initial observation [batch_size, size, size]
        """
        self.batch_size = batch_size
        r = self.config.ball_radius

        # Random initial position (keep ball inside boundaries)
        margin = r + 1
        self.pos = torch.rand(batch_size, 2, device=self.device) * \
                   (self.config.size - 2 * margin) + margin

        # Random initial velocity direction
        angle = torch.rand(batch_size, device=self.device) * 2 * torch.pi
        self.vel = torch.stack([
            torch.cos(angle) * self.config.speed,
            torch.sin(angle) * self.config.speed
        ], dim=1)

        return self._render()

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take one environment step.

        Returns:
            observation: [batch_size, size, size]
            position: [batch_size, 2]
            velocity: [batch_size, 2]
        """
        assert self.pos is not None, "Must call reset() first"

        # Update position
        self.pos = self.pos + self.vel

        # Bounce off boundaries
        r = self.config.ball_radius
        size = self.config.size

        # X boundaries
        left_bounce = self.pos[:, 0] < r
        right_bounce = self.pos[:, 0] > size - r
        self.pos[:, 0] = torch.where(left_bounce, 2 * r - self.pos[:, 0], self.pos[:, 0])
        self.pos[:, 0] = torch.where(right_bounce, 2 * (size - r) - self.pos[:, 0], self.pos[:, 0])
        self.vel[:, 0] = torch.where(left_bounce | right_bounce, -self.vel[:, 0], self.vel[:, 0])

        # Y boundaries
        top_bounce = self.pos[:, 1] < r
        bottom_bounce = self.pos[:, 1] > size - r
        self.pos[:, 1] = torch.where(top_bounce, 2 * r - self.pos[:, 1], self.pos[:, 1])
        self.pos[:, 1] = torch.where(bottom_bounce, 2 * (size - r) - self.pos[:, 1], self.pos[:, 1])
        self.vel[:, 1] = torch.where(top_bounce | bottom_bounce, -self.vel[:, 1], self.vel[:, 1])

        return self._render(), self.pos.clone(), self.vel.clone()

    def _render_ball(self) -> torch.Tensor:
        """
        Render the ball as a 2D image.

        Returns:
            clean_frame: [batch_size, size, size]
        """
        # Distance from each pixel to ball center
        # pos: [batch, 2], coords: [size, size]
        dx = self.x_coords.unsqueeze(0) - self.pos[:, 0:1, None]  # [batch, size, size]
        dy = self.y_coords.unsqueeze(0) - self.pos[:, 1:2, None]
        dist = torch.sqrt(dx ** 2 + dy ** 2)

        # Ball mask (hard circle)
        ball = (dist <= self.config.ball_radius).float() * self.config.signal_intensity

        return ball

    @abstractmethod
    def _render(self) -> torch.Tensor:
        """
        Render the current state as an observation.

        Subclasses implement this to add occlusion, noise, etc.

        Returns:
            observation: [batch_size, size, size]
        """
        pass

    def render_clean(self) -> torch.Tensor:
        """Render without noise (for visualization)."""
        return self._render_ball()

    def get_ground_truth(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ground truth state."""
        return self.pos.clone(), self.vel.clone()

    def generate_trajectory(
        self,
        seq_len: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a trajectory of observations.

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            observations: [seq_len, batch_size, size, size]
            positions: [seq_len, batch_size, 2]
            velocities: [seq_len, batch_size, 2]
        """
        obs_list = []
        pos_list = []
        vel_list = []

        obs = self.reset(batch_size)
        obs_list.append(obs)
        pos_list.append(self.pos.clone())
        vel_list.append(self.vel.clone())

        for _ in range(seq_len - 1):
            obs, pos, vel = self.step()
            obs_list.append(obs)
            pos_list.append(pos)
            vel_list.append(vel)

        return (
            torch.stack(obs_list, dim=0),
            torch.stack(pos_list, dim=0),
            torch.stack(vel_list, dim=0)
        )

    def generate_batch(
        self,
        num_trajectories: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple trajectories.

        Returns:
            observations: [num_trajectories, seq_len, size, size]
            positions: [num_trajectories, seq_len, 2]
            velocities: [num_trajectories, seq_len, 2]
        """
        obs, pos, vel = self.generate_trajectory(seq_len, num_trajectories)
        return (
            obs.transpose(0, 1),
            pos.transpose(0, 1),
            vel.transpose(0, 1)
        )
