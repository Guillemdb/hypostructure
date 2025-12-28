"""
NoisyTunnelEnv: 2D Bouncing Ball with Occlusion.

The "Tunnel" Test for Object Permanence.

A ball bounces in a 2D box, but a vertical "tunnel" (blind spot) in the
center of the screen occludes the ball when it passes through.

The Challenge:
- When the ball enters the tunnel, optimal reconstruction predicts zeros
- The recurrent state has no gradient signal during occlusion
- Standard models "forget" the ball and are surprised at re-emergence

The Physicist Requirement:
- z_macro must conserve momentum even when ∇L_recon = 0
- The physics engine must "hallucinate" the ball's trajectory
- This proves separation of Representation (pixels) from Dynamics (concept)

If the agent predicts the ball exiting at the correct position/time,
you have achieved OBJECT PERMANENCE.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from .base_2d_env import Base2DEnv, Base2DConfig


@dataclass
class TunnelConfig(Base2DConfig):
    """Configuration for the Tunnel environment."""

    # Tunnel parameters (vertical bar)
    tunnel_left: int = 24              # Left edge of tunnel
    tunnel_right: int = 40             # Right edge of tunnel (exclusive)

    # Inherited defaults
    size: int = 64
    ball_radius: float = 3.0
    speed: float = 2.0
    noise_sigma: float = 0.2
    signal_intensity: float = 1.0
    device: str = "cuda"


class NoisyTunnelEnv(Base2DEnv):
    """
    2D Bouncing Ball with Tunnel Occlusion.

    The ball bounces in a 64x64 box. A vertical tunnel (blind spot)
    in the center occludes the ball when it passes through.

    Physics:
    - Ball bounces off walls (conserves momentum)
    - Tunnel is purely visual - ball physics continues unchanged

    Observation:
    - Ball rendered as bright circle
    - Tunnel region (pixels tunnel_left:tunnel_right) always zero
    - Gaussian noise added everywhere

    This tests:
    - Object Permanence: Can track ball through occlusion
    - Causal Enclosure: Physics continues when observation is blank
    - Momentum Conservation: Exit position/velocity matches entry

    Expected behavior:
    - Baseline: Forgets ball in tunnel, wrong exit prediction
    - Physicist: Maintains physics simulation, correct exit prediction
    """

    def __init__(self, config: Optional[TunnelConfig] = None):
        self.tunnel_config = config or TunnelConfig()
        super().__init__(self.tunnel_config)

        # Create occlusion mask (1 = visible, 0 = occluded)
        self.occlusion_mask = torch.ones(
            self.tunnel_config.size,
            self.tunnel_config.size,
            device=self.device
        )
        self.occlusion_mask[:, self.tunnel_config.tunnel_left:self.tunnel_config.tunnel_right] = 0.0

    def _render(self) -> torch.Tensor:
        """
        Render with tunnel occlusion and noise.

        Returns:
            observation: [batch_size, size, size]
        """
        # Render clean ball
        ball = self._render_ball()

        # Apply occlusion
        ball = ball * self.occlusion_mask.unsqueeze(0)

        # Add noise
        noise = torch.randn_like(ball) * self.config.noise_sigma
        obs = ball + noise

        # Apply occlusion to noise too (tunnel is completely black)
        obs = obs * self.occlusion_mask.unsqueeze(0)

        return obs

    def render_clean(self) -> torch.Tensor:
        """Render without noise but with occlusion."""
        ball = self._render_ball()
        return ball * self.occlusion_mask.unsqueeze(0)

    def render_ground_truth(self) -> torch.Tensor:
        """Render without noise AND without occlusion (true state)."""
        return self._render_ball()

    def is_in_tunnel(self) -> torch.Tensor:
        """
        Check if ball is currently in the tunnel.

        Returns:
            Boolean tensor [batch_size]
        """
        return (
            (self.pos[:, 0] >= self.tunnel_config.tunnel_left) &
            (self.pos[:, 0] < self.tunnel_config.tunnel_right)
        )

    def generate_trajectory(
        self,
        seq_len: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate trajectory with tunnel tracking.

        Returns:
            observations: [seq_len, batch_size, size, size]
            positions: [seq_len, batch_size, 2]
            velocities: [seq_len, batch_size, 2]
            in_tunnel: [seq_len, batch_size]
        """
        obs_list = []
        pos_list = []
        vel_list = []
        tunnel_list = []

        obs = self.reset(batch_size)
        obs_list.append(obs)
        pos_list.append(self.pos.clone())
        vel_list.append(self.vel.clone())
        tunnel_list.append(self.is_in_tunnel())

        for _ in range(seq_len - 1):
            obs, pos, vel = self.step()
            obs_list.append(obs)
            pos_list.append(pos)
            vel_list.append(vel)
            tunnel_list.append(self.is_in_tunnel())

        return (
            torch.stack(obs_list, dim=0),
            torch.stack(pos_list, dim=0),
            torch.stack(vel_list, dim=0),
            torch.stack(tunnel_list, dim=0)
        )

    def generate_batch(
        self,
        num_trajectories: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple trajectories with tunnel tracking.

        Returns:
            observations: [num_trajectories, seq_len, size, size]
            positions: [num_trajectories, seq_len, 2]
            velocities: [num_trajectories, seq_len, 2]
            in_tunnel: [num_trajectories, seq_len]
        """
        obs, pos, vel, tunnel = self.generate_trajectory(seq_len, num_trajectories)
        return (
            obs.transpose(0, 1),
            pos.transpose(0, 1),
            vel.transpose(0, 1),
            tunnel.transpose(0, 1)
        )


def test_tunnel_env():
    """Test the tunnel environment."""
    print("Testing NoisyTunnelEnv...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TunnelConfig(device=device)
    env = NoisyTunnelEnv(config)

    # Test reset
    obs = env.reset(batch_size=4)
    assert obs.shape == (4, 64, 64), f"Expected (4, 64, 64), got {obs.shape}"
    print(f"  Reset: obs shape = {obs.shape}")

    # Verify tunnel is occluded
    tunnel_region = obs[:, :, 24:40]
    assert (tunnel_region == 0).all(), "Tunnel region should be zero"
    print(f"  Tunnel occlusion verified: pixels 24-40 are zero")

    # Test step
    obs, pos, vel = env.step()
    assert obs.shape == (4, 64, 64)
    assert pos.shape == (4, 2)
    assert vel.shape == (4, 2)
    print(f"  Step: obs={obs.shape}, pos={pos.shape}, vel={vel.shape}")

    # Generate trajectory
    obs, pos, vel, tunnel = env.generate_batch(num_trajectories=8, seq_len=64)
    assert obs.shape == (8, 64, 64, 64)
    assert tunnel.shape == (8, 64)
    print(f"  Batch: obs={obs.shape}, in_tunnel={tunnel.shape}")

    # Count tunnel events
    num_in_tunnel = tunnel.sum().item()
    total_frames = tunnel.numel()
    print(f"  Tunnel events: {num_in_tunnel} / {total_frames} ({100*num_in_tunnel/total_frames:.1f}%)")

    # Visualize single frame (ASCII)
    print("\n  Sample frame (t=32):")
    frame = obs[0, 32].cpu().numpy()
    for row in range(0, 64, 4):
        line = "  "
        for col in range(0, 64, 2):
            val = frame[row, col]
            if 24 <= col < 40:
                char = "█"  # Tunnel
            elif val > 0.5:
                char = "●"  # Ball
            elif val > 0.1:
                char = "○"
            else:
                char = " "
            line += char
        print(line)

    print(f"\n  Ball position at t=32: ({pos[0, 32, 0].item():.1f}, {pos[0, 32, 1].item():.1f})")
    print(f"  In tunnel: {tunnel[0, 32].item()}")

    print("\nAll tunnel tests passed!")


if __name__ == "__main__":
    test_tunnel_env()
