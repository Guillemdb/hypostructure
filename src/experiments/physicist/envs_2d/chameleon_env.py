"""
ChameleonEnv: Low SNR Bouncing Ball.

The "Chameleon" Test for Temporal Integration.

A ball bounces in a 2D box, but the ball is FAINT - its pixel intensity
is only slightly higher than the noise variance. In a single frame,
the ball is statistically indistinguishable from noise.

The Challenge:
- Single-frame VAE sees only noise (SNR ≈ 1)
- Ball can only be detected through temporal correlation
- Requires integrating evidence across multiple frames

The Physicist Requirement:
- z_macro must accumulate evidence over time
- z_micro must discard uncorrelated frame-to-frame noise
- Physics engine provides prior for where to "look"

If the agent can track the ball when single-frame detection fails,
you have achieved TEMPORAL INTEGRATION.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from .base_2d_env import Base2DEnv, Base2DConfig


@dataclass
class ChameleonConfig(Base2DConfig):
    """Configuration for the Chameleon environment."""

    # Low SNR settings
    signal_intensity: float = 0.3      # Ball brightness (faint!)
    noise_sigma: float = 0.2           # Noise std (comparable to signal)

    # Optional: tunnel for combined test
    add_tunnel: bool = False
    tunnel_left: int = 24
    tunnel_right: int = 40

    # Inherited defaults
    size: int = 64
    ball_radius: float = 3.0
    speed: float = 2.0
    device: str = "cuda"


class ChameleonEnv(Base2DEnv):
    """
    2D Bouncing Ball with Low Signal-to-Noise Ratio.

    The ball is faint (signal ≈ noise). In a single frame, the ball is
    statistically indistinguishable from the background noise.

    Physics:
    - Ball bounces off walls (standard dynamics)
    - Ball intensity = 0.3 (vs noise σ = 0.2)
    - Effective SNR ≈ 1.5 (barely detectable)

    This tests:
    - Temporal Integration: Must correlate across frames to detect
    - Macro/Micro Separation: Macro accumulates, Micro discards
    - Physics Prior: Velocity prediction helps "find" the ball

    Expected behavior:
    - Single-frame: ~50% detection (chance)
    - Multi-frame baseline: Poor tracking
    - Physicist: Good tracking through temporal physics prior

    Optionally includes a tunnel for combined occlusion + low SNR test.
    """

    def __init__(self, config: Optional[ChameleonConfig] = None):
        self.chameleon_config = config or ChameleonConfig()
        super().__init__(self.chameleon_config)

        # Optional tunnel occlusion
        if self.chameleon_config.add_tunnel:
            self.occlusion_mask = torch.ones(
                self.chameleon_config.size,
                self.chameleon_config.size,
                device=self.device
            )
            self.occlusion_mask[:, self.chameleon_config.tunnel_left:self.chameleon_config.tunnel_right] = 0.0
        else:
            self.occlusion_mask = None

    def _render(self) -> torch.Tensor:
        """
        Render with low intensity ball and high noise.

        Returns:
            observation: [batch_size, size, size]
        """
        # Render faint ball
        ball = self._render_ball()  # Already uses signal_intensity from config

        # Apply optional tunnel occlusion
        if self.occlusion_mask is not None:
            ball = ball * self.occlusion_mask.unsqueeze(0)

        # Add noise (comparable to signal!)
        noise = torch.randn_like(ball) * self.config.noise_sigma
        obs = ball + noise

        # Apply occlusion to noise too if tunnel enabled
        if self.occlusion_mask is not None:
            obs = obs * self.occlusion_mask.unsqueeze(0)

        return obs

    def render_clean(self) -> torch.Tensor:
        """Render the faint ball without noise."""
        ball = self._render_ball()
        if self.occlusion_mask is not None:
            ball = ball * self.occlusion_mask.unsqueeze(0)
        return ball

    def render_ground_truth(self) -> torch.Tensor:
        """Render bright ball without noise (true state, easy to see)."""
        # Temporarily override signal intensity
        original_intensity = self.config.signal_intensity
        self.config.signal_intensity = 1.0
        ball = self._render_ball()
        self.config.signal_intensity = original_intensity
        return ball

    def compute_snr(self) -> float:
        """Compute the signal-to-noise ratio."""
        return self.chameleon_config.signal_intensity / self.chameleon_config.noise_sigma

    def is_in_tunnel(self) -> torch.Tensor:
        """Check if ball is in tunnel (if tunnel enabled)."""
        if self.occlusion_mask is None:
            return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return (
            (self.pos[:, 0] >= self.chameleon_config.tunnel_left) &
            (self.pos[:, 0] < self.chameleon_config.tunnel_right)
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
            in_tunnel: [seq_len, batch_size] (all False if no tunnel)
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
        """Generate multiple trajectories."""
        obs, pos, vel, tunnel = self.generate_trajectory(seq_len, num_trajectories)
        return (
            obs.transpose(0, 1),
            pos.transpose(0, 1),
            vel.transpose(0, 1),
            tunnel.transpose(0, 1)
        )


def test_chameleon_env():
    """Test the chameleon environment."""
    print("Testing ChameleonEnv...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ChameleonConfig(device=device)
    env = ChameleonEnv(config)

    print(f"  SNR: {env.compute_snr():.2f} (signal={config.signal_intensity}, noise={config.noise_sigma})")

    # Test reset
    obs = env.reset(batch_size=4)
    assert obs.shape == (4, 64, 64)
    print(f"  Reset: obs shape = {obs.shape}")

    # Analyze signal vs noise
    clean = env.render_clean()
    obs_sample = obs[0].cpu().numpy()
    clean_sample = clean[0].cpu().numpy()

    ball_mask = clean_sample > 0.1
    if ball_mask.any():
        ball_pixels = obs_sample[ball_mask]
        bg_pixels = obs_sample[~ball_mask]
        print(f"  Ball region mean: {ball_pixels.mean():.3f} (expected: {config.signal_intensity:.3f})")
        print(f"  Background mean: {bg_pixels.mean():.3f} (expected: ~0)")
        print(f"  Background std: {bg_pixels.std():.3f} (expected: {config.noise_sigma:.3f})")

    # Generate trajectory
    obs, pos, vel, tunnel = env.generate_batch(num_trajectories=8, seq_len=64)
    assert obs.shape == (8, 64, 64, 64)
    print(f"  Batch: obs={obs.shape}")

    # Demonstrate temporal integration
    print("\n  Temporal integration demonstration:")
    # Average 10 frames - should reveal the ball better
    obs_single = obs[0, 32].cpu().numpy()  # Single frame
    obs_avg = obs[0, 28:38].mean(dim=0).cpu().numpy()  # 10-frame average

    print(f"  Single frame max: {obs_single.max():.3f}")
    print(f"  10-frame average max: {obs_avg.max():.3f}")
    print(f"  Improvement: {obs_avg.max() / max(obs_single.max(), 1e-6):.2f}x")

    # Visual comparison
    print("\n  Single frame (hard to see ball):")
    for row in range(0, 64, 8):
        line = "  "
        for col in range(0, 64, 2):
            val = obs_single[row, col]
            if val > 0.4:
                char = "●"
            elif val > 0.2:
                char = "○"
            elif val > 0.0:
                char = "·"
            else:
                char = " "
            line += char
        print(line)

    print(f"\n  Ball position: ({pos[0, 32, 0].item():.1f}, {pos[0, 32, 1].item():.1f})")

    # Test with tunnel
    print("\n  Testing Chameleon + Tunnel combo...")
    combo_config = ChameleonConfig(add_tunnel=True, device=device)
    combo_env = ChameleonEnv(combo_config)
    obs, pos, vel, tunnel = combo_env.generate_batch(8, 64)
    tunnel_count = tunnel.sum().item()
    print(f"  Tunnel events: {tunnel_count}")

    print("\nAll chameleon tests passed!")


if __name__ == "__main__":
    test_chameleon_env()
