"""
1D Noisy Bouncing Pulse Environment.

A mathematically rigorous test environment for the Physicist architecture.

The environment generates:
- A Gaussian pulse that bounces deterministically between boundaries
- Heavy Gaussian noise overlaid on each observation

The challenge: Can the agent learn that position/velocity determine the future,
while noise is irrelevant for prediction?

Ground truth physics:
    position_{t+1} = position_t + velocity_t * dt
    velocity_{t+1} = -velocity_t  (if hitting boundary)
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class NoisyPulseConfig:
    """Configuration for the Noisy Pulse environment."""

    width: int = 64                    # Number of pixels
    pulse_sigma: float = 3.0           # Gaussian pulse width (pixels)
    noise_sigma: float = 0.5           # Noise standard deviation
    velocity: float = 2.0              # Base velocity (pixels per step)
    dt: float = 1.0                    # Time step
    device: str = "cuda"               # Device to use


class NoisyPulseEnv:
    """
    1D Noisy Bouncing Pulse Environment.

    A Gaussian pulse bounces back and forth in a 1D strip.
    Observations are corrupted by heavy Gaussian noise.

    This is the simplest possible test of the Physicist architecture:
    - True latent dim = 2 (position, velocity sign)
    - Observable dim = 64 (pixels)
    - Noise dim = 64 (independent per pixel per frame)

    The physics is deterministic and simple:
        x_{t+1} = x_t + v_t * dt
        v_{t+1} = -v_t  if x hits boundary, else v_t

    Args:
        config: Environment configuration
    """

    def __init__(self, config: Optional[NoisyPulseConfig] = None):
        self.config = config or NoisyPulseConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # State variables (batched)
        self.position: Optional[torch.Tensor] = None      # [batch_size]
        self.velocity: Optional[torch.Tensor] = None      # [batch_size]
        self.batch_size: int = 0

        # Precompute pixel coordinates
        self.pixel_coords = torch.arange(
            self.config.width, dtype=torch.float32, device=self.device
        )  # [width]

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """
        Reset the environment.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initial observation [batch_size, width]
        """
        self.batch_size = batch_size

        # Random initial position in [pulse_sigma, width - pulse_sigma]
        margin = self.config.pulse_sigma * 2
        self.position = torch.rand(batch_size, device=self.device) * (self.config.width - 2 * margin) + margin

        # Random initial velocity direction
        self.velocity = torch.where(
            torch.rand(batch_size, device=self.device) > 0.5,
            torch.full((batch_size,), self.config.velocity, device=self.device),
            torch.full((batch_size,), -self.config.velocity, device=self.device)
        )

        return self._render()

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take one environment step.

        Returns:
            observation: Noisy observation [batch_size, width]
            position: Ground truth position [batch_size]
            velocity: Ground truth velocity [batch_size]
        """
        assert self.position is not None, "Must call reset() first"

        # Update position
        self.position = self.position + self.velocity * self.config.dt

        # Bounce off boundaries
        # Left boundary
        left_bounce = self.position < self.config.pulse_sigma
        self.position = torch.where(
            left_bounce,
            2 * self.config.pulse_sigma - self.position,
            self.position
        )
        self.velocity = torch.where(left_bounce, -self.velocity, self.velocity)

        # Right boundary
        right_bound = self.config.width - self.config.pulse_sigma
        right_bounce = self.position > right_bound
        self.position = torch.where(
            right_bounce,
            2 * right_bound - self.position,
            self.position
        )
        self.velocity = torch.where(right_bounce, -self.velocity, self.velocity)

        return self._render(), self.position.clone(), self.velocity.clone()

    def _render(self) -> torch.Tensor:
        """
        Render current state as noisy 1D observation.

        Returns:
            observation: [batch_size, width]
        """
        # Compute clean Gaussian pulse: exp(-0.5 * ((x - pos) / sigma)^2)
        # position: [batch_size], pixel_coords: [width]
        # Result: [batch_size, width]
        diff = self.pixel_coords.unsqueeze(0) - self.position.unsqueeze(1)  # [B, W]
        clean = torch.exp(-0.5 * (diff / self.config.pulse_sigma) ** 2)

        # Add noise
        noise = torch.randn_like(clean) * self.config.noise_sigma
        noisy = clean + noise

        return noisy

    def render_clean(self) -> torch.Tensor:
        """Render without noise (for visualization/debugging)."""
        diff = self.pixel_coords.unsqueeze(0) - self.position.unsqueeze(1)
        return torch.exp(-0.5 * (diff / self.config.pulse_sigma) ** 2)

    def get_ground_truth(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ground truth state.

        Returns:
            position: [batch_size]
            velocity: [batch_size]
        """
        return self.position.clone(), self.velocity.clone()

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
            observations: [seq_len, batch_size, width]
            positions: [seq_len, batch_size]
            velocities: [seq_len, batch_size]
        """
        obs_list = []
        pos_list = []
        vel_list = []

        # Reset and get initial observation
        obs = self.reset(batch_size)
        obs_list.append(obs)
        pos_list.append(self.position.clone())
        vel_list.append(self.velocity.clone())

        # Generate trajectory
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

        Args:
            num_trajectories: Number of trajectories
            seq_len: Length of each trajectory

        Returns:
            observations: [num_trajectories, seq_len, width]
            positions: [num_trajectories, seq_len]
            velocities: [num_trajectories, seq_len]
        """
        obs, pos, vel = self.generate_trajectory(seq_len, num_trajectories)
        # Transpose from [seq_len, batch, ...] to [batch, seq_len, ...]
        return obs.transpose(0, 1), pos.transpose(0, 1), vel.transpose(0, 1)


@dataclass
class OccludedPulseConfig(NoisyPulseConfig):
    """Configuration for the Occluded Pulse environment."""

    occlusion_start: int = 30          # Start of occlusion region
    occlusion_end: int = 34            # End of occlusion region (exclusive)


class OccludedPulseEnv(NoisyPulseEnv):
    """
    1D Noisy Bouncing Pulse with Occlusion.

    Same as NoisyPulseEnv but with a "black bar" in the middle of the screen
    where pixels are always 0. This tests OBJECT PERMANENCE:

    - When the ball passes behind the bar, it disappears from observation
    - The agent must maintain belief about the ball's position
    - Upon re-emergence, predictions should be accurate

    This is a test of Node 6 (GeomCheck) - can the agent maintain geometric
    structure even when observations are incomplete?

    Expected behavior:
    - Baseline: "Forgets" the ball when it's behind the bar, predictions fail
    - Physicist: Maintains physics simulation, predicts re-emergence correctly

    Args:
        config: Environment configuration with occlusion parameters
    """

    def __init__(self, config: Optional[OccludedPulseConfig] = None):
        self.occlusion_config = config or OccludedPulseConfig()
        super().__init__(self.occlusion_config)

        # Create occlusion mask
        self.occlusion_mask = torch.ones(
            self.occlusion_config.width, device=self.device
        )
        self.occlusion_mask[
            self.occlusion_config.occlusion_start:self.occlusion_config.occlusion_end
        ] = 0.0

    def _render(self) -> torch.Tensor:
        """Render with occlusion applied."""
        # Get base rendering (clean + noise)
        diff = self.pixel_coords.unsqueeze(0) - self.position.unsqueeze(1)
        clean = torch.exp(-0.5 * (diff / self.config.pulse_sigma) ** 2)
        noise = torch.randn_like(clean) * self.config.noise_sigma
        noisy = clean + noise

        # Apply occlusion mask
        noisy = noisy * self.occlusion_mask.unsqueeze(0)

        return noisy

    def render_clean(self) -> torch.Tensor:
        """Render without noise but with occlusion."""
        diff = self.pixel_coords.unsqueeze(0) - self.position.unsqueeze(1)
        clean = torch.exp(-0.5 * (diff / self.config.pulse_sigma) ** 2)
        return clean * self.occlusion_mask.unsqueeze(0)

    def render_ground_truth(self) -> torch.Tensor:
        """Render without noise AND without occlusion (true state)."""
        diff = self.pixel_coords.unsqueeze(0) - self.position.unsqueeze(1)
        return torch.exp(-0.5 * (diff / self.config.pulse_sigma) ** 2)

    def is_occluded(self) -> torch.Tensor:
        """
        Check if the ball is currently behind the occlusion.

        Returns:
            Boolean tensor [batch_size] indicating occlusion status
        """
        return (
            (self.position >= self.occlusion_config.occlusion_start) &
            (self.position < self.occlusion_config.occlusion_end)
        )

    def generate_trajectory(
        self,
        seq_len: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a trajectory with occlusion tracking.

        Returns:
            observations: [seq_len, batch_size, width]
            positions: [seq_len, batch_size]
            velocities: [seq_len, batch_size]
            occluded: [seq_len, batch_size] - bool mask of occlusion
        """
        obs_list = []
        pos_list = []
        vel_list = []
        occ_list = []

        # Reset and get initial observation
        obs = self.reset(batch_size)
        obs_list.append(obs)
        pos_list.append(self.position.clone())
        vel_list.append(self.velocity.clone())
        occ_list.append(self.is_occluded())

        # Generate trajectory
        for _ in range(seq_len - 1):
            obs, pos, vel = self.step()
            obs_list.append(obs)
            pos_list.append(pos)
            vel_list.append(vel)
            occ_list.append(self.is_occluded())

        return (
            torch.stack(obs_list, dim=0),
            torch.stack(pos_list, dim=0),
            torch.stack(vel_list, dim=0),
            torch.stack(occ_list, dim=0)
        )

    def generate_batch(
        self,
        num_trajectories: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple trajectories with occlusion tracking.

        Returns:
            observations: [num_trajectories, seq_len, width]
            positions: [num_trajectories, seq_len]
            velocities: [num_trajectories, seq_len]
            occluded: [num_trajectories, seq_len]
        """
        obs, pos, vel, occ = self.generate_trajectory(seq_len, num_trajectories)
        return (
            obs.transpose(0, 1),
            pos.transpose(0, 1),
            vel.transpose(0, 1),
            occ.transpose(0, 1)
        )


def test_environment():
    """Test the environment works correctly."""
    print("Testing NoisyPulseEnv...")

    config = NoisyPulseConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    env = NoisyPulseEnv(config)

    # Test reset
    obs = env.reset(batch_size=4)
    assert obs.shape == (4, 64), f"Expected (4, 64), got {obs.shape}"
    print(f"  Reset: obs shape = {obs.shape}")

    # Test step
    obs, pos, vel = env.step()
    assert obs.shape == (4, 64)
    assert pos.shape == (4,)
    assert vel.shape == (4,)
    print(f"  Step: obs={obs.shape}, pos={pos.shape}, vel={vel.shape}")

    # Test trajectory generation
    obs, pos, vel = env.generate_trajectory(seq_len=50, batch_size=8)
    assert obs.shape == (50, 8, 64)
    assert pos.shape == (50, 8)
    assert vel.shape == (50, 8)
    print(f"  Trajectory: obs={obs.shape}, pos={pos.shape}, vel={vel.shape}")

    # Test batch generation
    obs, pos, vel = env.generate_batch(num_trajectories=16, seq_len=32)
    assert obs.shape == (16, 32, 64)
    print(f"  Batch: obs={obs.shape}, pos={pos.shape}, vel={vel.shape}")

    # Verify physics: check position changes correctly
    env.reset(batch_size=1)
    pos0, vel0 = env.get_ground_truth()
    _, pos1, vel1 = env.step()
    expected_pos = pos0 + vel0 * config.dt
    # Account for bouncing
    if expected_pos.item() < config.pulse_sigma or expected_pos.item() > config.width - config.pulse_sigma:
        print(f"  Physics: Bounce detected")
    else:
        assert torch.allclose(pos1, expected_pos, atol=1e-5), "Physics mismatch!"
        print(f"  Physics: pos0={pos0.item():.2f}, vel={vel0.item():.2f}, pos1={pos1.item():.2f}")

    print("All tests passed!")


def test_occluded_environment():
    """Test the occluded environment."""
    print("\nTesting OccludedPulseEnv...")

    config = OccludedPulseConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    env = OccludedPulseEnv(config)

    # Test reset
    obs = env.reset(batch_size=4)
    assert obs.shape == (4, 64), f"Expected (4, 64), got {obs.shape}"
    print(f"  Reset: obs shape = {obs.shape}")

    # Verify occlusion mask is applied
    assert (obs[:, 30:34] == 0).all() or (obs[:, 30:34].abs() < 1e-6).all(), \
        "Occlusion not applied correctly"
    print(f"  Occlusion mask applied: pixels 30-34 are zeroed")

    # Generate trajectory and check occlusion tracking
    obs, pos, vel, occ = env.generate_batch(num_trajectories=8, seq_len=64)
    assert obs.shape == (8, 64, 64)
    assert occ.shape == (8, 64)
    print(f"  Trajectory: obs={obs.shape}, occluded={occ.shape}")

    # Count occlusion events
    num_occluded = occ.sum().item()
    print(f"  Total occluded frames: {num_occluded} / {occ.numel()}")

    # Verify that occlusion status matches position
    for i in range(8):
        for t in range(64):
            expected_occ = 30 <= pos[i, t].item() < 34
            actual_occ = occ[i, t].item()
            assert expected_occ == actual_occ, \
                f"Occlusion mismatch at traj={i}, t={t}: pos={pos[i,t].item():.2f}"

    print("  Occlusion status verified against position")

    # Visualize a single trajectory (text-based)
    print("\n  Sample trajectory visualization (8 timesteps):")
    for t in range(0, 64, 8):
        obs_t = obs[0, t].cpu().numpy()
        pos_t = pos[0, t].item()
        occ_t = "OCCLUDED" if occ[0, t].item() else ""
        # Simple ASCII visualization
        viz = "".join(["█" if v > 0.3 else "▒" if v > 0.1 else " " if v > -0.1 else "░"
                       for v in obs_t])
        # Mark occlusion region
        viz = viz[:30] + "|" + viz[30:34] + "|" + viz[34:]
        print(f"    t={t:2d}: {viz} pos={pos_t:5.1f} {occ_t}")

    print("\nAll occlusion tests passed!")


if __name__ == "__main__":
    test_environment()
    test_occluded_environment()
