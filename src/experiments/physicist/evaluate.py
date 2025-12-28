"""
Evaluation and visualization for Physicist experiment.

Provides:
- Multi-step prediction accuracy
- Position tracking correlation
- Latent trajectory visualization
- Reconstruction quality comparison
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .noisy_pulse_env import NoisyPulseEnv, NoisyPulseConfig
from .physicist_agent import PhysicistAgent1D, PhysicistConfig1D
from .baseline_agent import BaselineAgent1D, BaselineConfig1D


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    num_trajectories: int = 50
    seq_len: int = 100
    prediction_horizons: List[int] = None  # Will default to [1, 5, 10, 20]
    device: str = "cuda"

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 10, 20]


class Evaluator:
    """
    Evaluator for Physicist and Baseline agents.

    Metrics computed:
    - Reconstruction MSE (clean and noisy)
    - Multi-step prediction MSE
    - Position tracking R^2
    - Closure ratio
    - Latent interpretability (correlation with ground truth)
    """

    def __init__(
        self,
        agent: nn.Module,
        config: Optional[EvalConfig] = None,
        agent_name: str = "agent"
    ):
        self.agent = agent
        self.config = config or EvalConfig()
        self.agent_name = agent_name
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self.agent.to(self.device)
        self.agent.eval()

    def generate_test_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test data with both noisy and clean observations."""
        env_config = NoisyPulseConfig(device=str(self.device))
        env = NoisyPulseEnv(env_config)

        # Generate trajectories
        obs_noisy, pos, vel = env.generate_batch(
            self.config.num_trajectories,
            self.config.seq_len
        )

        # Generate clean version (same positions, no noise)
        # We'll reconstruct clean observations from positions
        pixel_coords = torch.arange(
            env_config.width, dtype=torch.float32, device=self.device
        )
        obs_clean = torch.zeros_like(obs_noisy)
        for t in range(self.config.seq_len):
            diff = pixel_coords.unsqueeze(0) - pos[:, t:t+1]
            obs_clean[:, t] = torch.exp(-0.5 * (diff / env_config.pulse_sigma) ** 2)

        return obs_noisy, obs_clean, pos, vel

    @torch.no_grad()
    def compute_reconstruction_mse(
        self,
        obs: torch.Tensor,
        clean_target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction MSE.

        Args:
            obs: Noisy observations [batch, seq_len, obs_dim]
            clean_target: Clean observations [batch, seq_len, obs_dim]

        Returns:
            Dictionary with MSE values
        """
        batch_size, seq_len, obs_dim = obs.shape

        # Flatten and encode
        obs_flat = obs.reshape(-1, obs_dim)
        clean_flat = clean_target.reshape(-1, obs_dim)

        # Reconstruct
        x_recon, _ = self.agent(obs_flat)

        # MSE against noisy input
        mse_noisy = F.mse_loss(x_recon, obs_flat).item()

        # MSE against clean target (the true test)
        mse_clean = F.mse_loss(x_recon, clean_flat).item()

        return {
            "mse_noisy": mse_noisy,
            "mse_clean": mse_clean,
        }

    @torch.no_grad()
    def compute_multistep_prediction_mse(
        self,
        obs: torch.Tensor,
    ) -> Dict[int, float]:
        """
        Compute multi-step prediction MSE.

        For each horizon k, predict z_{t+k} from z_t and measure error.

        Args:
            obs: Observations [batch, seq_len, obs_dim]

        Returns:
            Dictionary mapping horizon to MSE
        """
        batch_size, seq_len, obs_dim = obs.shape
        results = {}

        # Encode all observations
        obs_flat = obs.reshape(-1, obs_dim)

        # Get latent means (for prediction targets)
        if hasattr(self.agent, "encoder") and hasattr(self.agent.encoder, "macro_mean"):
            # Physicist agent
            latents = self.agent.encode(obs_flat)
            z = latents["macro_mean"].reshape(batch_size, seq_len, -1)
            is_physicist = True
        else:
            # Baseline agent
            latents = self.agent.encode(obs_flat)
            z = latents["mean"].reshape(batch_size, seq_len, -1)
            is_physicist = False

        for horizon in self.config.prediction_horizons:
            if horizon >= seq_len:
                continue

            # Get source and target latents
            z_source = z[:, :-horizon]  # [batch, seq_len-horizon, latent]
            z_target = z[:, horizon:]   # [batch, seq_len-horizon, latent]

            # Multi-step prediction
            z_pred = z_source.clone()
            for _ in range(horizon):
                if is_physicist:
                    z_pred = self.agent.predict_next_macro(
                        z_pred.reshape(-1, z_pred.shape[-1])
                    ).reshape(z_pred.shape)
                else:
                    z_pred, _ = self.agent.predict_next(z_pred)

            # Compute MSE
            mse = F.mse_loss(z_pred, z_target).item()
            results[horizon] = mse

        return results

    @torch.no_grad()
    def compute_position_tracking_r2(
        self,
        obs: torch.Tensor,
        true_pos: torch.Tensor
    ) -> float:
        """
        Compute R^2 for position tracking.

        Measures how well the latent encodes position.

        Args:
            obs: Observations [batch, seq_len, obs_dim]
            true_pos: Ground truth positions [batch, seq_len]

        Returns:
            R^2 value (1.0 = perfect tracking)
        """
        batch_size, seq_len, obs_dim = obs.shape

        # Flatten
        obs_flat = obs.reshape(-1, obs_dim)
        pos_flat = true_pos.reshape(-1)

        # Extract position estimate
        estimated_pos = self.agent.extract_position(obs_flat)

        # Compute correlation
        # R^2 = 1 - SS_res / SS_tot
        pos_mean = pos_flat.mean()
        ss_tot = ((pos_flat - pos_mean) ** 2).sum()
        ss_res = ((pos_flat - estimated_pos) ** 2).sum()

        # Handle case where estimated_pos might be scaled/shifted
        # Use best linear fit R^2
        x = estimated_pos - estimated_pos.mean()
        y = pos_flat - pos_mean

        if x.std() > 1e-8 and y.std() > 1e-8:
            correlation = (x * y).mean() / (x.std() * y.std())
            r2 = correlation.item() ** 2
        else:
            r2 = 0.0

        return r2

    @torch.no_grad()
    def compute_latent_statistics(
        self,
        obs: torch.Tensor,
        true_pos: torch.Tensor,
        true_vel: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute latent space statistics.

        Args:
            obs: Observations [batch, seq_len, obs_dim]
            true_pos: Ground truth positions [batch, seq_len]
            true_vel: Ground truth velocities [batch, seq_len]

        Returns:
            Dictionary with latent statistics
        """
        batch_size, seq_len, obs_dim = obs.shape

        obs_flat = obs.reshape(-1, obs_dim)
        pos_flat = true_pos.reshape(-1)
        vel_flat = true_vel.reshape(-1)

        latents = self.agent.encode(obs_flat)

        stats = {}

        if hasattr(self.agent, "encoder") and hasattr(self.agent.encoder, "macro_mean"):
            # Physicist agent
            z_macro = latents["macro_mean"]
            z_micro = latents["micro_mean"]

            # Macro variance (should be low-dimensional)
            stats["macro_variance"] = z_macro.var().item()
            stats["micro_variance"] = z_micro.var().item()

            # Correlation of z_macro with position/velocity
            for i in range(z_macro.shape[-1]):
                z_i = z_macro[:, i]
                # Position correlation
                r_pos = self._correlation(z_i, pos_flat)
                stats[f"macro_{i}_pos_corr"] = abs(r_pos)
                # Velocity correlation
                r_vel = self._correlation(z_i, vel_flat)
                stats[f"macro_{i}_vel_corr"] = abs(r_vel)

        else:
            # Baseline agent
            z = latents["mean"]
            stats["latent_variance"] = z.var().item()

            # Check first 2 components for position/velocity
            for i in range(min(2, z.shape[-1])):
                z_i = z[:, i]
                r_pos = self._correlation(z_i, pos_flat)
                stats[f"z_{i}_pos_corr"] = abs(r_pos)
                r_vel = self._correlation(z_i, vel_flat)
                stats[f"z_{i}_vel_corr"] = abs(r_vel)

        return stats

    def _correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation."""
        x = x - x.mean()
        y = y - y.mean()
        if x.std() > 1e-8 and y.std() > 1e-8:
            return ((x * y).mean() / (x.std() * y.std())).item()
        return 0.0

    def evaluate_all(self) -> Dict[str, any]:
        """
        Run all evaluations.

        Returns:
            Dictionary with all metrics
        """
        print(f"\nEvaluating {self.agent_name}...")

        # Generate test data
        obs_noisy, obs_clean, pos, vel = self.generate_test_data()
        print(f"  Test data: {obs_noisy.shape}")

        results = {"agent": self.agent_name}

        # Reconstruction MSE
        recon_mse = self.compute_reconstruction_mse(obs_noisy, obs_clean)
        results.update(recon_mse)
        print(f"  Reconstruction MSE (noisy): {recon_mse['mse_noisy']:.6f}")
        print(f"  Reconstruction MSE (clean): {recon_mse['mse_clean']:.6f}")

        # Multi-step prediction
        pred_mse = self.compute_multistep_prediction_mse(obs_noisy)
        results["prediction_mse"] = pred_mse
        print(f"  Prediction MSE:")
        for horizon, mse in pred_mse.items():
            print(f"    {horizon}-step: {mse:.6f}")

        # Position tracking
        r2 = self.compute_position_tracking_r2(obs_noisy, pos)
        results["position_r2"] = r2
        print(f"  Position tracking R^2: {r2:.4f}")

        # Closure ratio
        ratio = self.agent.compute_closure_ratio(obs_noisy)
        results["closure_ratio"] = ratio
        print(f"  Closure ratio: {ratio:.4f}")

        # Latent statistics
        latent_stats = self.compute_latent_statistics(obs_noisy, pos, vel)
        results["latent_stats"] = latent_stats
        print(f"  Latent statistics:")
        for k, v in latent_stats.items():
            print(f"    {k}: {v:.4f}")

        return results


def compare_agents(
    physicist: PhysicistAgent1D,
    baseline: BaselineAgent1D,
    config: Optional[EvalConfig] = None
) -> Dict[str, Dict]:
    """
    Compare Physicist and Baseline agents.

    Args:
        physicist: Trained Physicist agent
        baseline: Trained Baseline agent
        config: Evaluation configuration

    Returns:
        Dictionary with results for both agents
    """
    config = config or EvalConfig()

    print("=" * 60)
    print("EVALUATION COMPARISON")
    print("=" * 60)

    # Evaluate physicist
    physicist_eval = Evaluator(physicist, config, "physicist")
    physicist_results = physicist_eval.evaluate_all()

    # Evaluate baseline
    baseline_eval = Evaluator(baseline, config, "baseline")
    baseline_results = baseline_eval.evaluate_all()

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<25} {:>15} {:>15}".format("Metric", "Physicist", "Baseline"))
    print("-" * 55)

    # Key metrics
    metrics = [
        ("Recon MSE (clean)", "mse_clean"),
        ("Position R^2", "position_r2"),
        ("Closure ratio", "closure_ratio"),
    ]

    for name, key in metrics:
        p_val = physicist_results.get(key, 0)
        b_val = baseline_results.get(key, 0)
        winner = "<<" if p_val > b_val else ">>" if b_val > p_val else "=="

        if key in ["mse_clean"]:  # Lower is better
            winner = "<<" if p_val < b_val else ">>" if b_val < p_val else "=="

        print("{:<25} {:>15.4f} {:>15.4f} {}".format(name, p_val, b_val, winner))

    # Prediction MSE comparison
    print("\nMulti-step Prediction MSE:")
    for horizon in config.prediction_horizons:
        p_val = physicist_results["prediction_mse"].get(horizon, float("inf"))
        b_val = baseline_results["prediction_mse"].get(horizon, float("inf"))
        winner = "<<" if p_val < b_val else ">>" if b_val < p_val else "=="
        print("{:<25} {:>15.6f} {:>15.6f} {}".format(f"  {horizon}-step", p_val, b_val, winner))

    return {
        "physicist": physicist_results,
        "baseline": baseline_results,
    }


def quick_test():
    """Quick test of evaluation functions."""
    print("Testing Evaluator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create agents
    physicist = PhysicistAgent1D(PhysicistConfig1D(device=device))
    baseline = BaselineAgent1D(BaselineConfig1D(device=device))

    # Quick evaluation config
    config = EvalConfig(
        num_trajectories=10,
        seq_len=32,
        prediction_horizons=[1, 5],
        device=device
    )

    # Compare
    results = compare_agents(physicist, baseline, config)

    print("\nTest passed!")
    return results


if __name__ == "__main__":
    quick_test()
