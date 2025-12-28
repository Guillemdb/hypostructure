#!/usr/bin/env python3
"""
Chameleon Test: 2D Temporal Integration Evaluation.

This is the "Signal in Noise" Boss Fight for the Physicist architecture.
Tests whether agents can track a ball when single-frame SNR is ~1.

Key insight:
- Single-frame VAE sees essentially noise (signal ≈ noise variance)
- Ball can only be detected through temporal correlation
- Physics engine provides prior for where to "look"

Key metrics:
- Single-frame detection: How well does single-frame encoding work?
- Multi-frame tracking: Can the agent track over time?
- SNR improvement: Does temporal integration boost effective SNR?

If the Physicist can track when single-frame detection fails,
you have achieved TEMPORAL INTEGRATION.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .envs_2d import ChameleonEnv, ChameleonConfig
from .agents_2d import PhysicistAgent2D, PhysicistConfig2D, BaselineAgent2D, BaselineConfig2D


@dataclass
class ChameleonTestConfig:
    """Configuration for chameleon testing."""

    # Data
    num_train_trajectories: int = 500
    num_val_trajectories: int = 50
    seq_len: int = 64

    # Training
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Evaluation
    num_eval_trajectories: int = 100
    eval_seq_len: int = 100

    # Environment
    signal_intensity: float = 0.3  # Faint ball
    noise_sigma: float = 0.2       # High noise

    # Output
    output_dir: str = "outputs/chameleon_test"

    # Device
    device: str = "cuda"


def generate_chameleon_data(
    num_trajectories: int,
    seq_len: int,
    config: ChameleonTestConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data from chameleon environment."""
    env_config = ChameleonConfig(
        signal_intensity=config.signal_intensity,
        noise_sigma=config.noise_sigma,
        device=config.device
    )
    env = ChameleonEnv(env_config)
    return env.generate_batch(num_trajectories, seq_len)


class ChameleonTrainer:
    """Trainer for chameleon test."""

    def __init__(
        self,
        agent: nn.Module,
        config: ChameleonTestConfig,
        agent_name: str
    ):
        self.agent = agent
        self.config = config
        self.agent_name = agent_name
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.agent.to(self.device)
        self.optimizer = optim.AdamW(
            agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        self.output_dir = Path(config.output_dir) / agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict:
        """Full training loop."""
        print(f"\nTraining {self.agent_name}...")

        # Generate data
        print("  Generating training data...")
        obs, pos, vel, _ = generate_chameleon_data(
            self.config.num_train_trajectories,
            self.config.seq_len,
            self.config
        )
        train_loader = DataLoader(
            TensorDataset(obs, pos, vel),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        print("  Generating validation data...")
        val_obs, val_pos, val_vel, _ = generate_chameleon_data(
            self.config.num_val_trajectories,
            self.config.seq_len,
            self.config
        )

        history = {"train_loss": [], "val_loss": [], "closure_ratio": []}

        # Warmup epochs (first 20% of training)
        warmup_epochs = max(1, self.config.num_epochs // 5)

        for epoch in range(1, self.config.num_epochs + 1):
            # Calculate warmup ratio
            if epoch <= warmup_epochs:
                warmup_ratio = epoch / warmup_epochs
            else:
                warmup_ratio = 1.0

            # Train
            self.agent.train()
            epoch_loss = 0.0
            for batch_obs, batch_pos, batch_vel in train_loader:
                self.optimizer.zero_grad()
                # Pass warmup_ratio and epoch if agent supports it
                if hasattr(self.agent, 'config') and hasattr(self.agent.config, 'macro_dim'):
                    losses = self.agent.compute_loss(batch_obs, training=True, warmup_ratio=warmup_ratio, epoch=epoch)
                else:
                    losses = self.agent.compute_loss(batch_obs, training=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += losses["total"].item()

            epoch_loss /= len(train_loader)
            history["train_loss"].append(epoch_loss)

            # Validate
            if epoch % 5 == 0:
                self.agent.eval()
                with torch.no_grad():
                    val_losses = self.agent.compute_loss(val_obs, training=False)
                    val_loss = val_losses["total"].item()
                    ratio = self.agent.compute_closure_ratio(val_obs)

                history["val_loss"].append(val_loss)
                history["closure_ratio"].append(ratio)

                warmup_str = f" [warmup={warmup_ratio:.1f}]" if warmup_ratio < 1.0 else ""
                print(f"  Epoch {epoch}: train={epoch_loss:.4f}, val={val_loss:.4f}, closure={ratio:.4f}{warmup_str}")

            self.scheduler.step()

        # Save
        torch.save(self.agent.state_dict(), self.output_dir / "model.pt")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f)

        return history


@torch.no_grad()
def evaluate_chameleon(
    agent: nn.Module,
    config: ChameleonTestConfig,
    agent_name: str
) -> Dict:
    """
    Evaluate temporal integration on chameleon task.

    Key metrics:
    - position_error_overall: Overall position tracking error
    - position_r2: Correlation between true and estimated position
    - temporal_smoothness: How smooth is the latent trajectory?
    - snr_effective: Effective SNR through temporal integration
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    agent.eval()

    # Generate evaluation data
    obs, pos, vel, _ = generate_chameleon_data(
        config.num_eval_trajectories,
        config.eval_seq_len,
        config
    )

    batch_size, seq_len, H, W = obs.shape

    # Encode all
    obs_flat = obs.reshape(-1, H, W)
    latents = agent.encode(obs_flat)

    # Get z_macro (for Physicist) or z[:4] (for Baseline)
    if hasattr(agent, 'config') and hasattr(agent.config, 'macro_dim'):
        z_macro = latents["macro_mean"].reshape(batch_size, seq_len, -1)
    else:
        z_macro = latents["mean"][:, :4].reshape(batch_size, seq_len, 4)

    # Position estimates
    estimated_pos = agent.extract_position(obs_flat).reshape(batch_size, seq_len, 2)

    # Position error
    position_error = (estimated_pos - pos).norm(dim=-1).mean().item()

    # Position R^2
    def compute_r2(estimated: torch.Tensor, true: torch.Tensor) -> float:
        if len(estimated) == 0:
            return 0
        x = estimated - estimated.mean(dim=0, keepdim=True)
        y = true - true.mean(dim=0, keepdim=True)
        if x.std() > 1e-8 and y.std() > 1e-8:
            corr = (x * y).mean() / (x.std() * y.std())
            return corr.item() ** 2
        return 0

    r2 = compute_r2(
        estimated_pos.reshape(-1, 2),
        pos.reshape(-1, 2)
    )

    # Temporal smoothness (latent stability)
    z_delta = (z_macro[:, 1:] - z_macro[:, :-1]).norm(dim=-1)
    temporal_smoothness = z_delta.mean().item()

    # Single-frame vs multi-frame analysis
    # Compare position error at different temporal windows
    errors_by_window = {}
    for window in [1, 5, 10, 20]:
        if window == 1:
            # Single frame: just use frame-by-frame
            window_pos = estimated_pos
        else:
            # Multi-frame: smooth the estimates using simple moving average
            # Use unfold for simpler windowed averaging
            window_size = min(window, seq_len)
            kernel = torch.ones(window_size, device=device) / window_size

            # Average over time dimension for each coordinate
            smoothed_x = F.conv1d(
                estimated_pos[:, :, 0:1].transpose(1, 2),
                kernel.view(1, 1, -1),
                padding=window_size - 1
            )[:, :, :seq_len].transpose(1, 2)

            smoothed_y = F.conv1d(
                estimated_pos[:, :, 1:2].transpose(1, 2),
                kernel.view(1, 1, -1),
                padding=window_size - 1
            )[:, :, :seq_len].transpose(1, 2)

            window_pos = torch.cat([smoothed_x, smoothed_y], dim=-1)

        err = (window_pos - pos).norm(dim=-1).mean().item()
        errors_by_window[f"error_window_{window}"] = err

    # Compute effective SNR improvement from temporal integration
    # Compare error at window=1 vs window=10
    if errors_by_window["error_window_1"] > 1e-8:
        snr_improvement = errors_by_window["error_window_1"] / max(errors_by_window["error_window_10"], 1e-8)
    else:
        snr_improvement = 1.0

    results = {
        "agent": agent_name,
        "position_error": position_error,
        "position_r2": r2,
        "temporal_smoothness": temporal_smoothness,
        "snr_improvement": snr_improvement,
        **errors_by_window,
    }

    return results


def run_chameleon_test(config: Optional[ChameleonTestConfig] = None, quick: bool = False) -> Dict:
    """Run the full chameleon test."""
    if config is None:
        if quick:
            config = ChameleonTestConfig(
                num_train_trajectories=100,
                num_val_trajectories=20,
                seq_len=48,
                num_epochs=15,
                batch_size=8,
                num_eval_trajectories=50,
                eval_seq_len=80,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            config = ChameleonTestConfig(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    print("=" * 70)
    print("CHAMELEON TEST: 2D TEMPORAL INTEGRATION")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Quick mode: {quick}")
    print(f"SNR: {config.signal_intensity / config.noise_sigma:.2f}")

    # Train Physicist
    physicist = PhysicistAgent2D(PhysicistConfig2D(device=config.device))
    physicist_trainer = ChameleonTrainer(physicist, config, "physicist")
    physicist_history = physicist_trainer.train()

    # Train Baseline
    baseline = BaselineAgent2D(BaselineConfig2D(device=config.device))
    baseline_trainer = ChameleonTrainer(baseline, config, "baseline")
    baseline_history = baseline_trainer.train()

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    physicist_results = evaluate_chameleon(physicist, config, "Physicist")
    baseline_results = evaluate_chameleon(baseline, config, "Baseline")

    # Print comparison
    print("\n{:<30} {:>15} {:>15} {:>10}".format("Metric", "Physicist", "Baseline", "Winner"))
    print("-" * 70)

    metrics = [
        ("Position error", "position_error", "lower"),
        ("Position R²", "position_r2", "higher"),
        ("Temporal smoothness", "temporal_smoothness", "lower"),
        ("SNR improvement", "snr_improvement", "higher"),
        ("Error (1-frame)", "error_window_1", "lower"),
        ("Error (5-frame)", "error_window_5", "lower"),
        ("Error (10-frame)", "error_window_10", "lower"),
    ]

    physicist_wins = 0
    baseline_wins = 0

    for name, key, direction in metrics:
        p_val = physicist_results[key]
        b_val = baseline_results[key]

        if direction == "higher":
            winner = "Phys" if p_val > b_val else "Base" if b_val > p_val else "Tie"
            if p_val > b_val:
                physicist_wins += 1
            elif b_val > p_val:
                baseline_wins += 1
        else:
            winner = "Phys" if p_val < b_val else "Base" if b_val < p_val else "Tie"
            if p_val < b_val:
                physicist_wins += 1
            elif b_val < p_val:
                baseline_wins += 1

        print("{:<30} {:>15.4f} {:>15.4f} {:>10}".format(name, p_val, b_val, winner))

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if physicist_wins > baseline_wins:
        print("\n  PHYSICIST ACHIEVES TEMPORAL INTEGRATION!")
        print("  The split-brain architecture extracts signal from noise!")
    elif baseline_wins > physicist_wins:
        print("\n  BASELINE WINS (unexpected)")
        print("  Consider training longer or adjusting hyperparameters.")
    else:
        print("\n  INCONCLUSIVE")

    print(f"\n  Physicist wins: {physicist_wins}")
    print(f"  Baseline wins:  {baseline_wins}")

    # Key insight
    if physicist_results["snr_improvement"] > 1.5:
        print("\n  *** TEMPORAL INTEGRATION ACHIEVED ***")
        print(f"  SNR improvement through temporal integration: {physicist_results['snr_improvement']:.2f}x")
        print("  The physics engine helps 'find' the faint ball across frames!")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "physicist": physicist_results,
        "baseline": baseline_results,
        "physicist_wins": physicist_wins,
        "baseline_wins": baseline_wins,
        "snr": config.signal_intensity / config.noise_sigma,
    }

    with open(output_dir / "chameleon_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir / 'chameleon_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.quick:
        run_chameleon_test(quick=True)
    else:
        config = ChameleonTestConfig(
            num_epochs=args.epochs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        run_chameleon_test(config)
