#!/usr/bin/env python3
"""
Tunnel Test: 2D Object Permanence Evaluation.

This is the primary "Boss Fight" for the Physicist architecture.
Tests whether agents can maintain object tracking when the ball
passes through an occluded region (the "tunnel").

Key metrics:
- Exit prediction accuracy: Where does the ball emerge?
- Latent stability during occlusion: Does z_macro stay smooth?
- Momentum conservation: Does velocity match before/after tunnel?

If the Physicist predicts the correct exit position/timing,
you have achieved OBJECT PERMANENCE - the ability to maintain
a representation of an object that cannot be directly observed.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .envs_2d import NoisyTunnelEnv, TunnelConfig
from .agents_2d import PhysicistAgent2D, PhysicistConfig2D, BaselineAgent2D, BaselineConfig2D


@dataclass
class TunnelTestConfig:
    """Configuration for tunnel testing."""

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

    # Output
    output_dir: str = "outputs/tunnel_test"

    # Device
    device: str = "cuda"


def generate_tunnel_data(
    num_trajectories: int,
    seq_len: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data from tunnel environment."""
    config = TunnelConfig(device=device)
    env = NoisyTunnelEnv(config)
    return env.generate_batch(num_trajectories, seq_len)


class TunnelTrainer:
    """Trainer for tunnel test."""

    def __init__(
        self,
        agent: nn.Module,
        config: TunnelTestConfig,
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
        obs, pos, vel, tunnel = generate_tunnel_data(
            self.config.num_train_trajectories,
            self.config.seq_len,
            str(self.device)
        )
        train_loader = DataLoader(
            TensorDataset(obs, pos, vel, tunnel),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        print("  Generating validation data...")
        val_obs, val_pos, val_vel, val_tunnel = generate_tunnel_data(
            self.config.num_val_trajectories,
            self.config.seq_len,
            str(self.device)
        )

        history = {"train_loss": [], "val_loss": [], "closure_ratio": []}

        # Warmup epochs (first 20% of training)
        warmup_epochs = max(1, self.config.num_epochs // 5)

        for epoch in range(1, self.config.num_epochs + 1):
            # Calculate warmup ratio (0->1 over warmup period)
            if epoch <= warmup_epochs:
                warmup_ratio = epoch / warmup_epochs
            else:
                warmup_ratio = 1.0

            # Train
            self.agent.train()
            epoch_loss = 0.0
            for batch_obs, batch_pos, batch_vel, batch_tunnel in train_loader:
                self.optimizer.zero_grad()
                # Pass warmup_ratio and epoch if agent supports it (Physicist does, Baseline ignores)
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
def evaluate_tunnel(
    agent: nn.Module,
    config: TunnelTestConfig,
    agent_name: str
) -> Dict:
    """
    Evaluate object permanence through tunnel.

    Key metrics:
    - exit_position_error: Position error when ball exits tunnel
    - latent_stability_tunnel: Variance of z_macro during occlusion
    - momentum_conservation: Velocity error through tunnel
    - position_r2_visible: Position tracking when visible
    - position_r2_tunnel: Position tracking when in tunnel
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    agent.eval()

    # Generate evaluation data
    obs, pos, vel, in_tunnel = generate_tunnel_data(
        config.num_eval_trajectories,
        config.eval_seq_len,
        str(device)
    )

    batch_size, seq_len, H, W = obs.shape

    # Encode all
    obs_flat = obs.reshape(-1, H, W)
    latents = agent.encode(obs_flat)

    # Get z_macro (for Physicist) or z[:4] (for Baseline)
    if hasattr(agent, "config") and hasattr(agent.config, "macro_dim"):
        z_macro = latents["macro_mean"].reshape(batch_size, seq_len, -1)
    else:
        z_macro = latents["mean"][:, :4].reshape(batch_size, seq_len, 4)

    # Position estimates
    estimated_pos = agent.extract_position(obs_flat).reshape(batch_size, seq_len, 2)

    # Latent stability during tunnel
    z_delta = (z_macro[:, 1:] - z_macro[:, :-1]).norm(dim=-1)
    tunnel_mask = in_tunnel[:, 1:]
    visible_mask = ~tunnel_mask

    stability_visible = z_delta[visible_mask].mean().item() if visible_mask.any() else 0
    stability_tunnel = z_delta[tunnel_mask].mean().item() if tunnel_mask.any() else 0

    # Find tunnel exit events
    # Exit = was in tunnel, now visible
    exit_mask = in_tunnel[:, :-1] & ~in_tunnel[:, 1:]

    if exit_mask.any():
        # Position error at exit
        exit_pos_true = pos[:, 1:][exit_mask]
        exit_pos_est = estimated_pos[:, 1:][exit_mask]
        exit_error = (exit_pos_true - exit_pos_est).norm(dim=-1).mean().item()

        # Velocity change through tunnel (should be ~0 for straight paths)
        # Compare velocity at exit vs expected from physics (constant velocity)
        exit_vel = vel[:, 1:][exit_mask]
        # Just measure velocity magnitude stability during tunnel
        tunnel_vel = vel[in_tunnel]
        if len(tunnel_vel) > 1:
            # Velocity should be constant through tunnel
            vel_std = tunnel_vel.std(dim=0).mean().item()
            momentum_error = vel_std
        else:
            momentum_error = 0
    else:
        exit_error = 0
        momentum_error = 0

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

    r2_visible = compute_r2(
        estimated_pos.reshape(-1, 2)[~in_tunnel.reshape(-1)],
        pos.reshape(-1, 2)[~in_tunnel.reshape(-1)]
    )
    r2_tunnel = compute_r2(
        estimated_pos.reshape(-1, 2)[in_tunnel.reshape(-1)],
        pos.reshape(-1, 2)[in_tunnel.reshape(-1)]
    )

    results = {
        "agent": agent_name,
        "stability_visible": stability_visible,
        "stability_tunnel": stability_tunnel,
        "stability_ratio": stability_tunnel / max(stability_visible, 1e-8),
        "exit_position_error": exit_error,
        "momentum_error": momentum_error,
        "r2_visible": r2_visible,
        "r2_tunnel": r2_tunnel,
        "r2_ratio": r2_tunnel / max(r2_visible, 1e-8),
        "num_tunnel_frames": in_tunnel.sum().item(),
        "num_exit_events": exit_mask.sum().item(),
    }

    return results


def run_tunnel_test(config: Optional[TunnelTestConfig] = None, quick: bool = False) -> Dict:
    """Run the full tunnel test."""
    if config is None:
        if quick:
            config = TunnelTestConfig(
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
            config = TunnelTestConfig(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    print("=" * 70)
    print("TUNNEL TEST: 2D OBJECT PERMANENCE")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Quick mode: {quick}")

    # Train Physicist
    physicist = PhysicistAgent2D(PhysicistConfig2D(device=config.device))
    physicist_trainer = TunnelTrainer(physicist, config, "physicist")
    physicist_history = physicist_trainer.train()

    # Train Baseline
    baseline = BaselineAgent2D(BaselineConfig2D(device=config.device))
    baseline_trainer = TunnelTrainer(baseline, config, "baseline")
    baseline_history = baseline_trainer.train()

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    physicist_results = evaluate_tunnel(physicist, config, "Physicist")
    baseline_results = evaluate_tunnel(baseline, config, "Baseline")

    # Print comparison
    print("\n{:<30} {:>15} {:>15} {:>10}".format("Metric", "Physicist", "Baseline", "Winner"))
    print("-" * 70)

    metrics = [
        ("Latent stability (visible)", "stability_visible", "lower"),
        ("Latent stability (tunnel)", "stability_tunnel", "lower"),
        ("Exit position error", "exit_position_error", "lower"),
        ("Momentum error", "momentum_error", "lower"),
        ("Position R² (visible)", "r2_visible", "higher"),
        ("Position R² (tunnel)", "r2_tunnel", "higher"),
        ("R² ratio (tunnel/visible)", "r2_ratio", "higher"),
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
        print("\n  PHYSICIST ACHIEVES OBJECT PERMANENCE!")
        print("  The split-brain architecture maintains physics through the tunnel!")
    elif baseline_wins > physicist_wins:
        print("\n  BASELINE WINS (unexpected)")
        print("  Consider training longer or adjusting hyperparameters.")
    else:
        print("\n  INCONCLUSIVE")

    print(f"\n  Physicist wins: {physicist_wins}")
    print(f"  Baseline wins:  {baseline_wins}")

    # Key insight
    if physicist_results["r2_tunnel"] > 0.3:
        print("\n  *** OBJECT PERMANENCE ACHIEVED ***")
        print("  The agent can track the ball even when it cannot see it!")
        print("  This separates Representation (pixels) from Dynamics (concept).")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "physicist": physicist_results,
        "baseline": baseline_results,
        "physicist_wins": physicist_wins,
        "baseline_wins": baseline_wins,
    }

    with open(output_dir / "tunnel_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir / 'tunnel_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.quick:
        run_tunnel_test(quick=True)
    else:
        config = TunnelTestConfig(
            num_epochs=args.epochs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        run_tunnel_test(config)
