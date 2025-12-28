#!/usr/bin/env python3
"""
Occlusion Test: Object Permanence Evaluation.

This test evaluates whether agents can maintain belief about object position
when the object passes behind an occluding bar (pixels 30-34).

Key metrics:
- Latent trajectory smoothness during occlusion
- Prediction accuracy when object re-emerges
- Position tracking R^2 during visible vs occluded phases

Expected results:
- Baseline: Latents become noisy/random during occlusion, poor re-emergence prediction
- Physicist: Smooth latent trajectory through occlusion, accurate re-emergence prediction

This is a test of Node 6 (GeomCheck) - maintaining geometric structure with
incomplete observations. If it passes, you have primitive object permanence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noisy_pulse_env import OccludedPulseEnv, OccludedPulseConfig
from .physicist_agent import PhysicistAgent1D, PhysicistConfig1D
from .baseline_agent import BaselineAgent1D, BaselineConfig1D
from .train import TrainConfig, Trainer


@dataclass
class OcclusionTestConfig:
    """Configuration for occlusion testing."""

    # Training
    num_train_trajectories: int = 500
    num_val_trajectories: int = 50
    seq_len: int = 64
    num_epochs: int = 50
    batch_size: int = 32

    # Evaluation
    num_eval_trajectories: int = 100
    eval_seq_len: int = 100

    # Output
    output_dir: str = "outputs/occlusion_test"

    # Device
    device: str = "cuda"


def generate_occluded_data(
    config: OcclusionTestConfig,
    num_trajectories: int,
    seq_len: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data from occluded environment."""
    env_config = OccludedPulseConfig(device=device)
    env = OccludedPulseEnv(env_config)
    return env.generate_batch(num_trajectories, seq_len)


class OcclusionTrainer(Trainer):
    """Trainer adapted for occluded environment."""

    def generate_data(
        self,
        num_trajectories: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training data from occluded environment."""
        obs, pos, vel, occ = generate_occluded_data(
            self.config, num_trajectories, seq_len, str(self.device)
        )
        return obs, pos, vel


def train_on_occlusion(
    config: OcclusionTestConfig
) -> Tuple[PhysicistAgent1D, BaselineAgent1D, Dict, Dict]:
    """Train both agents on occluded environment."""
    device = config.device

    # Create training config
    train_config = TrainConfig(
        num_train_trajectories=config.num_train_trajectories,
        num_val_trajectories=config.num_val_trajectories,
        seq_len=config.seq_len,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        device=device,
    )

    # Train Physicist
    print("\n" + "=" * 60)
    print("TRAINING PHYSICIST ON OCCLUDED ENVIRONMENT")
    print("=" * 60)

    physicist = PhysicistAgent1D(PhysicistConfig1D(device=device))
    physicist_trainer = OcclusionTrainer(physicist, train_config, "physicist_occluded")
    physicist_history = physicist_trainer.train()

    # Train Baseline
    print("\n" + "=" * 60)
    print("TRAINING BASELINE ON OCCLUDED ENVIRONMENT")
    print("=" * 60)

    baseline = BaselineAgent1D(BaselineConfig1D(device=device))
    baseline_trainer = OcclusionTrainer(baseline, train_config, "baseline_occluded")
    baseline_history = baseline_trainer.train()

    return physicist, baseline, physicist_history, baseline_history


@torch.no_grad()
def evaluate_occlusion(
    agent: nn.Module,
    config: OcclusionTestConfig,
    agent_name: str
) -> Dict:
    """
    Evaluate agent's object permanence.

    Metrics:
    - latent_smoothness_visible: Variance of latent changes when visible
    - latent_smoothness_occluded: Variance of latent changes when occluded
    - reemergence_accuracy: Prediction error at re-emergence
    - position_r2_visible: Position tracking when visible
    - position_r2_occluded: Position tracking when occluded
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    agent.eval()

    # Generate evaluation data
    obs, pos, vel, occ = generate_occluded_data(
        config, config.num_eval_trajectories, config.eval_seq_len, str(device)
    )

    batch_size, seq_len, obs_dim = obs.shape

    # Encode all observations
    obs_flat = obs.reshape(-1, obs_dim)

    if hasattr(agent, "encoder") and hasattr(agent.encoder, "macro_mean"):
        # Physicist
        latents = agent.encode(obs_flat)
        z = latents["macro_mean"].reshape(batch_size, seq_len, -1)
        is_physicist = True
    else:
        # Baseline
        latents = agent.encode(obs_flat)
        z = latents["mean"].reshape(batch_size, seq_len, -1)
        is_physicist = False

    # Compute latent deltas
    z_delta = (z[:, 1:] - z[:, :-1]).norm(dim=-1)  # [batch, seq-1]
    occ_mask = occ[:, 1:]  # Align with deltas

    # Latent smoothness (lower = smoother trajectory)
    visible_mask = ~occ_mask
    smoothness_visible = z_delta[visible_mask].mean().item() if visible_mask.any() else 0
    smoothness_occluded = z_delta[occ_mask].mean().item() if occ_mask.any() else 0

    # Find re-emergence events (transition from occluded to visible)
    reemergence_mask = occ[:, :-1] & ~occ[:, 1:]  # Was occluded, now visible

    # Position tracking
    estimated_pos = agent.extract_position(obs_flat).reshape(batch_size, seq_len)

    # R^2 for visible
    pos_visible = pos[~occ]
    est_visible = estimated_pos[~occ]
    if len(pos_visible) > 0:
        r2_visible = compute_r2(est_visible, pos_visible)
    else:
        r2_visible = 0.0

    # R^2 for occluded (this tests object permanence!)
    pos_occluded = pos[occ]
    est_occluded = estimated_pos[occ]
    if len(pos_occluded) > 0:
        r2_occluded = compute_r2(est_occluded, pos_occluded)
    else:
        r2_occluded = 0.0

    # Prediction at re-emergence
    if reemergence_mask.any():
        # Get latents just before and at re-emergence
        z_before_reemergence = z[:, :-1][reemergence_mask]
        z_at_reemergence = z[:, 1:][reemergence_mask]

        # Predict
        if is_physicist:
            z_pred = agent.predict_next_macro(z_before_reemergence)
        else:
            z_pred, _ = agent.predict_next(z_before_reemergence.unsqueeze(1))
            z_pred = z_pred.squeeze(1)

        reemergence_error = F.mse_loss(z_pred, z_at_reemergence).item()
    else:
        reemergence_error = 0.0

    results = {
        "agent": agent_name,
        "smoothness_visible": smoothness_visible,
        "smoothness_occluded": smoothness_occluded,
        "smoothness_ratio": smoothness_occluded / max(smoothness_visible, 1e-8),
        "r2_visible": r2_visible,
        "r2_occluded": r2_occluded,
        "r2_ratio": r2_occluded / max(r2_visible, 1e-8),
        "reemergence_error": reemergence_error,
        "num_occlusion_frames": occ.sum().item(),
        "num_reemergence_events": reemergence_mask.sum().item(),
    }

    return results


def compute_r2(estimated: torch.Tensor, true: torch.Tensor) -> float:
    """Compute R^2 using correlation."""
    x = estimated - estimated.mean()
    y = true - true.mean()
    if x.std() > 1e-8 and y.std() > 1e-8:
        correlation = (x * y).mean() / (x.std() * y.std())
        return correlation.item() ** 2
    return 0.0


@torch.no_grad()
def visualize_latent_trajectory(
    agent: nn.Module,
    config: OcclusionTestConfig,
    agent_name: str,
    num_samples: int = 3
) -> str:
    """
    Visualize latent trajectories as ASCII art.

    Returns a string visualization showing:
    - Ground truth position
    - Estimated latent (z_macro[0] for Physicist, z[0] for Baseline)
    - Occlusion markers
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    agent.eval()

    # Generate a few trajectories
    obs, pos, vel, occ = generate_occluded_data(config, num_samples, 80, str(device))

    output = []
    output.append(f"\n{'='*70}")
    output.append(f"LATENT TRAJECTORY VISUALIZATION: {agent_name}")
    output.append(f"{'='*70}")

    for traj_idx in range(num_samples):
        output.append(f"\n--- Trajectory {traj_idx + 1} ---")

        # Get latent trajectory
        obs_traj = obs[traj_idx]  # [seq_len, obs_dim]
        pos_traj = pos[traj_idx].cpu().numpy()  # [seq_len]
        occ_traj = occ[traj_idx].cpu().numpy()  # [seq_len]

        latents = agent.encode(obs_traj)
        if hasattr(agent, "encoder") and hasattr(agent.encoder, "macro_mean"):
            z_traj = latents["macro_mean"][:, 0].cpu().numpy()  # First component
            z_traj_1 = latents["macro_mean"][:, 1].cpu().numpy()  # Second component
        else:
            z_traj = latents["mean"][:, 0].cpu().numpy()
            z_traj_1 = latents["mean"][:, 1].cpu().numpy() if latents["mean"].shape[1] > 1 else z_traj

        # Normalize for visualization
        pos_norm = (pos_traj - pos_traj.min()) / (pos_traj.max() - pos_traj.min() + 1e-8)
        z_norm = (z_traj - z_traj.min()) / (z_traj.max() - z_traj.min() + 1e-8)

        # ASCII plot
        height = 10
        width = min(80, len(pos_traj))
        step = max(1, len(pos_traj) // width)

        output.append("\n  Position (true) vs Latent[0]:")
        output.append("  " + "-" * (width + 2))

        for row in range(height, -1, -1):
            line = "  |"
            for col in range(width):
                t = col * step
                pos_row = int(pos_norm[t] * height)
                z_row = int(z_norm[t] * height)

                char = " "
                if pos_row == row and z_row == row:
                    char = "X"  # Both
                elif pos_row == row:
                    char = "o"  # Position
                elif z_row == row:
                    char = "+"  # Latent

                # Mark occlusion
                if occ_traj[t] and row == 0:
                    char = "█"

                line += char
            line += "|"
            output.append(line)

        output.append("  " + "-" * (width + 2))
        output.append("  Legend: o=true position, +=latent[0], X=both, █=occluded")

        # Occlusion timeline
        occ_line = "  Occ: "
        for col in range(width):
            t = col * step
            occ_line += "█" if occ_traj[t] else "·"
        output.append(occ_line)

        # Statistics
        visible_mask = ~occ_traj
        occluded_mask = occ_traj

        if visible_mask.any():
            visible_corr = abs(compute_r2(
                torch.tensor(z_traj[visible_mask]),
                torch.tensor(pos_traj[visible_mask])
            ))
        else:
            visible_corr = 0

        if occluded_mask.any():
            occluded_corr = abs(compute_r2(
                torch.tensor(z_traj[occluded_mask]),
                torch.tensor(pos_traj[occluded_mask])
            ))
        else:
            occluded_corr = 0

        output.append(f"  Correlation (visible): {visible_corr:.4f}")
        output.append(f"  Correlation (occluded): {occluded_corr:.4f}")

    return "\n".join(output)


def run_occlusion_test(
    config: Optional[OcclusionTestConfig] = None,
    quick: bool = False
) -> Dict:
    """
    Run the full occlusion test.

    This test determines if the Physicist architecture achieves
    object permanence - the ability to track objects through occlusion.
    """
    if config is None:
        if quick:
            config = OcclusionTestConfig(
                num_train_trajectories=100,
                num_val_trajectories=20,
                seq_len=48,
                num_epochs=15,
                batch_size=16,
                num_eval_trajectories=30,
                eval_seq_len=80,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            config = OcclusionTestConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    print("=" * 70)
    print("OCCLUSION TEST: OBJECT PERMANENCE EVALUATION")
    print("=" * 70)
    print(f"\nDevice: {config.device}")
    print(f"Quick mode: {quick}")

    # Train both agents
    physicist, baseline, p_history, b_history = train_on_occlusion(config)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    physicist_results = evaluate_occlusion(physicist, config, "Physicist")
    baseline_results = evaluate_occlusion(baseline, config, "Baseline")

    # Visualize latent trajectories
    physicist_viz = visualize_latent_trajectory(physicist, config, "Physicist")
    baseline_viz = visualize_latent_trajectory(baseline, config, "Baseline")

    print(physicist_viz)
    print(baseline_viz)

    # Print comparison
    print("\n" + "=" * 70)
    print("OBJECT PERMANENCE COMPARISON")
    print("=" * 70)

    print("\n{:<30} {:>15} {:>15} {:>10}".format(
        "Metric", "Physicist", "Baseline", "Winner"
    ))
    print("-" * 70)

    metrics = [
        ("Smoothness (visible)", "smoothness_visible", "lower"),
        ("Smoothness (occluded)", "smoothness_occluded", "lower"),
        ("Smoothness ratio", "smoothness_ratio", "lower"),
        ("Position R^2 (visible)", "r2_visible", "higher"),
        ("Position R^2 (occluded)", "r2_occluded", "higher"),
        ("R^2 ratio (occ/vis)", "r2_ratio", "higher"),
        ("Re-emergence error", "reemergence_error", "lower"),
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

    print("-" * 70)

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if physicist_wins > baseline_wins:
        verdict = "PHYSICIST ACHIEVES OBJECT PERMANENCE"
        detail = "The split-brain architecture maintains physics through occlusion!"
    elif baseline_wins > physicist_wins:
        verdict = "BASELINE WINS (unexpected)"
        detail = "Consider training longer or adjusting hyperparameters."
    else:
        verdict = "INCONCLUSIVE"
        detail = "Results are too close to call."

    print(f"\n  {verdict}")
    print(f"  {detail}")
    print(f"\n  Physicist wins: {physicist_wins}")
    print(f"  Baseline wins:  {baseline_wins}")

    # Key insight
    if physicist_results["r2_occluded"] > 0.5:
        print("\n  *** KEY INSIGHT ***")
        print("  R^2 during occlusion > 0.5 means the agent KNOWS where the ball is")
        print("  even when it cannot see it. This is OBJECT PERMANENCE.")
        print("  You have implemented a primitive form of geometric reasoning!")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "physicist": physicist_results,
        "baseline": baseline_results,
        "physicist_wins": physicist_wins,
        "baseline_wins": baseline_wins,
        "verdict": verdict,
    }

    with open(output_dir / "occlusion_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir / 'occlusion_results.json'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run occlusion test")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    if args.quick:
        run_occlusion_test(quick=True)
    else:
        config = OcclusionTestConfig(
            num_epochs=args.epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        run_occlusion_test(config)
