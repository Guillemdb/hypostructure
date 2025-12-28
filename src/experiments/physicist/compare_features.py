#!/usr/bin/env python3
"""
Compare Physicist configurations:
1. Baseline (standard VAE)
2. Physicist (default - backward compatible)
3. Physicist+ (new features: charted encoder, BRST, VICReg, topology)

Run with:
    python -m experiments.physicist.compare_features --quick
    python -m experiments.physicist.compare_features --epochs 50
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .envs_2d import NoisyTunnelEnv, TunnelConfig, ChameleonEnv, ChameleonConfig
from .agents_2d import (
    PhysicistAgent2D, PhysicistConfig2D,
    BaselineAgent2D, BaselineConfig2D
)


@dataclass
class CompareConfig:
    """Configuration for comparison experiment."""

    # Data
    num_train_trajectories: int = 300
    num_val_trajectories: int = 50
    seq_len: int = 64

    # Training
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-3

    # Evaluation
    num_eval_trajectories: int = 100

    # Output
    output_dir: str = "outputs/compare_features"

    # Device
    device: str = "cuda"


def count_parameters(agent) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in agent.parameters() if p.requires_grad)


def get_agent_configs(device: str, match_parameters: bool = True) -> Dict[str, tuple]:
    """
    Return agent configurations for comparison.

    Args:
        device: Device to use
        match_parameters: If True, adjust Physicist+ to have similar param count

    Returns dict of {name: (agent_class, config)}
    """
    configs = {
        # Standard VAE baseline
        "Baseline": (
            BaselineAgent2D,
            BaselineConfig2D(device=device)
        ),

        # Physicist with default config (backward compatible)
        "Physicist": (
            PhysicistAgent2D,
            PhysicistConfig2D(device=device)
        ),
    }

    if match_parameters:
        # To match ~1.9M parameters with 3 charts, we need to reduce hidden_channels
        # Default: hidden_channels=32 gives ~3.8M params with 3 charts
        # Reduced: hidden_channels=20 gives ~1.9M params with 3 charts
        configs["Physicist+"] = (
            PhysicistAgent2D,
            PhysicistConfig2D(
                device=device,
                # Reduced to match parameter count
                hidden_channels=20,         # Reduced from 32
                # Multi-chart routing
                num_charts=3,
                use_brst=True,
                # VICReg (decorrelated, non-collapsed representations)
                lambda_vicreg_var=1.0,
                lambda_vicreg_cov=0.5,
                # Topology (chart organization)
                lambda_entropy=0.5,
                lambda_balance=10.0,
                lambda_separation=1.0,
                chart_separation_dist=4.0,
                # BRST (near-orthogonal transforms)
                lambda_brst=0.01,
                # Warmup
                new_loss_warmup_epochs=10,
            )
        )
    else:
        # Original config without parameter matching
        configs["Physicist+"] = (
            PhysicistAgent2D,
            PhysicistConfig2D(
                device=device,
                num_charts=3,
                use_brst=True,
                lambda_vicreg_var=1.0,
                lambda_vicreg_cov=0.5,
                lambda_entropy=0.5,
                lambda_balance=10.0,
                lambda_separation=1.0,
                chart_separation_dist=4.0,
                lambda_brst=0.01,
                new_loss_warmup_epochs=10,
            )
        )

    return configs


def generate_tunnel_data(num_traj: int, seq_len: int, device: str):
    """Generate tunnel environment data."""
    config = TunnelConfig(device=device)
    env = NoisyTunnelEnv(config)
    return env.generate_batch(num_traj, seq_len)


def generate_chameleon_data(num_traj: int, seq_len: int, device: str):
    """Generate chameleon environment data."""
    config = ChameleonConfig(device=device)
    env = ChameleonEnv(config)
    return env.generate_batch(num_traj, seq_len)


def train_agent(
    agent,
    train_loader: DataLoader,
    val_obs: torch.Tensor,
    config: CompareConfig,
    agent_name: str
) -> Dict:
    """Train an agent and return history."""
    optimizer = optim.AdamW(agent.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    history = {"train_loss": [], "val_loss": [], "closure_ratio": []}
    warmup_epochs = max(1, config.num_epochs // 5)

    for epoch in range(1, config.num_epochs + 1):
        warmup_ratio = min(1.0, epoch / warmup_epochs)

        # Train
        agent.train()
        epoch_loss = 0.0
        for (batch_obs,) in train_loader:
            optimizer.zero_grad()

            # Pass epoch for new loss warmup
            if hasattr(agent, 'config') and hasattr(agent.config, 'macro_dim'):
                losses = agent.compute_loss(batch_obs, training=True, warmup_ratio=warmup_ratio, epoch=epoch)
            else:
                losses = agent.compute_loss(batch_obs, training=True)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses["total"].item()

        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        # Validate every 5 epochs
        if epoch % 5 == 0:
            agent.eval()
            with torch.no_grad():
                val_losses = agent.compute_loss(val_obs, training=False)
                val_loss = val_losses["total"].item()
                ratio = agent.compute_closure_ratio(val_obs)

            history["val_loss"].append(val_loss)
            history["closure_ratio"].append(ratio)

            # Show extra losses for Physicist+
            extra = ""
            if "vicreg_var" in val_losses:
                extra = f" | var={val_losses['vicreg_var'].item():.3f}"
            if "entropy" in val_losses:
                extra += f" ent={val_losses['entropy'].item():.3f}"
            if "brst" in val_losses:
                extra += f" brst={val_losses['brst'].item():.1f}"

            print(f"    Epoch {epoch:3d}: loss={epoch_loss:.4f} val={val_loss:.4f} closure={ratio:.4f}{extra}")

        scheduler.step()

    return history


@torch.no_grad()
def evaluate_agent(agent, obs: torch.Tensor, pos: torch.Tensor, in_tunnel: torch.Tensor) -> Dict:
    """Evaluate agent on tunnel task."""
    agent.eval()
    batch_size, seq_len, H, W = obs.shape

    # Encode
    obs_flat = obs.reshape(-1, H, W)
    latents = agent.encode(obs_flat)

    # Get z_macro
    if hasattr(agent, "config") and hasattr(agent.config, "macro_dim"):
        z_macro = latents["macro_mean"].reshape(batch_size, seq_len, -1)
    else:
        z_macro = latents["mean"][:, :4].reshape(batch_size, seq_len, 4)

    # Position estimates
    estimated_pos = agent.extract_position(obs_flat).reshape(batch_size, seq_len, 2)

    # Latent stability
    z_delta = (z_macro[:, 1:] - z_macro[:, :-1]).norm(dim=-1)
    tunnel_mask = in_tunnel[:, 1:]
    visible_mask = ~tunnel_mask

    stability_visible = z_delta[visible_mask].mean().item() if visible_mask.any() else 0
    stability_tunnel = z_delta[tunnel_mask].mean().item() if tunnel_mask.any() else 0

    # Position error
    pos_error = (estimated_pos - pos).norm(dim=-1).mean().item()
    pos_error_visible = (estimated_pos - pos).norm(dim=-1)[~in_tunnel].mean().item()
    pos_error_tunnel = (estimated_pos - pos).norm(dim=-1)[in_tunnel].mean().item() if in_tunnel.any() else 0

    return {
        "stability_visible": stability_visible,
        "stability_tunnel": stability_tunnel,
        "pos_error": pos_error,
        "pos_error_visible": pos_error_visible,
        "pos_error_tunnel": pos_error_tunnel,
    }


def run_comparison(config: Optional[CompareConfig] = None, match_parameters: bool = True):
    """Run the full comparison experiment."""
    if config is None:
        config = CompareConfig(device="cuda" if torch.cuda.is_available() else "cpu")

    device = config.device
    print("=" * 70)
    print("FEATURE COMPARISON: Baseline vs Physicist vs Physicist+")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Train trajectories: {config.num_train_trajectories}")
    print(f"Match parameters: {match_parameters}")

    # Generate data
    print("\nGenerating tunnel data...")
    train_obs, train_pos, train_vel, train_tunnel = generate_tunnel_data(
        config.num_train_trajectories, config.seq_len, device
    )
    val_obs, val_pos, val_vel, val_tunnel = generate_tunnel_data(
        config.num_val_trajectories, config.seq_len, device
    )
    eval_obs, eval_pos, eval_vel, eval_tunnel = generate_tunnel_data(
        config.num_eval_trajectories, config.seq_len, device
    )

    train_loader = DataLoader(
        TensorDataset(train_obs),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Train and evaluate each agent
    agent_configs = get_agent_configs(device, match_parameters=match_parameters)
    results = {}

    for name, (agent_class, agent_config) in agent_configs.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name}")
        print("=" * 70)

        # Show config details for Physicist+
        if name == "Physicist+":
            print(f"  num_charts={agent_config.num_charts}, use_brst={agent_config.use_brst}")
            print(f"  lambda_vicreg_var={agent_config.lambda_vicreg_var}, lambda_vicreg_cov={agent_config.lambda_vicreg_cov}")
            print(f"  lambda_entropy={agent_config.lambda_entropy}, lambda_balance={agent_config.lambda_balance}")
            print(f"  lambda_brst={agent_config.lambda_brst}")

        agent = agent_class(agent_config)
        num_params = sum(p.numel() for p in agent.parameters())
        print(f"  Parameters: {num_params:,}")

        history = train_agent(agent, train_loader, val_obs, config, name)
        eval_results = evaluate_agent(agent, eval_obs, eval_pos, eval_tunnel)

        results[name] = {
            "history": history,
            "eval": eval_results,
            "num_params": num_params,
        }

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:>12} {:>12} {:>12}".format(
        "Metric", "Baseline", "Physicist", "Physicist+"
    ))
    print("-" * 70)

    metrics = [
        ("Parameters", "num_params", "info"),
        ("Final train loss", lambda r: r["history"]["train_loss"][-1], "lower"),
        ("Final val loss", lambda r: r["history"]["val_loss"][-1], "lower"),
        ("Final closure ratio", lambda r: r["history"]["closure_ratio"][-1], "higher"),
        ("Latent stability (vis)", lambda r: r["eval"]["stability_visible"], "lower"),
        ("Latent stability (tun)", lambda r: r["eval"]["stability_tunnel"], "lower"),
        ("Position error", lambda r: r["eval"]["pos_error"], "lower"),
        ("Pos error (visible)", lambda r: r["eval"]["pos_error_visible"], "lower"),
        ("Pos error (tunnel)", lambda r: r["eval"]["pos_error_tunnel"], "lower"),
    ]

    for metric_name, key, direction in metrics:
        values = {}
        for name in ["Baseline", "Physicist", "Physicist+"]:
            if callable(key):
                values[name] = key(results[name])
            else:
                values[name] = results[name][key]

        # Format values
        if metric_name == "Parameters":
            formatted = [f"{v:,}" for v in values.values()]
        else:
            formatted = [f"{v:.4f}" for v in values.values()]

        # Find winner
        if direction == "lower":
            winner_val = min(values.values())
            winners = [k for k, v in values.items() if v == winner_val]
        elif direction == "higher":
            winner_val = max(values.values())
            winners = [k for k, v in values.items() if v == winner_val]
        else:
            winners = []

        # Add marker for winner
        row = []
        for i, name in enumerate(["Baseline", "Physicist", "Physicist+"]):
            val = formatted[i]
            if name in winners and direction != "info":
                val = f"{val} *"
            row.append(val)

        print("{:<25} {:>12} {:>12} {:>12}".format(metric_name, *row))

    print("\n* = best")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count wins
    wins = {"Baseline": 0, "Physicist": 0, "Physicist+": 0}
    for metric_name, key, direction in metrics:
        if direction == "info":
            continue
        values = {}
        for name in ["Baseline", "Physicist", "Physicist+"]:
            if callable(key):
                values[name] = key(results[name])
            else:
                values[name] = results[name][key]

        if direction == "lower":
            winner = min(values, key=values.get)
        else:
            winner = max(values, key=values.get)
        wins[winner] += 1

    print(f"\nMetrics won:")
    for name, count in wins.items():
        print(f"  {name}: {count}")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "eval": data["eval"],
            "num_params": data["num_params"],
            "final_train_loss": data["history"]["train_loss"][-1],
            "final_val_loss": data["history"]["val_loss"][-1],
            "final_closure_ratio": data["history"]["closure_ratio"][-1],
        }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'comparison_results.json'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Physicist configurations")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer epochs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--trajectories", type=int, default=300, help="Training trajectories")
    parser.add_argument("--no-match-params", action="store_true",
                        help="Don't match parameter counts (Physicist+ will have more params)")
    args = parser.parse_args()

    if args.quick:
        config = CompareConfig(
            num_train_trajectories=50,
            num_val_trajectories=20,
            seq_len=32,
            num_epochs=10,
            batch_size=8,
            num_eval_trajectories=30,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        config = CompareConfig(
            num_epochs=args.epochs,
            num_train_trajectories=args.trajectories,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    run_comparison(config, match_parameters=not args.no_match_params)
