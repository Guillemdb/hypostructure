#!/usr/bin/env python3
"""
Main entry point for the Physicist experiment.

This script runs the complete experiment:
1. Trains both Physicist and Baseline agents
2. Evaluates both agents
3. Generates comparison report

Usage:
    python -m experiments.physicist.run_experiment
    python -m experiments.physicist.run_experiment --quick  # Quick test
    python -m experiments.physicist.run_experiment --epochs 200 --output results/
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import time

import torch

from .noisy_pulse_env import NoisyPulseEnv, NoisyPulseConfig
from .physicist_agent import PhysicistAgent1D, PhysicistConfig1D
from .baseline_agent import BaselineAgent1D, BaselineConfig1D
from .train import TrainConfig, Trainer
from .evaluate import EvalConfig, compare_agents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Physicist vs Baseline experiment"
    )

    # Quick test mode
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with reduced settings"
    )

    # Training settings
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )

    # Data settings
    parser.add_argument(
        "--train-trajectories", type=int, default=1000,
        help="Number of training trajectories (default: 1000)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
        help="Sequence length (default: 64)"
    )

    # Output settings
    parser.add_argument(
        "--output", type=str, default="outputs/physicist",
        help="Output directory (default: outputs/physicist)"
    )

    # Device settings
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage"
    )

    # Mode selection
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train, skip evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate (requires checkpoints)"
    )
    parser.add_argument(
        "--physicist-checkpoint", type=str,
        help="Path to Physicist checkpoint for eval-only mode"
    )
    parser.add_argument(
        "--baseline-checkpoint", type=str,
        help="Path to Baseline checkpoint for eval-only mode"
    )

    return parser.parse_args()


def create_train_config(args) -> TrainConfig:
    """Create training config from arguments."""
    if args.quick:
        return TrainConfig(
            num_train_trajectories=100,
            num_val_trajectories=20,
            seq_len=32,
            batch_size=16,
            num_epochs=10,
            eval_interval=2,
            save_interval=10,
            log_interval=5,
            output_dir=args.output,
            device="cpu" if args.cpu else args.device,
        )

    return TrainConfig(
        num_train_trajectories=args.train_trajectories,
        num_val_trajectories=args.train_trajectories // 10,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        eval_interval=5,
        save_interval=20,
        log_interval=10,
        output_dir=args.output,
        device="cpu" if args.cpu else args.device,
    )


def create_eval_config(args) -> EvalConfig:
    """Create evaluation config from arguments."""
    if args.quick:
        return EvalConfig(
            num_trajectories=20,
            seq_len=32,
            prediction_horizons=[1, 5, 10],
            device="cpu" if args.cpu else args.device,
        )

    return EvalConfig(
        num_trajectories=100,
        seq_len=100,
        prediction_horizons=[1, 5, 10, 20, 50],
        device="cpu" if args.cpu else args.device,
    )


def train_agents(train_config: TrainConfig):
    """Train both Physicist and Baseline agents."""
    device = train_config.device

    # Create agents
    physicist_config = PhysicistConfig1D(device=device)
    physicist = PhysicistAgent1D(physicist_config)

    baseline_config = BaselineConfig1D(device=device)
    baseline = BaselineAgent1D(baseline_config)

    # Print model info
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    physicist_params = sum(p.numel() for p in physicist.parameters())
    baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"Physicist parameters: {physicist_params:,}")
    print(f"Baseline parameters:  {baseline_params:,}")
    print(f"Parameter ratio: {physicist_params / baseline_params:.2f}x")

    # Train Physicist
    print("\n" + "=" * 60)
    print("TRAINING PHYSICIST")
    print("=" * 60)

    physicist_trainer = Trainer(physicist, train_config, agent_name="physicist")
    physicist_history = physicist_trainer.train()

    # Train Baseline
    print("\n" + "=" * 60)
    print("TRAINING BASELINE")
    print("=" * 60)

    baseline_trainer = Trainer(baseline, train_config, agent_name="baseline")
    baseline_history = baseline_trainer.train()

    return physicist, baseline, physicist_history, baseline_history


def load_agents(args, device: str):
    """Load agents from checkpoints."""
    print("Loading agents from checkpoints...")

    # Load Physicist
    physicist_config = PhysicistConfig1D(device=device)
    physicist = PhysicistAgent1D(physicist_config)
    checkpoint = torch.load(args.physicist_checkpoint, map_location=device)
    physicist.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded Physicist from: {args.physicist_checkpoint}")

    # Load Baseline
    baseline_config = BaselineConfig1D(device=device)
    baseline = BaselineAgent1D(baseline_config)
    checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
    baseline.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded Baseline from: {args.baseline_checkpoint}")

    return physicist, baseline


def save_results(results: dict, output_dir: str):
    """Save evaluation results."""
    output_path = Path(output_dir) / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/torch values to Python types
    def convert(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_serializable = convert(results)

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_final_summary(results: dict):
    """Print final summary of experiment."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    physicist = results["physicist"]
    baseline = results["baseline"]

    # Key metrics comparison
    key_metrics = [
        ("Reconstruction (clean)", "mse_clean", "lower"),
        ("Position tracking R^2", "position_r2", "higher"),
        ("Closure ratio", "closure_ratio", "higher"),
    ]

    print("\n{:<30} {:>12} {:>12} {:>8}".format(
        "Metric", "Physicist", "Baseline", "Winner"
    ))
    print("-" * 65)

    physicist_wins = 0
    baseline_wins = 0

    for name, key, direction in key_metrics:
        p_val = physicist.get(key, 0)
        b_val = baseline.get(key, 0)

        if direction == "higher":
            if p_val > b_val:
                winner = "Phys"
                physicist_wins += 1
            elif b_val > p_val:
                winner = "Base"
                baseline_wins += 1
            else:
                winner = "Tie"
        else:
            if p_val < b_val:
                winner = "Phys"
                physicist_wins += 1
            elif b_val < p_val:
                winner = "Base"
                baseline_wins += 1
            else:
                winner = "Tie"

        print("{:<30} {:>12.4f} {:>12.4f} {:>8}".format(name, p_val, b_val, winner))

    print("-" * 65)
    print(f"\nPhysicist wins: {physicist_wins}, Baseline wins: {baseline_wins}")

    if physicist_wins > baseline_wins:
        print("\n>> PHYSICIST ARCHITECTURE WINS! <<")
        print("The split-brain structure successfully separates physics from noise.")
    elif baseline_wins > physicist_wins:
        print("\n>> BASELINE WINS (unexpected) <<")
        print("Consider checking hyperparameters or training longer.")
    else:
        print("\n>> TIE <<")
        print("Results are inconclusive - consider more epochs or data.")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("PHYSICIST EXPERIMENT")
    print("1D Noisy Bouncing Pulse - Split-Brain Architecture Test")
    print("=" * 60)

    # Determine device
    device = "cpu" if args.cpu else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\nDevice: {device}")
    print(f"Quick mode: {args.quick}")

    start_time = time.time()

    if args.eval_only:
        # Evaluation only mode
        if not args.physicist_checkpoint or not args.baseline_checkpoint:
            raise ValueError(
                "Must provide --physicist-checkpoint and --baseline-checkpoint for eval-only mode"
            )

        physicist, baseline = load_agents(args, device)
        eval_config = create_eval_config(args)
        results = compare_agents(physicist, baseline, eval_config)

    else:
        # Training mode
        train_config = create_train_config(args)
        print(f"\nTraining config:")
        print(f"  Epochs: {train_config.num_epochs}")
        print(f"  Trajectories: {train_config.num_train_trajectories}")
        print(f"  Sequence length: {train_config.seq_len}")
        print(f"  Batch size: {train_config.batch_size}")

        physicist, baseline, physicist_history, baseline_history = train_agents(train_config)

        if not args.train_only:
            # Evaluation
            print("\n" + "=" * 60)
            print("EVALUATION")
            print("=" * 60)

            eval_config = create_eval_config(args)
            results = compare_agents(physicist, baseline, eval_config)

            # Add training history
            results["physicist"]["training_history"] = physicist_history
            results["baseline"]["training_history"] = baseline_history

            # Save results
            save_results(results, args.output)

            # Print summary
            print_final_summary(results)
        else:
            results = {
                "physicist": {"training_history": physicist_history},
                "baseline": {"training_history": baseline_history},
            }

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
