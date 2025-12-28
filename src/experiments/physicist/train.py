"""
Training loop for Physicist experiment.

Trains both Physicist and Baseline agents on the Noisy Pulse environment
and tracks diagnostic metrics including closure ratio.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .noisy_pulse_env import NoisyPulseEnv, NoisyPulseConfig
from .physicist_agent import PhysicistAgent1D, PhysicistConfig1D
from .baseline_agent import BaselineAgent1D, BaselineConfig1D


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data generation
    num_train_trajectories: int = 1000
    num_val_trajectories: int = 100
    seq_len: int = 64

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Logging
    log_interval: int = 10           # Log every N batches
    eval_interval: int = 5           # Evaluate every N epochs
    save_interval: int = 20          # Save checkpoint every N epochs

    # Paths
    output_dir: str = "outputs/physicist"

    # Device
    device: str = "cuda"


class Trainer:
    """
    Trainer for Physicist and Baseline agents.

    Handles:
    - Data generation from environment
    - Training loop with loss tracking
    - Validation with closure ratio computation
    - Checkpointing and logging
    """

    def __init__(
        self,
        agent: nn.Module,
        config: TrainConfig,
        agent_name: str = "agent"
    ):
        self.agent = agent
        self.config = config
        self.agent_name = agent_name
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Move agent to device
        self.agent.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        # Setup output directory
        self.output_dir = Path(config.output_dir) / agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "closure_ratio": [],
            "epoch_time": [],
        }

    def generate_data(
        self,
        num_trajectories: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training data from environment."""
        env_config = NoisyPulseConfig(device=str(self.device))
        env = NoisyPulseEnv(env_config)

        obs, pos, vel = env.generate_batch(num_trajectories, seq_len)
        return obs, pos, vel

    def create_dataloader(
        self,
        obs: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor,
        shuffle: bool = True
    ) -> DataLoader:
        """Create dataloader from generated data."""
        dataset = TensorDataset(obs, pos, vel)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=True
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.agent.train()

        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        num_batches = 0

        for batch_idx, (obs, pos, vel) in enumerate(dataloader):
            obs = obs.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            losses = self.agent.compute_loss(obs, training=True)

            # Backward pass
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += losses["total"].item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item()

            num_batches += 1

            # Logging
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx}/{len(dataloader)}: "
                    f"loss={losses['total'].item():.4f}"
                )

        # Average losses
        avg_losses = {k: v / num_batches for k, v in loss_components.items()}
        avg_losses["total"] = total_loss / num_batches

        return avg_losses

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader
    ) -> Tuple[Dict[str, float], float]:
        """Validate and compute metrics."""
        self.agent.eval()

        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        closure_ratios = []
        num_batches = 0

        for obs, pos, vel in dataloader:
            obs = obs.to(self.device)

            # Compute losses
            losses = self.agent.compute_loss(obs, training=False)

            total_loss += losses["total"].item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item()

            # Compute closure ratio
            ratio = self.agent.compute_closure_ratio(obs)
            closure_ratios.append(ratio)

            num_batches += 1

        # Average
        avg_losses = {k: v / num_batches for k, v in loss_components.items()}
        avg_losses["total"] = total_loss / num_batches
        avg_closure = sum(closure_ratios) / len(closure_ratios)

        return avg_losses, avg_closure

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"]

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            Training history
        """
        print(f"Training {self.agent_name}...")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")

        # Generate data
        print("Generating training data...")
        train_obs, train_pos, train_vel = self.generate_data(
            self.config.num_train_trajectories,
            self.config.seq_len
        )
        train_loader = self.create_dataloader(train_obs, train_pos, train_vel)

        print("Generating validation data...")
        val_obs, val_pos, val_vel = self.generate_data(
            self.config.num_val_trajectories,
            self.config.seq_len
        )
        val_loader = self.create_dataloader(val_obs, val_pos, val_vel, shuffle=False)

        print(f"  Train: {train_obs.shape}")
        print(f"  Val: {val_obs.shape}")

        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")

            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_losses["total"])

            # Validate
            if epoch % self.config.eval_interval == 0:
                val_losses, closure_ratio = self.validate(val_loader)
                self.history["val_loss"].append(val_losses["total"])
                self.history["closure_ratio"].append(closure_ratio)

                print(f"  Train loss: {train_losses['total']:.4f}")
                print(f"  Val loss: {val_losses['total']:.4f}")
                print(f"  Closure ratio: {closure_ratio:.4f}")

            # Update learning rate
            self.scheduler.step()

            # Track time
            epoch_time = time.time() - start_time
            self.history["epoch_time"].append(epoch_time)
            print(f"  Time: {epoch_time:.1f}s")

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

        # Save final checkpoint
        self.save_checkpoint(self.config.num_epochs)

        # Save history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nSaved history: {history_path}")

        return self.history


def train_physicist(config: Optional[TrainConfig] = None) -> Tuple[PhysicistAgent1D, Dict]:
    """Train Physicist agent."""
    config = config or TrainConfig()

    agent_config = PhysicistConfig1D(device=config.device)
    agent = PhysicistAgent1D(agent_config)

    trainer = Trainer(agent, config, agent_name="physicist")
    history = trainer.train()

    return agent, history


def train_baseline(config: Optional[TrainConfig] = None) -> Tuple[BaselineAgent1D, Dict]:
    """Train Baseline agent."""
    config = config or TrainConfig()

    agent_config = BaselineConfig1D(device=config.device)
    agent = BaselineAgent1D(agent_config)

    trainer = Trainer(agent, config, agent_name="baseline")
    history = trainer.train()

    return agent, history


def train_both(config: Optional[TrainConfig] = None) -> Dict:
    """Train both agents and compare."""
    config = config or TrainConfig()

    print("=" * 60)
    print("PHYSICIST VS BASELINE EXPERIMENT")
    print("=" * 60)

    # Train physicist
    physicist, physicist_history = train_physicist(config)

    # Train baseline
    baseline, baseline_history = train_baseline(config)

    # Compare final metrics
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    physicist_closure = physicist_history["closure_ratio"][-1] if physicist_history["closure_ratio"] else 0
    baseline_closure = baseline_history["closure_ratio"][-1] if baseline_history["closure_ratio"] else 0

    print(f"Physicist closure ratio: {physicist_closure:.4f}")
    print(f"Baseline closure ratio:  {baseline_closure:.4f}")

    if physicist_closure > baseline_closure:
        print(">> Physicist wins!")
    else:
        print(">> Baseline wins (unexpected)")

    return {
        "physicist": physicist_history,
        "baseline": baseline_history,
    }


if __name__ == "__main__":
    # Quick test with small config
    config = TrainConfig(
        num_train_trajectories=100,
        num_val_trajectories=20,
        seq_len=32,
        num_epochs=10,
        eval_interval=2,
        save_interval=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    results = train_both(config)
