"""
HypoPPO v4 for Atari: Ms. Pacman with Nature DQN Architecture

Extends HypoPPO v4 with convolutional architecture from the Nature DQN paper:
- Conv1: 32 filters, 8x8 kernel, stride 4
- Conv2: 64 filters, 4x4 kernel, stride 2
- Conv3: 64 filters, 3x3 kernel, stride 1
- FC: 512 units
- Separate actor/critic heads

Plus hypostructure scaling enforcement (α, β) for adaptive trust regions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import gymnasium as gym
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

# Import the scaling components from hypoppo_v4
import sys
sys.path.insert(0, '/home/user/hypostructure')
from hypoppo_v4 import (
    ScalingCertificate,
    ScalingExponentEstimator,
    SubcriticalityEnforcer,
    Trajectory,
    TrajectoryBuffer
)


class NatureCNN(nn.Module):
    """
    Nature DQN CNN architecture for Atari.

    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 512) - feature vector
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            # Conv1: 84x84x4 -> 20x20x32
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Conv2: 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Conv3: 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate flattened size: 7x7x64 = 3136
        self.feature_size = 7 * 7 * 64

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU()
        )

    def forward(self, x):
        # Input shape: (batch, 4, 84, 84)
        # Normalize to [0, 1]
        x = x.float() / 255.0

        # Convolutional features
        x = self.conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        return x


class AtariActorCritic(nn.Module):
    """
    Actor-Critic with Nature DQN CNN backbone.

    Shared convolutional features -> separate actor/critic heads
    """

    def __init__(self, in_channels: int, action_dim: int):
        super().__init__()

        # Shared CNN backbone
        self.backbone = NatureCNN(in_channels)

        # Actor head (policy)
        self.actor_head = nn.Linear(512, action_dim)

        # Critic head (value)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, state):
        # Shared features
        features = self.backbone(state)

        # Policy logits
        logits = self.actor_head(features)

        # Value estimate
        value = self.critic_head(features)

        return logits, value

    def get_param_norm(self) -> float:
        """Get total parameter norm for scaling estimation."""
        total = 0.0
        for p in self.parameters():
            total += p.data.norm().item() ** 2
        return np.sqrt(total)


class FrameStack:
    """Stack of recent frames for Atari observation."""

    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        """Reset with initial frame."""
        # Convert to grayscale and resize to 84x84
        gray = self._preprocess(frame)
        for _ in range(self.num_frames):
            self.frames.append(gray)
        return self.get_state()

    def update(self, frame):
        """Add new frame and return stacked state."""
        gray = self._preprocess(frame)
        self.frames.append(gray)
        return self.get_state()

    def _preprocess(self, frame):
        """Convert RGB frame to 84x84 grayscale."""
        # Convert to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
        if len(frame.shape) == 3:
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = frame

        # Resize to 84x84 using simple subsampling
        import cv2
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        return resized.astype(np.uint8)

    def get_state(self):
        """Get current stacked state as (4, 84, 84)."""
        return np.stack(self.frames, axis=0)


class HypoPPOv4Atari:
    """
    HypoPPO v4 for Atari with Nature DQN CNN architecture.

    Includes:
    - Convolutional feature extraction
    - Frame stacking (4 frames)
    - Scaling exponent estimation (α, β)
    - Subcriticality enforcement
    """

    def __init__(self,
                 in_channels: int,
                 action_dim: int,
                 lr: float = 2.5e-4,
                 base_clip: float = 0.2,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 target_entropy: float = 0.5,
                 subcritical_margin: float = 0.1,
                 scaling_window: int = 100,
                 device: str = 'cpu'):

        self.device = device
        self.model = AtariActorCritic(in_channels, action_dim).to(device)
        self.base_lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        self.buffer = TrajectoryBuffer()

        # Average reward baseline
        self.rho = 0.0
        self.rho_lr = 0.01

        # PPO parameters
        self.base_clip = base_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Entropy
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # HYPOSTRUCTURE: Scaling exponent estimation
        self.scaling_estimator = ScalingExponentEstimator(
            window_size=scaling_window,
            subcritical_margin=subcritical_margin,
        )

        # HYPOSTRUCTURE: Subcriticality enforcement
        self.subcrit_enforcer = SubcriticalityEnforcer(
            base_clip=base_clip,
        )

        self.current_certificate: Optional[ScalingCertificate] = None
        self.update_count = 0

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE with average reward baseline."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        n = len(rewards)

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = (rewards[t] - self.rho +
                    self.gamma * next_val * (1 - dones[t]) - values[t])
            advantages[t] = last_gae = (delta +
                self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae)

        return advantages

    def _record_scaling_observation(self, loss: float, model: nn.Module):
        """Record observation for scaling exponent estimation."""
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)

        param_norm = model.get_param_norm()
        self.scaling_estimator.record_observation(loss, grad_norm, param_norm)

    def update(self, batch_size: int = 32, epochs: int = 4) -> dict:
        """Policy update with hypostructure scaling enforcement."""
        trajectories = self.buffer.get_trajectories()
        if not trajectories:
            return {}

        self.update_count += 1

        # Flatten trajectories
        all_states, all_actions, all_rewards = [], [], []
        all_old_log_probs, all_old_values, all_dones = [], [], []

        for traj in trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_rewards.extend(traj.rewards)
            all_old_log_probs.extend(traj.log_probs)
            all_old_values.extend(traj.values)
            all_dones.extend(traj.dones)

        # Update average reward
        self.rho = self.rho + self.rho_lr * (np.mean(all_rewards) - self.rho)

        # Convert to tensors
        states = torch.from_numpy(np.array(all_states)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        rewards = torch.FloatTensor(all_rewards).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        old_values = torch.FloatTensor(all_old_values).to(self.device)
        dones = torch.FloatTensor(all_dones).to(self.device)

        n_samples = len(states)

        # Compute advantages
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze(-1)
            advantages = self._compute_gae(rewards, values, dones, 0.0)
            targets = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # HYPOSTRUCTURE: Update scaling certificate
        self.current_certificate = self.scaling_estimator.update()

        # HYPOSTRUCTURE: Get enforcement parameters
        effective_clip, lr_scale = self.subcrit_enforcer.enforce(self.current_certificate)

        # Adjust learning rate based on criticality
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_scale

        # Training loop
        history = {
            'loss_policy': [], 'loss_value': [], 'loss_total': [],
            'entropy': [], 'kl': [], 'clip_fraction': []
        }

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_old_values = old_values[idx]
                mb_targets = targets[idx]

                # Forward pass
                logits, values = self.model(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                values = values.squeeze(-1)

                # PPO clipped objective
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - effective_clip, 1 + effective_clip) * mb_advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                clip_fraction = ((ratios < 1 - effective_clip) |
                                (ratios > 1 + effective_clip)).float().mean()

                # Value loss
                value_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -self.base_clip, self.base_clip
                )
                loss_v1 = (values - mb_targets) ** 2
                loss_v2 = (value_clipped - mb_targets) ** 2
                loss_value = 0.5 * torch.max(loss_v1, loss_v2).mean()

                # Entropy bonus
                alpha = self.log_alpha.exp()
                loss_entropy = -alpha * entropy.mean()

                # Total loss
                loss_total = loss_policy + 0.5 * loss_value + loss_entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss_total.backward()

                # HYPOSTRUCTURE: Record for scaling estimation
                self._record_scaling_observation(loss_total.item(), self.model)

                # Gradient clipping
                grad_clip = 0.5 * lr_scale
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

                # Entropy coefficient update
                loss_alpha = self.log_alpha * (entropy.mean().detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()

                # KL tracking
                with torch.no_grad():
                    kl = (mb_old_log_probs - new_log_probs).mean()

                history['loss_policy'].append(loss_policy.item())
                history['loss_value'].append(loss_value.item())
                history['loss_total'].append(loss_total.item())
                history['entropy'].append(entropy.mean().item())
                history['kl'].append(kl.item())
                history['clip_fraction'].append(clip_fraction.item())

        self.buffer.clear()

        # Compile statistics
        stats = {
            'loss_policy': np.mean(history['loss_policy']),
            'loss_value': np.mean(history['loss_value']),
            'loss_total': np.mean(history['loss_total']),
            'entropy': np.mean(history['entropy']),
            'kl': np.mean(history['kl']),
            'clip_frac': np.mean(history['clip_fraction']),
            'rho': self.rho,
            'effective_clip': effective_clip,
            'lr_scale': lr_scale,
        }

        if self.current_certificate:
            stats.update(self.current_certificate.to_dict())
            stats['barrier_status'] = self.current_certificate.barrier_status

        stats.update(self.subcrit_enforcer.get_stats())

        return stats


def evaluate(agent: HypoPPOv4Atari, env_name: str, n_episodes: int = 5) -> float:
    """Evaluate agent performance."""
    env = gym.make(env_name, frameskip=1)
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        frame_stack = FrameStack(num_frames=4)
        state = frame_stack.reset(obs)

        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            state = frame_stack.update(obs)
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


def train_mspacman():
    """Train HypoPPO v4 on Ms. Pacman."""
    print("=" * 70)
    print("HypoPPO v4 for Atari: Ms. Pacman")
    print("=" * 70)
    print()
    print("ARCHITECTURE: Nature DQN CNN")
    print("  • Conv1: 32 filters, 8x8 kernel, stride 4")
    print("  • Conv2: 64 filters, 4x4 kernel, stride 2")
    print("  • Conv3: 64 filters, 3x3 kernel, stride 1")
    print("  • FC: 512 units")
    print("  • Separate actor/critic heads")
    print()
    print("HYPOSTRUCTURE: Scaling enforcement (α, β)")
    print("  • Empirical estimation of scaling exponents")
    print("  • Adaptive trust regions based on criticality")
    print()

    env_name = "ALE/MsPacman-v5"
    env = gym.make(env_name, frameskip=1)

    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"Action dim: {action_dim}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    agent = HypoPPOv4Atari(
        in_channels=4,
        action_dim=action_dim,
        lr=2.5e-4,
        base_clip=0.2,
        target_entropy=0.5,
        subcritical_margin=0.1,
        scaling_window=100,
        device=device
    )

    max_steps = 1000000
    update_interval = 2048
    eval_interval = 50000

    print(f"Training for {max_steps} steps...")
    print(f"Update every {update_interval} steps")
    print("-" * 70)

    obs, _ = env.reset()
    frame_stack = FrameStack(num_frames=4)
    state = frame_stack.reset(obs)

    episode_reward = 0
    recent_rewards = deque(maxlen=20)

    agent.buffer.start_trajectory()

    for step in range(1, max_steps + 1):
        action, log_prob, value = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = frame_stack.update(obs)

        agent.buffer.add_step(state, action, reward, log_prob, value, done)
        episode_reward += reward
        state = next_state

        if done:
            recent_rewards.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            agent.buffer.start_trajectory()

        if step % update_interval == 0 and agent.buffer.total_steps() > 0:
            stats = agent.update(batch_size=32, epochs=4)

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0

            # Barrier status
            barrier = stats.get('barrier_status', 'unknown')
            if barrier == 'clear':
                barrier_sym = '✓'
            elif barrier == 'warning':
                barrier_sym = '⚠'
            else:
                barrier_sym = '✗'

            print(f"Step {step:7d} | "
                  f"R: {avg_reward:7.1f} | "
                  f"α: {stats.get('alpha', 0):5.2f} | "
                  f"β: {stats.get('beta', 0):5.2f} | "
                  f"α-β: {stats.get('criticality', 0):+5.2f} {barrier_sym} | "
                  f"clip: {stats.get('effective_clip', 0.2):.3f}")

            if step % 50000 == 0:
                print(f"         Subcritical: {stats.get('is_subcritical', False)} | "
                      f"Confidence: {stats.get('confidence', 0):.2f} | "
                      f"LR scale: {stats.get('lr_scale', 1.0):.2f}")

        if step % eval_interval == 0:
            eval_score = evaluate(agent, env_name, n_episodes=5)
            print(f"         >>> Evaluation: {eval_score:.1f}")

    env.close()

    print("-" * 70)
    print("Final Evaluation (10 episodes):")
    final_score = evaluate(agent, env_name, n_episodes=10)
    print(f"Average score: {final_score:.1f}")

    if agent.current_certificate:
        cert = agent.current_certificate
        print()
        print("Final Scaling State:")
        print(f"  α (height): {cert.alpha:.3f}")
        print(f"  β (dissipation): {cert.beta:.3f}")
        print(f"  Criticality (α - β): {cert.criticality:+.3f}")
        print(f"  Subcritical: {cert.is_subcritical}")
        print(f"  Barrier: {cert.barrier_status}")

    return agent, final_score


if __name__ == "__main__":
    agent, score = train_mspacman()
