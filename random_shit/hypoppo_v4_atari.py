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
import argparse
import time

# Register ALE environments
gym.register_envs(ale_py)

from hypoppo_v4 import (
    ScalingCertificate,
    ScalingExponentEstimator,
    SubcriticalityEnforcer,
    ScalingRegularizer,
    LayerwiseTrustEnforcer,
    SemanticScalingEstimator,
    ActorCriticCoherenceRegularizer,
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


class MLP(nn.Module):
    """
    Simple MLP for CartPole.
    
    Input: (batch, input_dim)
    Output: (batch, 64)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.feature_size = 64

    def forward(self, x):
        return self.net(x.float())


class AtariActorCritic(nn.Module):
    """
    Actor-Critic with Nature DQN CNN backbone.

    Shared convolutional features (Atari) or MLP (CartPole) -> separate actor/critic heads
    """

    def __init__(self, input_shape: tuple, action_dim: int):
        super().__init__()

        # Backbone selection
        if len(input_shape) == 3:
            # Atari: (C, H, W) -> NatureCNN
            self.backbone = NatureCNN(input_shape[0])
            feature_dim = 512
        else:
            # Vector env: (dim,) -> MLP
            self.backbone = MLP(input_shape[0])
            feature_dim = 64

        # Actor head (policy)
        self.actor_head = nn.Linear(feature_dim, action_dim)

        # Critic head (value)
        self.critic_head = nn.Linear(feature_dim, 1)

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



# FrameStack class removed - using Gymnasium wrappers instead


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
                 input_shape: tuple,
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
        self.model = AtariActorCritic(input_shape, action_dim).to(device)
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

        # HYPOSTRUCTURE: Scaling Regularization (New Mechanism)
        self.scaling_regularizer = ScalingRegularizer(
            subcritical_margin=subcritical_margin
        )

        # HYPOSTRUCTURE: Layerwise Trust (uses Adam's v for per-layer trust)
        self.layer_trust_enforcer = LayerwiseTrustEnforcer(
            optimizer=self.optimizer,
            base_trust=1.0,
            sensitivity=1.0,
        )

        # HYPOSTRUCTURE: Semantic Scaling (α_V, β_π)
        self.semantic_estimator = SemanticScalingEstimator()

        # HYPOSTRUCTURE: Coherence Regularization
        self.coherence_regularizer = ActorCriticCoherenceRegularizer()

        self.current_certificate: Optional[ScalingCertificate] = None
        self.update_count = 0

        # Performance timing for 'bells & whistles'
        self.timing = {
            'scaling_est_update': 0,
            'scaling_reg': 0,
            'adam_record': 0,
            'trust_scaling': 0,
            'semantic_record': 0,
            'total_updates': 0
        }

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            logits, value = self.model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def select_actions_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for a batch of states (vectorized)."""
        states_t = torch.from_numpy(states).to(self.device).float()

        with torch.no_grad():
            logits, values = self.model(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

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
        t_est = time.time()
        self.current_certificate = self.scaling_estimator.update()
        self.timing['scaling_est_update'] += time.time() - t_est

        # HYPOSTRUCTURE: Get enforcement parameters
        effective_clip, lr_scale = self.subcrit_enforcer.enforce(self.current_certificate)

        # Learning rate update and init timing
        self.timing['total_updates'] += 1

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

                # HYPOSTRUCTURE: Add regularization penalty
                if self.current_certificate:
                    t_reg = time.time()
                    loss_reg, penalty_coeff = self.scaling_regularizer.compute_penalty(
                        self.model, self.current_certificate
                    )
                    loss_total += loss_reg
                    self.timing['scaling_reg'] += time.time() - t_reg
                else:
                    loss_reg = 0.0
                    penalty_coeff = 0.0

                # Backward pass
                self.optimizer.zero_grad()
                loss_total.backward()

                # HYPOSTRUCTURE: Record for scaling estimation (uses Adam's v for β)
                t_record = time.time()
                self.scaling_estimator.record_from_adam(
                    self.optimizer, loss_total.item(), self.model
                )
                self.timing['adam_record'] += time.time() - t_record

                # HYPOSTRUCTURE: Apply per-layer trust scaling
                t_trust = time.time()
                self.layer_trust_enforcer.apply_trust_scaling(self.model)
                self.timing['trust_scaling'] += time.time() - t_trust

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

                # HYPOSTRUCTURE: Record semantic observations
                t_semantic = time.time()
                self.semantic_estimator.record(
                    value_mean=values.mean().item(),
                    value_var=values.var().item(),
                    bellman_var=(values - mb_targets).var().item(),
                    entropy=entropy.mean().item(),
                    kl_div=kl.item()
                )
                self.timing['semantic_record'] += time.time() - t_semantic

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
            'loss_reg': loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0.0,
            'penalty_coeff': penalty_coeff,
            'rho': self.rho,
            'effective_clip': effective_clip,
            'lr_scale': lr_scale,
        }

        if self.current_certificate:
            stats.update(self.current_certificate.to_dict())
            stats['barrier_status'] = self.current_certificate.barrier_status

        stats.update(self.subcrit_enforcer.get_stats())
        stats.update(self.layer_trust_enforcer.get_stats())

        # HYPOSTRUCTURE: Semantic scaling stats
        semantic_stats = self.semantic_estimator.estimate()
        stats.update(semantic_stats)
        stats.update(self.coherence_regularizer.get_stats())

        return stats


def evaluate(agent: HypoPPOv4Atari, env_name: str, n_episodes: int = 5, is_atari: bool = False) -> float:
    """Evaluate agent performance."""
    # Setup environment with wrappers
    if is_atari:
        env = gym.make(env_name, frameskip=1)
    else:
        env = gym.make(env_name)
    if is_atari:
        env = gym.wrappers.AtariPreprocessing(
            env, 
            noop_max=30, 
            frame_skip=4, 
            screen_size=84, 
            terminal_on_life_loss=False, 
            grayscale_obs=True, 
            grayscale_newaxis=True, 
            scale_obs=False
        )
    
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        # Transpose (84, 84, 1) -> (1, 84, 84)
        if is_atari:
            state = obs.transpose(2, 0, 1)
        else:
            state = obs

        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            # Transpose
            if is_atari:
                state = obs.transpose(2, 0, 1)
            else:
                state = obs
            
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


def train():
    """Train HypoPPO v4 with vectorized environments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cartpole', action='store_true', help='Run on CartPole-v1 for debugging')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--max-updates', type=int, default=None, help='Limit number of updates (useful for profiling)')
    args = parser.parse_args()

    env_name = "CartPole-v1" if args.cartpole else "ALE/Pong-v5"
    num_envs = args.num_envs
    
    print("=" * 70)
    print(f"HypoPPO v4: {env_name}")
    print("=" * 70)
    
    if args.cartpole:
        # Vectorized CartPole
        def make_env():
            return gym.make(env_name)
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
        input_shape = envs.single_observation_space.shape
        is_atari = False
        print(f"Backbone:  MLP (64x64) x{num_envs} envs")
    else:
        # Vectorized Atari (single for now, Atari is expensive)
        num_envs = 1
        def make_atari_env():
            env = gym.make(env_name, frameskip=1)
            env = gym.wrappers.AtariPreprocessing(
                env, noop_max=30, frame_skip=4, screen_size=84,
                terminal_on_life_loss=False, grayscale_obs=True,
                grayscale_newaxis=True, scale_obs=False
            )
            return env
        envs = gym.vector.SyncVectorEnv([make_atari_env for _ in range(num_envs)])
        input_shape = (1, 84, 84)
        is_atari = True
        print(f"Backbone: Nature CNN x{num_envs} envs")

    action_dim = envs.single_action_space.n
    print(f"Action dim: {action_dim}")
    print(f"Input shape: {input_shape}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    agent = HypoPPOv4Atari(
        input_shape=input_shape,
        action_dim=action_dim,
        lr=5e-3,  # Standard PPO LR
        base_clip=0.2,
        target_entropy=0.5,
        subcritical_margin=0.1,
        scaling_window=50,
        device=device
    )

    # OPTIMIZED HYPERPARAMETERS
    max_steps = 500000
    steps_per_env = 128  # Steps per env before update
    update_interval = steps_per_env * num_envs  # Total steps before update
    eval_interval = 25000
    batch_size = 64
    epochs = 3

    print(f"Training for {max_steps} steps...")
    print(f"Update every {update_interval} steps ({steps_per_env}/env x {num_envs} envs)")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print("-" * 70)

    # Reset all envs
    obs, _ = envs.reset()
    if is_atari:
        states = np.array([o.transpose(2, 0, 1) for o in obs])
    else:
        states = obs

    episode_rewards = np.zeros(num_envs)
    recent_rewards = deque(maxlen=100)

    agent.buffer.start_trajectory()
    
    total_steps = 0
    
    # Optional profiling
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    max_updates = args.max_updates if args.max_updates else (max_steps // update_interval)

    for update in range(1, max_updates + 1):
        # Collect steps_per_env steps from each env
        for _ in range(steps_per_env):
            # Batched action selection (MUCH FASTER)
            actions, log_probs, values = agent.select_actions_batch(states)
            
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            
            if is_atari:
                next_states = np.array([o.transpose(2, 0, 1) for o in next_obs])
            else:
                next_states = next_obs

            # Add each env's step to buffer
            for i in range(num_envs):
                agent.buffer.add_step(states[i], actions[i], rewards[i], 
                                     log_probs[i], values[i], dones[i])
                episode_rewards[i] += rewards[i]
                
                if dones[i]:
                    recent_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    agent.buffer.start_trajectory()
            
            states = next_states
            total_steps += num_envs

        # Update
        if agent.buffer.total_steps() > 0:
            stats = agent.update(batch_size=batch_size, epochs=epochs)

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0

            # Barrier status
            barrier = stats.get('barrier_status', 'unknown')
            barrier_sym = '✓' if barrier == 'clear' else ('⚠' if barrier == 'warning' else '✗')

            print(f"Step {total_steps:7d} | "
                  f"R: {avg_reward:7.1f} | "
                  f"α: {stats.get('alpha', 0):5.2f} β: {stats.get('beta', 0):5.2f} | "
                  f"αV: {stats.get('alpha_V', 0):5.2f} βπ: {stats.get('beta_pi', 0):5.2f} | "
                  f"clip: {stats.get('effective_clip', 0.2):.3f} {barrier_sym}")

        if total_steps % eval_interval < update_interval:
            eval_score = evaluate(agent, env_name, n_episodes=5, is_atari=is_atari)
            print(f"         >>> Evaluation: {eval_score:.1f}")

    envs.close()

    if args.profile:
        profiler.disable()
        # Also print detailed hypostructure timing
        if hasattr(agent, 'timing'):
            print("\n" + "=" * 70)
            print("HYPOSTRUCTURE 'BELLS & WHISTLES' TIMING:")
            print("=" * 70)
            t = agent.timing
            n = t['total_updates']
            print(f"Total updates: {n}")
            print(f"Scaling Est Update: {1000*t['scaling_est_update']/n:7.2f} ms/update")
            print(f"Scaling Reg      : {1000*t['scaling_reg']/n:7.2f} ms/update")
            print(f"Adam Record      : {1000*t['adam_record']/n:7.2f} ms/update")
            print(f"Trust Scaling    : {1000*t['trust_scaling']/n:7.2f} ms/update")
            print(f"Semantic Record  : {1000*t['semantic_record']/n:7.2f} ms/update")
            
        import pstats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\n" + "=" * 70)
        print("PROFILING RESULTS (top 20):")
        print("=" * 70)
        stats.print_stats(20)

    print("-" * 70)
    print("Final Evaluation (10 episodes):")
    final_score = evaluate(agent, env_name, n_episodes=10, is_atari=is_atari)
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
    agent, score = train()

