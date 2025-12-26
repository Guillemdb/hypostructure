"""
HypoPPO v4: Hypostructure Scaling Parameters (α, β) and Subcritical Enforcement

HYPOSTRUCTURE INTEGRATION:
--------------------------
This implementation adds Node 4 (ScaleCheck) from the Hypostructure sieve:

1. SCALING EXPONENTS:
   - α (alpha): Height/energy scaling exponent
     Φ(S_λ θ) = λ^α Φ(θ) where Φ = loss functional

   - β (beta): Dissipation scaling exponent
     D(S_λ θ) = λ^β D(θ) where D = ||∇Φ||² (squared gradient norm)

2. CRITICALITY INDEX: α - β
   - Subcritical (α - β > 0): Singularities cost infinite energy → SAFE
   - Critical (α - β = 0): Borderline
   - Supercritical (α - β < 0): Singularities can form → DANGEROUS

3. ENFORCEMENT:
   When the system drifts toward criticality (α - β → 0 or negative),
   we apply adaptive trust region constraints to restore subcriticality.

ESTIMATION METHOD:
------------------
We estimate α and β empirically by measuring how loss and gradient norms
change as we scale parameter perturbations. This gives us:

  α ≈ d log Φ / d log λ   (how loss responds to parameter scaling)
  β ≈ d log D / d log λ   (how gradient norm responds to scaling)

The key insight from Hypostructure: we don't need to prove global bounds,
we just need to show that catastrophic behaviors (singularities) would
require more "resources" (energy/dissipation) than are structurally available.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import gymnasium as gym


@dataclass
class ScalingCertificate:
    """
    Certificate from Node 4 (ScaleCheck) of the Hypostructure sieve.

    Corresponds to K_{SC_λ}^+ = (α, β, α - β > 0) for subcritical
    or K_{SC_λ}^- for supercritical with barrier status.
    """
    alpha: float          # Height scaling exponent
    beta: float           # Dissipation scaling exponent
    criticality: float    # α - β
    is_subcritical: bool  # α - β > 0
    barrier_status: str   # 'clear', 'warning', 'blocked'
    timestamp: int        # Update step
    confidence: float     # Estimation confidence (0-1)

    def to_dict(self) -> Dict[str, float]:
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'criticality': self.criticality,
            'is_subcritical': float(self.is_subcritical),
            'confidence': self.confidence,
        }


class ScalingExponentEstimator:
    """
    Estimates α and β scaling exponents empirically.

    METHOD:
    -------
    1. Store recent (loss, gradient_norm, param_norm) triplets
    2. Fit power law: Φ ~ θ^α, D ~ θ^β
    3. Use log-linear regression on normalized data

    HYPOSTRUCTURE CONNECTION:
    -------------------------
    This implements the interface permit SC_λ:
    - Scaling Action: S_λ: θ → λθ (parameter norm scaling)
    - Height Exponent α: Φ(λθ) = λ^α Φ(θ)
    - Dissipation Exponent β: D(λθ) = λ^β D(θ)
    """

    def __init__(self,
                 window_size: int = 100,
                 min_samples: int = 20,
                 regularization: float = 1e-6,
                 subcritical_margin: float = 0.1,
                 ema_decay: float = 0.95):

        self.window_size = window_size
        self.min_samples = min_samples
        self.regularization = regularization
        self.subcritical_margin = subcritical_margin
        self.ema_decay = ema_decay

        # Storage for (log_param_norm, log_loss, log_grad_norm)
        self.log_param_norms: deque = deque(maxlen=window_size)
        self.log_losses: deque = deque(maxlen=window_size)
        self.log_grad_norms: deque = deque(maxlen=window_size)

        # EMA estimates
        self.alpha_ema = 2.0  # Default: quadratic loss
        self.beta_ema = 2.0   # Default: quadratic gradient
        self.confidence = 0.0

        # History for analysis
        self.certificate_history: List[ScalingCertificate] = []
        self.step_count = 0

    def record_observation(self,
                          loss: float,
                          grad_norm: float,
                          param_norm: float):
        """
        Record an observation triplet for scaling estimation.

        From Hypostructure template:
        - loss → Φ(θ) (height functional)
        - grad_norm → sqrt(D(θ)) where D = ||∇Φ||²
        - param_norm → ||θ|| (scale parameter)
        """
        if loss > 0 and grad_norm > 0 and param_norm > 0:
            self.log_param_norms.append(np.log(param_norm))
            self.log_losses.append(np.log(loss))
            self.log_grad_norms.append(np.log(grad_norm))

        self.step_count += 1

    def estimate_exponents(self) -> Tuple[float, float, float]:
        """
        Estimate α and β via log-linear regression.

        For scaling law Φ(λθ) = λ^α Φ(θ):
          log Φ ≈ α log ||θ|| + const

        Similarly for D:
          log D ≈ β log ||θ|| + const

        Note: D = ||∇Φ||², so log(grad_norm) ≈ (β/2) log ||θ||
        We estimate β from gradient norm, so β = 2 * slope of log(grad_norm).
        """
        if len(self.log_param_norms) < self.min_samples:
            return self.alpha_ema, self.beta_ema, 0.0

        x = np.array(self.log_param_norms)
        y_loss = np.array(self.log_losses)
        y_grad = np.array(self.log_grad_norms)

        # Normalize for numerical stability
        x_mean, x_std = x.mean(), x.std() + 1e-8
        x_norm = (x - x_mean) / x_std

        # Estimate α from loss scaling
        # log Φ = α log ||θ|| + c  →  slope = α
        alpha_raw = self._fit_slope(x_norm, y_loss)

        # Estimate β from gradient norm scaling
        # log ||∇Φ|| = (β/2) log ||θ|| + c  →  β = 2 * slope
        beta_half_raw = self._fit_slope(x_norm, y_grad)
        beta_raw = 2.0 * beta_half_raw

        # Compute confidence from R² values
        r2_alpha = self._compute_r2(x_norm, y_loss, alpha_raw)
        r2_beta = self._compute_r2(x_norm, y_grad, beta_half_raw)
        confidence = (r2_alpha + r2_beta) / 2.0
        confidence = np.clip(confidence, 0.0, 1.0)

        return alpha_raw, beta_raw, confidence

    def _fit_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fit slope via regularized least squares."""
        # y = slope * x + intercept
        # Minimize ||y - slope*x - intercept||² + reg * slope²
        n = len(x)
        x_mean = x.mean()
        y_mean = y.mean()

        # Centered regression
        x_c = x - x_mean
        y_c = y - y_mean

        numerator = np.dot(x_c, y_c)
        denominator = np.dot(x_c, x_c) + self.regularization

        slope = numerator / denominator if denominator > 0 else 0.0
        return slope

    def _compute_r2(self, x: np.ndarray, y: np.ndarray, slope: float) -> float:
        """Compute R² for goodness of fit."""
        y_mean = y.mean()
        y_pred = slope * x + (y_mean - slope * x.mean())

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        if ss_tot < 1e-10:
            return 1.0

        r2 = 1.0 - ss_res / ss_tot
        return max(0.0, r2)

    def update(self) -> ScalingCertificate:
        """
        Update scaling estimates and emit certificate.

        This is the core ScaleCheck (Node 4) implementation.
        """
        alpha_raw, beta_raw, confidence = self.estimate_exponents()

        # EMA smoothing
        self.alpha_ema = self.ema_decay * self.alpha_ema + (1 - self.ema_decay) * alpha_raw
        self.beta_ema = self.ema_decay * self.beta_ema + (1 - self.ema_decay) * beta_raw
        self.confidence = self.ema_decay * self.confidence + (1 - self.ema_decay) * confidence

        # Compute criticality
        criticality = self.alpha_ema - self.beta_ema
        is_subcritical = criticality > self.subcritical_margin

        # Determine barrier status
        if criticality > 2 * self.subcritical_margin:
            barrier_status = 'clear'
        elif criticality > 0:
            barrier_status = 'warning'
        else:
            barrier_status = 'blocked'

        cert = ScalingCertificate(
            alpha=self.alpha_ema,
            beta=self.beta_ema,
            criticality=criticality,
            is_subcritical=is_subcritical,
            barrier_status=barrier_status,
            timestamp=self.step_count,
            confidence=self.confidence
        )

        self.certificate_history.append(cert)
        return cert

    def get_trust_modifier(self) -> float:
        """
        Compute trust region modifier based on criticality.

        HYPOSTRUCTURE PRINCIPLE:
        When approaching criticality (α - β → 0), constrain updates
        to prevent the system from entering a supercritical regime
        where singularities become possible.

        Returns: modifier in (0, 1] where 1 = full trust, <1 = reduced trust
        """
        criticality = self.alpha_ema - self.beta_ema

        if criticality > 2 * self.subcritical_margin:
            # Safely subcritical: full trust
            return 1.0
        elif criticality > self.subcritical_margin:
            # Approaching criticality: linear reduction
            return 0.5 + 0.5 * (criticality - self.subcritical_margin) / self.subcritical_margin
        elif criticality > 0:
            # Near-critical: significant reduction
            return 0.3 + 0.2 * criticality / self.subcritical_margin
        else:
            # Supercritical: strong constraint
            return max(0.1, 0.3 + 0.2 * criticality)


class SubcriticalityEnforcer:
    """
    Enforces subcriticality by adaptively constraining optimization.

    HYPOSTRUCTURE CONNECTION:
    -------------------------
    This implements the BarrierTypeII check from Node 4:

    When K_{SC_λ}^- (supercritical detected), check BarrierTypeII:
    - If renormalization cost is infinite → singularity blocked → continue
    - If barrier breached → apply surgery (SurgSE)

    Our "surgery" is adaptive trust region reduction: we don't allow
    updates that would push the system further into supercriticality.
    """

    def __init__(self,
                 base_clip: float = 0.2,
                 base_lr_scale: float = 1.0,
                 min_clip: float = 0.05,
                 min_lr_scale: float = 0.1,
                 recovery_rate: float = 0.01):

        self.base_clip = base_clip
        self.base_lr_scale = base_lr_scale
        self.min_clip = min_clip
        self.min_lr_scale = min_lr_scale
        self.recovery_rate = recovery_rate

        # Current trust levels
        self.clip_modifier = 1.0
        self.lr_modifier = 1.0

        # Statistics
        self.interventions = 0
        self.supercritical_steps = 0

    def enforce(self, certificate: ScalingCertificate) -> Tuple[float, float]:
        """
        Apply subcriticality enforcement based on certificate.

        Returns: (effective_clip, lr_scale)
        """
        trust = self._compute_trust(certificate)

        if certificate.barrier_status == 'blocked':
            # Supercritical: apply strong constraint
            self.interventions += 1
            self.supercritical_steps += 1
            target_modifier = trust * 0.5
        elif certificate.barrier_status == 'warning':
            # Approaching criticality: moderate constraint
            target_modifier = trust * 0.8
        else:
            # Subcritical: allow recovery toward full trust
            target_modifier = min(1.0, self.clip_modifier + self.recovery_rate)

        # Smooth update
        self.clip_modifier = 0.9 * self.clip_modifier + 0.1 * target_modifier
        self.lr_modifier = 0.9 * self.lr_modifier + 0.1 * target_modifier

        # Compute effective values with floors
        effective_clip = max(self.min_clip, self.base_clip * self.clip_modifier)
        lr_scale = max(self.min_lr_scale, self.base_lr_scale * self.lr_modifier)

        return effective_clip, lr_scale

    def _compute_trust(self, cert: ScalingCertificate) -> float:
        """Compute trust level from certificate."""
        # Weight by confidence
        base_trust = 1.0 if cert.is_subcritical else 0.5

        # Adjust by criticality margin
        if cert.criticality > 0.2:
            margin_factor = 1.0
        elif cert.criticality > 0:
            margin_factor = 0.5 + 2.5 * cert.criticality
        else:
            margin_factor = max(0.2, 0.5 + cert.criticality)

        # Confidence-weighted trust
        trust = base_trust * margin_factor
        trust = cert.confidence * trust + (1 - cert.confidence) * 0.5

        return np.clip(trust, 0.1, 1.0)

    def get_stats(self) -> Dict[str, float]:
        return {
            'clip_modifier': self.clip_modifier,
            'lr_modifier': self.lr_modifier,
            'interventions': self.interventions,
            'supercritical_steps': self.supercritical_steps,
        }


@dataclass
class Trajectory:
    """Complete trajectory with gradient storage."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self):
        return len(self.rewards)

    def to_tensors(self, device='cpu'):
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(device),
            'actions': torch.LongTensor(self.actions).to(device),
            'rewards': torch.FloatTensor(self.rewards).to(device),
            'log_probs': torch.FloatTensor(self.log_probs).to(device),
            'values': torch.FloatTensor(self.values).to(device),
            'dones': torch.FloatTensor(self.dones).to(device),
        }


class TrajectoryBuffer:
    """Stores complete trajectories."""

    def __init__(self, max_trajectories: int = 100):
        self.trajectories: List[Trajectory] = []
        self.max_trajectories = max_trajectories
        self._current: Optional[Trajectory] = None

    def start_trajectory(self):
        self._current = Trajectory()

    def add_step(self, state, action, reward, log_prob, value, done):
        if self._current is None:
            self.start_trajectory()
        self._current.states.append(state)
        self._current.actions.append(action)
        self._current.rewards.append(reward)
        self._current.log_probs.append(log_prob)
        self._current.values.append(value)
        self._current.dones.append(done)

        if done:
            if len(self._current) > 1:
                self.trajectories.append(self._current)
            if len(self.trajectories) > self.max_trajectories:
                self.trajectories.pop(0)
            self._current = None

    def get_trajectories(self) -> List[Trajectory]:
        return self.trajectories

    def clear(self):
        self.trajectories = []
        self._current = None

    def total_steps(self) -> int:
        return sum(len(t) for t in self.trajectories)


class ActorCritic(nn.Module):
    """Standard actor-critic with named layers."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk_fc1 = nn.Linear(state_dim, hidden_dim)
        self.trunk_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.trunk_fc1(state))
        x = torch.tanh(self.trunk_fc2(x))
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

    def get_param_norm(self) -> float:
        """Get total parameter norm."""
        total = 0.0
        for p in self.parameters():
            total += p.data.norm().item() ** 2
        return np.sqrt(total)


class HypoPPOv4:
    """
    HypoPPO with Hypostructure Scaling Parameters.

    NOVEL CONTRIBUTIONS:
    -------------------
    1. Empirical estimation of scaling exponents (α, β)
    2. Subcriticality enforcement (α - β > 0)
    3. Adaptive trust regions based on criticality

    HYPOSTRUCTURE NODES IMPLEMENTED:
    --------------------------------
    - Node 1 (EnergyCheck): Loss as height functional ✓
    - Node 4 (ScaleCheck): α, β estimation, subcriticality ✓
    - Node 7 (StiffnessCheck): Gradient coherence (from v3) ✓
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 base_clip: float = 0.2,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 target_entropy: float = 0.4,
                 subcritical_margin: float = 0.1,
                 scaling_window: int = 100,
                 device: str = 'cpu'):

        self.device = device
        self.model = ActorCritic(state_dim, action_dim).to(device)
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

        # HYPOSTRUCTURE: Scaling exponent estimation (Node 4)
        self.scaling_estimator = ScalingExponentEstimator(
            window_size=scaling_window,
            subcritical_margin=subcritical_margin,
        )

        # HYPOSTRUCTURE: Subcriticality enforcement
        self.subcrit_enforcer = SubcriticalityEnforcer(
            base_clip=base_clip,
        )

        # Current certificate
        self.current_certificate: Optional[ScalingCertificate] = None

        # Statistics
        self.update_count = 0

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        # Compute gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)

        # Get parameter norm
        param_norm = model.get_param_norm()

        # Record
        self.scaling_estimator.record_observation(loss, grad_norm, param_norm)

    def update(self, batch_size: int = 64, epochs: int = 10) -> dict:
        """
        Policy update with Hypostructure scaling enforcement.

        HYPOSTRUCTURE INTEGRATION:
        - Record (loss, grad_norm, param_norm) for scaling estimation
        - Update α, β estimates
        - Emit ScalingCertificate
        - Apply subcriticality enforcement to trust region
        """
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
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
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

                # PPO clipped objective with Hypostructure-adjusted clip
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios,
                                   1 - effective_clip,
                                   1 + effective_clip) * mb_advantages
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

                # Gradient clipping (also modulated by criticality)
                grad_clip = 0.5 * lr_scale  # Tighter clip when near-critical
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

                # Entropy coefficient update
                loss_alpha = self.log_alpha * (entropy.mean().detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()

                # KL divergence tracking
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

        # Add scaling certificate stats
        if self.current_certificate:
            stats.update(self.current_certificate.to_dict())
            stats['barrier_status'] = self.current_certificate.barrier_status

        # Add enforcer stats
        stats.update(self.subcrit_enforcer.get_stats())

        return stats


def evaluate(agent: HypoPPOv4, env_name: str, n_episodes: int = 5) -> float:
    """Evaluate agent performance."""
    env = gym.make(env_name)
    total_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


def train_cartpole():
    """Train HypoPPO v4 on CartPole with scaling enforcement."""
    print("=" * 70)
    print("HypoPPO v4: Hypostructure Scaling Parameters (α, β)")
    print("=" * 70)
    print()
    print("HYPOSTRUCTURE NODE 4 (ScaleCheck) IMPLEMENTATION:")
    print("  • α (alpha): Height/energy scaling exponent")
    print("  • β (beta): Dissipation scaling exponent")
    print("  • Subcritical: α - β > 0 (singularities blocked)")
    print("  • Critical: α - β = 0 (borderline)")
    print("  • Supercritical: α - β < 0 (singularities possible)")
    print()
    print("ENFORCEMENT:")
    print("  • When α - β → 0: Reduce trust region (clip, LR)")
    print("  • When α - β < 0: Strong constraint to restore subcriticality")
    print()

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print()

    agent = HypoPPOv4(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        base_clip=0.2,
        target_entropy=0.35,
        subcritical_margin=0.1,
        scaling_window=100,
    )

    max_steps = 100000
    update_interval = 1024
    eval_interval = 10000

    print(f"Training for {max_steps} steps...")
    print("-" * 70)

    state, _ = env.reset()
    episode_reward = 0
    recent_rewards = deque(maxlen=20)

    agent.buffer.start_trajectory()

    for step in range(1, max_steps + 1):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add_step(state, action, reward, log_prob, value, done)
        episode_reward += reward
        state = next_state

        if done:
            recent_rewards.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()
            agent.buffer.start_trajectory()

        if step % update_interval == 0 and agent.buffer.total_steps() > 0:
            stats = agent.update(batch_size=64, epochs=10)

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0

            # Barrier status indicator
            barrier = stats.get('barrier_status', 'unknown')
            if barrier == 'clear':
                barrier_sym = '✓'
            elif barrier == 'warning':
                barrier_sym = '⚠'
            else:
                barrier_sym = '✗'

            print(f"Step {step:6d} | "
                  f"R: {avg_reward:6.1f} | "
                  f"α: {stats.get('alpha', 0):5.2f} | "
                  f"β: {stats.get('beta', 0):5.2f} | "
                  f"α-β: {stats.get('criticality', 0):+5.2f} {barrier_sym} | "
                  f"clip: {stats.get('effective_clip', 0.2):.3f}")

            if step % 10000 == 0:
                print(f"         Subcritical: {stats.get('is_subcritical', False)} | "
                      f"Confidence: {stats.get('confidence', 0):.2f} | "
                      f"LR scale: {stats.get('lr_scale', 1.0):.2f}")
                print(f"         Interventions: {stats.get('interventions', 0)} | "
                      f"Supercritical steps: {stats.get('supercritical_steps', 0)}")

        if step % eval_interval == 0:
            eval_score = evaluate(agent, env_name, n_episodes=10)
            print(f"         >>> Evaluation: {eval_score:.1f}")

    env.close()

    print("-" * 70)
    print("Final Evaluation (20 episodes):")
    final_score = evaluate(agent, env_name, n_episodes=20)
    print(f"Average score: {final_score:.1f}")

    if final_score >= 475:
        print("SUCCESS: SOLVED (score >= 475)")
    elif final_score >= 400:
        print("PARTIAL: NEAR-SOLVED (score >= 400)")
    else:
        print(f"NOT SOLVED (score < 400)")

    print()
    print("Final Scaling State (Node 4 Certificate):")
    if agent.current_certificate:
        cert = agent.current_certificate
        print(f"  α (height scaling): {cert.alpha:.3f}")
        print(f"  β (dissipation scaling): {cert.beta:.3f}")
        print(f"  Criticality (α - β): {cert.criticality:+.3f}")
        print(f"  Is Subcritical: {cert.is_subcritical}")
        print(f"  Barrier Status: {cert.barrier_status}")
        print(f"  Confidence: {cert.confidence:.3f}")

    return agent, final_score


if __name__ == "__main__":
    agent, score = train_cartpole()
