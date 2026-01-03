"""
Hypo-PPO: PPO baseline aligned with CleanRL, plus optional Hypostructure losses.

Baseline: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
Regularizers (optional):
- Critic geometry regularization (stiffness + eikonal) on raw observations
- Actor smoothness (Zeno KL) between consecutive policies
- Riemannian variants (covariant dissipation + Lyapunov) using state-space metric
- Optional lightweight dynamics head for policy-gradient coupling
"""

from dataclasses import dataclass
import argparse
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

try:
    import gymnasium as gym
except ImportError as exc:
    raise ImportError("gymnasium is required to run hypo_ppo.py") from exc


@dataclass
class HypoPPOConfig:
    # Environment
    env_id: str = "Pendulum-v1"
    num_envs: int = 8
    total_timesteps: int = 1_000_000

    # PPO hyperparameters (CleanRL defaults)
    learning_rate: float = 3e-4
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Model
    hidden_dim: int = 64
    dynamics_hidden: int = 64
    dynamics_predict_delta: bool = True
    use_dynamics_head: bool = False
    lambda_dynamics: float = 0.0

    # Hypostructure losses (disabled by default)
    lambda_stiffness: float = 0.0
    lambda_eikonal: float = 0.0
    lambda_zeno: float = 0.0
    stiffness_epsilon: float = 0.1
    eikonal_target: float = 1.0
    # Riemannian variants (disabled by default)
    lambda_covariant: float = 0.0
    lambda_riemannian_lyapunov: float = 0.0
    riemannian_alpha: float = 0.1
    riemannian_metric: str = "grad_rms"  # grad_rms | obs_var | policy_fisher (adam_scalar alias)
    riemannian_eps: float = 1e-6
    riemannian_metric_clip: float = 1e3
    riemannian_value_floor: float = 1.0
    riemannian_use_model: bool = False
    # HJB Correspondence (disabled by default)
    lambda_hjb: float = 0.0  # Weight for HJB defect loss
    hjb_effort_weight: float = 0.01  # Weight for action cost in HJB

    # Wrappers (CleanRL defaults)
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Logging
    log_interval: int = 10

    # Misc
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cuda"

    # Computed in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, gamma, config: HypoPPOConfig):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if config.normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -config.clip_obs, config.clip_obs),
                observation_space=None,
            )
        if config.normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -config.clip_reward, config.clip_reward))
        return env

    return thunk


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class HypoPPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        use_dynamics_head: bool = False,
        dynamics_hidden: int = 64,
        dynamics_predict_delta: bool = True,
    ):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.use_dynamics_head = use_dynamics_head
        self.dynamics_predict_delta = dynamics_predict_delta
        if use_dynamics_head:
            self.dynamics = nn.Sequential(
                layer_init(nn.Linear(obs_dim + action_dim, dynamics_hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(dynamics_hidden, obs_dim), std=0.01),
            )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x), dist

    def predict_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamics_head:
            raise RuntimeError("Dynamics head is disabled. Enable use_dynamics_head to call predict_next().")
        inputs = torch.cat([obs, action], dim=-1)
        delta = self.dynamics(inputs)
        if self.dynamics_predict_delta:
            return obs + delta
        return delta


def _compute_critic_geometry_loss(
    agent: HypoPPOAgent,
    obs: torch.Tensor,
    stiffness_epsilon: float,
    eikonal_target: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs = obs.detach().clone().requires_grad_(True)
    value = agent.get_value(obs)
    grad = torch.autograd.grad(value.sum(), obs, create_graph=True)[0]
    grad_norm = grad.norm(dim=-1)

    loss_stiff = torch.relu(stiffness_epsilon - grad_norm).pow(2).mean()
    loss_eik = (grad_norm - eikonal_target).pow(2).mean()
    return loss_stiff, loss_eik


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _compute_metric_inverse(
    grad_v: torch.Tensor,
    agent: HypoPPOAgent,
    obs: torch.Tensor,
    mode: str,
    eps: float,
    clip_max: Optional[float],
) -> torch.Tensor:
    obs_dim = grad_v.shape[1]
    device = grad_v.device

    if mode == "grad_rms":
        metric = grad_v.pow(2).mean(dim=0).sqrt()
        metric_inv = 1.0 / (metric + eps)
        if clip_max:
            metric_inv = metric_inv.clamp(max=clip_max)
        return metric_inv

    if mode in ("obs_var", "adam_scalar"):
        obs_detached = obs.detach()
        var = obs_detached.var(dim=0, unbiased=False)
        metric_inv = 1.0 / (var + eps)
        if clip_max:
            metric_inv = metric_inv.clamp(max=clip_max)
        return metric_inv

    if mode == "policy_fisher":
        obs_fisher = obs.detach().clone().requires_grad_(True)
        action_mean = agent.actor_mean(obs_fisher)
        action_logstd = agent.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        grad_log_pi = torch.autograd.grad(log_prob.sum(), obs_fisher, create_graph=False)[0]
        fisher_diag = grad_log_pi.pow(2).mean(dim=0)
        metric_inv = 1.0 / (fisher_diag + eps)
        if clip_max:
            metric_inv = metric_inv.clamp(max=clip_max)
        return metric_inv

    return torch.ones(obs_dim, device=device)


def _compute_riemannian_losses(
    agent: HypoPPOAgent,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    metric_mode: str,
    alpha: float,
    eps: float,
    value_floor: float,
    metric_clip: float,
    mask: Optional[torch.Tensor],
    use_model: bool = False,
    policy_action: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Riemannian losses: Covariant Dissipation and Lyapunov Decay.

    CRITICAL DISTINCTION (Anti-Mixing Rule #2):
    This function computes the METRIC-WEIGHTED inner product:

        ⟨dV, f⟩_G = G^{ij} (∂_i V) f_j

    This is NOT the Lie derivative L_f V = ∂_i V · f^i (which is metric-independent).

    Physical interpretation:
    - Lie derivative (see compute_hjb_loss): Rate of value change along trajectory (for HJB)
    - Covariant inner product (this function): Penalizes movement in high-curvature
      (high-G) regions, even if value is decreasing. Acts as a "trust region" in
      information geometry.

    The covariant gate ensures the agent takes small steps where the metric is large
    (high uncertainty / high curvature), providing geometric safety beyond what the
    scalar Lie derivative can capture.

    Returns:
        covariant_loss: max(0, ⟨∇V, δs⟩_G)^2 - penalizes positive metric-weighted flow
        lyap_loss: Lyapunov decay violation with relative scaling
    """
    obs = obs.detach().clone().requires_grad_(True)
    value = agent.get_value(obs).view(-1)
    grad_v = torch.autograd.grad(value.sum(), obs, create_graph=True)[0]

    metric_inv = _compute_metric_inverse(
        grad_v,
        agent,
        obs,
        mode=metric_mode,
        eps=eps,
        clip_max=metric_clip,
    )

    obs_detached = obs.detach()
    if use_model:
        if policy_action is None:
            raise ValueError("policy_action must be provided when use_model=True.")
        pred_next = agent.predict_next(obs_detached, policy_action)
        delta_s = pred_next - obs_detached
    else:
        delta_s = next_obs - obs_detached
    vdot = (grad_v * delta_s * metric_inv).sum(dim=-1)

    covariant_violation = torch.relu(vdot)
    covariant_loss = _masked_mean(covariant_violation, mask)

    v_abs = value.abs().clamp_min(value_floor)
    rel_vdot = vdot / v_abs
    violation = torch.relu(rel_vdot + alpha)
    lyap_loss = _masked_mean(violation.pow(2), mask)

    return covariant_loss, lyap_loss


def compute_hjb_loss(
    agent: HypoPPOAgent,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    effort_weight: float = 0.01,
) -> torch.Tensor:
    """
    Compute HJB-Defect loss for value function learning.

    The Hamilton-Jacobi-Bellman equation (metric-INDEPENDENT Lie derivative):

        L_f V + D(z,a) = -R(z,a)

    Where:
        L_f V = dV(f) = ∂_i V · f^i = ∇V · f  (NO metric G)
        D(z,a) = control effort (action cost)
        R(z,a) = reward (negative potential flux)

    CRITICAL: The Lie derivative is metric-independent. This is the natural
    pairing dV(f) between the 1-form dV and the vector field f. The metric G
    appears in the COVARIANT gate (trust region), NOT here.

    Anti-Mixing Rule #2: "NO Metric in Lie Derivative"

    Dimensional check: [∇V · f] = (Energy/Length)(Length/Time) = Power ✓

    All terms have units of Power. Rewards are energy flux, not points.

    Args:
        agent: The PPO agent with critic
        obs: Current observations z_t, shape (batch, obs_dim)
        actions: Actions a_t, shape (batch, act_dim)
        rewards: Environmental rewards R(z,a), shape (batch,)
        next_obs: Next observations z_{t+1}, shape (batch, obs_dim)
        dones: Terminal flags, shape (batch,)
        effort_weight: Weight for action cost term

    Returns:
        Scalar HJB defect loss
    """
    # Compute Lie derivative: L_f V = ∇V · (z_{t+1} - z_t)
    # Note: NO metric G appears here - this is the key distinction
    obs_grad = obs.detach().clone().requires_grad_(True)
    v_now = agent.get_value(obs_grad).squeeze(-1)
    grad_v = torch.autograd.grad(v_now.sum(), obs_grad, create_graph=True)[0]

    # Dynamics vector field: f ≈ (z_{t+1} - z_t) / dt
    # We use discrete difference; dt=1 implicitly
    delta_z = next_obs - obs
    lie_derivative = (grad_v * delta_z).sum(dim=-1)  # NO metric here!

    # Mask terminal states (dynamics undefined at boundaries)
    non_terminal = (1 - dones.float())
    lie_derivative = lie_derivative * non_terminal

    # Control effort D(z,a) — quadratic action cost
    effort = effort_weight * actions.pow(2).sum(dim=-1)

    # HJB Defect: L_f V + D - R = 0
    # At optimality, the Lie derivative balances effort and reward:
    #   L_f V = R - D  (value increases toward high reward, minus effort)
    hjb_defect = lie_derivative + effort - rewards

    return hjb_defect.pow(2).mean()


def train(config: HypoPPOConfig) -> None:
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, i, config.gamma, config) for i in range(config.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))

    use_dynamics = config.use_dynamics_head or config.lambda_dynamics > 0 or config.riemannian_use_model
    agent = HypoPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        use_dynamics_head=use_dynamics,
        dynamics_hidden=config.dynamics_hidden,
        dynamics_predict_delta=config.dynamics_predict_delta,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    next_obs_storage = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    next_dones_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)

    prev_mask_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)
    prev_mean_storage = torch.zeros((config.num_steps, config.num_envs, action_dim)).to(device)
    prev_std_storage = torch.ones((config.num_steps, config.num_envs, action_dim)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    recent_returns = []

    for iteration in range(1, config.num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        prev_mask = torch.zeros(config.num_envs, device=device)
        prev_mean = torch.zeros((config.num_envs, action_dim), device=device)
        prev_std = torch.ones((config.num_envs, action_dim), device=device)

        for step in range(0, config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            prev_mask_storage[step] = prev_mask
            prev_mean_storage[step] = prev_mean
            prev_std_storage[step] = prev_std

            with torch.no_grad():
                action, logprob, _, value, dist = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                curr_mean = dist.mean
                curr_std = dist.stddev

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_obs_storage[step] = next_obs
            next_dones_storage[step] = next_done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        recent_returns.append(info["episode"]["r"])
                        if len(recent_returns) > 100:
                            recent_returns = recent_returns[-100:]
            elif "episode" in infos:
                episode_info = infos["episode"]
                episode_mask = infos.get("_episode", None)
                if episode_mask is None:
                    episode_mask = np.ones_like(episode_info["r"], dtype=bool)
                for idx, done_flag in enumerate(episode_mask):
                    if done_flag:
                        recent_returns.append(episode_info["r"][idx])
                if len(recent_returns) > 100:
                    recent_returns = recent_returns[-100:]

            prev_mask = 1.0 - next_done
            prev_mean = curr_mean
            prev_std = curr_std

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_next_obs = next_obs_storage.reshape((-1,) + envs.single_observation_space.shape)
        b_next_done = next_dones_storage.reshape(-1)
        b_prev_mask = prev_mask_storage.reshape(-1)
        b_prev_mean = prev_mean_storage.reshape(-1, action_dim)
        b_prev_std = prev_std_storage.reshape(-1, action_dim)

        b_inds = np.arange(config.batch_size)
        clipfracs = []

        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        stiff_loss = torch.tensor(0.0, device=device)
        eik_loss = torch.tensor(0.0, device=device)
        zeno_loss = torch.tensor(0.0, device=device)
        covariant_loss = torch.tensor(0.0, device=device)
        rlyap_loss = torch.tensor(0.0, device=device)
        dyn_loss = torch.tensor(0.0, device=device)
        hjb_loss = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)

        for _ in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, dist = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                if config.lambda_stiffness > 0 or config.lambda_eikonal > 0:
                    stiff_loss, eik_loss = _compute_critic_geometry_loss(
                        agent,
                        b_obs[mb_inds],
                        stiffness_epsilon=config.stiffness_epsilon,
                        eikonal_target=config.eikonal_target,
                    )
                    loss = loss + config.lambda_stiffness * stiff_loss + config.lambda_eikonal * eik_loss

                if config.lambda_zeno > 0:
                    prev_dist = Normal(b_prev_mean[mb_inds], b_prev_std[mb_inds])
                    kl = kl_divergence(dist, prev_dist).sum(dim=-1)
                    zeno_loss = _masked_mean(kl, b_prev_mask[mb_inds])
                    loss = loss + config.lambda_zeno * zeno_loss

                if config.lambda_dynamics > 0:
                    pred_next = agent.predict_next(b_obs[mb_inds], b_actions[mb_inds])
                    dyn_error = (pred_next - b_next_obs[mb_inds]).pow(2).mean(dim=-1)
                    dyn_loss = _masked_mean(dyn_error, 1.0 - b_next_done[mb_inds])
                    loss = loss + config.lambda_dynamics * dyn_loss

                if config.lambda_covariant > 0 or config.lambda_riemannian_lyapunov > 0:
                    mb_next_obs = b_next_obs[mb_inds]
                    mb_mask = 1.0 - b_next_done[mb_inds]
                    covariant_loss, rlyap_loss = _compute_riemannian_losses(
                        agent,
                        b_obs[mb_inds],
                        mb_next_obs,
                        metric_mode=config.riemannian_metric,
                        alpha=config.riemannian_alpha,
                        eps=config.riemannian_eps,
                        value_floor=config.riemannian_value_floor,
                        metric_clip=config.riemannian_metric_clip,
                        mask=mb_mask,
                        use_model=config.riemannian_use_model,
                        policy_action=dist.mean if config.riemannian_use_model else None,
                    )
                    if config.lambda_covariant > 0:
                        loss = loss + config.lambda_covariant * covariant_loss
                    if config.lambda_riemannian_lyapunov > 0:
                        loss = loss + config.lambda_riemannian_lyapunov * rlyap_loss

                # HJB Correspondence: Lie derivative (metric-independent)
                if config.lambda_hjb > 0:
                    hjb_loss = compute_hjb_loss(
                        agent,
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                        b_rewards[mb_inds],
                        b_next_obs[mb_inds],
                        b_next_done[mb_inds],
                        effort_weight=config.hjb_effort_weight,
                    )
                    loss = loss + config.lambda_hjb * hjb_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        if iteration % config.log_interval == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_reward = float(np.mean(recent_returns)) if recent_returns else 0.0
            clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0
            print(
                f"Update {iteration}/{config.num_iterations} | Steps: {global_step:,} | SPS: {sps} | "
                f"Avg Reward: {avg_reward:.2f} | ClipFrac: {clipfrac:.3f}"
            )
            print(
                f"  PPO: pg={pg_loss.item():.4f} vf={v_loss.item():.4f} ent={entropy_loss.item():.4f}"
            )
            if config.lambda_stiffness > 0 or config.lambda_eikonal > 0:
                print(f"  Geometry: stiff={stiff_loss.item():.4f} eik={eik_loss.item():.4f}")
            if config.lambda_zeno > 0:
                print(f"  Zeno: {zeno_loss.item():.4f}")
            if config.lambda_dynamics > 0:
                print(f"  Dynamics: {dyn_loss.item():.4f}")
            if config.lambda_covariant > 0 or config.lambda_riemannian_lyapunov > 0:
                print(
                    f"  Riemannian: cov={covariant_loss.item():.4f} lyap={rlyap_loss.item():.4f}"
                )
            if config.lambda_hjb > 0:
                print(f"  HJB: {hjb_loss.item():.4f}")

    envs.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hypo-PPO (CleanRL baseline + optional Hypostructure losses)")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dynamics_hidden", type=int, default=64)
    parser.add_argument(
        "--dynamics_predict_delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict delta state instead of absolute next state",
    )
    parser.add_argument(
        "--use_dynamics_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable lightweight dynamics head",
    )
    parser.add_argument("--lambda_dynamics", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument(
        "--anneal_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly anneal learning rate",
    )
    parser.add_argument(
        "--norm_adv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize advantages",
    )
    parser.add_argument(
        "--clip_vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use clipped value loss",
    )
    parser.add_argument("--lambda_stiffness", type=float, default=0.0)
    parser.add_argument("--lambda_eikonal", type=float, default=0.0)
    parser.add_argument("--lambda_zeno", type=float, default=0.0)
    parser.add_argument("--stiffness_epsilon", type=float, default=0.1)
    parser.add_argument("--eikonal_target", type=float, default=1.0)
    parser.add_argument("--lambda_covariant", type=float, default=0.0)
    parser.add_argument("--lambda_riemannian_lyapunov", type=float, default=0.0)
    parser.add_argument("--riemannian_alpha", type=float, default=0.1)
    parser.add_argument(
        "--riemannian_metric",
        type=str,
        default="grad_rms",
        choices=["grad_rms", "obs_var", "policy_fisher", "adam_scalar"],
    )
    parser.add_argument("--riemannian_eps", type=float, default=1e-6)
    parser.add_argument("--riemannian_metric_clip", type=float, default=1e3)
    parser.add_argument("--riemannian_value_floor", type=float, default=1.0)
    parser.add_argument(
        "--riemannian_use_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamics head to estimate delta for Riemannian losses",
    )
    parser.add_argument("--lambda_hjb", type=float, default=0.0)
    parser.add_argument("--hjb_effort_weight", type=float, default=0.01)
    parser.add_argument(
        "--normalize_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize observations",
    )
    parser.add_argument(
        "--normalize_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize rewards",
    )
    parser.add_argument("--clip_obs", type=float, default=10.0)
    parser.add_argument("--clip_reward", type=float, default=10.0)
    parser.add_argument(
        "--torch_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic CuDNN",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> HypoPPOConfig:
    return HypoPPOConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        dynamics_hidden=args.dynamics_hidden,
        dynamics_predict_delta=args.dynamics_predict_delta,
        use_dynamics_head=args.use_dynamics_head,
        lambda_dynamics=args.lambda_dynamics,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
        anneal_lr=args.anneal_lr,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        target_kl=args.target_kl,
        lambda_stiffness=args.lambda_stiffness,
        lambda_eikonal=args.lambda_eikonal,
        lambda_zeno=args.lambda_zeno,
        lambda_covariant=args.lambda_covariant,
        lambda_riemannian_lyapunov=args.lambda_riemannian_lyapunov,
        riemannian_alpha=args.riemannian_alpha,
        riemannian_metric=args.riemannian_metric,
        riemannian_eps=args.riemannian_eps,
        riemannian_metric_clip=args.riemannian_metric_clip,
        riemannian_value_floor=args.riemannian_value_floor,
        riemannian_use_model=args.riemannian_use_model,
        lambda_hjb=args.lambda_hjb,
        hjb_effort_weight=args.hjb_effort_weight,
        stiffness_epsilon=args.stiffness_epsilon,
        eikonal_target=args.eikonal_target,
        normalize_obs=args.normalize_obs,
        normalize_reward=args.normalize_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        torch_deterministic=args.torch_deterministic,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
