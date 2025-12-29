"""
Physicist Agent for Continuous Control.

Split-Brain VAE-PPO architecture that separates:
- z_macro (dim=16): Physics state (position, velocity, orientation)
- z_micro (dim=32): Noise/uncertainty encoding

Key architectural features:
1. Causal Enclosure: Physics engine only sees z_macro
2. Information Dropout: Randomly zero z_micro during training
3. Slowness Constraint: z_macro changes slowly across timesteps
4. BRST Constraints: Near-orthogonal weights for geometric preservation
5. PPO-based training: Vectorized environments
6. Lyapunov Stability: V̇(s) ≤ -α*V(s) for exponential convergence
7. Eikonal Regularization: ||∇V|| ≈ 1 for valid geodesic distance
8. VICReg Integration: Variance-Invariance-Covariance for collapse prevention

See fragile-index.md for theoretical background.

Usage:
    python physicist_agent_continuous.py --env_id Ant-v5 --total_timesteps 1000000
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import math
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Local imports - handle both module and script execution
try:
    from .layers import BRSTLinear, compute_total_brst_defect
    from .losses import (
        vicreg_loss, lyapunov_loss, eikonal_loss, gradient_stiffness_loss,
        zeno_loss, sync_vae_wm_loss, compute_ruppeiner_metric
    )
except ImportError:
    from layers import BRSTLinear, compute_total_brst_defect
    from losses import (
        vicreg_loss, lyapunov_loss, eikonal_loss, gradient_stiffness_loss,
        zeno_loss, sync_vae_wm_loss, compute_ruppeiner_metric
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PhysicistConfig:
    """Configuration for Physicist Agent (environment-agnostic)."""

    # Environment (set dynamically)
    env_id: str = "Ant-v5"         # MuJoCo Ant; use Isaac-Velocity-Flat-Ant-v0 for Isaac Lab
    obs_dim: int = 105             # MuJoCo Ant observation size
    action_dim: int = 8            # Joint torques
    action_low: Optional[np.ndarray] = field(default=None, repr=False)
    action_high: Optional[np.ndarray] = field(default=None, repr=False)
    squash_actions: bool = True    # Tanh-squash Gaussian to action bounds

    # Latent space
    hidden_dim: int = 256          # MLP hidden size
    macro_dim: int = 16            # z_macro: physics state
    micro_dim: int = 32            # z_micro: noise/uncertainty encoding
    sample_latent: bool = False    # Use VAE sampling for policy/value (False = mean)

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Physicist losses (existing)
    lambda_closure: float = 1.0    # Causal enclosure
    lambda_slowness: float = 0.1   # Temporal smoothness
    lambda_kl_micro: float = 0.01  # Push z_micro to N(0,I)
    lambda_kl_macro: float = 0.001 # Light KL on z_macro
    lambda_brst: float = 0.001     # Orthogonality
    info_dropout_prob: float = 0.3 # Drop z_micro during training
    use_brst: bool = True          # Enable BRST constraints

    # Lyapunov & Geometric losses (NEW)
    lambda_lyapunov: float = 1.0   # Stability constraint: V̇ ≤ -α*V
    lyapunov_alpha: float = 0.1    # Decay rate for Lyapunov
    lambda_eikonal: float = 0.1    # Distance function: ||∇V|| ≈ 1
    lambda_stiffness: float = 0.01 # Gradient lower bound: ||∇V|| ≥ ε
    stiffness_epsilon: float = 0.1 # Minimum gradient norm
    lambda_zeno: float = 0.1       # Action smoothness: KL(π_t || π_{t-1})
    lambda_sync: float = 0.1       # VAE-WM synchronization
    lambda_vicreg: float = 1.0     # VICReg (variance-invariance-covariance)
    vicreg_noise_std: float = 0.01 # Perturbation noise for VICReg

    # Training
    num_envs: int = 4096           # Parallel environments
    num_steps: int = 16            # Steps per rollout
    learning_rate: float = 3e-4
    anneal_lr: bool = True         # Anneal learning rate
    num_minibatches: int = 4
    update_epochs: int = 4
    total_timesteps: int = 30_000_000

    # Logging
    log_interval: int = 10         # Log every N updates
    save_interval: int = 100       # Save checkpoint every N updates
    exp_name: str = "physicist"
    seed: int = 1

    # Device
    device: str = "cuda"


# =============================================================================
# Network Components
# =============================================================================

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights and constant bias."""
    if hasattr(layer, 'linear'):
        # For BRSTLinear
        nn.init.orthogonal_(layer.linear.weight, std)
        nn.init.constant_(layer.linear.bias, bias_const)
    elif hasattr(layer, 'weight'):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_micro_dropout(z_micro: torch.Tensor, drop_micro) -> torch.Tensor:
    if drop_micro is None:
        return z_micro
    if isinstance(drop_micro, bool):
        return torch.zeros_like(z_micro) if drop_micro else z_micro
    mask = drop_micro.to(device=z_micro.device, dtype=z_micro.dtype)
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    return z_micro * (1.0 - mask)


class PhysicistEncoder(nn.Module):
    """
    Encoder that produces split latent: z_macro + z_micro.

    Architecture:
        obs → Shared MLP → Split heads → (z_macro, z_micro)

    The encoder learns to separate physics-relevant features (macro)
    from noise features (micro) via the causal enclosure loss.
    """

    def __init__(self, config: PhysicistConfig):
        super().__init__()
        self.config = config

        # Choose layer type based on BRST setting
        LinearLayer = BRSTLinear if config.use_brst else nn.Linear

        # Shared feature extraction
        self.shared = nn.Sequential(
            layer_init(LinearLayer(config.obs_dim, config.hidden_dim)),
            nn.LayerNorm(config.hidden_dim),
            nn.Tanh(),
            layer_init(LinearLayer(config.hidden_dim, config.hidden_dim)),
            nn.LayerNorm(config.hidden_dim),
            nn.Tanh(),
        )

        # z_macro branch (physics state)
        self.macro_mean = layer_init(nn.Linear(config.hidden_dim, config.macro_dim), std=0.01)
        self.macro_logvar = layer_init(nn.Linear(config.hidden_dim, config.macro_dim), std=0.01)

        # z_micro branch (noise encoding)
        self.micro_mean = layer_init(nn.Linear(config.hidden_dim, config.micro_dim), std=0.01)
        self.micro_logvar = layer_init(nn.Linear(config.hidden_dim, config.micro_dim), std=0.01)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to split latent.

        Args:
            x: Observation [batch, obs_dim]

        Returns:
            macro_mean: [batch, macro_dim]
            macro_logvar: [batch, macro_dim]
            micro_mean: [batch, micro_dim]
            micro_logvar: [batch, micro_dim]
        """
        h = self.shared(x)

        macro_mean = self.macro_mean(h)
        macro_logvar = self.macro_logvar(h)

        micro_mean = self.micro_mean(h)
        micro_logvar = self.micro_logvar(h)

        return macro_mean, macro_logvar, micro_mean, micro_logvar


class PhysicsEngine(nn.Module):
    """
    Physics engine that predicts z_macro_{t+1} from z_macro_t.

    CRITICAL: This module is BLIND to z_micro.
    The causal enclosure loss ensures this is sufficient for prediction.

    Uses residual prediction: z_next = z_current + delta(z_current)
    """

    def __init__(self, config: PhysicistConfig):
        super().__init__()
        self.config = config

        # Choose layer type based on BRST setting
        LinearLayer = BRSTLinear if config.use_brst else nn.Linear

        # Residual dynamics prediction
        self.dynamics = nn.Sequential(
            layer_init(LinearLayer(config.macro_dim, 64)),
            nn.Tanh(),
            layer_init(LinearLayer(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, config.macro_dim), std=0.01),  # Small init for residual
        )

    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        """
        Predict next z_macro using residual dynamics.

        Args:
            z_macro: Current macro latent [batch, macro_dim]

        Returns:
            z_macro_pred: Predicted next macro latent [batch, macro_dim]
        """
        delta = self.dynamics(z_macro)
        return z_macro + delta  # Residual connection


class PhysicistActor(nn.Module):
    """
    Actor network: (z_macro, z_micro) → action distribution.

    Outputs mean and log_std for a diagonal Gaussian policy.
    During inference, z_micro can be dropped for more robust behavior.
    """

    def __init__(self, config: PhysicistConfig):
        super().__init__()
        self.config = config

        latent_dim = config.macro_dim + config.micro_dim

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(latent_dim, config.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(config.hidden_dim, config.hidden_dim)),
            nn.Tanh(),
        )

        # Action mean
        self.action_mean = layer_init(nn.Linear(config.hidden_dim, config.action_dim), std=0.01)

        # Learnable log_std (state-independent)
        self.action_logstd = nn.Parameter(torch.zeros(1, config.action_dim))

        if config.action_low is not None and config.action_high is not None and config.squash_actions:
            action_low = torch.as_tensor(config.action_low, dtype=torch.float32)
            action_high = torch.as_tensor(config.action_high, dtype=torch.float32)
            action_scale = (action_high - action_low) / 2.0
            action_bias = (action_high + action_low) / 2.0
            action_scale = torch.clamp(action_scale, min=1e-6)
            self.squash_actions = True
        else:
            action_scale = torch.ones(config.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(config.action_dim, dtype=torch.float32)
            self.squash_actions = False

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
        self.register_buffer("log_action_scale", torch.log(action_scale))

    def forward(
        self,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        drop_micro: Optional[torch.Tensor] = None
    ) -> Normal:
        """
        Get action distribution from latent.

        Args:
            z_macro: Macro latent [batch, macro_dim]
            z_micro: Micro latent [batch, micro_dim]
            drop_micro: If set, zero out z_micro (bool or per-sample mask)

        Returns:
            Action distribution (Normal)
        """
        z_micro = apply_micro_dropout(z_micro, drop_micro)

        z = torch.cat([z_macro, z_micro], dim=-1)
        h = self.backbone(z)

        action_mean = self.action_mean(h)
        action_std = torch.exp(self.action_logstd.expand_as(action_mean))

        return Normal(action_mean, action_std)


class PhysicistCritic(nn.Module):
    """
    Critic network: (z_macro, z_micro) → value.

    Estimates V(s) from the latent representation.
    Also provides gradients ∇V for Lyapunov/Eikonal losses.
    """

    def __init__(self, config: PhysicistConfig):
        super().__init__()
        self.config = config

        latent_dim = config.macro_dim + config.micro_dim

        self.network = nn.Sequential(
            layer_init(nn.Linear(latent_dim, config.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(config.hidden_dim, config.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(config.hidden_dim, 1), std=1.0),
        )

    def forward(
        self,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        drop_micro: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get value estimate from latent.

        Args:
            z_macro: Macro latent [batch, macro_dim]
            z_micro: Micro latent [batch, micro_dim]
            drop_micro: If set, zero out z_micro (bool or per-sample mask)

        Returns:
            Value estimate [batch, 1]
        """
        z_micro = apply_micro_dropout(z_micro, drop_micro)

        z = torch.cat([z_macro, z_micro], dim=-1)
        return self.network(z)


# =============================================================================
# Complete Agent
# =============================================================================

class PhysicistAgent(nn.Module):
    """
    Complete Physicist Agent for Continuous Control.

    Architecture:
        Encoder: obs → (z_macro, z_micro)  [Split-Brain VAE]
        Physics: z_macro_t → z_macro_{t+1} [World Model, blind to z_micro]
        Actor: (z_macro, z_micro) → action distribution [Policy]
        Critic: (z_macro, z_micro) → value [V(s), also used for Lyapunov]

    Training Losses (14 total):
        PPO: Policy gradient + value loss + entropy bonus
        Physicist:
            - L_closure: ||WM(z_macro_t) - z_macro_{t+1}||²
            - L_slowness: ||z_macro_t - z_macro_{t-1}||²
            - L_kl_micro: KL(q(z_micro) || N(0,I))
            - L_kl_macro: KL(q(z_macro) || N(0,I))
            - L_brst: ||W^TW - I||²
        Lyapunov & Geometric:
            - L_lyapunov: max(0, V̇ + α*V)²
            - L_eikonal: (||∇V|| - 1)²
            - L_stiffness: max(0, ε - ||∇V||)²
        Synchronization:
            - L_sync: ||z_enc - sg(WM_pred)||²
            - L_zeno: KL(π_t || π_{t-1})
            - L_vicreg: invariance + variance + covariance
    """

    def __init__(self, config: Optional[PhysicistConfig] = None):
        super().__init__()
        self.config = config or PhysicistConfig()

        self.encoder = PhysicistEncoder(self.config)
        self.physics = PhysicsEngine(self.config)
        self.actor = PhysicistActor(self.config)
        self.critic = PhysicistCritic(self.config)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, obs: torch.Tensor, sample_latent: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Encode observation to latent.

        Args:
            obs: Observation [batch, obs_dim]

        Returns:
            Dictionary with z_macro, z_micro, and their parameters
        """
        macro_mean, macro_logvar, micro_mean, micro_logvar = self.encoder(obs)

        if sample_latent is None:
            sample_latent = self.config.sample_latent

        if sample_latent:
            z_macro = self.reparameterize(macro_mean, macro_logvar)
            z_micro = self.reparameterize(micro_mean, micro_logvar)
        else:
            z_macro = macro_mean
            z_micro = micro_mean

        return {
            "z_macro": z_macro,
            "z_micro": z_micro,
            "macro_mean": macro_mean,
            "macro_logvar": macro_logvar,
            "micro_mean": micro_mean,
            "micro_logvar": micro_logvar,
        }

    def get_value(
        self,
        obs: torch.Tensor,
        drop_micro: Optional[torch.Tensor] = None,
        sample_latent: Optional[bool] = None
    ) -> torch.Tensor:
        """Get value estimate for observations."""
        latents = self.encode(obs, sample_latent=sample_latent)
        return self.critic(latents["z_macro"], latents["z_micro"], drop_micro)

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _squash_action(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(pre_tanh)
        return action * self.actor.action_scale + self.actor.action_bias

    def _unsquash_action(self, action: torch.Tensor) -> torch.Tensor:
        scaled = (action - self.actor.action_bias) / self.actor.action_scale
        return torch.clamp(scaled, -1.0 + 1e-6, 1.0 - 1e-6)

    def _squash_logdet(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        log_det = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_det = log_det + self.actor.log_action_scale
        return log_det.sum(dim=-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        drop_micro: Optional[torch.Tensor] = None,
        sample_latent: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get action, log_prob, entropy, and value.

        Args:
            obs: Observation [batch, obs_dim]
            action: If provided, compute log_prob for this action
            drop_micro: If set, zero z_micro for robust behavior (bool or mask)
            sample_latent: If set, sample VAE latents for policy/value

        Returns:
            action: Sampled or input action [batch, action_dim]
            log_prob: Log probability [batch]
            entropy: Policy entropy [batch]
            value: Value estimate [batch, 1]
            latents: Dictionary of latent variables
        """
        latents = self.encode(obs, sample_latent=sample_latent)
        z_macro = latents["z_macro"]
        z_micro = latents["z_micro"]

        # Get action distribution
        dist = self.actor(z_macro, z_micro, drop_micro)

        if self.actor.squash_actions:
            if action is None:
                pre_tanh = dist.rsample()
                action = self._squash_action(pre_tanh)
            else:
                pre_tanh = self._atanh(self._unsquash_action(action))

            log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
            log_prob = log_prob - self._squash_logdet(pre_tanh)
            entropy = dist.entropy().sum(dim=-1)
        else:
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # Get value
        value = self.critic(z_macro, z_micro, drop_micro)

        return action, log_prob, entropy, value, latents

    def predict_next_macro(self, z_macro: torch.Tensor) -> torch.Tensor:
        """Predict next z_macro using physics engine."""
        return self.physics(z_macro)

    def compute_brst_loss(self) -> torch.Tensor:
        """Compute total BRST defect across encoder and physics."""
        if not self.config.use_brst:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total = compute_total_brst_defect(self.encoder)
        total = total + compute_total_brst_defect(self.physics)
        return total

    def compute_physicist_losses(
        self,
        latents_t: Dict[str, torch.Tensor],
        latents_tp1: Dict[str, torch.Tensor],
        obs_t: Optional[torch.Tensor] = None,
        action_mean_t: Optional[torch.Tensor] = None,
        action_std_t: Optional[torch.Tensor] = None,
        action_mean_prev: Optional[torch.Tensor] = None,
        action_std_prev: Optional[torch.Tensor] = None,
        mask_tp1: Optional[torch.Tensor] = None,
        mask_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all Physicist-specific losses (14 total).

        Skips expensive computations when corresponding weight is 0.

        Args:
            latents_t: Latent variables at time t
            latents_tp1: Latent variables at time t+1
            obs_t: Observations at time t (for VICReg)
            action_mean_t: Current policy mean (for Zeno)
            action_std_t: Current policy std (for Zeno)
            action_mean_prev: Previous policy mean (for Zeno)
            action_std_prev: Previous policy std (for Zeno)
            mask_tp1: Optional mask for valid t→t+1 transitions (1 = include)
            mask_prev: Optional mask for valid t-1→t transitions (1 = include)

        Returns:
            Dictionary of Physicist losses
        """
        device = latents_t["z_macro"].device
        zero = torch.tensor(0.0, device=device)
        losses = {}
        cfg = self.config

        def masked_mean(per_sample: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                return per_sample.mean()
            mask = mask.to(device=per_sample.device, dtype=per_sample.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (per_sample * mask).sum() / denom

        # =================================================================
        # EXISTING LOSSES (from original implementation)
        # =================================================================

        # Causal enclosure: physics engine should predict z_macro_{t+1}
        # Always compute z_macro_pred since it's used by sync loss too
        z_macro_pred = self.predict_next_macro(latents_t["z_macro"])

        if cfg.lambda_closure > 0:
            z_macro_target = latents_tp1["macro_mean"].detach()
            per_sample = F.mse_loss(z_macro_pred, z_macro_target, reduction="none").mean(dim=-1)
            losses["closure"] = masked_mean(per_sample, mask_tp1)
        else:
            losses["closure"] = zero

        # Slowness: z_macro should change slowly
        if cfg.lambda_slowness > 0:
            per_sample = F.mse_loss(
                latents_t["z_macro"], latents_tp1["z_macro"], reduction="none"
            ).mean(dim=-1)
            losses["slowness"] = masked_mean(per_sample, mask_tp1)
        else:
            losses["slowness"] = zero

        # KL divergence for z_micro (push to N(0,I))
        if cfg.lambda_kl_micro > 0:
            losses["kl_micro"] = -0.5 * torch.mean(
                1 + latents_t["micro_logvar"] -
                latents_t["micro_mean"].pow(2) -
                latents_t["micro_logvar"].exp()
            )
        else:
            losses["kl_micro"] = zero

        # Light KL on z_macro
        if cfg.lambda_kl_macro > 0:
            losses["kl_macro"] = -0.5 * torch.mean(
                1 + latents_t["macro_logvar"] -
                latents_t["macro_mean"].pow(2) -
                latents_t["macro_logvar"].exp()
            )
        else:
            losses["kl_macro"] = zero

        # BRST loss (orthogonality)
        if cfg.lambda_brst > 0 and cfg.use_brst:
            losses["brst"] = self.compute_brst_loss()
        else:
            losses["brst"] = zero

        # =================================================================
        # LYAPUNOV & GEOMETRIC LOSSES (skip if weight=0)
        # =================================================================

        # Lyapunov stability: V̇ ≤ -α*V
        if cfg.lambda_lyapunov > 0:
            losses["lyapunov"] = lyapunov_loss(
                self.critic, self.physics, latents_t["z_macro"],
                alpha=cfg.lyapunov_alpha,
                micro_dim=cfg.micro_dim
            )
        else:
            losses["lyapunov"] = zero

        # Eikonal: ||∇V|| ≈ 1 (valid distance function)
        if cfg.lambda_eikonal > 0:
            losses["eikonal"] = eikonal_loss(
                self.critic, latents_t["z_macro"],
                micro_dim=cfg.micro_dim
            )
        else:
            losses["eikonal"] = zero

        # Gradient stiffness: ||∇V|| ≥ ε (no vanishing gradients)
        if cfg.lambda_stiffness > 0:
            losses["stiffness"] = gradient_stiffness_loss(
                self.critic, latents_t["z_macro"],
                epsilon=cfg.stiffness_epsilon,
                micro_dim=cfg.micro_dim
            )
        else:
            losses["stiffness"] = zero

        # =================================================================
        # SYNCHRONIZATION LOSSES (skip if weight=0)
        # =================================================================

        # VAE-WM sync: encoder should match world model predictions
        if cfg.lambda_sync > 0:
            per_sample = F.mse_loss(
                latents_tp1["macro_mean"],
                z_macro_pred.detach(),
                reduction="none"
            ).mean(dim=-1)
            losses["sync"] = masked_mean(per_sample, mask_tp1)
        else:
            losses["sync"] = zero

        # Zeno constraint: smooth policy changes
        if cfg.lambda_zeno > 0 and (
            action_mean_t is not None and action_std_t is not None and
            action_mean_prev is not None and action_std_prev is not None
        ):
            losses["zeno"] = zeno_loss(
                action_mean_t, action_std_t,
                action_mean_prev, action_std_prev,
                mask=mask_prev
            )
        else:
            losses["zeno"] = zero

        # VICReg: variance-invariance-covariance (collapse prevention)
        if cfg.lambda_vicreg > 0 and obs_t is not None and self.training:
            # Perturb observations slightly
            noise = cfg.vicreg_noise_std * torch.randn_like(obs_t)
            obs_perturbed = obs_t + noise

            # Encode perturbed observations
            with torch.no_grad():
                macro_mean_pert, _, _, _ = self.encoder(obs_perturbed)

            losses["vicreg"] = vicreg_loss(
                latents_t["macro_mean"],
                macro_mean_pert,
                lambda_inv=25.0,
                lambda_var=25.0,
                lambda_cov=1.0
            )
        else:
            losses["vicreg"] = zero

        return losses


# =============================================================================
# Training Loop
# =============================================================================

def make_env(config: PhysicistConfig):
    """Create vectorized environment (MuJoCo or Isaac Lab)."""
    # Skip to dummy environment if requested
    if os.environ.get("PHYSICIST_USE_DUMMY", "0") == "1":
        print("Using dummy environment (PHYSICIST_USE_DUMMY=1)")
        return DummyVecEnv(config)

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required unless PHYSICIST_USE_DUMMY=1") from exc

    # Check if it's an Isaac Lab environment
    if "Isaac" in config.env_id:
        try:
            import isaaclab_tasks  # noqa: F401
            if config.env_id not in gym.envs.registry:
                raise ImportError(f"Environment {config.env_id} not registered")
            env = gym.make(config.env_id, num_envs=config.num_envs, device=config.device)
            return IsaacLabWrapper(env, config)
        except (ImportError, Exception) as e:
            print(f"Isaac Lab not available ({e}). Falling back to MuJoCo Ant-v5.")
            config.env_id = "Ant-v5"

    # Use Gymnasium MuJoCo environment with SyncVectorEnv
    print(f"Creating {config.num_envs} parallel MuJoCo environments: {config.env_id}")
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make(config.env_id) for _ in range(config.num_envs)
    ])
    return GymnasiumWrapper(env, config)


class IsaacLabWrapper:
    """Wrapper to normalize Isaac Lab environment interface."""

    def __init__(self, env, config: PhysicistConfig):
        self.env = env
        self.config = config
        self.num_envs = config.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def _select_obs(self, obs):
        if isinstance(obs, dict):
            if "policy" in obs:
                return obs["policy"]
            return next(iter(obs.values()))
        return obs

    def reset(self):
        obs, info = self.env.reset()
        obs = self._select_obs(obs)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._select_obs(obs)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.bool, device=self.device)
            truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()


class GymnasiumWrapper:
    """Wrapper to normalize Gymnasium SyncVectorEnv interface to match training loop."""

    def __init__(self, env, config: PhysicistConfig):
        self.env = env
        self.config = config
        self.num_envs = config.num_envs
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def reset(self):
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, done, truncated, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()


class DummyVecEnv:
    """Dummy vectorized environment for testing without Isaac Lab."""

    def __init__(self, config: PhysicistConfig):
        self.config = config
        self.num_envs = config.num_envs
        self.observation_space = type('Space', (), {
            'shape': (config.obs_dim,),
            'dtype': np.float32
        })()
        self.action_space = type('Space', (), {
            'shape': (config.action_dim,),
            'dtype': np.float32,
            'low': -1.0,
            'high': 1.0
        })()
        self.device = torch.device(config.device)

    def reset(self):
        obs = torch.randn(self.num_envs, self.config.obs_dim, device=self.device)
        return obs, {}

    def step(self, action):
        obs = torch.randn(self.num_envs, self.config.obs_dim, device=self.device)
        reward = torch.randn(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, reward, done, truncated, {}

    def close(self):
        pass


def train_physicist(config: Optional[PhysicistConfig] = None):
    """
    Train Physicist Agent on continuous control environments.

    Args:
        config: Training configuration

    Returns:
        Trained agent and training history
    """
    config = config or PhysicistConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create environment
    env = make_env(config)

    # Update config with actual dimensions
    # Fix #4: Handle both Box and Dict observation spaces (Isaac Lab uses Dict)
    if hasattr(env.observation_space, 'shape'):
        config.obs_dim = env.observation_space.shape[0]
    elif hasattr(env.observation_space, 'spaces'):
        # Isaac Lab Dict space - prefer 'policy' key, fallback to first available
        spaces = env.observation_space.spaces
        obs_space = spaces.get('policy', list(spaces.values())[0])
        config.obs_dim = obs_space.shape[0]

    if hasattr(env.action_space, 'shape'):
        config.action_dim = env.action_space.shape[0]
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        if np.isfinite(action_low).all() and np.isfinite(action_high).all():
            config.action_low = action_low
            config.action_high = action_high
        else:
            config.action_low = None
            config.action_high = None

    # Create agent
    agent = PhysicistAgent(config).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Storage for rollouts
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    # Initialize storage tensors
    obs_storage = torch.zeros((config.num_steps, config.num_envs, config.obs_dim), device=device)
    actions_storage = torch.zeros((config.num_steps, config.num_envs, config.action_dim), device=device)
    logprobs_storage = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_storage = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_storage = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_storage = torch.zeros((config.num_steps, config.num_envs), device=device)

    # Latent storage for Physicist losses
    z_macro_storage = torch.zeros((config.num_steps, config.num_envs, config.macro_dim), device=device)
    macro_mean_storage = torch.zeros((config.num_steps, config.num_envs, config.macro_dim), device=device)
    macro_logvar_storage = torch.zeros((config.num_steps, config.num_envs, config.macro_dim), device=device)
    micro_mean_storage = torch.zeros((config.num_steps, config.num_envs, config.micro_dim), device=device)
    micro_logvar_storage = torch.zeros((config.num_steps, config.num_envs, config.micro_dim), device=device)

    # Temporal observation storage for proper temporal pairs (Fixes #2: temporal indexing)
    next_obs_storage = torch.zeros((config.num_steps, config.num_envs, config.obs_dim), device=device)
    prev_obs_storage = torch.zeros((config.num_steps, config.num_envs, config.obs_dim), device=device)
    prev_dones_storage = torch.zeros((config.num_steps, config.num_envs), device=device)
    drop_micro_storage = torch.zeros((config.num_steps, config.num_envs), dtype=torch.bool, device=device)

    # Training history (all 14 losses + metrics)
    history = {
        "episode_returns": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        # Physicist losses (existing)
        "closure_loss": [],
        "slowness_loss": [],
        "kl_micro": [],
        "kl_macro": [],
        "brst_loss": [],
        # Lyapunov & Geometric losses (new)
        "lyapunov_loss": [],
        "eikonal_loss": [],
        "stiffness_loss": [],
        # Synchronization losses (new)
        "sync_loss": [],
        "zeno_loss": [],
        "vicreg_loss": [],
        # Ruppeiner metrics (from Adam)
        "ruppeiner_condition": [],
        "ruppeiner_flatness": [],
    }

    # Initialize environment
    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()

    # Episode tracking for reward logging
    episode_rewards = torch.zeros(config.num_envs, device=device)
    episode_lengths = torch.zeros(config.num_envs, dtype=torch.long, device=device)
    recent_returns = []  # Store recent episode returns for averaging

    print(f"Starting training: {num_updates} updates, {batch_size} batch size")
    print(f"Config: env={config.env_id}, obs_dim={config.obs_dim}, action_dim={config.action_dim}")
    print(f"Losses: 14 total (PPO + Physicist + Lyapunov + Sync)")

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if config.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # Rollout phase
        prev_obs = obs.clone()  # Initialize prev_obs for Zeno loss (Fix #3)
        prev_done = torch.zeros(config.num_envs, dtype=torch.bool, device=device)

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_storage[step] = obs
            prev_obs_storage[step] = prev_obs  # Store previous obs for Zeno (Fix #3)
            prev_dones_storage[step] = prev_done.float()

            with torch.no_grad():
                # Information dropout during training
                drop_micro = torch.rand(config.num_envs, device=device) < config.info_dropout_prob

                action, logprob, _, value, latents = agent.get_action_and_value(
                    obs, drop_micro=drop_micro
                )

                # Store latents for Physicist losses
                z_macro_storage[step] = latents["z_macro"]
                macro_mean_storage[step] = latents["macro_mean"]
                macro_logvar_storage[step] = latents["macro_logvar"]
                micro_mean_storage[step] = latents["micro_mean"]
                micro_logvar_storage[step] = latents["micro_logvar"]

            drop_micro_storage[step] = drop_micro
            actions_storage[step] = action
            logprobs_storage[step] = logprob
            values_storage[step] = value.flatten()

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.bool, device=device)
            else:
                done = done.to(device=device, dtype=torch.bool)
            if not isinstance(truncated, torch.Tensor):
                truncated = torch.tensor(truncated, dtype=torch.bool, device=device)
            else:
                truncated = truncated.to(device=device, dtype=torch.bool)
            done = torch.logical_or(done, truncated)

            rewards_storage[step] = reward
            dones_storage[step] = done.float()
            next_obs_storage[step] = next_obs  # Store next obs for temporal pairs (Fix #2)

            # Track episode rewards
            episode_rewards += reward
            episode_lengths += 1

            # Check for completed episodes
            if done.any():
                for env_idx in range(config.num_envs):
                    if done[env_idx]:
                        recent_returns.append(episode_rewards[env_idx].item())
                        episode_rewards[env_idx] = 0
                        episode_lengths[env_idx] = 0
                # Keep only last 100 episodes for averaging
                if len(recent_returns) > 100:
                    recent_returns = recent_returns[-100:]

            prev_obs = obs.clone()  # Update prev_obs for next iteration (Fix #3)
            prev_done = done
            obs = next_obs

        # Bootstrap value
        with torch.no_grad():
            next_drop_micro = torch.rand(config.num_envs, device=device) < config.info_dropout_prob
            next_value = agent.get_value(obs, drop_micro=next_drop_micro).flatten()

            # GAE computation
            advantages = torch.zeros_like(rewards_storage)
            lastgaelam = 0

            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - dones_storage[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]

                delta = rewards_storage[t] + config.gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values_storage

        # Flatten batches
        b_obs = obs_storage.reshape(-1, config.obs_dim)
        b_actions = actions_storage.reshape(-1, config.action_dim)
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)

        # Flatten latent storage (kept for reference but no longer used for losses)
        b_z_macro = z_macro_storage.reshape(-1, config.macro_dim)
        b_macro_mean = macro_mean_storage.reshape(-1, config.macro_dim)
        b_macro_logvar = macro_logvar_storage.reshape(-1, config.macro_dim)
        b_micro_mean = micro_mean_storage.reshape(-1, config.micro_dim)
        b_micro_logvar = micro_logvar_storage.reshape(-1, config.micro_dim)

        # Flatten temporal observation storage (Fix #2 & #3)
        b_next_obs = next_obs_storage.reshape(-1, config.obs_dim)
        b_prev_obs = prev_obs_storage.reshape(-1, config.obs_dim)
        b_dones = dones_storage.reshape(-1)
        b_prev_dones = prev_dones_storage.reshape(-1)
        b_drop_micro = drop_micro_storage.reshape(-1)

        # Indices for minibatching
        b_inds = np.arange(batch_size)

        # Update phase
        clipfracs = []
        # Initialize loss accumulators for logging (avoid unbound errors)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        phys_losses = {k: torch.tensor(0.0, device=device) for k in [
            "closure", "slowness", "kl_micro", "kl_macro", "brst",
            "lyapunov", "eikonal", "stiffness", "sync", "zeno", "vicreg"
        ]}
        num_updates_this_batch = 0

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Get new action distribution and values
                _, newlogprob, entropy, newvalue, mb_latents = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    drop_micro=b_drop_micro[mb_inds]
                )

                # PPO losses
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Physicist losses (all 14)
                # Fix #1: Use mb_latents from get_action_and_value (has gradients!)
                latents_t = mb_latents

                # Fix #2: Re-encode next observations for proper temporal pairs
                latents_tp1 = agent.encode(b_next_obs[mb_inds])

                # Fix #3: Re-encode previous observations for proper Zeno loss
                latents_prev = agent.encode(b_prev_obs[mb_inds])

                # Get current and previous policy distributions for Zeno
                dist_curr = agent.actor(latents_t["z_macro"], latents_t["z_micro"])
                dist_prev = agent.actor(latents_prev["z_macro"], latents_prev["z_micro"])

                mb_not_done = 1.0 - b_dones[mb_inds]
                mb_not_prev_done = 1.0 - b_prev_dones[mb_inds]

                phys_losses = agent.compute_physicist_losses(
                    latents_t, latents_tp1,
                    obs_t=b_obs[mb_inds],
                    action_mean_t=dist_curr.mean,
                    action_std_t=dist_curr.stddev,
                    action_mean_prev=dist_prev.mean.detach(),  # Stop gradient on previous
                    action_std_prev=dist_prev.stddev.detach(),
                    mask_tp1=mb_not_done,
                    mask_prev=mb_not_prev_done,
                )

                # Combined loss (14 terms total)
                loss = (
                    # PPO losses
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + config.vf_coef * v_loss
                    # Existing Physicist losses
                    + config.lambda_closure * phys_losses["closure"]
                    + config.lambda_slowness * phys_losses["slowness"]
                    + config.lambda_kl_micro * phys_losses["kl_micro"]
                    + config.lambda_kl_macro * phys_losses["kl_macro"]
                    + config.lambda_brst * phys_losses["brst"]
                    # Lyapunov & Geometric losses (NEW)
                    + config.lambda_lyapunov * phys_losses["lyapunov"]
                    + config.lambda_eikonal * phys_losses["eikonal"]
                    + config.lambda_stiffness * phys_losses["stiffness"]
                    # Synchronization losses (NEW)
                    + config.lambda_sync * phys_losses["sync"]
                    + config.lambda_zeno * phys_losses["zeno"]
                    + config.lambda_vicreg * phys_losses["vicreg"]
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        # Logging
        if update % config.log_interval == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_reward = np.mean(recent_returns) if recent_returns else 0.0

            # Compute Ruppeiner metric from Adam statistics
            ruppeiner = compute_ruppeiner_metric(optimizer)

            print(f"Update {update}/{num_updates} | "
                  f"Steps: {global_step:,} | "
                  f"SPS: {sps} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Episodes: {len(recent_returns)}")
            print(f"  PPO: pg={pg_loss.item():.4f} vf={v_loss.item():.4f} ent={entropy_loss.item():.4f}")
            print(f"  Physicist: closure={phys_losses['closure'].item():.4f} "
                  f"slow={phys_losses['slowness'].item():.4f} "
                  f"brst={phys_losses['brst'].item():.4f}")
            print(f"  Lyapunov: lyap={phys_losses['lyapunov'].item():.4f} "
                  f"eik={phys_losses['eikonal'].item():.4f} "
                  f"stiff={phys_losses['stiffness'].item():.4f}")
            print(f"  Sync: sync={phys_losses['sync'].item():.4f} "
                  f"zeno={phys_losses['zeno'].item():.4f} "
                  f"vicreg={phys_losses['vicreg'].item():.4f}")
            print(f"  Ruppeiner: κ={ruppeiner['condition_number']:.1f} "
                  f"flat={ruppeiner['flatness']:.2e}")

            # Record all losses in history
            history["episode_returns"].append(avg_reward)
            history["policy_loss"].append(pg_loss.item())
            history["value_loss"].append(v_loss.item())
            history["entropy"].append(entropy_loss.item())
            # Physicist losses
            history["closure_loss"].append(phys_losses["closure"].item())
            history["slowness_loss"].append(phys_losses["slowness"].item())
            history["kl_micro"].append(phys_losses["kl_micro"].item())
            history["kl_macro"].append(phys_losses["kl_macro"].item())
            history["brst_loss"].append(phys_losses["brst"].item())
            # Lyapunov & Geometric
            history["lyapunov_loss"].append(phys_losses["lyapunov"].item())
            history["eikonal_loss"].append(phys_losses["eikonal"].item())
            history["stiffness_loss"].append(phys_losses["stiffness"].item())
            # Synchronization
            history["sync_loss"].append(phys_losses["sync"].item())
            history["zeno_loss"].append(phys_losses["zeno"].item())
            history["vicreg_loss"].append(phys_losses["vicreg"].item())
            # Ruppeiner metrics
            history["ruppeiner_condition"].append(ruppeiner["condition_number"])
            history["ruppeiner_flatness"].append(ruppeiner["flatness"])

        # Checkpointing
        if update % config.save_interval == 0:
            checkpoint_path = f"checkpoints/{config.exp_name}_update{update}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "update": update,
                "global_step": global_step,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    env.close()
    return agent, history


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Physicist Agent on MuJoCo/Isaac Lab")

    # Environment
    parser.add_argument("--env_id", type=str, default="Ant-v5", help="Environment ID (e.g., Ant-v5, HalfCheetah-v5)")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--num_steps", type=int, default=64, help="Steps per rollout per environment")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--update_epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")

    # Physicist losses (existing)
    parser.add_argument("--use_brst", action="store_true", default=True, help="Use BRST constraints")
    parser.add_argument("--lambda_closure", type=float, default=1.0, help="Closure loss weight")
    parser.add_argument("--lambda_slowness", type=float, default=0.1, help="Slowness loss weight")
    parser.add_argument("--lambda_kl_micro", type=float, default=0.01, help="KL micro loss weight")
    parser.add_argument("--lambda_kl_macro", type=float, default=0.001, help="KL macro loss weight")
    parser.add_argument("--lambda_brst", type=float, default=0.001, help="BRST loss weight")
    parser.add_argument("--info_dropout_prob", type=float, default=0.3, help="Probability of dropping z_micro")
    parser.add_argument("--sample_latent", action="store_true", default=False, help="Sample latents for policy/value")

    # Lyapunov & Geometric losses (NEW - all active by default)
    parser.add_argument("--lambda_lyapunov", type=float, default=1.0, help="Lyapunov stability loss weight")
    parser.add_argument("--lyapunov_alpha", type=float, default=0.1, help="Lyapunov decay rate")
    parser.add_argument("--lambda_eikonal", type=float, default=0.1, help="Eikonal (||∇V||≈1) loss weight")
    parser.add_argument("--lambda_stiffness", type=float, default=0.01, help="Gradient stiffness loss weight")
    parser.add_argument("--stiffness_epsilon", type=float, default=0.1, help="Min gradient norm threshold")

    # Synchronization losses (NEW - all active by default)
    parser.add_argument("--lambda_zeno", type=float, default=0.1, help="Zeno (action smoothness) loss weight")
    parser.add_argument("--lambda_sync", type=float, default=0.1, help="VAE-WM sync loss weight")
    parser.add_argument("--lambda_vicreg", type=float, default=1.0, help="VICReg loss weight")
    parser.add_argument("--vicreg_noise_std", type=float, default=0.01, help="VICReg perturbation noise")

    # Misc
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--exp_name", type=str, default="physicist", help="Experiment name")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N updates")
    parser.add_argument(
        "--squash_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tanh-squashed Gaussian actions"
    )

    args = parser.parse_args()

    config = PhysicistConfig(
        # Environment
        env_id=args.env_id,
        num_envs=args.num_envs,
        # Training
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        # Physicist losses (existing)
        use_brst=args.use_brst,
        lambda_closure=args.lambda_closure,
        lambda_slowness=args.lambda_slowness,
        lambda_kl_micro=args.lambda_kl_micro,
        lambda_kl_macro=args.lambda_kl_macro,
        lambda_brst=args.lambda_brst,
        info_dropout_prob=args.info_dropout_prob,
        sample_latent=args.sample_latent,
        # Lyapunov & Geometric losses
        lambda_lyapunov=args.lambda_lyapunov,
        lyapunov_alpha=args.lyapunov_alpha,
        lambda_eikonal=args.lambda_eikonal,
        lambda_stiffness=args.lambda_stiffness,
        stiffness_epsilon=args.stiffness_epsilon,
        # Synchronization losses
        lambda_zeno=args.lambda_zeno,
        lambda_sync=args.lambda_sync,
        lambda_vicreg=args.lambda_vicreg,
        vicreg_noise_std=args.vicreg_noise_std,
        # Misc
        seed=args.seed,
        device=args.device,
        exp_name=args.exp_name,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        squash_actions=args.squash_actions,
    )

    _, history = train_physicist(config)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    if history["episode_returns"]:
        print(f"Final Avg Reward: {history['episode_returns'][-1]:.2f}")
        print(f"Best Avg Reward:  {max(history['episode_returns']):.2f}")
    if history["policy_loss"]:
        print(f"Final Policy Loss: {history['policy_loss'][-1]:.4f}")
    else:
        print("Final Policy Loss: n/a (no logs captured; try --log_interval 1)")
    print("="*60)


def test_agent():
    """Test the Physicist agent without Isaac Lab."""
    print("Testing PhysicistAgent...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PhysicistConfig(device=device, num_envs=8, obs_dim=60, action_dim=8)
    agent = PhysicistAgent(config).to(device)

    # Test single observation
    obs = torch.randn(8, 60, device=device)

    # Test encoding
    latents = agent.encode(obs)
    print(f"  Encode: z_macro={latents['z_macro'].shape}, z_micro={latents['z_micro'].shape}")

    # Test action/value
    action, logprob, entropy, value, _ = agent.get_action_and_value(obs)
    print(f"  Action: {action.shape}, LogProb: {logprob.shape}, Value: {value.shape}")

    # Test physics engine
    z_macro_next = agent.predict_next_macro(latents["z_macro"])
    print(f"  Physics: {latents['z_macro'].shape} → {z_macro_next.shape}")

    # Test Physicist losses
    latents_t = agent.encode(obs)
    obs_next = torch.randn(8, 60, device=device)
    latents_tp1 = agent.encode(obs_next)
    phys_losses = agent.compute_physicist_losses(latents_t, latents_tp1)
    print(f"  Physicist losses: {', '.join(f'{k}={v.item():.4f}' for k, v in phys_losses.items())}")

    # Test BRST loss
    brst_loss = agent.compute_brst_loss()
    print(f"  BRST loss: {brst_loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("All tests passed!")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test_agent()
    else:
        main()
