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

        # Storage for Adam-based v (cleaner β estimation)
        self.log_v_from_adam: deque = deque(maxlen=window_size)

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

    def record_from_adam(self, optimizer: torch.optim.Adam, loss: float, model: nn.Module):
        """
        Record observations using Adam's internal state for cleaner β estimation.

        ADVANTAGE:
        - Adam's v = EMA of ||∇Φ||² is already smoothed (β₂ = 0.999)
        - Much cleaner signal than instantaneous grad_norm
        - Zero extra compute (v is already calculated)
        """
        param_norm = 0.0
        for p in model.parameters():
            param_norm += p.data.pow(2).sum().item()
        param_norm = np.sqrt(param_norm)

        # Extract v_total from Adam state
        v_total = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p, {})
                if 'exp_avg_sq' in state:
                    v_total += state['exp_avg_sq'].sum().item()

        # Record for α estimation (loss-based)
        if loss > 0 and param_norm > 0:
            self.log_losses.append(np.log(loss))
            self.log_param_norms.append(np.log(param_norm))

        # Record for β estimation (Adam-based, cleaner)
        if v_total > 0 and param_norm > 0:
            self.log_v_from_adam.append(np.log(v_total))

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

        # Normalize for numerical stability
        x_mean, x_std = x.mean(), x.std() + 1e-8
        x_norm = (x - x_mean) / x_std

        # Estimate α from loss scaling
        # log Φ = α log ||θ|| + c  →  slope = α
        alpha_raw = self._fit_slope(x_norm, y_loss)

        # Estimate β: prefer Adam-based v if available
        if len(self.log_v_from_adam) >= self.min_samples:
            # Use Adam's v directly (already ||∇Φ||², so no factor of 2)
            y_v = np.array(self.log_v_from_adam)
            # Align lengths (they should be same, but just in case)
            min_len = min(len(x_norm), len(y_v))
            beta_raw = self._fit_slope(x_norm[-min_len:], y_v[-min_len:])
            r2_beta = self._compute_r2(x_norm[-min_len:], y_v[-min_len:], beta_raw)
        elif len(self.log_grad_norms) >= self.min_samples:
            # Fallback: use grad_norm (less clean)
            y_grad = np.array(self.log_grad_norms)
            beta_half_raw = self._fit_slope(x_norm, y_grad)
            beta_raw = 2.0 * beta_half_raw
            r2_beta = self._compute_r2(x_norm, y_grad, beta_half_raw)
        else:
            beta_raw = self.beta_ema
            r2_beta = 0.0

        # Compute confidence from R² values
        r2_alpha = self._compute_r2(x_norm, y_loss, alpha_raw)
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


class ScalingRegularizer:
    """
    Regularizes model energy (param norm) when scaling exponents are critical.

    HYPOSTRUCTURE CONNECTION:
    -------------------------
    Implements a soft constraint version of "Energy Check" (Node 1) feedback.
    If ScaleCheck (Node 4) indicates criticality (α ≈ β), we increase the cost
    of "Energy" (||θ||) to force the system into a lower energy state,
    which typically restores subcriticality.
    """

    def __init__(self,
                 reg_scale: float = 1e-3,
                 subcritical_margin: float = 0.1):
        self.reg_scale = reg_scale
        self.subcritical_margin = subcritical_margin
        self.last_penalty_coeff = 0.0

    def compute_penalty(self, model: nn.Module, cert: ScalingCertificate) -> Tuple[torch.Tensor, float]:
        """
        Compute regularization penalty.

        Returns: (penalty_loss, penalty_coefficient)
        """
        # Criticality gap: how far are we from safe margin?
        # Safe: criticality > margin
        # Danger: criticality < margin

        gap = self.subcritical_margin - cert.criticality

        if gap <= 0:
            # Safe zone: no penalty
            self.last_penalty_coeff = 0.0
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device), 0.0

        # Danger zone: scale penalty by depth of violation
        # Linear scaling: coeff = scale * gap
        penalty_coeff = self.reg_scale * gap

        # If supercritical (criticality < 0), amplify penalty
        if cert.criticality < 0:
            penalty_coeff *= 2.0

        self.last_penalty_coeff = penalty_coeff

        # Reg term: coeff * ||θ||²
        # (We use squared norm for smoother gradients)
        param_norm_sq = 0.0
        for p in model.parameters():
            param_norm_sq += p.pow(2).sum()

        loss = penalty_coeff * param_norm_sq
        return loss, penalty_coeff


class LayerwiseTrustEnforcer:
    """
    Per-layer trust region enforcement using Adam's second moment (v).

    HYPOSTRUCTURE CONNECTION:
    -------------------------
    Adam's v ≈ EMA of ||∇Φ||² per parameter, which is the dissipation D.
    High v → high local β → layer is "critical" → apply tighter trust.
    Low v → stable layer → allow looser trust.

    This creates an adaptive, per-layer trust region derived from the
    optimizer's internal state, at zero extra computational cost.
    """

    def __init__(self,
                 optimizer: torch.optim.Adam,
                 base_trust: float = 1.0,
                 sensitivity: float = 1.0,
                 min_trust: float = 0.1,
                 max_trust: float = 2.0):
        """
        Args:
            optimizer: Adam optimizer to extract v from.
            base_trust: Default trust multiplier when v is at global mean.
            sensitivity: How aggressively to scale trust based on v.
            min_trust: Minimum trust multiplier (safety floor).
            max_trust: Maximum trust multiplier (cap).
        """
        self.optimizer = optimizer
        self.base_trust = base_trust
        self.sensitivity = sensitivity
        self.min_trust = min_trust
        self.max_trust = max_trust

        # Statistics
        self.layer_trusts: Dict[str, float] = {}
        self.global_v_mean = 1.0
        self.update_count = 0

    def compute_layer_trusts(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute per-layer trust multipliers from Adam's v statistics.

        Returns dict mapping parameter names to trust multipliers.
        """
        self.update_count += 1

        # First pass: compute global v mean for normalization
        v_values = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                if 'exp_avg_sq' in state:
                    v = state['exp_avg_sq']
                    v_values.append(v.mean().item())

        if not v_values:
            # Optimizer not yet initialized
            return {name: self.base_trust for name, _ in model.named_parameters()}

        self.global_v_mean = np.mean(v_values) + 1e-8

        # Second pass: compute per-layer trust
        layer_trusts = {}
        param_to_name = {id(p): name for name, p in model.named_parameters()}

        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_id = id(p)
                if param_id not in param_to_name:
                    continue

                name = param_to_name[param_id]
                state = self.optimizer.state.get(p, {})

                if 'exp_avg_sq' in state:
                    v = state['exp_avg_sq']
                    v_mean = v.mean().item()

                    # Trust formula: lower trust for higher v (more critical)
                    # trust = base / (1 + sensitivity * (v / global_mean - 1))
                    v_ratio = v_mean / self.global_v_mean
                    trust = self.base_trust / (1.0 + self.sensitivity * (v_ratio - 1.0))

                    # Clamp to bounds
                    trust = np.clip(trust, self.min_trust, self.max_trust)
                else:
                    trust = self.base_trust

                layer_trusts[name] = trust

        self.layer_trusts = layer_trusts
        return layer_trusts

    def apply_trust_scaling(self, model: nn.Module):
        """
        Scale gradients in-place based on per-layer trust.

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        """
        layer_trusts = self.compute_layer_trusts(model)

        for name, param in model.named_parameters():
            if param.grad is not None and name in layer_trusts:
                trust = layer_trusts[name]
                param.grad.mul_(trust)

    def get_stats(self) -> Dict[str, float]:
        """Return statistics about trust distribution."""
        if not self.layer_trusts:
            return {'trust_mean': 1.0, 'trust_min': 1.0, 'trust_max': 1.0}

        trusts = list(self.layer_trusts.values())
        return {
            'trust_mean': np.mean(trusts),
            'trust_min': np.min(trusts),
            'trust_max': np.max(trusts),
            'global_v_mean': self.global_v_mean,
        }


class SemanticScalingEstimator:
    """
    Estimates semantic scaling exponents for Actor-Critic.

    HYPOSTRUCTURE INTERPRETATION:
    -----------------------------
    - α_V (Value/Height): How value estimates scale with training progress.
      High α_V = value function captures more "potential energy" from rewards.

    - β_π (Policy/Dissipation): How policy divergence (KL/entropy) scales.
      High β_π = policy changes are costly (high dissipation).

    SUBCRITICALITY in Actor-Critic:
    - We want α_V > β_π: value should improve faster than policy collapses.
    - If β_π > α_V: policy is changing too fast relative to value improvement.
    """

    def __init__(self, window_size: int = 100, ema_decay: float = 0.95):
        self.window_size = window_size
        self.ema_decay = ema_decay

        # Value tracking (Height)
        self.value_means: deque = deque(maxlen=window_size)
        self.value_vars: deque = deque(maxlen=window_size)
        self.bellman_residual_vars: deque = deque(maxlen=window_size)

        # Policy tracking (Dissipation)
        self.entropies: deque = deque(maxlen=window_size)
        self.kl_divs: deque = deque(maxlen=window_size)

        # Step tracking (for scaling regression)
        self.steps: deque = deque(maxlen=window_size)

        # EMA estimates
        self.alpha_V_ema = 1.0
        self.beta_pi_ema = 1.0

        self.step_count = 0

    def record(self, value_mean: float, value_var: float, 
               bellman_var: float, entropy: float, kl_div: float):
        """Record semantic observations."""
        self.step_count += 1
        self.steps.append(self.step_count)

        self.value_means.append(abs(value_mean) + 1e-8)
        self.value_vars.append(value_var + 1e-8)
        self.bellman_residual_vars.append(bellman_var + 1e-8)
        self.entropies.append(entropy + 1e-8)
        self.kl_divs.append(abs(kl_div) + 1e-8)

    def estimate(self) -> Dict[str, float]:
        """
        Estimate semantic scaling exponents.

        α_V: How value variance grows with steps (stability of value learning)
        β_π: How KL divergence grows with steps (rate of policy change)
        """
        if len(self.steps) < 20:
            return {
                'alpha_V': self.alpha_V_ema,
                'beta_pi': self.beta_pi_ema,
                'semantic_criticality': self.alpha_V_ema - self.beta_pi_ema,
                'bellman_var': 0.0,
            }

        log_steps = np.log(np.array(self.steps))
        x = (log_steps - log_steps.mean()) / (log_steps.std() + 1e-8)

        # α_V: Value variance scaling
        # Higher = value estimates becoming more confident/stable
        log_value_var = np.log(np.array(self.value_vars))
        alpha_V_raw = -self._fit_slope(x, log_value_var)  # Negative: decreasing var is good

        # β_π: KL divergence scaling
        # Higher = policy changes becoming larger
        log_kl = np.log(np.array(self.kl_divs))
        beta_pi_raw = self._fit_slope(x, log_kl)

        # EMA smoothing
        self.alpha_V_ema = self.ema_decay * self.alpha_V_ema + (1 - self.ema_decay) * alpha_V_raw
        self.beta_pi_ema = self.ema_decay * self.beta_pi_ema + (1 - self.ema_decay) * beta_pi_raw

        return {
            'alpha_V': self.alpha_V_ema,
            'beta_pi': self.beta_pi_ema,
            'semantic_criticality': self.alpha_V_ema - self.beta_pi_ema,
            'bellman_var': np.mean(self.bellman_residual_vars),
        }

    def _fit_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple slope estimation."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = np.dot(x_c, x_c) + 1e-8
        return np.dot(x_c, y_c) / denom


class ActorCriticCoherenceRegularizer:
    """
    Regularizes based on Actor-Critic coherence and Bellman residual.

    REGULARIZATION TERMS:
    ---------------------
    1. Bellman Residual Variance: Penalizes high variance in TD errors.
       High variance = unstable value estimates.

    2. Gradient Coherence: Penalizes when actor and critic gradients conflict.
       Low coherence = actor and critic fighting each other.

    3. Semantic Subcriticality: Penalizes when β_π > α_V.
       Policy changing faster than value improving = unstable.
    """

    def __init__(self,
                 bellman_weight: float = 0.01,
                 coherence_weight: float = 0.01,
                 semantic_weight: float = 0.001):
        self.bellman_weight = bellman_weight
        self.coherence_weight = coherence_weight
        self.semantic_weight = semantic_weight

        self.last_coherence = 1.0
        self.last_bellman_penalty = 0.0

    def compute_bellman_variance_penalty(self, 
                                         values: torch.Tensor,
                                         targets: torch.Tensor) -> torch.Tensor:
        """
        Penalty for high variance in Bellman residuals.
        """
        residuals = values - targets
        residual_var = residuals.var()

        self.last_bellman_penalty = residual_var.item()
        return self.bellman_weight * residual_var

    def compute_gradient_coherence(self,
                                   actor_grads: List[torch.Tensor],
                                   critic_grads: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Compute coherence between actor and critic gradients.

        Coherence = cosine similarity of flattened gradients.
        Penalty = 1 - coherence (penalize when they point opposite ways).
        """
        # Flatten and concatenate gradients
        actor_flat = torch.cat([g.flatten() for g in actor_grads if g is not None])
        critic_flat = torch.cat([g.flatten() for g in critic_grads if g is not None])

        if len(actor_flat) == 0 or len(critic_flat) == 0:
            return torch.tensor(0.0), 1.0

        # Make same length (take min)
        min_len = min(len(actor_flat), len(critic_flat))
        actor_flat = actor_flat[:min_len]
        critic_flat = critic_flat[:min_len]

        # Cosine similarity
        dot = torch.dot(actor_flat, critic_flat)
        norm_a = actor_flat.norm() + 1e-8
        norm_c = critic_flat.norm() + 1e-8
        coherence = dot / (norm_a * norm_c)

        self.last_coherence = coherence.item()

        # Penalty for low/negative coherence
        penalty = self.coherence_weight * (1.0 - coherence)
        return penalty, coherence.item()

    def compute_semantic_penalty(self,
                                 semantic_stats: Dict[str, float]) -> torch.Tensor:
        """
        Penalty when policy divergence outpaces value improvement.
        """
        alpha_V = semantic_stats.get('alpha_V', 1.0)
        beta_pi = semantic_stats.get('beta_pi', 1.0)

        # Penalize when β_π > α_V (supercritical in semantic sense)
        gap = beta_pi - alpha_V
        if gap > 0:
            return torch.tensor(self.semantic_weight * gap)
        return torch.tensor(0.0)

    def get_stats(self) -> Dict[str, float]:
        return {
            'grad_coherence': self.last_coherence,
            'bellman_penalty': self.last_bellman_penalty,
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
