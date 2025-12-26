"""
Barrier Saturation Surgery Module

This module implements dynamic clipping per layer using hypostructure theory.
Instead of fixed epsilon values for clipping barriers (e.g., torch.relu(5.0 - dist)),
we use learnable per-layer barriers that adapt through barrier saturation surgery.

Theory:
-------
In hypostructure theory, a barrier is a topological constraint that prevents
charts/regions from overlapping. Traditional fixed barriers are rigid.

Barrier Saturation Surgery extends this by:
1. **Barrier**: Topological separation threshold between charts
2. **Saturation**: Asymptotic approach to optimal threshold via learning
3. **Surgery**: Topological cut-and-paste operation where barriers adjust per layer depth

Mathematical Formulation:
-------------------------
For layer depth d ∈ [0, 1] and training progress t:
    epsilon(d, t) = epsilon_base * saturation_fn(d) * schedule_fn(t)

Where:
- epsilon_base: Base clipping threshold (learnable or fixed)
- saturation_fn(d): Depth-dependent modulation (surgical component)
- schedule_fn(t): Optional temporal annealing schedule

Axioms Encoded:
---------------
- **Axiom Cap (Capacity)**: Barriers prevent chart collapse
- **Axiom TB (Topological Background)**: Surgery respects layer topology
- **Axiom D (Dissipation)**: Saturation allows gradual boundary adjustment
"""

import torch
import torch.nn as nn
import math


class BarrierSatSurgery(nn.Module):
    """
    Learnable barrier saturation surgery for dynamic clipping.

    This module computes per-layer epsilon values for clipping operations,
    replacing fixed thresholds with adaptive barriers.

    Args:
        num_layers: Number of layers/stages in the architecture
        base_epsilon: Initial base epsilon value (default: 4.0)
        learnable: Whether epsilon values are learnable (default: True)
        surgery_mode: Type of surgical modulation:
            - 'linear': Linear interpolation across depth
            - 'sigmoid': Smooth saturation curve
            - 'exponential': Exponential decay/growth
            - 'cosine': Cosine annealing across layers
        temporal_schedule: Optional temporal annealing schedule:
            - None: No temporal scheduling
            - 'warmup': Linear warmup from 0 to base
            - 'decay': Exponential decay
            - 'cosine': Cosine annealing
    """

    def __init__(
        self,
        num_layers: int = 1,
        base_epsilon: float = 4.0,
        learnable: bool = True,
        surgery_mode: str = 'sigmoid',
        temporal_schedule: str = None,
        min_epsilon: float = 1.0,
        max_epsilon: float = 10.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.surgery_mode = surgery_mode
        self.temporal_schedule = temporal_schedule
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

        # Learnable per-layer base epsilon values
        if learnable:
            self.epsilon_base = nn.Parameter(
                torch.full((num_layers,), base_epsilon, dtype=torch.float32)
            )
            # Learnable surgery modulation parameters
            self.surgery_alpha = nn.Parameter(torch.tensor(1.0))
            self.surgery_beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer(
                'epsilon_base',
                torch.full((num_layers,), base_epsilon, dtype=torch.float32)
            )
            self.register_buffer('surgery_alpha', torch.tensor(1.0))
            self.register_buffer('surgery_beta', torch.tensor(0.0))

        # Training step counter for temporal scheduling
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    def surgical_modulation(self, layer_idx: int) -> torch.Tensor:
        """
        Compute surgical modulation based on layer depth.

        This implements the "surgery" component - topological adjustment
        of barriers based on layer position in the architecture.

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            Modulation factor to apply to base epsilon
        """
        # Normalize layer depth to [0, 1]
        if self.num_layers == 1:
            depth = torch.tensor(0.5, device=self.epsilon_base.device)
        else:
            depth = layer_idx / (self.num_layers - 1)
            depth = torch.tensor(depth, device=self.epsilon_base.device)

        if self.surgery_mode == 'linear':
            # Linear interpolation: epsilon increases/decreases linearly with depth
            modulation = self.surgery_alpha * depth + self.surgery_beta

        elif self.surgery_mode == 'sigmoid':
            # Sigmoid saturation: smooth transition with adjustable steepness
            # Centered at depth=0.5, steepness controlled by alpha
            x = self.surgery_alpha * (depth - 0.5)
            modulation = torch.sigmoid(x) + self.surgery_beta

        elif self.surgery_mode == 'exponential':
            # Exponential decay/growth
            modulation = torch.exp(self.surgery_alpha * depth) + self.surgery_beta

        elif self.surgery_mode == 'cosine':
            # Cosine annealing: smooth periodic modulation
            modulation = 0.5 * (1 + torch.cos(math.pi * depth * self.surgery_alpha)) + self.surgery_beta

        else:
            # Default: no modulation
            modulation = torch.tensor(1.0, device=self.epsilon_base.device)

        # Ensure modulation is positive
        modulation = torch.abs(modulation) + 1e-6

        return modulation

    def temporal_annealing(self) -> torch.Tensor:
        """
        Compute temporal annealing schedule factor.

        This implements time-dependent barrier adjustment during training.

        Returns:
            Temporal scaling factor
        """
        if self.temporal_schedule is None:
            return torch.tensor(1.0, device=self.epsilon_base.device)

        step = self.step.float()

        if self.temporal_schedule == 'warmup':
            # Linear warmup: 0 -> 1 over first 1000 steps
            warmup_steps = 1000.0
            factor = torch.clamp(step / warmup_steps, 0.0, 1.0)

        elif self.temporal_schedule == 'decay':
            # Exponential decay: 1 -> 0.1 over time
            decay_rate = 0.0001
            factor = torch.exp(-decay_rate * step)
            factor = torch.clamp(factor, 0.1, 1.0)

        elif self.temporal_schedule == 'cosine':
            # Cosine annealing: smooth periodic schedule
            period = 10000.0
            factor = 0.5 * (1 + torch.cos(math.pi * (step % period) / period))
            factor = 0.5 + 0.5 * factor  # Scale to [0.5, 1.0]

        else:
            factor = torch.tensor(1.0, device=self.epsilon_base.device)

        return factor

    def get_epsilon(self, layer_idx: int = 0) -> torch.Tensor:
        """
        Get the epsilon value for a specific layer.

        Args:
            layer_idx: Layer index (0 to num_layers-1)

        Returns:
            Dynamic epsilon value for clipping
        """
        # Ensure layer_idx is in bounds
        layer_idx = min(layer_idx, self.num_layers - 1)

        # Get base epsilon for this layer
        epsilon = self.epsilon_base[layer_idx]

        # Apply surgical modulation (depth-dependent)
        surgery_factor = self.surgical_modulation(layer_idx)
        epsilon = epsilon * surgery_factor

        # Apply temporal annealing
        temporal_factor = self.temporal_annealing()
        epsilon = epsilon * temporal_factor

        # Clamp to valid range
        epsilon = torch.clamp(epsilon, self.min_epsilon, self.max_epsilon)

        return epsilon

    def forward(self, distances: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Apply barrier clipping to distance tensor.

        This replaces: torch.relu(fixed_epsilon - distances)
        With: torch.relu(dynamic_epsilon - distances)

        Args:
            distances: Distance tensor to clip
            layer_idx: Layer index for dynamic epsilon

        Returns:
            Clipped distances (hinge loss)
        """
        epsilon = self.get_epsilon(layer_idx)
        return torch.relu(epsilon - distances)

    def step_schedule(self):
        """Increment the training step counter for temporal scheduling."""
        self.step += 1

    def get_epsilon_stats(self) -> dict:
        """
        Get statistics about current epsilon values across layers.

        Returns:
            Dictionary with epsilon statistics
        """
        epsilons = [self.get_epsilon(i).item() for i in range(self.num_layers)]
        return {
            'epsilons': epsilons,
            'mean': sum(epsilons) / len(epsilons),
            'min': min(epsilons),
            'max': max(epsilons),
            'surgery_alpha': self.surgery_alpha.item(),
            'surgery_beta': self.surgery_beta.item(),
            'step': self.step.item(),
        }


class MultiBarrierSurgery(nn.Module):
    """
    Multiple barrier surgery modules for different loss components.

    This allows different barrier configurations for:
    - Chart separation barriers
    - Variance floor barriers
    - Load balancing barriers
    - etc.

    Args:
        num_layers: Number of layers in architecture
        barrier_configs: Dictionary mapping barrier names to configurations
    """

    def __init__(self, num_layers: int = 1, barrier_configs: dict = None):
        super().__init__()

        if barrier_configs is None:
            # Default configurations for common barriers
            barrier_configs = {
                'separation': {
                    'base_epsilon': 5.0,
                    'surgery_mode': 'sigmoid',
                    'temporal_schedule': 'warmup',
                },
                'variance': {
                    'base_epsilon': 1.0,
                    'surgery_mode': 'linear',
                    'temporal_schedule': None,
                },
                'covariance': {
                    'base_epsilon': 3.0,
                    'surgery_mode': 'exponential',
                    'temporal_schedule': 'decay',
                },
            }

        # Create barrier modules
        self.barriers = nn.ModuleDict()
        for name, config in barrier_configs.items():
            self.barriers[name] = BarrierSatSurgery(
                num_layers=num_layers,
                **config
            )

    def forward(self, barrier_name: str, distances: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Apply named barrier clipping.

        Args:
            barrier_name: Name of the barrier to use
            distances: Distance tensor
            layer_idx: Layer index

        Returns:
            Clipped distances
        """
        if barrier_name not in self.barriers:
            raise ValueError(f"Unknown barrier: {barrier_name}. Available: {list(self.barriers.keys())}")

        return self.barriers[barrier_name](distances, layer_idx)

    def get_epsilon(self, barrier_name: str, layer_idx: int = 0) -> torch.Tensor:
        """Get epsilon value for a specific barrier and layer."""
        if barrier_name not in self.barriers:
            raise ValueError(f"Unknown barrier: {barrier_name}")

        return self.barriers[barrier_name].get_epsilon(layer_idx)

    def step_schedule(self):
        """Increment all barrier schedules."""
        for barrier in self.barriers.values():
            barrier.step_schedule()

    def get_all_stats(self) -> dict:
        """Get statistics for all barriers."""
        return {name: barrier.get_epsilon_stats() for name, barrier in self.barriers.items()}


# Convenience function for quick integration
def create_barrier_surgery(
    num_layers: int = 1,
    mode: str = 'single',
    **kwargs
) -> nn.Module:
    """
    Factory function for creating barrier surgery modules.

    Args:
        num_layers: Number of layers
        mode: 'single' for single barrier, 'multi' for multiple barriers
        **kwargs: Additional arguments passed to the module

    Returns:
        Barrier surgery module
    """
    if mode == 'single':
        return BarrierSatSurgery(num_layers=num_layers, **kwargs)
    elif mode == 'multi':
        return MultiBarrierSurgery(num_layers=num_layers, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'multi'")
