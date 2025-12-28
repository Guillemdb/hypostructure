"""
Physicist Experiment: Testing the Split-Brain VAE-RNN Architecture.

This experiment validates the Physicist architecture's ability to separate
macro-level physics (signal) from micro-level noise using the 1D Noisy
Bouncing Pulse environment - a mathematically rigorous test of:

1. Renormalization: 64 pixels → 1 float (position)
2. Causal Enclosure: Physics doesn't need noise
3. Inertial Manifold: Position is slow, noise is fast

See fragile-index.md Section 9 for theoretical background.
"""

from .noisy_pulse_env import (
    NoisyPulseEnv,
    NoisyPulseConfig,
    OccludedPulseEnv,
    OccludedPulseConfig,
)
from .physicist_agent import PhysicistAgent1D, PhysicistConfig1D
from .baseline_agent import BaselineAgent1D, BaselineConfig1D

__all__ = [
    "NoisyPulseEnv",
    "NoisyPulseConfig",
    "OccludedPulseEnv",
    "OccludedPulseConfig",
    "PhysicistAgent1D",
    "PhysicistConfig1D",
    "BaselineAgent1D",
    "BaselineConfig1D",
]
