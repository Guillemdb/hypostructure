"""
2D Physicist Agents with CNN encoders.

These agents process 64x64 2D observations using convolutional networks
while maintaining the split-brain architecture for physics separation.
"""

from .physicist_2d import PhysicistAgent2D, PhysicistConfig2D
from .baseline_2d import BaselineAgent2D, BaselineConfig2D

__all__ = [
    "PhysicistAgent2D",
    "PhysicistConfig2D",
    "BaselineAgent2D",
    "BaselineConfig2D",
]
