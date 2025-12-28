"""
2D Physicist Environments: Boss Fights.

These environments test advanced capabilities:
- Tunnel: Object permanence through occlusion
- Chameleon: Temporal integration with low SNR
- ThreeBody: Chaotic dynamics and energy conservation
"""

from .base_2d_env import Base2DEnv, Base2DConfig
from .tunnel_env import NoisyTunnelEnv, TunnelConfig
from .chameleon_env import ChameleonEnv, ChameleonConfig

__all__ = [
    "Base2DEnv",
    "Base2DConfig",
    "NoisyTunnelEnv",
    "TunnelConfig",
    "ChameleonEnv",
    "ChameleonConfig",
]
