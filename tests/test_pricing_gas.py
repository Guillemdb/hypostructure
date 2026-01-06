import math
import sys
from pathlib import Path

import pytest


pytest.importorskip("panel")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fragile.pricing_gas import PricingGas  # noqa: E402


def test_pricing_gas_runs_with_cloning() -> None:
    gas = PricingGas(
        N=256,
        s0=100.0,
        r=0.0,
        mu=0.0,
        sigma=0.0,
        t=1.0,
        steps_per_year=4,
        strike=100.0,
        option_type="call",
        reward_mode="none",
        fitness_mode="weights",
        barrier=None,
    )
    price, state = gas.run_pricing(n_steps=2)
    assert math.isfinite(price)
    assert state.weights.shape[0] == gas.N


def test_pricing_gas_handles_immediate_barrier() -> None:
    gas = PricingGas(
        N=128,
        s0=100.0,
        r=0.0,
        mu=0.0,
        sigma=0.0,
        t=1.0,
        steps_per_year=4,
        strike=100.0,
        option_type="call",
        reward_mode="none",
        fitness_mode="weights",
        barrier=100.0,
        barrier_type="up-and-out",
    )
    price, state = gas.run_pricing(n_steps=1)
    assert price == 0.0
    assert state.alive.sum().item() == 0
