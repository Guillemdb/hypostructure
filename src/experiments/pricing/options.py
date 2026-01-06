#!/usr/bin/env python3
"""
Fragile Pricing Gas experiment (SMC + barrier killing) with analytic and MC benchmarks.

This script implements an SMC / Interacting Particle System estimator for option pricing
in the spirit of docs/source/sketches/fragile/pricing_gas.md and pricing.md. It includes
Reiner-Rubinstein barrier benchmarks and a Heston Monte Carlo reference for Asian exotics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class OptionExperimentConfig:
    s0: float = 100.0
    r: float = 0.02
    mu: float = 0.02
    t: float = 1.0
    steps_per_year: int = 252
    n_paths: int = 20000
    engine: str = "smc"
    dynamics: str = "regime-gbm"
    sigma_low: float = 0.15
    sigma_high: float = 0.35
    p_switch: float = 0.02
    p_high_init: float = 0.5
    infer_steps: int = 20
    risk_aversion: float = 2.0
    info_weight: float = 0.5
    option_type: str = "call"
    payoff_kind: str = "vanilla"
    strikes: tuple[float, ...] = (80, 90, 100, 110, 120)
    barrier: float | None = None
    barrier_type: str = "up-and-out"
    pg_reward_mode: str = "none"
    pg_fitness_mode: str = "weights"
    pg_potential_mode: str = "none"
    pg_potential_scale: float = 1.0
    pg_reward_scale: float = 1.0
    pg_companion_epsilon: float = 0.1
    pg_clone_p_max: float = 1.0
    pg_clone_epsilon: float = 0.01
    pg_clone_sigma_x: float = 0.1
    pg_clone_alpha: float = 0.5
    heston_v0: float = 0.04
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.5
    heston_rho: float = -0.7
    heston_benchmark_paths: int = 0
    bb_paths: int = 0
    fg_alpha: float = 1.0
    fg_beta: float = 1.0
    fg_eta: float = 0.1
    fg_lambda_alg: float = 0.0
    fg_sigma_min: float = 1e-8
    fg_epsilon_dist: float = 1e-8
    fg_A: float = 2.0
    fg_rho: float | None = None
    ess_resample: float = 0.30
    ess_fail: float = 0.07
    weight_max: float = 0.12
    kill_rate_max: float = 0.15
    over_resample_limit: int = 8
    stale_frac_max: float = 0.02
    gamma_multiplier: float = 2.5
    seed: int = 7
    output_dir: str = "outputs/pricing"
    save_json: bool = True
    plot: bool = False
    example: str | None = None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    s0: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    if t <= 0:
        payoff = max(s0 - strike, 0.0) if option_type == "call" else max(strike - s0, 0.0)
        return payoff
    if sigma <= 0:
        forward = s0 * math.exp(r * t)
        payoff = max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
        return math.exp(-r * t) * payoff
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return s0 * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - s0 * _norm_cdf(-d1)


def black_scholes_gamma(
    s0: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    if t <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    return _norm_pdf(d1) / (s0 * sigma * math.sqrt(t))


def barrier_up_and_out_call_rr(
    s0: float,
    strike: float,
    barrier: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    if t <= 0.0 or sigma <= 0.0:
        return 0.0 if barrier <= s0 else max(s0 - strike, 0.0)
    if barrier <= s0:
        return 0.0
    if barrier <= strike:
        raise ValueError("Reiner-Rubinstein up-and-out call requires barrier > strike.")
    mu = (r - 0.5 * sigma ** 2) / (sigma ** 2)
    vol_sqrt = sigma * math.sqrt(t)
    x1 = math.log(s0 / strike) / vol_sqrt + (1 + mu) * vol_sqrt
    x2 = math.log(s0 / barrier) / vol_sqrt + (1 + mu) * vol_sqrt
    y1 = math.log(barrier * barrier / (s0 * strike)) / vol_sqrt + (1 + mu) * vol_sqrt
    y2 = math.log(barrier / s0) / vol_sqrt + (1 + mu) * vol_sqrt

    term_a = s0 * _norm_cdf(x1) - strike * math.exp(-r * t) * _norm_cdf(x1 - vol_sqrt)
    term_b = s0 * _norm_cdf(x2) - strike * math.exp(-r * t) * _norm_cdf(x2 - vol_sqrt)
    term_c = (
        s0 * (barrier / s0) ** (2 * (mu + 1)) * _norm_cdf(y1)
        - strike * math.exp(-r * t) * (barrier / s0) ** (2 * mu) * _norm_cdf(y1 - vol_sqrt)
    )
    term_d = (
        s0 * (barrier / s0) ** (2 * (mu + 1)) * _norm_cdf(y2)
        - strike * math.exp(-r * t) * (barrier / s0) ** (2 * mu) * _norm_cdf(y2 - vol_sqrt)
    )

    return max(term_a - term_b - term_c + term_d, 0.0)


def barrier_down_and_out_put_rr(
    s0: float,
    strike: float,
    barrier: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    if t <= 0.0 or sigma <= 0.0:
        return 0.0 if barrier >= s0 else max(strike - s0, 0.0)
    if barrier >= s0:
        return 0.0
    if barrier >= strike:
        raise ValueError("Reiner-Rubinstein down-and-out put requires barrier < strike.")
    mu = (r - 0.5 * sigma ** 2) / (sigma ** 2)
    vol_sqrt = sigma * math.sqrt(t)
    x1 = math.log(s0 / strike) / vol_sqrt + (1 + mu) * vol_sqrt
    x2 = math.log(s0 / barrier) / vol_sqrt + (1 + mu) * vol_sqrt
    y1 = math.log(barrier * barrier / (s0 * strike)) / vol_sqrt + (1 + mu) * vol_sqrt
    y2 = math.log(barrier / s0) / vol_sqrt + (1 + mu) * vol_sqrt

    term_a = strike * math.exp(-r * t) * _norm_cdf(-x1 + vol_sqrt) - s0 * _norm_cdf(-x1)
    term_b = strike * math.exp(-r * t) * _norm_cdf(-x2 + vol_sqrt) - s0 * _norm_cdf(-x2)
    term_c = (
        strike * math.exp(-r * t) * (barrier / s0) ** (2 * mu) * _norm_cdf(y1 - vol_sqrt)
        - s0 * (barrier / s0) ** (2 * (mu + 1)) * _norm_cdf(y1)
    )
    term_d = (
        strike * math.exp(-r * t) * (barrier / s0) ** (2 * mu) * _norm_cdf(y2 - vol_sqrt)
        - s0 * (barrier / s0) ** (2 * (mu + 1)) * _norm_cdf(y2)
    )

    return max(term_a - term_b + term_c - term_d, 0.0)


def _log_mean_exp(values: np.ndarray) -> float:
    max_val = float(np.max(values))
    return max_val + math.log(float(np.mean(np.exp(values - max_val))))


def weighted_entropic_price(
    values: np.ndarray,
    weights: np.ndarray,
    log_norm: float,
    gamma: float,
) -> float:
    if abs(gamma) < 1e-8:
        return math.exp(log_norm) * float(np.mean(weights * values))
    eps = 1e-12
    log_weights = np.log(weights + eps)
    max_val = float(np.max(log_weights - gamma * values))
    log_sum = max_val + math.log(float(np.mean(np.exp(log_weights - gamma * values - max_val))))
    return -(log_norm + log_sum) / gamma


def bs_effective_sigma(cfg: OptionExperimentConfig) -> float:
    return math.sqrt(
        (1.0 - cfg.p_high_init) * cfg.sigma_low ** 2
        + cfg.p_high_init * cfg.sigma_high ** 2
    )


def rr_barrier_kind(cfg: OptionExperimentConfig, strike: float) -> str | None:
    if cfg.barrier is None:
        return None
    if cfg.barrier_type == "up-and-out" and cfg.option_type == "call":
        if cfg.barrier <= max(cfg.s0, strike):
            return None
        kind = "up-and-out-call"
    elif cfg.barrier_type == "down-and-out" and cfg.option_type == "put":
        if cfg.barrier >= min(cfg.s0, strike):
            return None
        kind = "down-and-out-put"
    else:
        return None
    if not math.isclose(cfg.sigma_low, cfg.sigma_high, rel_tol=1e-9, abs_tol=1e-9):
        return None
    if not math.isclose(cfg.mu, cfg.r, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return kind


def format_rr_report(results: list[dict[str, object]]) -> str:
    rows = []
    for row in results:
        rr_price = row.get("barrier_rr")
        if rr_price is None:
            continue
        rows.append(
            {
                "strike": row["strike"],
                "smc": row["smc_price"],
                "rr": rr_price,
                "diff": row["rr_diff"],
                "bs": row["black_scholes"],
            }
        )
    if not rows:
        return ""

    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.6f}"

    headers = ["Strike", "SMC", "RR", "SMC-RR", "BS"]
    values = [
        [fmt(row["strike"]), fmt(row["smc"]), fmt(row["rr"]), fmt(row["diff"]), fmt(row["bs"])]
        for row in rows
    ]
    col_widths = [
        max(len(headers[idx]), max(len(row[idx]) for row in values))
        for idx in range(len(headers))
    ]

    def build_row(items: list[str]) -> str:
        return " | ".join(item.rjust(col_widths[idx]) for idx, item in enumerate(items))

    lines = [build_row(headers), build_row(["-" * w for w in col_widths])]
    lines.extend(build_row(row) for row in values)
    return "\n".join(lines)


def format_heston_report(results: list[dict[str, object]]) -> str:
    rows = []
    for row in results:
        heston_price = row.get("heston_benchmark")
        if heston_price is None:
            continue
        rows.append(
            {
                "strike": row["strike"],
                "smc": row["smc_price"],
                "heston": heston_price,
                "diff": row["heston_diff"],
            }
        )
    if not rows:
        return ""

    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.6f}"

    headers = ["Strike", "SMC", "Heston", "SMC-Heston"]
    values = [
        [fmt(row["strike"]), fmt(row["smc"]), fmt(row["heston"]), fmt(row["diff"])]
        for row in rows
    ]
    col_widths = [
        max(len(headers[idx]), max(len(row[idx]) for row in values))
        for idx in range(len(headers))
    ]

    def build_row(items: list[str]) -> str:
        return " | ".join(item.rjust(col_widths[idx]) for idx, item in enumerate(items))

    lines = [build_row(headers), build_row(["-" * w for w in col_widths])]
    lines.extend(build_row(row) for row in values)
    return "\n".join(lines)


def option_payoff(s_t: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(s_t - strike, 0.0)
    return np.maximum(strike - s_t, 0.0)


def _barrier_breached(s: np.ndarray, cfg: OptionExperimentConfig) -> np.ndarray:
    if cfg.barrier is None:
        return np.zeros_like(s, dtype=bool)
    if cfg.barrier_type == "up-and-out":
        return s >= cfg.barrier
    if cfg.barrier_type == "down-and-out":
        return s <= cfg.barrier
    raise ValueError(f"Unknown barrier type: {cfg.barrier_type}")


def _resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("All particle weights are zero; resampling impossible.")
    probs = weights / total
    return rng.choice(weights.shape[0], size=weights.shape[0], replace=True, p=probs)


def estimate_info_cost_from_stats(
    sum_r: np.ndarray,
    sum_r2: np.ndarray,
    count: int,
    dt: float,
    cfg: OptionExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        return np.zeros_like(sum_r), np.full_like(sum_r, cfg.p_high_init)

    var = (sum_r2 - (sum_r ** 2) / count) / max(count - 1, 1)
    var = np.clip(var, 0.0, None)
    vol_est = np.sqrt(var) / math.sqrt(dt)

    if cfg.sigma_high == cfg.sigma_low:
        q_high = np.full_like(vol_est, cfg.p_high_init)
    else:
        q_high = (vol_est - cfg.sigma_low) / (cfg.sigma_high - cfg.sigma_low)
        q_high = np.clip(q_high, 0.0, 1.0)

    eps = 1e-12
    prior = min(max(cfg.p_high_init, eps), 1.0 - eps)
    qh = np.clip(q_high, eps, 1.0 - eps)
    ql = 1.0 - qh
    kl = qh * np.log(qh / prior) + ql * np.log(ql / (1.0 - prior))
    return kl, q_high


def simulate_pricing_gas(cfg: OptionExperimentConfig) -> dict[str, object]:
    if cfg.engine in {"pricing-gas", "fragile-gas"}:
        return simulate_pricing_gas_clone(cfg)
    if cfg.dynamics == "heston":
        return _simulate_pricing_gas_heston(cfg)
    return _simulate_pricing_gas_regime(cfg)


def simulate_pricing_gas_clone(cfg: OptionExperimentConfig) -> dict[str, object]:
    try:
        import torch
        from fragile.core.cloning import CloneOperator
        from fragile.core.companion_selection import CompanionSelection
        from fragile.pricing_gas import PricingGas
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pricing-gas/fragile-gas requires optional dependencies (panel, torch). "
            "Install them or use engine=smc."
        ) from exc

    sigma_eff = bs_effective_sigma(cfg)
    fitness_op = None
    fitness_mode = cfg.pg_fitness_mode
    if cfg.engine == "fragile-gas":
        try:
            from fragile.core.fitness import FitnessOperator
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "fragile-gas requires fragile.core.fitness dependencies."
            ) from exc
        fitness_op = FitnessOperator(
            alpha=cfg.fg_alpha,
            beta=cfg.fg_beta,
            eta=cfg.fg_eta,
            lambda_alg=cfg.fg_lambda_alg,
            sigma_min=cfg.fg_sigma_min,
            epsilon_dist=cfg.fg_epsilon_dist,
            A=cfg.fg_A,
            rho=cfg.fg_rho,
        )
        if fitness_mode == "weights":
            fitness_mode = "linear"
    gas = PricingGas(
        N=cfg.n_paths,
        d=1,
        s0=cfg.s0,
        r=cfg.r,
        mu=cfg.mu,
        sigma=sigma_eff,
        t=cfg.t,
        steps_per_year=cfg.steps_per_year,
        strike=cfg.strikes[0],
        option_type=cfg.option_type,
        payoff_kind=cfg.payoff_kind,
        barrier=cfg.barrier,
        barrier_type=cfg.barrier_type,
        reward_mode=cfg.pg_reward_mode,
        fitness_mode=fitness_mode,
        potential_mode=cfg.pg_potential_mode,
        potential_scale=cfg.pg_potential_scale,
        reward_scale=cfg.pg_reward_scale,
        companion_selection=CompanionSelection(method="cloning", epsilon=cfg.pg_companion_epsilon),
        cloning=CloneOperator(
            p_max=cfg.pg_clone_p_max,
            epsilon_clone=cfg.pg_clone_epsilon,
            sigma_x=cfg.pg_clone_sigma_x,
            alpha_restitution=cfg.pg_clone_alpha,
        ),
        fitness_op=fitness_op,
    )

    state = gas.initialize_state()
    ess_trace: list[float] = []
    kill_trace: list[float] = []
    kill_step_trace: list[float] = []
    max_weight_trace: list[float] = []
    resample_steps = 0
    resample_streak = 0
    max_resample_streak = 0

    prev_alive = state.alive.clone()

    for _ in range(gas.n_steps):
        state, _, info = gas.step(state, return_info=True)
        weights_pre = info.get("weights_pre_clone", state.weights)
        weights = state.weights
        weight_sum = float(torch.sum(weights))
        if weight_sum <= 0.0:
            break
        ess = (float(torch.sum(weights_pre)) ** 2) / float(torch.sum(weights_pre**2))
        ess_trace.append(ess / cfg.n_paths)
        alive_mask = info["alive_mask"]
        killed_step = float(torch.mean((prev_alive & ~alive_mask).float()))
        kill_step_trace.append(killed_step)
        kill_trace.append(1.0 - float(torch.mean(alive_mask.float())))
        max_weight_trace.append(float(torch.max(weights_pre / torch.sum(weights_pre))))

        num_cloned = int(info.get("num_cloned", 0))
        if num_cloned > 0:
            resample_steps += 1
            resample_streak += 1
            max_resample_streak = max(max_resample_streak, resample_streak)
        else:
            resample_streak = 0

        prev_alive = alive_mask.clone()

    terminal_prices = state.x[:, 0].detach().cpu().numpy()
    avg_prices = None
    if state.running_sum is not None:
        avg_prices = (state.running_sum / float(state.step_index + 1))[:, 0].detach().cpu().numpy()

    return {
        "terminal_prices": terminal_prices,
        "avg_prices": avg_prices,
        "weights": state.weights.detach().cpu().numpy(),
        "log_norm": float(state.log_norm),
        "info_cost": np.zeros(cfg.n_paths, dtype=float),
        "q_high": np.zeros(cfg.n_paths, dtype=float),
        "ess_trace": ess_trace,
        "kill_trace": kill_trace,
        "kill_step_trace": kill_step_trace,
        "max_weight_trace": max_weight_trace,
        "resample_steps": resample_steps,
        "max_resample_streak": max_resample_streak,
    }


def _simulate_pricing_gas_regime(cfg: OptionExperimentConfig) -> dict[str, object]:
    n_steps = max(1, int(cfg.t * cfg.steps_per_year))
    dt = cfg.t / n_steps
    rng = np.random.default_rng(cfg.seed)

    s = np.full(cfg.n_paths, cfg.s0, dtype=float)
    regimes = (rng.random(cfg.n_paths) < cfg.p_high_init).astype(np.int32)
    weights = np.ones(cfg.n_paths, dtype=float)
    alive = np.ones(cfg.n_paths, dtype=bool)

    sum_r = np.zeros(cfg.n_paths, dtype=float)
    sum_r2 = np.zeros(cfg.n_paths, dtype=float)
    ret_count = 0
    running_sum = s.copy() if cfg.payoff_kind == "asian-arithmetic" else None

    ess_trace: list[float] = []
    kill_trace: list[float] = []
    kill_step_trace: list[float] = []
    max_weight_trace: list[float] = []
    resample_steps = 0
    max_resample_streak = 0
    resample_streak = 0

    log_norm = 0.0

    for step in range(1, n_steps + 1):
        sigma = np.where(regimes == 1, cfg.sigma_high, cfg.sigma_low)
        z = rng.standard_normal(cfg.n_paths)
        s_prev = s
        s = s * np.exp((cfg.mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        if running_sum is not None:
            running_sum += s

        if step <= cfg.infer_steps:
            log_ret = np.log(s / s_prev)
            sum_r += log_ret
            sum_r2 += log_ret ** 2
            ret_count += 1

        breached = _barrier_breached(s, cfg)
        alive_prev = alive.copy()
        alive = alive & ~breached
        killed_step = float(np.mean(alive_prev & ~alive))
        kill_step_trace.append(killed_step)

        weights *= math.exp(-cfg.r * dt)
        weights[~alive] = 0.0

        mean_w = float(np.mean(weights))
        if mean_w <= 0.0:
            break
        log_norm += math.log(mean_w)
        weights /= mean_w

        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        ess_ratio = float(ess / cfg.n_paths)
        ess_trace.append(ess_ratio)
        kill_trace.append(1.0 - float(np.mean(alive)))
        max_weight_trace.append(float(np.max(weights / np.sum(weights))))

        if step < n_steps and ess_ratio < cfg.ess_resample:
            indices = _resample_indices(weights, rng)
            s = s[indices]
            regimes = regimes[indices]
            alive = alive[indices]
            sum_r = sum_r[indices]
            sum_r2 = sum_r2[indices]
            if running_sum is not None:
                running_sum = running_sum[indices]
            weights = np.ones(cfg.n_paths, dtype=float)
            resample_steps += 1
            resample_streak += 1
            max_resample_streak = max(max_resample_streak, resample_streak)
        else:
            resample_streak = 0

        flip = rng.random(cfg.n_paths) < cfg.p_switch
        regimes = np.where(flip, 1 - regimes, regimes)

    info_cost, q_high = estimate_info_cost_from_stats(sum_r, sum_r2, ret_count, dt, cfg)
    avg_prices = None
    if running_sum is not None:
        avg_prices = running_sum / (n_steps + 1)

    return {
        "terminal_prices": s,
        "avg_prices": avg_prices,
        "weights": weights,
        "log_norm": log_norm,
        "info_cost": info_cost,
        "q_high": q_high,
        "ess_trace": ess_trace,
        "kill_trace": kill_trace,
        "kill_step_trace": kill_step_trace,
        "max_weight_trace": max_weight_trace,
        "resample_steps": resample_steps,
        "max_resample_streak": max_resample_streak,
    }


def _simulate_pricing_gas_heston(cfg: OptionExperimentConfig) -> dict[str, object]:
    n_steps = max(1, int(cfg.t * cfg.steps_per_year))
    dt = cfg.t / n_steps
    rng = np.random.default_rng(cfg.seed)

    s = np.full(cfg.n_paths, cfg.s0, dtype=float)
    v = np.full(cfg.n_paths, cfg.heston_v0, dtype=float)
    weights = np.ones(cfg.n_paths, dtype=float)
    alive = np.ones(cfg.n_paths, dtype=bool)
    running_sum = s.copy() if cfg.payoff_kind == "asian-arithmetic" else None

    ess_trace: list[float] = []
    kill_trace: list[float] = []
    kill_step_trace: list[float] = []
    max_weight_trace: list[float] = []
    resample_steps = 0
    max_resample_streak = 0
    resample_streak = 0
    log_norm = 0.0

    for step in range(1, n_steps + 1):
        z_v = rng.standard_normal(cfg.n_paths)
        z_s = rng.standard_normal(cfg.n_paths)
        z_s = cfg.heston_rho * z_v + math.sqrt(1.0 - cfg.heston_rho ** 2) * z_s

        # Full truncation Euler for variance with correlated asset shock.
        v_pos = np.maximum(v, 0.0)
        v = v + cfg.heston_kappa * (cfg.heston_theta - v_pos) * dt
        v = v + cfg.heston_xi * np.sqrt(v_pos) * math.sqrt(dt) * z_v
        v = np.maximum(v, 0.0)
        v_bar = 0.5 * (v_pos + v)
        s = s * np.exp((cfg.mu - 0.5 * v_bar) * dt + np.sqrt(v_bar * dt) * z_s)
        if running_sum is not None:
            running_sum += s

        breached = _barrier_breached(s, cfg)
        alive_prev = alive.copy()
        alive = alive & ~breached
        killed_step = float(np.mean(alive_prev & ~alive))
        kill_step_trace.append(killed_step)

        weights *= math.exp(-cfg.r * dt)
        weights[~alive] = 0.0

        mean_w = float(np.mean(weights))
        if mean_w <= 0.0:
            break
        log_norm += math.log(mean_w)
        weights /= mean_w

        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        ess_ratio = float(ess / cfg.n_paths)
        ess_trace.append(ess_ratio)
        kill_trace.append(1.0 - float(np.mean(alive)))
        max_weight_trace.append(float(np.max(weights / np.sum(weights))))

        if step < n_steps and ess_ratio < cfg.ess_resample:
            indices = _resample_indices(weights, rng)
            s = s[indices]
            v = v[indices]
            alive = alive[indices]
            if running_sum is not None:
                running_sum = running_sum[indices]
            weights = np.ones(cfg.n_paths, dtype=float)
            resample_steps += 1
            resample_streak += 1
            max_resample_streak = max(max_resample_streak, resample_streak)
        else:
            resample_streak = 0

    avg_prices = None
    if running_sum is not None:
        avg_prices = running_sum / (n_steps + 1)

    info_cost = np.zeros(cfg.n_paths, dtype=float)
    q_high = np.zeros(cfg.n_paths, dtype=float)

    return {
        "terminal_prices": s,
        "avg_prices": avg_prices,
        "weights": weights,
        "log_norm": log_norm,
        "info_cost": info_cost,
        "q_high": q_high,
        "ess_trace": ess_trace,
        "kill_trace": kill_trace,
        "kill_step_trace": kill_step_trace,
        "max_weight_trace": max_weight_trace,
        "resample_steps": resample_steps,
        "max_resample_streak": max_resample_streak,
    }


def _simulate_heston_benchmark(
    cfg: OptionExperimentConfig,
    n_paths: int,
    seed: int,
) -> dict[str, object]:
    n_steps = max(1, int(cfg.t * cfg.steps_per_year))
    dt = cfg.t / n_steps
    rng = np.random.default_rng(seed)

    s = np.full(n_paths, cfg.s0, dtype=float)
    v = np.full(n_paths, cfg.heston_v0, dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    running_sum = s.copy() if cfg.payoff_kind == "asian-arithmetic" else None

    for _ in range(1, n_steps + 1):
        z_v = rng.standard_normal(n_paths)
        z_s = rng.standard_normal(n_paths)
        z_s = cfg.heston_rho * z_v + math.sqrt(1.0 - cfg.heston_rho ** 2) * z_s

        # Full truncation Euler for variance with correlated asset shock.
        v_pos = np.maximum(v, 0.0)
        v = v + cfg.heston_kappa * (cfg.heston_theta - v_pos) * dt
        v = v + cfg.heston_xi * np.sqrt(v_pos) * math.sqrt(dt) * z_v
        v = np.maximum(v, 0.0)
        v_bar = 0.5 * (v_pos + v)
        s = s * np.exp((cfg.mu - 0.5 * v_bar) * dt + np.sqrt(v_bar * dt) * z_s)
        if running_sum is not None:
            running_sum += s

        breached = _barrier_breached(s, cfg)
        alive = alive & ~breached

    avg_prices = None
    if running_sum is not None:
        avg_prices = running_sum / (n_steps + 1)

    return {
        "terminal_prices": s,
        "avg_prices": avg_prices,
        "alive": alive,
    }


def build_diagnostics(
    cfg: OptionExperimentConfig,
    sim: dict[str, object],
    gamma_ratio: float,
) -> dict[str, object]:
    ess_trace = np.array(sim["ess_trace"], dtype=float)
    kill_trace = np.array(sim["kill_trace"], dtype=float)
    kill_step_trace = np.array(sim["kill_step_trace"], dtype=float)
    max_weight_trace = np.array(sim["max_weight_trace"], dtype=float)

    ess_min = float(np.min(ess_trace)) if ess_trace.size else 0.0
    ess_mean = float(np.mean(ess_trace)) if ess_trace.size else 0.0
    ess_fail_steps = int(np.sum(ess_trace < cfg.ess_fail))
    max_weight_frac = float(np.max(max_weight_trace)) if max_weight_trace.size else 0.0
    max_kill_step = float(np.max(kill_step_trace)) if kill_step_trace.size else 0.0
    max_kill_frac = float(np.max(kill_trace)) if kill_trace.size else 0.0

    stale_frac = 0.0
    flags = {
        "ess_fail": ess_fail_steps > 0,
        "weight_collapse": max_weight_frac > cfg.weight_max,
        "kill_rate": max_kill_step > cfg.kill_rate_max,
        "over_resample": sim["max_resample_streak"] > cfg.over_resample_limit,
        "stale_data": stale_frac > cfg.stale_frac_max,
        "stiffness": gamma_ratio > cfg.gamma_multiplier,
    }

    return {
        "ess_min": ess_min,
        "ess_mean": ess_mean,
        "ess_fail_steps": ess_fail_steps,
        "max_weight_fraction": max_weight_frac,
        "max_kill_step": max_kill_step,
        "max_kill_fraction": max_kill_frac,
        "resample_steps": sim["resample_steps"],
        "max_resample_streak": sim["max_resample_streak"],
        "stale_frac": stale_frac,
        "gamma_ratio": gamma_ratio,
        "thresholds": {
            "ess_resample": cfg.ess_resample,
            "ess_fail": cfg.ess_fail,
            "weight_max": cfg.weight_max,
            "kill_rate_max": cfg.kill_rate_max,
            "over_resample_limit": cfg.over_resample_limit,
            "stale_frac_max": cfg.stale_frac_max,
            "gamma_multiplier": cfg.gamma_multiplier,
        },
        "flags": flags,
    }


def brownian_bridge_barrier_price(
    cfg: OptionExperimentConfig,
    strike: float,
    n_paths: int,
    sigma_eff: float,
    rng: np.random.Generator,
) -> float:
    if cfg.barrier is None:
        raise ValueError("Brownian bridge benchmark requires a barrier.")
    if sigma_eff <= 0.0:
        payoff = max(cfg.s0 - strike, 0.0) if cfg.option_type == "call" else max(strike - cfg.s0, 0.0)
        return math.exp(-cfg.r * cfg.t) * payoff

    n_steps = max(1, int(cfg.t * cfg.steps_per_year))
    dt = cfg.t / n_steps
    drift = (cfg.r - 0.5 * sigma_eff ** 2) * dt
    vol = sigma_eff * math.sqrt(dt)

    s = np.full(n_paths, cfg.s0, dtype=float)
    alive = np.ones(n_paths, dtype=bool)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        s_next = s * np.exp(drift + vol * z)

        if cfg.barrier_type == "up-and-out":
            hit = (s >= cfg.barrier) | (s_next >= cfg.barrier)
            alive &= ~hit
            mask = alive
            if mask.any():
                log_b_s = np.log(cfg.barrier / s[mask])
                log_b_s_next = np.log(cfg.barrier / s_next[mask])
                p_cross = np.exp(-2.0 * log_b_s * log_b_s_next / (sigma_eff ** 2 * dt))
                cross = rng.random(mask.sum()) < p_cross
                idx = np.where(mask)[0]
                alive[idx[cross]] = False
        else:
            hit = (s <= cfg.barrier) | (s_next <= cfg.barrier)
            alive &= ~hit
            mask = alive
            if mask.any():
                log_s_b = np.log(s[mask] / cfg.barrier)
                log_s_next_b = np.log(s_next[mask] / cfg.barrier)
                p_cross = np.exp(-2.0 * log_s_b * log_s_next_b / (sigma_eff ** 2 * dt))
                cross = rng.random(mask.sum()) < p_cross
                idx = np.where(mask)[0]
                alive[idx[cross]] = False

        s = s_next
        if not alive.any():
            break

    if cfg.option_type == "call":
        payoff = np.maximum(s - strike, 0.0)
    else:
        payoff = np.maximum(strike - s, 0.0)
    payoff = payoff * alive
    return math.exp(-cfg.r * cfg.t) * float(np.mean(payoff))


def example_barrier_failure_config() -> OptionExperimentConfig:
    return OptionExperimentConfig(
        s0=100.0,
        r=0.02,
        mu=0.02,
        t=1.0,
        steps_per_year=252,
        n_paths=20000,
        sigma_low=0.35,
        sigma_high=0.35,
        p_switch=0.0,
        p_high_init=0.0,
        infer_steps=20,
        risk_aversion=0.0,
        info_weight=0.0,
        option_type="call",
        strikes=(100.0,),
        barrier=105.0,
        barrier_type="up-and-out",
        example="barrier-failure",
    )


def example_barrier_rr_report_config() -> OptionExperimentConfig:
    return OptionExperimentConfig(
        s0=100.0,
        r=0.02,
        mu=0.02,
        t=1.0,
        steps_per_year=252,
        n_paths=40000,
        sigma_low=0.25,
        sigma_high=0.25,
        p_switch=0.0,
        p_high_init=0.0,
        infer_steps=20,
        risk_aversion=0.0,
        info_weight=0.0,
        option_type="call",
        strikes=(90.0, 100.0, 110.0, 120.0),
        barrier=130.0,
        barrier_type="up-and-out",
        example="barrier-rr-report",
    )


def example_heston_asian_benchmark_config() -> OptionExperimentConfig:
    return OptionExperimentConfig(
        s0=100.0,
        r=0.02,
        mu=0.02,
        t=1.0,
        steps_per_year=252,
        n_paths=40000,
        dynamics="heston",
        sigma_low=0.2,
        sigma_high=0.2,
        p_switch=0.0,
        p_high_init=0.0,
        infer_steps=20,
        risk_aversion=0.0,
        info_weight=0.0,
        option_type="call",
        payoff_kind="asian-arithmetic",
        strikes=(100.0,),
        barrier=None,
        heston_v0=0.04,
        heston_kappa=1.5,
        heston_theta=0.04,
        heston_xi=0.6,
        heston_rho=-0.7,
        heston_benchmark_paths=120000,
        example="heston-asian-benchmark",
    )


def example_pricing_gas_clone_config() -> OptionExperimentConfig:
    return OptionExperimentConfig(
        s0=100.0,
        r=0.02,
        mu=0.02,
        t=1.0,
        steps_per_year=252,
        n_paths=20000,
        engine="pricing-gas",
        dynamics="regime-gbm",
        sigma_low=0.35,
        sigma_high=0.35,
        p_switch=0.0,
        p_high_init=0.0,
        infer_steps=20,
        risk_aversion=0.0,
        info_weight=0.0,
        option_type="call",
        payoff_kind="vanilla",
        strikes=(100.0,),
        barrier=105.0,
        barrier_type="up-and-out",
        pg_reward_mode="none",
        pg_fitness_mode="weights",
        pg_potential_mode="none",
        pg_potential_scale=1.0,
        pg_reward_scale=1.0,
        pg_companion_epsilon=0.1,
        pg_clone_p_max=1.0,
        pg_clone_epsilon=0.01,
        pg_clone_sigma_x=0.1,
        pg_clone_alpha=0.5,
        example="pricing-gas-clone",
    )


def example_fragile_gas_clone_config() -> OptionExperimentConfig:
    return OptionExperimentConfig(
        s0=100.0,
        r=0.02,
        mu=0.02,
        t=1.0,
        steps_per_year=252,
        n_paths=20000,
        engine="fragile-gas",
        dynamics="regime-gbm",
        sigma_low=0.35,
        sigma_high=0.35,
        p_switch=0.0,
        p_high_init=0.0,
        infer_steps=20,
        risk_aversion=0.0,
        info_weight=0.0,
        option_type="call",
        payoff_kind="vanilla",
        strikes=(100.0,),
        barrier=105.0,
        barrier_type="up-and-out",
        pg_reward_mode="distance",
        pg_fitness_mode="linear",
        pg_potential_mode="none",
        pg_potential_scale=1.0,
        pg_reward_scale=1.0,
        pg_companion_epsilon=0.1,
        pg_clone_p_max=1.0,
        pg_clone_epsilon=0.01,
        pg_clone_sigma_x=0.1,
        pg_clone_alpha=0.5,
        fg_alpha=1.0,
        fg_beta=1.0,
        fg_eta=0.1,
        fg_lambda_alg=0.0,
        fg_sigma_min=1e-8,
        fg_epsilon_dist=1e-8,
        fg_A=2.0,
        fg_rho=None,
        example="fragile-gas-clone",
    )


def run_experiment(cfg: OptionExperimentConfig) -> dict[str, object]:
    if cfg.engine == "pricing-gas":
        if cfg.pg_reward_mode != "none" and len(cfg.strikes) > 1:
            raise ValueError("pricing-gas with reward-based fitness requires a single strike.")
        if cfg.pg_potential_mode != "none" and len(cfg.strikes) > 1:
            raise ValueError("pricing-gas with potential mode requires a single strike.")
    if cfg.engine == "fragile-gas":
        if len(cfg.strikes) > 1:
            raise ValueError("fragile-gas requires a single strike for reward shaping.")
    sim = simulate_pricing_gas(cfg)
    s_terminal = sim["terminal_prices"]
    weights = sim["weights"]
    log_norm = float(sim["log_norm"])
    info_cost = sim["info_cost"]
    avg_prices = sim.get("avg_prices")

    sigma_eff = bs_effective_sigma(cfg)
    payoff_underlier = s_terminal
    if cfg.payoff_kind == "asian-arithmetic":
        if avg_prices is None:
            raise ValueError("Asian payoff requested but avg_prices not available.")
        payoff_underlier = avg_prices

    results = []
    gammas = []
    gap_ratios = []
    rr_diffs = []
    heston_diffs = []
    bb_diffs = []
    heston_benchmark = None
    if cfg.dynamics == "heston" and cfg.heston_benchmark_paths > 0:
        heston_benchmark = _simulate_heston_benchmark(
            cfg,
            cfg.heston_benchmark_paths,
            seed=cfg.seed + 101,
        )
        bench_underlier = heston_benchmark.get("avg_prices")
        if cfg.payoff_kind != "asian-arithmetic":
            bench_underlier = heston_benchmark["terminal_prices"]
        heston_benchmark["payoff_underlier"] = bench_underlier
    bb_applicable = bool(
        cfg.barrier is not None
        and cfg.dynamics == "regime-gbm"
        and math.isclose(cfg.sigma_low, cfg.sigma_high, rel_tol=1e-9, abs_tol=1e-9)
    )
    bb_rng = np.random.default_rng(cfg.seed + 73)
    bb_paths = cfg.bb_paths if cfg.bb_paths > 0 else cfg.n_paths
    for strike in cfg.strikes:
        payoff = option_payoff(payoff_underlier, strike, cfg.option_type)
        if float(np.sum(weights)) <= 0.0:
            smc_price = 0.0
            fragile_price = 0.0
        else:
            smc_price = math.exp(log_norm) * float(np.mean(weights * payoff))
            adjusted = payoff - cfg.info_weight * info_cost
            fragile_price = weighted_entropic_price(adjusted, weights, log_norm, cfg.risk_aversion)
        bs_price = None
        if cfg.payoff_kind == "vanilla":
            bs_price = black_scholes_price(cfg.s0, strike, cfg.t, cfg.r, sigma_eff, cfg.option_type)
            gammas.append(black_scholes_gamma(cfg.s0, strike, cfg.t, cfg.r, sigma_eff))
            gap_ratios.append(abs(smc_price - bs_price) / max(bs_price, 1e-12))
        rr_price = None
        rr_diff = None
        rr_kind = rr_barrier_kind(cfg, strike)
        if rr_kind == "up-and-out-call":
            rr_price = barrier_up_and_out_call_rr(
                cfg.s0,
                strike,
                cfg.barrier if cfg.barrier is not None else 0.0,
                cfg.t,
                cfg.r,
                sigma_eff,
            )
        elif rr_kind == "down-and-out-put":
            rr_price = barrier_down_and_out_put_rr(
                cfg.s0,
                strike,
                cfg.barrier if cfg.barrier is not None else 0.0,
                cfg.t,
                cfg.r,
                sigma_eff,
            )
        if rr_price is not None:
            rr_diff = smc_price - rr_price
            rr_diffs.append(rr_diff)
        heston_price = None
        heston_diff = None
        if heston_benchmark is not None:
            bench_underlier = heston_benchmark["payoff_underlier"]
            bench_alive = heston_benchmark["alive"]
            bench_payoff = option_payoff(bench_underlier, strike, cfg.option_type)
            heston_price = math.exp(-cfg.r * cfg.t) * float(np.mean(bench_payoff * bench_alive))
            heston_diff = smc_price - heston_price
            heston_diffs.append(heston_diff)
        bb_price = None
        bb_diff = None
        if bb_applicable:
            bb_price = brownian_bridge_barrier_price(
                cfg,
                strike,
                n_paths=bb_paths,
                sigma_eff=sigma_eff,
                rng=bb_rng,
            )
            bb_diff = smc_price - bb_price
            bb_diffs.append(bb_diff)
        results.append(
            {
                "strike": float(strike),
                "smc_price": smc_price,
                "fragile_price": fragile_price,
                "black_scholes": bs_price,
                "smc_diff": smc_price - bs_price if bs_price is not None else None,
                "fragile_diff": fragile_price - bs_price if bs_price is not None else None,
                "barrier_rr": rr_price,
                "rr_diff": rr_diff,
                "heston_benchmark": heston_price,
                "heston_diff": heston_diff,
                "bb_mc": bb_price,
                "bb_diff": bb_diff,
            }
        )

    diffs = np.array([row["smc_diff"] for row in results if row["smc_diff"] is not None], dtype=float)
    fragile_diffs = np.array(
        [row["fragile_diff"] for row in results if row["fragile_diff"] is not None],
        dtype=float,
    )
    gamma_ratio = float(max(gammas) / (np.median(gammas) + 1e-12)) if gammas else 0.0
    diagnostics = build_diagnostics(cfg, sim, gamma_ratio)
    barrier_gap_max = float(max(gap_ratios)) if gap_ratios else 0.0
    rr_diffs_arr = np.array(rr_diffs, dtype=float) if rr_diffs else np.array([])
    heston_diffs_arr = np.array(heston_diffs, dtype=float) if heston_diffs else np.array([])
    bb_diffs_arr = np.array(bb_diffs, dtype=float) if bb_diffs else np.array([])

    summary = {
        "smc_mae": float(np.mean(np.abs(diffs))) if diffs.size else None,
        "smc_rmse": float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else None,
        "fragile_mae": float(np.mean(np.abs(fragile_diffs))) if fragile_diffs.size else None,
        "fragile_rmse": float(np.sqrt(np.mean(fragile_diffs ** 2))) if fragile_diffs.size else None,
        "rr_mae": float(np.mean(np.abs(rr_diffs_arr))) if rr_diffs_arr.size else None,
        "rr_rmse": float(np.sqrt(np.mean(rr_diffs_arr ** 2))) if rr_diffs_arr.size else None,
        "heston_mae": float(np.mean(np.abs(heston_diffs_arr))) if heston_diffs_arr.size else None,
        "heston_rmse": float(np.sqrt(np.mean(heston_diffs_arr ** 2))) if heston_diffs_arr.size else None,
        "bb_mae": float(np.mean(np.abs(bb_diffs_arr))) if bb_diffs_arr.size else None,
        "bb_rmse": float(np.sqrt(np.mean(bb_diffs_arr ** 2))) if bb_diffs_arr.size else None,
        "engine": cfg.engine,
        "sigma_eff": sigma_eff,
        "q_high_mean": float(np.mean(sim["q_high"])),
        "barrier_gap_max": barrier_gap_max,
        "barrier_failure": bool(cfg.barrier is not None and barrier_gap_max > 0.5),
        "rr_applicable": bool(rr_diffs_arr.size),
        "heston_benchmark": heston_benchmark is not None,
        "bb_applicable": bb_applicable,
    }

    output = {
        "config": asdict(cfg),
        "summary": summary,
        "diagnostics": diagnostics,
        "results": results,
        "example": cfg.example,
    }

    if cfg.save_json:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pricing_gas_{cfg.option_type}.json"
        out_path.write_text(json.dumps(output, indent=2))

    if cfg.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError:
            print("matplotlib not available; skipping plot.")
        else:
            strikes = [row["strike"] for row in results]
            smc_prices = [row["smc_price"] for row in results]
            bs_prices = [row["black_scholes"] for row in results]
            rr_prices = [row["barrier_rr"] for row in results]
            heston_prices = [row["heston_benchmark"] for row in results]
            plt.figure(figsize=(8, 4))
            plt.plot(strikes, smc_prices, marker="o", label="pricing_gas")
            if any(price is not None for price in bs_prices):
                plt.plot(strikes, bs_prices, marker="x", label="black-scholes")
            if all(price is not None for price in rr_prices):
                plt.plot(strikes, rr_prices, marker="s", label="barrier-rr")
            if all(price is not None for price in heston_prices):
                plt.plot(strikes, heston_prices, marker="^", label="heston-mc")
            plt.xlabel("Strike")
            plt.ylabel("Price")
            plt.title("Pricing Gas vs Black-Scholes")
            plt.legend()
            plt.tight_layout()
            out_path = Path(cfg.output_dir) / f"pricing_gas_{cfg.option_type}.png"
            plt.savefig(out_path)
            plt.close()

    return output


def parse_strikes(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> OptionExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Fragile Pricing Gas option pricing vs Black-Scholes benchmark"
    )
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--steps-per-year", type=int, default=252)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument(
        "--engine",
        type=str,
        default="smc",
        choices=["smc", "pricing-gas", "fragile-gas"],
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="regime-gbm",
        choices=["regime-gbm", "heston"],
    )
    parser.add_argument("--sigma-low", type=float, default=0.15)
    parser.add_argument("--sigma-high", type=float, default=0.35)
    parser.add_argument("--p-switch", type=float, default=0.02)
    parser.add_argument("--p-high-init", type=float, default=0.5)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--risk-aversion", type=float, default=2.0)
    parser.add_argument("--info-weight", type=float, default=0.5)
    parser.add_argument("--option-type", type=str, default="call", choices=["call", "put"])
    parser.add_argument(
        "--payoff-kind",
        type=str,
        default="vanilla",
        choices=["vanilla", "asian-arithmetic"],
    )
    parser.add_argument("--strikes", type=str, default="80,90,100,110,120")
    parser.add_argument("--barrier", type=float, default=None)
    parser.add_argument(
        "--barrier-type",
        type=str,
        default="up-and-out",
        choices=["up-and-out", "down-and-out"],
    )
    parser.add_argument(
        "--pg-reward-mode",
        type=str,
        default="none",
        choices=["none", "payoff", "distance"],
    )
    parser.add_argument(
        "--pg-fitness-mode",
        type=str,
        default="weights",
        choices=["weights", "reward", "linear"],
    )
    parser.add_argument(
        "--pg-potential-mode",
        type=str,
        default="none",
        choices=["none", "distance"],
    )
    parser.add_argument("--pg-potential-scale", type=float, default=1.0)
    parser.add_argument("--pg-reward-scale", type=float, default=1.0)
    parser.add_argument("--pg-companion-epsilon", type=float, default=0.1)
    parser.add_argument("--pg-clone-p-max", type=float, default=1.0)
    parser.add_argument("--pg-clone-epsilon", type=float, default=0.01)
    parser.add_argument("--pg-clone-sigma-x", type=float, default=0.1)
    parser.add_argument("--pg-clone-alpha", type=float, default=0.5)
    parser.add_argument("--heston-v0", type=float, default=0.04)
    parser.add_argument("--heston-kappa", type=float, default=2.0)
    parser.add_argument("--heston-theta", type=float, default=0.04)
    parser.add_argument("--heston-xi", type=float, default=0.5)
    parser.add_argument("--heston-rho", type=float, default=-0.7)
    parser.add_argument("--heston-benchmark-paths", type=int, default=0)
    parser.add_argument("--bb-paths", type=int, default=0)
    parser.add_argument("--fg-alpha", type=float, default=1.0)
    parser.add_argument("--fg-beta", type=float, default=1.0)
    parser.add_argument("--fg-eta", type=float, default=0.1)
    parser.add_argument("--fg-lambda-alg", type=float, default=0.0)
    parser.add_argument("--fg-sigma-min", type=float, default=1e-8)
    parser.add_argument("--fg-epsilon-dist", type=float, default=1e-8)
    parser.add_argument("--fg-a", dest="fg_A", type=float, default=2.0)
    parser.add_argument("--fg-rho", type=float, default=None)
    parser.add_argument("--ess-resample", type=float, default=0.30)
    parser.add_argument("--ess-fail", type=float, default=0.07)
    parser.add_argument("--weight-max", type=float, default=0.12)
    parser.add_argument("--kill-rate-max", type=float, default=0.15)
    parser.add_argument("--over-resample-limit", type=int, default=8)
    parser.add_argument("--stale-frac-max", type=float, default=0.02)
    parser.add_argument("--gamma-multiplier", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs/pricing")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--example",
        type=str,
        default="none",
        choices=[
            "none",
            "barrier-failure",
            "barrier-rr-report",
            "heston-asian-benchmark",
            "pricing-gas-clone",
            "fragile-gas-clone",
        ],
        help="Run a preset scenario (overrides most parameters).",
    )
    args = parser.parse_args()

    if args.example == "barrier-failure":
        cfg = example_barrier_failure_config()
        cfg.output_dir = args.output_dir
        cfg.plot = args.plot
        cfg.save_json = not args.no_save
        cfg.seed = args.seed
        cfg.n_paths = args.n_paths
        return cfg
    if args.example == "barrier-rr-report":
        cfg = example_barrier_rr_report_config()
        cfg.output_dir = args.output_dir
        cfg.plot = args.plot
        cfg.save_json = not args.no_save
        cfg.seed = args.seed
        cfg.n_paths = args.n_paths
        return cfg
    if args.example == "heston-asian-benchmark":
        cfg = example_heston_asian_benchmark_config()
        cfg.output_dir = args.output_dir
        cfg.plot = args.plot
        cfg.save_json = not args.no_save
        cfg.seed = args.seed
        cfg.n_paths = args.n_paths
        if args.heston_benchmark_paths > 0:
            cfg.heston_benchmark_paths = args.heston_benchmark_paths
        return cfg
    if args.example == "pricing-gas-clone":
        cfg = example_pricing_gas_clone_config()
        cfg.output_dir = args.output_dir
        cfg.plot = args.plot
        cfg.save_json = not args.no_save
        cfg.seed = args.seed
        cfg.n_paths = args.n_paths
        return cfg
    if args.example == "fragile-gas-clone":
        cfg = example_fragile_gas_clone_config()
        cfg.output_dir = args.output_dir
        cfg.plot = args.plot
        cfg.save_json = not args.no_save
        cfg.seed = args.seed
        cfg.n_paths = args.n_paths
        return cfg

    mu = args.mu if args.mu is not None else args.r

    return OptionExperimentConfig(
        s0=args.s0,
        r=args.r,
        mu=mu,
        t=args.t,
        steps_per_year=args.steps_per_year,
        n_paths=args.n_paths,
        engine=args.engine,
        dynamics=args.dynamics,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
        p_switch=args.p_switch,
        p_high_init=args.p_high_init,
        infer_steps=args.infer_steps,
        risk_aversion=args.risk_aversion,
        info_weight=args.info_weight,
        option_type=args.option_type,
        payoff_kind=args.payoff_kind,
        strikes=parse_strikes(args.strikes),
        barrier=args.barrier,
        barrier_type=args.barrier_type,
        pg_reward_mode=args.pg_reward_mode,
        pg_fitness_mode=args.pg_fitness_mode,
        pg_potential_mode=args.pg_potential_mode,
        pg_potential_scale=args.pg_potential_scale,
        pg_reward_scale=args.pg_reward_scale,
        pg_companion_epsilon=args.pg_companion_epsilon,
        pg_clone_p_max=args.pg_clone_p_max,
        pg_clone_epsilon=args.pg_clone_epsilon,
        pg_clone_sigma_x=args.pg_clone_sigma_x,
        pg_clone_alpha=args.pg_clone_alpha,
        heston_v0=args.heston_v0,
        heston_kappa=args.heston_kappa,
        heston_theta=args.heston_theta,
        heston_xi=args.heston_xi,
        heston_rho=args.heston_rho,
        heston_benchmark_paths=args.heston_benchmark_paths,
        bb_paths=args.bb_paths,
        fg_alpha=args.fg_alpha,
        fg_beta=args.fg_beta,
        fg_eta=args.fg_eta,
        fg_lambda_alg=args.fg_lambda_alg,
        fg_sigma_min=args.fg_sigma_min,
        fg_epsilon_dist=args.fg_epsilon_dist,
        fg_A=args.fg_A,
        fg_rho=args.fg_rho,
        ess_resample=args.ess_resample,
        ess_fail=args.ess_fail,
        weight_max=args.weight_max,
        kill_rate_max=args.kill_rate_max,
        over_resample_limit=args.over_resample_limit,
        stale_frac_max=args.stale_frac_max,
        gamma_multiplier=args.gamma_multiplier,
        seed=args.seed,
        output_dir=args.output_dir,
        save_json=not args.no_save,
        plot=args.plot,
        example=None,
    )


def main() -> None:
    cfg = parse_args()
    output = run_experiment(cfg)
    print("Pricing Gas vs Black-Scholes results")
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["diagnostics"], indent=2))
    if cfg.example == "barrier-rr-report":
        report = format_rr_report(output["results"])
        if report:
            print("RR Benchmark Report")
            print(report)
    if cfg.example == "heston-asian-benchmark":
        report = format_heston_report(output["results"])
        if report:
            print("Heston Benchmark Report")
            print(report)


if __name__ == "__main__":
    main()
