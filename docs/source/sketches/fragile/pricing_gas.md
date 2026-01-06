---
title: "Fragile Pricing Gas: SMC Option Pricing (Asset Pricing Adaptation)"
---

# Fragile Pricing Gas: SMC Option Pricing (Asset Pricing Adaptation)

## 0. Positioning and Scope

This document rewrites the Geometric Gas proof object as an **asset pricing algorithm**. The result is a **Sequential Monte Carlo (SMC) / Interacting Particle System (IPS)** method specialized for option pricing, consistent with the **Fragile Market** theory in `docs/source/sketches/fragile/pricing.md` {cite}`delmoral2004feynman,doucet2001sequential`.

The core idea:
- **Geometric Gas** = a killed diffusion with selection/cloning.
- **Pricing Gas** = an SMC estimator for path-dependent payoffs under the **Feynman-Kac** representation of prices {cite}`kac1949distribution,oksendal2003stochastic,delmoral2004feynman`.

This is not an equilibrium derivation; it is an **estimator** that can be validated by Sieve checks and benchmarked against Black-Scholes (see `src/experiments/pricing/options.py`).

---

## 1. Mathematical Bridge: Feynman-Kac for Pricing

Let $S_t$ follow a risk-neutral SDE on $[0,T]$ with short rate $r_t$. For payoff $\Pi(S_{0:T})$,

the price is
$$
P_0 = \mathbb{E}^{\mathbb{Q}}\left[ e^{-\int_0^T r_t dt} \Pi(S_{0:T}) \right].
$$

This is a Feynman-Kac expectation {cite}`kac1949distribution,oksendal2003stochastic`. The Geometric Gas algorithm already estimates such expectations using **killing + cloning + weights**, which maps directly to option pricing {cite}`delmoral2004feynman,glasserman2004monte`.

---

## 2. Concept Mapping (Gas -> Finance)

| Geometric Gas Concept | Financial Concept |
| --- | --- |
| Walkers $x_i$ | Monte Carlo paths (asset trajectories) |
| Bounds $B$ / killing | Barrier conditions (knock-out, default) |
| Fitness $V$ | Payoff weight / importance sampling score |
| Cloning (resampling) | Variance reduction / splitting |
| Kinetic dynamics | Asset SDE (GBM, Heston, etc.) |
| QSD | Conditional law of survivors (no barrier hit) |

---

## 3. Pricing Gas Algorithm (SMC Form)

### 3.1 State

Each particle $i$ carries:
- $S_t^i$: asset state (or multi-asset vector),
- $Y_t^i$: path feature state (running average, barrier flag, drawdown),
- $w_t^i$: particle weight.

### 3.2 Dynamics (Risk-Neutral SDE)

Replace BAOAB/viscous dynamics with the pricing SDE. Example GBM:
$$
dS_t = r S_t dt + \sigma S_t dW_t.
$$
For path-dependent options, update $Y_t$ deterministically each step.

### 3.3 Killing (Barrier Conditions)

If a barrier is breached, set $w_t^i = 0$ and mark particle as dead. This directly models knock-outs and default.

### 3.4 Weighting (Feynman-Kac)

Let $G_t(S_t, Y_t)$ be the incremental potential. Update weights:
$$
\tilde{w}_{t+\Delta t}^i = w_t^i \cdot \exp(-r \Delta t) \cdot \exp(-G_t \Delta t).
$$
For standard options, $G_t=0$ and all weighting is applied at maturity via the payoff.

### 3.5 Resampling (Cloning)

When the effective sample size (ESS) drops below a threshold, resample:
- Sample ancestors proportional to $\tilde{w}_{t+\Delta t}^i$.
- Copy their states $S, Y$ and reset weights to uniform.
- This is the financial version of cloning {cite}`delmoral2004feynman,doucet2001sequential`.

Resampling keeps the particle cloud focused on rare events (barrier survival, deep OTM reach) {cite}`cerou2007adaptive,glasserman2004monte`.

### 3.6 Estimator

At maturity:
$$
\hat{P}_0 = \frac{1}{N} \sum_{i=1}^N w_T^i \Pi(S_{0:T}^i).
$$
The estimator is unbiased if weight updates and resampling are implemented correctly (track normalization constants if needed).

---

## 4. Algorithm Modifications vs Geometric Gas (Surgery)

To adapt the Geometric Gas algorithm to pricing:

1. **Replace dynamics**
   - Remove viscous coupling.
   - Use risk-neutral SDE (GBM, Heston, local vol, etc.).

2. **Flatten the potential**
   - No energy wells. The only bias is via weights.
   - Fitness is used as a **weight** or **resampling score**, not a force.

3. **Track weights**
   - Pricing requires explicit Radon-Nikodym correction.
   - Always store and propagate weights.

4. **Barrier killing**
   - Killing becomes a hard boundary event.
   - Survivors are resampled to maintain population.

This converts the Gas optimizer into a **pricing estimator**.

---

## 5. Sieve Alignment (Fragile Market Checks)

Pricing Gas aligns to the Fragile Market Sieve:

- **BoundaryCheck ($\mathrm{Bound}_\partial$):** data feed quality for rates, vols, barriers.
- **Overload/Starve ($\mathrm{Bound}_B$, $\mathrm{Bound}_\Sigma$):** detect over- or under-resampling.
- **Compactness ($C_\mu$):** weights should not collapse to a single particle.
- **Stiffness ($\mathrm{LS}_\sigma$):** monitor payoff curvature; high curvature requires more particles.

Operational diagnostics:
- ESS ratio $\text{ESS}/N$ should stay above a minimum.
- Kill rate should not exceed a stability threshold unless modeled as stress.

---

## 6. Use Cases

### 6.1 Barrier Options

- Kill particles when barrier is hit.
- Clone survivors.
- Estimate survival-adjusted payoff.

Benefit: drastically reduces variance relative to naive MC {cite}`glasserman2004monte`.

### 6.2 Deep OTM Options

- Use a soft fitness potential to push particles toward the strike.
- Resample based on proximity to strike.
- Track weights to keep estimator unbiased.

### 6.3 Basket and High-Dimensional Options

- Diffusion explores correlated asset space.
- Resampling focuses on payoff-relevant correlation regimes.

---

## 7. Minimal Implementation Blueprint

```
initialize N particles at S0
weights w_i = 1/N
for t in time grid:
    propagate SDE for each particle
    update path state Y
    kill particles that breach barriers (w=0)
    apply weight increment exp(-r dt - G dt)
    if ESS < threshold:
        resample proportional to weights
        reset weights to 1/N
return price estimate = sum_i w_i * payoff(S_path)
```

This is the asset-pricing version of the Geometric Gas algorithm {cite}`delmoral2004feynman`.

---

## 8. Output and Benchmarking

A valid pricing run should report:
- price estimate and confidence interval,
- ESS trace over time,
- kill rate trace,
- Sieve diagnostic outcomes.

Benchmark against Black-Scholes for vanilla options. For barrier exotics under GBM, compare to the Reiner-Rubinstein closed-form benchmark (up-and-out calls and down-and-out puts; see {cite}`hull2018options`) implemented as `barrier_rr` in `src/experiments/pricing/options.py`. For stochastic-volatility exotics (e.g., Asian under Heston {cite}`heston1993stochastic`), compare to a high-sample Heston Monte Carlo benchmark (full truncation Euler; see {cite}`andersen2008simple`) reported as `heston_benchmark`. For other exotics, compare to high-sample Monte Carlo or known analytic bounds {cite}`glasserman2004monte`.

---

### 8.1 Example: Pricing Gas Succeeds, BS Fails (Barrier)

Vanilla Black-Scholes ignores barrier killing, so it **overprices** barrier knock-outs. The Pricing Gas estimator does not.

Example (up-and-out call):
- $S_0=100$, $K=100$, $B=105$, $T=1$, $r=0.02$, $\sigma=0.35$.
- Many paths hit the barrier early; the correct price is much lower than vanilla BS.

Run:
```bash
python src/experiments/pricing/options.py --example barrier-failure
```
Expected outcome: `barrier_failure` is `true` and `smc_price` is far below the BS price.

For a quantitative RR benchmark table (SMC vs RR vs BS), run:
```bash
python src/experiments/pricing/options.py --example barrier-rr-report
```

For an explicit **cloning-based** run using the Euclidean Gas clone operator, run:
```bash
python src/experiments/pricing/options.py --example pricing-gas-clone
```

### 8.2 Example: Heston Asian Benchmark (Arithmetic)

Pricing Gas should agree with a high-sample Heston benchmark for a path-dependent exotic.

Example (arithmetic Asian call under Heston):
- $S_0=100$, $K=100$, $T=1$, $r=0.02$.
- Heston: $v_0=0.04$, $\theta=0.04$, $\kappa=1.5$, $\xi=0.6$, $\rho=-0.7$.

Run:
```bash
python src/experiments/pricing/options.py --example heston-asian-benchmark
```
Expected outcome: `heston_benchmark` is `true`, and `heston_mae` is small (Pricing Gas tracks the Heston MC reference).

## 9. Appendix: Mapping to `options.py`

The experiment in `src/experiments/pricing/options.py` instantiates the Pricing Gas blueprint. The canonical cloning implementation lives in `src/fragile/pricing_gas.py` (Euclidean Gas clone operator + pricing SDE).

- **Propagation:** `simulate_pricing_gas` evolves either a two-regime GBM or a Heston SDE (via full truncation Euler) depending on `dynamics`.
- **Barrier killing:** `_barrier_breached` and the alive mask implement knockout/default events (Section 3.3).
- **Weighting:** per-step discounting `exp(-r dt)` updates particle weights (Section 3.4).
- **Resampling:** ESS-triggered resampling keeps the particle cloud alive (Section 3.5).
- **Payoff:** `option_payoff` evaluates the terminal payoff; `payoff_kind=asian-arithmetic` swaps in the arithmetic average.
- **Pricing functional:** `weighted_entropic_price` applies an entropic certainty-equivalent with the SMC normalizer, matching the thermoeconomic SDF idea (Section 4 in `pricing.md`).
- **Benchmark:** `black_scholes_price` is the vanilla reference, `barrier_rr` provides Reiner-Rubinstein barrier benchmarks, and `heston_benchmark` is the high-sample Heston MC reference for Asian exotics.

## 10. Sieve Checklist with Thresholds

Calibrated defaults for **liquid equity index options** (1Y maturity, daily steps, 15-35% vol). Tune for other regimes:

- **ESS ratio:** resample if $\text{ESS}/N < 0.30$; hard fail if $< 0.07$ for more than 5 steps.
- **Weight concentration:** max normalized weight $\max_i w_i > 0.12$ signals collapse (trigger resample).
- **Kill rate:** per-step kill fraction $> 0.15$ indicates barrier domination (increase $N$ or adjust proposal).
- **Over-resampling:** resampling every step for $> 8$ consecutive steps suggests degeneracy.
- **Boundary data quality:** if volatility input or rate feeds are stale for more than $2\%$ of steps, fail BoundaryCheck ($\mathrm{Bound}_\partial$).
- **Stiffness:** if payoff curvature proxy (finite-diff gamma) exceeds a threshold (e.g., 2.5x historical median), increase $N$ and reduce $\Delta t$.

## 11. Summary

The Geometric Gas proof object is a **general IPS engine**. With the modifications above, it becomes a **Fragile Pricing Gas** algorithm for option pricing:
- Correct under Feynman-Kac,
- Efficient for rare events,
- Auditable via Sieve diagnostics,
- Compatible with the theory in `docs/source/sketches/fragile/pricing.md`.

---

## References

```{bibliography}
:filter: docname in docnames
```
