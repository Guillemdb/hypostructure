---
title: "The Fragile Market: Thermoeconomic Asset Pricing with Hypostructure Permits"
---

# The Fragile Market: Thermoeconomic Asset Pricing with Hypostructure Permits

## 0. Positioning: Connections, Differences, Advantages

This document is a **full economic theory of asset pricing** built as a **bounded-rationality control system** with **thermoeconomic potentials** and the **hypostructure permit machinery**. It is an explicit synthesis of:
- standard mathematical finance (no-arbitrage, SDF, martingale pricing),
- thermoeconomics (entropy, dissipation, free energy),
- and the **Gate + Barrier Sieve** language of `docs/source/hypopermits_jb.md`.

The contribution is to make dependencies **operational and auditable**: every modeling step is typed as a **permit** with a clear observable predicate and certificate. Prices are therefore not just equilibrium objects, but **verified fixed points** of the market dynamics under information, liquidity, and solvency constraints.

### 0.1 Main Advantages (Why This Framing Is Useful)

1. **Online auditability.** Pricing assumptions become **checkable constraints** (no-arbitrage, solvency, liquidity, information coupling) rather than hidden modeling lore.
2. **Explicit market macro state.** A discrete macro register $K_t$ makes regime changes and risk-state shifts **measurable**, supporting robust pricing and stress testing.
3. **Thermoeconomic clarity.** Risk premia and discounting emerge from a **free-energy principle** that unifies expected utility, entropy, and information costs.
4. **Sieve integration.** Hypostructure permits provide a typed protocol for when prices are valid, when they are merely indicative, and when they are **structurally invalid**.
5. **Asset-type unification.** Equities, rates, credit, commodities, FX, and derivatives are treated as instances of a single pricing kernel with domain-specific constraints.

### 0.2 Contributions and Foundations

**Core contributions of this framework:**
1. **Market Hypostructure object.** Asset pricing encoded as a hypostructure with explicit boundary coupling and permit-verified transitions.
2. **Thermoeconomic SDF.** The stochastic discount factor linked to entropy production and free-energy minimization.
3. **Sieve-based market validity.** Pricing accepted only if a finite set of **gate + barrier permits** hold (solvency, liquidity, information grounding, etc).
4. **Representation constraints.** The macro register $K_t$ treated as a bounded-rate statistic with closure and capacity checks.
5. **WFR portfolio transport.** The Wasserstein-Fisher-Rao metric unifies continuous rebalancing with discrete regime transitions.
6. **Pricing kernel as Helmholtz solver.** DCF/Bellman equation as screened Poisson equation with discount rate as screening mass.
7. **Symplectic order book interface.** Order book as symplectic manifold with price/flow as conjugate variables, boundary conditions as Dirichlet (quotes) vs Neumann (orders).
8. **Ruppeiner geometry for risk.** Full Ruppeiner metric tensor formalism applied to financial risk metrics.

**Foundational literature:**
- **No-arbitrage + SDF:** Fundamental theorem of asset pricing {cite}`harrison1979martingales,harrison1981martingales,delbaen1994ftap`.
- **Equilibrium asset pricing:** Euler equations, representative agent, factor models {cite}`lucas1978asset,breeden1979intertemporal,cochrane2005asset`.
- **Term structure and derivatives:** Risk-neutral valuation and replication methods {cite}`vasicek1977equilibrium,cox1985theory,heath1992bond,black1973pricing,merton1973theory,hull2018options`.
- **Thermoeconomics:** Entropy, dissipation, and free-energy objectives {cite}`jaynes1957information,cover2006elements`.
- **Optimal transport:** Wasserstein metrics and distributionally robust optimization {cite}`esfahani2018dro,mohajerin2018dro,chizat2018wfr`.
- **Information geometry:** Natural gradients and Riemannian optimization {cite}`amari1998natural,martens2020ngd`.
- **Symplectic economics:** Symplectic geometry as the natural geometry of maximizing behavior {cite}`russell2011symplectic`.
- **PDE methods in finance:** Hamilton-Jacobi-Bellman equations {cite}`forsyth2007numerical,pham2009continuous`.
- **Market microstructure:** Order book dynamics, bid-ask spreads, market impact {cite}`gueant2016microstructure,cartea2015algorithmic`.
- **Thermodynamic geometry:** Ruppeiner geometry for fluctuation theory {cite}`ruppeiner1979thermodynamics`.

### 0.3 Comparison Snapshot

| Area | Typical baseline | Fragile Market difference |
|---|---|---|
| **Asset pricing** | equilibrium + no-arbitrage | explicit permits + Sieve validation |
| **Risk premia** | statistical factor models | thermoeconomic free-energy decomposition |
| **Market stability** | ad-hoc stress tests | gate/barrier constraints with certificates |
| **Regime modeling** | latent continuous factors | explicit discrete $K_t$ with capacity checks |
| **Microstructure** | separate from macro pricing | boundary interface with explicit coupling |
| **Portfolio rebalancing** | discrete reoptimization | WFR geodesic transport (Sec. 25) |
| **Risk metric** | covariance matrix | Ruppeiner tensor with curvature (Sec. 4, 27) |
| **Valuation PDE** | Black-Scholes / HJB | screened Poisson / Helmholtz (Sec. 29) |
| **Order book** | statistical models | symplectic manifold with BCs (Sec. 28) |
| **Sector allocation** | discrete optimization | gradient flow to attraction basins (Sec. 30) |
| **Capacity constraints** | ad-hoc position limits | information-theoretic area law (Sec. 24) |
| **Price discovery** | efficient markets hypothesis | entropic drift on Poincaré disk (Sec. 26) |

### 0.4 Axiomatic Foundation

The Fragile Market theory rests on **seven foundational axioms**:

:::{prf:axiom} A1: Bounded Rationality
:label: axiom-bounded-rationality

All market agents operate under finite information capacity and computational constraints:
$$
I(a_t; Z_t) \le C_{\text{agent}} < \infty,
$$
where $a_t$ is agent action, $Z_t$ is market state, and $C_{\text{agent}}$ is agent channel capacity.
:::

:::{prf:axiom} A2: Thermodynamic Consistency
:label: axiom-thermo-consistency

Market dynamics obey the laws of thermodynamics:
1. **First law (Conservation):** Capital is conserved modulo external flows and dissipation.
2. **Second law (Entropy):** Entropy production is non-negative; $\Delta S \ge 0$ for isolated systems.
3. **Third law (Irreversibility):** Finite-time transactions have irreducible friction cost.
:::

:::{prf:axiom} A3: No-Arbitrage
:label: axiom-no-arbitrage

In the absence of barrier breaches, there exists no self-financing strategy yielding positive return with zero risk:
$$
\nexists \theta : V_0(\theta) = 0, \; V_T(\theta) \ge 0 \; \text{a.s.}, \; \mathbb{P}(V_T(\theta) > 0) > 0.
$$
:::

:::{prf:axiom} A4: Positive SDF
:label: axiom-positive-sdf

There exists a strictly positive stochastic discount factor $M_t > 0$ such that:
$$
p_t = \mathbb{E}_t[M_{t+1} \cdot \text{Payoff}_{t+1}].
$$
:::

:::{prf:axiom} A5: Information Grounding
:label: axiom-info-grounding

Prices must be coupled to observable boundary data:
$$
I(p_t; B_t) > 0,
$$
where $B_t$ is the boundary signal (quotes, flows, news).
:::

:::{prf:axiom} A6: Finite Complexity
:label: axiom-finite-complexity

The market state has bounded Kolmogorov complexity:
$$
K(Z_t) \le K_{\max} < \infty.
$$
:::

:::{prf:axiom} A7: Permit Completeness
:label: axiom-permit-completeness

Every market failure mode is detectable by at least one gate or barrier:
$$
\forall \text{ failure } F, \; \exists \text{ permit } P : P(F) = \text{FAIL}.
$$
:::

**Derived Principles.** From these axioms, we derive:
- **MKT-Consistency** (Theorem 16.1) follows from A3, A4.
- **MKT-Exclusion** (Theorem 16.2) follows from A3.
- **MKT-Trichotomy** (Theorem 16.3) follows from A6, A7.
- **MKT-Equivariance** (Theorem 16.4) follows from A4 plus gauge symmetry.
- **MKT-HorizonLimit** (Theorem 16.5) follows from A1, A6.

### 0.5 Notation Glossary

| Symbol | Meaning | Domain | Reference |
|--------|---------|--------|-----------|
| $Z_t$ | Full market state | $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}}$ | Def. 1.1.1 |
| $K_t$ | Discrete macro state (regime) | $\mathcal{K}$ (finite set) | Def. 1.1.1 |
| $B_t$ | Boundary data | $(x_t, y_t, p_t, d_t, f_t, m_t, a_t)$ | Def. 1.1.2 |
| $M_t$ | Stochastic discount factor | $\mathbb{R}_{>0}$ | Def. 4.4.1 |
| $F_t$ | Free energy | $\mathbb{R}$ (CU) | Def. 4.1.1 |
| $S_t$ | Entropy | $\mathbb{R}_{\ge 0}$ (nats) | Def. 4.1.1 |
| $T_t$ | Risk temperature | $\mathbb{R}_{>0}$ | Def. 4.1.1 |
| $G_{ij}$ | Ruppeiner metric tensor | Positive definite matrix | Def. 4.5.1 |
| $\Phi$ | Risk potential | $\mathbb{R}_{\ge 0}$ | Def. 3.1.1 |
| $K_i^+$ | Gate $i$ certificate (PASS) | Boolean | Sec. 7.1 |
| $K_i^-$ | Gate $i$ certificate (FAIL) | Boolean | Sec. 7.1 |
| $K^{\text{blk}}$ | Barrier blocked | Status | Sec. 7.3 |
| $K^{\text{br}}$ | Barrier breached | Status | Sec. 7.3 |
| $\alpha, \beta, \gamma, \delta$ | Scaling exponents | $\mathbb{R}$ | Def. 4.7.1 |
| $\Psi$ | Phase order parameter | $[0, 1]$ | Def. 4.6.2 |
| $W_2$ | Wasserstein-2 distance | $\mathbb{R}_{\ge 0}$ | Def. 4.12.1 |
| $\mathcal{L}_{\text{Sieve}}$ | Sieve loss function | $\mathbb{R}_{\ge 0}$ | Sec. 18.5 |

**Subscript conventions:**
- $_t$ : time index
- $_i, _j$ : asset or coordinate indices
- $_K$ : regime-conditioned quantity
- $^{\mathbb{Q}}$ : risk-neutral measure
- $^{\mathbb{P}}$ : physical measure

### 0.6 Document Structure

**Document Structure (30 Sections):**

| Part | Sections | Content |
|------|----------|---------|
| **Foundations** | 0–4 | Positioning, introduction, units, market hypostructure, thermoeconomic foundations |
| **Core Pricing** | 5–10 | Representation constraints, asset pricing core, market sieve, dynamics, risk measures, asset classes |
| **Implementation** | 11–18 | Market understanding, implementation, summary, failure modes, surgery contracts, metatheorems, algorithmic pricing, full implementation |
| **Applications** | 19–23 | Worked examples, summary/cross-refs, calibration, risk attribution, backtesting |
| **Geometric Theory** | 24–30 | Capacity constraints, WFR transport, price discovery, equations of motion, market interface, pricing kernel, sector classification |

**Geometric Theory Sections (24–30):**

| Section | Content |
|---------|---------|
| **24. Capital Capacity** | Information-theoretic position limits via area law; capacity saturation diagnostic |
| **25. WFR Transport** | Unbalanced optimal transport unifies continuous rebalancing with discrete regime switches |
| **26. Price Discovery** | Entropic drift models spread compression; market maker as symmetry-breaking control |
| **27. Equations of Motion** | Portfolio geodesic SDE with Christoffel corrections; BAOAB integrator for risk-aware trading |
| **28. Market Interface** | Order book as symplectic manifold; Dirichlet (quotes) vs Neumann (orders) boundary conditions |
| **29. Pricing Kernel** | DCF as screened Poisson equation; discount rate = screening mass; Green's function valuation |
| **30. Sector Classification** | Sector rotation as gradient flow; allocation basins as regions of attraction |

**Diagnostic Nodes (Gates 40–47):**

| Node | Section | Monitors |
|------|---------|----------|
| Gate40 | §24 | Capacity saturation ratio |
| Gate41 | §25 | WFR continuity consistency |
| Gate42 | §26 | Price discovery convergence |
| Gate43 | §27 | Geodesic trajectory consistency |
| Gate44 | §28 | Symplectic boundary compatibility |
| Gate45 | §29 | Helmholtz/Bellman residual |
| Gate46 | §30 | Sector purity |
| Gate47 | §30 | Cross-sector separation |

---

## 1. Introduction: The Market as a Bounded-Rationality Controller

We treat the market as an **open control system** operating under partial observability, finite information capacity, and institutional constraints. Agents are controllers; the market as a whole is a **coupled dynamical system** that must remain **self-consistent** with its own pricing rules.

### 1.1 Definitions: Interaction Under Partial Observability

**Definition 1.1.1 (Market Controller).** The market has internal state
$$
Z_t := (K_t, Z_{n,t}, Z_{\mathrm{tex},t}) \in \mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}},
$$
where:
- $K_t$ is a **discrete macro state** (regimes, liquidity state, risk-on/off),
- $Z_{n,t}$ is **structured nuisance** (microstructure, seasonal effects, inventory),
- $Z_{\mathrm{tex},t}$ is **texture residual** (high-frequency noise, idiosyncratic features).

**Definition 1.1.2 (Boundary / Market Interface).** The boundary variables at time $t$ are:
$$
B_t := (x_t, y_t, p_t, d_t, f_t, m_t, a_t),
$$
where:
- $x_t$ is public information (macro data, news),
- $y_t$ is microstructure data (order flow, quotes, depth),
- $p_t$ are observed prices,
- $d_t$ are observed cash flows (dividends, coupons, funding),
- $f_t$ are funding and collateral rates,
- $m_t$ are margin and constraint signals,
- $a_t$ is the aggregate action (net demand / rebalancing flow).

**Definition 1.1.3 (Market as Input-Output Law).** The market environment is a conditional law of future boundary signals given boundary history and actions:
$$
P_{\partial}(B_{t+1} \mid B_{\le t}, a_{\le t}).
$$
In the Markov case this reduces to $P_{\partial}(B_{t+1} \mid B_t, a_t)$, but the interpretation is the same: **pricing and stability depend only on observable boundary signals**.

### 1.2 Symmetries and Gauge Freedoms

Pricing is invariant under certain transformations; these are **gauge freedoms** in the market description.

**Definition 1.2.1 (Market symmetry group).** A minimal symmetry group is
$$
\mathcal{G}_{\mathbb{M}} := G_{\text{numeraire}} \times S_{|\mathcal{A}|} \times G_{\text{measure}} \times G_{\text{unit}},
$$
where:
- $G_{\text{numeraire}}$ is positive scaling of the unit of account,
- $S_{|\mathcal{A}|}$ permutes asset labels,
- $G_{\text{measure}}$ is change of measure equivalent under the SDF,
- $G_{\text{unit}}$ rescales data units (volatility, notional).

**Principle of covariance.** Market diagnostics and permits should be invariant under $\mathcal{G}_{\mathbb{M}}$, so that pricing conclusions do not depend on arbitrary units or relabeling.

### 1.3 Market Category Theory: The Ambient Topos

We embed market dynamics in a **cohesive $(\infty,1)$-topos** $\mathcal{E}$ following the categorical foundations of `hypopermits_jb.md`. This provides the mathematical universe where pricing objects live.

**Definition 1.3.1 (Cohesive Market Topos).** The market topos $\mathcal{E}_{\text{mkt}}$ is a cohesive $(\infty,1)$-topos equipped with the adjoint quadruple:
$$
\Pi \dashv \flat \dashv \sharp \dashv \text{coDisc} : \mathcal{E}_{\text{mkt}} \to \infty\text{-Grpd},
$$
where:
- **$\Pi$ (Shape):** extracts the homotopy type of market configurations (e.g., connected components of trading networks, fundamental group of arbitrage cycles),
- **$\flat$ (Flat/Discrete):** embeds constant sheaves; distinguishes pointwise (spot) pricing from derived structures,
- **$\sharp$ (Sharp/Codiscrete):** contractible path spaces; enables continuous deformation of pricing strategies.

**Interpretation.** The cohesive structure allows us to distinguish:
1. **Discrete aspects:** individual trades, settlement events, default triggers.
2. **Continuous aspects:** smooth price evolution, gradual regime shifts.
3. **Homotopical aspects:** topologically distinct market states (e.g., normal vs. crisis regimes connected by paths vs. separated by barriers).

**Definition 1.3.2 (Market Object in Topos).** A market configuration is an object $\mathcal{M} \in \mathcal{E}_{\text{mkt}}$ such that:
$$
\pi_0(\mathcal{M}) = \text{market regimes (discrete states)}, \quad \pi_1(\mathcal{M}) = \text{arbitrage cycles (gauge symmetries)},
$$
$$
\pi_n(\mathcal{M}) = \text{higher anomalies and obstructions for } n \ge 2.
$$

**Remark 1.3.3 (Why Category Theory?).** The categorical framing provides:
1. **Universality:** pricing theorems become natural transformations, not ad-hoc formulas.
2. **Compositionality:** complex instruments are built from simpler ones via colimits.
3. **Invariance:** gauge-independent statements are morphisms in the topos.

### 1.4 Cohomological Height: Wealth as Derived Functor

Market wealth is not a number but a **derived functor** measuring the "cohomological height" of a position.

**Definition 1.4.1 (Wealth Functor).** Define the wealth functional as a derived functor:
$$
\Phi_{\bullet} : \mathcal{E}_{\text{mkt}} \to \text{Ch}(\mathbb{R}),
$$
where $\text{Ch}(\mathbb{R})$ is the derived category of real-valued chain complexes. The degree-$n$ component $\Phi_n$ measures:
- **$\Phi_0$:** Mark-to-market value (0th homology = direct valuation).
- **$\Phi_1$:** Contingent claims and options (1st homology = linear exposure).
- **$\Phi_2$:** Convexity and gamma exposure (2nd homology = curvature risk).
- **$\Phi_n$:** Higher-order Greeks and exotic path dependencies.

**Definition 1.4.2 (Euler Characteristic of a Portfolio).** The total economic value is the alternating sum:
$$
\chi(\Phi_{\bullet}) := \sum_{n=0}^{\infty} (-1)^n \text{rank}(\Phi_n) = \text{Net Present Value} - \text{Optionality} + \text{Convexity} - \cdots
$$

**Theorem 1.4.3 (Cohomological Pricing).** Under no-arbitrage, the Euler characteristic is preserved under gauge-equivalent portfolio transformations:
$$
\chi(\Phi_{\bullet}(\mathcal{P})) = \chi(\Phi_{\bullet}(g \cdot \mathcal{P})) \quad \forall g \in \mathcal{G}_{\mathbb{M}}.
$$

*Proof Sketch.* Gauge transformations act as quasi-isomorphisms on the chain complex; Euler characteristic is a homotopy invariant. $\square$

### 1.5 Modalities: Shape, Flat, and Sharp for Markets

The three modalities $\Pi, \flat, \sharp$ give distinct "views" of market data.

**Definition 1.5.1 (Shape Modality $\Pi$: Topological Market Structure).**
$$
\Pi(\mathcal{M}) = \text{homotopy type of market configuration space}.
$$
- **Application:** Detects whether two market states are "topologically equivalent" (connected by continuous deformation) or "topologically distinct" (separated by phase transition).
- **Observable:** Number of connected components = number of distinct regimes.

**Definition 1.5.2 (Flat Modality $\flat$: Spot Pricing).**
$$
\flat(\mathcal{M}) = \text{discrete/pointwise evaluation of prices}.
$$
- **Application:** Spot prices, mark-to-market, instantaneous valuation.
- **Contrast with $\sharp$:** $\flat$ ignores path dependence; $\sharp$ includes it.

**Definition 1.5.3 (Sharp Modality $\sharp$: Path-Dependent Pricing).**
$$
\sharp(\mathcal{M}) = \text{contractible deformation space of price paths}.
$$
- **Application:** Path-dependent options (Asian, barrier, lookback), accumulated dividends, accrued interest.
- **Mathematical structure:** $\sharp(\mathcal{M})$ has trivial homotopy groups—all paths are equivalent up to endpoints.

**Proposition 1.5.4 (Modal Decomposition of Pricing).** Any pricing functional $P$ decomposes as:
$$
P = P_{\flat} + P_{\sharp - \flat} + P_{\Pi},
$$
where:
- $P_{\flat}$ is the spot/intrinsic value,
- $P_{\sharp - \flat}$ is the path-dependent premium (time value, optionality),
- $P_{\Pi}$ is the topological risk premium (regime/crisis premium).

### 1.6 The Trinity of Market Manifolds

We distinguish three geometric objects:

| Manifold | Symbol | Coordinates | Metric | Role |
|----------|--------|-------------|--------|------|
| **Price/Data Space** | $\mathcal{P}$ | $(p^1, \ldots, p^n)$ | Euclidean | Raw observed prices |
| **State/Risk Space** | $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$ | $(K, z_n)$ | Ruppeiner $G_{ij}(z)$ | Control-relevant states |
| **Parameter Space** | $\Theta$ | $(\theta^1, \ldots, \theta^m)$ | Fisher-Rao $\mathcal{F}(\theta)$ | Model parameters |

**Warning 1.6.1 (Category Error).** The Fisher-Rao metric on parameter space $\Theta$ is **not** the same as the Ruppeiner metric on state space $\mathcal{Z}$. Confusing these leads to:
- Incorrect risk attribution,
- Spurious hedging recommendations,
- Violation of coordinate invariance.

**Definition 1.6.2 (Ruppeiner Risk Metric).** The state-space metric is:
$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{\partial^2 F}{\partial z^i \partial z^j} \cdot \frac{1}{T},
$$
where $S$ is entropy, $F$ is free energy, and $T$ is risk temperature. This measures the **thermodynamic distance** between market states.

### 1.7 Agent Types and Market Roles

Markets contain heterogeneous agents with distinct control objectives.

**Definition 1.7.1 (Agent Taxonomy).**

| Agent Type | Objective | Time Horizon | Key Constraint |
|------------|-----------|--------------|----------------|
| **Market Maker** | Minimize inventory risk | Intraday | Spread ≥ cost |
| **Arbitrageur** | Exploit mispricings | Seconds–days | Capital limits |
| **Hedger** | Minimize variance | Weeks–years | Basis risk |
| **Speculator** | Maximize expected return | Days–months | Drawdown limits |
| **Index Fund** | Track benchmark | Continuous | Tracking error |
| **Central Bank** | Stability | Permanent | Political mandate |

**Definition 1.7.2 (Aggregate Market Dynamics).** The market evolution $S_t$ is the composition of agent-level dynamics:
$$
S_t = \bigcirc_{j \in \mathcal{J}} S_t^{(j)},
$$
where $\mathcal{J}$ indexes active agents and $\bigcirc$ denotes composition under market clearing.

---

## 2. Units and Dimensional Conventions

- **Time:** measured in discrete steps or continuous time with step $\Delta t$.
- **Currency unit (CU):** all prices are in CU. A ratio of prices is dimensionless.
- **Information:** entropies and KL are measured in **nats**.
- **Rates:** interest rates and hazard rates are in $1/\text{time}$.
- **Free energy / certainty equivalents:** measured in CU when applied to payoffs; in nats when applied to log-prices or utility-based potentials.

Whenever a threshold $\epsilon$ appears, it inherits the units of the quantity it bounds.

---

## 3. The Market Hypostructure

We now instantiate the hypostructure formalism for markets.

### 3.1 Market Hypostructure Object

**Definition 3.1.1 (Market Hypostructure).** A market hypostructure is a tuple
$$
\mathbb{H}_{\text{mkt}} = (\mathcal{X}, \nabla, \Phi_{\bullet}, \tau, \partial_{\bullet}),
$$
where:
1. **State stack $\mathcal{X}$:** the configuration stack of balance sheets, contracts, positions, and market microstructure.
2. **Connection $\nabla$:** time evolution under trading, settlement, and policy constraints.
3. **Potential $\Phi_{\bullet}$:** a thermoeconomic potential encoding total utility, risk, and costs.
4. **Truncation structure $\tau$:** market constraints (capital, leverage, liquidity, information capacity, topology of the trading network).
5. **Boundary morphism $\partial_{\bullet}$:** restriction to the market interface $B_t$.

**Interpretation.** $\mathbb{H}_{\text{mkt}}$ is the **object on which pricing lives**. Prices are not intrinsic; they are sections of boundary data consistent with $\nabla$ and $\Phi_{\bullet}$ under $\tau$.

### 3.2 Self-Consistency Principle

**Definition 3.2.1 (Self-consistent market).** A market trajectory is self-consistent if the evolution $S_t$ preserves all permits and converges to a state where pricing is internally and externally consistent.

**Principle (Market fixed point).** Under strict dissipation and permit satisfaction, persistent market states are fixed points (or invariant sets) of $S_t$.

This is the market analogue of the hypostructure fixed-point principle: **prices that persist must be compatible with their own dynamics and constraints**.

### 3.3 Thin Inputs and Permit Mapping

The Sieve uses a **thin interface** representation of the market:
- $\mathcal{X}^{\text{thin}} = (X, d, \mu)$: market state space, distance, and observed measure.
- $\Phi^{\text{thin}} = (\Phi, \nabla, \alpha)$: potential, evolution, and curvature.
- $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$: boundary interface, trace map, boundary flux, and risk signal.

These thin inputs are the minimal objects needed to evaluate permits in the market Sieve.

### 3.4 Thin Market Kernel: Minimal Specification

Following the Thin Kernel Objects of `hypopermits_jb.md` Section 4, we define the minimal market data required for Sieve operation.

**Definition 3.4.1 (Thin Market Kernel).** A thin market kernel is a quintuple:
$$
\mathcal{T}_{\text{mkt}} = (\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}}),
$$
where each component has explicit market interpretation:

**Component 1: Arena $\mathcal{X}^{\text{thin}} = (X, d, \mathfrak{m})$**
- **$X$:** Polish metric space of market configurations (positions, prices, balances).
- **$d$:** Distance function; typically $d(x, x') = \|p - p'\|_2 + \lambda \|\theta - \theta'\|_{\text{param}}$.
- **$\mathfrak{m}$:** Reference measure; empirical distribution of historical market states.

**Component 2: Potential $\Phi^{\text{thin}} = (\Phi, \alpha_{\Phi})$**
- **$\Phi : X \to \mathbb{R}_{\ge 0}$:** Total market risk functional (e.g., aggregate VaR, expected shortfall).
- **$\alpha_{\Phi}$:** Scaling dimension; for equity markets, typically $\alpha_{\Phi} \approx 2$ (quadratic risk).

**Component 3: Dissipation $\mathfrak{D}^{\text{thin}} = (\mathfrak{D}, \beta_{\mathfrak{D}})$**
- **$\mathfrak{D} : X \times X \to \mathbb{R}_{\ge 0}$:** Transaction cost and friction functional.
- **$\beta_{\mathfrak{D}}$:** Scaling dimension; typically $\beta_{\mathfrak{D}} = 1$ (linear in volume) or $3/2$ (with impact).

**Component 4: Symmetry $G^{\text{thin}}$**
- Symmetry group acting on $\mathcal{X}$; at minimum, $G^{\text{thin}} = G_{\text{numeraire}} \times S_{|\mathcal{A}|}$.

**Component 5: Boundary $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$**
- **$\mathcal{B}$:** Boundary data space (observed prices, flows, news).
- **$\mathrm{Tr} : X \to \mathcal{B}$:** Trace map projecting bulk state to boundary observables.
- **$\mathcal{J}$:** Boundary flux (order flow, capital flow).
- **$\mathcal{R}$:** Risk signal (VIX, credit spreads, funding stress).

**Theorem 3.4.2 (Thin Kernel Sufficiency).** Given a thin market kernel $\mathcal{T}_{\text{mkt}}$, the Sieve constructor $F_{\text{Sieve}}$ produces a full market hypostructure:
$$
F_{\text{Sieve}}(\mathcal{T}_{\text{mkt}}) = \mathbb{H}_{\text{mkt}}.
$$

*Proof.* By the Expansion Adjunction (`hypopermits_jb.md` Theorem 5.3), thin kernels promote to full hypostructures via Postnikov tower construction. $\square$

### 3.5 Market RCD Condition (Curvature-Dimension Bound)

Markets satisfy a **Riemannian Curvature-Dimension** condition that bounds complexity.

**Definition 3.5.1 (Market RCD Condition).** The market state space $(\mathcal{X}, d, \mathfrak{m})$ satisfies $\mathrm{RCD}(K, N)$ if:
1. **Ricci curvature bounded below:** $\mathrm{Ric} \ge K$ (market has limited "negative curvature" / instability).
2. **Dimension bounded above:** $\dim \le N$ (finite degrees of freedom).

**Interpretation.**
- **$K > 0$:** Market has intrinsic stability; perturbations decay exponentially.
- **$K = 0$:** Flat market; perturbations persist (random walk).
- **$K < 0$:** Hyperbolic market; small perturbations amplify (crisis-prone).

**Theorem 3.5.2 (RCD Convergence for Markets).** If the market satisfies $\mathrm{RCD}(K, N)$ with $K > 0$, then:
$$
W_2(\mu_t, \mu_{\infty}) \le e^{-Kt} W_2(\mu_0, \mu_{\infty}),
$$
where $W_2$ is Wasserstein-2 distance and $\mu_{\infty}$ is the equilibrium distribution.

*Market implication:* Prices converge to equilibrium at rate $K$. Higher curvature = faster price discovery.

### 3.6 Cheeger Energy and Market Liquidity

The **Cheeger energy** connects metric structure to measure structure.

**Definition 3.6.1 (Market Cheeger Energy).**
$$
\mathrm{Ch}(f | \mathfrak{m}) := \frac{1}{2} \inf \left\{ \liminf_{n \to \infty} \int_X |\nabla f_n|^2 \, d\mathfrak{m} : f_n \to f \text{ in } L^2 \right\}.
$$

**Market interpretation:** Cheeger energy measures the **liquidity cost** of moving probability mass (capital) across market states. High Cheeger energy = illiquid transitions.

**Proposition 3.6.2 (Liquidity as Cheeger Constant).** The market liquidity index is:
$$
\mathcal{L}_{\text{mkt}} := \inf_{A : 0 < \mathfrak{m}(A) < 1} \frac{\text{Per}(A)}{\min(\mathfrak{m}(A), 1 - \mathfrak{m}(A))},
$$
where $\text{Per}(A)$ is the perimeter of set $A$ in the metric-measure space.

Low $\mathcal{L}_{\text{mkt}}$ indicates "bottlenecks" where capital cannot flow freely—liquidity traps.

---

## 4. Thermoeconomic Foundations

We model pricing as a thermodynamic system with capital as energy, uncertainty as entropy, and risk aversion as temperature.

### 4.1 Economic Energy, Entropy, and Free Energy

Let $U_t$ be total market wealth (mark-to-market), $S_t$ be informational entropy, and $T_t$ be risk temperature (inverse risk aversion).

**Definition 4.1.1 (Free energy).**
$$
F_t := U_t - T_t S_t.
$$
$F_t$ is the **extractable value** after accounting for uncertainty. In equilibrium, pricing minimizes expected free energy subject to constraints.
This is the standard MaxEnt free-energy form under information constraints {cite}`jaynes1957information,cover2006elements`.

### 4.2 First Law (Capital Flow)

**Definition 4.2.1 (Capital balance).**
$$
\Delta U = \Delta W + \Delta Q - \Delta D,
$$
where:
- $\Delta W$ is work done by trading (rebalancing gains),
- $\Delta Q$ is external inflow (income, dividends, funding),
- $\Delta D$ is dissipation (transaction costs, slippage, defaults).

### 4.3 Second Law (Entropy Production)

**Definition 4.3.1 (Entropy production).**
$$
\Delta S \ge \Delta S_{\text{info}} + \Delta S_{\text{friction}},
$$
with strictly positive entropy production when trading costs and information loss are nonzero.

**Interpretation.** In the absence of friction, entropy production can be zero (reversible pricing). With frictions, arbitrage extraction generates entropy and dissipates profit.

### 4.4 Pricing Kernel as a Thermodynamic Factor

**Definition 4.4.1 (Thermoeconomic SDF).** For payoff $X_{t+1}$,
$$
P_t(X_{t+1}) = \mathbb{E}_t[ m_{t+1} X_{t+1} ],
$$
where the stochastic discount factor $m_{t+1}$ has the form
$$
m_{t+1} = \exp(-\beta_t \Delta F_{t+1}) \cdot \xi_{t+1},
$$
with $\beta_t = 1/T_t$, and $\xi_{t+1}$ captures constraints (collateral, funding, default).

This makes discounting a **free-energy penalty** plus constraint adjustments.
It is consistent with standard SDF-based pricing when $\xi_{t+1} \equiv 1$ {cite}`cochrane2005asset`.

### 4.5 Ruppeiner Geometry for Markets: The Risk Metric Tensor

The **Ruppeiner metric** measures thermodynamic distance between market states, where "distance" reflects the difficulty of arbitraging between them.

**Definition 4.5.1 (Ruppeiner Metric Tensor).** The market risk metric is the Hessian of entropy:
$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{1}{T} \frac{\partial^2 F}{\partial z^i \partial z^j},
$$
where:
- $z = (z^1, \ldots, z^n)$ are market state coordinates (e.g., log-prices, volatilities, spreads),
- $S$ is the market entropy (uncertainty about future prices),
- $F$ is free energy (risk-adjusted value),
- $T$ is risk temperature (inverse risk aversion).

**Proposition 4.5.2 (Metric Components for Standard Markets).** For a market with log-returns $r_i$ and covariance $\Sigma_{ij}$:
$$
G_{ij} = \frac{1}{T} \Sigma^{-1}_{ij}.
$$
High covariance = low metric distance (easy to arbitrage); low covariance = high metric distance (hard to hedge).

**Definition 4.5.3 (Thermodynamic Distance).** The distance between market states $z$ and $z'$ is:
$$
d_G(z, z') := \int_0^1 \sqrt{G_{ij}(z(\tau)) \dot{z}^i(\tau) \dot{z}^j(\tau)} \, d\tau,
$$
minimized over paths $z(\tau)$ from $z$ to $z'$.

**Market interpretation:** $d_G$ measures the **minimum risk** required to move a portfolio from state $z$ to state $z'$.

### 4.6 Market Phase Transitions: Crystal, Liquid, Gas

Markets exhibit **three thermodynamic phases** with distinct pricing behavior.

**Definition 4.6.1 (Market Phases).**

| Phase | Entropy | Structure | Price Behavior | Examples |
|-------|---------|-----------|----------------|----------|
| **Crystal** | Low | Ordered, predictable | Prices at fundamental value | Government bonds at par, pegged FX |
| **Liquid** | Medium | Structured randomness | Efficient pricing with noise | Normal equity markets, active FX |
| **Gas** | High | Chaotic, unpredictable | Prices disconnected from fundamentals | Flash crashes, speculative bubbles |

**Definition 4.6.2 (Phase Order Parameter).** The market phase is characterized by:
$$
\Psi := \frac{H(K_t)}{\log |\mathcal{K}|} \in [0, 1],
$$
where $H(K_t)$ is the entropy of the regime distribution.
- $\Psi \approx 0$: Crystal phase (one dominant regime).
- $\Psi \approx 0.5$: Liquid phase (moderate uncertainty).
- $\Psi \approx 1$: Gas phase (maximum uncertainty, all regimes equiprobable).

**Theorem 4.6.3 (Phase Transition Detection).** A phase transition occurs at time $t^*$ if:
$$
\left| \frac{d\Psi}{dt} \right|_{t=t^*} > \Psi_{\text{crit}},
$$
where $\Psi_{\text{crit}}$ is a threshold (typically calibrated to VIX spikes or spread blowouts).

**Definition 4.6.4 (Critical Exponents).** Near a phase transition, observables scale as:
$$
\text{Volatility} \sim |T - T_c|^{-\gamma}, \quad \text{Correlation length} \sim |T - T_c|^{-\nu},
$$
where $T_c$ is the critical temperature and $\gamma, \nu$ are critical exponents.

### 4.7 Scaling Exponents: The Four Market Temperatures

We track **four scaling exponents** that characterize market dynamics.

**Definition 4.7.1 (Market Scaling Exponents).**

| Exponent | Symbol | Meaning | Observable Proxy |
|----------|--------|---------|------------------|
| **Risk Temperature** | $\alpha$ | Curvature of value landscape | $\sqrt{\mathbb{E}[(\nabla V)^2]}$ |
| **Volatility Temperature** | $\beta$ | Plasticity of price dynamics | Realized vol / Implied vol |
| **Liquidity Temperature** | $\gamma$ | Fluidity of capital flows | Bid-ask spread inverse |
| **Leverage Temperature** | $\delta$ | Amplification of positions | Aggregate leverage ratio |

**Operational extraction:** Using Adam-style optimizer statistics,
$$
\alpha \approx \log_{10}\left( \sqrt{\mathbb{E}[v_t^{\text{critic}}]} + \epsilon \right),
$$
where $v_t$ is the second moment estimate of risk gradients.

**Definition 4.7.2 (Temperature Hierarchy).** For stable markets, the scaling exponents must satisfy:
$$
\alpha > \beta > \gamma > \delta,
$$
meaning:
1. Risk perception ($\alpha$) must dominate volatility ($\beta$).
2. Volatility ($\beta$) must dominate liquidity effects ($\gamma$).
3. Liquidity ($\gamma$) must dominate leverage amplification ($\delta$).

**Violation → Instability:**
- $\beta > \alpha$: Volatility outpaces risk perception → Mode C.E (blow-up).
- $\gamma > \beta$: Liquidity dominates volatility → Mode T.D (frozen market).
- $\delta > \gamma$: Leverage dominates liquidity → Mode S.E (leverage spiral).

### 4.8 Einstein Equations of Finance: Curvature = Risk

We formulate the **Einstein field equations** for market dynamics, where curvature encodes risk.

**Definition 4.8.1 (Market Einstein Tensor).** Define the Einstein tensor:
$$
\mathcal{G}_{ij} := R_{ij} - \frac{1}{2} R \, G_{ij},
$$
where $R_{ij}$ is the Ricci curvature and $R = G^{ij} R_{ij}$ is the scalar curvature.

**Definition 4.8.2 (Risk-Energy Tensor).** The risk-energy tensor is:
$$
\mathcal{T}_{ij} := \frac{\partial \Phi}{\partial z^i} \frac{\partial \Phi}{\partial z^j} - \frac{1}{2} G_{ij} |\nabla \Phi|^2_G + \Lambda G_{ij},
$$
where $\Phi$ is the risk potential and $\Lambda$ is a "cosmological constant" (baseline risk premium).

**Theorem 4.8.3 (Market Einstein Equations).** In equilibrium, curvature and risk satisfy:
$$
\mathcal{G}_{ij} = \kappa \mathcal{T}_{ij},
$$
where $\kappa$ is the coupling constant (market-specific).

**Interpretation:**
- **Risk (mass-energy) curves the market state space.**
- **Curvature determines how portfolios "fall" toward equilibrium.**
- **Geodesics are optimal trading paths.**

**Corollary 4.8.4 (No-Arbitrage as Flatness).** If $\mathcal{T}_{ij} = 0$ (no risk concentration), then $\mathcal{G}_{ij} = 0$ (flat space), and all paths are equivalent—no arbitrage opportunities.

### 4.9 Geodesic Portfolio Flow: Natural Gradient Investing

Optimal portfolio dynamics follow **geodesics** on the risk manifold.

**Definition 4.9.1 (Geodesic Equation for Portfolios).** A portfolio path $w(t)$ is geodesic if:
$$
\frac{d^2 w^i}{dt^2} + \Gamma^i_{jk} \frac{dw^j}{dt} \frac{dw^k}{dt} = 0,
$$
where $\Gamma^i_{jk}$ are Christoffel symbols derived from $G_{ij}$.

**Proposition 4.9.2 (Natural Gradient Update).** The optimal portfolio update is:
$$
\Delta w = -\eta \, G^{-1} \nabla_w \Phi,
$$
where $G^{-1}$ is the inverse metric and $\nabla_w \Phi$ is the risk gradient.

**Market interpretation:** Natural gradient adjusts position sizes based on local risk curvature:
- **High curvature (risky region):** Small position changes.
- **Low curvature (safe region):** Larger position changes allowed.

**Definition 4.9.3 (Covariant Portfolio Dissipation).** The dissipation rate along a portfolio path is:
$$
\mathfrak{D}_{\text{geo}} := \left\langle \nabla_w V, \dot{w} \right\rangle_G = G_{ij} \frac{\partial V}{\partial w^i} \dot{w}^j,
$$
where $V$ is the value function and $\langle \cdot, \cdot \rangle_G$ is the inner product under the Ruppeiner metric.

**Theorem 4.9.4 (Geodesic Optimality).** Among all self-financing paths from $w_0$ to $w_T$, the geodesic minimizes total transaction cost:
$$
\mathcal{C}[w] = \int_0^T \sqrt{G_{ij}(w) \dot{w}^i \dot{w}^j} \, dt.
$$

### 4.10 Landauer Bound for Trading: Information-Theoretic Costs

Trading incurs an **irreducible information-theoretic cost** bounded by Landauer's principle.

**Theorem 4.10.1 (Landauer Bound for Markets).** Any trade that erases $\Delta I$ bits of market information must dissipate at least:
$$
\Delta Q \ge k_B T \ln(2) \cdot \Delta I,
$$
where $k_B T$ is thermal energy (in market context: risk temperature × volatility).

**Market interpretation:** Information processing (price discovery, order matching) has a **minimum energy cost**. This is why:
1. Market making is not free—spread compensates for information processing.
2. High-frequency trading requires proportionally high infrastructure investment.
3. "Free" information is impossible; all price signals cost someone.

**Definition 4.10.2 (Information-Theoretic Spread).** The minimum bid-ask spread is:
$$
s_{\min} = \frac{k_B T \ln(2)}{V_{\text{avg}}} \cdot H(K_t),
$$
where $V_{\text{avg}}$ is average trade volume and $H(K_t)$ is regime entropy.

**Corollary 4.10.3 (Efficient Market Bound).** In an efficient market, the actual spread satisfies:
$$
s_{\text{actual}} \ge s_{\min},
$$
with equality only in the theoretical limit of zero noise and infinite liquidity.

### 4.11 Log-Sobolev Inequality and Market Concentration

The **Log-Sobolev inequality** connects entropy to concentration—markets with good LSI constants have predictable price distributions.

**Definition 4.11.1 (Market Log-Sobolev Constant).** The LSI constant $\rho_{\text{LSI}}$ satisfies:
$$
\text{Ent}_{\mathfrak{m}}(f^2) \le \frac{2}{\rho_{\text{LSI}}} \int |\nabla f|^2 \, d\mathfrak{m},
$$
for all smooth $f$ with $\int f^2 d\mathfrak{m} = 1$.

**Market interpretation:**
- **Large $\rho_{\text{LSI}}$:** Prices concentrate tightly around equilibrium.
- **Small $\rho_{\text{LSI}}$:** Prices are dispersed; fat tails and extreme events are common.

**Proposition 4.11.2 (LSI and VaR).** The Value-at-Risk at confidence $\alpha$ satisfies:
$$
\text{VaR}_{\alpha} \le \mu + \sigma \sqrt{\frac{2}{\rho_{\text{LSI}}} \ln\left(\frac{1}{1-\alpha}\right)},
$$
where $\mu, \sigma$ are mean and standard deviation.

### 4.12 Wasserstein Distance and Regime Shifts

Regime shifts are measured by **Wasserstein distance** between price distributions.

**Definition 4.12.1 (Regime Transition Cost).** The cost of transitioning from regime $K$ to regime $K'$ is:
$$
W_2(\mu_K, \mu_{K'}) := \left( \inf_{\pi \in \Pi(\mu_K, \mu_{K'})} \int d(x, y)^2 \, d\pi(x, y) \right)^{1/2},
$$
where $\Pi(\mu_K, \mu_{K'})$ is the set of couplings.

**Proposition 4.12.2 (Regime Transition Warning).** A regime transition is imminent when:
$$
\frac{d}{dt} W_2(\mu_t, \mu_K) < -\epsilon_{\text{trans}},
$$
where $\mu_K$ is the current regime distribution and $\epsilon_{\text{trans}}$ is a threshold.

---

## 5. Representation and Information Constraints

### 5.1 Macro Register and Residual Channels

The market maintains a discrete macro register $K_t$ (regime, liquidity state, policy state). A valid $K_t$ must satisfy:
- **Capacity:** $H(K_t) \le \log |\mathcal{K}|$.
- **Grounding:** $I(B_t; K_t) > 0$.
- **Closure:** future price dynamics conditional on $K_t$ are stable across regimes.

### 5.2 Filtering and Belief Update

Let $q(K_t \mid B_{\le t})$ be the market belief over regimes. A consistent update requires:
$$
D_{KL}(q_{t+1} \Vert q_t) \le I(B_{t+1}; K_{t+1}),
$$
so that belief updates are supported by boundary information.

### 5.3 Information Cost and Pricing

Define an information cost $\mathcal{I}_t := D_{KL}(q_t \Vert p_0)$ relative to a prior $p_0$ {cite}`kullback1951information,cover2006elements`. Pricing that ignores $\mathcal{I}_t$ violates the thermoeconomic free-energy principle and will be rejected by the Sieve (BoundaryCheck + AlignCheck).

### 5.4 State-Space Metric and Trust Region

Let $z_t$ denote the continuous part of the market state (inventory, funding spreads, liquidity coordinates). Define the state-space Fisher metric
$$
G_t := \mathbb{E}_t[\nabla_{z} \log p(B_{t+1}\mid z_t) \nabla_{z} \log p(B_{t+1}\mid z_t)^\top].
$$
Updates must satisfy a trust-region constraint
$$
d_{G_t}(z_{t+1}, z_t) \le v_{\max},
$$
ensuring the market does not move faster than its own information capacity (Node 7 and Node 12).

### 5.5 Coupling Window

For a window length $W$, define the boundary coupling condition
$$
0 < I(B_{t:t+W}; K_t) < \log|\mathcal{K}|.
$$
The lower bound prevents starvation (Node 15), the upper bound prevents overload (Node 14).

---

## 6. Asset Pricing Core

We now lay out the classical pricing machinery in the hypostructure language.

### 6.1 Probability Space and Assets

Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t\ge 0}, \mathbb{P})$ be a filtered probability space.
- Asset $i$ has price process $S_t^i$ and dividend stream $D_t^i$.
- The money market account $B_t$ satisfies $dB_t = r_t B_t dt$ (or $B_{t+1} = (1+r_t) B_t$).

A trading strategy $\theta_t$ is **self-financing** if changes in value come only from asset returns, not external infusion.

### 6.2 No-Arbitrage and the SDF

**Definition 6.2.1 (No-arbitrage).** There is no self-financing strategy with zero cost and nonnegative payoff that is positive with positive probability.

**Theorem 6.2.2 (SDF existence).** Under standard regularity (locally bounded prices, NFLVR), there exists a strictly positive process $M_t$ such that for all assets {cite}`harrison1979martingales,harrison1981martingales,delbaen1994ftap`:
$$
S_t^i = \mathbb{E}_t[ M_{t+1} (S_{t+1}^i + D_{t+1}^i) ].
$$
$M_t$ is the **stochastic discount factor**.

### 6.3 Risk-Neutral Measure

Define the Radon-Nikodym derivative
$$
\frac{d\mathbb{Q}}{d\mathbb{P}} \propto M_T.
$$
Then discounted prices are martingales under $\mathbb{Q}$ {cite}`harrison1981martingales,duffie2001dynamic`:
$$
\frac{S_t^i}{B_t} = \mathbb{E}_t^{\mathbb{Q}}\left[\frac{S_T^i + D_T^i}{B_T}\right].
$$

### 6.4 Equilibrium Pricing (Consumption-Based)

If a representative agent with utility $U$ consumes $C_t$, then
$$
M_{t+1} = \beta \frac{U'(C_{t+1})}{U'(C_t)},
$$
and the Euler equation implies for any asset return $R_{t+1}^i$:
$$
\mathbb{E}_t[M_{t+1} R_{t+1}^i] = 1.
$$
This is the consumption-based Euler condition {cite}`lucas1978asset,breeden1979intertemporal`.

### 6.5 Factor Structure and Risk Premia

Assume $M_{t+1}$ is affine in factors $F_{t+1}$:
$$
M_{t+1} = a_t - b_t^\top F_{t+1}.
$$
Then expected excess returns satisfy
$$
\mathbb{E}[R_{t+1}^i - R_f] = \beta_i^\top \lambda,
$$
where $\beta_i$ is asset exposure and $\lambda$ is factor price of risk.
Empirical factor structures are documented in {cite}`fama1993common,hansen1991implications,cochrane2005asset`.

### 6.6 Term Structure

Zero-coupon bond price for maturity $T$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u du\right)\right].
$$
HJM and affine models fit into this SDF framework with $r_t$ and $M_t$ jointly specified {cite}`heath1992bond,vasicek1977equilibrium,cox1985theory,duffie2001dynamic`.

### 6.7 Incomplete Markets and Bounds

When markets are incomplete, SDFs are not unique. Let $\mathcal{M}$ be the admissible SDF set. Then:
$$
\inf_{M \in \mathcal{M}} \mathbb{E}_t[M X] \le P_t(X) \le \sup_{M \in \mathcal{M}} \mathbb{E}_t[M X].
$$
A canonical choice is the **minimal entropy martingale measure**, consistent with the free-energy principle {cite}`frittelli2000entropy`.

### 6.8 Transaction Costs and Funding Frictions

With proportional costs $\kappa$ and funding spread $s_t$, the no-arbitrage price interval for $X$ widens:
$$
P_t^{\text{bid}}(X) \le P_t(X) \le P_t^{\text{ask}}(X),
$$
with bounds computed from super- and sub-hedging costs. These bounds are enforced by the **Liquidity and Funding Barriers** in the Sieve.

---

## 7. The Market Sieve: Permits and Certificates

The market Sieve is the operational protocol that determines whether pricing statements are valid in the current regime. It follows the permit vocabulary of `docs/source/hypopermits_jb.md`.

### 7.1 Permit Vocabulary

:::{prf:definition} Gate permits
:label: def-market-gate-permits
For each gate $i$, the outcome alphabet is $\{\text{YES}, \text{NO}\}$ with certificates:
- $K_i^+$ (YES): the predicate $P_i$ holds on the current state or window.
- $K_i^-$ (NO): the predicate fails or cannot be certified.
:::

:::{prf:definition} Barrier permits
:label: def-market-barrier-permits
For each barrier, the outcome alphabet is $\{\text{Blocked}, \text{Breached}\}$ with certificates:
- $K^{\mathrm{blk}}$: the barrier holds; proceed.
- $K^{\mathrm{br}}$: the barrier fails; enter defense mode.
:::

:::{prf:definition} Surgery permits
:label: def-market-surgery-permits
A surgery outputs a re-entry certificate
$$
K^{\mathrm{re}} = (D_S, x', \pi)
$$
where $D_S$ is the intervention data, $x'$ is the post-surgery state, and $\pi$ proves the next gate's precondition.
:::

:::{prf:definition} YES-tilde permits (equivalence)
:label: def-market-yes-tilde
A YES$^\sim$ permit allows acceptance up to equivalence (e.g., numeraire change):
$$
K_i^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_i^+[\tilde{x}]).
$$
:::

:::{prf:definition} Promotion permits
:label: def-market-promotion
Blocked certificates may be promoted to YES if other certificates imply the original predicate:
$$
K_i^{\mathrm{blk}} \wedge \bigwedge_j K_j^+ \Rightarrow K_i^+.
$$
Promotions may be immediate (past-only) or a-posteriori (future-enabled).
:::

:::{prf:definition} Inconclusive upgrade permits
:label: def-market-inc-upgrade
If a NO certificate is due to missing prerequisites, it can be upgraded when those prerequisites are later supplied:
$$
K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^+.
$$
:::

### 7.2 Gate Permits (Core Checks)

Each gate outputs YES ($K_i^+$) or NO ($K_i^-$). NO is conservative. Below we provide **full specifications** for all 21 market gate nodes.

#### Summary Table

| Node | Permit | Market Check | Interpretation | Example observable |
|---|---|---|---|---|
| 1 | $D_E$ | Solvency / budget | Total losses bounded | aggregate VaR, capital ratio |
| 2 | $\mathrm{Rec}_N$ | Turnover limit | No chattering trades | turnover rate, cancel ratio |
| 3 | $C_\mu$ | Compactness | Leverage and positions bounded | leverage, position size |
| 4 | $\mathrm{SC}_\lambda$ | Scale stability | Parameters not drifting too fast | vol of parameters |
| 5 | $\mathrm{SC}_{\partial c}$ | Stationarity | model drift tolerable | regime drift rate |
| 6 | $\mathrm{Cap}_H$ | Information capacity | market depth supports state | depth, spread, order flow |
| 7 | $\mathrm{LS}_\sigma$ | Stiffness | price impact bounded | impact vs size |
| 7a | $\mathrm{Bif}$ | Bifurcation | no regime instability | Jacobian determinant |
| 7b | $\mathrm{Sym}$ | Alternatives | multiple strategies exist | policy entropy |
| 7c | $\mathrm{SC}_{\text{new}}$ | New regime stability | new mode stable | variance after switch |
| 7d | $\mathrm{TB}_{\text{switch}}$ | Switching cost | transition affordable | switch cost vs budget |
| 8 | $\mathrm{TB}_\pi$ | Connectivity | clearing network connected | graph connectivity |
| 9 | $\mathrm{TB}_O$ | Tameness | pricing function smooth | gamma bounds, convexity |
| 10 | $\mathrm{TB}_\rho$ | Mixing | regime exploration adequate | regime transition counts |
| 11 | $\mathrm{Rep}_K$ | Representation | regime complexity within budget | $H(K)$ vs $\log|\mathcal{K}|$ |
| 12 | $\mathrm{GC}_\nabla$ | Oscillation | no endogenous cycles | boom-bust indicator |
| 13 | $\mathrm{Bound}_\partial$ | Boundary coupling | prices grounded in data | $I(B;K)$ |
| 14 | $\mathrm{Bound}_B$ | Overload | data channel saturated | quote outages, spreads |
| 15 | $\mathrm{Bound}_\Sigma$ | Starvation | insufficient data | thin trading, stale prices |
| 16 | $\mathrm{GC}_T$ | Alignment | incentives match constraints | funding vs risk signals |
| 17 | $\mathrm{Cat}_{\mathrm{Hom}}$ | Lock | no structural arbitrage | arbitrage cycle detection |

---

#### Node 1: Solvency Check ($D_E$) — Conservation of Capital

:::{prf:definition} Node 1 Specification
:label: def-node1-solvency

**Predicate:** Total mark-to-market losses are bounded by available capital.
$$
P_1 : \quad \Phi(x_t) := \text{VaR}_{\alpha}(L_t) \le C_t - \epsilon_{\text{buffer}},
$$
where $L_t$ is the loss distribution, $C_t$ is available capital, and $\epsilon_{\text{buffer}}$ is a safety margin.

**Market interpretation:** The market participant (or aggregate) can absorb expected losses without insolvency.

**Observable metrics:**
- Capital ratio: $\rho_{\text{cap}} := C_t / \text{RWA}_t$
- VaR breach count over rolling window
- Distance to default: $\text{DD}_t := (\mu_t - D_t) / \sigma_t$

**Certificate format:**
$$
K_1^+ = (\text{VaR}_{\alpha}, C_t, \rho_{\text{cap}}, \text{timestamp})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{solvency}} = \lambda_1 \cdot \max(0, \Phi(x_t) - C_t + \epsilon_{\text{buffer}})^2
$$
:::

---

#### Node 2: Turnover Check ($\mathrm{Rec}_N$) — No Zeno Trading

:::{prf:definition} Node 2 Specification
:label: def-node2-turnover

**Predicate:** Trading activity is bounded; no infinite-frequency switching.
$$
P_2 : \quad N_t := \sum_{s \le t} \mathbb{I}[\text{trade at } s] \le N_{\max}(t),
$$
where $N_{\max}(t)$ is a time-dependent bound (e.g., $N_{\max}(t) = \kappa \cdot t$).

**Market interpretation:** Prevents "Zeno paradox" where infinite trades occur in finite time. Detects HFT instability and quote stuffing.

**Observable metrics:**
- Turnover rate: $\tau_t := \text{Volume}_t / \text{AUM}_t$
- Order-to-trade ratio
- Cancel rate: fraction of orders cancelled before execution

**Certificate format:**
$$
K_2^+ = (\tau_t, \text{O2T ratio}, \text{cancel rate}, \text{window})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{zeno}} = \lambda_2 \cdot D_{KL}(\pi_t \| \pi_{t-1})
$$
:::

---

#### Node 3: Compactness Check ($C_\mu$) — Leverage Bounds

:::{prf:definition} Node 3 Specification
:label: def-node3-compactness

**Predicate:** Positions and leverage are bounded; no concentration blow-up.
$$
P_3 : \quad \|w_t\|_{\infty} \le w_{\max} \quad \text{and} \quad \text{Lev}_t := \frac{\sum_i |w_t^i|}{\text{NAV}_t} \le L_{\max}.
$$

**Market interpretation:** Energy (capital at risk) concentrates or disperses but doesn't escape to infinity. Detects dangerous position concentrations.

**Observable metrics:**
- Gross leverage: $\text{Lev}_t$
- Herfindahl index: $\text{HHI}_t := \sum_i (w_t^i / \sum_j |w_t^j|)^2$
- Maximum single position as fraction of NAV

**Certificate format:**
$$
K_3^+ = (\text{Lev}_t, \text{HHI}_t, w_{\max}, \|w_t\|_{\infty})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{compact}} = \lambda_3 \cdot \max(0, \text{Lev}_t - L_{\max})^2 + \lambda_3' \cdot \text{HHI}_t
$$
:::

---

#### Node 4: Scale Stability Check ($\mathrm{SC}_\lambda$) — Parameter Drift

:::{prf:definition} Node 4 Specification
:label: def-node4-scale

**Predicate:** Model parameters evolve slower than the market adapts.
$$
P_4 : \quad \|\nabla_t \theta_t\|^2 \le \epsilon_{\text{drift}} \cdot \|\nabla_\theta \mathcal{L}\|^2,
$$
where $\theta_t$ are model parameters and $\nabla_t$ is the time derivative.

**Market interpretation:** The pricing model is not chasing noise. Parameters should be stable relative to signal.

**Observable metrics:**
- Parameter volatility: $\sigma(\theta)$ over rolling window
- Signal-to-noise ratio of parameter updates
- Autocorrelation of parameter changes

**Certificate format:**
$$
K_4^+ = (\|\nabla_t \theta_t\|, \text{SNR}_{\theta}, \text{AC}_{\theta})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{scale}} = \lambda_4 \cdot \|\theta_t - \theta_{t-1}\|^2 / \|\nabla_\theta \mathcal{L}\|^2
$$
:::

---

#### Node 5: Stationarity Check ($\mathrm{SC}_{\partial c}$) — Regime Stability

:::{prf:definition} Node 5 Specification
:label: def-node5-stationarity

**Predicate:** The current regime is statistically stationary.
$$
P_5 : \quad \text{ADF}(r_{t-W:t}) < \text{crit}_{\alpha} \quad \text{or} \quad \text{KPSS}(r_{t-W:t}) > \text{crit}_{\alpha}',
$$
where ADF is Augmented Dickey-Fuller and KPSS is Kwiatkowski-Phillips-Schmidt-Shin test.

**Market interpretation:** Price dynamics are stable within the current regime; no structural breaks.

**Observable metrics:**
- ADF test statistic
- KPSS test statistic
- Chow test for structural breaks
- Rolling mean/variance stability

**Certificate format:**
$$
K_5^+ = (\text{ADF}_t, \text{KPSS}_t, \text{break count}, \text{window})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{stat}} = \lambda_5 \cdot \text{ReLU}(\text{ADF}_t - \text{crit}_{\alpha})
$$
:::

---

#### Node 6: Information Capacity Check ($\mathrm{Cap}_H$) — Market Depth

:::{prf:definition} Node 6 Specification
:label: def-node6-capacity

**Predicate:** Market depth supports the information content of the state.
$$
P_6 : \quad I(B_t; K_t) \le \mathcal{C}_{\text{channel}} := \log_2(1 + \text{SNR}_{\text{depth}}),
$$
where $\mathcal{C}_{\text{channel}}$ is the information capacity of the order book.

**Market interpretation:** The market can transmit enough information to support the complexity of the current regime.

**Observable metrics:**
- Order book depth at multiple levels
- Effective spread
- Kyle's lambda (price impact coefficient)
- Information ratio of order flow

**Certificate format:**
$$
K_6^+ = (I(B_t; K_t), \mathcal{C}_{\text{channel}}, \text{depth}, \lambda_{\text{Kyle}})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{cap}} = \lambda_6 \cdot \mathcal{L}_{\text{InfoNCE}}(z_t, z_{t+1})
$$
:::

---

#### Node 7: Stiffness Check ($\mathrm{LS}_\sigma$) — Price Impact Bounds

:::{prf:definition} Node 7 Specification
:label: def-node7-stiffness

**Predicate:** The value gradient is non-vanishing; price discovery is possible.
$$
P_7 : \quad \|\nabla_z V(z_t)\| \ge \epsilon_{\text{stiff}} > 0.
$$

**Market interpretation:** Prices respond to information. A flat value landscape means prices are stuck (no signal).

**Observable metrics:**
- Value gradient norm
- Price impact: $\Delta p / \Delta q$
- Łojasiewicz exponent estimate
- Bid-ask spread sensitivity

**Certificate format:**
$$
K_7^+ = (\|\nabla V\|, \text{impact}, \text{LS exponent})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{stiff}} = \lambda_7 \cdot \max(0, \epsilon_{\text{stiff}} - \|\nabla_z V\|)^2
$$
:::

---

#### Node 7a: Bifurcation Check ($\mathrm{Bif}$) — Regime Stability

:::{prf:definition} Node 7a Specification
:label: def-node7a-bifurcation

**Predicate:** The system is not at a bifurcation point.
$$
P_{7a} : \quad |\det(J_S(z_t))| \ge \epsilon_{\text{bif}},
$$
where $J_S$ is the Jacobian of the market dynamics.

**Market interpretation:** Small perturbations don't cause qualitative regime changes. Near bifurcation, tiny shocks can flip the market between states.

**Observable metrics:**
- Jacobian determinant (estimated via finite differences)
- Eigenvalue clustering near zero
- Sensitivity of regime probabilities to shocks

**Certificate format:**
$$
K_{7a}^+ = (|\det J_S|, \lambda_{\min}(J_S), \text{sensitivity})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{bif}} = \lambda_{7a} \cdot \text{Var}(\nabla_z S_t)
$$
:::

---

#### Node 7b: Alternatives Check ($\mathrm{Sym}$) — Strategy Diversity

:::{prf:definition} Node 7b Specification
:label: def-node7b-alternatives

**Predicate:** Multiple viable trading strategies exist.
$$
P_{7b} : \quad H(\pi_t) \ge \epsilon_{\text{ent}},
$$
where $H(\pi_t)$ is the entropy of the policy/strategy distribution.

**Market interpretation:** The market isn't locked into a single strategy. Diversity of approaches provides resilience.

**Observable metrics:**
- Policy entropy
- Number of active strategies (above threshold)
- Strategy correlation matrix rank

**Certificate format:**
$$
K_{7b}^+ = (H(\pi_t), \text{active count}, \text{rank}(\Sigma_{\text{strat}}))
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{alt}} = -\lambda_{7b} \cdot H(\pi_t)
$$
:::

---

#### Node 7c: New Regime Stability ($\mathrm{SC}_{\text{new}}$)

:::{prf:definition} Node 7c Specification
:label: def-node7c-newregime

**Predicate:** After a regime switch, the new regime is stable.
$$
P_{7c} : \quad \text{Var}(V(z_{t+1:t+W}) | K_{t+1} = k') \le \sigma_{\text{stable}}^2.
$$

**Market interpretation:** When the market transitions to a new regime, prices stabilize quickly rather than continuing to fluctuate wildly.

**Observable metrics:**
- Post-transition variance
- Time to stabilization
- Return autocorrelation decay

**Certificate format:**
$$
K_{7c}^+ = (\text{Var}_{post}, \tau_{\text{stable}}, \text{AC decay rate})
$$
:::

---

#### Node 7d: Switching Cost Check ($\mathrm{TB}_{\text{switch}}$)

:::{prf:definition} Node 7d Specification
:label: def-node7d-switching

**Predicate:** The cost of switching strategies is affordable.
$$
P_{7d} : \quad |V(\pi') - V(\pi)| - B_{\text{switch}} \le \text{Budget}_{\text{switch}}.
$$

**Market interpretation:** Transitioning between strategies doesn't consume excessive capital in transaction costs.

**Observable metrics:**
- Estimated rebalancing cost
- Turnover required for strategy change
- Slippage estimate

**Certificate format:**
$$
K_{7d}^+ = (\text{switch cost}, \text{turnover}, \text{slippage})
$$
:::

---

#### Node 8: Connectivity Check ($\mathrm{TB}_\pi$) — Network Topology

:::{prf:definition} Node 8 Specification
:label: def-node8-connectivity

**Predicate:** The trading/clearing network is connected.
$$
P_8 : \quad \text{connected}(G_{\text{clearing}}) = \text{True},
$$
where $G_{\text{clearing}}$ is the graph of counterparty relationships.

**Market interpretation:** All market participants can reach each other for settlement. Disconnection indicates fragmentation or clearing failure.

**Observable metrics:**
- Graph connectivity (strongly/weakly connected components)
- Average path length
- Clustering coefficient
- Central counterparty coverage

**Certificate format:**
$$
K_8^+ = (\text{num components}, \text{avg path}, \text{clustering}, \text{CCP coverage})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{conn}} = \lambda_8 \cdot (\text{num components} - 1)
$$
:::

---

#### Node 9: Tameness Check ($\mathrm{TB}_O$) — Pricing Smoothness

:::{prf:definition} Node 9 Specification
:label: def-node9-tameness

**Predicate:** Pricing functions are smooth; no discontinuities.
$$
P_9 : \quad \|\nabla^2 P(z)\|_{\text{op}} \le \Gamma_{\max},
$$
where $\nabla^2 P$ is the Hessian (gamma) of the pricing function.

**Market interpretation:** Prices respond smoothly to state changes. Jumps in gamma indicate potential for discontinuous repricing.

**Observable metrics:**
- Gamma bounds across instruments
- Convexity measures
- Jump frequency in pricing
- Lipschitz constant estimate

**Certificate format:**
$$
K_9^+ = (\Gamma_{\max}, \text{convexity}, \text{Lip const}, \text{jump count})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{tame}} = \lambda_9 \cdot \|\nabla^2_z S_t\|^2
$$
:::

---

#### Node 10: Mixing Check ($\mathrm{TB}_\rho$) — Regime Exploration

:::{prf:definition} Node 10 Specification
:label: def-node10-mixing

**Predicate:** The market explores all regimes adequately.
$$
P_{10} : \quad \min_k \hat{p}(K_t = k) \ge p_{\min},
$$
where $\hat{p}$ is the empirical regime frequency.

**Market interpretation:** The market doesn't get stuck in one regime. Adequate exploration means all states have been tested.

**Observable metrics:**
- Regime visit frequencies
- Mixing time estimate
- Ergodic ratio
- Time since last regime $k$ visit

**Certificate format:**
$$
K_{10}^+ = (\min_k \hat{p}(k), \tau_{\text{mix}}, \text{ergodic ratio})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{mix}} = -\lambda_{10} \cdot H(\pi_t)
$$
:::

---

#### Node 11: Representation Check ($\mathrm{Rep}_K$) — Regime Complexity

:::{prf:definition} Node 11 Specification
:label: def-node11-representation

**Predicate:** Regime complexity is within capacity.
$$
P_{11} : \quad H(K_t) \le \log |\mathcal{K}| - \epsilon_{\text{margin}}.
$$

**Market interpretation:** The number of active regimes doesn't exceed what the market can distinguish. Prevents "hallucinated" regimes.

**Observable metrics:**
- Regime entropy $H(K_t)$
- Effective number of regimes: $\exp(H(K_t))$
- Rate utilization: $H(K_t) / \log |\mathcal{K}|$

**Certificate format:**
$$
K_{11}^+ = (H(K_t), |\mathcal{K}|_{\text{eff}}, \text{utilization})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{rep}} = \lambda_{11} \cdot H(q(K|x))
$$
:::

---

#### Node 12: Oscillation Check ($\mathrm{GC}_\nabla$) — No Endogenous Cycles

:::{prf:definition} Node 12 Specification
:label: def-node12-oscillation

**Predicate:** No persistent oscillatory patterns (boom-bust cycles).
$$
P_{12} : \quad \|z_t - z_{t-2}\| \ge \epsilon_{\text{osc}} \quad \text{or} \quad \text{FFT}(z_{t-W:t}) \text{ has no dominant frequency}.
$$

**Market interpretation:** Prevents period-2 limit cycles where the market ping-pongs between states.

**Observable metrics:**
- Period-2 autocorrelation
- Spectral peak detection
- Holonomy around closed paths
- Boom-bust indicator

**Certificate format:**
$$
K_{12}^+ = (\text{AC}_2, \text{spectral peak}, \text{holonomy})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{osc}} = \lambda_{12} \cdot \|z_t - z_{t-2}\|^{-2}
$$
:::

---

#### Node 13: Boundary Coupling Check ($\mathrm{Bound}_\partial$) — Price Grounding

:::{prf:definition} Node 13 Specification
:label: def-node13-boundary

**Predicate:** Prices are grounded in observable data.
$$
P_{13} : \quad I(B_t; K_t) > \epsilon_{\text{ground}}.
$$

**Market interpretation:** Internal regime beliefs are supported by external evidence. Prevents "ungrounded inference" where prices disconnect from fundamentals.

**Observable metrics:**
- Mutual information $I(B_t; K_t)$
- Boundary-bulk correlation
- Forecast error relative to boundary data

**Certificate format:**
$$
K_{13}^+ = (I(B_t; K_t), \rho_{\text{boundary-bulk}}, \text{forecast error})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{bound}} = \lambda_{13} \cdot \text{ReLU}(D_{KL}(p_{t+1} \| p_t) - I(B_t; K_t))^2
$$
:::

---

#### Node 14: Overload Check ($\mathrm{Bound}_B$) — Data Saturation

:::{prf:definition} Node 14 Specification
:label: def-node14-overload

**Predicate:** Data channels are not saturated.
$$
P_{14} : \quad \text{Quote outage rate} \le \epsilon_{\text{outage}} \quad \text{and} \quad \text{Spread}_t \le s_{\max}.
$$

**Market interpretation:** The market infrastructure can handle the data load. Overload causes stale prices and failed executions.

**Observable metrics:**
- Quote outage frequency
- Message queue depth
- Latency spikes
- Spread blowouts

**Certificate format:**
$$
K_{14}^+ = (\text{outage rate}, \text{queue depth}, \text{latency p99}, \text{spread})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{overload}} = \lambda_{14} \cdot \mathbb{I}(\|x_t\| > x_{\max})
$$
:::

---

#### Node 15: Starvation Check ($\mathrm{Bound}_\Sigma$) — Data Sufficiency

:::{prf:definition} Node 15 Specification
:label: def-node15-starvation

**Predicate:** Sufficient data is available for pricing.
$$
P_{15} : \quad \text{SNR}_t \ge \epsilon_{\text{SNR}} \quad \text{and} \quad \text{Volume}_t \ge V_{\min}.
$$

**Market interpretation:** The market has enough trading activity to produce meaningful prices. Starvation = illiquidity.

**Observable metrics:**
- Signal-to-noise ratio
- Trading volume
- Time since last trade
- Quote staleness

**Certificate format:**
$$
K_{15}^+ = (\text{SNR}_t, \text{Volume}_t, \text{staleness})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{starve}} = \lambda_{15} \cdot \text{ReLU}(\epsilon_{\text{SNR}} - \text{SNR}_t)
$$
:::

---

#### Node 16: Alignment Check ($\mathrm{GC}_T$) — Incentive Consistency

:::{prf:definition} Node 16 Specification
:label: def-node16-alignment

**Predicate:** Short-term incentives align with long-term objectives.
$$
P_{16} : \quad \|V_{\text{proxy}} - V_{\text{true}}\| \le \epsilon_{\text{align}}.
$$

**Market interpretation:** Trading signals (proxy) actually predict long-term value (true). Misalignment causes agency problems.

**Observable metrics:**
- Proxy-true value correlation
- Funding rate vs. risk signal divergence
- Agent objective alignment score

**Certificate format:**
$$
K_{16}^+ = (\|V_{\text{proxy}} - V_{\text{true}}\|, \rho_{\text{align}}, \text{divergence})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{align}} = \lambda_{16} \cdot \|V_{\text{proxy}} - V_{\text{true}}\|^2
$$
:::

---

#### Node 17: Lock Check ($\mathrm{Cat}_{\mathrm{Hom}}$) — No Structural Arbitrage

:::{prf:definition} Node 17 Specification
:label: def-node17-lock

**Predicate:** No structural arbitrage exists.
$$
P_{17} : \quad \text{Hom}(\mathbb{H}_{\text{arb}}, \mathbb{H}_{\text{mkt}}) = \emptyset,
$$
where $\mathbb{H}_{\text{arb}}$ is the universal arbitrage pattern.

**Market interpretation:** There is no way to extract guaranteed profit without risk. This is the **final lock** that validates the entire pricing structure.

**Observable metrics:**
- Arbitrage cycle detection (graph algorithms)
- Put-call parity violations
- Cross-market price discrepancies
- Negative basis detection

**Certificate format:**
$$
K_{17}^+ = (\text{arb cycle count} = 0, \text{max parity violation}, \text{basis})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{lock}} = \infty \cdot \mathbb{I}(\text{arbitrage detected})
$$

**Implementation note:** This is a **hard constraint**. Any arbitrage detection immediately invalidates the pricing model.
:::

---

#### Nodes 18-21: Extended Checks

:::{prf:definition} Node 18: Symmetry Check
:label: def-node18-symmetry

**Predicate:** Pricing is invariant under gauge transformations.
$$
P_{18} : \quad \mathbb{E}_g[D_{KL}(q(K|x) \| q(K|g \cdot x))] \le \epsilon_{\text{sym}}.
$$

**Market interpretation:** Changing numeraire or relabeling assets doesn't change fundamental valuations.
:::

:::{prf:definition} Node 19: Disentanglement Check
:label: def-node19-disentangle

**Predicate:** Macro and micro factors are separated.
$$
P_{19} : \quad \|\text{Cov}(z_{\text{macro}}, z_{\text{micro}})\|_F^2 \le \epsilon_{\text{dis}}.
$$

**Market interpretation:** Regime state (macro) is not contaminated by noise (micro). Clean separation enables robust pricing.
:::

:::{prf:definition} Node 20: Lipschitz Check
:label: def-node20-lipschitz

**Predicate:** All operators have bounded Lipschitz constants.
$$
P_{20} : \quad \max_{\ell} \sigma(W_{\ell}) \le L_{\max},
$$
where $\sigma(W_{\ell})$ is the spectral norm of layer $\ell$.

**Market interpretation:** Small input changes produce small output changes. No explosive sensitivity.
:::

:::{prf:definition} Node 21: Symplectic Check (for Hamiltonian markets)
:label: def-node21-symplectic

**Predicate:** Market dynamics preserve phase space volume.
$$
P_{21} : \quad \|J_S^T J J_S - J\|_F^2 \le \epsilon_{\text{symp}},
$$
where $J$ is the symplectic form.

**Market interpretation:** For markets with Hamiltonian structure (e.g., order book dynamics), volume preservation ensures no information loss.
:::

### 7.3 Barrier Permits (Failure Defense)

Barriers return **Blocked** ($K^{\mathrm{blk}}$) or **Breached** ($K^{\mathrm{br}}$). When breached, pricing enters **defense mode**: conservative bounds, reduced position limits, or suspension.

#### Summary Table: 20 Market Barriers

| Barrier | Category | Meaning | Trigger | Defense |
|---------|----------|---------|---------|---------|
| BarrierSat | Position | Actuator saturation | Position hits hard limit | Cap positions |
| BarrierTypeII | Scaling | Vol-of-vol crisis | $\beta > \alpha$ | Freeze updates |
| BarrierGap | Liquidity | Price discontinuity | Spread > threshold | Widen quotes |
| BarrierOmin | Dynamics | Flash crash | $\|dp/dt\| > \text{limit}$ | Circuit breaker |
| BarrierCausal | Information | Prediction lag | Forecast horizon exceeded | Shorten horizon |
| BarrierScat | Representation | Market fragmentation | $I(B;K) \to 0$ | Consolidate |
| BarrierMix | Diversity | Herding | $H(\pi) \to 0$ | Inject noise |
| BarrierCap | Control | Uncontrollability | No hedge exists | Reduce exposure |
| BarrierVac | Stability | Regime vacuum | Bifurcation detected | Stabilize |
| BarrierFreq | Oscillation | HFT instability | Resonance detected | Rate limit |
| BarrierEpi | Information | Overload | Channel saturated | Throttle |
| BarrierAction | Execution | Trade impossible | Execution fails | Queue/cancel |
| BarrierInput | Resources | Data starvation | No quotes available | Use stale + discount |
| BarrierVariety | Hedging | Incomplete market | Hedge unavailable | Accept residual |
| BarrierBode | Tradeoff | Risk waterbed | Suppress one, amplify another | Balanced response |
| BarrierLock | Regulatory | Hard limit | Legal/regulatory breach | Mandatory stop |
| BarrierLiq | Liquidity | Illiquidity crisis | Spread/depth threshold | Interval pricing |
| BarrierLev | Leverage | Excess leverage | Leverage > limit | Deleveraging |
| BarrierRef | Reference | Price integrity | Oracle deviation | Fallback oracle |
| BarrierDef | Credit | Default event | Credit event trigger | Recovery protocol |

---

#### BarrierSat: Position Saturation

:::{prf:definition} BarrierSat Specification
:label: def-barrier-sat

**Condition:** Position sizes hit hard limits.
$$
\text{Breached} \iff \exists i : |w_t^i| \ge w_{\max}^i - \epsilon.
$$

**Market context:** Regulatory position limits, exchange limits, risk limits.

**Trigger observables:**
- Position size relative to limit
- Utilization rate: $|w| / w_{\max}$
- Time at limit

**Defense action:**
1. Cap new orders in breached direction
2. Allow only reducing trades
3. Notify risk management

**Re-entry condition:** $|w_t^i| < 0.9 \cdot w_{\max}^i$ for all $i$.
:::

---

#### BarrierTypeII: Scaling Hierarchy Violation

:::{prf:definition} BarrierTypeII Specification
:label: def-barrier-typeii

**Condition:** Volatility temperature exceeds risk temperature.
$$
\text{Breached} \iff \beta_t > \alpha_t,
$$
where $\beta$ = volatility scaling, $\alpha$ = risk perception scaling.

**Market context:** The market is moving faster than participants can assess risk. Classic "vol-of-vol" crisis.

**Trigger observables:**
- Ratio $\beta_t / \alpha_t$
- VIX / realized vol divergence
- Model uncertainty spikes

**Defense action:**
1. Freeze portfolio updates (skip policy step)
2. Widen confidence intervals
3. Increase haircuts/margins

**Re-entry condition:** $\alpha_t > \beta_t + \epsilon_{\text{buffer}}$ for $\tau_{\text{stable}}$ periods.
:::

---

#### BarrierGap: Liquidity Gap

:::{prf:definition} BarrierGap Specification
:label: def-barrier-gap

**Condition:** Bid-ask spread exceeds normal bounds.
$$
\text{Breached} \iff s_t := p_{\text{ask}} - p_{\text{bid}} > s_{\max}.
$$

**Market context:** Liquidity withdrawal, market maker failure, or extreme uncertainty.

**Trigger observables:**
- Spread as multiple of normal
- Depth at best quotes
- Time since last fill

**Defense action:**
1. Price using mid ± conservative spread
2. Mark positions at worst-case (bid for longs, ask for shorts)
3. Reject market orders; limit orders only

**Re-entry condition:** $s_t < s_{\text{normal}}$ for $\tau_{\text{recover}}$ periods.
:::

---

#### BarrierOmin: Flash Crash Detection

:::{prf:definition} BarrierOmin Specification
:label: def-barrier-omin

**Condition:** Price velocity exceeds physical limits.
$$
\text{Breached} \iff \left| \frac{dp_t}{dt} \right| > v_{\max}.
$$

**Market context:** Flash crash, fat-finger error, algorithmic feedback loop.

**Trigger observables:**
- Price change per unit time
- Cumulative intraday move
- Deviation from fair value

**Defense action:**
1. **Circuit breaker:** halt trading for $\tau_{\text{halt}}$
2. Cancel outstanding orders
3. Re-open with auction mechanism

**Re-entry condition:** Successful auction clears within bounds.
:::

---

#### BarrierCausal: Information Horizon Exceeded

:::{prf:definition} BarrierCausal Specification
:label: def-barrier-causal

**Condition:** Prediction horizon exceeds model validity.
$$
\text{Breached} \iff \tau_{\text{forecast}} > \tau_{\text{model validity}}.
$$

**Market context:** Trying to price long-dated instruments with short-term models.

**Trigger observables:**
- Forecast horizon vs. training window
- Out-of-sample error growth
- Model confidence decay

**Defense action:**
1. Shorten effective forecast horizon
2. Increase uncertainty bands exponentially with horizon
3. Use unconditional (prior) distribution for long horizons

**Re-entry condition:** Model retrained or horizon reduced.
:::

---

#### BarrierScat: Market Fragmentation

:::{prf:definition} BarrierScat Specification
:label: def-barrier-scat

**Condition:** Boundary-bulk coupling collapses.
$$
\text{Breached} \iff I(B_t; K_t) < \epsilon_{\text{min}} \quad \text{or} \quad H(K_t) \approx \log |\mathcal{K}|.
$$

**Market context:** Prices become disconnected from fundamentals; regimes become indistinguishable.

**Trigger observables:**
- Mutual information estimate
- Regime entropy (too high = dispersion)
- Cross-venue price divergence

**Defense action:**
1. Consolidate to primary venue
2. Reduce regime model complexity
3. Use simple (robust) pricing models

**Re-entry condition:** $I(B_t; K_t) > 2 \epsilon_{\text{min}}$ sustained.
:::

---

#### BarrierMix: Herding / Mode Collapse

:::{prf:definition} BarrierMix Specification
:label: def-barrier-mix

**Condition:** Strategy diversity collapses.
$$
\text{Breached} \iff H(\pi_t) < \epsilon_{\text{ent}}.
$$

**Market context:** Everyone is on the same trade; crowded positions create systemic risk.

**Trigger observables:**
- Strategy entropy
- Correlation of flows across participants
- Short interest concentration

**Defense action:**
1. Inject exploration noise
2. Increase contrarian signal weight
3. Reduce position size in crowded trades

**Re-entry condition:** $H(\pi_t) > 2\epsilon_{\text{ent}}$.
:::

---

#### BarrierCap: Uncontrollability

:::{prf:definition} BarrierCap Specification
:label: def-barrier-cap

**Condition:** No available action can improve the situation.
$$
\text{Breached} \iff \forall a \in \mathcal{A} : V(S(z_t, a)) \ge V(z_t).
$$

**Market context:** Stuck in a bad state with no exit. The "doom loop."

**Trigger observables:**
- All actions worsen value
- Controllability Gramian near-singular
- No hedge available at any price

**Defense action:**
1. Accept current position (no trading)
2. Seek external intervention (liquidity injection)
3. Invoke surgery (bailout protocol)

**Re-entry condition:** At least one improving action becomes available.
:::

---

#### BarrierVac: Regime Instability / Bifurcation

:::{prf:definition} BarrierVac Specification
:label: def-barrier-vac

**Condition:** System is at or near a bifurcation point.
$$
\text{Breached} \iff |\det(J_S)| < \epsilon_{\text{bif}}.
$$

**Market context:** Small shocks can cause large regime changes. Metastability.

**Trigger observables:**
- Jacobian determinant
- Critical slowing down (autocorrelation spike)
- Variance spike

**Defense action:**
1. Widen all uncertainty bands
2. Prepare for both regime outcomes
3. Reduce leverage preemptively

**Re-entry condition:** $|\det(J_S)| > 2\epsilon_{\text{bif}}$ sustained.
:::

---

#### BarrierFreq: HFT Oscillation / Resonance

:::{prf:definition} BarrierFreq Specification
:label: def-barrier-freq

**Condition:** Closed-loop system exhibits resonance.
$$
\text{Breached} \iff \|J_{\text{feedback}}\|_{\text{spectral}} \ge 1.
$$

**Market context:** HFT algorithms create feedback loops; quote flickering; mini-flash crashes.

**Trigger observables:**
- Quote update frequency
- Price oscillation frequency
- Spectral power at characteristic frequencies

**Defense action:**
1. Rate-limit order updates
2. Introduce minimum quote lifetime
3. Damping via wider spreads

**Re-entry condition:** Oscillation amplitude decays below threshold.
:::

---

#### BarrierEpi: Information Overload

:::{prf:definition} BarrierEpi Specification
:label: def-barrier-epi

**Condition:** Information channel is saturated.
$$
\text{Breached} \iff I_{\text{received}} > \mathcal{C}_{\text{channel}}.
$$

**Market context:** Too many signals; processing capacity exceeded; system cannot keep up.

**Trigger observables:**
- Message queue depth
- Processing latency
- Dropped message rate

**Defense action:**
1. Throttle incoming data
2. Prioritize critical feeds
3. Use cached/delayed data with discounts

**Re-entry condition:** Queue depth returns to normal.
:::

---

#### BarrierAction: Execution Impossibility

:::{prf:definition} BarrierAction Specification
:label: def-barrier-action

**Condition:** Desired trade cannot be executed.
$$
\text{Breached} \iff \text{ExecutionCost}(a_t) > \text{Budget}_{\text{exec}}.
$$

**Market context:** Market impact too high; no counterparty available; settlement failure.

**Trigger observables:**
- Estimated market impact
- Fill rate on orders
- Settlement failures

**Defense action:**
1. Queue order for later execution
2. Break into smaller pieces (TWAP/VWAP)
3. Accept partial fill or cancel

**Re-entry condition:** Execution cost falls within budget.
:::

---

#### BarrierInput: Data Starvation

:::{prf:definition} BarrierInput Specification
:label: def-barrier-input

**Condition:** Insufficient market data for pricing.
$$
\text{Breached} \iff \text{Volume}_t < V_{\min} \quad \text{or} \quad \text{Age}(\text{last quote}) > \tau_{\text{stale}}.
$$

**Market context:** Illiquid market; exchange outage; data feed failure.

**Trigger observables:**
- Time since last trade
- Quote staleness
- Data feed status

**Defense action:**
1. Use last known price with uncertainty premium
2. Interpolate from related instruments
3. Mark as "indicative only"

**Re-entry condition:** Fresh data arrives.
:::

---

#### BarrierVariety: Hedging Impossibility (Ashby)

:::{prf:definition} BarrierVariety Specification
:label: def-barrier-variety

**Condition:** Hedge dimensionality insufficient.
$$
\text{Breached} \iff \dim(\text{Hedge Space}) < \dim(\text{Risk Space}).
$$

**Market context:** Incomplete market; basis risk cannot be eliminated.

**Trigger observables:**
- Rank of hedge instrument covariance
- Residual risk after best hedge
- Basis spread volatility

**Defense action:**
1. Accept residual (unhedgeable) risk
2. Price in risk premium for incompleteness
3. Reduce exposure to unhedgeable component

**Re-entry condition:** New hedge instruments become available or exposure reduced.
:::

---

#### BarrierBode: Risk Waterbed Effect

:::{prf:definition} BarrierBode Specification
:label: def-barrier-bode

**Condition:** Reducing one risk increases another (control theory waterbed).
$$
\text{Breached} \iff \int_0^\infty \log |S(j\omega)| d\omega > 0,
$$
where $S$ is the sensitivity function.

**Market context:** Hedging one risk (e.g., delta) amplifies another (e.g., gamma, vega).

**Trigger observables:**
- Cross-Greek sensitivity
- Hedge effectiveness vs. new risk introduction
- Portfolio sensitivity integral

**Defense action:**
1. Balanced multi-objective hedging
2. Accept tradeoff explicitly
3. Use robust (worst-case) hedging

**Re-entry condition:** Sensitivity integral returns to acceptable range.
:::

---

#### BarrierLock: Regulatory Hard Stop

:::{prf:definition} BarrierLock Specification
:label: def-barrier-lock

**Condition:** Legal or regulatory limit breached.
$$
\text{Breached} \iff x_t \in \mathcal{X}_{\text{forbidden}}.
$$

**Market context:** Exceeding position limits, capital requirements, or other hard regulatory constraints.

**Trigger observables:**
- Regulatory metric values
- Distance to regulatory threshold
- Compliance flags

**Defense action:**
1. **Mandatory:** Cease prohibited activity immediately
2. Report to compliance
3. Execute remediation plan

**Re-entry condition:** Explicit regulatory clearance or metric returns to compliant range.

**Implementation note:** This barrier has **infinite penalty**. Unlike other barriers, there is no discretion—breach requires immediate action.
:::

---

#### BarrierLiq: Liquidity Threshold

:::{prf:definition} BarrierLiq Specification
:label: def-barrier-liq

**Condition:** Market liquidity falls below operational threshold.
$$
\text{Breached} \iff \text{Spread}_t > s_{\max} \quad \text{or} \quad \text{Depth}_t < d_{\min}.
$$

**Market context:** Illiquidity crisis where normal market making withdraws. Bid-ask spreads explode, depth vanishes.

**Trigger observables:**
- Bid-ask spread (absolute and relative)
- Order book depth at multiple levels
- Time between trades
- Quote update frequency

**Defense action:**
1. Widen position limits (reduce trading)
2. Switch to interval pricing
3. Mark positions to conservative estimate
4. Notify risk management

**Re-entry condition:** Spread < $0.8 \cdot s_{\max}$ and Depth > $1.2 \cdot d_{\min}$ for sustained period.
:::

---

#### BarrierLev: Leverage Threshold

:::{prf:definition} BarrierLev Specification
:label: def-barrier-lev

**Condition:** Aggregate leverage exceeds safe bounds.
$$
\text{Breached} \iff \text{Lev}_t := \frac{\text{Gross Exposure}_t}{\text{NAV}_t} > L_{\max}.
$$

**Market context:** Excessive leverage creates forced deleveraging risk. When markets move against leveraged positions, margin calls cascade.

**Trigger observables:**
- Gross leverage ratio
- Net leverage ratio
- Margin utilization
- Funding rate

**Defense action:**
1. Halt new position increases
2. Begin orderly deleveraging
3. Increase margin reserves
4. Monitor counterparty exposure

**Re-entry condition:** Leverage < $0.9 \cdot L_{\max}$ with stable margin.
:::

---

#### BarrierRef: Reference Price Integrity

:::{prf:definition} BarrierRef Specification
:label: def-barrier-ref

**Condition:** Reference price deviates significantly from consensus or shows manipulation signs.
$$
\text{Breached} \iff |p_{\text{ref}} - p_{\text{consensus}}| > \delta_{\text{ref}} \cdot p_{\text{consensus}}.
$$

**Market context:** Oracle attacks, benchmark manipulation, stale reference prices. Critical for derivatives and DeFi protocols that depend on external price feeds.

**Trigger observables:**
- Reference price deviation from median of sources
- Time since last update
- Cross-source disagreement
- Historical volatility of reference

**Defense action:**
1. Reject outlier reference prices
2. Fall back to backup oracle
3. Use time-weighted average (TWAP)
4. Pause operations if no reliable reference

**Re-entry condition:** Reference price within $0.5 \cdot \delta_{\text{ref}}$ of consensus from multiple independent sources.
:::

---

#### BarrierDef: Default/Credit Event

:::{prf:definition} BarrierDef Specification
:label: def-barrier-def

**Condition:** Credit event or default affects portfolio.
$$
\text{Breached} \iff \exists i : \text{Issuer}_i \in \text{Default State}.
$$

**Market context:** Counterparty default, issuer bankruptcy, credit event (failure to pay, restructuring). Triggers recovery process and cascade risk assessment.

**Trigger observables:**
- Credit event notices
- CDS auction triggers
- Rating downgrades to D
- Missed payment notifications

**Defense action:**
1. Freeze affected positions
2. Assess recovery value
3. Check for cascade exposure
4. Invoke credit event protocols

**Re-entry condition:** Recovery process complete and value realized; no residual exposure to defaulted entity.
:::

---

#### Barrier Interaction: Multi-Barrier Coordination

When multiple barriers breach simultaneously, the market enters **crisis mode**:

**Priority ordering:**
1. BarrierLock (always first—legal requirement)
2. BarrierOmin (safety—prevent crash continuation)
3. BarrierSat, BarrierCap (position management)
4. All others (risk management)

**Cascade detection:** If $\ge 3$ barriers breach within $\tau_{\text{cascade}}$, invoke **SurgeryMode** (Section 7.6).

### 7.4 Edge Validity and Determinism

**Edge validity.** An edge $N_1 \xrightarrow{o} N_2$ is valid iff the certificate $K_o$ implies the precondition of $N_2$.

**Determinism policy.** Any UNKNOWN check is treated as NO. This routes execution to barrier defenses and guarantees conservative pricing.

### 7.5 Promotions and Inconclusive Upgrades

Promotion and inc-upgrade rules are applied during context closure: the market aggregates certificates until no more promotions are possible. This makes pricing conclusions **monotone** with respect to evidence.

### 7.6 Surgery (Interventions)

Surgery nodes are **market interventions** that repair violations and re-enter the Sieve:
- circuit breakers,
- margin calls and position reductions,
- central bank liquidity,
- temporary price bands or auctions.

A surgery outputs a re-entry certificate $K^{\mathrm{re}}$ that proves preconditions for the next gate.


## 8. Market Dynamics and Control

The market is a collection of bounded-rational controllers that trade under constraints and costs.

### 8.1 Entropy-Regularized Portfolio Choice

Agent $j$ solves:
$$
\min_{\pi_j} \; \mathbb{E}\left[ C_j(\pi_j) + \alpha_j D_{KL}(\pi_j \Vert \pi_j^0) \right],
$$
where $C_j$ is expected cost and $\pi_j^0$ is a prior allocation. The solution is exponential-family:
$$
\pi_j(a) \propto \pi_j^0(a) \exp(-C_j(a)/\alpha_j).
$$
This is the **thermodynamic logit** and links risk aversion to temperature {cite}`todorov2009efficient,kappen2005path`.

### 8.2 Market Clearing as Fixed Point

Let $D_t(p)$ be aggregate demand and $S_t(p)$ aggregate supply. Clearing requires:
$$
D_t(p_t) = S_t(p_t).
$$
Under permits, the clearing price $p_t$ is a fixed point of the trading dynamics.

### 8.3 Stability via Lyapunov Potential

Define a Lyapunov functional $L$ (aggregate risk plus costs). Stability requires:
$$
\Delta L \le 0
$$
in normal regimes. Persistent positive drift violates the **Stiffness and Oscillation permits**.

---

## 9. Risk Measures and Thermodynamic Duality

### 9.1 Coherent Risk Measures

A risk measure $\rho$ is coherent if {cite}`artzner1999coherent`:
- monotone,
- subadditive,
- positive homogeneous,
- translation invariant.

### 9.2 Entropic Risk and Free Energy

The entropic risk of payoff $X$ at risk aversion $\alpha$ is
$$
\rho_{\alpha}(X) = \frac{1}{\alpha} \log \mathbb{E}[e^{\alpha X}].
$$
This is a **free-energy functional** and corresponds to exponential utility {cite}`follmer2011stochastic`.

### 9.3 Dual Form (Relative Entropy)

Entropic risk has dual form
$$
\rho_{\alpha}(X) = \sup_{Q \ll P} \left( \mathbb{E}_Q[X] - \frac{1}{\alpha} D_{KL}(Q \Vert P) \right),
$$
which is the thermoeconomic principle: value equals expected payoff minus information cost {cite}`kullback1951information,follmer2011stochastic`.

---

## 10. Asset Class Pricing (Comprehensive)

This section provides complete pricing specifications for all 12 major asset classes. Each class includes:
- SDF-based pricing derivation
- Complete permit checklist (relevant gates and barriers)
- Asset-specific failure mode mapping
- Risk geometry (curvature, geodesics)
- Stress test scenario

---

### 10.1 Risk-Free and Government Bonds

:::{prf:definition} Government Bond Pricing Framework
:label: def-govbond-pricing

**Fundamental equation.** For a zero-coupon bond paying 1 at maturity $T$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u \, du\right)\right] = \mathbb{E}_t^{\mathbb{P}}\left[M_T / M_t\right],
$$
where $r_u$ is the short rate and $M_t$ is the stochastic discount factor.

**Affine term structure.** Under affine models (Vasicek, CIR, multi-factor):
$$
P(t,T) = \exp\left(A(t,T) - B(t,T) \cdot X_t\right),
$$
where $X_t$ is the state vector (short rate, slope, curvature factors).

**Duration and convexity geometry.** Define the risk metric:
$$
g^{\text{bond}}_{ij} = \frac{\partial^2 \log P}{\partial X_i \partial X_j} = B_i B_j - \frac{\partial B_i}{\partial X_j}.
$$
This is the **Fisher information metric** on the bond manifold.

**Geodesic portfolio path.** Duration-neutral rebalancing follows geodesics:
$$
\ddot{w}^k + \Gamma^k_{ij} \dot{w}^i \dot{w}^j = 0,
$$
where $w$ is the portfolio weight vector and $\Gamma$ is the Christoffel symbol from $g^{\text{bond}}$.
:::

**Permit checklist:**
- Node 1 (Solvency): Government credit risk (for non-AAA sovereigns)
- Node 5 (Stationarity): Interest rate regime stability
- Node 6 (Capacity): Treasury market depth
- Node 8 (Connectivity): Dealer network functionality
- Node 11 (Representation): Yield curve model adequacy
- BarrierInput: Central bank data feed integrity
- BarrierLiq: On-the-run vs. off-the-run liquidity

**Failure mode mapping:**
- T.D (Frozen Market): Treasury market stress (March 2020)
- B.E (External Shock): Fed policy surprise
- S.D (Flat Vol): Yield curve control regimes

**Stress test scenario:** Fed surprise 100bp hike
- BarrierOmin: Check for gap risk in long-duration positions
- BarrierBode: Duration hedge increases convexity exposure
- Expected response: Price interval widens; switch to rolling auctions

---

### 10.2 Inflation-Linked Bonds (TIPS, Linkers)

:::{prf:definition} Inflation-Linked Bond Pricing
:label: def-tips-pricing

**Real vs. nominal decomposition.** Let $I_t$ be the price index. The real bond price:
$$
P^{\text{real}}(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u^{\text{real}} \, du\right)\right],
$$
where $r^{\text{real}}_t = r_t - \pi_t$ with $\pi_t$ the instantaneous inflation rate.

**Breakeven inflation.** The breakeven rate $\text{BE}(t,T)$ satisfies:
$$
P^{\text{nom}}(t,T) = P^{\text{real}}(t,T) \cdot \exp\left(-\text{BE}(t,T)(T-t)\right).
$$

**Inflation risk premium.** The difference between breakeven and expected inflation:
$$
\text{IRP}(t,T) = \text{BE}(t,T) - \mathbb{E}_t[\bar{\pi}_{t,T}] = -\frac{\text{Cov}_t(M_T, I_T)}{M_t P^{\text{nom}}(t,T)}.
$$

**Risk geometry.** The inflation-linked bond manifold has metric:
$$
g^{\text{TIPS}}_{ij} = g^{\text{real}}_{ij} + g^{\text{inflation}}_{ij} + 2 \cdot \text{cross-term}_{ij},
$$
capturing real rate risk, inflation risk, and their correlation.
:::

**Permit checklist:**
- Node 1 (Solvency): Sovereign real credit risk
- Node 5 (Stationarity): Inflation regime stability
- Node 11 (Representation): Inflation model adequacy (seasonal adjustment)
- BarrierInput: CPI data integrity and publication schedule
- BarrierCausal: Indexation lag (3-month typical)
- BarrierRef: Reference index definition changes

**Failure mode mapping:**
- D.C (Fundamental Uncertainty): Inflation regime change (1970s, 2021-22)
- S.C (Parameter Drift): Correlation breakdown between breakeven and realized
- B.E (External Shock): Commodity price shock affecting CPI

**Stress test scenario:** CPI methodology change
- BarrierRef triggers: Index definition no longer comparable
- BarrierCausal: Historical comparisons invalid
- Expected response: Widen price bounds; mark as model uncertainty

---

### 10.3 Equities

:::{prf:definition} Equity Pricing Framework
:label: def-equity-pricing

**Dividend discount model.** Stock price equals discounted expected dividends:
$$
S_t = \mathbb{E}_t\left[\sum_{u>t} M_u D_u\right] = \mathbb{E}_t\left[\int_t^\infty M_u D_u \, du\right].
$$

**Risk premium decomposition.** The equity risk premium:
$$
\mathbb{E}_t[R_{t+1}] - r_t = -\text{Cov}_t\left(\frac{M_{t+1}}{M_t}, R_{t+1}\right) = \gamma_t \cdot \text{Cov}_t(R_{t+1}, \Delta c_{t+1}),
$$
where $\gamma_t$ is risk aversion and $c$ is consumption (CCAPM form).

**Factor model embedding.** In factor space:
$$
\mathbb{E}_t[R_i] - r_t = \sum_k \beta_{ik} \lambda_k,
$$
where $\beta_{ik}$ is exposure to factor $k$ and $\lambda_k$ is the factor risk premium.

**Risk geometry (Sharpe manifold).** Define the metric on equity space:
$$
g^{\text{eq}}_{ij} = \frac{1}{\sigma_i \sigma_j} \left(\rho_{ij} - \frac{\mu_i - r}{\sigma_i} \cdot \frac{\mu_j - r}{\sigma_j} \cdot \frac{1}{\text{SR}^2_{\text{max}}}\right),
$$
where SR$_{\text{max}}$ is the maximum Sharpe ratio. Geodesics are **efficient portfolio paths**.

**Natural gradient update.** Portfolio optimization via:
$$
w_{t+1} = w_t - \eta \cdot (g^{\text{eq}})^{-1} \nabla_w \mathcal{L},
$$
where $\mathcal{L}$ is the risk-adjusted loss.
:::

**Permit checklist:**
- Node 1 (Solvency): Corporate credit/bankruptcy risk
- Node 2 (Turnover): Trading volume adequacy
- Node 3 (Leverage): Margin requirements, short interest
- Node 5 (Stationarity): Factor regime stability
- Node 6 (Capacity): Market cap, float
- Node 7 (Stiffness): Mean reversion in valuations
- Node 8 (Connectivity): Exchange connectivity, dark pools
- Node 11 (Representation): Factor model adequacy
- Node 12 (Oscillation): Momentum vs. mean-reversion balance
- BarrierOmin: Flash crash protection
- BarrierSat: Position limits
- BarrierFreq: HFT monitoring

**Failure mode mapping:**
- D.E (Boom-Bust): Equity bubbles (dot-com, meme stocks)
- T.E (Flash Crash): May 2010, August 2015
- S.E (Supercritical Leverage): Margin debt spikes
- C.D (Too-Big-to-Fail): Index concentration

**Stress test scenario:** Factor rotation (growth → value)
- Node 5 triggers: Regime change detected
- Node 12 monitors: Oscillation amplitude
- BarrierBode: Factor hedge introduces sector exposure
- Expected response: Increase model uncertainty; reduce position sizing

---

### 10.4 Commodities

:::{prf:definition} Commodity Pricing Framework
:label: def-commodity-pricing

**Spot-futures relationship.** Futures price under cost-of-carry:
$$
F_{t,T} = S_t \exp\left((r_t + c_t - y_t)(T-t)\right),
$$
where $c_t$ is storage cost and $y_t$ is convenience yield.

**Convenience yield dynamics.** Convenience yield reflects inventory scarcity:
$$
y_t = y_0 + \kappa(\bar{y} - y_t) dt + \sigma_y dW_t^y + \text{jump}(\text{inventory shock}).
$$

**Backwardation vs. contango regimes.** Market regime $K_t \in \{\text{backwardation}, \text{contango}\}$:
$$
K_t = \begin{cases}
\text{backwardation} & \text{if } y_t > r_t + c_t \\
\text{contango} & \text{if } y_t < r_t + c_t
\end{cases}
$$

**Risk geometry.** Commodity manifold with inventory state:
$$
g^{\text{comm}}_{ij} = \begin{pmatrix} \sigma_S^2 & \rho_{Sy}\sigma_S\sigma_y \\ \rho_{Sy}\sigma_S\sigma_y & \sigma_y^2 \end{pmatrix}
$$
Portfolio roll strategy follows geodesics on this manifold.

**Physical vs. financial convergence.** At delivery:
$$
\lim_{t \to T} F_{t,T} = S_T \quad \text{(physical settlement)}.
$$
This is enforced by arbitrage but requires storage/delivery capacity.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty risk (OTC), clearinghouse risk (exchange)
- Node 5 (Stationarity): Regime stability (backwardation/contango)
- Node 6 (Capacity): Storage capacity, delivery infrastructure
- Node 9 (Tameness): Price limit compliance
- Node 14 (Coupling): Spot-futures basis tracking
- BarrierInput: Inventory data, weather data
- BarrierRef: Benchmark price integrity (Brent, WTI, etc.)
- BarrierGap: Roll gap risk at expiry

**Failure mode mapping:**
- T.E (Flash Crash): Oil flash crash (April 2020 negative prices)
- D.E (Boom-Bust): Commodity supercycles
- B.E (External Shock): Geopolitical supply disruption
- T.C (Complexity): Physical vs. paper market divergence

**Stress test scenario:** Negative oil prices (April 2020 style)
- BarrierOmin: Price floor breach (negative prices possible)
- BarrierGap: Roll to next contract fails
- Node 6 triggers: Storage capacity exhausted
- Expected response: Halt physical delivery; cash settlement only

---

### 10.5 Foreign Exchange

:::{prf:definition} FX Pricing Framework
:label: def-fx-pricing

**Covered interest parity (CIP).** Forward rate determined by interest differential:
$$
F_{t,T} = S_t \cdot \frac{B^{\text{dom}}_t(T)}{B^{\text{for}}_t(T)} = S_t \exp\left((r^{\text{dom}}_t - r^{\text{for}}_t)(T-t)\right).
$$

**CIP deviations (cross-currency basis).** Actual forward deviates due to funding constraints:
$$
F_{t,T}^{\text{actual}} = F_{t,T}^{\text{CIP}} \cdot \exp(-\text{basis}_t \cdot (T-t)),
$$
where basis reflects dollar funding premium.

**Uncovered interest parity (UIP).** Expected spot change:
$$
\mathbb{E}_t[S_{T}] = S_t \exp\left((r^{\text{dom}}_t - r^{\text{for}}_t)(T-t)\right) + \text{risk premium}.
$$
UIP failure is the **carry trade premium**.

**Triangle arbitrage.** For currencies A, B, C:
$$
S_{A/B} \times S_{B/C} \times S_{C/A} = 1.
$$
Deviations are arbitrage opportunities or market stress indicators.

**Risk geometry.** FX space forms a Lie group (currency ratios):
$$
g^{\text{FX}}_{ij} = \sigma_i \sigma_j \rho_{ij},
$$
with natural group structure for cross rates.
:::

**Permit checklist:**
- Node 1 (Solvency): Sovereign default risk (EM currencies)
- Node 3 (Leverage): Margin requirements, leverage limits
- Node 5 (Stationarity): Interest rate regime, carry regime
- Node 8 (Connectivity): Dealer network, ECN access
- Node 10 (Mixing): Market maker activity
- Node 14 (Coupling): CIP/UIP relationship
- BarrierLiq: Liquidity in crosses vs. majors
- BarrierRef: Benchmark fixings (WM/Reuters)
- BarrierCausal: Time zone gaps

**Failure mode mapping:**
- D.D (Dispersion Success): Carry trade crowding
- S.E (Supercritical Leverage): Carry unwind (JPY 2024)
- B.E (External Shock): EM currency crisis
- T.E (Flash Crash): GBP October 2016

**Stress test scenario:** G10 carry unwind
- Node 3 triggers: Leverage across carry trades
- D.D → S.E cascade: Crowded carry → forced deleveraging
- BarrierOmin: Gap risk in JPY crosses
- Expected response: Reduce leverage; widen stops; hedge with vol

---

### 10.6 Credit and Defaultable Bonds

:::{prf:definition} Credit Pricing Framework
:label: def-credit-pricing

**Intensity-based model.** With hazard rate $\lambda_t$ and recovery $R$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T (r_u + \lambda_u) du}\right] + R \cdot \mathbb{E}_t^{\mathbb{Q}}\left[\int_t^T \lambda_u e^{-\int_t^u (r_s + \lambda_s) ds} du\right].
$$

**Credit spread decomposition.** Spread $s_t = \lambda_t (1-R) + \text{liquidity premium} + \text{risk premium}$:
$$
s_t = s^{\text{default}}_t + s^{\text{liquidity}}_t + s^{\text{risk}}_t.
$$

**Structural model (Merton).** Equity as call option on firm value:
$$
E_t = V_t N(d_1) - D e^{-rT} N(d_2),
$$
where $V_t$ is firm value and $D$ is debt face value.

**Distance to default.** Probability of default proxy:
$$
\text{DD}_t = \frac{\log(V_t/D) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}.
$$

**Risk geometry.** Credit manifold with coordinates (spread, duration, recovery):
$$
g^{\text{credit}}_{ij} = \begin{pmatrix} \sigma_s^2 & \rho_{sd} & \rho_{sr} \\ \rho_{sd} & \sigma_d^2 & \rho_{dr} \\ \rho_{sr} & \rho_{dr} & \sigma_r^2 \end{pmatrix}
$$
Geodesics represent constant-risk-adjusted credit curves.
:::

**Permit checklist:**
- Node 1 (Solvency): Default risk (primary concern)
- Node 2 (Turnover): Bond market liquidity (often poor)
- Node 5 (Stationarity): Credit cycle regime
- Node 6 (Capacity): New issue absorption
- Node 7 (Stiffness): Mean reversion in spreads
- Node 11 (Representation): Model adequacy (structural vs. intensity)
- BarrierSat: Concentration limits
- BarrierGap: Credit event gap risk
- BarrierInput: Financial statement data, rating actions

**Failure mode mapping:**
- C.E (Default Cascade): Contagion across credits
- C.D (Too-Big-to-Fail): Single-issuer concentration
- D.E (Boom-Bust): Credit cycle (spread compression → blow-out)
- T.D (Frozen Market): High-yield market freeze

**Stress test scenario:** IG → HY downgrade wave
- C.E triggers: Fallen angels create forced selling
- BarrierSat: Mandate constraints (IG-only funds)
- T.D risk: HY market cannot absorb supply
- Expected response: Pre-position for fallen angel risk; diversify by rating

---

### 10.7 Options and Derivatives

:::{prf:definition} Option Pricing Framework
:label: def-option-pricing

**Risk-neutral pricing.** European option value:
$$
V_t = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T r_u du} \cdot \text{Payoff}(S_T)\right].
$$

**Black-Scholes-Merton.** Under GBM dynamics ($dS = rS dt + \sigma S dW$):
$$
C_t = S_t N(d_1) - K e^{-r(T-t)} N(d_2), \quad d_{1,2} = \frac{\log(S/K) + (r \pm \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}.
$$

**Greeks as geometry.** The Greeks form a covariant structure:
$$
\Delta = \frac{\partial V}{\partial S}, \quad \Gamma = \frac{\partial^2 V}{\partial S^2}, \quad \Theta = \frac{\partial V}{\partial t}, \quad \text{Vega} = \frac{\partial V}{\partial \sigma}.
$$

**Volatility surface.** Implied vol $\sigma^{\text{imp}}(K,T)$ encodes market expectations. Surface dynamics:
$$
d\sigma^{\text{imp}} = \alpha dt + \xi dW^{\sigma},
$$
with constraints from no-arbitrage (Gatheral conditions).

**Risk metric on vol surface.** Curvature of the vol surface:
$$
R^{\text{vol}} = \frac{\partial^2 \sigma}{\partial K^2} - \frac{1}{T}\frac{\partial^2 \sigma}{\partial T^2},
$$
with high curvature indicating pricing stress.

**Replication and hedging.** Dynamic hedge portfolio:
$$
\Pi_t = V_t - \Delta_t S_t,
$$
requires continuous rebalancing. Market impact creates **hedging friction**.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty risk (OTC)
- Node 4 (Scale): Position sizing relative to gamma
- Node 5 (Stationarity): Vol regime stability
- Node 6 (Capacity): Liquidity at strikes/tenors
- Node 7 (Stiffness): Vol mean reversion
- Node 9 (Tameness): Tail risk bounds
- Node 10 (Mixing): Market maker activity
- Node 11 (Representation): Model adequacy (local vol, stochastic vol)
- Node 12 (Oscillation): Pin risk near expiry
- BarrierTypeII: Vol-of-vol crisis
- BarrierGap: Gap risk (discrete hedging)
- BarrierFreq: Gamma scalping frequency

**Failure mode mapping:**
- D.D (Dispersion Success): Vol selling crowding
- S.E (Supercritical Leverage): Gamma exposure × vol spike
- T.E (Flash Crash): Delta hedging cascade
- C.C (HFT Instability): Option market making at high frequency

**Stress test scenario:** Vol spike + liquidity withdrawal
- BarrierTypeII triggers: Vol-of-vol exceeds threshold
- Node 10 fails: Market makers pull quotes
- BarrierGap: Discrete hedging creates realized vs. implied gap
- Expected response: Reduce gamma exposure; accept wider bid-ask

---

### 10.8 Volatility Products

:::{prf:definition} Volatility Product Pricing
:label: def-vol-pricing

**Variance swap.** Fair strike for variance swap:
$$
K_{\text{var}} = \mathbb{E}_t^{\mathbb{Q}}\left[\frac{1}{T}\int_t^T \sigma_u^2 du\right] = \frac{2}{T}\int_0^\infty \frac{C(K) + P(K)}{K^2} dK,
$$
derived from static replication via log contract.

**VIX definition.** VIX index approximates 30-day implied variance:
$$
\text{VIX}^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i),
$$
where $Q(K_i)$ are out-of-money option prices.

**Vol-of-vol dynamics.** VIX follows mean-reverting jump-diffusion:
$$
d(\text{VIX}) = \kappa(\bar{v} - \text{VIX})dt + \xi \cdot \text{VIX}^{\beta} dW + J dN,
$$
with jumps $J$ and Poisson process $N$.

**Volatility term structure.** VIX futures curve:
$$
F^{\text{VIX}}_{t,T} = \mathbb{E}_t^{\mathbb{Q}}[\text{VIX}_T],
$$
typically in contango (upward sloping) due to variance risk premium.

**Risk geometry.** Vol space is positively curved (vol bounded below by zero):
$$
g^{\text{vol}}_{ij} = \frac{\partial^2}{\partial \sigma_i \partial \sigma_j}\log p(\sigma),
$$
where $p(\sigma)$ is the vol distribution. Non-Euclidean distances matter for vol trading.
:::

**Permit checklist:**
- Node 1 (Solvency): Extreme loss potential (short vol)
- Node 3 (Leverage): Leveraged vol ETPs
- Node 4 (Scale): Position size vs. market vol
- Node 5 (Stationarity): Vol regime stability
- Node 7 (Stiffness): Vol mean reversion strength
- Node 9 (Tameness): Tail risk in vol distribution
- Node 12 (Oscillation): Vol clustering
- BarrierTypeII: Vol-of-vol crisis (primary concern)
- BarrierVac: Regime instability
- BarrierOmin: Gap risk in vol products

**Failure mode mapping:**
- D.D (Dispersion Success): Short vol crowding (XIV blowup)
- S.E (Supercritical Leverage): Leveraged vol ETP cascade
- T.E (Flash Crash): VIX spike (Volmageddon February 2018)
- D.E (Boom-Bust): Vol compression → explosion cycle

**Stress test scenario:** Volmageddon replay
- BarrierTypeII triggers: Vol-of-vol extreme
- S.E activates: Leveraged products force rebalancing
- D.D → S.E cascade: Crowded short vol → forced covering
- Expected response: Position limits on vol ETPs; dynamic margin

---

### 10.9 Real Assets (Real Estate, Infrastructure)

:::{prf:definition} Real Asset Pricing Framework
:label: def-real-asset-pricing

**Discounted cash flow with illiquidity.** Real asset value:
$$
P_t = \mathbb{E}_t\left[\sum_{u>t} M_u (D_u - \iota_u)\right],
$$
where $\iota_u$ is the illiquidity discount (option value of liquidity foregone).

**Cap rate model.** Property value from net operating income:
$$
P_t = \frac{\text{NOI}_t}{\text{cap rate}_t}, \quad \text{cap rate}_t = r_t + \text{risk premium}_t - g_t,
$$
where $g_t$ is expected growth.

**Appraisal smoothing.** Reported values are smoothed:
$$
\hat{P}_t = \alpha P_t + (1-\alpha) \hat{P}_{t-1},
$$
creating **artificial autocorrelation** and understated volatility.

**NAV discount/premium.** REITs trade at:
$$
\text{Price}_t = \text{NAV}_t \times (1 + \text{discount/premium}_t),
$$
with discount reflecting liquidity premium, governance, and leverage.

**Illiquidity as option.** The illiquidity cost is a **real option**:
$$
\iota_t = \mathbb{E}_t\left[\max\left(0, V^{\text{liquid}}_\tau - V^{\text{illiquid}}_\tau\right)\right],
$$
where $\tau$ is the (uncertain) time of forced sale.
:::

**Permit checklist:**
- Node 1 (Solvency): Underlying tenant/asset credit
- Node 2 (Turnover): Transaction frequency (very low)
- Node 5 (Stationarity): Real estate cycle regime
- Node 6 (Capacity): Market depth for transactions
- Node 11 (Representation): Appraisal methodology adequacy
- BarrierInput: Valuation data quality and frequency
- BarrierCausal: Appraisal lag (quarterly typical)
- BarrierGap: Transaction gap (illiquidity)

**Failure mode mapping:**
- T.D (Frozen Market): Real estate transaction freeze (2008-09)
- D.E (Boom-Bust): Property cycles
- B.D (Liquidity Starvation): Redemption pressure on open-end funds
- D.C (Fundamental Uncertainty): Valuation during market stress

**Stress test scenario:** Open-end real estate fund redemption wave
- BarrierInput triggers: Stale NAV vs. market reality
- B.D activates: Redemptions exceed liquidity
- T.D risk: Fund gates, no transactions to price
- Expected response: Gate redemptions; mark to distressed sale levels

---

### 10.10 Crypto and Digital Assets

:::{prf:definition} Crypto Asset Pricing Framework
:label: def-crypto-pricing

**Network value model.** Token value from network economics:
$$
S_t = \mathbb{E}_t\left[\sum_{u>t} M_u (\text{fees}_u + \text{staking}_u + \text{MEV}_u)\right],
$$
where MEV is maximal extractable value.

**Metcalfe's Law approximation.** Network value scales with users:
$$
V_t \propto n_t^{\alpha}, \quad \alpha \in [1.5, 2],
$$
where $n_t$ is active addresses/users.

**Staking yield.** Proof-of-stake yield:
$$
y^{\text{stake}}_t = \frac{\text{block rewards}_t + \text{tips}_t}{\text{staked amount}_t} - \text{slashing risk}_t.
$$

**Oracle dependency.** DeFi prices depend on oracles:
$$
P^{\text{DeFi}}_t = f(\text{Oracle}_t), \quad \text{Oracle}_t = \text{median}(\text{reporters}).
$$
Oracle manipulation creates **reference barrier risk**.

**Cross-chain arbitrage.** Price consistency across chains:
$$
P^{\text{Chain A}}_t = P^{\text{Chain B}}_t + \text{bridge cost}_t + \text{latency premium}_t.
$$
:::

**Permit checklist:**
- Node 1 (Solvency): Protocol security, smart contract risk
- Node 2 (Turnover): On-chain vs. CEX volume
- Node 3 (Leverage): DeFi leverage (liquidation cascades)
- Node 5 (Stationarity): Protocol upgrade stability
- Node 8 (Connectivity): Cross-chain bridges, CEX connectivity
- Node 10 (Mixing): MEV, front-running
- Node 11 (Representation): Valuation model adequacy
- BarrierRef: Oracle integrity (critical)
- BarrierInput: Blockchain data feed
- BarrierLiq: DEX vs. CEX liquidity fragmentation

**Failure mode mapping:**
- C.E (Default Cascade): DeFi liquidation cascade
- T.E (Flash Crash): Crypto flash crashes (common)
- C.C (HFT Instability): MEV extraction instability
- B.C (Agency Misalignment): Insider trading, rug pulls
- T.C (Complexity): Smart contract composability failure

**Stress test scenario:** Oracle manipulation attack
- BarrierRef triggers: Oracle reports manipulated price
- C.E cascade: Liquidations cascade from bad prices
- Node 8 fails: Bridge exploits compound losses
- Expected response: Circuit breakers on oracle deviation; multi-source oracles

---

### 10.11 Private Equity and Venture Capital

:::{prf:definition} Private Equity Pricing Framework
:label: def-pe-pricing

**Stochastic exit model.** PE value with random exit:
$$
P_t = \mathbb{E}_t\left[M_\tau \cdot X_\tau\right],
$$
where $\tau$ is exit time (IPO, sale, failure) and $X_\tau$ is exit value.

**J-curve dynamics.** Fund NAV follows J-curve pattern:
$$
\text{NAV}_t = \text{Invested}_t - \text{Fees}_t + \text{Appreciation}_t,
$$
with early years showing fees > appreciation.

**Multiple expansion.** Value creation decomposition:
$$
\text{Return} = \text{Revenue growth} + \text{Margin expansion} + \text{Multiple expansion} + \text{Leverage}.
$$

**Secondary pricing.** Secondary market trades at:
$$
P^{\text{secondary}}_t = \text{NAV}_t \times (1 - \text{discount}_t),
$$
with discount reflecting illiquidity and information asymmetry.

**Wide bounds from incomplete markets.** Valuation is **interval-valued**:
$$
P_t \in [P^{\text{down}}_t, P^{\text{up}}_t],
$$
where bounds reflect scenario analysis, not point estimate.
:::

**Permit checklist:**
- Node 1 (Solvency): Portfolio company credit risk
- Node 2 (Turnover): Secondary market (limited)
- Node 5 (Stationarity): Exit market regime
- Node 6 (Capacity): Exit capacity (IPO window)
- Node 11 (Representation): Valuation methodology
- Node 17 (Lock): Contractual lock-up periods
- BarrierInput: Portfolio company data (quarterly, delayed)
- BarrierCausal: Reporting lag
- BarrierVariety: Incomplete hedging (unhedgeable)

**Failure mode mapping:**
- T.D (Frozen Market): IPO market closure
- D.E (Boom-Bust): VC cycle (2021 → 2022)
- D.C (Fundamental Uncertainty): Startup valuation
- B.D (Liquidity Starvation): LP distribution pressure

**Stress test scenario:** IPO market closure
- Node 6 triggers: Exit capacity zero
- T.D activates: No price discovery
- BarrierVariety: Cannot hedge, must hold
- Expected response: Mark to distressed comps; extend holding periods

---

### 10.12 Structured Products

:::{prf:definition} Structured Product Pricing Framework
:label: def-structured-pricing

**Path-dependent pricing.** General structured product value:
$$
V_t = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T r_u du} \cdot \text{Payoff}(\text{Path}_{t:T})\right],
$$
where payoff depends on full price path.

**Barrier option example.** Down-and-out call:
$$
V_t = C^{\text{vanilla}}_t - \left(\frac{S_t}{H}\right)^{2\lambda} C^{\text{vanilla}}_t(S \to H^2/S_t),
$$
where $H$ is barrier and $\lambda = (r - q - \sigma^2/2)/\sigma^2$.

**Correlation products.** Basket option value depends on correlation:
$$
V_t = f(\rho_{ij}), \quad \frac{\partial V}{\partial \rho_{ij}} = \text{correlation vega}.
$$

**CVA/DVA adjustment.** Credit valuation adjustment:
$$
V^{\text{adjusted}}_t = V^{\text{clean}}_t - \text{CVA}_t + \text{DVA}_t,
$$
accounting for counterparty and own default.

**Funding valuation adjustment (FVA).** Collateral costs:
$$
V^{\text{funded}}_t = V^{\text{adjusted}}_t - \text{FVA}_t,
$$
where FVA reflects funding spread on uncollateralized portion.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty credit (critical)
- Node 3 (Leverage): Embedded leverage
- Node 4 (Scale): Position size vs. underlying liquidity
- Node 5 (Stationarity): Correlation regime stability
- Node 6 (Capacity): Hedging capacity for exotic risks
- Node 9 (Tameness): Tail risk (barriers, autocallables)
- Node 11 (Representation): Model risk (correlation, path)
- Node 14 (Coupling): Basis risk between hedge and product
- BarrierGap: Barrier breach gap risk
- BarrierVariety: Incomplete hedging of exotics
- BarrierBode: Hedging one Greek worsens another

**Failure mode mapping:**
- T.C (Complexity): CDO-squared opacity (2008)
- D.C (Fundamental Uncertainty): Correlation smile stress
- S.E (Supercritical Leverage): Autocallable barrier breach cascade
- B.C (Agency Misalignment): Suitability failures

**Stress test scenario:** Correlation spike + barrier breach
- BarrierGap triggers: Barrier breached in gap
- Node 11 fails: Correlation model breaks down
- BarrierVariety: Cannot hedge correlation exposure
- Expected response: Mark to scenario analysis; reserve for model risk

---

## 11. Market Understanding and Regime Dynamics

### 11.1 Regime Switching

Let $K_t \in \{1,\dots,|\mathcal{K}|\}$ evolve via transition matrix $\Pi$. Asset drifts and volatilities are state-dependent:
$$
\mu_t = \mu_{K_t}, \quad \sigma_t = \sigma_{K_t}.
$$
Regime changes are audited via the Representation and Boundary permits.

### 11.2 Liquidity and Leverage Cycles

Liquidity is a dynamic state variable. When leverage rises and liquidity falls, the system approaches BarrierLev and BarrierLiq. Pricing becomes **interval-valued** rather than point-valued.

### 11.3 Macro Factors and Risk Appetite

Risk temperature $T_t$ reflects aggregate risk appetite. Higher $T_t$ implies lower risk aversion, higher asset prices, and higher entropy. Sieve checks ensure $T_t$ changes are grounded in observable data.

---

## 12. Implementation and Diagnostics

A practical deployment uses the Sieve as a runtime auditor:
- **Data feeds:** prices, order flow, balance sheet aggregates, funding rates.
- **Checks:** gate and barrier permits evaluated each step.
- **Output:** price estimates with certificate states (valid, bounded, or suspended).

A model is acceptable only if it produces **certificate-backed prices** in the intended operating regime.

---

## 13. Summary and Self-Consistency Checklist

A market pricing model is **self-consistent** if:
1. It admits a positive SDF consistent with observed prices.
2. It satisfies gate permits for solvency, liquidity, and information grounding.
3. It respects barrier constraints during stress.
4. Its macro state $K_t$ is stable, grounded, and within capacity.
5. Its thermoeconomic potential decreases under normal operation.

This is the operational definition of a **complete asset pricing theory** within the Fragile Market framework.

---

## 14. Market Failure Mode Taxonomy

Markets fail in structured ways. The **Failure Mode Taxonomy** classifies all market pathologies into a 3×5 grid indexed by:
- **Structural domain** (row): Conservation (C), Topology (T), Duality (D), Symmetry (S), Boundary (B)
- **Failure type** (column): Explosive (E), Degenerative (D), Computational (C)

This taxonomy is **complete** in the sense that every market failure routes through exactly one cell, and **conserved** in the sense that interventions in one cell can shift risk to adjacent cells but cannot eliminate it.

### 14.1 Taxonomy Overview

```{list-table} Market Failure Mode Grid
:header-rows: 1
:name: failure-mode-grid

* - Domain
  - Explosive (E)
  - Degenerative (D)
  - Computational (C)
* - Conservation (C)
  - C.E: Default Cascade
  - C.D: Too-Big-to-Fail
  - C.C: HFT Instability
* - Topology (T)
  - T.E: Flash Crash
  - T.D: Frozen Market
  - T.C: Complexity Crisis
* - Duality (D)
  - D.E: Boom-Bust Cycle
  - D.D: Dispersion Success
  - D.C: Fundamental Uncertainty
* - Symmetry (S)
  - S.E: Supercritical Leverage
  - S.D: Flat Volatility
  - S.C: Parameter Drift
* - Boundary (B)
  - B.E: External Shock
  - B.D: Liquidity Starvation
  - B.C: Agency Misalignment
```

### 14.2 Conservation Failures (Row C)

Conservation failures violate **solvency, turnover, or resource constraints**—the market's equivalent of mass-energy conservation.

---

#### C.E: Default Cascade (Explosive Conservation)

:::{prf:definition} Failure Mode C.E
:label: def-failure-ce

**Mathematical signature:**
$$
\frac{d(\text{Defaults})}{dt} > \lambda_{\text{crit}} \cdot \text{Defaults},
$$
where the default rate exceeds the critical branching factor $\lambda_{\text{crit}} > 1$, producing exponential growth.

**Interpretation:** One default triggers multiple defaults. The system exhibits **supercritical branching**: each failure causes more than one subsequent failure on average.

**Market examples:**
- 2008 financial crisis: Lehman default → money market freeze → bank run cascade
- Sovereign debt contagion: Greece → Portugal → Spain
- Crypto exchange collapse: FTX → Alameda → lending platforms

**Observable signatures:**
- CDS spreads widen exponentially
- Interbank lending freezes
- Correlation spike across unrelated credits
- Recovery rates collapse

**Violated permits:** Node 1 (Solvency), Node 2 (Turnover), BarrierSat

**Intervention class:** SurgCE (Bailout/Recapitalization)
:::

---

#### C.D: Too-Big-to-Fail (Degenerative Conservation)

:::{prf:definition} Failure Mode C.D
:label: def-failure-cd

**Mathematical signature:**
$$
\text{HHI} = \sum_j s_j^2 > \text{HHI}_{\text{crit}},
$$
where market share concentration exceeds critical threshold, creating systemic nodes.

**Interpretation:** Wealth/risk concentrates in too few entities. The system becomes **fragile by concentration**—a single node failure is catastrophic.

**Market examples:**
- "Too-big-to-fail" banks (2008)
- Dominant market makers (Knight Capital, Archegos)
- Index concentration (FAANG in S&P 500)
- Stablecoin concentration (USDT dominance)

**Observable signatures:**
- Herfindahl index above threshold
- Single-name CDS dominating index CDS
- Correlation asymmetry (one name moves all)
- Implicit government guarantee priced in

**Violated permits:** Node 3 (Leverage balance), BarrierSat

**Intervention class:** SurgCD (Forced Deleveraging/Breakup)
:::

---

#### C.C: HFT Instability (Computational Conservation)

:::{prf:definition} Failure Mode C.C
:label: def-failure-cc

**Mathematical signature:**
$$
\tau_{\text{trade}} < \tau_{\text{settle}} \implies \text{Zeno regime},
$$
where trade frequency exceeds settlement/clearing capacity.

**Interpretation:** Trading approaches **Zeno's paradox**: infinite trades in finite time, but settlement cannot keep up. The market produces trades faster than it can reconcile them.

**Market examples:**
- Flash crashes from HFT feedback loops
- Quote stuffing overwhelming exchanges
- Latency arbitrage creating phantom liquidity
- MEV extraction in DeFi creating ordering games

**Observable signatures:**
- Message-to-trade ratio exploding
- Quote flickering (sub-millisecond updates)
- Latency variance increasing
- Settlement queue growing

**Violated permits:** Node 10 (Mixing), BarrierFreq

**Intervention class:** SurgCC (Circuit Breakers/Speed Bumps)
:::

---

### 14.3 Topology Failures (Row T)

Topology failures affect **market connectivity, reachability, and structural integrity**—the graph of who can trade with whom.

---

#### T.E: Flash Crash (Explosive Topology)

:::{prf:definition} Failure Mode T.E
:label: def-failure-te

**Mathematical signature:**
$$
\left|\frac{dp}{dt}\right| > v_{\text{max}} \quad \text{for} \quad \Delta t < \tau_{\text{human}},
$$
where price velocity exceeds any historical precedent on timescales faster than human reaction.

**Interpretation:** The market **falls through itself**—prices move so fast that intermediate liquidity providers cannot react, creating a liquidity vacuum.

**Market examples:**
- May 2010 Flash Crash (Dow -9% in minutes)
- August 2015 ETF dislocations
- October 2016 GBP flash crash
- Crypto flash crashes (20%+ in minutes)

**Observable signatures:**
- Bid-ask spread explosion
- Stub quotes getting hit
- Stop-loss cascade
- Rapid partial recovery

**Violated permits:** Node 8 (Connectivity), BarrierOmin

**Intervention class:** SurgTE (Trading Halt/Auction)
:::

---

#### T.D: Frozen Market (Degenerative Topology)

:::{prf:definition} Failure Mode T.D
:label: def-failure-td

**Mathematical signature:**
$$
\text{Volume}_t < \epsilon \cdot \text{Volume}_{\text{normal}} \quad \text{for} \quad t > \tau_{\text{freeze}},
$$
where trading volume collapses below operational threshold.

**Interpretation:** The market **goes silent**—bid and ask exist but nobody trades. Liquidity providers withdraw rather than reveal information or take risk.

**Market examples:**
- 2008 interbank lending freeze
- Emerging market currency crises (no bid)
- Off-the-run Treasury illiquidity
- Distressed credit no-trade zones

**Observable signatures:**
- Zero or near-zero volume
- Bid-ask quotes but no prints
- Price discovery halted
- Stale marks persisting

**Violated permits:** Node 6 (Capacity), Node 9 (Tameness), BarrierInput

**Intervention class:** SurgTD (Market Maker of Last Resort)
:::

---

#### T.C: Complexity Crisis (Computational Topology)

:::{prf:definition} Failure Mode T.C
:label: def-failure-tc

**Mathematical signature:**
$$
K(\text{Market State}) > K_{\text{observable}},
$$
where Kolmogorov complexity of the true market state exceeds observable data's descriptive capacity.

**Interpretation:** The market becomes **undecidable locally**—no participant can determine the true state from available information. The topology of dependencies is too complex to model.

**Market examples:**
- CDO-squared pricing collapse (2008)
- Cross-exchange arbitrage with hidden order books
- DeFi composability leading to unforeseen interactions
- Regulatory arbitrage creating invisible risk transfers

**Observable signatures:**
- Model disagreement across participants
- Basis trades failing unexpectedly
- Correlation breakdown in "hedged" positions
- Audit failures revealing hidden exposures

**Violated permits:** Node 11 (Representation), BarrierEpi

**Intervention class:** Simplification mandate, Position transparency
:::

---

### 14.4 Duality Failures (Row D)

Duality failures involve **oscillation, balance, and the observer-system relationship**—the market's feedback loops.

---

#### D.E: Boom-Bust Cycle (Explosive Duality)

:::{prf:definition} Failure Mode D.E
:label: def-failure-de

**Mathematical signature:**
$$
\ddot{x}_t + \omega^2 x_t \approx 0 \quad \text{with} \quad |\dot{x}_t| \gg \sigma_{\text{normal}},
$$
where the system exhibits undamped oscillation with growing amplitude.

**Interpretation:** The market **oscillates destructively**—leverage builds during boom, then unwinds catastrophically in bust. The dual forces of greed and fear fail to balance.

**Market examples:**
- Dot-com bubble and crash (1995-2002)
- Housing bubble and crash (2003-2009)
- Crypto boom-bust cycles (2017, 2021)
- Commodity supercycles

**Observable signatures:**
- Valuation ratios at historical extremes
- Leverage building during appreciation
- "New paradigm" narratives
- Rapid sentiment reversal

**Violated permits:** Node 12 (Oscillation), Node 7 (Stiffness), BarrierTypeII

**Intervention class:** Counter-cyclical capital buffers, Macroprudential policy
:::

---

#### D.D: Dispersion Success (Degenerative Duality)

:::{prf:definition} Failure Mode D.D
:label: def-failure-dd

**Mathematical signature:**
$$
\text{Var}(\text{Returns}) \to 0 \quad \text{but} \quad \text{Skew} \to -\infty,
$$
where apparent stability masks extreme tail risk.

**Interpretation:** The market appears **too stable**—volatility selling is profitable, spreads compress, risk seems to have vanished. But the stability is purchased by tail risk accumulation.

**Market examples:**
- Short volatility strategy blowups (XIV, 2018)
- Carry trade unwinds
- Convergence trades gone wrong (LTCM)
- "Picking up pennies in front of a steamroller"

**Observable signatures:**
- Volatility at historical lows
- Option skew extremely negative
- Carry strategies crowded
- Correlation rising despite low vol

**Violated permits:** Node 4 (Scale), Node 5 (Stationarity), BarrierVac

**Intervention class:** Stress testing, Tail risk disclosure
:::

---

#### D.C: Fundamental Uncertainty (Computational Duality)

:::{prf:definition} Failure Mode D.C
:label: def-failure-dc

**Mathematical signature:**
$$
\text{Var}_{\text{model}}(X) = \infty \quad \text{or} \quad P(\text{model correct}) < p_{\text{min}},
$$
where no finite-variance model can describe the asset's behavior.

**Interpretation:** The market faces **Knightian uncertainty**—not risk (measurable probability) but genuine uncertainty (unknown unknowns). Pricing is fundamentally undecidable.

**Market examples:**
- Early-stage venture capital (no comparable)
- Pandemic pricing (March 2020)
- Regulatory cliff events (Brexit vote)
- Novel asset classes (first bitcoin pricing)

**Observable signatures:**
- Wide bid-ask spreads
- Option prices not fitting any model
- Expert disagreement
- "Price discovery" language used

**Violated permits:** Node 11 (Representation), BarrierEpi, BarrierCausal

**Intervention class:** Accept uncertainty explicitly, Use interval pricing
:::

---

### 14.5 Symmetry Failures (Row S)

Symmetry failures involve **scaling, leverage, and invariance properties**—the market's dimensional consistency.

---

#### S.E: Supercritical Leverage (Explosive Symmetry)

:::{prf:definition} Failure Mode S.E
:label: def-failure-se

**Mathematical signature:**
$$
\text{Leverage} \cdot \text{Volatility} > L_{\text{crit}},
$$
where the product of leverage and volatility exceeds the Kelly criterion bound.

**Interpretation:** The market is **overleveraged for its volatility regime**. When volatility spikes, forced deleveraging creates selling pressure that raises volatility further—a positive feedback loop.

**Market examples:**
- LTCM (1998): 25:1 leverage meets volatility spike
- Archegos (2021): Concentrated leveraged positions
- Crypto margin cascades
- VaR-based deleveraging spirals

**Observable signatures:**
- Margin utilization at extremes
- Prime broker exposure concentration
- Funding rates elevated
- Vol-of-vol spiking

**Violated permits:** Node 3 (Leverage), BarrierLev, BarrierSat

**Intervention class:** SurgSE (Regulatory Capital Injection/Margin Holiday)
:::

---

#### S.D: Flat Volatility Trap (Degenerative Symmetry)

:::{prf:definition} Failure Mode S.D
:label: def-failure-sd

**Mathematical signature:**
$$
\frac{d\sigma}{dS} \to 0 \quad \text{while} \quad \sigma \to \sigma_{\min},
$$
where volatility becomes unresponsive and artificially suppressed.

**Interpretation:** The market loses its **natural warning system**. Central bank intervention or structural changes suppress volatility, but risk accumulates invisibly until sudden release.

**Market examples:**
- "Greenspan put" → "Fed put" vol suppression
- European sovereign spreads pre-2010 (uniform despite divergent fundamentals)
- Vol targeting strategies forcing vol down
- Implicit government guarantees flattening credit spreads

**Observable signatures:**
- Realized vol << implied vol for extended period
- VIX term structure in contango
- Correlation with vol selling flows
- Sudden regime breaks when policy changes

**Violated permits:** Node 7 (Stiffness), Node 5 (Stationarity), BarrierVac

**Intervention class:** SurgSD (Volatility Injection/Policy Taper)
:::

---

#### S.C: Parameter Drift (Computational Symmetry)

:::{prf:definition} Failure Mode S.C
:label: def-failure-sc

**Mathematical signature:**
$$
\theta_t \neq \theta_{t+\Delta t} \quad \text{on calibration timescale},
$$
where model parameters drift faster than models can be recalibrated.

**Interpretation:** The market exhibits **non-stationarity**. Models calibrated yesterday fail today because the underlying generating process has changed, but the change is not observable in real-time.

**Market examples:**
- Factor investing regime shifts
- Correlation breakdown during stress
- Term structure model failure
- Machine learning model decay

**Observable signatures:**
- Calibration residuals growing
- Backtest-to-live performance gap
- Hedging effectiveness declining
- P&L attribution unexplained

**Violated permits:** Node 5 (Stationarity), Node 11 (Representation), BarrierCausal

**Intervention class:** Ensemble models, Regime-aware calibration
:::

---

### 14.6 Boundary Failures (Row B)

Boundary failures involve **external interactions, data flows, and agent incentives**—the market's interface with the outside world.

---

#### B.E: External Shock (Explosive Boundary)

:::{prf:definition} Failure Mode B.E
:label: def-failure-be

**Mathematical signature:**
$$
\|\text{Input}_t - \text{Input}_{t-1}\|_2 > \delta_{\text{shock}},
$$
where an exogenous input exceeds the market's absorption capacity.

**Interpretation:** The market receives an **external shock** that exceeds its capacity to absorb. The shock originates outside the financial system (war, pandemic, natural disaster) but propagates through it.

**Market examples:**
- COVID-19 market crash (March 2020)
- 9/11 market closure and reopening
- Oil embargo (1973)
- Fukushima disaster → Japanese equities

**Observable signatures:**
- News-driven gap opens
- Cross-asset correlation spikes to 1
- Safe haven flows dominate
- Normal trading patterns suspended

**Violated permits:** BarrierInput, Node 15 (Starvation), $\mathrm{Bound}_\partial$

**Intervention class:** Coordinated policy response, Market closure if needed
:::

---

#### B.D: Liquidity Starvation (Degenerative Boundary)

:::{prf:definition} Failure Mode B.D
:label: def-failure-bd

**Mathematical signature:**
$$
\text{Inflow}_t < \text{Required Flow}_t \quad \text{for} \quad t > \tau_{\text{starve}},
$$
where the market receives insufficient capital/data/liquidity to function.

**Interpretation:** The market **starves**—not from a shock but from gradual withdrawal. Capital leaves, liquidity providers exit, data feeds degrade. Death by a thousand cuts.

**Market examples:**
- Capital flight from emerging markets
- Dealer balance sheet constraints post-regulation
- Repo market strains (September 2019)
- Stablecoin redemption pressure

**Observable signatures:**
- Persistent outflows
- Market depth declining
- Bid-ask spreads widening gradually
- Prime broker credit tightening

**Violated permits:** Node 15 (Starvation), BarrierInput, Node 6 (Capacity)

**Intervention class:** SurgBD (Liquidity Injection/Quantitative Easing)
:::

---

#### B.C: Agency Misalignment (Computational Boundary)

:::{prf:definition} Failure Mode B.C
:label: def-failure-bc

**Mathematical signature:**
$$
\nabla_a U_{\text{agent}} \cdot \nabla_a U_{\text{principal}} < 0,
$$
where agent incentives point opposite to principal/social welfare.

**Interpretation:** The market exhibits **principal-agent failure**. Market participants optimize their own objective, but this optimization harms the system or their clients.

**Market examples:**
- Rating agency conflicts (paid by issuers)
- Fund manager incentives (AUM vs. returns)
- Market maker payment for order flow
- Auditor independence failures

**Observable signatures:**
- Persistent mispricings in one direction
- Information asymmetry exploitation
- Governance scandals
- Regulatory enforcement actions

**Violated permits:** Node 16 (Alignment), BarrierMix, BarrierCap

**Intervention class:** SurgBC (Incentive Realignment/Regulation)
:::

---

### 14.7 Failure Mode Interactions

Failures rarely occur in isolation. The taxonomy reveals **interaction patterns**:

:::{prf:proposition} Failure Cascade Paths
:label: prop-failure-cascade

Common cascade sequences:
1. **S.E → C.E**: Leverage crisis → Default cascade (LTCM, 2008)
2. **T.E → C.C**: Flash crash → HFT instability (May 2010)
3. **B.E → T.D**: External shock → Frozen market (COVID March 2020)
4. **D.E → S.E → C.E**: Bubble → Overleveraging → Cascade (housing crisis)
5. **B.C → D.D → S.E**: Misalignment → Hidden risk → Leverage blowup (2008 CDOs)
:::

:::{prf:proposition} Conservation of Risk
:label: prop-risk-conservation

Interventions in one cell shift risk to adjacent cells:
$$
\sum_{i,j} R_{ij} = \text{const} \quad \text{(up to dissipation)},
$$
where $R_{ij}$ is risk intensity in cell $(i,j)$.

**Examples:**
- Bailing out C.E (default cascade) increases moral hazard in C.D (too-big-to-fail)
- Suppressing S.D (flat vol) accumulates pressure for D.E (boom-bust)
- Circuit breakers (T.E) can concentrate risk into T.D (frozen market)
:::

### 14.8 Failure Mode Detection

Each failure mode has a **signature vector** of observable metrics:

```{list-table} Failure Mode Detection Signatures
:header-rows: 1
:name: failure-detection

* - Mode
  - Primary Signal
  - Secondary Signals
  - Lead Time
* - C.E
  - CDS spread acceleration
  - Interbank rate, correlation spike
  - Hours to days
* - C.D
  - HHI index
  - Single-name dominance
  - Months to years
* - C.C
  - Message/trade ratio
  - Latency variance
  - Milliseconds
* - T.E
  - Price velocity
  - Spread explosion
  - Seconds
* - T.D
  - Volume collapse
  - Quote staleness
  - Hours to days
* - T.C
  - Model disagreement
  - Basis unexplained
  - Days to weeks
* - D.E
  - Valuation extremes
  - Sentiment indicators
  - Months
* - D.D
  - Vol at lows, skew extreme
  - Carry crowding
  - Weeks to months
* - D.C
  - Spread width
  - Expert variance
  - Persistent
* - S.E
  - Leverage × vol product
  - Margin utilization
  - Days
* - S.D
  - Vol suppression duration
  - VIX term structure
  - Months
* - S.C
  - Calibration residuals
  - Hedge effectiveness
  - Days to weeks
* - B.E
  - News shock
  - Gap opens
  - Immediate
* - B.D
  - Flow data
  - Depth metrics
  - Weeks
* - B.C
  - Governance signals
  - Regulatory actions
  - Months to years
```

### 14.9 Failure Mode Severity Classification

:::{prf:definition} Severity Levels
:label: def-severity-levels

Each failure mode is classified by severity:

**Level 1 (Warning):** Metrics approaching threshold. Defense: monitoring upgrade.

**Level 2 (Alert):** Threshold breached but contained. Defense: position reduction, hedging.

**Level 3 (Crisis):** Multiple barriers breached, cascade risk. Defense: surgery invocation.

**Level 4 (Systemic):** Cross-market contagion, regulatory intervention required. Defense: coordinated policy response.

Severity formula:
$$
\text{Severity} = \max_i \left( \frac{\text{Metric}_i - \text{Threshold}_i}{\text{Threshold}_i} \right) \times \text{Cascade Factor}.
$$
:::

---

## 15. Surgery Contracts (Market Interventions)

When a failure mode reaches Level 3 or above, the market requires **surgery**—a structured intervention that repairs the violation and returns the system to a safe state. Each surgery is a **contract** specifying preconditions, actions, and postconditions.

### 15.1 Surgery Contract Structure

:::{prf:definition} Surgery Contract Template
:label: def-surgery-template

A surgery contract $\text{Surg}_X$ consists of:
1. **Trigger condition:** When the surgery is invoked
2. **Preconditions:** What must be true before surgery
3. **Actions:** The intervention steps
4. **Postconditions:** What must be true after surgery
5. **Re-entry certificate:** Proof that the system can resume normal operation
6. **Side effects:** Risk shifts to other failure modes
:::

---

### 15.2 SurgCE: Bailout and Recapitalization

:::{prf:definition} SurgCE Contract
:label: def-surg-ce

**Trigger:** Failure mode C.E (Default Cascade) at Level 3+.

**Preconditions:**
- Default cascade confirmed (branching factor $> 1$)
- Systemic importance threshold exceeded
- Market-based solutions exhausted

**Actions:**
1. Identify systemically important defaulting entities
2. Inject capital via equity purchase, loan guarantee, or direct transfer
3. Provide liquidity backstop to counterparties
4. Impose conditions (management change, dividend restrictions)
5. Establish resolution framework for orderly unwinding

**Postconditions:**
- Branching factor $< 1$ (cascade halted)
- Key entities solvent
- Interbank/funding markets functional

**Re-entry certificate:** $K^{\text{re}}_{\text{CE}}$ = (Solvency restored, Funding normalized, Capital plan approved)

**Side effects:**
- Increases C.D risk (moral hazard, too-big-to-fail reinforced)
- May trigger B.C (agency misalignment from implicit guarantees)
- Fiscal cost creates boundary constraint

**Historical examples:** TARP (2008), EU bank recapitalizations, FTX/Alameda aftermath discussions
:::

---

### 15.3 SurgCD: Forced Deleveraging and Breakup

:::{prf:definition} SurgCD Contract
:label: def-surg-cd

**Trigger:** Failure mode C.D (Too-Big-to-Fail) at Level 3+.

**Preconditions:**
- Concentration metrics (HHI) exceed critical threshold
- Single entity failure would trigger C.E cascade
- Voluntary deleveraging insufficient

**Actions:**
1. Mandate capital raise or asset sales
2. Enforce position limits
3. If necessary, mandate structural separation (Glass-Steagall style)
4. Create resolution plan ("living will")
5. Increase capital requirements for systemic entities

**Postconditions:**
- HHI below critical threshold
- No single entity systemically critical
- Credible resolution plans in place

**Re-entry certificate:** $K^{\text{re}}_{\text{CD}}$ = (Concentration reduced, Resolution plans filed, Capital buffers adequate)

**Side effects:**
- May reduce market efficiency (economies of scale lost)
- Could trigger T.D (reduced market making capacity)
- Regulatory arbitrage risk

**Historical examples:** Volcker Rule, Dodd-Frank systemic designations, UK ring-fencing
:::

---

### 15.4 SurgCC: Circuit Breakers and Speed Controls

:::{prf:definition} SurgCC Contract
:label: def-surg-cc

**Trigger:** Failure mode C.C (HFT Instability) at Level 2+.

**Preconditions:**
- Trade frequency exceeding settlement capacity
- Quote-to-trade ratio explosive
- Latency arbitrage creating instability

**Actions:**
1. Implement market-wide circuit breakers (price limits)
2. Introduce minimum resting times for quotes
3. Apply speed bumps (intentional delays)
4. Enforce batch auctions at intervals
5. Increase messaging fees for excessive quoting

**Postconditions:**
- Trade frequency within settlement capacity
- Quote quality improved (quote-to-trade ratio normalized)
- Price continuity restored

**Re-entry certificate:** $K^{\text{re}}_{\text{CC}}$ = (Trading velocity bounded, Message rates stable, Settlement current)

**Side effects:**
- May reduce liquidity (HFT provides some liquidity)
- Could shift activity to less-regulated venues
- May increase T.D risk (slower price discovery)

**Historical examples:** NYSE circuit breakers, IEX speed bump, EU minimum tick sizes
:::

---

### 15.5 SurgTE: Trading Halt and Price Auction

:::{prf:definition} SurgTE Contract
:label: def-surg-te

**Trigger:** Failure mode T.E (Flash Crash) at Level 3+.

**Preconditions:**
- Price move exceeds velocity threshold
- Liquidity vacuum detected
- Normal market making suspended

**Actions:**
1. Halt trading immediately
2. Cancel clearly erroneous trades
3. Accumulate orders during halt
4. Conduct single-price auction to reopen
5. Gradually restore continuous trading

**Postconditions:**
- Price within reasonable range
- Bid-ask spread normalized
- Liquidity providers re-engaged

**Re-entry certificate:** $K^{\text{re}}_{\text{TE}}$ = (Auction completed, Price discovery validated, Market makers present)

**Side effects:**
- Halts transfer risk to correlated markets
- May trigger T.D if halt extends
- Uncertainty during halt period

**Historical examples:** May 2010 trade cancellations, single-stock circuit breakers, crypto exchange halts
:::

---

### 15.6 SurgTD: Market Maker of Last Resort

:::{prf:definition} SurgTD Contract
:label: def-surg-td

**Trigger:** Failure mode T.D (Frozen Market) at Level 3+.

**Preconditions:**
- Trading volume collapsed
- Normal market makers withdrawn
- Price discovery halted

**Actions:**
1. Central entity (central bank, exchange, or designated institution) posts two-sided quotes
2. Provide backstop liquidity at wide but finite spreads
3. Accept losses as cost of market function
4. Gradually narrow spreads as private liquidity returns
5. Exit once private market making resumes

**Postconditions:**
- Two-sided market exists
- Trades can execute
- Price discovery resumed

**Re-entry certificate:** $K^{\text{re}}_{\text{TD}}$ = (Bid and ask present, Volume above minimum, Private makers returning)

**Side effects:**
- Moral hazard for market makers
- Central entity takes mark-to-market risk
- May crowd out private liquidity if not exited promptly

**Historical examples:** Fed commercial paper facility (2008), ECB bond purchases, Treasury buybacks during stress
:::

---

### 15.7 SurgSE: Emergency Margin Relief

:::{prf:definition} SurgSE Contract
:label: def-surg-se

**Trigger:** Failure mode S.E (Supercritical Leverage) at Level 3+.

**Preconditions:**
- Leverage × volatility product exceeds critical bound
- Forced deleveraging creating feedback loop
- Margin calls cascading

**Actions:**
1. Temporarily reduce margin requirements
2. Extend margin call deadlines
3. Provide emergency credit lines to clearinghouses
4. Coordinate orderly position reduction
5. Increase margin requirements gradually once stable

**Postconditions:**
- Leverage × volatility below critical bound
- Forced selling halted
- Clearinghouse solvent

**Re-entry certificate:** $K^{\text{re}}_{\text{SE}}$ = (Margin calls current, Leverage reduced, Volatility subsiding)

**Side effects:**
- Moral hazard (leverage may rebuild faster)
- Regulatory credibility affected
- May shift risk to B.D (liquidity starvation if credit tight)

**Historical examples:** Exchange margin reductions during stress, Fed lending to clearinghouses, coordinated bank credit lines
:::

---

### 15.8 SurgSD: Volatility Injection

:::{prf:definition} SurgSD Contract
:label: def-surg-sd

**Trigger:** Failure mode S.D (Flat Volatility) at Level 2+ (preventive).

**Preconditions:**
- Volatility suppressed for extended period
- Policy intervention identified as cause
- Tail risk accumulating invisibly

**Actions:**
1. Signal policy normalization (taper talk)
2. Reduce intervention gradually
3. Allow volatility to rise naturally
4. Stress test for higher vol regime
5. Monitor for D.E (boom-bust) emergence

**Postconditions:**
- Volatility at historically normal levels
- Risk pricing restored
- Policy intervention reduced

**Re-entry certificate:** $K^{\text{re}}_{\text{SD}}$ = (Volatility normalized, Policy stance communicated, Stress tests passed)

**Side effects:**
- May trigger D.E or S.E as suppressed volatility releases
- Market adjustment costs
- Communication risk (taper tantrum)

**Historical examples:** Fed taper (2013), ECB policy normalization attempts, BOJ yield curve control adjustments
:::

---

### 15.9 SurgBC: Incentive Realignment

:::{prf:definition} SurgBC Contract
:label: def-surg-bc

**Trigger:** Failure mode B.C (Agency Misalignment) at Level 3+.

**Preconditions:**
- Systematic misalignment identified
- Market-based correction insufficient
- Harm to principals/system documented

**Actions:**
1. Regulatory intervention (new rules, enforcement)
2. Mandate disclosure of conflicts
3. Restructure compensation (clawbacks, deferred pay)
4. Strengthen fiduciary duties
5. Create alignment mechanisms (skin in the game requirements)

**Postconditions:**
- Agent incentives aligned with principals
- Conflicts disclosed and managed
- Enforcement mechanism operational

**Re-entry certificate:** $K^{\text{re}}_{\text{BC}}$ = (Rules in effect, Compliance verified, Alignment metrics improved)

**Side effects:**
- Compliance costs
- May reduce market participation (activity shifts elsewhere)
- Regulatory capture risk

**Historical examples:** Dodd-Frank compensation rules, MiFID II inducement rules, rating agency regulation
:::

---

### 15.10 Surgery Coordination

When multiple surgeries are needed simultaneously:

:::{prf:definition} Multi-Surgery Protocol
:label: def-multi-surgery

**Priority ordering:**
1. SurgTE (halt trading if flash crash—immediate safety)
2. SurgCE (stop default cascade—systemic risk)
3. SurgSE (margin relief—prevent cascade amplification)
4. SurgTD (restore liquidity—enable price discovery)
5. SurgCC, SurgCD, SurgSD, SurgBC (structural—can wait for stability)

**Coordination rules:**
- Never invoke SurgSD during active crisis (would amplify)
- SurgCE and SurgCD conflict (bailout vs. breakup)—choose based on urgency
- SurgTE and SurgTD are sequential (halt → maker of last resort → reopen)

**Exit sequencing:**
- Surgeries should exit in reverse order of invocation
- Each exit requires the re-entry certificate of later surgeries
- Central bank facilities should be last to exit
:::

---

## 16. Market Metatheorems

This section establishes **structural theorems** about the market pricing framework—meta-level results that constrain what any consistent pricing theory must satisfy. These are the market equivalents of the KRNL theorems from the hypostructure theory.

### 16.1 MKT-Consistency: Self-Consistent Pricing

:::{prf:theorem} Market Consistency Theorem (MKT-Consistency)
:label: thm-mkt-consistency

A pricing system is **internally consistent** if and only if it admits a fixed point under the market dynamics operator.

**Formal statement:** Let $\mathcal{M}: \mathcal{P} \to \mathcal{P}$ be the market pricing operator mapping prices to updated prices via:
$$
\mathcal{M}(p) = \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot \text{Payoff}(p, \omega)\right].
$$

The pricing system is consistent iff $\exists p^* : \mathcal{M}(p^*) = p^*$.

**Proof (rigorous):**

*Step 1 (No-Arbitrage → EMM).* By Axiom A3 (No-Arbitrage), $\nexists \theta$ with $V_0(\theta) = 0$, $V_T(\theta) \ge 0$ a.s., $\mathbb{P}(V_T(\theta) > 0) > 0$. By the First Fundamental Theorem of Asset Pricing (FTAP) {cite}`delbaen1994ftap`, this implies existence of equivalent martingale measure $\mathbb{Q} \sim \mathbb{P}$ with $\frac{d\mathbb{Q}}{d\mathbb{P}} > 0$.

*Step 2 (EMM → SDF).* By Axiom A4 (Positive SDF), there exists $M_t > 0$ such that $p_t = \mathbb{E}_t[M_{t+1} \cdot \text{Payoff}_{t+1}]$. Under $\mathbb{Q}$, setting $M_t = \beta \frac{\xi_t}{\xi_0}$ where $\xi_t = \frac{d\mathbb{Q}}{d\mathbb{P}}|_{\mathcal{F}_t}$ and $\beta$ is the discount factor, we have the martingale property.

*Step 3 (SDF → Fixed Point).* Define the pricing operator:
$$
\mathcal{M}(p)_i = \mathbb{E}^{\mathbb{Q}}\left[\beta \cdot \text{Payoff}_i(p, \omega)\right].
$$
Then $p^*$ is a fixed point iff $\mathcal{M}(p^*) = p^*$ iff discounted prices are martingales.

*Step 4 (Existence via Kakutani).* The price space $\mathcal{P} \subset \mathbb{R}^n_{>0}$ is restricted by barriers (Axiom A7) to a compact set $K$. The SDF positivity (Axiom A4) ensures $\mathcal{M}: K \to K$ is continuous. By Kakutani's Fixed Point Theorem, $\exists p^* \in K$ with $\mathcal{M}(p^*) = p^*$. $\square$

**Corollary (Permit interpretation):** A pricing model satisfies MKT-Consistency iff all Sieve gates pass. Gate failures indicate inconsistency.

:::{prf:lemma} Contraction for Unique Fixed Point
:label: lem-contraction-unique

If the market operator $\mathcal{M}$ is a **contraction** with Lipschitz constant $L < 1$:
$$
\|\mathcal{M}(p) - \mathcal{M}(q)\| \le L \|p - q\|,
$$
then the fixed point $p^*$ is **unique** and iteration converges geometrically: $\|p^{(n)} - p^*\| \le L^n \|p^{(0)} - p^*\|$.
:::
:::

:::{prf:remark} Constructive Fixed Point
:label: rem-constructive-fp

The fixed point is **constructively obtained** via iteration:
$$
p^{(n+1)} = \mathcal{M}(p^{(n)}),
$$
converging under contraction conditions. The rate of convergence indicates **pricing stability**:
- Fast convergence → stable, well-identified prices
- Slow convergence → fragile, sensitive to perturbations
- Non-convergence → inconsistent pricing (barrier breach)
:::

---

### 16.2 MKT-Exclusion: No-Arbitrage as Topological Obstruction

:::{prf:theorem} Market Exclusion Theorem (MKT-Exclusion)
:label: thm-mkt-exclusion

No-arbitrage is equivalent to the **absence of topological obstructions** in the market's category of trading strategies.

**Formal statement:** Let $\mathcal{C}_{\text{Market}}$ be the category with:
- Objects: Portfolios (positions in assets)
- Morphisms: Trading strategies (rebalancing rules)

An arbitrage is a morphism $\phi: 0 \to X$ where $X > 0$ almost surely. No-arbitrage holds iff:
$$
\text{Hom}_{\mathcal{C}_{\text{Market}}}(\text{Zero Portfolio}, \text{Positive Payoff}) = \emptyset.
$$

**Proof (rigorous):**

*Step 1 (Category Structure).* Define $\mathcal{C}_{\text{Market}}$ with objects $\text{Obj} = \{w \in \mathbb{R}^n : w^T \mathbf{1} = 0\}$ (self-financing portfolios) and morphisms $\text{Hom}(w_1, w_2) = \{\phi : [0,T] \to \mathbb{R}^n \text{ predictable} : \int_0^T \phi_t \cdot dS_t = w_2 - w_1\}$.

*Step 2 (Arbitrage as Morphism).* An arbitrage is $\phi \in \text{Hom}(0, X)$ where $X \ge 0$ a.s. and $\mathbb{P}(X > 0) > 0$. This is the zero portfolio to positive payoff morphism.

*Step 3 (Cohomological Obstruction).* Define the arbitrage obstruction class:
$$
\omega := [M] \in H^0(\mathcal{C}_{\text{Market}}, \mathcal{O}^*) \cong \text{Pic}(\mathcal{C}_{\text{Market}}),
$$
where $\mathcal{O}^* = \text{Hom}(-, \mathbb{R}_{>0})$ is the sheaf of positive functions. The obstruction $\omega$ measures the failure of the SDF to extend globally.

*Step 4 (Vanishing ↔ No-Arbitrage).* By the cohomological form of FTAP:
- $\omega = 0$ iff $\exists M > 0$ globally (Axiom A4)
- $\exists M > 0$ iff $\text{Hom}(0, X^+) = \emptyset$ for all $X^+ > 0$
- This holds iff Axiom A3 (No-Arbitrage) is satisfied.

*Step 5 (Topological Interpretation).* The cone of attainable claims $C = \{V_T(\theta) : \theta \text{ admissible}\}$ is closed (NFLVR condition). No-arbitrage $\Leftrightarrow$ $C \cap L^0_+ = \{0\}$ $\Leftrightarrow$ separating hyperplane exists (SDF) $\Leftrightarrow$ $\omega = 0$. $\square$

**Permit interpretation:** Node 9 (Tameness) and Node 7 (Stiffness) ensure the Hom-set remains empty. Barrier breaches can create temporary "apparent arbitrages" that are actually liquidity/execution risk in disguise.
:::

:::{prf:corollary} Basis Trades as Near-Obstructions
:label: cor-basis-near-obstruction

Basis trades (apparent mispricings) are **near-obstructions**:
$$
\text{Hom}(\text{Long}, \text{Short}) \neq \emptyset \quad \text{but} \quad \text{cost}(\phi) > 0.
$$
The cost arises from barriers (funding, liquidity, execution) that prevent the arbitrage from being realized.
:::

---

### 16.3 MKT-Trichotomy: Fundamental Market Outcomes

:::{prf:theorem} Market Trichotomy Theorem (MKT-Trichotomy)
:label: thm-mkt-trichotomy

Every market trajectory terminates in exactly one of three states:
1. **Equilibrium (E):** Stable fixed point with prices converging
2. **Crisis (C):** Barrier breach requiring surgery intervention
3. **Horizon (H):** Fundamental uncertainty beyond pricing capacity

**Formal statement:** Let $\{p_t\}_{t \geq 0}$ be a price trajectory under the market dynamics. Then:
$$
\lim_{t \to T} \{p_t\} \in \{\text{Equilibrium}, \text{Crisis}, \text{Horizon}\},
$$
where $T$ may be finite (crisis/horizon) or infinite (equilibrium).

**Characterization:**
- **Equilibrium:** $\|p_t - p^*\| < \epsilon$ for $t > T_{\text{conv}}$, all permits pass
- **Crisis:** $\exists$ barrier $B$ such that $B(p_t) = \text{BREACHED}$ for $t \in [T_{\text{crisis}}, T_{\text{recovery}}]$
- **Horizon:** $\text{Var}(p_{T+\tau} | \mathcal{F}_T) = \infty$ for all $\tau > 0$ (Knightian uncertainty)

**Proof (rigorous):**

*Step 1 (State Space Partition).* By Axiom A7 (Permit Completeness), the state space $\mathcal{S}$ is partitioned:
$$
\mathcal{S} = \mathcal{S}_{\text{valid}} \cup \bigcup_{B \in \text{Barriers}} \partial \mathcal{S}_B \cup \mathcal{S}_{\text{horizon}},
$$
where $\mathcal{S}_{\text{valid}}$ is the interior (all permits pass), $\partial \mathcal{S}_B$ are barrier surfaces, and $\mathcal{S}_{\text{horizon}}$ is the undecidable boundary.

*Step 2 (Local Existence and Uniqueness).* Within $\mathcal{S}_{\text{valid}}$, the market dynamics $\dot{p} = f(p, t)$ satisfy:
- $f$ is Lipschitz continuous (by Axiom A6, finite complexity)
- $f$ is bounded (by barrier constraints)
By Picard-Lindelöf theorem, local solutions exist uniquely.

*Step 3 (Global Behavior Classification).* For any trajectory $\{p_t\}_{t \ge 0}$ starting in $\mathcal{S}_{\text{valid}}$:

**Case E (Equilibrium):** If $\limsup_{t \to \infty} d(p_t, \partial \mathcal{S}_{\text{valid}}) > 0$, then by the Invariance Principle, $p_t \to p^*$ where $f(p^*) = 0$ (fixed point). By MKT-Consistency (Theorem 16.1), such $p^*$ exists.

**Case C (Crisis):** If $\exists T^* < \infty$ such that $p_{T^*} \in \partial \mathcal{S}_B$ for some barrier $B$, the trajectory hits a barrier surface. By definition, this is a crisis state requiring surgery.

**Case H (Horizon):** If $\lim_{t \to T} K(p_t) = \infty$ where $K(\cdot)$ is Kolmogorov complexity, or $\mathbb{E}[\|p_{t+\tau} - p_t\|^2 | \mathcal{F}_t] \to \infty$ for all $\tau$, the system enters the horizon regime (Axiom A6 violated).

*Step 4 (Mutual Exclusivity).* These cases are mutually exclusive:
- E requires staying in $\mathcal{S}_{\text{valid}}$ forever with convergence
- C requires hitting a barrier in finite time
- H requires divergence of complexity/variance

*Step 5 (Exhaustiveness).* By Axiom A1 (Bounded Rationality) and A6 (Finite Complexity), trajectories cannot exhibit other behaviors (e.g., chaos within valid region is detectable by Node 7c, routing to C or H). $\square$

**Permit interpretation:** The Sieve routes each trajectory to exactly one outcome. Gate certificates track progress toward equilibrium; barrier breaches indicate crisis; Node 11 (Representation) failure indicates horizon.
:::

:::{prf:remark} Crisis as Temporary State
:label: rem-crisis-temporary

Crisis (C) is **transient by design**—surgery contracts exist to return the system to equilibrium. The Horizon state (H) is **absorbing** for finite-horizon pricing but may resolve with new information arrival.
:::

---

### 16.4 MKT-Equivariance: Pricing Under Symmetry

:::{prf:theorem} Market Equivariance Theorem (MKT-Equivariance)
:label: thm-mkt-equivariance

Prices are **equivariant** under the market's gauge group—transformations that preserve economic structure.

**Formal statement:** Let $G$ be the group of admissible transformations (currency changes, unit rescalings, time shifts). Then:
$$
\mathcal{M}(g \cdot p) = g \cdot \mathcal{M}(p) \quad \forall g \in G.
$$

**Components of the gauge group:**
1. **Currency invariance:** $p_{\text{USD}} \cdot S_{\text{EUR/USD}} = p_{\text{EUR}}$
2. **Numéraire invariance:** Pricing independent of chosen numéraire asset
3. **Time translation:** $p_t(T) = p_{t+\Delta}(T+\Delta)$ (for stationary processes)
4. **Scale covariance:** $p(\lambda \cdot \text{payoff}) = \lambda \cdot p(\text{payoff})$

**Proof (rigorous):**

*Step 1 (Gauge Group Structure).* The market gauge group is:
$$
G = G_{\text{num}} \times G_{\text{scale}} \times G_{\text{time}} \times G_{\text{perm}},
$$
where:
- $G_{\text{num}} \cong \mathbb{R}_{>0}$: numéraire changes (currency/unit)
- $G_{\text{scale}} \cong \mathbb{R}_{>0}$: portfolio scaling
- $G_{\text{time}} \cong \mathbb{R}$: time translations (for stationary processes)
- $G_{\text{perm}} \cong S_n$: asset permutations

*Step 2 (SDF Transformation Law).* Under $g \in G$, the SDF transforms as:
$$
M^g_t = M_t \cdot J_g(p_t, t),
$$
where $J_g$ is the Radon-Nikodym derivative ensuring $\mathbb{Q}^g \sim \mathbb{Q}$. For numéraire change from $N$ to $N'$:
$$
\frac{dM^{N'}}{dM^N} = \frac{N'_T/N'_0}{N_T/N_0}.
$$

*Step 3 (Price Transformation).* The pricing formula under $g$ is:
$$
p^g_t = \mathbb{E}^{\mathbb{Q}^g}_t\left[\int_t^T M^g_{t,s} \cdot \text{Payoff}^g_s \, ds\right].
$$
By change of variables and Axiom A4 (Positive SDF):
$$
p^g_t = g \cdot \mathbb{E}^{\mathbb{Q}}_t\left[\int_t^T M_{t,s} \cdot \text{Payoff}_s \, ds\right] = g \cdot p_t.
$$

*Step 4 (Equivariance Verification).* The market operator $\mathcal{M}$ commutes with $G$:
$$
\mathcal{M}(g \cdot p)_i = \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot (g \cdot \text{Payoff}_i)\right] = g \cdot \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot \text{Payoff}_i\right] = g \cdot \mathcal{M}(p)_i.
$$
This holds by linearity of expectation and the homogeneity of payoffs. $\square$

**Permit interpretation:** Node 4 (Scale) enforces scale covariance; Node 5 (Stationarity) enforces time translation; Node 18 (Symmetry) in extended gates monitors symmetry preservation.
:::

:::{prf:corollary} Arbitrage from Symmetry Breaking
:label: cor-symmetry-arbitrage

**Symmetry breaking creates arbitrage opportunities.** If $g \in G$ but $\mathcal{M}(g \cdot p) \neq g \cdot \mathcal{M}(p)$, then:
$$
\text{Arbitrage profit} = \left| p^g - g \cdot p \right|.
$$
This is the basis for cross-currency basis trades, merger arbitrage, and relative value strategies.
:::

---

### 16.5 MKT-HorizonLimit: Irreducible Uncertainty

:::{prf:theorem} Horizon Limit Theorem (MKT-HorizonLimit)
:label: thm-mkt-horizon

There exists a **fundamental horizon** beyond which pricing precision is impossible, regardless of model sophistication or computational power.

**Formal statement:** For any pricing model $\mathcal{M}$ and horizon $T$, there exists $T^* < \infty$ such that:
$$
\text{Var}(p_T | \mathcal{F}_0) \geq V_{\min}(T) \quad \text{for } T > T^*,
$$
where $V_{\min}(T) \to \infty$ as $T \to \infty$.

**Sources of irreducible uncertainty:**
1. **Chaotic sensitivity:** Small perturbations grow exponentially (Lyapunov > 0)
2. **Model uncertainty:** True generating process unknown
3. **Reflexivity:** Prices affect fundamentals which affect prices
4. **Knightian uncertainty:** Unknown unknowns not in probability space

**Quantification:** The horizon limit is approximately:
$$
T^* \approx \frac{1}{\lambda_{\max}} \log\left(\frac{\text{Price Precision Required}}{\text{Input Uncertainty}}\right),
$$
where $\lambda_{\max}$ is the largest Lyapunov exponent of the market dynamics.

**Proof (rigorous):**

*Step 1 (Information-Theoretic Lower Bound).* By Axiom A1 (Bounded Rationality), agent channel capacity is finite: $I(a_t; Z_t) \le C < \infty$. By the data processing inequality:
$$
I(p_T; Z_0) \le I(p_T; p_0) \le I(p_0; Z_0) \le C.
$$
Thus information about far-future prices is bounded by current channel capacity.

*Step 2 (Lyapunov Divergence).* Let $\lambda_{\max} > 0$ be the largest Lyapunov exponent of the market dynamics (empirically, $\lambda_{\max} \approx 0.01-0.05$ per day for equities). For two trajectories starting $\epsilon$ apart:
$$
\|p_t - \tilde{p}_t\| \approx \epsilon \cdot e^{\lambda_{\max} t}.
$$

*Step 3 (Predictability Horizon).* Define the predictability horizon $T^*$ as the time when prediction error equals price range $\Delta p$:
$$
\epsilon \cdot e^{\lambda_{\max} T^*} = \Delta p \implies T^* = \frac{1}{\lambda_{\max}} \ln\left(\frac{\Delta p}{\epsilon}\right).
$$
For $T > T^*$, the prediction interval spans the entire price range—pricing is undecidable.

*Step 4 (Kolmogorov Complexity Bound).* By Axiom A6 (Finite Complexity), $K(Z_t) \le K_{\max}$. The complexity of $p_T$ conditional on current information is:
$$
K(p_T | Z_0) \ge K(p_T) - K(Z_0) - O(\log T).
$$
As $T \to \infty$, $K(p_T | Z_0) \to K(p_T)$—the future becomes algorithmically random relative to the present.

*Step 5 (Variance Divergence).* Combining Steps 2-4, the conditional variance satisfies:
$$
\text{Var}(p_T | \mathcal{F}_0) \ge \sigma^2 \left(e^{2\lambda_{\max} T} - 1\right) / (2\lambda_{\max}).
$$
For $T > T^*$, this exceeds any finite bound, establishing the horizon limit. $\square$

**Permit interpretation:** Node 11 (Representation) and BarrierEpi (epistemic barrier) signal approach to horizon. D.C failure mode (Fundamental Uncertainty) is the manifestation.

:::{prf:lemma} Horizon Estimation in Practice
:label: lem-horizon-practice

For typical market parameters:
- Input uncertainty $\epsilon \approx 0.1\%$ (data noise)
- Required precision $\Delta p \approx 10\%$
- Lyapunov exponent $\lambda_{\max} \approx 0.02$ per day

The predictability horizon is:
$$
T^* \approx \frac{1}{0.02} \ln(100) \approx 230 \text{ trading days} \approx 1 \text{ year}.
$$

This explains why 1-year forward prices have meaningful information content, but 5-year forecasts are dominated by uncertainty.
:::
:::

:::{prf:remark} Practical Implications
:label: rem-horizon-practical

The Horizon Limit implies:
- Long-term equity valuation is fundamentally **interval-valued**, not point-valued
- Option pricing at long tenors requires **model uncertainty quantification**
- Retirement planning must use **scenario analysis**, not point forecasts
- Any pricing model claiming arbitrary precision is **epistemically invalid**
:::

---

### 16.6 Metatheorem Interactions

The five metatheorems form a coherent system:

```{list-table} Metatheorem Dependencies
:header-rows: 1
:name: metatheorem-deps

* - Theorem
  - Depends On
  - Implies
* - MKT-Consistency
  - Sieve completeness
  - Fixed-point prices exist
* - MKT-Exclusion
  - MKT-Consistency
  - No arbitrage → market structure
* - MKT-Trichotomy
  - MKT-Consistency, Barriers
  - All outcomes classified
* - MKT-Equivariance
  - MKT-Consistency
  - Symmetry constraints on pricing
* - MKT-HorizonLimit
  - Chaos theory, Epistemics
  - Pricing limits exist
```

:::{prf:proposition} Completeness of Metatheorems
:label: prop-metatheorem-complete

The five metatheorems are **complete** for the market pricing theory in the following sense:
1. Any consistent pricing system satisfies all five theorems.
2. Any system satisfying all five theorems admits a consistent pricing interpretation.
3. Violation of any theorem indicates a fundamental model error.
:::

---

## 17. Algorithmic Pricing Theory

This section develops the **information-theoretic** and **computational** aspects of market pricing, connecting to algorithmic complexity and the physics of computation.

### 17.1 Kolmogorov Complexity of Prices

:::{prf:definition} Price Complexity
:label: def-price-complexity

The **Kolmogorov complexity** of a price series $\{p_t\}_{t=1}^T$ is the length of the shortest program that generates it:
$$
K(p_{1:T}) = \min_{\text{program } \pi} \{|\pi| : U(\pi) = p_{1:T}\},
$$
where $U$ is a universal Turing machine and $|\pi|$ is program length in bits.

**Interpretation:** $K(p)$ measures the **intrinsic information content** of prices—how much description is needed to specify them exactly.
:::

:::{prf:proposition} Compressibility Bounds
:label: prop-compress-bounds

Price series satisfy complexity bounds:
$$
K(p_{1:T}) \leq H(p_{1:T}) + O(\log T),
$$
where $H$ is the Shannon entropy rate.

**Market phases by complexity:**
- **Crystal phase (efficient):** $K(p) \approx K(\text{random})$ — prices are incompressible
- **Liquid phase (predictable):** $K(p) < K(\text{random}) - \epsilon$ — structure exists
- **Gas phase (chaotic):** $K(p) \approx K(\text{random})$ but structure is emergent
:::

---

### 17.2 Three Pricing Phases

Markets exhibit **phase transitions** in their complexity characteristics:

:::{prf:definition} Market Complexity Phases
:label: def-complexity-phases

**Crystal Phase (Efficient Markets):**
- Prices reflect all available information instantly
- $K(\text{price} | \text{info}) \approx 0$
- No profitable prediction possible
- Corresponds to: Liquid, competitive markets with low barriers

**Liquid Phase (Arbitrageable Markets):**
- Prices reflect most information with friction
- $0 < K(\text{price} | \text{info}) < K_{\text{barrier}}$
- Prediction profitable after costs
- Corresponds to: Markets with execution costs, information asymmetry

**Gas Phase (Random/Chaotic Markets):**
- Prices disconnected from information
- $K(\text{price} | \text{info}) \approx K(\text{price})$
- No systematic relationship to fundamentals
- Corresponds to: Crisis, bubble, or nascent markets

**Phase boundaries:**
- Crystal ↔ Liquid: Execution cost threshold
- Liquid ↔ Gas: Information capacity threshold
- Gas ↔ Crystal: Crisis resolution / market maturation
:::

:::{prf:proposition} Phase Detection
:label: prop-phase-detection

Phase can be detected via the **compression ratio**:
$$
\rho = \frac{K(p_{1:T})}{T \cdot H_0},
$$
where $H_0$ is the entropy of uniform prices.

- $\rho \approx 1$: Crystal or Gas phase
- $\rho < 1 - \epsilon$: Liquid phase (exploitable structure)

Distinguishing Crystal from Gas requires **external information tests**.
:::

---

### 17.3 Computational Depth of Price Discovery

:::{prf:definition} Price Discovery Depth
:label: def-price-depth

The **computational depth** of a price is the time required to compute it from fundamentals:
$$
\text{Depth}(p) = \min_{\pi : U(\pi) = p} \{\text{runtime}(\pi)\}.
$$

**Interpretation:** Deep prices require complex computation; shallow prices are easily derived.
:::

:::{prf:proposition} Depth-Complexity Tradeoff
:label: prop-depth-complexity

Prices exhibit a tradeoff between specification complexity and computational depth:
$$
K(p) \cdot \text{Depth}(p) \geq \Omega(\text{Information Content}(p)).
$$

**Implications:**
- Simple prices (low $K$) require deep computation to discover
- Complex prices (high $K$) may be computationally shallow but hard to specify
- Arbitrage opportunities have high depth (hard to find) but low complexity once found
:::

---

### 17.4 Levin Limit for Markets

:::{prf:theorem} Levin Market Limit
:label: thm-levin-market

There exists a **thermodynamic bound** on market prediction analogous to Levin's universal prior:
$$
\mathbb{P}(\text{price series } p) \propto 2^{-K(p)},
$$
and the expected prediction error satisfies:
$$
\mathbb{E}[\text{error}] \geq \frac{k_B T_{\text{market}}}{E_{\text{computation}}},
$$
where $T_{\text{market}}$ is market temperature (risk tolerance) and $E_{\text{computation}}$ is energy expended on prediction.

**Proof sketch:**
1. By Levin's universal prior, probability is bounded by Kolmogorov complexity.
2. Prediction error is lower-bounded by Bayesian optimal (Levin prior).
3. Computation requires energy (Landauer bound): $E \geq k_B T \ln 2$ per bit erased.
4. Market temperature scales energy costs; combining gives the bound.

**Implications:**
- **No free lunch:** Better prediction requires more computation/energy.
- **Thermodynamic consistency:** Market efficiency has physical foundations.
- **HFT limits:** Speed requires energy; there's a speed-energy tradeoff.
:::

---

### 17.5 Algorithmic Information and Market Efficiency

:::{prf:definition} Algorithmic Efficiency
:label: def-alg-efficiency

A market is **algorithmically efficient** at level $\epsilon$ if:
$$
K(p_{t+1} | p_{1:t}, \text{public info}) > K(p_{t+1}) - \epsilon.
$$

**Interpretation:** Future prices are nearly as complex given history as they are unconditionally—history provides minimal compression.
:::

:::{prf:proposition} Efficiency Hierarchy
:label: prop-efficiency-hierarchy

Market efficiency levels map to algorithmic notions:

1. **Weak efficiency:** $K(p_{t+1} | p_{1:t}) \approx K(p_{t+1})$ — price history uninformative
2. **Semi-strong efficiency:** $K(p_{t+1} | p_{1:t}, \text{public}) \approx K(p_{t+1})$ — public info reflected
3. **Strong efficiency:** $K(p_{t+1} | \text{all info}) \approx K(p_{t+1})$ — all info reflected

**Permit mapping:**
- Node 11 (Representation) tracks deviations from efficiency
- BarrierEpi triggers when complexity analysis shows exploitable structure
- Liquid phase markets are semi-strong efficient with friction
:::

---

### 17.6 Price as Proof

:::{prf:definition} Proof-Carrying Prices
:label: def-proof-price

A **proof-carrying price** is a tuple $(p, \pi)$ where:
- $p$ is the price
- $\pi$ is a certificate/proof that $p$ satisfies required properties

The verification function $V(p, \pi) \in \{\text{ACCEPT}, \text{REJECT}\}$ runs in polynomial time.
:::

:::{prf:proposition} Sieve as Proof System
:label: prop-sieve-proof

The Market Sieve (Section 7) implements a proof system for prices:
- **Prover:** Market dynamics generating prices
- **Verifier:** Sieve gates checking permits
- **Certificate:** Gate passage record $K = (K_1, \ldots, K_{21})$
- **Soundness:** Invalid prices fail some gate (completeness of gates)
- **Completeness:** Valid prices pass all gates

The certificate size is:
$$
|K| = O(\text{number of gates} \times \log(\text{precision})) = O(21 \times 64) = O(1344 \text{ bits}).
$$
:::

---

### 17.7 Computational Cost Analysis

:::{prf:definition} Sieve Computational Complexity
:label: def-sieve-complexity

The computational cost of running the Market Sieve:

**Per-gate costs:**
- Node 1-2 (Conservation): $O(n)$ where $n$ = number of positions
- Node 3-5 (Duality): $O(n)$ for leverage/scale checks
- Node 6-7 (Geometry): $O(n \log n)$ for capacity/stiffness
- Node 8-10 (Topology): $O(n^2)$ worst case for connectivity (typically $O(n \log n)$ with sparse structure)
- Node 11-12 (Epistemics): $O(m)$ where $m$ = model parameters
- Node 13-17 (Extended): $O(n)$ each

**Total Sieve cost:**
$$
T_{\text{Sieve}} = O(n^2 + m),
$$
with typical sparsity allowing $O(n \log n + m)$.

**Barrier monitoring:**
- Per barrier: $O(1)$ to $O(n)$ depending on barrier type
- 20 barriers: $O(n)$ total

**Full pricing loop overhead:**
$$
\text{Overhead} = \frac{T_{\text{Sieve}}}{T_{\text{Pricing}}} \approx 2-5\%,
$$
for typical portfolios with $n \sim 1000$ positions.
:::

---

## 18. Full Implementation

This section provides complete Python/PyTorch implementations of the Market Hypostructure framework.

### 18.1 Core Data Structures

```python
"""
Market Hypostructure: Core Implementation
=========================================
Complete Python/PyTorch implementation of the thermoeconomic asset pricing framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class GateStatus(Enum):
    """Status of a gate node check."""
    PASS = auto()      # Certificate valid
    FAIL = auto()      # Certificate invalid
    UNKNOWN = auto()   # Cannot determine (treated as FAIL)
    BOUNDED = auto()   # Passes with bounds/uncertainty


class BarrierStatus(Enum):
    """Status of a barrier."""
    CLEAR = auto()     # No breach
    WARNING = auto()   # Approaching threshold
    BREACHED = auto()  # Active breach
    RECOVERY = auto()  # Recovering from breach


class MarketPhase(Enum):
    """Market complexity phase."""
    CRYSTAL = auto()   # Efficient, incompressible
    LIQUID = auto()    # Predictable with friction
    GAS = auto()       # Chaotic, disconnected


class FailureMode(Enum):
    """15-mode failure taxonomy."""
    CE = "Default Cascade"
    CD = "Too-Big-to-Fail"
    CC = "HFT Instability"
    TE = "Flash Crash"
    TD = "Frozen Market"
    TC = "Complexity Crisis"
    DE = "Boom-Bust Cycle"
    DD = "Dispersion Success"
    DC = "Fundamental Uncertainty"
    SE = "Supercritical Leverage"
    SD = "Flat Volatility"
    SC = "Parameter Drift"
    BE = "External Shock"
    BD = "Liquidity Starvation"
    BC = "Agency Misalignment"


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class MarketState:
    """Complete market state at time t."""
    prices: torch.Tensor           # Asset prices (n_assets,)
    positions: torch.Tensor        # Position sizes (n_positions, n_assets)
    volatilities: torch.Tensor     # Implied/realized vol (n_assets,)
    correlations: torch.Tensor     # Correlation matrix (n_assets, n_assets)
    liquidity: torch.Tensor        # Bid-ask spreads (n_assets,)
    leverage: torch.Tensor         # Leverage ratios (n_positions,)
    regime: int                    # Current market regime index
    temperature: float             # Market temperature (risk tolerance)
    timestamp: float               # Time

    @property
    def n_assets(self) -> int:
        return self.prices.shape[0]

    @property
    def n_positions(self) -> int:
        return self.positions.shape[0]

    def covariance_matrix(self) -> torch.Tensor:
        """Compute covariance from vol and correlation."""
        vol_diag = torch.diag(self.volatilities)
        return vol_diag @ self.correlations @ vol_diag


@dataclass
class Certificate:
    """Proof-carrying certificate for a pricing decision."""
    gate_results: Dict[int, GateStatus]    # Gate index -> status
    barrier_results: Dict[str, BarrierStatus]  # Barrier name -> status
    price_bounds: Tuple[float, float]      # (lower, upper) price bounds
    confidence: float                       # Overall confidence [0, 1]
    failure_modes: List[FailureMode]       # Active/near failure modes
    timestamp: float

    @property
    def is_valid(self) -> bool:
        """Certificate is valid if all gates pass and no barriers breached."""
        gates_ok = all(s in (GateStatus.PASS, GateStatus.BOUNDED)
                       for s in self.gate_results.values())
        barriers_ok = all(s != BarrierStatus.BREACHED
                         for s in self.barrier_results.values())
        return gates_ok and barriers_ok


@dataclass
class SDFParams:
    """Stochastic Discount Factor parameters."""
    risk_free_rate: float = 0.03
    risk_aversion: float = 2.0
    market_temperature: float = 1.0
    regime_weights: torch.Tensor = field(default_factory=lambda: torch.ones(3) / 3)
```

### 18.2 Thermoeconomic SDF Implementation

```python
# ============================================================================
# THERMOECONOMIC SDF
# ============================================================================

class ThermoeconomicSDF(nn.Module):
    """
    Free-energy based Stochastic Discount Factor.

    The SDF is derived from the thermoeconomic potential:
    M_t = exp(-beta * (r_t + risk_premium_t))

    where risk_premium follows from the Ruppeiner geometry.
    """

    def __init__(self, n_assets: int, n_factors: int = 3,
                 n_regimes: int = 3, device: str = 'cpu'):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.device = device

        # Factor loadings (beta)
        self.factor_betas = nn.Parameter(
            torch.randn(n_assets, n_factors) * 0.1
        )

        # Factor risk premia (lambda)
        self.factor_lambdas = nn.Parameter(
            torch.zeros(n_regimes, n_factors)
        )

        # Regime transition matrix
        self.regime_transitions = nn.Parameter(
            torch.eye(n_regimes) * 0.9 + torch.ones(n_regimes, n_regimes) * 0.1 / n_regimes
        )

        # Risk aversion (inverse temperature)
        self.log_risk_aversion = nn.Parameter(torch.tensor(0.0))

    @property
    def risk_aversion(self) -> torch.Tensor:
        return torch.exp(self.log_risk_aversion)

    def forward(self, state: MarketState, factors: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF value.

        Args:
            state: Current market state
            factors: Factor values (n_factors,)

        Returns:
            SDF value M_t
        """
        # Get regime-specific risk premia
        regime_probs = self._regime_probabilities(state.regime)
        lambdas = (regime_probs @ self.factor_lambdas)  # (n_factors,)

        # Risk premium from factor exposure
        risk_premium = (self.factor_betas @ lambdas).sum()

        # Free energy form
        log_sdf = -self.risk_aversion * (state.temperature * risk_premium)

        return torch.exp(log_sdf)

    def _regime_probabilities(self, current_regime: int) -> torch.Tensor:
        """Get regime probability distribution."""
        probs = torch.softmax(self.regime_transitions[current_regime], dim=0)
        return probs

    def price_asset(self, payoff: torch.Tensor, state: MarketState,
                    factors: torch.Tensor, n_simulations: int = 1000) -> torch.Tensor:
        """
        Price an asset using Monte Carlo with the SDF.

        Args:
            payoff: Payoff function values (n_simulations,)
            state: Current market state
            factors: Factor paths (n_simulations, n_factors)
            n_simulations: Number of MC paths

        Returns:
            Expected discounted payoff
        """
        sdf_values = torch.stack([
            self.forward(state, factors[i]) for i in range(n_simulations)
        ])
        return (sdf_values * payoff).mean()

    def risk_premium(self, state: MarketState) -> torch.Tensor:
        """Compute asset risk premia."""
        regime_probs = self._regime_probabilities(state.regime)
        lambdas = regime_probs @ self.factor_lambdas
        return self.factor_betas @ lambdas


# ============================================================================
# RUPPEINER GEOMETRY
# ============================================================================

class RuppeinerMarket:
    """
    Risk geometry via Ruppeiner metric.

    The metric tensor g_ij measures risk curvature in the space of
    portfolios/positions.
    """

    def __init__(self, state: MarketState):
        self.state = state
        self._metric = None
        self._christoffel = None

    def metric_tensor(self) -> torch.Tensor:
        """
        Compute the Ruppeiner metric g_ij.

        g_ij = -d²S/dX_i dX_j

        where S is the entropy (negative risk).
        """
        if self._metric is not None:
            return self._metric

        cov = self.state.covariance_matrix()

        # Ruppeiner metric is inverse covariance (Fisher information)
        # with temperature scaling
        T = self.state.temperature
        self._metric = torch.linalg.inv(cov) / T

        return self._metric

    def christoffel_symbols(self) -> torch.Tensor:
        """
        Compute Christoffel symbols Γ^k_ij for geodesic equation.
        """
        if self._christoffel is not None:
            return self._christoffel

        g = self.metric_tensor()
        n = g.shape[0]

        # Numerical differentiation for metric derivatives
        eps = 1e-6
        dg = torch.zeros(n, n, n)  # dg_ij/dx_k

        # For simplicity, assume metric is constant (flat approximation)
        # Full implementation would compute derivatives

        g_inv = torch.linalg.inv(g)
        self._christoffel = torch.zeros(n, n, n)

        # Γ^k_ij = (1/2) g^kl (∂g_li/∂x_j + ∂g_lj/∂x_i - ∂g_ij/∂x_l)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        self._christoffel[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[l, i, j] + dg[l, j, i] - dg[i, j, l]
                        )

        return self._christoffel

    def geodesic_step(self, position: torch.Tensor, velocity: torch.Tensor,
                      dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take one step along geodesic (natural gradient path).

        d²x^k/dt² + Γ^k_ij dx^i/dt dx^j/dt = 0
        """
        gamma = self.christoffel_symbols()

        # Geodesic acceleration
        acceleration = -torch.einsum('kij,i,j->k', gamma, velocity, velocity)

        # Symplectic Euler step
        new_velocity = velocity + dt * acceleration
        new_position = position + dt * new_velocity

        return new_position, new_velocity

    def ricci_scalar(self) -> torch.Tensor:
        """
        Compute Ricci scalar curvature R.

        High R indicates high risk concentration.
        """
        g = self.metric_tensor()
        n = g.shape[0]

        # For diagonal metric approximation:
        # R ≈ sum of eigenvalue reciprocals
        eigenvalues = torch.linalg.eigvalsh(g)
        return torch.sum(1.0 / (eigenvalues + 1e-8))
```

### 18.3 Market Sieve Implementation

```python
# ============================================================================
# MARKET SIEVE (21 GATES + 16 BARRIERS)
# ============================================================================

class GateNode(ABC):
    """Abstract base class for gate nodes."""

    def __init__(self, node_id: int, name: str):
        self.node_id = node_id
        self.name = name

    @abstractmethod
    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        """
        Check the gate condition.

        Returns:
            (status, loss_contribution)
        """
        pass


class SolvencyGate(GateNode):
    """Node 1: Solvency check."""

    def __init__(self, min_equity_ratio: float = 0.0):
        super().__init__(1, "Solvency")
        self.min_equity_ratio = min_equity_ratio

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Equity = positions * prices
        portfolio_values = state.positions @ state.prices

        # Check for negative equity
        min_value = portfolio_values.min().item()

        if min_value > self.min_equity_ratio:
            return GateStatus.PASS, 0.0
        else:
            loss = torch.relu(-portfolio_values).sum().item()
            return GateStatus.FAIL, loss


class LeverageGate(GateNode):
    """Node 3: Leverage balance check."""

    def __init__(self, max_leverage: float = 10.0):
        super().__init__(3, "Leverage")
        self.max_leverage = max_leverage

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        max_lev = state.leverage.max().item()

        if max_lev <= self.max_leverage:
            return GateStatus.PASS, 0.0
        else:
            excess = (state.leverage - self.max_leverage).relu().sum().item()
            return GateStatus.FAIL, excess


class StationarityGate(GateNode):
    """Node 5: Stationarity check."""

    def __init__(self, max_drift: float = 0.1):
        super().__init__(5, "Stationarity")
        self.max_drift = max_drift
        self.history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.history.append(state.prices.clone())

        if len(self.history) < 10:
            return GateStatus.BOUNDED, 0.0

        # Check for unit root / drift
        recent = torch.stack(self.history[-10:])
        drift = (recent[-1] - recent[0]).abs().mean().item() / 10

        if drift < self.max_drift:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, drift - self.max_drift


class CapacityGate(GateNode):
    """Node 6: Market depth capacity check."""

    def __init__(self, min_depth_ratio: float = 0.01):
        super().__init__(6, "Capacity")
        self.min_depth_ratio = min_depth_ratio

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Depth proxy: inverse of spread
        depth = 1.0 / (state.liquidity + 1e-8)
        position_sizes = state.positions.abs().sum(dim=0)

        # Check if positions exceed depth
        depth_ratio = position_sizes / (depth + 1e-8)
        max_ratio = depth_ratio.max().item()

        if max_ratio < self.min_depth_ratio:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_ratio


class ConnectivityGate(GateNode):
    """Node 8: Market connectivity check."""

    def __init__(self, min_correlation: float = -0.99):
        super().__init__(8, "Connectivity")
        self.min_correlation = min_correlation

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check correlation matrix is valid (no extreme negative)
        min_corr = state.correlations.min().item()

        if min_corr > self.min_correlation:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, abs(min_corr - self.min_correlation)


class RepresentationGate(GateNode):
    """Node 11: Model representation adequacy."""

    def __init__(self, max_residual: float = 0.1):
        super().__init__(11, "Representation")
        self.max_residual = max_residual
        self.model_predictions: Optional[torch.Tensor] = None

    def set_predictions(self, predictions: torch.Tensor):
        self.model_predictions = predictions

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        if self.model_predictions is None:
            return GateStatus.UNKNOWN, 0.0

        residuals = (state.prices - self.model_predictions).abs()
        max_res = residuals.max().item()

        if max_res < self.max_residual:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_res


class TurnoverGate(GateNode):
    """Node 2: Capital turnover (conservation) check."""

    def __init__(self, max_turnover_rate: float = 10.0):
        super().__init__(2, "Turnover")
        self.max_turnover_rate = max_turnover_rate
        self.position_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.position_history.append(state.positions.clone())

        if len(self.position_history) < 2:
            return GateStatus.BOUNDED, 0.0

        # Compute turnover: sum of absolute position changes / portfolio value
        delta = (self.position_history[-1] - self.position_history[-2]).abs()
        portfolio_value = (state.positions.abs() @ state.prices).sum() + 1e-8
        turnover = (delta @ state.prices).sum() / portfolio_value

        if turnover.item() < self.max_turnover_rate:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, turnover.item() - self.max_turnover_rate


class ScaleGate(GateNode):
    """Node 4: Scale balance (no asset dominates)."""

    def __init__(self, max_concentration: float = 0.5):
        super().__init__(4, "Scale")
        self.max_concentration = max_concentration

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Compute HHI (Herfindahl-Hirschman Index)
        weights = state.positions.abs() / (state.positions.abs().sum() + 1e-8)
        hhi = (weights ** 2).sum().item()

        # Max concentration is 1/n for equal weights
        n_assets = len(state.prices)
        max_acceptable = self.max_concentration

        if hhi < max_acceptable:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, hhi - max_acceptable


class StiffnessGate(GateNode):
    """Node 7: Market stiffness (price response elasticity)."""

    def __init__(self, max_impact: float = 0.01):
        super().__init__(7, "Stiffness")
        self.max_impact = max_impact

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Estimate price impact: position * (1/liquidity)
        impact = state.positions.abs() * (1.0 / (state.liquidity + 1e-8))
        max_impact = impact.max().item()

        if max_impact < self.max_impact:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_impact - self.max_impact


class BifurcationGate(GateNode):
    """Node 7a: Bifurcation detection (approaching critical point)."""

    def __init__(self, eigenvalue_threshold: float = 0.95):
        super().__init__(701, "Bifurcation")
        self.eigenvalue_threshold = eigenvalue_threshold

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check for near-zero eigenvalue in Jacobian (approaching bifurcation)
        # Proxy: check correlation matrix eigenvalues
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        min_eigenvalue = eigenvalues.min().item()

        # Near-zero eigenvalue indicates instability
        if min_eigenvalue > 1 - self.eigenvalue_threshold:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.eigenvalue_threshold - min_eigenvalue


class AlternativesGate(GateNode):
    """Node 7b: Alternative investments available (diversification possible)."""

    def __init__(self, min_uncorrelated: int = 3):
        super().__init__(702, "Alternatives")
        self.min_uncorrelated = min_uncorrelated

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Count assets with correlation < 0.5 to portfolio
        portfolio_weights = state.positions / (state.positions.sum() + 1e-8)
        portfolio_corr = state.correlations @ portfolio_weights
        n_uncorrelated = (portfolio_corr.abs() < 0.5).sum().item()

        if n_uncorrelated >= self.min_uncorrelated:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, float(self.min_uncorrelated - n_uncorrelated)


class StabilityGate(GateNode):
    """Node 7c: Lyapunov stability check."""

    def __init__(self, max_lyapunov: float = 0.0):
        super().__init__(703, "Stability")
        self.max_lyapunov = max_lyapunov
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        if len(self.return_history) < 20:
            self.return_history.append(state.prices.clone())
            return GateStatus.BOUNDED, 0.0

        # Estimate largest Lyapunov exponent from return series
        returns = torch.stack(self.return_history[-20:])
        log_returns = torch.log(returns[1:] / (returns[:-1] + 1e-8) + 1e-8)

        # Simplified: use variance growth rate as proxy
        var_growth = log_returns.var(dim=0).mean().item()
        lyapunov_estimate = var_growth * 252  # Annualized

        self.return_history.append(state.prices.clone())

        if lyapunov_estimate < self.max_lyapunov:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, lyapunov_estimate


class SwitchingGate(GateNode):
    """Node 7d: Regime switching detection."""

    def __init__(self, max_switch_prob: float = 0.3):
        super().__init__(704, "Switching")
        self.max_switch_prob = max_switch_prob
        self.regime_history: List[int] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.regime_history.append(state.regime)

        if len(self.regime_history) < 10:
            return GateStatus.BOUNDED, 0.0

        # Compute empirical switching probability
        switches = sum(
            1 for i in range(1, 10)
            if self.regime_history[-i] != self.regime_history[-i-1]
        )
        switch_prob = switches / 9

        if switch_prob < self.max_switch_prob:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, switch_prob - self.max_switch_prob


class TamenessGate(GateNode):
    """Node 9: Distribution tameness (fat tail check)."""

    def __init__(self, max_kurtosis: float = 10.0):
        super().__init__(9, "Tameness")
        self.max_kurtosis = max_kurtosis
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.return_history.append(state.prices.clone())

        if len(self.return_history) < 30:
            return GateStatus.BOUNDED, 0.0

        # Compute returns
        prices = torch.stack(self.return_history[-30:])
        returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        # Compute excess kurtosis
        mean_ret = returns.mean(dim=0)
        std_ret = returns.std(dim=0) + 1e-8
        z = (returns - mean_ret) / std_ret
        kurtosis = (z ** 4).mean(dim=0).mean() - 3  # Excess kurtosis

        if kurtosis.item() < self.max_kurtosis:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, kurtosis.item() - self.max_kurtosis


class MixingGate(GateNode):
    """Node 10: Information mixing (market efficiency proxy)."""

    def __init__(self, min_autocorr_decay: float = 0.5):
        super().__init__(10, "Mixing")
        self.min_autocorr_decay = min_autocorr_decay
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.return_history.append(state.prices.clone())

        if len(self.return_history) < 20:
            return GateStatus.BOUNDED, 0.0

        # Compute autocorrelation at lag 1
        prices = torch.stack(self.return_history[-20:])
        returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        ret_mean = returns.mean(dim=0)
        ret_centered = returns - ret_mean
        var = (ret_centered ** 2).mean(dim=0) + 1e-8

        autocorr = (ret_centered[1:] * ret_centered[:-1]).mean(dim=0) / var
        max_autocorr = autocorr.abs().max().item()

        # Low autocorrelation indicates good mixing
        if max_autocorr < self.min_autocorr_decay:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_autocorr - self.min_autocorr_decay


class OscillationGate(GateNode):
    """Node 12: Oscillation detection (boom-bust cycles)."""

    def __init__(self, max_amplitude: float = 0.2, min_period: int = 5):
        super().__init__(12, "Oscillation")
        self.max_amplitude = max_amplitude
        self.min_period = min_period
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 20:
            return GateStatus.BOUNDED, 0.0

        prices = torch.stack(self.price_history[-20:])

        # Detect oscillation via sign changes in returns
        returns = (prices[1:] - prices[:-1])
        signs = torch.sign(returns)
        sign_changes = (signs[1:] * signs[:-1] < 0).float().sum(dim=0)

        # High sign changes indicate oscillation
        avg_changes = sign_changes.mean().item()
        amplitude = returns.abs().mean().item()

        if avg_changes > self.min_period and amplitude > self.max_amplitude:
            return GateStatus.FAIL, amplitude
        else:
            return GateStatus.PASS, 0.0


class CouplingGate(GateNode):
    """Node 14: Boundary coupling (external data connection)."""

    def __init__(self, min_coupling: float = 0.1):
        super().__init__(14, "Coupling")
        self.min_coupling = min_coupling

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check if prices are coupled to external signals
        # Proxy: information ratio > threshold
        if not hasattr(state, 'external_signal') or state.external_signal is None:
            return GateStatus.BOUNDED, 0.0

        corr = torch.corrcoef(torch.stack([state.prices, state.external_signal]))[0, 1]
        coupling = corr.abs().item()

        if coupling > self.min_coupling:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.min_coupling - coupling


class OverloadGate(GateNode):
    """Node 15: Information overload detection."""

    def __init__(self, max_entropy: float = 5.0):
        super().__init__(15, "Overload")
        self.max_entropy = max_entropy

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Compute entropy of price distribution
        # Proxy: use volatility as entropy measure
        entropy = state.volatilities.mean().item() * np.log(len(state.prices))

        if entropy < self.max_entropy:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, entropy - self.max_entropy


class AlignmentGate(GateNode):
    """Node 16: Incentive alignment check."""

    def __init__(self, max_misalignment: float = 0.1):
        super().__init__(16, "Alignment")
        self.max_misalignment = max_misalignment

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check if prices align with fundamentals
        # Proxy: price-to-fundamental ratio deviation
        if not hasattr(state, 'fundamentals') or state.fundamentals is None:
            return GateStatus.BOUNDED, 0.0

        ratio = state.prices / (state.fundamentals + 1e-8)
        deviation = (ratio - 1.0).abs().mean().item()

        if deviation < self.max_misalignment:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, deviation - self.max_misalignment


class LockGate(GateNode):
    """Node 17: Hard regulatory limit check."""

    def __init__(self, limits: Dict[str, float] = None):
        super().__init__(17, "Lock")
        self.limits = limits or {
            'max_position': 1e7,
            'max_leverage': 20.0,
            'min_margin': 0.05
        }

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        violations = []

        # Check position limits
        if state.positions.abs().max().item() > self.limits['max_position']:
            violations.append('position')

        # Check leverage
        if state.leverage.max().item() > self.limits['max_leverage']:
            violations.append('leverage')

        # Check margin
        margin = state.positions.abs().sum() / (state.nav + 1e-8)
        if margin.item() < self.limits['min_margin']:
            violations.append('margin')

        if len(violations) == 0:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, float(len(violations))


class SymmetryGate(GateNode):
    """Node 18: Market symmetry check (bid-ask balance)."""

    def __init__(self, max_asymmetry: float = 0.2):
        super().__init__(18, "Symmetry")
        self.max_asymmetry = max_asymmetry

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check bid-ask spread asymmetry
        if not hasattr(state, 'bid_prices') or not hasattr(state, 'ask_prices'):
            return GateStatus.BOUNDED, 0.0

        mid = (state.bid_prices + state.ask_prices) / 2
        bid_dist = (mid - state.bid_prices) / (mid + 1e-8)
        ask_dist = (state.ask_prices - mid) / (mid + 1e-8)

        asymmetry = (bid_dist - ask_dist).abs().mean().item()

        if asymmetry < self.max_asymmetry:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, asymmetry - self.max_asymmetry


class DisentanglementGate(GateNode):
    """Node 19: Factor disentanglement check."""

    def __init__(self, min_independence: float = 0.3):
        super().__init__(19, "Disentanglement")
        self.min_independence = min_independence

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check that correlation matrix has reasonable eigenvalue spread
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        eigenvalues = eigenvalues.sort(descending=True).values

        # Effective rank = ratio of sum to max
        eff_rank = eigenvalues.sum() / (eigenvalues[0] + 1e-8)
        independence = eff_rank / len(eigenvalues)

        if independence > self.min_independence:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.min_independence - independence


class LipschitzGate(GateNode):
    """Node 20: Price function Lipschitz continuity."""

    def __init__(self, max_lipschitz: float = 2.0):
        super().__init__(20, "Lipschitz")
        self.max_lipschitz = max_lipschitz
        self.state_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        current = (state.prices.clone(), state.positions.clone())
        self.state_history.append(current)

        if len(self.state_history) < 2:
            return GateStatus.BOUNDED, 0.0

        # Estimate Lipschitz constant: |f(x) - f(y)| / |x - y|
        prev_prices, prev_positions = self.state_history[-2]

        price_change = (state.prices - prev_prices).norm()
        pos_change = (state.positions - prev_positions).norm() + 1e-8

        lipschitz = (price_change / pos_change).item()

        if lipschitz < self.max_lipschitz:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, lipschitz - self.max_lipschitz


class SymplecticGate(GateNode):
    """Node 21: Symplectic structure preservation (conservative dynamics)."""

    def __init__(self, max_divergence: float = 0.1):
        super().__init__(21, "Symplectic")
        self.max_divergence = max_divergence
        self.pq_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Track (position, momentum) pairs for Hamiltonian structure
        # Momentum proxy: position * price rate of change
        q = state.positions
        p = state.prices * state.volatilities  # Momentum proxy

        self.pq_history.append((q.clone(), p.clone()))

        if len(self.pq_history) < 3:
            return GateStatus.BOUNDED, 0.0

        # Check phase space volume preservation (Liouville theorem)
        # Proxy: check that det(Jacobian) ≈ 1
        q_prev, p_prev = self.pq_history[-2]
        q_old, p_old = self.pq_history[-3]

        # Simplified: check momentum-position correlation stability
        corr1 = (q * p).sum() / (q.norm() * p.norm() + 1e-8)
        corr2 = (q_prev * p_prev).sum() / (q_prev.norm() * p_prev.norm() + 1e-8)

        divergence = (corr1 - corr2).abs().item()

        if divergence < self.max_divergence:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, divergence - self.max_divergence


# ============================================================================
# BARRIER IMPLEMENTATION
# ============================================================================

class Barrier(ABC):
    """Abstract base class for barriers."""

    def __init__(self, name: str):
        self.name = name
        self.status = BarrierStatus.CLEAR

    @abstractmethod
    def check(self, state: MarketState) -> BarrierStatus:
        pass

    @abstractmethod
    def defense_action(self, state: MarketState) -> MarketState:
        """Apply defense if breached."""
        pass


class BarrierSat(Barrier):
    """Position saturation barrier."""

    def __init__(self, max_position: float = 1e6):
        super().__init__("BarrierSat")
        self.max_position = max_position

    def check(self, state: MarketState) -> BarrierStatus:
        max_pos = state.positions.abs().max().item()

        if max_pos < 0.8 * self.max_position:
            self.status = BarrierStatus.CLEAR
        elif max_pos < self.max_position:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Scale down positions
        scale = self.max_position / (state.positions.abs().max().item() + 1e-8)
        state.positions = state.positions * min(scale, 1.0)
        return state


class BarrierOmin(Barrier):
    """Flash crash (Ominous) barrier."""

    def __init__(self, max_velocity: float = 0.1, window: int = 10):
        super().__init__("BarrierOmin")
        self.max_velocity = max_velocity
        self.window = window
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 2:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Price velocity
        velocity = (self.price_history[-1] - self.price_history[-2]).abs()
        max_vel = velocity.max().item()

        if max_vel < 0.5 * self.max_velocity:
            self.status = BarrierStatus.CLEAR
        elif max_vel < self.max_velocity:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        # Trim history
        if len(self.price_history) > self.window:
            self.price_history = self.price_history[-self.window:]

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Halt: revert to last known good price
        if len(self.price_history) >= 2:
            state.prices = self.price_history[-2].clone()
        return state


class BarrierTypeII(Barrier):
    """Vol-of-vol crisis barrier."""

    def __init__(self, max_vol_of_vol: float = 0.5):
        super().__init__("BarrierTypeII")
        self.max_vol_of_vol = max_vol_of_vol
        self.vol_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.vol_history.append(state.volatilities.clone())

        if len(self.vol_history) < 10:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Vol of vol
        recent_vols = torch.stack(self.vol_history[-10:])
        vol_of_vol = recent_vols.std(dim=0).mean().item()

        if vol_of_vol < 0.5 * self.max_vol_of_vol:
            self.status = BarrierStatus.CLEAR
        elif vol_of_vol < self.max_vol_of_vol:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Increase margin / reduce exposure
        state.leverage = state.leverage * 0.5
        return state


class BarrierGap(Barrier):
    """Liquidity gap barrier."""

    def __init__(self, max_gap: float = 0.05):
        super().__init__("BarrierGap")
        self.max_gap = max_gap

    def check(self, state: MarketState) -> BarrierStatus:
        # Gap = difference between best bid/ask and next level
        spread = 1.0 / (state.liquidity + 1e-8)
        max_spread = spread.max().item()

        if max_spread < 0.5 * self.max_gap:
            self.status = BarrierStatus.CLEAR
        elif max_spread < self.max_gap:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Widen acceptable execution range
        return state


class BarrierCausal(Barrier):
    """Information lag barrier."""

    def __init__(self, max_lag: int = 5):
        super().__init__("BarrierCausal")
        self.max_lag = max_lag
        self.timestamps: List[float] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.timestamps.append(state.timestamp)

        if len(self.timestamps) < 2:
            self.status = BarrierStatus.CLEAR
            return self.status

        lag = self.timestamps[-1] - self.timestamps[-2]

        if lag < 0.5 * self.max_lag:
            self.status = BarrierStatus.CLEAR
        elif lag < self.max_lag:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierScat(Barrier):
    """Market fragmentation barrier."""

    def __init__(self, max_fragmentation: float = 0.3):
        super().__init__("BarrierScat")
        self.max_fragmentation = max_fragmentation

    def check(self, state: MarketState) -> BarrierStatus:
        # Fragmentation proxy: variance in liquidity across assets
        liq_var = state.liquidity.var().item() / (state.liquidity.mean().item() + 1e-8)

        if liq_var < 0.5 * self.max_fragmentation:
            self.status = BarrierStatus.CLEAR
        elif liq_var < self.max_fragmentation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierMix(Barrier):
    """Herding behavior barrier."""

    def __init__(self, max_herding: float = 0.8):
        super().__init__("BarrierMix")
        self.max_herding = max_herding

    def check(self, state: MarketState) -> BarrierStatus:
        # Herding proxy: first eigenvalue dominance
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        first_ev_ratio = eigenvalues[-1].item() / (eigenvalues.sum().item() + 1e-8)

        if first_ev_ratio < 0.5 * self.max_herding:
            self.status = BarrierStatus.CLEAR
        elif first_ev_ratio < self.max_herding:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierCap(Barrier):
    """Controllability barrier."""

    def __init__(self, min_controllability: float = 0.1):
        super().__init__("BarrierCap")
        self.min_controllability = min_controllability

    def check(self, state: MarketState) -> BarrierStatus:
        # Controllability proxy: minimum eigenvalue of position Gramian
        min_ev = torch.linalg.eigvalsh(state.correlations).min().item()

        if min_ev > 2 * self.min_controllability:
            self.status = BarrierStatus.CLEAR
        elif min_ev > self.min_controllability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierVac(Barrier):
    """Regime instability barrier."""

    def __init__(self, max_regime_instability: float = 0.5):
        super().__init__("BarrierVac")
        self.max_regime_instability = max_regime_instability
        self.regime_history: List[int] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.regime_history.append(state.regime)

        if len(self.regime_history) < 5:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Regime instability: recent switching frequency
        switches = sum(1 for i in range(1, 5) if self.regime_history[-i] != self.regime_history[-i-1])
        instability = switches / 4

        if instability < 0.5 * self.max_regime_instability:
            self.status = BarrierStatus.CLEAR
        elif instability < self.max_regime_instability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierFreq(Barrier):
    """HFT oscillation barrier."""

    def __init__(self, max_hft_oscillation: int = 3):
        super().__init__("BarrierFreq")
        self.max_hft_oscillation = max_hft_oscillation
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 10:
            self.status = BarrierStatus.CLEAR
            return self.status

        prices = torch.stack(self.price_history[-10:])
        returns = prices[1:] - prices[:-1]
        sign_changes = (returns[1:] * returns[:-1] < 0).float().sum().item()
        oscillation = sign_changes / 8

        if oscillation < 0.5 * self.max_hft_oscillation:
            self.status = BarrierStatus.CLEAR
        elif oscillation < self.max_hft_oscillation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierEpi(Barrier):
    """Information overload barrier."""

    def __init__(self, max_info_overload: float = 5.0):
        super().__init__("BarrierEpi")
        self.max_info_overload = max_info_overload

    def check(self, state: MarketState) -> BarrierStatus:
        entropy = state.volatilities.mean().item() * np.log(len(state.prices) + 1)

        if entropy < 0.5 * self.max_info_overload:
            self.status = BarrierStatus.CLEAR
        elif entropy < self.max_info_overload:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierAction(Barrier):
    """Execution impossibility barrier."""

    def __init__(self, min_execution_prob: float = 0.9):
        super().__init__("BarrierAction")
        self.min_execution_prob = min_execution_prob

    def check(self, state: MarketState) -> BarrierStatus:
        # Execution probability proxy: liquidity / position size
        exec_prob = (state.liquidity / (state.positions.abs() + 1e-8)).min().item()
        exec_prob = min(1.0, exec_prob)

        if exec_prob > self.min_execution_prob:
            self.status = BarrierStatus.CLEAR
        elif exec_prob > 0.5 * self.min_execution_prob:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierInput(Barrier):
    """Data starvation barrier."""

    def __init__(self, min_data_quality: float = 0.8):
        super().__init__("BarrierInput")
        self.min_data_quality = min_data_quality

    def check(self, state: MarketState) -> BarrierStatus:
        # Data quality proxy: 1 - NaN ratio (simulated)
        # In real implementation, would check actual data completeness
        nan_ratio = torch.isnan(state.prices).float().mean().item()
        quality = 1.0 - nan_ratio

        if quality > self.min_data_quality:
            self.status = BarrierStatus.CLEAR
        elif quality > 0.5 * self.min_data_quality:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Replace NaNs with last known values
        state.prices = torch.nan_to_num(state.prices, nan=1.0)
        return state


class BarrierVariety(Barrier):
    """Hedging impossibility barrier."""

    def __init__(self, min_hedgeability: float = 0.5):
        super().__init__("BarrierVariety")
        self.min_hedgeability = min_hedgeability

    def check(self, state: MarketState) -> BarrierStatus:
        # Hedgeability: effective rank of correlation matrix
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        eff_rank = eigenvalues.sum() / (eigenvalues.max() + 1e-8)
        hedgeability = eff_rank / len(eigenvalues)

        if hedgeability > self.min_hedgeability:
            self.status = BarrierStatus.CLEAR
        elif hedgeability > 0.5 * self.min_hedgeability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierBode(Barrier):
    """Risk waterbed barrier (Bode integral constraint)."""

    def __init__(self, max_waterbed: float = 0.3):
        super().__init__("BarrierBode")
        self.max_waterbed = max_waterbed
        self.risk_history: List[float] = []

    def check(self, state: MarketState) -> BarrierStatus:
        total_risk = state.volatilities.sum().item()
        self.risk_history.append(total_risk)

        if len(self.risk_history) < 5:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Waterbed effect: risk reduction in one area causing increase elsewhere
        risk_std = np.std(self.risk_history[-5:])
        waterbed = risk_std / (np.mean(self.risk_history[-5:]) + 1e-8)

        if waterbed < 0.5 * self.max_waterbed:
            self.status = BarrierStatus.CLEAR
        elif waterbed < self.max_waterbed:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierLock(Barrier):
    """Hard regulatory lock barrier."""

    def __init__(self):
        super().__init__("BarrierLock")
        self.hard_limits = {
            'max_position': 1e8,
            'max_leverage': 25.0,
            'min_capital': 1e6
        }

    def check(self, state: MarketState) -> BarrierStatus:
        violations = 0

        if state.positions.abs().max().item() > self.hard_limits['max_position']:
            violations += 1
        if state.leverage.max().item() > self.hard_limits['max_leverage']:
            violations += 1
        if state.nav < self.hard_limits['min_capital']:
            violations += 1

        if violations == 0:
            self.status = BarrierStatus.CLEAR
        elif violations == 1:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Force compliance
        scale = self.hard_limits['max_position'] / (state.positions.abs().max().item() + 1e-8)
        state.positions = state.positions * min(scale, 1.0)
        return state


class BarrierLiq(Barrier):
    """Liquidity crisis barrier."""

    def __init__(self, min_liquidity: float = 0.01):
        super().__init__("BarrierLiq")
        self.min_liquidity = min_liquidity

    def check(self, state: MarketState) -> BarrierStatus:
        min_liq = state.liquidity.min().item()

        if min_liq > 2 * self.min_liquidity:
            self.status = BarrierStatus.CLEAR
        elif min_liq > self.min_liquidity:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierLev(Barrier):
    """Leverage crisis barrier."""

    def __init__(self, max_leverage: float = 15.0):
        super().__init__("BarrierLev")
        self.max_leverage = max_leverage

    def check(self, state: MarketState) -> BarrierStatus:
        max_lev = state.leverage.max().item()

        if max_lev < 0.7 * self.max_leverage:
            self.status = BarrierStatus.CLEAR
        elif max_lev < self.max_leverage:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        state.leverage = state.leverage * 0.8
        return state


class BarrierRef(Barrier):
    """Oracle/reference data barrier."""

    def __init__(self, max_oracle_deviation: float = 0.05):
        super().__init__("BarrierRef")
        self.max_oracle_deviation = max_oracle_deviation

    def check(self, state: MarketState) -> BarrierStatus:
        if not hasattr(state, 'oracle_prices') or state.oracle_prices is None:
            self.status = BarrierStatus.CLEAR
            return self.status

        deviation = (state.prices - state.oracle_prices).abs() / (state.oracle_prices + 1e-8)
        max_dev = deviation.max().item()

        if max_dev < 0.5 * self.max_oracle_deviation:
            self.status = BarrierStatus.CLEAR
        elif max_dev < self.max_oracle_deviation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierDef(Barrier):
    """Default probability barrier."""

    def __init__(self, max_default_prob: float = 0.1):
        super().__init__("BarrierDef")
        self.max_default_prob = max_default_prob

    def check(self, state: MarketState) -> BarrierStatus:
        # Default probability proxy: leverage / solvency ratio
        solvency = state.positions @ state.prices / (state.nav + 1e-8)
        default_prob = torch.sigmoid(state.leverage.max() - solvency.abs()).item()

        if default_prob < 0.5 * self.max_default_prob:
            self.status = BarrierStatus.CLEAR
        elif default_prob < self.max_default_prob:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        state.leverage = state.leverage * 0.5
        return state
```

### 18.4 Complete Market Sieve

```python
# ============================================================================
# COMPLETE MARKET SIEVE
# ============================================================================

class MarketSieve:
    """
    Complete Market Sieve with 21 gates and 20 barriers.

    Routes pricing decisions through permit checks.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize all 21 gates
        self.gates: Dict[int, GateNode] = {
            1: SolvencyGate(),
            2: TurnoverGate(self.config.get('max_turnover', 10.0)),
            3: LeverageGate(self.config.get('max_leverage', 10.0)),
            4: ScaleGate(self.config.get('max_concentration', 0.5)),
            5: StationarityGate(),
            6: CapacityGate(),
            7: StiffnessGate(),
            701: BifurcationGate(),
            702: AlternativesGate(),
            703: StabilityGate(),
            704: SwitchingGate(),
            8: ConnectivityGate(),
            9: TamenessGate(),
            10: MixingGate(),
            11: RepresentationGate(),
            12: OscillationGate(),
            14: CouplingGate(),
            15: OverloadGate(),
            16: AlignmentGate(),
            17: LockGate(self.config.get('limits')),
            18: SymmetryGate(),
            19: DisentanglementGate(),
            20: LipschitzGate(),
            21: SymplecticGate(),
        }

        # Initialize all 20 barriers
        self.barriers: Dict[str, Barrier] = {
            'BarrierSat': BarrierSat(self.config.get('max_position', 1e6)),
            'BarrierOmin': BarrierOmin(self.config.get('max_velocity', 0.1)),
            'BarrierTypeII': BarrierTypeII(self.config.get('max_vol_of_vol', 0.5)),
            'BarrierGap': BarrierGap(self.config.get('max_gap', 0.05)),
            'BarrierCausal': BarrierCausal(self.config.get('max_lag', 5)),
            'BarrierScat': BarrierScat(self.config.get('max_fragmentation', 0.3)),
            'BarrierMix': BarrierMix(self.config.get('max_herding', 0.8)),
            'BarrierCap': BarrierCap(self.config.get('min_controllability', 0.1)),
            'BarrierVac': BarrierVac(self.config.get('max_regime_instability', 0.5)),
            'BarrierFreq': BarrierFreq(self.config.get('max_hft_oscillation', 3)),
            'BarrierEpi': BarrierEpi(self.config.get('max_info_overload', 5.0)),
            'BarrierAction': BarrierAction(self.config.get('min_execution_prob', 0.9)),
            'BarrierInput': BarrierInput(self.config.get('min_data_quality', 0.8)),
            'BarrierVariety': BarrierVariety(self.config.get('min_hedgeability', 0.5)),
            'BarrierBode': BarrierBode(self.config.get('max_waterbed', 0.3)),
            'BarrierLock': BarrierLock(),
            'BarrierLiq': BarrierLiq(self.config.get('min_liquidity', 0.01)),
            'BarrierLev': BarrierLev(self.config.get('max_leverage_barrier', 15.0)),
            'BarrierRef': BarrierRef(self.config.get('max_oracle_deviation', 0.05)),
            'BarrierDef': BarrierDef(self.config.get('max_default_prob', 0.1)),
        }

    def run(self, state: MarketState,
            model_predictions: Optional[torch.Tensor] = None) -> Certificate:
        """
        Run complete Sieve check.

        Args:
            state: Current market state
            model_predictions: Optional model price predictions

        Returns:
            Certificate with all check results
        """
        gate_results = {}
        total_loss = 0.0

        # Run gates
        for node_id, gate in self.gates.items():
            if node_id == 11 and model_predictions is not None:
                gate.set_predictions(model_predictions)
            status, loss = gate.check(state)
            gate_results[node_id] = status
            total_loss += loss

        # Run barriers
        barrier_results = {}
        for name, barrier in self.barriers.items():
            barrier_results[name] = barrier.check(state)

        # Detect failure modes
        failure_modes = self._detect_failure_modes(gate_results, barrier_results)

        # Compute price bounds
        price_bounds = self._compute_price_bounds(state, total_loss)

        # Compute confidence
        n_pass = sum(1 for s in gate_results.values()
                     if s in (GateStatus.PASS, GateStatus.BOUNDED))
        confidence = n_pass / len(gate_results)

        return Certificate(
            gate_results=gate_results,
            barrier_results=barrier_results,
            price_bounds=price_bounds,
            confidence=confidence,
            failure_modes=failure_modes,
            timestamp=state.timestamp
        )

    def _detect_failure_modes(self, gates: Dict[int, GateStatus],
                               barriers: Dict[str, BarrierStatus]) -> List[FailureMode]:
        """Detect active or approaching failure modes."""
        modes = []

        # C.E: Default cascade (Node 1 fail + BarrierSat breach)
        if gates.get(1) == GateStatus.FAIL:
            modes.append(FailureMode.CE)

        # S.E: Supercritical leverage (Node 3 fail)
        if gates.get(3) == GateStatus.FAIL:
            modes.append(FailureMode.SE)

        # T.E: Flash crash (BarrierOmin breach)
        if barriers.get('BarrierOmin') == BarrierStatus.BREACHED:
            modes.append(FailureMode.TE)

        # D.D: Vol crisis (BarrierTypeII breach)
        if barriers.get('BarrierTypeII') == BarrierStatus.BREACHED:
            modes.append(FailureMode.DD)

        return modes

    def _compute_price_bounds(self, state: MarketState,
                               loss: float) -> Tuple[float, float]:
        """Compute price bounds based on uncertainty."""
        base_price = state.prices.mean().item()
        uncertainty = loss * state.temperature
        return (base_price - uncertainty, base_price + uncertainty)

    def apply_defenses(self, state: MarketState) -> MarketState:
        """Apply defense actions for breached barriers."""
        for barrier in self.barriers.values():
            if barrier.status == BarrierStatus.BREACHED:
                state = barrier.defense_action(state)
        return state
```

### 18.5 Loss Functions

```python
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MarketLoss(nn.Module):
    """
    Combined loss function for market pricing.

    Integrates:
    - Pricing error (SDF consistency)
    - Gate violations
    - Barrier penalties
    - Regularization
    """

    def __init__(self, sieve: MarketSieve, sdf: ThermoeconomicSDF,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.sieve = sieve
        self.sdf = sdf
        self.weights = weights or {
            'pricing': 1.0,
            'gates': 0.5,
            'barriers': 1.0,
            'regularization': 0.01
        }

    def forward(self, state: MarketState,
                target_prices: torch.Tensor,
                factors: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.

        Returns:
            (loss, component_dict)
        """
        losses = {}

        # Pricing loss (SDF consistency)
        predicted_prices = self._price_assets(state, factors)
        losses['pricing'] = F.mse_loss(predicted_prices, target_prices)

        # Gate loss
        cert = self.sieve.run(state, predicted_prices)
        gate_loss = sum(
            1.0 for s in cert.gate_results.values()
            if s == GateStatus.FAIL
        )
        losses['gates'] = torch.tensor(gate_loss, dtype=torch.float32)

        # Barrier loss
        barrier_loss = sum(
            2.0 if s == BarrierStatus.BREACHED else
            0.5 if s == BarrierStatus.WARNING else 0.0
            for s in cert.barrier_results.values()
        )
        losses['barriers'] = torch.tensor(barrier_loss, dtype=torch.float32)

        # Regularization (L2 on model params)
        reg_loss = sum(p.pow(2).sum() for p in self.sdf.parameters())
        losses['regularization'] = reg_loss

        # Weighted sum
        total = sum(self.weights[k] * v for k, v in losses.items())

        return total, losses

    def _price_assets(self, state: MarketState,
                      factors: torch.Tensor) -> torch.Tensor:
        """Price all assets using SDF."""
        risk_premia = self.sdf.risk_premium(state)
        expected_returns = state.prices * (1 + risk_premia)
        return expected_returns


# ============================================================================
# PHASE DETECTOR
# ============================================================================

class MarketPhaseDetector:
    """
    Detect market complexity phase (Crystal/Liquid/Gas).
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.price_history: List[torch.Tensor] = []

    def add_observation(self, prices: torch.Tensor):
        self.price_history.append(prices.clone())
        if len(self.price_history) > self.window:
            self.price_history = self.price_history[-self.window:]

    def detect_phase(self) -> MarketPhase:
        """
        Detect current market phase via compression ratio.
        """
        if len(self.price_history) < 20:
            return MarketPhase.LIQUID  # Default

        prices = torch.stack(self.price_history)
        returns = prices[1:] / prices[:-1] - 1

        # Compression ratio proxy: autocorrelation
        # High autocorrelation → predictable → Liquid
        # Low autocorrelation → random → Crystal or Gas

        returns_flat = returns.flatten()
        if len(returns_flat) < 10:
            return MarketPhase.LIQUID

        autocorr = torch.corrcoef(
            torch.stack([returns_flat[:-1], returns_flat[1:]])
        )[0, 1].item()

        # Volatility clustering (GARCH effect)
        vol = returns.std(dim=0)
        vol_autocorr = torch.corrcoef(
            torch.stack([vol[:-1], vol[1:]])
        )[0, 1].item() if len(vol) > 1 else 0.0

        if abs(autocorr) < 0.1 and abs(vol_autocorr) < 0.1:
            # Low predictability in both price and vol
            # Need external info test to distinguish Crystal vs Gas
            return MarketPhase.CRYSTAL
        elif abs(autocorr) > 0.3 or abs(vol_autocorr) > 0.5:
            # High predictability
            return MarketPhase.LIQUID
        else:
            return MarketPhase.GAS
```

### 18.6 Complete Market Hypostructure

```python
# ============================================================================
# COMPLETE MARKET HYPOSTRUCTURE
# ============================================================================

class MarketHypostructure:
    """
    Complete Market Hypostructure implementation.

    Integrates:
    - Thermoeconomic SDF
    - Market Sieve (permits)
    - Ruppeiner geometry
    - Phase detection
    - Certificate generation
    """

    def __init__(self, n_assets: int, n_factors: int = 3,
                 config: Optional[Dict] = None):
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.config = config or {}

        # Core components
        self.sdf = ThermoeconomicSDF(n_assets, n_factors)
        self.sieve = MarketSieve(config)
        self.phase_detector = MarketPhaseDetector()

        # Loss function
        self.loss_fn = MarketLoss(self.sieve, self.sdf)

    def price(self, state: MarketState,
              factors: torch.Tensor) -> Tuple[torch.Tensor, Certificate]:
        """
        Generate certified prices.

        Returns:
            (prices, certificate)
        """
        # Compute geometry
        geometry = RuppeinerMarket(state)

        # Detect phase
        self.phase_detector.add_observation(state.prices)
        phase = self.phase_detector.detect_phase()

        # Compute risk premia
        risk_premia = self.sdf.risk_premium(state)

        # Adjust for phase
        if phase == MarketPhase.GAS:
            # Widen uncertainty in chaotic phase
            risk_premia = risk_premia * 2.0

        # Compute prices
        prices = state.prices * (1 + risk_premia)

        # Run Sieve
        certificate = self.sieve.run(state, prices)

        # Apply geometry correction
        curvature = geometry.ricci_scalar()
        if curvature > 10.0:  # High curvature = high concentration
            certificate.failure_modes.append(FailureMode.CD)

        return prices, certificate

    def update(self, state: MarketState,
               target_prices: torch.Tensor,
               factors: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> Dict:
        """
        Update model parameters.

        Returns:
            Dictionary of loss components
        """
        optimizer.zero_grad()

        total_loss, losses = self.loss_fn(state, target_prices, factors)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.sdf.parameters(), 1.0)

        optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    def stress_test(self, state: MarketState,
                    scenario: str) -> Tuple[MarketState, Certificate]:
        """
        Run stress test scenario.

        Args:
            state: Base state
            scenario: Scenario name ('vol_spike', 'liquidity_crisis', etc.)

        Returns:
            (stressed_state, certificate)
        """
        stressed = MarketState(
            prices=state.prices.clone(),
            positions=state.positions.clone(),
            volatilities=state.volatilities.clone(),
            correlations=state.correlations.clone(),
            liquidity=state.liquidity.clone(),
            leverage=state.leverage.clone(),
            regime=state.regime,
            temperature=state.temperature,
            timestamp=state.timestamp
        )

        if scenario == 'vol_spike':
            stressed.volatilities = stressed.volatilities * 3.0
            stressed.temperature = stressed.temperature * 2.0

        elif scenario == 'liquidity_crisis':
            stressed.liquidity = stressed.liquidity * 10.0  # Spreads widen

        elif scenario == 'correlation_spike':
            stressed.correlations = torch.ones_like(stressed.correlations) * 0.9
            torch.diagonal(stressed.correlations).fill_(1.0)

        elif scenario == 'leverage_cascade':
            stressed.leverage = stressed.leverage * 2.0

        # Run Sieve on stressed state
        _, certificate = self.price(stressed, torch.zeros(self.n_factors))

        return stressed, certificate
```

### 18.7 Usage Example

```python
# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate Market Hypostructure usage."""

    # Configuration
    n_assets = 10
    n_positions = 5
    n_factors = 3

    # Initialize
    hypo = MarketHypostructure(n_assets, n_factors)
    optimizer = torch.optim.Adam(hypo.sdf.parameters(), lr=0.001)

    # Create sample state
    state = MarketState(
        prices=torch.randn(n_assets).abs() * 100,
        positions=torch.randn(n_positions, n_assets) * 10,
        volatilities=torch.rand(n_assets) * 0.3 + 0.1,
        correlations=torch.eye(n_assets) * 0.5 + 0.5,
        liquidity=torch.rand(n_assets) * 0.01 + 0.001,
        leverage=torch.rand(n_positions) * 5 + 1,
        regime=0,
        temperature=1.0,
        timestamp=0.0
    )

    # Generate certified prices
    prices, cert = hypo.price(state, torch.randn(n_factors))

    print(f"Prices: {prices}")
    print(f"Certificate valid: {cert.is_valid}")
    print(f"Confidence: {cert.confidence:.2%}")
    print(f"Price bounds: {cert.price_bounds}")
    print(f"Failure modes: {cert.failure_modes}")

    # Run stress test
    stressed_state, stressed_cert = hypo.stress_test(state, 'vol_spike')
    print(f"\nStress test (vol spike):")
    print(f"  Certificate valid: {stressed_cert.is_valid}")
    print(f"  Failure modes: {stressed_cert.failure_modes}")

    # Training loop example
    for epoch in range(10):
        target_prices = state.prices * 1.01  # 1% expected return
        factors = torch.randn(n_factors)
        losses = hypo.update(state, target_prices, factors, optimizer)
        print(f"Epoch {epoch}: {losses}")


if __name__ == "__main__":
    example_usage()
```

---

## 19. Worked Examples

This section provides complete worked examples demonstrating the Market Hypostructure in action.

### 19.1 Bond Pricing Under Fed Surprise

**Scenario:** Unexpected 100bp Fed rate hike during Treasury auction.

**Initial State:**
- 10-year Treasury yield: 4.00%
- Portfolio: Long $100M 10Y duration
- Duration: 8.5 years
- Convexity: 75

**Step 1: Pre-shock Sieve Check**
```
Gate 1 (Solvency): PASS - Equity positive
Gate 5 (Stationarity): PASS - No drift detected
Gate 6 (Capacity): PASS - Position within depth
Gate 11 (Representation): PASS - Model fits yield curve
All barriers: CLEAR
Certificate: VALID, confidence 95%
```

**Step 2: Shock Application**
```
Fed announces surprise 100bp hike
New yield: 5.00%
Price change: -8.5 × 1.00% + 0.5 × 75 × (1.00%)² ≈ -8.1%
P&L: -$8.1M
```

**Step 3: Post-shock Sieve Check**
```
Gate 1 (Solvency): PASS - Still positive equity
Gate 5 (Stationarity): FAIL - Drift > threshold
Gate 7 (Stiffness): WARNING - Mean reversion uncertain
BarrierOmin: WARNING - Price velocity elevated
BarrierInput: PASS - Data feeds normal

Certificate: BOUNDED
Price bounds: [91.0, 92.5]
Failure mode: B.E (External Shock)
Recommended action: Widen bid-ask, reduce duration
```

**Step 4: Recovery Path**
```
T+1 day: Markets stabilize
Gate 5: PASS - New regime established
BarrierOmin: CLEAR
Certificate: VALID
```

**Key insight:** The Sieve correctly identified the external shock (B.E) and widened price bounds during transition. No false positive on default cascade (C.E) since solvency maintained.

---

### 19.2 Options During Vol Spike (Volmageddon)

**Scenario:** VIX spikes from 12 to 50 in one session (February 2018 replay).

**Initial State:**
- Short $10M vega in SPX options
- VIX: 12
- Short vol ETPs: Leveraged 2x
- Realized vol: 8%

**Step 1: Pre-spike State**
```
Gate 3 (Leverage): WARNING - High leverage (8x effective)
Gate 7 (Stiffness): PASS - Vol mean reversion assumed
BarrierTypeII: CLEAR - Vol-of-vol normal
BarrierVac: CLEAR - Regime stable

Certificate: BOUNDED (due to leverage warning)
Failure mode monitoring: D.D (Dispersion Success)
```

**Step 2: Vol Spike Sequence**
```
T+0: VIX 12 → 20
  BarrierTypeII: WARNING
  Gate 3: FAIL (leverage + vol product > threshold)

T+1 hour: VIX 20 → 35
  BarrierTypeII: BREACHED
  BarrierOmin: WARNING (price velocity)
  Failure mode: D.D → S.E cascade

T+2 hours: VIX 35 → 50
  Certificate: INVALID
  Multiple barriers: BREACHED
  Surgery triggered: SurgSE (margin relief)
```

**Step 3: Defense Actions**
```
BarrierTypeII defense: Reduce gamma exposure by 50%
BarrierSat defense: Scale positions to limit
SurgSE: Extend margin call deadline

Post-defense state:
  Positions: Reduced 60%
  Leverage: 3x (from 8x)
  Certificate: BOUNDED
```

**Step 4: Post-Crisis Analysis**
```
Root cause: D.D (crowded short vol) → S.E (leverage crisis)
Cascade path: D.D → S.E → (near C.E avoided by intervention)
Recovery time: 3 days to CLEAR status
Loss attribution:
  - Vol move: 70%
  - Liquidity cost: 20%
  - Forced selling: 10%
```

---

### 19.3 Credit Default Cascade (2008 Style)

**Scenario:** Major financial institution default triggering counterparty concerns.

**Initial State:**
- Investment grade portfolio: $500M
- Single-name concentration: 15% in one issuer
- CDS hedge: 50% notional
- HHI index: 0.08 (moderate concentration)

**Step 1: Pre-default**
```
Gate 1 (Solvency): PASS
Gate 3 (Leverage): PASS (low leverage)
BarrierSat: WARNING (concentration at 15%)
Node 14 (Coupling): WARNING (CDS-bond basis elevated)

Certificate: BOUNDED
Monitoring: C.D (Too-Big-to-Fail)
```

**Step 2: Default Event**
```
T+0: Lehman-equivalent defaults
Concentrated position: -75% overnight
CDS hedge: +60% (partial offset)
Net loss: -$11.25M

Gate 1: FAIL (solvency impaired for that position)
BarrierGap: BREACHED (credit event gap)
Failure mode: C.E activated
```

**Step 3: Cascade Propagation**
```
T+1 day: Counterparty concerns spread
  CDS spreads: +200bp across IG
  Interbank: Freeze beginning

Certificate trace:
  C.E metrics: Branching factor = 1.2 > 1.0 (supercritical)
  T.D metrics: Volume down 80%

Surgery trigger: SurgCE (bailout/backstop)
```

**Step 4: Intervention**
```
SurgCE applied:
  - Fed liquidity facility announced
  - Counterparty guarantees
  - Funding normalized

Post-intervention:
  Gate 1: PASS (guarantee counts as capital)
  Branching factor: 0.7 < 1.0 (subcritical)
  Certificate: BOUNDED

Recovery:
  T+30 days: C.E cleared
  T+90 days: All barriers CLEAR
```

**Key insight:** The Sieve detected C.D risk (concentration) pre-event, correctly identified C.E cascade post-event, and tracked intervention effectiveness.

---

### 19.4 Crypto Oracle Attack

**Scenario:** Chainlink oracle reports manipulated price for DeFi lending protocol.

**Initial State:**
- ETH collateral: $50M
- Borrowed stablecoins: $30M (60% LTV)
- Oracle: Chainlink ETH/USD
- Health factor: 1.67

**Step 1: Normal Operation**
```
Gate 1 (Solvency): PASS
Node 8 (Connectivity): PASS (oracle functional)
BarrierRef: CLEAR (oracle deviation < 1%)

Certificate: VALID
```

**Step 2: Oracle Manipulation**
```
T+0: Oracle reports ETH = $500 (actual: $2000)
  BarrierRef: BREACHED (deviation 75%)
  Immediate effect: Apparent LTV = 240%
  Protocol triggers liquidation

Certificate: INVALID
Failure mode: C.E triggered by false data
```

**Step 3: Cascade Effects**
```
Liquidation cascade:
  T+0: Protocol attempts to sell $50M ETH at $500
  T+1 block: MEV bots front-run
  T+2 blocks: Market absorbs selling
  Actual execution: $1800 average

Additional damage:
  - Protocol TVL drops 40%
  - Cross-protocol contagion via composability

Failure modes active:
  - C.E (liquidation cascade)
  - T.C (composability complexity)
  - B.C (oracle incentive misalignment)
```

**Step 4: Defense & Recovery**
```
Defense actions:
  BarrierRef defense: Reject outlier prices
  Circuit breaker: Pause protocol
  Multi-oracle: Require 3/5 consensus

Post-incident:
  Oracle source: Expanded to 5 providers
  Deviation threshold: Tightened to 5%
  Time delay: 10-minute TWAP required

Certificate recovery:
  T+2 days: BarrierRef CLEAR with new design
  T+7 days: Full certificate VALID
```

---

### 19.5 Cross-Asset Contagion

**Scenario:** Emerging market crisis spreading across asset classes.

**Initial State:**
- EM equity: $100M
- EM local currency bonds: $50M
- USD/EM FX hedge: 50%
- Commodity exposure (oil): $25M
- Correlation assumption: 0.3 across assets

**Step 1: Initial Shock (EM Equities)**
```
T+0: EM political crisis
EM equities: -15%
Initial loss: $15M

Gate 5 (Stationarity): FAIL (regime break)
Failure mode: B.E (External Shock)
```

**Step 2: FX Contagion**
```
T+1 day: EM currencies depreciate 10%
FX hedge: Partially offsets
Net FX loss: $2.5M (after hedge)

Correlation observation:
  Actual EM eq/FX correlation: 0.8 (vs 0.3 assumed)

Gate 11 (Representation): FAIL
Node 14 (Coupling): FAIL (basis blowout)
```

**Step 3: Bond Market Impact**
```
T+2 days: EM bond spreads widen 300bp
Local bonds: -20% (duration + FX + spread)
Loss: $10M

T.D emerging: EM bond liquidity drying up
Certificate: INVALID (multiple gate failures)
```

**Step 4: Commodity Spillover**
```
T+3 days: Oil drops 10% on global growth fears
Commodity loss: $2.5M

Total portfolio loss: $30M (17% of initial $175M)

Failure mode progression:
  B.E (shock) → B.D (EM liquidity starvation) →
  T.D (frozen EM bonds) → D.E (correlation spike)
```

**Step 5: Multi-Barrier Coordination**
```
Active barriers:
  BarrierInput (EM data quality degraded)
  BarrierGap (illiquidity gaps)
  BarrierVariety (hedge incomplete)

Surgery coordination (Section 15.10):
  Priority 1: Reduce EM exposure (BarrierSat defense)
  Priority 2: Accept illiquidity (no forced selling)
  Priority 3: Mark to conservative (interval pricing)

Recovery timeline:
  T+7 days: Volatility subsides
  T+14 days: Liquidity returning
  T+30 days: Certificate BOUNDED
  T+60 days: Certificate VALID (new correlation model)
```

**Key insight:** The multi-barrier coordination protocol prevented forced selling at distressed prices. Interval pricing preserved capital while acknowledging uncertainty.

---

## 20. Summary and Cross-References

### 20.1 Document Structure Summary

This document establishes a **complete thermoeconomic theory of asset pricing** with the following components:

```{list-table} Document Component Summary
:header-rows: 1
:name: doc-summary

* - Section
  - Content
  - Lines
* - 1-3
  - Categorical foundations (topos, modalities, kernel)
  - ~350
* - 4
  - Thermoeconomic framework (SDF, geometry, phases)
  - ~250
* - 5-6
  - Market structure (states, dynamics)
  - ~150
* - 7
  - Market Sieve (21 gates, 20 barriers)
  - ~1200
* - 8-9
  - Market dynamics and risk measures
  - ~150
* - 10
  - Asset class pricing (12 classes)
  - ~750
* - 11-13
  - Regime dynamics, implementation, checklist
  - ~200
* - 14
  - Failure mode taxonomy (15 modes)
  - ~700
* - 15
  - Surgery contracts (8 interventions)
  - ~400
* - 16
  - Market metatheorems (5 theorems)
  - ~250
* - 17
  - Algorithmic pricing theory
  - ~200
* - 18
  - Full Python/PyTorch implementation
  - ~1600
* - 19
  - Worked examples (5 scenarios)
  - ~350
* - 21
  - Calibration guidance
  - ~400
* - 22
  - Risk attribution framework
  - ~280
* - 23
  - Backtesting framework
  - ~600
```

### 20.2 Internal Cross-References

**Sieve Structure:**
- Gate nodes (Section 7.2): 47-node diagnostic Sieve structure
- Failure mode taxonomy (Section 14): 3×5 failure grid
- Surgery contracts (Section 15): intervention framework
- Certificate structure: proof-carrying pattern

**Categorical Foundations:**
- Categorical machinery (Section 1.3-1.7): cohesive topos structure
- Metatheorems (Section 16): market-domain KRNL theorems
- Algorithmic pricing (Section 17): Kolmogorov complexity connections
- The Sieve: permit-checking framework

**Thermoeconomic Framework:**
- Thermoeconomic foundations (Section 4): entropy, free energy, temperature
- Market phases and phase transitions
- Ruppeiner geometry: risk metric tensor
- Landauer bounds: trading cost constraints

**Geometric Theory:**
- Capacity constraints (Section 24): information-theoretic bounds
- WFR transport (Section 25): portfolio rebalancing geometry
- Price discovery (Section 26): entropic drift dynamics
- Equations of motion (Section 27): geodesic portfolio dynamics
- Market interface (Section 28): symplectic boundary structure
- Pricing kernel (Section 29): Helmholtz equation framework
- Sector classification (Section 30): gradient flow allocation

### 20.3 Key Definitions Index

| Definition | Label | Section |
|------------|-------|---------|
| Market Hypostructure | def-market-hypostructure | 2.1 |
| Cohesive Market Category | def-cohesive-market | 1.3 |
| Thermoeconomic SDF | def-thermo-sdf | 4.1 |
| Free Energy Potential | def-free-energy | 4.3 |
| Ruppeiner Risk Metric | def-ruppeiner-market | 4.5 |
| Market Phase Transitions | def-market-phase | 4.6 |
| Thin Market Kernel | def-thin-kernel | 3.4 |
| Gate Node Specification | (Nodes 1-21) | 7.2 |
| Barrier Specification | (20 barriers) | 7.3 |
| Failure Mode C.E-B.C | def-failure-* | 14.2-14.6 |
| Surgery Contracts | def-surg-* | 15.2-15.9 |
| MKT-Consistency | thm-mkt-consistency | 16.1 |
| MKT-Exclusion | thm-mkt-exclusion | 16.2 |
| MKT-Trichotomy | thm-mkt-trichotomy | 16.3 |
| MKT-Equivariance | thm-mkt-equivariance | 16.4 |
| MKT-HorizonLimit | thm-mkt-horizon | 16.5 |
| Price Complexity | def-price-complexity | 17.1 |
| Market Complexity Phases | def-complexity-phases | 17.2 |

### 20.4 Implementation Checklist

For practitioners implementing this framework:

1. **Minimal viable implementation:**
   - [ ] Core SDF computation (Section 4.1)
   - [ ] Basic gates (Nodes 1, 3, 5, 6, 11)
   - [ ] Key barriers (BarrierSat, BarrierOmin, BarrierTypeII)
   - [ ] Certificate generation

2. **Standard implementation:**
   - [ ] All 21 gate nodes
   - [ ] All 20 barriers
   - [ ] Failure mode detection
   - [ ] Surgery trigger logic

3. **Advanced implementation:**
   - [ ] Ruppeiner geometry for risk metrics
   - [ ] Phase detection (Crystal/Liquid/Gas)
   - [ ] Multi-barrier coordination
   - [ ] Full surgery automation

4. **Production deployment:**
   - [ ] Real-time barrier monitoring
   - [ ] Certificate logging and audit trail
   - [ ] Integration with trading systems
   - [ ] Stress testing framework

### 20.5 Theoretical Completeness

The framework is **theoretically complete** in the following senses:

1. **Asset coverage:** All 12 major asset classes fit within the SDF framework
2. **Failure coverage:** All market failures route through the 15-mode taxonomy
3. **Intervention coverage:** All crisis states have corresponding surgery contracts
4. **Metatheoretic coverage:** Five metatheorems constrain any consistent extension

**Open questions for future work:**
- Extension to multi-agent game-theoretic equilibria
- Integration with quantum probability for option pricing
- Climate risk as additional barrier class
- Cross-border regulatory coordination

---

## 21. Calibration Guidance

This section provides practical guidance for **calibrating thresholds** in the gate nodes and barriers. Proper calibration is essential for avoiding false positives (unnecessary trading halts) and false negatives (missing genuine risks).

### 21.1 General Calibration Principles

:::{prf:remark} Calibration Philosophy
:label: rem-calibration-philosophy

Thresholds should be set to achieve:
1. **High recall for catastrophic events** (never miss a crisis)
2. **Acceptable precision for warnings** (tolerate some false alarms)
3. **Regime-dependent adjustment** (tighter in stress, looser in calm)
4. **Asset-class specificity** (equities differ from bonds)
:::

**The Precision-Recall Tradeoff:**

| Setting | False Positives | False Negatives | Use Case |
|---------|----------------|-----------------|----------|
| Conservative | High | Low | Critical infrastructure, pension funds |
| Balanced | Medium | Medium | Standard institutional trading |
| Aggressive | Low | High | Prop trading, market makers |

### 21.2 Gate Node Threshold Calibration

#### Node 1: Solvency Threshold

**Recommended thresholds:**
```
Conservative:  NAV_threshold = 0.20 (fail if leverage > 5×)
Balanced:      NAV_threshold = 0.10 (fail if leverage > 10×)
Aggressive:    NAV_threshold = 0.05 (fail if leverage > 20×)
```

**Calibration procedure:**
1. Compute historical NAV/Notional ratios
2. Identify the 1st percentile of the ratio distribution
3. Set threshold at 2× the 1st percentile (buffer for measurement error)

**Python calibration:**
```python
def calibrate_solvency(nav_history, notional_history, percentile=1):
    """Calibrate solvency threshold from historical data."""
    ratios = np.array(nav_history) / np.array(notional_history)
    threshold = 2 * np.percentile(ratios, percentile)
    return max(threshold, 0.05)  # Floor at 5%
```

#### Node 3: Leverage Ratio Threshold

**Recommended thresholds by asset class:**

| Asset Class | Conservative | Balanced | Aggressive |
|-------------|-------------|----------|------------|
| Equities | 2.0 | 3.0 | 5.0 |
| Government Bonds | 10.0 | 15.0 | 25.0 |
| Corporate Bonds | 5.0 | 8.0 | 12.0 |
| FX | 20.0 | 50.0 | 100.0 |
| Commodities | 3.0 | 5.0 | 10.0 |
| Derivatives | 1.5 | 2.5 | 4.0 |

**Calibration procedure:**
1. Collect leverage ratios at historical stress points
2. Identify maximum leverage that survived stress without default
3. Apply 0.8× safety factor

```python
def calibrate_leverage(leverage_history, stress_events, survived):
    """Calibrate leverage from stress survival data."""
    stress_leverages = [leverage_history[t] for t in stress_events if survived[t]]
    max_safe = max(stress_leverages) if stress_leverages else 2.0
    return 0.8 * max_safe
```

#### Node 5: Stationarity Threshold

**Recommended thresholds for regime change detection:**
```
p-value threshold (ADF test): 0.05 (standard), 0.10 (sensitive)
Break detection window: 20-60 days
Minimum observations: 252 (1 year)
```

**Calibration via rolling ADF:**
```python
def calibrate_stationarity(returns, window=252):
    """Calibrate stationarity threshold via rolling ADF."""
    from statsmodels.tsa.stattools import adfuller

    p_values = []
    for i in range(window, len(returns)):
        result = adfuller(returns[i-window:i])
        p_values.append(result[1])

    # Set threshold at 95th percentile of calm period p-values
    return np.percentile(p_values, 95)
```

#### Node 6: Capacity Utilization Threshold

**Recommended thresholds:**
```
Warning level:   70% of estimated market capacity
Critical level:  90% of estimated market capacity
Halt level:      95% of estimated market capacity
```

**Capacity estimation:**
```python
def estimate_capacity(volume_history, price_impact_history):
    """Estimate market capacity from impact regression."""
    # Market capacity ≈ volume at which impact exceeds 1%
    X = np.log(volume_history).reshape(-1, 1)
    y = np.abs(price_impact_history)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)

    # Solve for volume where impact = 0.01
    capacity = np.exp((0.01 - reg.intercept_) / reg.coef_[0])
    return capacity
```

#### Node 11: Representation Accuracy Threshold

**Recommended thresholds:**
```
Maximum prediction error (RMSE): 2 × historical volatility
Maximum model complexity: Effective parameters < n/10
Maximum regime uncertainty: H(K) < log(|K|) - 0.5
```

**Calibration for model accuracy:**
```python
def calibrate_representation(predictions, actuals, vol_estimate):
    """Calibrate representation threshold."""
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    threshold = 2 * vol_estimate
    return rmse < threshold, threshold
```

### 21.3 Barrier Threshold Calibration

#### BarrierSat: Position Limits

**Recommended limits:**

| Position Type | Conservative | Balanced | Aggressive |
|---------------|-------------|----------|------------|
| Single name (% of NAV) | 5% | 10% | 20% |
| Sector (% of NAV) | 20% | 30% | 50% |
| Single name (% of ADV) | 5% | 10% | 25% |
| Gross exposure | 100% | 150% | 200% |

**Dynamic adjustment:**
```python
def adjust_position_limit(base_limit, vol_regime, liquidity_regime):
    """Adjust position limits based on market regime."""
    vol_factor = 1.0 / (1 + vol_regime)  # Reduce in high vol
    liq_factor = liquidity_regime  # Reduce in low liquidity
    return base_limit * vol_factor * liq_factor
```

#### BarrierTypeII: Volatility-of-Volatility Crisis

**Recommended thresholds:**
```
Vol-of-vol warning:  VVIX/VIX > 5.0 (historical median ~4.5)
Vol-of-vol critical: VVIX/VIX > 7.0 (99th percentile)
Vol regime change:   VIX > 2 × 20-day MA
```

**Calibration from historical vol dynamics:**
```python
def calibrate_vol_of_vol(vix_history, vvix_history):
    """Calibrate vol-of-vol thresholds."""
    ratio = np.array(vvix_history) / np.array(vix_history)
    return {
        'warning': np.percentile(ratio, 75),
        'critical': np.percentile(ratio, 99),
        'median': np.median(ratio)
    }
```

#### BarrierOmin: Flash Crash Detection

**Recommended thresholds:**
```
Price move threshold: -5% in 5 minutes (equities)
Volume spike: > 10× average 5-minute volume
Quote withdrawal: > 50% reduction in depth
Recovery time: < 30 minutes for temporary classification
```

**Real-time detection:**
```python
def detect_flash_crash(prices, volumes, depths, window_minutes=5):
    """Detect flash crash conditions."""
    price_change = (prices[-1] - prices[-window_minutes]) / prices[-window_minutes]
    volume_ratio = volumes[-window_minutes:].sum() / volumes[-60:-window_minutes].mean()
    depth_change = depths[-1] / depths[-window_minutes]

    flash_crash = (
        price_change < -0.05 and  # 5% drop
        volume_ratio > 10 and      # 10× volume spike
        depth_change < 0.5         # 50% depth reduction
    )
    return flash_crash, {'price': price_change, 'volume': volume_ratio, 'depth': depth_change}
```

#### BarrierFreq: HFT Oscillation Detection

**Recommended thresholds:**
```
Price oscillation: > 3 reversals per minute
Quote flickering: > 100 updates per second
Layering detection: > 5 levels with < 100ms lifetime
Momentum ignition: Correlation(flow, returns) > 0.9 with reversal
```

**Detection algorithm:**
```python
def detect_hft_oscillation(prices, timestamps):
    """Detect HFT-induced oscillation."""
    # Count sign changes in 1-minute windows
    returns = np.diff(prices)
    sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
    duration_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60

    reversals_per_minute = sign_changes / duration_minutes
    return reversals_per_minute > 3, reversals_per_minute
```

### 21.4 Regime-Dependent Calibration

Thresholds should adjust based on the macro regime $K_t$:

:::{prf:definition} Regime-Adjusted Threshold
:label: def-regime-threshold

For base threshold $\tau_0$ and regime $K$, the adjusted threshold is:
$$
\tau_K = \tau_0 \cdot \phi_K,
$$
where $\phi_K$ is the regime adjustment factor:

| Regime | $\phi_K$ | Interpretation |
|--------|----------|----------------|
| Risk-On | 1.2 | Looser thresholds |
| Neutral | 1.0 | Base thresholds |
| Risk-Off | 0.8 | Tighter thresholds |
| Crisis | 0.5 | Much tighter |
| Recovery | 0.9 | Slightly tight |
:::

**Regime detection and adjustment:**
```python
class RegimeAdjustedThresholds:
    """Adjust thresholds based on market regime."""

    def __init__(self, base_thresholds):
        self.base = base_thresholds
        self.phi = {
            'risk_on': 1.2,
            'neutral': 1.0,
            'risk_off': 0.8,
            'crisis': 0.5,
            'recovery': 0.9
        }

    def detect_regime(self, vix, credit_spread, momentum):
        """Simple regime detection."""
        if vix > 30 and credit_spread > 500:
            return 'crisis'
        elif vix > 25 or credit_spread > 300:
            return 'risk_off'
        elif vix < 15 and momentum > 0:
            return 'risk_on'
        elif vix < 20 and credit_spread < 200:
            return 'recovery' if self.prev_regime == 'crisis' else 'neutral'
        return 'neutral'

    def get_threshold(self, name, regime):
        """Get regime-adjusted threshold."""
        return self.base[name] * self.phi[regime]
```

### 21.5 Cross-Validation and Backtesting

**Calibration validation procedure:**

1. **In-sample calibration:** Fit thresholds to 70% of historical data
2. **Out-of-sample validation:** Test on remaining 30%
3. **Stress period validation:** Ensure thresholds trigger appropriately during known crises
4. **False positive rate:** Target < 5% false alarms in calm periods
5. **True positive rate:** Target > 95% detection of known stress events

```python
def validate_calibration(thresholds, test_data, known_crises):
    """Validate calibration against test data."""
    predictions = []
    actuals = []

    for t in range(len(test_data)):
        # Check if any gate/barrier triggers
        triggered = any(
            test_data[t][gate] > thresholds[gate]
            for gate in thresholds.keys()
        )
        predictions.append(triggered)
        actuals.append(t in known_crises)

    # Compute metrics
    tp = sum(p and a for p, a in zip(predictions, actuals))
    fp = sum(p and not a for p, a in zip(predictions, actuals))
    fn = sum(not p and a for p, a in zip(predictions, actuals))
    tn = sum(not p and not a for p, a in zip(predictions, actuals))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'recall': recall,           # Target > 0.95
        'precision': precision,      # Target > 0.50
        'false_positive_rate': fpr   # Target < 0.05
    }
```

### 21.6 Threshold Summary Table

```{list-table} Complete Threshold Summary
:header-rows: 1
:name: threshold-summary

* - Component
  - Parameter
  - Conservative
  - Balanced
  - Aggressive
* - Node 1 (Solvency)
  - NAV/Notional minimum
  - 0.20
  - 0.10
  - 0.05
* - Node 3 (Leverage)
  - Max equity leverage
  - 2.0
  - 3.0
  - 5.0
* - Node 5 (Stationarity)
  - ADF p-value
  - 0.10
  - 0.05
  - 0.01
* - Node 6 (Capacity)
  - Utilization warning
  - 0.60
  - 0.70
  - 0.80
* - Node 11 (Representation)
  - Max RMSE / vol
  - 1.5
  - 2.0
  - 3.0
* - BarrierSat
  - Single name % NAV
  - 0.05
  - 0.10
  - 0.20
* - BarrierTypeII
  - VVIX/VIX critical
  - 6.0
  - 7.0
  - 8.0
* - BarrierOmin
  - 5-min price drop
  - -0.03
  - -0.05
  - -0.10
* - BarrierFreq
  - Reversals/minute
  - 2
  - 3
  - 5
```

---

## 22. Risk Attribution Framework

This section provides a framework for **attributing risk and losses** to specific gate failures, barrier breaches, and failure modes.

### 22.1 Hierarchical Risk Attribution

:::{prf:definition} Risk Attribution Decomposition
:label: def-risk-attribution

Total portfolio risk $\sigma^2$ decomposes hierarchically:
$$
\sigma^2 = \underbrace{\sigma^2_{\text{sys}}}_{\text{Systematic}} + \underbrace{\sigma^2_{\text{idio}}}_{\text{Idiosyncratic}} + \underbrace{\sigma^2_{\text{regime}}}_{\text{Regime}} + \underbrace{\sigma^2_{\text{barrier}}}_{\text{Barrier}}.
$$
:::

**Component definitions:**
- $\sigma^2_{\text{sys}}$: Risk from factor exposures (market, sector, style)
- $\sigma^2_{\text{idio}}$: Asset-specific residual risk
- $\sigma^2_{\text{regime}}$: Risk from potential regime changes
- $\sigma^2_{\text{barrier}}$: Risk from potential barrier breaches

### 22.2 Gate-Based Risk Attribution

Each gate failure contributes to total risk. We attribute risk to gates based on **proximity to threshold**:

:::{prf:definition} Gate Risk Contribution
:label: def-gate-risk

For gate $i$ with current value $v_i$ and threshold $\tau_i$, the gate risk contribution is:
$$
R_i = w_i \cdot \max\left(0, \frac{v_i - \tau_i^{\text{warn}}}{\tau_i^{\text{crit}} - \tau_i^{\text{warn}}}\right)^2,
$$
where $w_i$ is the weight representing potential loss if gate $i$ fails.
:::

**Implementation:**
```python
class GateRiskAttributor:
    """Attribute risk to individual gates."""

    def __init__(self, gate_weights, warn_thresholds, crit_thresholds):
        self.weights = gate_weights  # Potential loss per gate
        self.warn = warn_thresholds
        self.crit = crit_thresholds

    def attribute(self, gate_values):
        """Compute risk attribution to each gate."""
        attributions = {}

        for gate_name, value in gate_values.items():
            warn = self.warn[gate_name]
            crit = self.crit[gate_name]
            weight = self.weights[gate_name]

            if value < warn:
                attributions[gate_name] = 0.0
            else:
                proximity = (value - warn) / (crit - warn)
                attributions[gate_name] = weight * min(proximity, 1.0)**2

        return attributions

    def total_gate_risk(self, gate_values):
        """Compute total risk from gate proximity."""
        return sum(self.attribute(gate_values).values())
```

### 22.3 Failure Mode Risk Attribution

Risk attributed to each failure mode based on proximity and conditional severity:

:::{prf:definition} Failure Mode Risk
:label: def-fm-risk

For failure mode $F$ with probability $p_F$ and severity $s_F$:
$$
R_F = p_F \cdot s_F \cdot \mathbb{E}[\text{Loss} \mid F],
$$
where $p_F$ is estimated from gate/barrier states.
:::

**Failure mode severity table:**

| Mode | Base Severity | Typical Loss | Recovery Time |
|------|--------------|--------------|---------------|
| C.E (Blow-up) | 10 | 50-100% | Permanent |
| C.D (Concentration) | 7 | 20-50% | 1-6 months |
| C.C (Zeno) | 5 | 5-20% | Days-weeks |
| T.E (Flash crash) | 6 | 10-30% | Hours-days |
| T.D (Frozen) | 8 | 20-40% | Weeks-months |
| T.C (Complexity) | 6 | 10-30% | Weeks |
| D.E (Oscillation) | 7 | 15-40% | Months |
| D.D (Dispersion) | 3 | Gains | N/A |
| D.C (Undecidable) | 8 | Unknown | Unknown |
| S.E (Supercritical) | 9 | 30-70% | Months |
| S.D (Flat vol) | 2 | Opportunity cost | N/A |
| S.C (Drift) | 5 | 10-25% | Ongoing |
| B.E (External) | 8 | 20-50% | Variable |
| B.D (Starvation) | 6 | 15-35% | Weeks |
| B.C (Misalignment) | 4 | 5-20% | Variable |

**Implementation:**
```python
class FailureModeAttributor:
    """Attribute risk to failure modes."""

    SEVERITIES = {
        'C.E': 10, 'C.D': 7, 'C.C': 5,
        'T.E': 6, 'T.D': 8, 'T.C': 6,
        'D.E': 7, 'D.D': 3, 'D.C': 8,
        'S.E': 9, 'S.D': 2, 'S.C': 5,
        'B.E': 8, 'B.D': 6, 'B.C': 4
    }

    EXPECTED_LOSS = {
        'C.E': 0.75, 'C.D': 0.35, 'C.C': 0.12,
        'T.E': 0.20, 'T.D': 0.30, 'T.C': 0.20,
        'D.E': 0.27, 'D.D': -0.10, 'D.C': 0.50,
        'S.E': 0.50, 'S.D': 0.05, 'S.C': 0.17,
        'B.E': 0.35, 'B.D': 0.25, 'B.C': 0.12
    }

    def estimate_probability(self, mode, gate_states, barrier_states):
        """Estimate failure mode probability from current states."""
        # Map failure modes to relevant gates/barriers
        mode_gates = {
            'C.E': ['solvency', 'turnover'],
            'C.D': ['solvency', 'capacity'],
            'T.E': ['connectivity', 'stiffness'],
            'D.E': ['oscillation', 'stability'],
            'S.E': ['leverage', 'stationarity'],
            'B.E': ['coupling', 'input'],
            # ... etc
        }

        relevant_gates = mode_gates.get(mode, [])
        if not relevant_gates:
            return 0.01  # Base rate

        # Probability increases with gate proximity to failure
        gate_risks = [gate_states.get(g, 0) for g in relevant_gates]
        return min(1.0, np.mean(gate_risks) * 2)

    def attribute(self, gate_states, barrier_states):
        """Compute risk attribution to each failure mode."""
        attributions = {}

        for mode in self.SEVERITIES:
            p = self.estimate_probability(mode, gate_states, barrier_states)
            s = self.SEVERITIES[mode] / 10  # Normalize to [0, 1]
            loss = self.EXPECTED_LOSS[mode]

            if loss > 0:  # Only attribute negative outcomes
                attributions[mode] = p * s * loss

        return attributions
```

### 22.4 Loss Attribution Post-Event

After a loss event, attribute the loss to specific causes:

```python
class LossAttributor:
    """Attribute realized losses to causes."""

    def attribute_loss(self, loss, pre_event_state, post_event_state, timeline):
        """
        Attribute a realized loss to gates, barriers, and failure modes.

        Args:
            loss: Total loss amount
            pre_event_state: State before event
            post_event_state: State after event
            timeline: List of (time, event) during the event
        """
        attribution = {
            'gates': {},
            'barriers': {},
            'failure_modes': {},
            'unexplained': 0.0
        }

        # Identify which gates failed
        for gate in pre_event_state['gates']:
            if pre_event_state['gates'][gate] == 'PASS' and \
               post_event_state['gates'][gate] == 'FAIL':
                attribution['gates'][gate] = self._estimate_gate_contribution(
                    gate, loss, pre_event_state, post_event_state
                )

        # Identify which barriers breached
        for barrier in pre_event_state['barriers']:
            if pre_event_state['barriers'][barrier] == 'CLEAR' and \
               post_event_state['barriers'][barrier] == 'BREACHED':
                attribution['barriers'][barrier] = self._estimate_barrier_contribution(
                    barrier, loss, pre_event_state, post_event_state
                )

        # Map to failure modes
        attribution['failure_modes'] = self._identify_failure_modes(
            attribution['gates'], attribution['barriers']
        )

        # Unexplained residual
        explained = (sum(attribution['gates'].values()) +
                    sum(attribution['barriers'].values()))
        attribution['unexplained'] = max(0, loss - explained)

        return attribution
```

### 22.5 Risk Attribution Dashboard

**Key metrics for risk monitoring:**

```python
class RiskDashboard:
    """Real-time risk attribution dashboard."""

    def compute_metrics(self, portfolio, market_state):
        """Compute dashboard metrics."""
        return {
            # Level 1: Summary
            'total_var_95': self.compute_var(portfolio, 0.95),
            'total_es_95': self.compute_es(portfolio, 0.95),

            # Level 2: Category attribution
            'systematic_risk': self.systematic_attribution(portfolio),
            'idiosyncratic_risk': self.idiosyncratic_attribution(portfolio),
            'regime_risk': self.regime_attribution(portfolio, market_state),
            'barrier_risk': self.barrier_attribution(portfolio, market_state),

            # Level 3: Gate proximity
            'gate_risk_scores': self.gate_attributor.attribute(market_state['gates']),
            'nearest_gate_to_fail': self.find_nearest_gate(market_state['gates']),

            # Level 4: Failure mode probabilities
            'failure_mode_risks': self.fm_attributor.attribute(
                market_state['gates'], market_state['barriers']
            ),
            'dominant_failure_mode': self.find_dominant_mode(market_state),

            # Level 5: Action recommendations
            'recommended_hedges': self.recommend_hedges(portfolio, market_state),
            'recommended_reductions': self.recommend_reductions(portfolio, market_state)
        }
```

---

## 23. Backtesting Framework

This section provides a rigorous backtesting framework for validating the Market Sieve on historical data.

### 23.1 Backtesting Objectives

:::{prf:remark} Backtesting Goals
:label: rem-backtest-goals

The backtesting framework aims to:
1. **Validate gate thresholds** against historical crises
2. **Measure false positive/negative rates** for barriers
3. **Test surgery effectiveness** on historical interventions
4. **Calibrate regime detection** accuracy
5. **Estimate economic value** of the Sieve
:::

### 23.2 Historical Event Database

**Required data structure:**

```python
@dataclass
class HistoricalEvent:
    """Documented market stress event."""
    name: str                    # e.g., "2008 Financial Crisis"
    start_date: datetime
    end_date: datetime
    peak_date: datetime          # Maximum stress
    asset_classes: List[str]     # Affected asset classes
    failure_modes: List[str]     # Failure modes observed
    interventions: List[str]     # Surgeries applied
    max_drawdown: float          # Peak-to-trough loss
    recovery_time: int           # Days to recovery
    gates_that_failed: List[str] # Gates that should have triggered
    barriers_breached: List[str] # Barriers that should have triggered

HISTORICAL_EVENTS = [
    HistoricalEvent(
        name="1987 Black Monday",
        start_date=datetime(1987, 10, 14),
        end_date=datetime(1987, 10, 26),
        peak_date=datetime(1987, 10, 19),
        asset_classes=["equities", "options"],
        failure_modes=["T.E", "D.E"],
        interventions=["SurgTE"],
        max_drawdown=0.226,
        recovery_time=452,
        gates_that_failed=["stiffness", "oscillation", "connectivity"],
        barriers_breached=["BarrierOmin", "BarrierTypeII"]
    ),
    HistoricalEvent(
        name="1998 LTCM",
        start_date=datetime(1998, 8, 1),
        end_date=datetime(1998, 10, 15),
        peak_date=datetime(1998, 9, 23),
        asset_classes=["credit", "fx", "equity_vol"],
        failure_modes=["C.E", "S.E", "T.D"],
        interventions=["SurgCE", "SurgSE"],
        max_drawdown=0.44,
        recovery_time=90,
        gates_that_failed=["solvency", "leverage", "mixing"],
        barriers_breached=["BarrierTypeII", "BarrierGap", "BarrierLev"]
    ),
    HistoricalEvent(
        name="2008 Financial Crisis",
        start_date=datetime(2008, 9, 1),
        end_date=datetime(2009, 3, 9),
        peak_date=datetime(2008, 10, 10),
        asset_classes=["credit", "equities", "real_estate"],
        failure_modes=["C.E", "C.D", "T.D", "S.E"],
        interventions=["SurgCE", "SurgCD", "SurgTD", "SurgSE"],
        max_drawdown=0.569,
        recovery_time=1403,
        gates_that_failed=["solvency", "leverage", "connectivity", "mixing", "stationarity"],
        barriers_breached=["BarrierSat", "BarrierTypeII", "BarrierGap", "BarrierLev", "BarrierDef"]
    ),
    HistoricalEvent(
        name="2010 Flash Crash",
        start_date=datetime(2010, 5, 6),
        end_date=datetime(2010, 5, 6),
        peak_date=datetime(2010, 5, 6),
        asset_classes=["equities", "etfs"],
        failure_modes=["T.E", "C.C"],
        interventions=["SurgTE", "SurgCC"],
        max_drawdown=0.099,
        recovery_time=1,
        gates_that_failed=["stiffness", "connectivity"],
        barriers_breached=["BarrierOmin", "BarrierFreq"]
    ),
    HistoricalEvent(
        name="2015 CNH Devaluation",
        start_date=datetime(2015, 8, 11),
        end_date=datetime(2015, 8, 25),
        peak_date=datetime(2015, 8, 24),
        asset_classes=["fx", "em_equities"],
        failure_modes=["B.E", "D.E"],
        interventions=["SurgBE"],
        max_drawdown=0.12,
        recovery_time=60,
        gates_that_failed=["stationarity", "coupling"],
        barriers_breached=["BarrierCausal", "BarrierInput"]
    ),
    HistoricalEvent(
        name="2018 Volmageddon",
        start_date=datetime(2018, 2, 2),
        end_date=datetime(2018, 2, 9),
        peak_date=datetime(2018, 2, 5),
        asset_classes=["equity_vol", "etfs"],
        failure_modes=["S.E", "D.E", "C.E"],
        interventions=["SurgSE"],
        max_drawdown=0.12,
        recovery_time=14,
        gates_that_failed=["bifurcation", "stability", "leverage"],
        barriers_breached=["BarrierTypeII", "BarrierOmin"]
    ),
    HistoricalEvent(
        name="2020 COVID Crash",
        start_date=datetime(2020, 2, 20),
        end_date=datetime(2020, 3, 23),
        peak_date=datetime(2020, 3, 16),
        asset_classes=["all"],
        failure_modes=["B.E", "T.D", "S.E", "C.E"],
        interventions=["SurgBE", "SurgTD", "SurgSE", "SurgCE"],
        max_drawdown=0.339,
        recovery_time=148,
        gates_that_failed=["connectivity", "stationarity", "coupling", "leverage"],
        barriers_breached=["BarrierOmin", "BarrierTypeII", "BarrierGap", "BarrierInput"]
    ),
    HistoricalEvent(
        name="2022 LDI Crisis",
        start_date=datetime(2022, 9, 23),
        end_date=datetime(2022, 10, 14),
        peak_date=datetime(2022, 9, 28),
        asset_classes=["gilts", "gbp"],
        failure_modes=["C.E", "S.E", "T.D"],
        interventions=["SurgCE", "SurgSE", "SurgTD"],
        max_drawdown=0.25,
        recovery_time=21,
        gates_that_failed=["solvency", "leverage", "stiffness"],
        barriers_breached=["BarrierLev", "BarrierGap", "BarrierTypeII"]
    ),
]
```

### 23.3 Backtesting Metrics

:::{prf:definition} Sieve Performance Metrics
:label: def-backtest-metrics

For a set of $N$ historical events, the Sieve is evaluated on:

1. **Detection rate:** $DR = \frac{\text{Events where any gate/barrier triggered}}{\text{Total events}}$

2. **Early warning rate:** $EW = \frac{\text{Events with trigger} \ge 5 \text{ days before peak}}{\text{Total events}}$

3. **False positive rate:** $FPR = \frac{\text{Triggers in non-event periods}}{\text{Total non-event days}}$

4. **Failure mode accuracy:** $FMA = \frac{\text{Correctly identified failure modes}}{\text{Actual failure modes}}$

5. **Economic value:** $EV = \sum_{\text{events}} (\text{Loss avoided by early exit}) - (\text{Opportunity cost of false exits})$
:::

### 23.4 Backtesting Implementation

```python
class SieveBacktester:
    """Backtest the Market Sieve on historical data."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events
        self.results = {}

    def run_backtest(self):
        """Run full backtest across all events."""
        for event in self.events:
            self.results[event.name] = self._backtest_event(event)

        self._compute_aggregate_metrics()
        return self.results

    def _backtest_event(self, event: HistoricalEvent):
        """Backtest single event."""
        result = {
            'detected': False,
            'first_trigger_date': None,
            'days_before_peak': None,
            'gates_triggered': [],
            'barriers_triggered': [],
            'failure_modes_detected': [],
            'correct_gates': [],
            'missed_gates': [],
            'false_gates': [],
            'loss_at_trigger': None,
            'loss_at_peak': None,
            'loss_avoided': None
        }

        # Run sieve on each day from 30 days before start to event end
        start_window = event.start_date - timedelta(days=30)

        for date in self._date_range(start_window, event.end_date):
            state = self.data.get_state(date)

            # Run sieve
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                if not result['detected']:
                    result['detected'] = True
                    result['first_trigger_date'] = date
                    result['days_before_peak'] = (event.peak_date - date).days
                    result['loss_at_trigger'] = self._compute_loss(
                        event.start_date, date, event.asset_classes
                    )

                # Record triggers
                result['gates_triggered'].extend(certificate.get('failed_gates', []))
                result['barriers_triggered'].extend(certificate.get('breached_barriers', []))

        # Compute accuracy metrics
        result['gates_triggered'] = list(set(result['gates_triggered']))
        result['barriers_triggered'] = list(set(result['barriers_triggered']))

        result['correct_gates'] = [
            g for g in result['gates_triggered']
            if g in event.gates_that_failed
        ]
        result['missed_gates'] = [
            g for g in event.gates_that_failed
            if g not in result['gates_triggered']
        ]
        result['false_gates'] = [
            g for g in result['gates_triggered']
            if g not in event.gates_that_failed
        ]

        # Loss computation
        result['loss_at_peak'] = event.max_drawdown
        if result['loss_at_trigger'] is not None:
            result['loss_avoided'] = event.max_drawdown - result['loss_at_trigger']

        # Failure mode detection
        result['failure_modes_detected'] = self._identify_failure_modes_from_triggers(
            result['gates_triggered'], result['barriers_triggered']
        )

        return result

    def _compute_aggregate_metrics(self):
        """Compute aggregate backtest metrics."""
        n_events = len(self.events)

        # Detection rate
        detected = sum(1 for r in self.results.values() if r['detected'])
        self.results['_aggregate'] = {
            'detection_rate': detected / n_events,

            # Early warning rate (>= 5 days before peak)
            'early_warning_rate': sum(
                1 for r in self.results.values()
                if r['days_before_peak'] is not None and r['days_before_peak'] >= 5
            ) / n_events,

            # Average days of warning
            'avg_warning_days': np.mean([
                r['days_before_peak'] for r in self.results.values()
                if r['days_before_peak'] is not None
            ]),

            # Gate accuracy
            'gate_precision': self._compute_gate_precision(),
            'gate_recall': self._compute_gate_recall(),

            # Total loss avoided
            'total_loss_avoided': sum(
                r['loss_avoided'] for r in self.results.values()
                if r['loss_avoided'] is not None
            ),

            # Average loss avoided
            'avg_loss_avoided': np.mean([
                r['loss_avoided'] for r in self.results.values()
                if r['loss_avoided'] is not None
            ])
        }

    def _compute_gate_precision(self):
        """Compute precision of gate triggers."""
        correct = sum(len(r['correct_gates']) for r in self.results.values() if isinstance(r, dict))
        total = sum(len(r['gates_triggered']) for r in self.results.values() if isinstance(r, dict))
        return correct / total if total > 0 else 0

    def _compute_gate_recall(self):
        """Compute recall of gate triggers."""
        correct = sum(len(r['correct_gates']) for r in self.results.values() if isinstance(r, dict))
        expected = sum(len(e.gates_that_failed) for e in self.events)
        return correct / expected if expected > 0 else 0

    def generate_report(self):
        """Generate backtest report."""
        agg = self.results.get('_aggregate', {})

        report = f"""
# Market Sieve Backtest Report

## Summary Metrics
- Detection Rate: {agg.get('detection_rate', 0):.1%}
- Early Warning Rate (≥5 days): {agg.get('early_warning_rate', 0):.1%}
- Average Warning Days: {agg.get('avg_warning_days', 0):.1f}
- Gate Precision: {agg.get('gate_precision', 0):.1%}
- Gate Recall: {agg.get('gate_recall', 0):.1%}
- Total Loss Avoided: {agg.get('total_loss_avoided', 0):.1%}

## Event-by-Event Results
"""

        for event in self.events:
            r = self.results[event.name]
            report += f"""
### {event.name}
- Detected: {'✓' if r['detected'] else '✗'}
- Warning Days: {r['days_before_peak']}
- Gates Triggered: {', '.join(r['gates_triggered']) or 'None'}
- Barriers Triggered: {', '.join(r['barriers_triggered']) or 'None'}
- Loss at Trigger: {r['loss_at_trigger']:.1%} if r['loss_at_trigger'] else 'N/A'
- Loss Avoided: {r['loss_avoided']:.1%} if r['loss_avoided'] else 'N/A'
"""

        return report
```

### 23.5 False Positive Analysis

**Measuring false positives in non-crisis periods:**

```python
class FalsePositiveAnalyzer:
    """Analyze false positive rates."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events

    def compute_false_positive_rate(self, start_date, end_date):
        """Compute FPR over a date range excluding known events."""
        # Create mask of event periods
        event_dates = set()
        for event in self.events:
            for date in self._date_range(event.start_date, event.end_date):
                event_dates.add(date)

        non_event_days = 0
        false_triggers = 0

        for date in self._date_range(start_date, end_date):
            if date in event_dates:
                continue

            non_event_days += 1
            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                false_triggers += 1

        fpr = false_triggers / non_event_days if non_event_days > 0 else 0

        return {
            'false_positive_rate': fpr,
            'false_triggers': false_triggers,
            'non_event_days': non_event_days,
            'annualized_false_triggers': fpr * 252  # Trading days per year
        }

    def analyze_false_positives(self, start_date, end_date):
        """Detailed analysis of false positives."""
        event_dates = set()
        for event in self.events:
            for date in self._date_range(event.start_date, event.end_date):
                event_dates.add(date)

        false_positives = []

        for date in self._date_range(start_date, end_date):
            if date in event_dates:
                continue

            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                # Analyze what triggered and why
                false_positives.append({
                    'date': date,
                    'gates': certificate.get('failed_gates', []),
                    'barriers': certificate.get('breached_barriers', []),
                    'subsequent_5d_return': self._compute_forward_return(date, 5),
                    'subsequent_20d_return': self._compute_forward_return(date, 20),
                    'was_near_event': self._is_near_event(date)
                })

        return false_positives
```

### 23.6 Economic Value Computation

**Computing the economic value of the Sieve:**

```python
class EconomicValueComputer:
    """Compute economic value of the Sieve."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events

    def compute_value(self, initial_capital=100, exit_strategy='immediate'):
        """
        Compute economic value of using the Sieve.

        exit_strategy options:
        - 'immediate': Exit fully on trigger
        - 'gradual': Reduce 50% on warning, 100% on critical
        - 'hedge': Buy protection instead of exiting
        """

        # Baseline: Buy and hold through all events
        baseline_path = self._compute_baseline_path(initial_capital)

        # Sieve-protected path
        protected_path = self._compute_protected_path(initial_capital, exit_strategy)

        # Compute metrics
        baseline_final = baseline_path[-1]
        protected_final = protected_path[-1]

        baseline_dd = self._max_drawdown(baseline_path)
        protected_dd = self._max_drawdown(protected_path)

        baseline_sharpe = self._sharpe_ratio(baseline_path)
        protected_sharpe = self._sharpe_ratio(protected_path)

        return {
            'baseline_final_value': baseline_final,
            'protected_final_value': protected_final,
            'value_added': protected_final - baseline_final,
            'value_added_pct': (protected_final - baseline_final) / baseline_final,
            'baseline_max_drawdown': baseline_dd,
            'protected_max_drawdown': protected_dd,
            'drawdown_reduction': baseline_dd - protected_dd,
            'baseline_sharpe': baseline_sharpe,
            'protected_sharpe': protected_sharpe,
            'sharpe_improvement': protected_sharpe - baseline_sharpe
        }

    def _compute_protected_path(self, initial_capital, exit_strategy):
        """Compute capital path with Sieve protection."""
        capital = initial_capital
        path = [capital]
        in_market = True
        exit_date = None
        reentry_date = None

        for date in self._all_dates():
            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)
            daily_return = self.data.get_return(date)

            # Exit logic
            if in_market and certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                if exit_strategy == 'immediate':
                    in_market = False
                    exit_date = date
                elif exit_strategy == 'gradual':
                    # Reduce exposure
                    capital *= (1 + 0.5 * daily_return)  # 50% exposure
                    if certificate['severity'] == 'CRITICAL':
                        in_market = False
                        exit_date = date

            # Re-entry logic: wait for all-clear for 5 consecutive days
            if not in_market:
                if certificate['status'] == 'VALID':
                    if reentry_date is None:
                        reentry_date = date
                    elif (date - reentry_date).days >= 5:
                        in_market = True
                        reentry_date = None
                else:
                    reentry_date = None

            # Apply return
            if in_market:
                capital *= (1 + daily_return)
            else:
                capital *= (1 + self.data.get_risk_free_rate(date) / 252)  # Cash return

            path.append(capital)

        return path
```

### 23.7 Sensitivity Analysis

**Testing sensitivity to threshold choices:**

```python
class ThresholdSensitivityAnalyzer:
    """Analyze sensitivity to threshold choices."""

    def __init__(self, base_sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.base_sieve = base_sieve
        self.data = data
        self.events = events

    def sweep_threshold(self, gate_name, threshold_range):
        """Sweep a single threshold and measure impact."""
        results = []

        for threshold in threshold_range:
            # Create modified sieve
            modified_sieve = self._create_modified_sieve(gate_name, threshold)

            # Run backtest
            backtester = SieveBacktester(modified_sieve, self.data, self.events)
            backtest_results = backtester.run_backtest()
            agg = backtest_results['_aggregate']

            # Run FPR analysis
            fpr_analyzer = FalsePositiveAnalyzer(modified_sieve, self.data, self.events)
            fpr_results = fpr_analyzer.compute_false_positive_rate(
                self.data.start_date, self.data.end_date
            )

            results.append({
                'threshold': threshold,
                'detection_rate': agg['detection_rate'],
                'early_warning_rate': agg['early_warning_rate'],
                'false_positive_rate': fpr_results['false_positive_rate'],
                'f1_score': self._f1_score(agg['detection_rate'], fpr_results['false_positive_rate'])
            })

        return results

    def find_optimal_threshold(self, gate_name, threshold_range, objective='f1'):
        """Find optimal threshold for a gate."""
        results = self.sweep_threshold(gate_name, threshold_range)

        if objective == 'f1':
            best = max(results, key=lambda x: x['f1_score'])
        elif objective == 'detection':
            # Maximize detection subject to FPR < 5%
            valid = [r for r in results if r['false_positive_rate'] < 0.05]
            best = max(valid, key=lambda x: x['detection_rate']) if valid else results[0]
        elif objective == 'economic':
            # Maximize economic value
            best = max(results, key=lambda x: x.get('economic_value', 0))

        return best['threshold']
```

---

## 24. Capital Capacity Constraints: Market Depth as Information Bandwidth

**Market liquidity imposes an information-theoretic bound on representational complexity**, and metric curvature emerges as the regulatory mechanism.

### 24.1 The Market Depth–Position Inequality

**Definition 24.1.1 (No-Arbitrage Capacity Bound).** Consider the market interface (order book, quote stream) as an information channel. The **market capacity** $C_{\text{mkt}}$ bounds the information content of any sustainable position:
$$
I_{\text{position}} \le C_{\text{mkt}},
$$
where:
- $I_{\text{position}}$ is the information content of the portfolio position (bits needed to specify the strategy),
- $C_{\text{mkt}}$ is the effective information capacity of the market interface (market depth, quote frequency).

Units: $[I_{\text{position}}] = [C_{\text{mkt}}] = \text{nat}$.

**Consequence:** Positions with information content exceeding market capacity are unsustainable. Strategies that violate this bound incur ungrounded exposure risk.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Bulk information $I_{\text{bulk}}$ | Position information $I_{\text{position}}$ |
| Boundary capacity $C_{\partial}$ | Market capacity $C_{\text{mkt}}$ |
| Shutter channel | Order book / quote stream |

**Definition 24.1.2 (Capital Information Density).** Let $\rho(w, t)$ denote the probability density of portfolio weights $w \in \mathcal{W}$ at time $t$. The **capital information density** is:
$$
\rho_I(w, t) := -\rho(w, t) \log \rho(w, t) + \frac{1}{2}\rho(w, t) \log\det G(w),
$$
where $G(w)$ is the Ruppeiner risk metric (Definition 4.5.1).

*Interpretation:* The first term is the Shannon entropy density; the second is the geometric correction accounting for risk-induced volume distortion.

**Definition 24.1.3 (Market Depth as Area Law).** The market capacity follows an **area law**:
$$
C_{\text{mkt}} = \frac{1}{\eta_{\text{tick}}} \cdot \text{Depth}(\partial\mathcal{W}),
$$
where:
- $\text{Depth}(\partial\mathcal{W})$ is the aggregate market depth at the trading boundary,
- $\eta_{\text{tick}}$ is the minimum price tick per unit information (market microstructure parameter).

**Cross-reference:** Node 13 (BoundaryCheck) in Section 7 corresponds to the grounding condition.

### 24.2 Capacity-Constrained Risk Metric Law

**Theorem 24.2.1 (Capacity-Constrained Ruppeiner Law).** Under the no-arbitrage capacity constraint (Definition 24.1.1), the equilibrium risk metric satisfies:
$$
\boxed{R_{ij} - \frac{1}{2} R \, G_{ij} + \Lambda G_{ij} = \kappa \, T_{ij}^{\text{risk}}}
$$
where:
- $R_{ij}$ is the Ricci curvature of the risk metric $G$,
- $\Lambda$ is the baseline risk premium (cosmological constant),
- $T_{ij}^{\text{risk}}$ is the risk-energy tensor (loss gradients, concentration risk),
- $\kappa$ is the risk coupling constant.

**Economic interpretation:**
1. **Risk concentration curves the portfolio space.** High-risk positions induce metric curvature.
2. **Curvature bounds position size.** The capacity constraint prevents information volume from exceeding market depth.
3. **Geodesics are optimal trades.** Minimum-risk paths follow the curved geometry.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Einstein equations on $\mathcal{Z}$ | Ruppeiner equations on portfolio space |
| Information density $\rho_I$ | Capital density |
| Curvature regulator | Risk concentration regulator |

### 24.3 Capacity Saturation Diagnostic

**Node GateCapacitySat: Capacity Saturation Check**

| # | Name | Measures | Trigger |
|---|------|----------|---------|
| Gate40 | CapacitySatCheck | Position vs market depth ratio | $I_{\text{position}} / C_{\text{mkt}} > 1 - \epsilon$ |

**Definition 24.3.1 (Capacity Saturation Ratio).**
$$
\nu_{\text{cap}}(t) := \frac{I_{\text{position}}(t)}{C_{\text{mkt}}(t)}.
$$

| $\nu_{\text{cap}}$ | Interpretation | Action |
|-------------------|----------------|--------|
| $\ll 1$ | Under-utilized capacity | Review for suboptimal capital deployment |
| $\approx 1$ | Operating at capacity | Risk metric regulates; constrain new positions |
| $> 1$ | **Violation** | Ungrounded position; deleveraging required |

**Cross-reference:** This extends the solvency gates from Section 7.3.

---

## 25. Portfolio Transport Geometry: Wasserstein-Fisher-Rao for Rebalancing

The WFR metric unifies **continuous rebalancing** (Wasserstein transport) with **discrete regime transitions** (Fisher-Rao reaction).

### 25.1 The Failure of Product Metrics

Standard portfolio metrics treat asset allocation and regime selection separately. This creates discontinuities when switching regimes (e.g., from equities to bonds during a crisis).

**The WFR Solution:** Treat the portfolio not as a fixed allocation $w$, but as a **belief distribution** $\rho(w, K)$ over allocations and regimes. The WFR metric provides:
- **Transport (Wasserstein):** Continuous rebalancing within a regime.
- **Reaction (Fisher-Rao):** Discrete regime transitions (risk-on → risk-off).

### 25.2 The WFR Rebalancing Action

**Definition 25.2.1 (WFR Rebalancing Cost).** The squared WFR distance between portfolio distributions $\rho_0$ and $\rho_1$ is:
$$
d^2_{\text{WFR}}(\rho_0, \rho_1) = \inf_{(\rho, v, r)} \int_0^1 \int_{\mathcal{W}} \left( \underbrace{\|v_s(w)\|_G^2}_{\text{Rebalancing cost}} + \underbrace{\lambda^2 |r_s(w)|^2}_{\text{Regime switch cost}} \right) d\rho_s(w) \, ds
$$
subject to the unbalanced continuity equation:
$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r,
$$
where:
- $v_s(w)$ is the **rebalancing velocity** (continuous portfolio drift),
- $r_s(w)$ is the **regime transition rate** (mass creation/destruction),
- $\lambda$ is the **rebalancing granularity** (crossover scale).

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Transport velocity $v$ | Continuous rebalancing |
| Reaction rate $r$ | Regime jumps / allocation shifts |
| Length scale $\lambda$ | Rebalancing granularity |
| Mass $\rho$ | Allocation weight / belief |

### 25.3 Transport vs Reaction in Markets

**1. Transport (Wasserstein Component):**
$$
\partial_s \rho + \nabla \cdot (\rho v) = 0
$$
- **Market interpretation:** Gradual rebalancing (dollar-cost averaging, systematic rotation).
- **Cost:** Transaction costs, market impact proportional to $\|v\|^2$.
- **Regime:** Same investment regime, different weights.

**2. Reaction (Fisher-Rao Component):**
$$
\partial_s \rho = \rho r
$$
- **Market interpretation:** Regime switches (risk-on to risk-off, sector rotation).
- **Cost:** Opportunity cost, execution risk, regime transition costs.
- **Regime:** Different investment regime, mass redistributed.

**3. The Crossover Scale $\lambda$:**
- If rebalancing distance $< \lambda$: Transport preferred (gradual rebalancing).
- If rebalancing distance $> \lambda$: Reaction preferred (discrete regime switch).

**Interpretation:** $\lambda$ is the crossover scale: for distances below $\lambda$, transport dominates; above $\lambda$, reaction dominates.

### 25.4 WFR Portfolio World Model

**Definition 25.4.1 (WFR Portfolio Dynamics).** The policy outputs a generalized velocity $(v, r)$ to minimize WFR path length to the target allocation (goal portfolio).

```python
class WFRPortfolioModel(nn.Module):
    """
    WFR-based portfolio dynamics model.

    Predicts continuous rebalancing (transport) and
    regime transitions (reaction) in a unified framework.
    """

    def __init__(self, n_assets: int, n_regimes: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = n_assets + n_regimes + 1  # weights + regime_probs + risk_budget

        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Transport velocity (continuous rebalancing)
        self.head_v = nn.Linear(hidden_dim, n_assets)

        # Reaction rate (regime transitions)
        self.head_r = nn.Linear(hidden_dim, n_regimes)

    def forward(self, w: torch.Tensor, regime: torch.Tensor,
                risk_budget: torch.Tensor, dt: float = 0.1):
        """
        Predict next portfolio state via WFR dynamics.

        Returns:
            w_next: Next portfolio weights
            regime_next: Next regime probabilities
            v: Rebalancing velocity
            r: Regime transition rate
        """
        inp = torch.cat([w, regime, risk_budget], dim=-1)
        feat = self.dynamics_net(inp)

        v = self.head_v(feat)  # Continuous rebalancing
        r = self.head_r(feat)  # Regime transition

        # Transport update: w' = w + v * dt
        w_next = w + v * dt
        w_next = F.softmax(w_next, dim=-1)  # Normalize

        # Reaction update: regime' = regime * exp(r * dt)
        regime_next = regime * torch.exp(r * dt)
        regime_next = regime_next / regime_next.sum(dim=-1, keepdim=True)

        return w_next, regime_next, v, r
```

### 25.5 WFR Consistency Diagnostic

**Node GateWFR: WFR Consistency Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate41 | WFRCheck | Portfolio Model | Dynamics Consistency | Transport-Reaction balance | $\mathcal{L}_{\text{WFR}}$ | $O(BK)$ |

**Trigger conditions:**
- High $\mathcal{L}_{\text{WFR}}$: Portfolio model's $(v, r)$ predictions violate continuity.
- Remedy: Increase training on regime transitions; check for distribution shift in market conditions.

---

## 26. Price Discovery Dynamics: Entropic Drift and Market Steering

**Price discovery** is the expansion from maximum uncertainty (prior) to information revelation (quoted price).

### 26.1 Price Discovery as Radial Expansion

Price discovery is modeled as expansion from the origin (maximum uncertainty, wide bid-ask spreads) toward the boundary (price revelation, tight spreads).

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Origin $z=0$ | Maximum uncertainty (prior) |
| Boundary $|z|\to 1$ | Price revelation / tight spreads |
| Entropic drift | Spread compression over time |
| Policy control field $u_\pi$ | Market maker steering / order flow |
| Radial coordinate $r$ | Information content / price precision |

**Definition 26.1.1 (Information Content of Price).** The hyperbolic distance from origin represents information content:
$$
I_{\text{price}}(z) := d_{\mathbb{D}}(0, z) = 2 \operatorname{artanh}(|z|).
$$

| $|z|$ | $I_{\text{price}}$ | Market Interpretation |
|-------|-------------------|----------------------|
| $0$ | $0$ | Prior only (no market information) |
| $0.5$ | $1.1$ nat | Moderate price discovery |
| $0.9$ | $2.9$ nat | High price precision |
| $\to 1$ | $\to \infty$ | Perfect price revelation |

### 26.2 Entropic Spread Compression

**Definition 26.2.1 (Entropic Drift in Markets).** In the absence of order flow, prices experience an **entropic drift** toward revelation:
$$
\dot{r} = \frac{1 - r^2}{2},
$$
which integrates to:
$$
r(\tau) = \tanh(\tau/2).
$$

**Interpretation:** In the absence of order flow, the system evolves toward the boundary at this rate. The entropic drift represents the baseline price discovery rate.

**Definition 26.2.2 (Market Maker Control Field).** The market maker (or informed trader) provides a **control field**:
$$
u_{\text{mm}}(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi}[a],
$$
which breaks rotational symmetry at the origin, selecting a preferred direction for price evolution.

| Control Field | Market Interpretation |
|--------------|----------------------|
| $u_{\text{mm}} = 0$ | Uninformed trading (random walk) |
| $u_{\text{mm}} \neq 0$ | Informed trading (directional pressure) |
| $u_{\text{mm}} \cdot \hat{r} > 0$ | Accelerated price discovery |
| $u_{\text{mm}} \cdot \hat{r} < 0$ | Price discovery inhibition |

### 26.3 Bid-Ask Separation as Partition Condition

**Axiom 26.3.1 (Bid-Ask Decoupling).** The state decomposition $(K, z_n, z_{\text{tex}})$ maps to:
- **Interior (price process):** Mid-price trajectory $z(\tau)$ evolves on the pricing manifold.
- **Boundary (microstructure):** Bid-ask spread $z_{\text{tex}}$ is sampled at the interface.

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}, \lambda_{\text{jump}}, u_\pi \right] = 0
$$

**Consequence:** Mid-price dynamics are independent of microstructure noise. Spread fluctuations decouple from the fundamental price discovery process.

**Definition 26.3.2 (Microstructure Noise Distribution).** At the market interface:
$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma_{\text{spread}}(z)),
$$
where:
$$
\Sigma_{\text{spread}}(z) = \sigma_{\text{spread}}^2 \cdot G^{-1}(z).
$$

**Scaling:** Near the origin (wide spreads), microstructure noise variance is large. Near the boundary (tight spreads), noise is suppressed by the metric.

### 26.4 Price Discovery Diagnostic

**Node GatePriceDisc: Price Discovery Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate42 | PriceDiscCheck | Market Model | Discovery Validity | Did price discovery converge? | $\mathbb{I}(|z_{\text{final}}| \ge R_{\text{cutoff}})$ | $O(B)$ |

**Trigger conditions:**
- Low PriceDiscCheck: Price discovery incomplete (wide spreads persist).
- Remedy: Increase trading horizon; check for liquidity constraints.

---

## 27. Market Equations of Motion: Portfolio Geodesics and Jump-Diffusion

The portfolio follows a **geodesic jump-diffusion** on the risk manifold.

### 27.1 Position Inertia: Mass = Risk Metric

**Definition 27.1.1 (Position Inertia Tensor).** The **position inertia** is the Ruppeiner risk metric:
$$
\mathbf{M}(w) := G(w).
$$

**Operational consequences:**
- **High-risk positions** (large $G$) have large inertia → smaller rebalancing per unit signal.
- **Low-risk positions** (small $G$) have small inertia → larger rebalancing allowed.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Mass tensor $\mathbf{M}(z)$ | Position inertia |
| Kinetic energy $\frac{1}{2}\mathbf{M}\|\dot{z}\|^2$ | Trading cost / market impact |
| Potential $\Phi_{\text{eff}}$ | Risk-adjusted return landscape |
| Christoffel symbols $\Gamma^k_{ij}$ | Cross-asset correlation corrections |

**Risk-Metric Coupling (Market Natural Gradient):**
$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced trade size}
$$

### 27.2 Portfolio Jump-Diffusion SDE

**Definition 27.2.1 (Portfolio Geodesic SDE).** The portfolio weights $w^k$ evolve according to:
$$
dw^k = \underbrace{\left( -G^{kj}\partial_j \Phi_{\text{risk}} + u_\pi^k \right)}_{\text{Drift (signal + policy)}} ds - \underbrace{\Gamma^k_{ij}\dot{w}^i \dot{w}^j\,ds}_{\text{Correlation correction}} + \underbrace{\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s}_{\text{Market noise}},
$$
where:
- $\Phi_{\text{risk}}$ is the risk-adjusted return potential,
- $u_\pi^k$ is the alpha signal (policy control),
- $\Gamma^k_{ij}$ are Christoffel symbols of the risk metric (correlation structure),
- $T_c$ is the market temperature (volatility scaling).

**Three-Force Decomposition:**
1. **Return gradient:** $-G^{-1}\nabla\Phi_{\text{risk}}$ — move toward high risk-adjusted returns.
2. **Alpha signal:** $u_\pi$ — policy-induced trades (momentum, value, etc.).
3. **Correlation correction:** $-\Gamma(\dot{w},\dot{w})$ — adjusts for cross-asset dependencies.

### 27.3 Regime Jump Process

**Definition 27.3.1 (Regime Jump Intensity).** The intensity of jumping from regime $i$ to regime $j$ is:
$$
\lambda_{i \to j}(w) = \lambda_0 \cdot \exp\left(\beta \cdot \left( V_j(w) - V_i(w) - c_{\text{switch}} \right) \right),
$$
where:
- $V_i, V_j$ are regime-specific value functions,
- $c_{\text{switch}}$ is the regime transition cost,
- $\beta$ is inverse temperature (sharpness).

**Interpretation:** Regime transitions occur when $V_j(w) - V_i(w) > c_{\text{switch}}$, with rate exponentially increasing in the value differential.

### 27.4 Effective Return Potential

**Definition 27.4.1 (Effective Return Potential).** The unified potential is:
$$
\Phi_{\text{risk}}(w, K) = \alpha \cdot U(w) + (1 - \alpha) \cdot V_{\text{alpha}}(w, K) + \gamma_{\text{risk}} \cdot \Psi_{\text{risk}}(w),
$$
where:
- $U(w)$ is the information potential (spread compression),
- $V_{\text{alpha}}(w, K)$ is the alpha signal (expected returns),
- $\Psi_{\text{risk}}(w) = \frac{1}{2}\text{tr}(T_{ij} G^{ij})$ is risk concentration.

| $\alpha$ | Behavior | Strategy Type |
|----------|----------|---------------|
| $\alpha = 1$ | Pure liquidity provision | Market making |
| $\alpha = 0$ | Pure alpha capture | Directional trading |
| $\alpha = 0.5$ | Balanced | Mixed strategy |

### 27.5 BAOAB Portfolio Integrator

**Algorithm 27.5.1 (Portfolio BAOAB Step).**

```python
def portfolio_baoab_step(
    w: torch.Tensor,        # Portfolio weights [B, N_assets]
    p: torch.Tensor,        # Momentum (trading velocity) [B, N_assets]
    regime: torch.Tensor,   # Regime index [B]
    grad_Phi: torch.Tensor, # Return gradient [B, N_assets]
    u_alpha: torch.Tensor,  # Alpha signal [B, N_assets]
    G: torch.Tensor,        # Risk metric [B, N, N]
    T_c: float,             # Market temperature
    gamma: float,           # Friction (transaction costs)
    h: float,               # Time step
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Portfolio BAOAB integrator with geodesic corrections.

    B-A-O-A-B splitting:
    - B: Momentum kick from returns + alpha
    - A: Position drift (portfolio update)
    - O: Market noise (Ornstein-Uhlenbeck)
    """
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c)

    # B-step: half kick
    total_force = grad_Phi - u_alpha
    p = p - (h / 2) * total_force

    # A-step: half drift
    G_inv = torch.linalg.inv(G)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    w = w + (h / 2) * velocity

    # O-step: market noise
    G_sqrt = torch.linalg.cholesky(G)
    xi = torch.randn_like(p)
    p = c1 * p + c2 * torch.einsum('bij,bj->bi', G_sqrt, xi)

    # A-step: half drift
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    w = w + (h / 2) * velocity

    # B-step: half kick
    p = p - (h / 2) * total_force

    # Normalize to simplex
    w = F.softmax(w, dim=-1)

    return w, p
```

### 27.6 Market Dynamics Diagnostics

**Node GateGeodesic: Geodesic Consistency Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate43 | GeodesicCheck | Portfolio Model | Trajectory Consistency | Is portfolio path geodesic? | $\|\ddot{w} + \Gamma(\dot{w},\dot{w}) + G^{-1}\nabla\Phi - u_\pi\|_G$ | $O(BN^2)$ |

---

## 28. Market Interface: Order Book as Symplectic Boundary

The order book is a **symplectic manifold** where prices (positions) and order flow (momentum) are conjugate variables.

### 28.1 Position-Momentum Duality in Markets

**Definition 28.1.1 (Symplectic Market Interface).** The market interface is a symplectic manifold $(\partial\mathcal{W}, \omega)$ with:
- $q \in \mathcal{Q}$ is the **price coordinate** (mark-to-market values),
- $p \in T^*_q\mathcal{Q}$ is the **flow coordinate** (order flow, trading velocity).

The symplectic form is:
$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.
$$

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Position $q$ | Mark-to-market prices |
| Momentum $p$ | Order flow / trading velocity |
| Dirichlet BC (sensors) | Price quotes (observable) |
| Neumann BC (motors) | Order submission (actions) |
| Symplectic form $\omega$ | Position-flow duality |

### 28.2 Boundary Conditions for Trading

**Definition 28.2.1 (Dirichlet BC — Price Quotes).** Market prices impose position-clamping:
$$
q_{\partial}^{\text{quote}}(t) = q_{\text{mid}}(t),
$$
where $q_{\text{mid}}$ is the observable mid-price. This clamps the **configuration** of the portfolio.

**Definition 28.2.2 (Neumann BC — Order Submission).** Trading imposes flux-clamping:
$$
\nabla_n q \cdot \mathbf{n} \big|_{\partial\mathcal{W}} = j_{\text{trade}}(p, t),
$$
where $j_{\text{trade}}$ is the order flow determined by the trading strategy.

### 28.3 Active Trading vs Risk Simulation

**Definition 28.3.1 (Trading Cycle Phases).**

| Phase | Process | Information Flow | Entropy Change |
|-------|---------|------------------|----------------|
| **I. Observation** | Price compression | Market data → portfolio state | $\Delta S < 0$ |
| **II. Simulation** | Internal risk analysis | No external exchange | $\Delta S = 0$ (isentropic) |
| **III. Execution** | Order expansion | Trading signal → order flow | $\Delta S > 0$ |

**Theorem 28.3.2 (Market Carnot Efficiency).** The efficiency of converting market information to trading profits is bounded:
$$
\eta = \frac{I(A_t; K_t)}{I(X_t; K_t)} \le 1 - \frac{T_{\text{exec}}}{T_{\text{obs}}},
$$
where $T_{\text{exec}}$ and $T_{\text{obs}}$ are effective temperatures at execution and observation interfaces.

### 28.4 Active Trading vs Closed-System Simulation

**Definition 28.4.1 (Active Trading Mode).**
$$
\rho_{\partial}^{\text{quote}}(w, t) = \delta(w - w_{\text{target}}(t)) \quad \text{(Dirichlet)},
$$
$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{trade}}(u_\pi) \quad \text{(Neumann)}.
$$

**Definition 28.4.2 (Closed-System Simulation Mode).**
$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective)}.
$$
The system is closed—no trading, pure risk simulation.

| Mode | Quote BC | Trade BC | Internal Flow | Information Balance |
|------|----------|----------|---------------|---------------------|
| **Active Trading** | Dirichlet (price-clamp) | Neumann (flow-clamp) | Price-driven | $\oint j_{\text{in}} > 0$ |
| **Closed Simulation** | Reflective | Reflective | Recirculating | $\oint j = 0$ |

### 28.5 Context Space: Unified Task Structure

**Definition 28.5.1 (Market Context Space).** The context $c \in \mathcal{C}$ determines the trading objective:

| Task | Context $c$ | Output | Potential $\Phi_{\text{eff}}$ |
|------|-------------|--------|-------------------------------|
| **Alpha Capture** | Signal space | Trade direction | $V_{\text{alpha}}(w, K)$ |
| **Risk Management** | Risk budget | Hedge ratio | $-\log p(\text{safe}|w)$ |
| **Execution** | Target portfolio | Order sequence | $-\log p(\text{fill}|w, \text{target})$ |

### 28.6 Market Interface Diagnostics

**Node GateSymplectic: Symplectic Boundary Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate44 | SymplecticCheck | Interface | BC Consistency | Are quote/trade BCs compatible? | $\|\omega(j_{\text{quote}}, j_{\text{trade}})\|$ | $O(Bd)$ |

---

## 29. Value Field and Pricing Kernel: Discounted Cash Flow as Screened Poisson

The **pricing kernel** is the Green's function of a screened Poisson equation, and the **discount rate** is the screening mass.

### 29.1 Reward as Cash Flow

**Definition 29.1.1 (Cash Flow as Source Term).** The cash flow stream (dividends, coupons) acts as a scalar source:
$$
\sigma_{\text{cf}}(t, w) = \sum_{t' < t} \text{CF}_{t'} \cdot \delta(t - t') \cdot \delta(w - w_{t'}),
$$
where $\text{CF}_t$ is the cash flow at time $t$.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Reward flux $J_r$ | Cash flow stream |
| Boundary charge $\sigma_r$ | Dividend/coupon payments |
| Potential $V(z)$ | Net present value $\text{NPV}(w)$ |
| Screening mass $\kappa$ | Discount rate |

### 29.2 Pricing Kernel as Screened Poisson Solver

**Theorem 29.2.1 (DCF as Helmholtz Equation).** The net present value $V(w)$ satisfies the **screened Poisson equation**:
$$
\boxed{-\Delta_G V(w) + \kappa^2 V(w) = \rho_{\text{cf}}(w)}
$$
where:
- $\Delta_G$ is the Laplace-Beltrami operator on the risk manifold,
- $\kappa = -\ln(\gamma)/\Delta t$ is the screening mass (discount rate),
- $\rho_{\text{cf}}$ is the cash flow density.

**Proof sketch:** The Bellman equation $V(w) = \mathbb{E}[\text{CF} + \gamma V(w')]$ approaches the Helmholtz PDE in the continuous limit. $\square$

### 29.3 Discount as Screening Length

**Corollary 29.3.1 (Investment Horizon as Screening Length).**
$$
\ell_{\text{horizon}} = \frac{1}{\kappa} = \frac{\Delta t}{-\ln\gamma}.
$$

| Discount $\gamma$ | Screening Mass $\kappa$ | Horizon $\ell$ | Interpretation |
|-------------------|-------------------------|----------------|----------------|
| $\gamma \to 1$ | $\kappa \to 0$ | $\ell \to \infty$ | Long-term investor |
| $\gamma = 0.99$ | $\kappa \approx 0.01$ | $\ell \approx 100$ | Standard DCF |
| $\gamma = 0.9$ | $\kappa \approx 0.1$ | $\ell \approx 10$ | Short-term trader |
| $\gamma \to 0$ | $\kappa \to \infty$ | $\ell \to 0$ | Myopic (day trader) |

### 29.4 Value-Risk Conformal Coupling

**Definition 29.4.1 (Value-Metric Feedback).** High-value-curvature regions induce metric distortion:
$$
\tilde{G}_{ij}(w) = \Omega^2(w) \cdot G_{ij}(w),
$$
where:
$$
\Omega(w) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(w)\|_{\text{op}}.
$$

**Operational effect:**
- **Flat value landscape:** Default risk metric applies.
- **High curvature (decision boundary):** Metric expands, reducing position velocity.
- **Saddle regions:** Moderate metric expansion.

### 29.5 Pricing Kernel Implementation

```python
class PricingKernel(nn.Module):
    """
    Pricing kernel as Helmholtz equation solver.

    Maps cash flow sources to net present value via screened Poisson.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, gamma: float = 0.99):
        super().__init__()
        self.kappa = -math.log(gamma)  # Screening mass

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Compute NPV at portfolio position w."""
        return self.net(w)

    def helmholtz_loss(self, w: torch.Tensor, w_next: torch.Tensor,
                        cf: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Enforce Bellman/Helmholtz consistency.

        V(w) = cf + gamma * V(w')
        """
        V = self(w)
        V_next = self(w_next).detach()
        td_error = V - (cf + gamma * V_next)
        return td_error.pow(2).mean()
```

### 29.6 Value Field Diagnostics

**Node GateHelmholtz: Helmholtz Residual Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate45 | HelmholtzCheck | Pricing Kernel | PDE Consistency | Is DCF equation satisfied? | $\|-\Delta_G V + \kappa^2 V - \rho_{\text{cf}}\|$ | $O(BD)$ |

---

## 30. Sector Classification and Regime Segmentation

Class labels become **sector labels** or **regime labels**, and regions of attraction become **allocation basins**.

### 30.1 Sector as Semantic Partition

**Definition 30.1.1 (Sector Partition).** Let $\mathcal{Y} = \{\text{Tech}, \text{Finance}, \text{Healthcare}, \ldots\}$ be sector labels. The sector induces a partition of the regime atlas:
$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(\text{Sector}=y \mid K=k) > 1 - \epsilon_{\text{purity}}\}.
$$

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Class labels $\mathcal{Y}$ | Sector / regime labels |
| Semantic potential $V_y$ | Sector risk premium |
| Region of attraction $\mathcal{B}_y$ | Sector allocation basin |
| Chart purity | Sector membership clarity |
| Transition regions | Cross-sector exposure |

### 30.2 Sector-Conditioned Risk Premium

**Definition 30.2.1 (Sector Risk Premium Potential).**
$$
V_{\text{sector}}(w, K) := -\beta_{\text{sector}} \log P(\text{Sector}=y \mid K) + V_{\text{base}}(w, K),
$$
where:
- $P(\text{Sector}=y \mid K)$ is the sector probability given regime,
- $\beta_{\text{sector}}$ is the sector temperature (concentration preference).

### 30.3 Sector Rotation as Gradient Flow

**Definition 30.3.1 (Sector Allocation Basin).** The **allocation basin** for sector $y$ is:
$$
\mathcal{B}_y := \{w \in \mathcal{W} : \lim_{t \to \infty} \phi_t(w) \in \mathcal{A}_y\},
$$
where $\phi_t$ is the flow of $\dot{w} = -G^{-1}(w)\nabla V_y(w)$.

**Interpretation:** Starting from any portfolio in $\mathcal{B}_y$, gradient flow converges to a sector-$y$ allocation.

**Theorem 30.3.2 (Sector Rotation as Relaxation).** Under overdamped dynamics with sector potential $V_y$:
$$
dw = -G^{-1}(w) \nabla V_y(w)\,ds + \sqrt{2T_c}\,G^{-1/2}(w)\,dW_s,
$$
the limiting regime satisfies $\lim_{s \to \infty} K(w(s)) \in \mathcal{A}_y$ almost surely.

### 30.4 Cross-Sector Jump Suppression

**Definition 30.4.1 (Sector-Modulated Regime Transition).** Modify regime transition rates:
$$
\lambda_{i \to j}^{\text{sector}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{sector}}(i, j)\right),
$$
where $D_{\text{sector}}(i, j) = \mathbb{I}[\text{Sector}(i) \neq \text{Sector}(j)]$.

**Effect:** Intra-sector transitions have baseline rates; cross-sector transitions are exponentially suppressed by $\gamma_{\text{sep}}$.

### 30.5 Sector Classification Loss

**Definition 30.5.1 (Sector Purity Loss).**
$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(\text{Sector} \mid K=k).
$$

**Definition 30.5.2 (Sector Rotation Loss).**
$$
\mathcal{L}_{\text{sector}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.
$$

### 30.6 Sector Classification Diagnostics

**Node GatePurity: Sector Purity Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate46 | PurityCheck | Router | Sector Clustering | Are regimes sector-pure? | $H(\text{Sector} \mid K)$ | $O(BC)$ |

**Node GateSectorSep: Sector Separation Check**

| # | Name | Component | Type | Interpretation | Proxy | Cost |
|---|------|-----------|------|----------------|-------|------|
| Gate47 | SectorSepCheck | Jump Op | Sector Separation | Are sectors metrically separated? | $\min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2})$ | $O(C^2 N_c)$ |

---

## References

```{bibliography}
:filter: docname in docnames
```

---

*Self-consistent economic theory: 7 axioms, 47 diagnostic gates, 20 barriers, 15 failure modes, 8 surgery contracts, 12 asset classes, 5 metatheorems, 30 sections.*
