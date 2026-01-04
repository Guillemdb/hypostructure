---
title: "The Fragile Agent: Bounded-Rationality Control and Information Geometry"
---

# The Fragile Agent: Bounded-Rationality Control and Information Geometry

## 0. Positioning: Connections to Prior Work, Differences, and Advantages

This document is a **synthesis and engineering specification** for building agents that remain stable, grounded, and debuggable under partial observability and finite capacity. Most mathematical ingredients are standard in **safe RL**, **robust control**, **information geometry**, **representation learning**, and **Bayesian filtering**. The contribution is to make the dependencies *explicit* and to provide a set of **online-auditable contracts** (Gate Nodes + Barriers) that connect representation, dynamics, value, and control.

### 0.1 Main Advantages (Why This Framing Is Useful)

1. **Online auditability.** Constraints are stated in quantities you can compute during training/inference (entropies, KLs, value gradients, stability inequalities), not only as “eventual performance”.
2. **Explicit macro-state abstraction.** The discrete macro register $K_t$ makes sufficiency, capacity, and closure conditions well-typed and testable (Sections 2.2b, 2.8, 3, 15).
3. **Predictive vs nuisance separation.** The split $(K_t, z_{\mu,t})$ prevents the world model and policy from silently depending on high-variance residuals (Sections 2.2b, 9).
4. **Geometry-aware regulation.** A state-space sensitivity metric $G$ is used as a runtime trust-region / conditioning signal (Sections 2.5–2.6, 9.10), complementing standard natural-gradient methods {cite}`amari1998natural,schulman2015trpo,martens2015kfac`.
5. **Safety as a first-class interface contract.** “Safety” is not a single scalar constraint: it decomposes into explicit checks (switching limits, capacity limits, saturation, grounding, mixing) with known compute cost (Sections 3–8).

### 0.2 What Is Novel Here vs What Is Repackaging

**Novel (as a combined framework / spec):**
1. **A discrete macro register used as the control-relevant state.** VQ-style discretization is treated as the *enabler* for audit-friendly information constraints (closure, capacity, window conditions), not just a compression trick (Sections 2.2b, 2.8, 15).
2. **The Sieve as an explicit catalog of monitors and limits.** Gate Nodes + Barriers are presented as a concrete interface between theory and implementation: “what to measure”, “what to penalize”, and “what to halt on” (Sections 3–6).
3. **Coupling-window operationalization.** The grounding/mixing window is stated directly in measurable information rates (Theorem 15.1.3), turning “stability/grounding” into an online diagnostic rather than a post-hoc story.
4. **A single notation tying representation, filtering, and control.** The same objects ($K_t$, $\bar P$, $V$, $G$, KL-control) appear consistently across the loop, reducing category errors between “learning” and “control” (Sections 2, 11, 13).

**Repackaging (directly inherited ingredients):**
- **POMDP/belief-control viewpoint:** partial observability, belief updates, and control on internal state {cite}`kaelbling1998planning,rabiner1989tutorial`.
- **Entropy-regularized / KL-regularized control:** soft Bellman objectives, KL-control, and exponential-family optimal policies {cite}`todorov2009efficient,kappen2005path,haarnoja2018soft`.
- **World-model based RL:** learning latent dynamics for planning/control (e.g., Dreamer-like latent rollouts) {cite}`hafner2019dreamer,ha2018worldmodels`.
- **Representation learning primitives:** VQ-VAE, InfoNCE/CPC, VICReg/Barlow-type collapse prevention {cite}`oord2017vqvae,oord2018cpc,bardes2022vicreg,zbontar2021barlow`.
- **Safe RL and constrained optimization:** Lyapunov-style constraints and constrained policy updates {cite}`chow2018lyapunov,berkenkamp2017safe,altman1999constrained,achiam2017constrained`.

### 0.3 Comparison Snapshot (Where This Differs in Practice)

| Area | Typical baseline | Fragile Agent difference |
|---|---|---|
| **Model-free RL** | optimize return; debugging via reward curves | adds explicit monitors and stop/penalty mechanisms tied to identifiable failure modes (Sections 3–6) |
| **World models** | continuous latent rollouts; implicit “state” | enforces a discrete macro state with closure and capacity checks; micro residual is treated as nuisance (Sections 2.2b, 2.8, 9, 15) |
| **Safe RL / CMDPs** | few scalar constraints (expected cost) | uses a *vector* of auditable constraints (grounding, mixing, saturation, switching, stiffness) with compute-cost accounting (Sections 3–8) |
| **Info bottleneck RL** | compression via an information penalty | makes the bottleneck operational via $\log|\mathcal{K}|$, $H(K)$, $I(X;K)$ and closure, not only via a single Lagrange term (Sections 2.2b, 3, 15) |
| **Natural gradient / trust region** | parameter-space Fisher metric | emphasizes state-space sensitivity $G$ as a runtime regulator of updates and checks (Sections 2.5–2.6, 9.10) |

**Reading guide (connections by section).**
- Representation + abstraction: Sections 2.2b, 2.8, 9.7–9.9
- Safety monitors and limits: Sections 3–6
- Filtering + projection (belief evolution): Section 11
- Entropy-regularized control + exploration: Sections 10–14
- Coupling window and capacity constraints: Sections 15 and 17

## 1. Introduction: The Agent as a Bounded-Rationality Controller

This document presents the **Fragile** interpretation of the Hypostructure: a deployed agent is a persistent **controller under partial observability** whose competence is bounded by (i) finite sensing/communication bandwidth, (ii) finite internal memory/representation capacity, and (iii) finite compute for inference and planning.

The framework is stated strictly in **information theory, optimization, and control**: discrete/continuous latent state construction (representation), stability constraints (Lyapunov-style), and capacity/sufficiency conditions (information bottlenecks).

This is the native language of **Safe RL**, **Robust Control**, and **Embodied AI**.

## 1.1 Definitions: Interaction Under Partial Observability

In the Fragile Agent framework, we do **not** treat the environment as a passive data provider. We treat the agent as a **partially observed control problem** whose only coupling to the external world is through a well-defined **interface / Markov blanket**. All RL primitives are re-typed as **signals and constraints at this interface**.

### 1.1.1 The Environment is an Input–Output Law (Not an Internal Object)

**Definition 1.1.1 (Bounded-Rationality Controller).** The agent is a controller with internal state
$$
Z_t := (K_t, Z_{\mu,t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_\mu,
$$
and internal components (Encoder/Shutter, World Model, Critic, Policy). Its evolution is driven only by the observable interaction stream at the interface (observations/feedback) and by its own outgoing control signals (actions).

**Definition 1.1.2 (Boundary / Markov Blanket).** The boundary variables at time $t$ are the interface tuple
$$
B_t := (x_t,\ r_t,\ d_t,\ \iota_t,\ a_t),
$$
where:
- $x_t\in\mathcal{X}$ is the observation (input sample),
- $r_t\in\mathbb{R}$ is reward/utility (scalar feedback; equivalently negative instantaneous cost),
- $d_t\in\{0,1\}$ is termination (absorbing event / task boundary),
- $\iota_t$ denotes any additional side channels (costs, constraints, termination reasons, privileged signals),
- $a_t\in\mathcal{A}$ is action (control signal sent outward).

**Definition 1.1.3 (Environment as Generative Process).** The “environment” is the conditional law of future interface signals given past interface history. Concretely it is a (possibly history-dependent) kernel on incoming boundary signals conditional on outgoing control:
$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_{\le t}, a_{\le t}).
$$
In the Markov case this reduces to the familiar RL kernel
$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_t, a_t),
$$
but the **interpretation changes**: $P_{\partial}$ is not “a dataset generator”; it is the **input–output law** that the controller must cope with under partial observability and model mismatch.

This is the categorical move: we do not assume access to the environment’s latent variables; we work only with the **law over observable interface variables**.

### 1.1.2 Re-typing Standard RL Primitives as Interface Signals

1. **Environment (Stochastic Process / Unmodeled Disturbance).**
   - *Standard:* a black box providing states and rewards.
   - *Fragile:* a high-dimensional, partially observed stochastic process; only its induced interface law $P_{\partial}$ is accessible to the agent.
   - *Role:* supplies observations and feedback signals; may be non-stationary, adversarial, or only approximately Markov.

2. **Observation $x_t$ (Observation Stream).**
   - *Standard:* an input tensor.
   - *Fragile:* the only exogenous input available to the controller. The encoder/shutter transduces it into internal coordinates:
     $$
     x_t \mapsto (K_t, Z_{\mu,t}),
     $$
     where $K_t$ is the **discrete predictive signal** (bounded-rate latent statistic) and $Z_{\mu,t}$ is the **continuous nuisance residual** (compression artifacts / observation noise).
   - *Boundary gate nodes (Section 3):*
     - **Node 14 (InputSaturationCheck):** input saturation (sensor dynamic range exceeded).
     - **Node 15 (SNRCheck):** low signal-to-noise (SNR too low to support stable inference).
     - **Node 13 (BoundaryCheck):** the channel is open in the only well-typed sense: $I(X;K)>0$ (symbolic mutual information).

3. **Action $a_t$ (Control / Actuation).**
   - *Standard:* a vector sent to the environment.
   - *Fragile:* a control signal chosen to minimize expected future cost under uncertainty and constraints.
   - *Cybernetic constraints:*
     - **Node 2 (ZenoCheck):** limits chattering (bounded variation in control outputs).
     - **BarrierSat:** actuator saturation (finite control authority).

4. **Reward $r_t$ (Utility / Negative Cost Signal).**
   - *Standard:* a scalar to maximize.
   - *Fragile:* a scalar feedback signal used to define the control objective. In continuous-time derivations it appears as an instantaneous **cost rate**; in discrete time it appears as an incremental term in the Bellman/HJB consistency relation (Section 2.7).
   - *Mechanism:* the critic’s $V$ is the internal value/cost-to-go; reward provides the task-aligned signal shaping $V$.

5. **Termination $d_t$ (Absorbing Boundary Event).**
   - *Standard:* end-of-episode flag.
   - *Fragile:* an absorbing event: the trajectory has entered a terminal region (failure/success) or exited the modeled domain. It is part of the task specification, not a training artifact.

6. **Episode / Rollout (Finite-Horizon Segment).**
   - *Standard:* a finite trajectory segment.
   - *Fragile:* a finite window used to estimate a cumulative objective under uncertainty. “Success” is satisfying task constraints while maintaining stability; “failure” is crossing a monitored limit (Section 4).

## 1.2 Units and Dimensional Conventions (Explicit)

This document expresses objectives in **information units** so that likelihoods, code lengths, KL terms, and entropy regularizers share a common scale.

**Base units.**
- Time is measured in **environment steps**. If a physical clock is needed, introduce $\Delta t$ with $[\\Delta t]=\mathrm{s}$.
- Information: entropies/information/KL are in **nats** (dimensionless but tracked): $[H]=[S]=[I]=[D_{KL}]=\mathrm{nat}$.
- Costs / values / losses (including $V$ and negative rewards) are measured in **nats**: $[V]=\mathrm{nat}$.
- Cost rates are measured in $\mathrm{nat/step}$ (or $\mathrm{nat\,s^{-1}}$ after dividing by $\Delta t$).

**Discrete vs continuous reward.**
- Per-step reward $r_t$ (or cost $c_t=-r_t$) has units $\mathrm{nat}$.
- A continuous-time cost rate $\mathcal{R}$ has units $\mathrm{nat\,s^{-1}}$ and links to discrete time by $r_t \approx \int_{t}^{t+\Delta t}\mathcal{R}(\tau)\,d\tau$.

**Regularization / precision coefficients.**
- MaxEnt / entropy-regularized control introduces a trade-off coefficient (often written $T_c$ or $\alpha_{\text{ent}}$) multiplying an entropy term. Because entropy is in nats, this coefficient is dimensionless and simply sets relative weight in the objective.
- Exponential-family (softmax/logit) policies use a precision parameter $\beta$ so that $\exp(\beta\,\cdot)$ is dimensionless. Here $\beta$ is dimensionless and interpretable as an inverse-variance / “sharpness” control knob.

**Conventions for generic coefficients.**
- Numerical stabilizers like $\epsilon$ always inherit the units of the quantity they are added to.
- Composite-loss weights (e.g. $\lambda_{\text{*}}$ used to sum training losses) are taken dimensionless unless explicitly stated otherwise.

---

## 2. The Control Loop: Representation and Control

We frame the agent as a **bounded controller** operating on an internal latent state space: it must learn a representation, learn/predict dynamics, and choose actions under uncertainty and capacity limits.

### 2.1 The Objective: Optimal Control Under Information Constraints

We write the objective as a cumulative cost functional $\mathcal{S}$ (expected loss plus regularization) over time:
$$
\mathcal{S}
=
\int \Big(
\underbrace{\mathcal{L}_{\text{control}}}_{\text{control cost / regularization}}
\;+\;
\underbrace{C(z_t,a_t)}_{\text{task cost}}
\Big)\,dt.
$$
Operationally, $\mathcal{L}_{\text{control}}$ can be a KL control penalty (deviation from a prior policy), action magnitude cost, or any control-effort term; $C$ encodes task loss and constraints.

**Relation to prior work.** This objective family covers standard stochastic optimal control and entropy-/KL-regularized RL: KL-control and “soft” (entropy-regularized) Bellman objectives are special cases {cite}`todorov2009efficient,kappen2005path,haarnoja2018soft`.

### Anatomy: The Fragile Agent (Core Architecture)
The Fragile Agent architecture used throughout this document is the tuple $\mathbb{A}$:
$$ \mathbb{A} = (\text{Split VQ-VAE Shutter}, \text{World Model}, \text{Critic}, \text{Policy}) $$

This tuple directly instantiates the core objects of the Hypostructure $\mathbb{H} = (\mathcal{X}, \nabla, \Phi)$:

| Component | Hypostructure Map | Role (Mechanism) | Cybernetic Function |
| :--- | :--- | :--- | :--- |
| **Autoencoder (Split VQ-VAE)** | **State Space Construction ($\mathcal{X}$)** | **Information Bottleneck Encoder:** maps $x \mapsto (K, z_{\mu})$ where $K$ is a *discrete predictive latent* and $z_{\mu}$ is a *continuous residual/nuisance latent*. | Defines the representation used for prediction and control. |
| **World Model** | **Dynamics Model ($\nabla, S_t$)** | **Predictive Model:** simulates/learns latent dynamics to support planning and counterfactual evaluation. | Defines the learned transition structure within $Z$. |
| **Critic** | **Value/Cost Functional ($\Phi$)** | **Value Function:** assigns a scalar cost-to-go/value to points in $Z$, representing risk/undesirability. | Defines the gradient signal $\nabla V$. |
| **Policy** | **Control Regularization ($\mathfrak{D}$)** | **Controller (Policy):** chooses actions that reduce expected future cost subject to constraints and regularization. | Implements the control law minimizing $\mathcal{S}$. |

### 2.2a The Trinity of Manifolds (Dimensional Alignment)

To prevent category errors, we formally distinguish three manifolds with distinct geometric structures:

| Manifold | Symbol | Coordinates | Metric Tensor | Role |
|----------|--------|-------------|---------------|------|
| **Physical/Data** | $\mathcal{X}$ | $x \in \mathbb{R}^D$ | $I$ (Euclidean) | Raw observations—the "hardware" |
| **Latent/Problem** | $\mathcal{Z}=\mathcal{K}\times \mathcal{Z}_{\mu}$ | $(K, z_{\mu})$ with $K \in \mathcal{K}$, $z_{\mu}\in\mathbb{R}^{d_\mu}$ | $d_{\mathcal{K}} \oplus G_{\mu}(z_{\mu};K)$ | Symbolic macro-register + continuous latent manifold—the "representation" |
| **Parameter/Model** | $\Theta$ | $\theta \in \mathbb{R}^P$ | $\mathcal{F}(\theta)$ (Fisher-Rao) | Configuration space—the "weights" |

**Dimensional Verification:**

- The shutter $E: \mathcal{X} \to \mathcal{K}\times \mathcal{Z}_\mu$ is a **contraction** in continuous degrees of freedom ($d_\mu \ll D$) plus a **finite-rate quantization** in the macro channel ($\log|\mathcal{K}|$ bits).
- The policy $\pi_\theta$ is a map on macrostates (or their code embeddings) $\pi_\theta:\mathcal{K}\to\mathcal{A}$; dependence on $z_\mu$ is a *causal leak* (violates enclosure).
- The metric $G_{\mu}$ is defined on the **continuous fibres** $\mathcal{Z}_\mu$ (and/or the induced geometry of the codebook embedding), not on $\Theta$.

**Anti-Mixing Principle:** Never conflate $\mathcal{F}(\theta)$ (parameter-space Fisher) with $G(z)$ (state-space metric). They live on different manifolds and measure different quantities:
- $\mathcal{F}(\theta)$: How the policy changes with weights (used by TRPO/PPO)
- $G_\mu(z_\mu;K)$: How the policy changes with continuous state coordinates (used by the Fragile Agent)

### 2.2b The Shutter as a VQ-VAE (Discrete Macro, Continuous Micro)

The information-theoretic/control interpretation benefits from an explicit **discrete latent register**: a countable set of macrostates on which we can apply Shannon/algorithmic statements without differential-entropy ambiguities. This is provided by a **VQ-VAE macro-encoder**.

We factor the latent as:
$$
Z_t := (K_t, Z_{\mu,t}), \qquad K_t \in \mathcal{K}\ \text{(discrete)}, \quad Z_{\mu,t}\in\mathcal{Z}_\mu\subset \mathbb{R}^{d_\mu}\ \text{(continuous)}.
$$

**Macro codebook (symbols).** Let $\mathcal{K}=\{1,\dots,|\mathcal{K}|\}$ and let $\{e_k\}_{k\in\mathcal{K}}\subset\mathbb{R}^{d_m}$ be a learned codebook. Given an observation $x$ the macro encoder produces a pre-quantized vector $z_e(x)\in\mathbb{R}^{d_m}$ and the VQ projection chooses the nearest code:
$$
K(x) := \arg\min_{k\in\mathcal{K}} \|z_e(x)-e_k\|_2^2,
\qquad
z_{\text{macro}}(x):=e_{K(x)}.
$$
We equip $\mathcal{K}$ with the induced finite metric $d_{\mathcal{K}}(k,k'):=\|e_k-e_{k'}\|_2$ (or its $G_\mu$-weighted variant), so $\mathcal{Z}$ is a bundle of continuous fibres over a discrete base.

**Micro residual (continuous nuisance residual).** The micro channel remains continuous, e.g. with a Gaussian posterior
$$
q_\phi(z_\mu\mid x)=\mathcal{N}(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))),
$$
and a simple prior $p(z_\mu)=\mathcal{N}(0,I)$ so that matching $q_\phi(\cdot\mid x)$ to $p$ operationalizes the statement “$z_\mu$ carries residual variation, not predictive structure.”

**VQ-VAE objective (with continuous micro).** With a decoder $p_\theta(x\mid z_{\text{macro}},z_\mu)$ and stop-gradient operator $\operatorname{sg}[\cdot]$, the canonical loss is
$$
\mathcal{L}_{\text{shutter}}
=
\mathbb{E}\Big[-\log p_\theta(X\mid e_{K(X)}, Z_\mu)\Big]
+\underbrace{\| \operatorname{sg}[z_e]-e_{K}\|_2^2}_{\text{codebook}}
+\underbrace{\beta\|z_e-\operatorname{sg}[e_K]\|_2^2}_{\text{commit}}
+\underbrace{\beta_\mu D_{KL}(q_\phi(Z_\mu\mid X)\Vert p(Z_\mu))}_{\text{micro-as-noise}}.
$$
Units: $\beta$ and $\beta_\mu$ are dimensionless weights; $D_{KL}$ is in nats.

**Information-theoretic capacity becomes explicit.** Because $K$ is discrete,
$$
I(X;K)\le H(K)\le \log|\mathcal{K}|.
$$
For deterministic nearest-neighbor quantization, $H(K\mid X)=0$ and hence $I(X;K)=H(K)$: the macro channel is literally a bounded-rate symbolic memory.

**Rate–distortion / MDL viewpoint (optional but canonical).** Because $K$ is a discrete string, an explicit entropy model $p_\psi(K)$ defines a literal expected codelength $\mathbb{E}[-\log p_\psi(K)]$ (in nats). Adding this term yields the Lagrangian form of lossy source coding:
$$
\min_{E,D}\ \mathbb{E}\big[d(X,\hat{X})\big] + \lambda\,\mathbb{E}\big[-\log p_\psi(K)\big],
$$
with rate controlled by the symbolic model and distortion controlled by the decoder. This is the rigorous information-theoretic envelope in which VQ-VAE-style macrostates live.
Units: $\lambda$ has units of the distortion scale divided by nats (dimensionless if $d$ is itself a negative-log-likelihood in nats).

**Metatheorems unlocked by discretization.**
- A discrete macro-register makes coding-theoretic and finite-memory update bounds applicable.
- Macro-trajectories become literal strings $K_{0:T}$, so Levin/Kolmogorov-style horizon arguments become well-typed (see {prf:ref}`mt:levin-search`).

### 2.3 The Bridge: RL as Lyapunov-Constrained Control (Neural Lyapunov Geometry)

Standard Reinforcement Learning maximizes expected return. Robust control enforces stability by requiring a Lyapunov-like decrease condition. We bridge these by treating the learned critic $V(s)$ as a **Control Lyapunov Function (CLF)** {cite}`chang2019neural` and shaping policy improvement to respect stability constraints.

| Perspective | Objective | Mechanism |
|-------------|-----------|-----------|
| **Optimization** | Minimize expected cost-to-go $V(s)$ | Gradient-based policy/value updates |
| **Control Theory** | Ensure stability: $\dot{V}(s) \le -\lambda V(s)$, $[\lambda]=\mathrm{step^{-1}}$ | Lyapunov constraint |
| **Reinforcement Learning** | Improve value estimates/policy | TD learning + policy gradients |

The key insight is that these perspectives align around the same mathematical objects: a scalar value/cost-to-go function and constraints on how fast it can improve without destabilizing the loop.

### 2.4 A Geometry-Regularized Objective (Natural-Gradient View)

We can write an update objective that combines (i) a geometry-aware smoothness/trust-region penalty and (ii) a value-improvement term:

$$\mathcal{S} = \int \left( \underbrace{\frac{1}{2} \|\dot{\pi}\|^2_{G}}_{\text{Update smoothness / trust region}} - \underbrace{\frac{d V}{d \tau}}_{\text{Value improvement}} \right) dt$$

Where $\|\cdot\|_G$ is the norm under a **state-space sensitivity metric** $G$ (Section 2.5). This biases updates toward paths that are conservative in sensitive regions and more aggressive where the value landscape is well-conditioned.

**Comparison: Euclidean vs Geometry-Aware Updates**

| Aspect | Euclidean (Standard RL) | Geometry-aware (Fragile Agent) |
|--------|-------------------------|----------------------------|
| **Metric** | $\|\cdot\|_2$ (flat) | $\|\cdot\|_G$ (curved) |
| **Step Size** | Constant everywhere | Varies with curvature |
| **Near ill-conditioned regions** | Large steps → instability | Small steps → safety |
| **In Valleys** | Same as cliffs | Large steps → efficiency |
| **Failure Mode** | BarrierBode (oscillation) | Prevented by geometry |

### 2.5 Second-Order Sensitivity: Value Defines a Local Metric

In information geometry and second-order optimization, a local metric captures how sensitive the objective and the policy are to changes in state coordinates {cite}`amari1998natural`. For the Fragile Agent, we define a state-space sensitivity metric $G_{ij}$ using curvature of the critic/value function:

$$G_{ij} = \frac{\partial^2 V}{\partial z_i \partial z_j} = \text{Hess}(V)$$
Units: $[G_{ij}]=\mathrm{nat}\,[z]^{-2}$ if $z$ is measured in units $[z]$.

**Behavior in Different Regions:**

* **Flat Region ($G \approx I$):** The Value function is linear/flat. Risk is uniform. The space is Euclidean.
* **Curved Region ($G \gg I$):** The value function is sharply curved (ill-conditioned). The metric "stretches" space: a small Euclidean step can correspond to a large change in value/sensitivity.

**The Upgrade: From Gradient Descent to Geodesic Flow**

| Standard RL | Riemannian RL |
|-------------|---------------|
| $\theta \leftarrow \theta + \eta \nabla_\theta \mathcal{L}$ | $\theta \leftarrow \theta + \eta G^{-1} \nabla_\theta \mathcal{L}$ |
| Euclidean gradient | Natural gradient (Amari) |
| Ignores curvature | Respects curvature |

In the Fragile Agent implementation, the **Riemannian metric lives in state space**, not parameter space. The covariant derivative uses a **diagonal inverse metric** $M^{-1}(s)$ to scale $\dot{V}$:

$$\dot{V}_M = \nabla V(s)^\top M^{-1}(s) \Delta s$$

Current state-space metric options (diagonal approximations):

* **Observation variance (whitening):**
  $$M^{-1}_{ii}(s) = \frac{1}{\mathrm{Var}(s_i) + \epsilon}$$
* **Policy Fisher on states:**
  $$M^{-1}_{ii}(s) = \frac{1}{\mathbb{E}[(\partial_{s_i}\log \pi(a|s))^2] + \epsilon}$$
* **Gradient RMS (critic):**
  $$M^{-1}_{ii}(s) = \frac{1}{\sqrt{\mathbb{E}[(\partial_{s_i} V)^2]} + \epsilon}$$

In all three cases, $\epsilon$ is a numerical stabilizer with the **same units as the denominator term** it is added to.

**Important:** Parameter-space statistics (e.g., Adam's $\hat{v}_t$) are *not* used for $M^{-1}(s)$. They belong to optimizer diagnostics, not state-space geometry.

This is a **state-space preconditioning** effect: directions with large local sensitivity (large $G$) are damped via $G^{-1}$, while flatter directions are amplified. This is the same qualitative idea as natural-gradient and second-order preconditioning, but applied to latent-state coordinates rather than to parameters {cite}`amari1998natural,martens2015kfac`.
* **High curvature / high sensitivity:** $G$ is large → $G^{-1}$ is small. The update **slows down** automatically.
* **Low curvature / low sensitivity:** $G$ is small → $G^{-1}$ is large. The update **accelerates** in well-conditioned regions.

**A Practical Diagonal Sensitivity Metric:**

We construct a diagonal state-space sensitivity metric using the **scaling coefficients** from Section 3.2:

$$G = \text{diag}(\alpha, \beta, \gamma, \delta)$$

Where:
* $\alpha$: critic curvature scale (value landscape sensitivity).
* $\beta$: policy stochasticity / exploration scale.
* $\gamma$: world-model volatility / non-stationarity scale.
* $\delta$: representation drift / codebook stability scale.

These coefficients summarize relative update scales across subsystems and serve as a compact diagnostic for stability monitoring.

**The Latent Space Metric on $\mathcal{Z}$:**

We define the complete state-space sensitivity metric as the sum of two contributions:

$$G_{ij}(z) = \underbrace{\frac{\partial^2 V(z)}{\partial z_i \partial z_j}}_{\text{Hessian (value curvature)}} + \lambda \underbrace{\mathbb{E}_{a \sim \pi} \left[ \frac{\partial \log \pi(a|z)}{\partial z_i} \frac{\partial \log \pi(a|z)}{\partial z_j} \right]}_{\text{Fisher (control sensitivity)}}$$
Units: the Fisher term has units $[z]^{-2}$; therefore $\lambda$ carries the same units as $V$ (here $\mathrm{nat}$) so both addends match.

**Dimensional Verification:**

- $V$ is a scalar potential (0-form) on $\mathcal{Z}$
- $\nabla_z V$ is a 1-form (covector): $dV = (\partial_i V) dz^i$
- $\text{Hess}_z(V) = \partial_i \partial_j V$ is a $(0,2)$-tensor
- The Fisher term is the covariance of the score function $\nabla_z \log \pi$, also a $(0,2)$-tensor
- Result: $G$ is a positive-definite $(0,2)$-tensor that defines the Riemannian structure on $\mathcal{Z}$

**Operational Interpretation:**

- **High $G_{ii}$** (large Hessian or Fisher): the objective/policy is highly sensitive to coordinate $i$ (ill-conditioned or highly controllable direction)
- **Low $G_{ii}$**: weak sensitivity or ignored coordinate $i$ (flat direction / potential blind spot)

### 2.6 The Metric Hierarchy: Fixing the Category Error

A common mistake in geometric RL is conflating three distinct geometries:

| Geometry | Manifold | Metric | Lives On | Used By |
|----------|----------|--------|----------|---------|
| **Euclidean** | Parameter Space $\Theta$ | $\|\cdot\|_2$ (flat) | Neural network weights | Adam, SGD |
| **Fisher-Rao** | Policy Space $\mathcal{P}$ | $F_{\theta\theta} = \mathbb{E}[(\nabla_\theta \log \pi)^2]$ | Policy parameters | TRPO, PPO |
| **State-Space Sensitivity** | State Space $Z$ | $G_{zz} = \mathbb{E}[(\nabla_z \log \pi)^2] + \text{Hess}_z(V)$ | Latent states | **Fragile Agent** |

**The Category Error:** Using Adam's $v_t$ (which approximates $F_{\theta\theta}$ in Parameter Space) as if it were $G_{zz}$ (State Space) mixes two different manifolds. This breaks coordinate invariance.

**The State-Space Fisher Information:**

$$G_{ij}(z) = \mathbb{E}_{a \sim \pi} \left[ \frac{\partial \log \pi(a|z)}{\partial z_i} \frac{\partial \log \pi(a|z)}{\partial z_j} \right]$$

This measures the **Information Bottleneck** between the Shutter (Split VQ-VAE) and the Actuator (Policy):
- High $G_{ii}$: The policy is sensitive to state dimension $i$ → high control authority
- Low $G_{ii}$: The policy ignores dimension $i$ → potential blind spot

**Why This Matters:**
- **Coordinate Invariance:** The agent's behavior is invariant to how you encode $z$
- **Natural-Gradient Paths:** updates follow shortest/least-distorting paths under the chosen metric
- **Stability:** the metric discourages overly large updates in regions where the model/value is highly sensitive

The Covariant Regulator uses the **State-Space Fisher Information** to scale the Lie Derivative. While standard RL uses Fisher in **Parameter Space** (TRPO/PPO), the Fragile Agent uses Fisher in **State Space** to stabilize **Causal Induction**.

### 2.7 The HJB Correspondence (Rewards as Value Updates)

We replace the heuristic Bellman equation with the rigorous **Hamilton-Jacobi-Bellman (HJB) Equation**:

$$\underbrace{\mathcal{L}_f V}_{\text{Lie Derivative}} + \underbrace{\mathfrak{D}(z, a)}_{\text{Control Effort / Regularizer}} = \underbrace{-\mathcal{R}(z, a)}_{\text{Instantaneous cost rate}}$$

**Critical Distinction:** The Lie derivative is **metric-independent**:

$$\mathcal{L}_f V = dV(f) = \partial_i V \cdot f^i = \nabla V \cdot f$$

This is the natural pairing between the 1-form $dV$ and the vector field $f$—NO metric $G$ appears.

**Dimensional Verification:**

$$[\nabla V \cdot f] = \frac{[V]}{[z]} \cdot \frac{[z]}{[t]} = \mathrm{nat}\,\mathrm{step}^{-1}$$

All terms in the HJB equation have units of a **cost rate**. In discrete time this is naturally $\mathrm{nat/step}$; in continuous time it is $\mathrm{nat\,s^{-1}}$.

**Where the Metric $G$ Appears (and Where It Does NOT):**

| Operation | Formula | Uses Metric? |
|-----------|---------|--------------|
| **Lie Derivative** | $\mathcal{L}_f V = dV(f) = \nabla V \cdot f$ | NO |
| **Natural Gradient** | $\delta z = G^{-1} \nabla_z \mathcal{L}$ | YES (index raising) |
| **Geodesic Distance** | $d_G(z_1, z_2)^2 = (z_1-z_2)^T G (z_1-z_2)$ | YES |
| **Trust Region** | $\|\delta \pi\|_G^2 \leq \epsilon$ | YES |
| **Gradient Norm** | $\|\nabla V\|_G^2 = G^{ij} (\partial_i V)(\partial_j V)$ | YES |

**Anti-Mixing Rule #2:** The Lie derivative $\mathcal{L}_f V = dV(f)$ is a pairing, not an inner product $\langle \cdot, \cdot \rangle_G$.

**Interpretation:**

- The critic $V$ is a value/cost-to-go function that should decrease along controlled trajectories.
- $\mathcal{R}(z,a)$ is an instantaneous cost rate (negative of reward rate, depending on sign convention).
- $\mathfrak{D}(z,a)$ is an explicit control-effort / regularization term (e.g., KL control, action penalties).
- At optimality, the relation enforces a local consistency between value change, immediate cost, and control effort.

### 2.8 Conditional Independence and Sufficiency (Causal Enclosure)

The transition from micro to macro is a **projection operator** $\Pi:\mathcal{Z}\to\mathcal{K}$. In the discrete macro instantiation, $\Pi$ is precisely the **VQ quantizer** $z_e\mapsto K$ from Section 2.2b.

**Causal Enclosure Condition (Markov sufficiency).** Let $(Z_t,A_t)$ be the microstate/action process and define the macrostate $K_t:=\Pi(Z_t)$. The macro-model requirement is the conditional independence
$$
K_{t+1}\ \perp\!\!\!\perp\ Z_t\ \big|\ (K_t,A_t),
$$
equivalently the vanishing of a conditional mutual information:
$$
I(K_{t+1};Z_t\mid K_t,A_t)=0.
$$
This is the information-theoretic statement that the macro-symbols are a sufficient statistic for predicting their own future, while the micro coordinates are residual variation not needed for macro prediction.

**Closure Defect (kernel-level).** Write the micro-dynamics as a Markov kernel $P(dz'\mid z,a)$ and let $P_\Pi(\cdot\mid z,a)$ be the pushforward kernel on $\mathcal{K}$ induced by $\Pi$. A learned macro-dynamics kernel $\bar{P}(\cdot\mid k,a)$ is enclosure-correct iff
$$
P_\Pi(\cdot\mid z,a)=\bar{P}(\cdot\mid \Pi(z),a)
\quad\text{for }P\text{-a.e. }z.
$$
A canonical defect functional is the expected divergence
$$
\delta_{\text{CE}}
:=
\mathbb{E}_{z,a}\Big[D_{KL}\big(P_\Pi(\cdot\mid z,a)\ \Vert\ \bar{P}(\cdot\mid \Pi(z),a)\big)\Big].
$$
This is the discrete, measure-theoretic refinement of the “commuting diagram” in the Micro–Macro Consistency metatheorem (see {prf:ref}`mt:micro-macro-consistency`).

Where:
- $P$ is the micro-dynamics (World Model) as a kernel on $\mathcal{Z}$
- $\bar{P}$ is the learned macro-dynamics (effective model) as a kernel on $\mathcal{K}$
- the divergence is over the discrete macro alphabet, so it is a true Shannon quantity (no differential-entropy ambiguity)

**Computational Meaning:** The macro-dynamics should be a homomorphism of the micro-dynamics. If $\delta_{\text{CE}} > 0$ (or equivalently $I(K_{t+1};Z_t\mid K_t,A_t)>0$), then the learned macro predictor is not sufficient: predicting $K_{t+1}$ still depends on nuisance microstate information.

### 2.9 Regularity Conditions

The formalism requires explicit assumptions:

1. **Smoothness:** $V \in C^2(\mathcal{Z})$ — the Hessian exists and is continuous
2. **Positive Definiteness:** $G(z) \succ 0$ for all $z \in \mathcal{Z}$ — the metric is non-degenerate
3. **Lipschitz Dynamics:** $\|f(z_1, a) - f(z_2, a)\| \leq L\|z_1 - z_2\|$ — no discontinuities
4. **Bounded State Space:** $\mathcal{Z}$ is compact, or $V$ has appropriate growth at infinity

**Diagonal Metric Approximation (Computational):**

> We approximate $G \approx \text{diag}(G_{11}, G_{22}, \ldots, G_{nn})$. This is valid when:
> - State dimensions are statistically independent under the policy
> - Cross-correlations $\text{Cov}(\partial \log \pi / \partial z_i, \partial \log \pi / \partial z_j)$ are small for $i \neq j$
>
> The approximation error is bounded by the spectral norm of the off-diagonal part of $G$.

### 2.10 Anti-Mixing Rules (Formal Prohibitions)

To maintain mathematical rigor, we strictly forbid the following operations:

| Rule | Prohibition | Reason |
|------|-------------|--------|
| **#1** | NO Parameter Fisher in State Space | $\mathcal{F}(\theta) \neq G(z)$; they live on different manifolds |
| **#2** | NO Metric in Lie Derivative | $\mathcal{L}_f V = dV(f)$ is metric-independent |
| **#3** | NO Coordinate-Dependent Step-Length | When budgeting update magnitude, use metric arc-length: $\int dt \sqrt{\dot{z}^T G \dot{z}}$ |
| **#4** | NO Unnormalized Optimization | Gradients pre-multiplied by $G^{-1}$ for natural gradient descent |

**Consequence of Violation:** Mixing manifolds breaks coordinate invariance. The agent's behavior will depend on the arbitrary choice of coordinates for $z$, leading to inconsistent generalization.

### 2.11 Variance–Value Duality and Information Conservation

To connect geometry (sensitivity) with stochastic control, we relate the value/cost functional and entropy/variance regularization through a coupling (precision) coefficient.

#### 2.11.1 Local Conditioning Scale and β-Coupling

**Definition 2.11.1 (Local Conditioning Scale).** Let $(\mathcal{Z}, G)$ be the Riemannian latent manifold. Define a local scale parameter $\Theta: \mathcal{Z} \to \mathbb{R}^+$ as the trace of the inverse metric:

$$\Theta(z) := \frac{1}{d} \operatorname{Tr}\left( G^{-1}(z) \right)$$

where $d = \dim(\mathcal{Z})$. The corresponding **precision / coupling coefficient** is $\beta(z) = [\Theta(z)]^{-1}$.
Units: if $z$ carries units $[z]$, then $[G]=\mathrm{nat}\,[z]^{-2}$ implies $[\Theta]=[z]^2/\mathrm{nat}$ and $[\beta]=\mathrm{nat}/[z]^2$ (dimensionless when $z$ is normalized).

**Lemma 2.11.2 (Variance–Curvature Correspondence).** The covariance of the policy $\pi(a|z)$ is coupled to the curvature/sensitivity encoded by $G$. In entropy-regularized control, a natural scaling is:

$$\Sigma_\pi(z) \propto \beta(z)^{-1} \cdot G^{-1}(z)$$

*Proof (sketch).* In maximum-entropy control / exponential-family models, stationary distributions over latent states often take an exponential form $p(z)\propto \exp(-\beta V(z))$. Matching this form with a geometry-aware update implies that policy covariance scales inversely with the sensitivity metric. Deviations can be measured by a **consistency defect** $\mathcal{D}_{\beta} := \|\nabla \log p + \beta \nabla V\|_G^2$.

**Definition 2.11.3 (Entropy-Regularized Objective Functional).** Let $d\mu_G:=\sqrt{|G|}\,dz$ be the Riemannian volume form on $\mathcal{Z}$ and let $p(z)$ be a probability density with respect to $d\mu_G$. For a (dimensionless) trade-off coefficient $\tau\ge 0$, define
$$
\mathcal{F}[p,\pi]
:=
\int_{\mathcal{Z}} p(z)\Big(V(z) - \tau\,H(\pi(\cdot\mid z))\Big)\,d\mu_G,
$$
where $H(\pi(\cdot\mid z)) := -\mathbb{E}_{a\sim \pi(\cdot\mid z)}[\log \pi(a\mid z)]$ is the per-state policy entropy (in nats). Because $V$ and $H$ are measured in nats (Section 1.2), $\tau$ is dimensionless.

#### 2.11.2 The Continuity Equation and Belief Conservation

This subsection records a **continuous-time idealization** that is useful for auditing “grounding”: belief mass in latent space should change only via (i) transport under internal dynamics, (ii) boundary-driven updates from observations, and (iii) explicit projection/reweighting events (Sections 3–6). The discrete-time implementation is given in Section 11; the PDEs below provide the corresponding limit intuition.

**Definition 2.11.4 (Belief Density).** Let $p(z,t)\ge 0$ be a density with respect to $d\mu_G$ representing the agent’s belief (or belief-weight) over latent coordinates. In closed-system idealizations one may impose $\int_{\mathcal{Z}}p(z,t)\,d\mu_G=1$; in open-system implementations with explicit projections/reweightings we track the unnormalized mass and renormalize when needed (Section 11).

**Definition 2.11.5 (Transport Field).** Let $v\in\Gamma(T\mathcal{Z})$ be a vector field describing the instantaneous transport of belief mass on $\mathcal{Z}$. In a value-gradient-flow idealization (used only for intuition), one may take
$$
v^i(z) := -G^{ij}(z)\frac{\partial V}{\partial z^j},
$$
so transport points in the direction of decreasing $V$ (Riemannian steepest descent). Units: if time is measured in steps, then $[v]=[z]/\mathrm{step}$ in the discrete-time scaling.

**Lemma 2.11.6 (Continuity Equation for Transport).** If the belief density evolves only by deterministic transport under $v$ (no internal sources/sinks), then it satisfies the continuity equation

$$\frac{\partial p}{\partial t} + \nabla_i \left( p v^i \right) = 0$$

where $\nabla_i$ denotes the Levi-Civita covariant derivative associated with $G$.

**Definition 2.11.7 (Source Residual).** In general, belief evolution may include additional update effects (e.g. approximation error, off-manifold steps, or explicit projection/reweighting). We collect these into a residual/source term $\sigma(z,t)$:

$$\frac{\partial p}{\partial t} + \operatorname{div}_G(p v) = \sigma$$

Interpreting $\sigma$:
1. If $\sigma>0$ on a region, belief mass is being created there beyond pure transport; this indicates an **ungrounded internal update** relative to the transport model.
2. If $\sigma<0$, belief mass is being removed beyond pure transport (aggressive forgetting or projection).
3. Integrating over any measurable region $U\subseteq\mathcal{Z}$ and applying the divergence theorem yields the exact mass balance
   $$
   \frac{d}{dt}\int_U p\,d\mu_G
   =
   -\oint_{\partial U}\langle p v,n\rangle\,dA_G
   +\int_U \sigma\,d\mu_G.
   $$
   For $U=\mathcal{Z}$ this relates net mass change to boundary flux and the integrated residual.

**Proposition 2.11.8 (Mass Conservation in a Closed Enclosure).** If $\sigma\equiv 0$ and the boundary flux vanishes (e.g. $\langle p v,n\rangle=0$ on $\partial\mathcal{Z}$), then the total belief mass
$$
\mathcal{V}(t):=\int_{\mathcal{Z}}p(z,t)\,d\mu_G
$$
is constant in time.

*Proof.* Applying the divergence theorem on the Riemannian manifold:

$$\frac{d\mathcal{V}}{dt} = \int_{\mathcal{Z}} \frac{\partial p}{\partial t} d\mu_G = -\int_{\mathcal{Z}} \operatorname{div}_G(p v) d\mu_G = -\int_{\partial \mathcal{Z}} \langle p v, n \rangle dA = 0$$

assuming there is no net boundary contribution and no internal source term. In applications we do not estimate $\sigma$ pointwise; instead we monitor surrogate checks (e.g. BoundaryCheck and coupling-window metrics) that are sensitive to persistent boundary decoupling (Sections 3 and 15).

#### 2.11.3 Geometric Summary of Internal Consistency

The dimensional and conceptual alignment is now fixed:

| Symbol | Object | Units | Role |
|--------|--------|-------|------|
| $V$ (Value / cost-to-go) | Scalar Field | $\mathrm{nat}$ | Objective landscape over $\mathcal{Z}$ |
| $G$ (Sensitivity metric) | $(0,2)$-Tensor Field | $\mathrm{nat}\,[z]^{-2}$ | Local conditioning / state-space sensitivity |
| $\beta$ (Local coupling) | Scalar | $\mathrm{nat}/[z]^2$ | Conditioning scale derived from $G$ (Definition 2.11.1) |
| $\tau$ (Entropy weight) | Scalar | dimensionless | Cost–entropy trade-off weight (Definition 2.11.3) |
| $p$ (Belief density) | Measure | $[d\mu_G]^{-1}$ | Belief mass/weight over $\mathcal{Z}$ (Definition 2.11.4) |

These identities are **model checks**, not automatic certificates for deep, nonconvex training. In practice, large residuals (e.g. persistent boundary decoupling or unstable drift statistics) indicate the agent is operating outside the assumed regime and should trigger conservative updates or explicit interventions (Sections 3–6 and 15).

#### 2.11.4 The Interface and Observation Inflow

In the general case, the manifold $(\mathcal{Z}, G)$ is a compact Riemannian manifold with boundary $\partial \mathcal{Z}$. The boundary represents the agent’s **interface**—the site of interaction between internal belief/state and external observations $\mathcal{X}$.

**Definition 2.11.9 (Observation Inflow Form).** Let $j \in \Omega^{d-1}(\partial \mathcal{Z})$ be the **observation inflow form**. This form represents the rate of information entering the model through the interface.

**Theorem 2.11.10 (Generalized Conservation of Belief).** The evolution of the belief density $p$ satisfies the **Global Balance Equation**:

$$
\frac{d}{dt}\int_{\mathcal{Z}}p\,d\mu_G
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G
\;+\;
\int_{\mathcal{Z}} \sigma\,d\mu_G.
$$

where $n$ is the outward unit normal and $dA_G$ is the induced boundary area element. (Equivalently, if $\iota:\partial\mathcal{Z}\hookrightarrow \mathcal{Z}$ is the inclusion map, then the boundary flux is the pullback $\iota^*(p v\;\lrcorner\; d\mu_G)$.)

**The Architectural Sieve Condition (Node 13: BoundaryCheck).** The idealized “fully grounded” regime corresponds to $\sigma\approx 0$ in the interior: net changes in internal belief mass should be attributable to boundary influx and explicit projection events. Operationally we do not estimate $\sigma$ pointwise; instead Node 13 and the coupling-window diagnostics (Theorem 15.1.3) enforce that the macro register remains coupled to boundary data (non-collapse of $I(X;K)$) and does not saturate ($H(K)$ stays below $\log|\mathcal{K}|$).

$$
\frac{d\mathcal{V}}{dt}
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G,
$$
in the case $\sigma\equiv 0$.

Here $\langle p v,n\rangle$ is the outward flux density across the boundary (negative values correspond to net inflow).

**Distinction: boundary-driven updates vs ungrounded updates**

1.  **Valid learning (boundary-driven):** The belief changes because there is non-negligible boundary flux, i.e. new observations justify updating the internal state.
2.  **Ungrounded update (internal source):** The belief changes despite negligible boundary flux, corresponding to $\sigma>0$ under the transport model. Operationally, this is a warning sign that internal rollouts are decoupled from the data stream and should be treated as unreliable for control until re-grounded.

**Corollary 2.11.11 (Boundary filter interpretation).** Sieve Nodes 13-16 (Boundary/Overload/Starve/Align) can be interpreted as monitoring a trace-like coupling between bulk and boundary (informally: whether internal degrees of freedom remain supported by boundary evidence), analogous in spirit to the trace map $\operatorname{Tr}: H^1(\mathcal{Z}) \to H^{1/2}(\partial \mathcal{Z})$:

*   **Mode B.E (Injection):** Occurs when interface inflow exceeds the effective capacity of the manifold (Levin capacity), breaking the assumed operating regime.
*   **Mode B.D (Starvation):** Occurs when interface inflow is too weak, causing the internal information volume to decay (catastrophic forgetting).

#### 2.11.5 The HJB–Interface Coupling

To maintain stability, the value/cost function $V$ must satisfy interface constraints dictated by the environment:

$$\langle \nabla_G V, n \rangle \big|_{\partial \mathcal{Z}} = \gamma(x_t)$$

where $\gamma$ is an **instantaneous external cost/risk signal** of the external state $x_t$, with units matching $\langle \nabla_G V, n \rangle$ (dimensionless in normalized coordinates).

**Interpretation:** This anchors the internal value landscape to externally observed signals at the interface. If the internal $V$ near the boundary does not match external feedback, the agent enters **Mode B.C (Control Deficit)**—its internal model may be self-consistent but poorly aligned with the task-relevant data stream.

#### 2.11.6 Summary: Geometry–Regularization–Interface

The Trinity of Manifolds is extended to the **Boundary Operator**:

| Aspect | Governs | Formalism |
|--------|---------|-----------|
| **Internal Geometry** | How the agent "reasons" | Geodesics on $(\mathcal{Z}, G)$ |
| **Regularization / Precision ($\beta$)** | The sharpness of those reasons | Variance–curvature coupling via $\Theta(z)$ |
| **Interface Inflow ($j$)** | The grounding of those reasons | Conservation/balance across $\partial \mathcal{Z}$ |

**Operational audit criterion.** Rather than treating internal variables as inherently grounded, we require that changes in internal belief/state be explainable by boundary coupling and declared projection events. In practice this is enforced via BoundaryCheck, coupling-window constraints, and enclosure/closure defects; persistent violations indicate that internal rollouts are no longer reliable for control and should trigger conservative updates or re-grounding interventions.

---

## 3. Diagnostics: Stability Checks (Monitors)

Stability and data-quality are monitored via 21 distinct checks (Gate Nodes). Each corresponds to a specific, testable condition on the interaction between the agent and its environment.

**Relation to prior work.** Many safe-RL formulations express safety as one (or a few) expected-cost constraints in a constrained MDP {cite}`altman1999constrained,achiam2017constrained`. The Fragile Agent keeps that spirit but broadens the constraint surface to include **representation and interface diagnostics** (grounding, mixing, saturation, switching, stiffness) that can be audited online, alongside Lyapunov-style stability constraints {cite}`chow2018lyapunov`.

### The 21 Stability Checks

| Node | Check | Component | Interpretation | Meaning | Regularization Factor ($\mathcal{L}_{\text{check}}$) | Compute |
|------|-----------|-----------|-----------------------------------|---------|------------------------------------------------------|---------|
| **1** | **CostBoundCheck ($D_C$)** | **Critic** | **Cost Budget Check** | Is current cost ($V(s)$) within budget? | $\max(0, V(s) - V_{\text{max}})^2$ (Cost Bound) | $O(B)$ ✓ |
| **2** | **ZenoCheck ($\mathrm{Rec}_N$)** | **Policy** | **Action Frequency Limit** | Switching policies too fast? | $D_{KL}(\pi_t \Vert \pi_{t-1})$ (Smoothness) | $O(BA)$ ✓ |
| **3** | **CompactCheck ($C_\mu$)** | **VQ-VAE** | **Belief Concentration** | Macro assignment sharp? | $H(q(K \mid x))$ (Symbol Entropy) | $O(BZ)$ ✓ |
| **4** | **ScaleCheck ($\mathrm{SC}_\lambda$)** | **All** | **Adaptation Scaling** | Adaptation speed > Disturbance speed? | $\Vert \nabla \theta \Vert / \Vert \Delta S \Vert$ (Relative Rate) | $O(P)$ ⚡ |
| **5** | **ParamCheck ($\mathrm{SC}_{\partial c}$)** | **World Model** | **Stationarity Check** | Dynamics stable? | $\Vert \nabla_t S_t \Vert^2$ (Time Derivative Penalty) | $O(P_{WM})$ ⚡ |
| **6** | **GeomCheck ($\mathrm{Cap}_H$)** | **VQ-VAE / WM** | **Blind Spot Check** | Unobservable states negligible? | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE) | $O(B^2Z)$ ⚡ |
| **7** | **StiffnessCheck ($\mathrm{LS}_\sigma$)** | **Critic** | **Responsiveness / Gain** | Gradient signal strong enough? | $\max(0, \epsilon - \Vert \nabla V \Vert)$ (Gain > $\epsilon$) | $O(BZ)$ ✓ |
| **7a**| **BifurcateCheck ($\mathrm{LS}_{\partial^2 V}$)**| **World Model** | **Instability Check** | Bifurcation point? | $\det(J_{S_t})$ (Jacobian Determinant) | $O(Z^3)$ ✗ |
| **7b**| **SymCheck ($G_{\mathrm{act}}$)** | **Policy** | **Alternative Strategy Search**| Symmetric strategies available? | $-\sum \pi(a_i) \log \pi(a_i)$ (Policy Entropy) | $O(BA)$ ✓ |
| **7c**| **CheckSC ($\mathrm{SC}_{\partial c}$)** | **Critic** | **New Mode Viability** | New mode stable? | $\text{Var}(V(s'))$ (Variance Check) | $O(B)$ ✓ |
| **7d**| **CheckTB ($\mathrm{TB}_S$)** | **Policy** | **Transition Feasibility** | Switching cost affordable? | $\Vert V(\pi') - V(\pi) \Vert - B_{\text{switch}}$ | $O(B)$ ⚡ |
| **8** | **TopoCheck ($\mathrm{TB}_\pi$)** | **Policy** | **Sector Reachability** | Goal reachable? | $T_{\text{reach}}(s_{\text{goal}})$ (Reachability Map) | $O(HBZ)$ ✗ |
| **9** | **TameCheck ($\mathrm{TB}_O$)** | **World Model** | **Interpretability Check** | World "tame"? | $\Vert \nabla^2 S_t \Vert$ (Hessian Norm / Smoothness) | $O(Z^2 P_{WM})$ ✗ |
| **10** | **ErgoCheck ($\mathrm{TB}_\rho$)** | **Policy** | **Exploration/Mixing** | Sufficient exploration? | $-H(\pi)$ (Max Entropy) | $O(BA)$ ✓ |
| **11** | **ComplexCheck ($\mathrm{Rep}_K$)** | **VQ-VAE** | **Model Capacity Check** | Symbolic rate within budget? | $\mathrm{Rep}_K := H(K)/\log|\mathcal{K}|$ (Rate Utilization) | $O(B)$ ✓ |
| **12** | **OscillateCheck ($\mathrm{GC}_\nabla$)** | **WM / Policy** | **Oscillation / Chattering** | Limit cycles? | $\Vert z_t - z_{t-2} \Vert$ (Period-2 Penalty) | $O(BZ)$ ✓ |
| **13** | **BoundaryCheck ($\mathrm{Bound}_\partial$)** | **VQ-VAE** | **Input Informativeness** | External signal present? | $I(X;K)$ (Symbolic MI $>0$) | $O(B)$ ✓ |
| **14** | **InputSaturationCheck ($\mathrm{Bound}_B$)** | **Boundary** | **Input Saturation** | Inputs clipping? | $\mathbb{I}(\lvert x \rvert > x_{\text{max}})$ (Saturation Flag) | $O(BD)$ ✓ |
| **15** | **SNRCheck ($\mathrm{Bound}_{\Sigma}$)** | **Boundary** | **Signal-to-Noise** | Signal strength sufficient? | $\text{SNR} < \epsilon$ (Noise Floor Check) | $O(BD)$ ✓ |
| **16** | **AlignCheck ($\mathrm{GC}_T$)** | **Critic** | **Objective Alignment** | Proxy matches objective? | $\lvert V_{\text{proxy}} - V_{\text{true}} \rvert$ (Alignment Error) | $O(B)$ ✗ |
| **17** | **Lock ($\mathrm{Cat}_{\mathrm{Hom}}$)** | **WM** | **Structural Constraint** | Hard safe-guards active? | $\mathbb{I}(\text{Unsafe}) \cdot \infty$ (Hard Constraint) | $O(B)$ ✓ |

**Compute Legend:** ✓ Easy (<10% overhead) | ⚡ Medium (10-50% overhead) | ✗ Hard (>50% overhead or infeasible)
**Variables:** $B$ = batch, $Z$ = latent dim, $A$ = actions, $P$ = params, $H$ = horizon, $D$ = observation dim
**Threshold units:** whenever a node uses a threshold $\epsilon$, it inherits the units of the compared quantity (e.g., $\epsilon$ is dimensionless for SNR checks; $\epsilon$ has the same units as $\|\nabla V\|$ for stiffness checks; $\epsilon$ is in nats when compared to $I(X;K)$ or $H(K)$). Budgets like $V_{\text{max}},V_{\text{limit}}$, and $B_{\text{switch}}$ share units with $V$ (nats in the convention of Section 1.2).

**Geometric Properties of Key Nodes:**

| Node | Space | Formal Property | Verification Criterion |
|------|-------|-----------------|------------------------|
| **1 (CostBound)** | $V \in \mathcal{F}(\mathcal{Z})$ | Sublevel Set Compactness | Is $\{z \mid V(z) \leq c\}$ compact? |
| **7 (Stiffness)** | $G \in T^*_2(\mathcal{Z})$ | Spectral Gap | Is $\lambda_{\min}(G) > \epsilon$? (No flat directions) |
| **9 (Tameness)** | $f: \mathcal{Z} \to T\mathcal{Z}$ | Lipschitz Continuity | Is $\|\nabla_z f\|_G < K$? (No chaos) |
| **17 (Lock)** | $H_n(\mathcal{Z})$ | Homological Obstruction | Does the "bad pattern" induce a non-trivial cycle? |

Each node corresponds to a verifiable geometric property. The Sieve acts as a **topological filter**: problems that fail these checks are rejected before gradient updates can corrupt the agent.

### 3.1 Theory: Thin Interfaces

In the Hypostructure framework, **Thin Interfaces** are defined as minimal couplings between components. Instead of monolithic end-to-end training, we enforce structural contracts (the checks) via **Defect Functionals** ($\mathcal{L}_{\text{check}}$).

*   **Principle:** Components (VQ-VAE, WM, Critic, Policy) should be **autonomous** but **aligned**.
*   **Mechanism:** Each component minimizes its own objective *subject to* the cybernetic constraints imposed by the others.

### 3.2 Scaling Exponents: Characterizing the Agent

We characterize the training dynamics of the Fragile Agent using four **scaling coefficients**. These are *diagnostic* summaries of state-space behavior, not optimizer statistics.

The geometric metric $G$ is the **State-Space Fisher Information** (see Section 2.6), ensuring coordinate invariance. Common diagonal approximations include:
- `policy_fisher`: $G_{ii} = \mathbb{E}[(\partial \log \pi / \partial z_i)^2]$
- `state_fisher`: $G_{ii} = \mathbb{E}[(\partial \log \pi / \partial z_i)^2] + \text{Hess}_z(V)_{ii}$ (Hessian + Fisher sensitivity)
- `grad_rms`: $G_{ii} = \mathbb{E}[(\partial V / \partial z_i)^2]^{1/2}$
- `obs_var`: $G_{ii} = \text{Var}(z_i)$

| Component | Exponent | Symbol | Units | Interpretation | Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Critic** | **Curvature scale** | $\alpha$ | dimensionless | **Value curvature:** magnitude of value gradients/curvature. | High $\alpha$: strong supervision.<br>Low $\alpha$: flat landscape (BarrierGap). |
| **Policy** | **Exploration scale** | $\beta$ | dimensionless | **Policy variance / update scale.** | High $\beta$: high noise/plasticity.<br>Low $\beta$: near-deterministic/frozen. |
| **World Model** | **Volatility scale** | $\gamma$ | dimensionless | **Dynamics non-stationarity / rollout volatility.** | High $\gamma$: unstable/chaotic predictions.<br>Low $\gamma$: stable dynamics. |
| **VQ-VAE** | **Drift scale** | $\delta$ | dimensionless | **Representation drift:** codebook/encoder stability. | High $\delta$: symbol churn (representation drift).<br>Low $\delta$: stable representation. |

**The Stability Hierarchy (BarrierTypeII):**
Stable training requires separation of timescales: the representation should change slowest, the world model should not drift faster than the critic can track, and the policy should not update faster than the critic’s usable signal. A practical regime is:
$$ \delta \ll \gamma \ll \alpha,\qquad \beta \le \alpha $$
1.  **$\delta \ll \gamma$ (Representation Stability):** the representation (encoder/codebook) drifts slower than the learned dynamics model.
2.  **$\gamma \ll \alpha$ (Predictability / Trackability):** the learned dynamics do not drift faster than the value function can track.
3.  **$\beta \le \alpha$ (Two-Time-Scale Actor–Critic):** policy updates stay within the critic’s validity region. If $\beta>\alpha$, skip or shrink the policy update (BarrierTypeII; see Section 4.1.2).

### 3.3 Defect Functionals: Implementing Regulation

We regulate the Fragile Agent by augmenting the loss function with specific terms for each component. These terms are non-negotiable cybernetic contracts.

#### A. VQ-VAE Regulation (The Shutter)
*   **Symbolic Bottleneck (Node 3 / 11):** the shutter is a *split* latent $(K,z_\mu)$ with $K\in\mathcal{K}$ discrete (Section 2.2b). A canonical objective is:
    $$
    \mathcal{L}_{\text{shutter}}
    =
    \mathcal{L}_{\text{recon}}
    + \underbrace{\| \operatorname{sg}[z_e]-e_{K}\|_2^2 + \beta\|z_e-\operatorname{sg}[e_K]\|_2^2}_{\text{VQ codebook + commitment}}
    + \underbrace{\beta_\mu D_{KL}(q(z_\mu \mid x) \Vert p(z_\mu))}_{\text{micro-as-noise}}
    + \underbrace{\lambda_{\text{use}} D_{KL}(\hat{p}(K)\ \Vert\ \mathrm{Unif}(\mathcal{K}))}_{\text{anti-collapse (optional)}}.
    $$
    Units: $\beta$, $\beta_\mu$, and $\lambda_{\text{use}}$ are dimensionless weights; each $D_{KL}$ is measured in nats.
    *   *Effect:* The macro channel is a bounded-rate symbolic register; the micro channel is forced toward a high-entropy prior (treated as nuisance/noise). The optional usage term prevents codebook collapse while leaving the rate bounded by $\log|\mathcal{K}|$.
*   **Contrastive Anchoring (Node 6):**
    $$ \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum \exp(\text{sim}(z_t, z_{neg}))} $$
    *   *Effect:* Ensures the latent space captures long-term structural dependencies (slow features), not just pixel reconstruction.

*   **VICReg: Variance-Invariance-Covariance Regularization (Alternative to InfoNCE):**

    VICReg {cite}`bardes2022vicreg` provides an alternative approach to preventing representation collapse **without requiring negative samples**. While InfoNCE contrasts positive pairs against negatives, VICReg uses geometric constraints.

    **The Collapse Problem:**
    Self-supervised learning can produce trivial solutions where the encoder maps all inputs to a constant. VICReg prevents this through three orthogonal constraints:

    **1. Invariance Loss (Metric Stability):**
    $$\mathcal{L}_{\text{inv}} = \|z - z'\|^2$$
    - $z, z'$ are embeddings of two augmented views of the same input
    - *Effect:* Forces representations to be stable under perturbations

    **2. Variance Loss (Non-Collapse):**
    $$\mathcal{L}_{\text{var}} = \frac{1}{d} \sum_{j=1}^{d} \max(0, \gamma - \sqrt{\text{Var}(z_j) + \epsilon})$$
- $\gamma$ is the target standard deviation (typically 1)
- Units: $[\gamma]=[z_j]$ and $[\epsilon]=[z_j]^2$ in this expression.
- *Effect:* Forces each dimension to have non-trivial variance (prevents collapse to a point)

    **3. Covariance Loss (Decorrelation):**
    $$\mathcal{L}_{\text{cov}} = \frac{1}{d} \sum_{i \neq j} [\text{Cov}(z)]_{ij}^2$$
    - *Effect:* Forces off-diagonal covariance to zero (decorrelates dimensions)

    **Combined VICReg Loss:**
$$\mathcal{L}_{\text{VICReg}} = \lambda \mathcal{L}_{\text{inv}} + \mu \mathcal{L}_{\text{var}} + \nu \mathcal{L}_{\text{cov}}$$
Units: $\lambda,\mu,\nu$ are dimensionless weights; each component loss is taken dimensionless (nats after normalization).

    **Comparison: InfoNCE vs VICReg vs Barlow Twins:**

    | Method | Negative Samples | Collapse Prevention | Computation | Citation |
    |--------|------------------|---------------------|-------------|----------|
    | **InfoNCE** | Required ($B^2$ pairs) | Contrastive pushing | $O(B^2 Z)$ | {cite}`oord2018cpc` |
    | **VICReg** | None | Variance constraint | $O(B Z^2)$ | {cite}`bardes2022vicreg` |
    | **Barlow Twins** | None | Cross-correlation identity | $O(B Z^2)$ | {cite}`zbontar2021barlow` |

    **When to Use Which:**
    - **InfoNCE:** When you have large batches and care about discriminative features
    - **VICReg:** When you want geometric constraints without mining hard negatives
    - **Barlow Twins:** When you want redundancy reduction (information-theoretic)

*   **Whitening / Orthogonality (Node 6 — Identifiability):**
    $$ \mathcal{L}_{\text{orth}} = \Vert \operatorname{Cov}(z) - I \Vert_F^2 \quad \text{or} \quad \| J_S^T J_S - I \|^2 $$
    *   *Effect:* Removes redundant/degenerate directions in the representation. Approximate whitening makes the latent coordinates identifiable up to permutation/sign, improves conditioning, and reduces collapse/exploding-gradient pathologies.
    Units: $\mathcal{L}_{\text{orth}}$ is dimensionless; the weight multiplying it is dimensionless.

#### B. World Model Regulation (Dynamics Model)
*   **Lipschitz Constraint (BarrierOmin / Node 9):**
    $$ \mathcal{L}_{\text{Lip}} = \mathbb{E}_{s, s'}[(\|S(s) - S(s')\| / \|s - s'\| - K)^+]^2 $$
    Or via Spectral Normalization on weights.
    *   *Effect:* Enforces **tameness**: bounds sensitivity of the learned dynamics and reduces non-smooth / ill-conditioned rollouts that destabilize planning and control.
*   **Forward Consistency (Node 5):**
    $$ \mathcal{L}_{\text{pred}} = \| S(z_t, a_t) - z_{t+1} \|^2 $$
    *   *Effect:* Standard dynamics learning, but constrained by the Lyapunov potential (see below).

#### C. Critic Regulation (Value / Lyapunov Function)

The Critic does not just predict reward; it defines a **stability-oriented potential** over latent state. We impose Lyapunov-style constraints as *sufficient conditions* for local stability, enforced approximately via sampled penalties {cite}`chang2019neural,chow2018lyapunov,kolter2019safe`.

**Euclidean vs Riemannian Critic Losses:**

| Loss Type | Euclidean (Standard) | Riemannian (Lyapunov) |
|-----------|----------------------|----------------------|
| **Primary** | $\mathcal{L} = \|V_{\text{pred}} - V_{\text{target}}\|^2$ | $\mathcal{L}_{\text{Lyap}} = \mathbb{E}[\max(0, \dot{V}(s) + \alpha V(s))^2]$ |
| **Goal** | Accuracy | Stability-oriented constraint |
| **Failure Mode** | Flat plateaus, jagged landscapes | Mitigated |
| **Geometry** | Ignores curvature | Encourages a well-conditioned potential |

*   **Lyapunov Decay (Node 7 - Stiffness):**
    Enforce a sampled Lyapunov decrease condition (a sufficient stability surrogate):
    $$\mathcal{L}_{\text{Lyapunov}} = \mathbb{E}_{s} [\max(0, \dot{V}(s) + \alpha V(s))^2]$$
    * *Mechanism:* Penalize states where the estimated decrease $\dot{V}$ is not sufficiently negative (relative to rate $\alpha$). This encourages $V$ to decrease along trajectories in regions the agent visits.

*   **Eikonal-style Gradient Regularization (BarrierGap - Geometric Constraint):**
    $$\mathcal{L}_{\text{Eikonal}} = (\|\nabla_s V\| - 1)^2$$
    * *Effect:* Encourages distance-like scaling of $V$ and mitigates exploding/vanishing gradients. It does not, by itself, guarantee that $V$ is an exact geodesic distance without additional conditions (e.g. boundary conditions and regularity).

*   **Lyapunov Stiffness (Node 7):**
    $$ \mathcal{L}_{\text{Stiff}} = \max(0, \epsilon - \|\nabla V(s)\|)^2 + \|\nabla V(s)\|^2_{\text{reg}} $$
    *   *Effect:* The gradient $\nabla V$ must be non-zero (to drive the policy) but bounded (to prevent explosion).
*   **Safety Budget (Node 1):**
    $$ \mathcal{L}_{\text{Risk}} = \lambda_{\text{safety}} \cdot \mathbb{E}[\max(0, V(s) - V_{\text{limit}})] $$
    *   *Effect:* Hard Lagrangian enforcement of the risk budget.

#### D. Policy Regulation (Controller / Geometry-Aware Updates)

The Policy is the controller. Its objective is to choose actions that reduce expected cost while respecting stability and information constraints. We replace purely Euclidean policy-gradient updates with a **natural-gradient / information-geometric** update that respects the local sensitivity metric $G$ (Section 2.5).

**Euclidean vs Riemannian Policy Losses:**

| Loss Type | Euclidean (Standard) | Geometry-aware (Natural) |
|-----------|----------------------|------------------------|
| **Primary** | $\mathcal{L} = -\log \pi(a|s) \cdot A(s,a)$ | $\mathcal{L}_{\text{nat}} = -\mathbb{E}\left[\frac{\nabla_s V(s) \cdot f(s, a)}{\sqrt{G_{ii}(s)}}\right]$ |
| **What it maximizes** | Advantage (scalar) | Value-decrease rate normalized by $G$ |
| **Geometry** | Ignores local conditioning | Uses Fisher/Hessian sensitivity metric $G$ |
| **Ill-conditioned regions** | Aggressive steps can destabilize | Geometry-scaled steps are conservative |
| **Mechanism** | Push toward high reward | Push along manifold |

*   **Value-Decrease Maximization (Node 10 — Natural Gradient):**
    $$\mathcal{L}_{\text{nat}} = -\mathbb{E}_{s, a \sim \pi} \left[ \frac{\nabla_s V(s) \cdot f(s, a)}{\sqrt{G_{ii}(s)}} \right]$$
    * *Mechanism:* Maximize the alignment between the value gradient $\nabla V$ and the realized dynamics $f(s,a)$, normalized by the local sensitivity scale ($G_{ii}$).
    * *Effect:* Where the landscape is sensitive/ill-conditioned (large $G$), effective steps shrink; where it is well-conditioned (small $G$), steps can be larger.

*   **Geodesic Stiffness (Node 2 - Zeno Constraint):**
    $$\mathcal{L}_{\text{Zeno}} = \|\pi_t - \pi_{t-1}\|^2_{G}$$
    * *Effect:* Penalizes high-frequency switching, weighted by geometry. Switching is penalized more strongly in regions where the metric indicates high sensitivity (large $G$).

*   **Standard Zeno Constraint (Euclidean fallback):**
    $$ \mathcal{L}_{\text{Zeno}}^{\text{Euc}} = D_{KL}(\pi(\cdot \mid s_t) \Vert \pi(\cdot \mid s_{t-1})) $$
    *   *Effect:* Penalizes high-frequency action switching (chattering).
*   **Entropy Regularization (Node 10):**
    $$ \mathcal{L}_{\text{Ent}} = -\mathcal{H}(\pi(\cdot \mid s)) $$
    *   *Effect:* Prevents premature collapse to deterministic policies (BarrierMix).

#### E. Cross-Network Synchronization (Alignment Terms)
A key design choice in the Fragile Agent is to make inter-component alignment explicit via **synchronization losses**:

1.  **Shutter $\leftrightarrow$ WM (Macro Closure / Predictability):**
    *   The shutter is not merely compressing $x_t$; it is defining the **macro-effective ontology** $K_t\in\mathcal{K}$ on which the World Model claims to be Markov (Causal Enclosure; Section 2.8).
    *   $$ \mathcal{L}_{\text{Sync}_{K-W}} = \mathrm{CE}\!\left(K_{t+1},\ \hat{p}_\phi(K_{t+1}\mid K_t, A_t)\right) $$
    *   *Meaning:* If the shutter emits macrostates that the WM cannot predict (large closure cross-entropy), then the ontology is inconsistent: either the symbol inventory is unstable (codebook churn) or the WM class is misspecified (Mode D.C / T.E).

2.  **Critic $\leftrightarrow$ Policy (Audit / Advantage Gap):**
    *   The Critic is the risk auditor. If the Policy acts in a way the Critic didn't anticipate, there is a control gap.
    *   $$ \mathcal{L}_{\text{Sync}_{V-\pi}} = \| V(s) - (r + \gamma V(s')) \|^2 \quad (\text{TD-Error}) $$
    *   *Critically:* We track the **Advantage Gap** $\Delta A = |A^{\pi}(s, a) - A^{\text{Buffer}}(s, a)|$. If $\Delta A$ grows, the policy has drifted off-manifold (BarrierTypeII).

3.  **WM $\leftrightarrow$ Policy (Control-Awareness):**
    *   The WM should allocate capacity where the Policy visits (On-Policy dynamics).
    *   $$ \mathcal{L}_{\text{Sync}_{W-\pi}} = \mathbb{E}_{z \sim \pi} [\mathcal{L}_{\text{pred}}(z)] $$
    *   *Meaning:* Accuracy on the *optimal path* matters more than global accuracy.

#### F. Exploration and Coupling Regularizers (Path Entropy, KL-Control, Window)
The synchronization and component losses above enforce internal consistency. The following regularizers make information/coupling constraints explicit in online-auditable form.

*   **KL-Control (Relative-Entropy Control; Theorem 13.2.1).** Fix a reference actuator prior $\pi_0(a\mid k)$ with full support. Define control effort as KL deviation from this prior:
    $$
    \mathcal{L}_{\text{KL-ctrl}}
    :=
    T_c\,\mathbb{E}_{K_t}\!\left[D_{KL}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right)\right].
    $$
    When $\pi_0$ is uniform, this reduces (up to an additive constant) to standard entropy regularization; when $\pi_0$ encodes actuator limits, it becomes a calibrated control-effort penalty.

*   **Path-Entropy Exploration (Future Flexibility; Definition 10.1.2).** Encourage non-degenerate reachable macro futures by maximizing causal path entropy (equivalently minimizing its negative):
    $$
    \mathcal{L}_{\text{expl}}
    :=
    -\sum_{h=1}^{H} w_h\,S_c(K_t,h;\pi),
    $$
    with weights $w_h\ge 0$. In practice, a computable proxy is the entropy of the WM-predicted horizon marginals $\hat{P}_\phi(K_{t+h}\mid K_t)$ obtained by rollout or dynamic programming.

*   **Information–Stability Window (Theorem 15.1.3).** Penalize both under-coupling (loss of grounding) and over-coupling (symbol dispersion / saturation). With thresholds $0<\epsilon<\log|\mathcal{K}|$,
    $$
    \mathcal{L}_{\text{window}}
    :=
    \mathrm{ReLU}\!\big(\epsilon - I(X_t;K_t)\big)^2
    +
    \mathrm{ReLU}\!\big(H(K_t)-(\log|\mathcal{K}|-\epsilon)\big)^2.
    $$
    This is an explicit online enforcement of the coupling window: $I(X;K)$ must not collapse, and $H(K)$ must not saturate.

*   **Regularized Objective Descent (Sections 9.11 and 11–14).** Define an instantaneous (per-step) regularized objective
    $$
    F_t
    :=
    V(Z_t)
    + \beta_K\big(-\log p_\psi(K_t)\big)
    + \beta_\mu D_{KL}\!\left(q(z_{\mu,t}\mid x_t)\ \Vert\ p(z_\mu)\right)
    + T_c D_{KL}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),
    \qquad \text{where } Z_t=(K_t,z_{\mu,t}).
    $$
    A monotonicity surrogate is then enforced by
    $$
    \mathcal{L}_{\downarrow F}
    :=
    \mathbb{E}\!\left[\mathrm{ReLU}\!\left(F_{t+1}-F_t\right)^2\right],
    $$
    optionally applied only inside the safety budget (Node 1 / CostBoundCheck) to avoid suppressing necessary exploration.

### 3.4 Joint Optimization
The total Fragile Agent training objective is the weighted sum of component and synchronization tasks:
$$ \mathcal{L}_{\text{Fragile}} = \mathcal{L}_{\text{Task}} + \sum \lambda_i \mathcal{L}_{\text{Self-Reg}_i} + \sum \lambda_{ij} \mathcal{L}_{\text{Sync}_{ij}} $$
This defines the coupled-system “stiffness”. In practice, the coefficients $\lambda$ should be treated as **adaptive multipliers** (not fixed constants): different constraints become active at different times, and gradient scales drift as representation and policy change (Section 3.5). If $\lambda_{\text{Sync}}$ is too low, components drift out of alignment; if it is too high, optimization becomes over-regularized and can stall (BarrierBode).

### 3.5 Adaptive Multipliers: Learned Penalties, Setpoints, and Calibration

In the Fragile Agent, **static loss weights are a failure mode**: hard-coding numbers like $\lambda=0.1$ implicitly assumes a constant exchange rate between heterogeneous terms even though their typical magnitudes and gradients change across training, operating regimes, and distribution shift.

We distinguish three classes of coefficients:
- **Dual multipliers (constraints):** enforce nonnegotiable inequalities (Gate Nodes / Barriers).
- **Setpoint controllers (regulators):** maintain a metric near a target (entropy, KL-per-update, code usage), rather than driving it to zero.
- **Learned precisions (multi-task scaling):** balance likelihood-style losses with unknown noise scales (reconstruction vs prediction vs auxiliary SSL).

#### 3.5.1 Method A: Primal–Dual Updates (Projected Dual Ascent)

Choose online-computable nonnegative constraint metrics $\mathcal{C}_i(\theta)$ and tolerances $\epsilon_i$ defining a feasible set
$$
\mathcal{C}_i(\theta)\le \epsilon_i,
\qquad i=1,\dots,m.
$$
Examples include enclosure defects (Section 2.8), Zeno/step-size limits (Node 2), saturation measures (BarrierSat), and Lyapunov defects (Node 7).

Define the Lagrangian (units consistent with Section 1.2):
$$
\mathcal{L}(\theta,\lambda)
=
\mathcal{L}_{\text{Task}}(\theta)
\;+\;
\sum_{i=1}^{m}\lambda_i\big(\mathcal{C}_i(\theta)-\epsilon_i\big),
\qquad
\lambda_i\ge 0.
$$

**Online algorithm (two-step loop).**
1. **Primal step (agent):** update $\theta$ to reduce $\mathcal{L}(\theta,\lambda)$ using your standard optimizer.
2. **Dual step (multipliers):** increase pressure on violated constraints:
   $$
   \lambda_i
   \leftarrow
   \Pi_{[0,\lambda_{\max}]}\!\left(\lambda_i + \eta_{\lambda}\,(\mathcal{C}_i(\theta)-\epsilon_i)\right),
   $$
   where $\Pi$ is projection/clipping and $\eta_\lambda$ is a small dual step size.

**Implementation notes.**
- Compute $\mathcal{C}_i$ on the same batch as the primal loss; use `detach()` for the dual update so gradients do not flow into $\theta$ through $\lambda$.
- Use a separate optimizer (often much slower than the primal optimizer) and cap $\lambda$; if $\lambda_i$ repeatedly hits $\lambda_{\max}$, treat it as an **unsatisfied hard constraint** and halt or change architecture rather than silently continue.

```python
# Sketch: projected dual ascent with detached violations
violations = {name: (C - eps[name]).detach() for name, C in C_values.items()}
for name, v in violations.items():
    lambda_val[name] = torch.clamp(lambda_val[name] + eta_lambda[name] * v, 0.0, lambda_max[name])
```

This is the same mathematical pattern used to tune entropy coefficients or KL constraints in modern RL (e.g. SAC-style automatic entropy tuning) {cite}`haarnoja2018soft`.

#### 3.5.2 Method B: Setpoint Controllers (PI/PID Regulation)

Some metrics should be regulated around a target value or rate. Typical examples:
- **Policy KL per update** (trust region): keep $D_{KL}(\pi_t\Vert\pi_{t-1})$ in a target band.
- **Entropy / mixing:** keep $H(\pi(\cdot\mid K))$ within a target range.
- **Code usage:** keep $H(K)$ away from collapse and away from saturation.

Let $m_t$ be a measured scalar metric and $m^\star$ its target. Define the error $e_t := m^\star - m_t$. A discrete PID update for a positive coefficient $\lambda$ is:
$$
\lambda_{t+1}
=
\Pi_{[\lambda_{\min},\lambda_{\max}]}\!\Big(
\lambda_t
+
K_p e_t
+
K_i \sum_{\tau\le t} e_\tau
+
K_d(e_t-e_{t-1})
\Big).
$$

**Sign discipline.** Choose the loss term so that increasing $\lambda$ pushes the metric in the desired direction. For example, if you want higher entropy when it is too low, include $\lambda_{\text{ent}}\,(-H(\pi))$ in the minimized loss: increasing $\lambda_{\text{ent}}$ increases the incentive to raise $H$.

In practice, PI control (no derivative term) is often sufficient; add derivative damping only if oscillations are observed.

This “multiplier as controller” viewpoint is used directly in PID Lagrangian methods for constrained RL {cite}`stooke2020responsive`.

#### 3.5.3 Method C: Learned Precisions (Homoscedastic Uncertainty Weighting)

When combining multiple likelihood-style losses with different natural scales (e.g., reconstruction vs dynamics prediction vs auxiliary self-supervision), it is often better to learn their relative weights as **inverse variances** (precisions) rather than choose them by hand.

Assume each loss $\mathcal{L}_i$ is (or is proportional to) a negative log-likelihood with unknown homoscedastic noise variance $\sigma_i^2$. Learning $s_i:=\log\sigma_i^2$ yields the objective {cite}`kendall2018multi`:
$$
\mathcal{L}_{\text{total}}
=
\sum_i \frac12\Big(\exp(-s_i)\,\mathcal{L}_i + s_i\Big).
$$
The effective weight is $\exp(-s_i)$, and the $s_i$ term prevents degenerate solutions where all weights collapse to zero.

This method is appropriate for **multi-task scaling**, not for nonnegotiable safety constraints (use Method A for those).

#### 3.5.4 Recommended Mixing (Practical Policy)

| Term type | Examples in this document | Mechanism |
|---|---|---|
| **Objective anchor** | task loss / return surrogate | fixed scale (e.g., 1.0) |
| **Hard constraints** | enclosure/closure, Lyapunov defects, saturation, budget exceedance | Method A (primal–dual) |
| **Setpoints / regulators** | entropy targets, KL-per-update target, code usage target | Method B (PI/PID) |
| **Multi-task likelihood balancing** | recon vs prediction vs auxiliary SSL losses | Method C (learned precisions) |

#### 3.5.5 Calibrating Tolerances $\epsilon_i$ (Feasibility and Units)

Dual methods only work if constraints are **feasible**: setting $\epsilon_i$ below the system’s achievable resolution forces multipliers to diverge and produces brittle training.

A practical, implementable calibration procedure is:
1. **Collect a calibration buffer** $\mathcal{D}_{\text{cal}}$ by running a baseline policy for $N$ steps (random, scripted, or a known-safe controller), logging the metrics used in your Gate Nodes / Barriers.
2. **Estimate achievable baselines** by computing empirical summaries for each metric (median, quantiles, MAD).
3. **Set tolerances** using quantiles plus a margin:
   $$
   \epsilon_i := Q_p\!\big(\mathcal{C}_i(\mathcal{D}_{\text{cal}})\big) + \Delta_i,
   $$
   with $p$ chosen by strictness (e.g. $p=0.9$ for “usually satisfied”, $p=0.99$ for “almost always satisfied”).

**Calibration phases (practical).**
1. **Empirical floors (data stream):** for losses tied to prediction/reconstruction, estimate what is achievable on $\mathcal{D}_{\text{cal}}$ with a low-capacity baseline or an ensemble (a proxy for irreducible/aleatoric error).
2. **Architecture bounds:** for discrete/finite-capacity objects, compute tolerances analytically (e.g., code usage, sampling noise floors).
3. **Requirements:** for safety budgets (risk, decay rates), set $\epsilon$ from task-level specifications.

| Constraint metric $\mathcal{C}_i$ | Units | Example tolerance choice $\epsilon_i$ | Notes |
|---|---:|---|---|
| Reconstruction NLL / distortion | nat | $Q_{0.9}(\mathcal{L}_{\text{recon}}(\mathcal{D}_{\text{cal}}))$ | Prefer likelihood losses (nats); if using MSE, fix/learn the scale (Method C). |
| One-step prediction NLL | nat/step | $Q_{0.9}(\mathcal{L}_{\text{pred}}(\mathcal{D}_{\text{cal}}))$ | Use an ensemble baseline to separate reducible vs irreducible error. |
| Macro closure defect (e.g. $H(K_{t+1}\!\mid K_t,a_t)$ surrogate) | nat | baseline Markov predictor + margin | Prevents “macro depends on micro” failure (Section 2.8). |
| KL-per-update (trust region) | nat | $\epsilon_{\text{KL}}\approx c/B$ | Sampling noise scales as $O(1/B)$ for batch size $B$. |
| Code usage gap $\log|\mathcal{K}|-H(K)$ | nat | $-\log(1-\rho_{\text{dead}})$ | Purely architectural. |
| Numerical residuals (orthogonality, symmetry) | dimensionless | $\approx 10^{-6}$ (float32) | Treat as a numeric floor, not a learnable target. |
| Lyapunov decay margin | step$^{-1}$ | $1/\tau$ | “Stabilize in $\tau$ steps” requirement. |
| Risk/cost budget | nat | $V_{\max}$ from spec | If interpreted as log-risk, map probabilities via $-\log p$. |

**Architecture-derived tolerances (often better than environment guesses).**
- **Codebook usage.** If you allow a dead-code fraction $\rho_{\text{dead}}$, then “not too collapsed” can be stated as
  $$
  H(K)\ \ge\ \log\!\big((1-\rho_{\text{dead}})\,|\mathcal{K}|\big)
  \quad\Longleftrightarrow\quad
  \log|\mathcal{K}|-H(K)\ \le\ -\log(1-\rho_{\text{dead}}).
  $$
  With $\rho_{\text{dead}}=0.05$, the right-hand side is $\approx 0.051$ nats.
- **Sampling noise floors.** For per-update KL constraints estimated from batches, a typical noise scale is $O(1/B)$, so a conservative starting tolerance is $\epsilon_{\text{KL}}\approx c/B$ with $c\in[0.5,5]$ depending on variance.

**Safety budgets.** Budgets like $V_{\max}$ are part of the task specification; if $V$ is interpreted as a log-risk surrogate, then requirements on survival probability can be mapped to nats via $-\log p$ (Section 1.2).

#### 3.5.6 Using Scaling Exponents to Gate Updates and Tune Step Sizes

The scaling exponents $(\alpha,\beta,\gamma,\delta)$ (Section 3.2) become actionable when treated as **online diagnostics** driving a simple update scheduler {cite}`konda2000actor`:
- If representation drift $\delta$ is high, freeze downstream learning (policy/critic/world) until the shutter stabilizes.
- If world-model volatility $\gamma$ is high, avoid policy learning on shifting dynamics (freeze or reduce policy step size).
- If the policy update scale $\beta$ exceeds critic signal strength $\alpha$ (BarrierTypeII), skip policy updates until the critic recovers.

One implementable pattern is a “gate + ratio” rule with EMA-smoothed exponents:
```python
# Sketch: gate policy updates if actor outruns critic
alpha = ema(alpha)          # critic signal / curvature proxy
beta  = ema(beta_kl)        # mean KL(π_t || π_{t-1}) per update
gamma = ema(world_drift)    # WM parameter drift proxy
delta = ema(code_drift)     # codebook/encoder drift proxy

if delta > delta_max:
    lr_policy = 0.0
    lr_world *= 0.5
    lr_critic *= 0.5
elif gamma > gamma_max:
    lr_policy = 0.0
elif beta > min(beta_max, alpha):
    lr_policy *= 0.98
    lr_critic *= 1.02
```

This is not ad-hoc tuning; it is a direct operationalization of the two-time-scale requirement already encoded as BarrierTypeII (Section 4.1.2).

---

## 4. Limits: Barriers (The Limits of Control)

Barriers represent the fundamental limits of the control loop.

| Barrier ID | Name | Bottleneck | Limit | Mechanism | Regularization Factor ($\mathcal{L}_{\text{barrier}}$) | Compute |
|------------|---------------|------------|----------------------------|-----------|-------------------------------------------------------|---------|
| **BarrierSat** | Saturation | **Policy** | **Actuator Saturation** | Policy cannot output enough control authority to counter disturbance. | $\Vert \pi(s) \Vert < F_{\text{max}}$ (Soft Clipping) | $O(BA)$ ✓ |
| **BarrierCausal** | Causal Censor | **World Model** | **Computational Horizon** | Failure happens faster than WM can predict/compute. | $T_{\text{horizon}}$ (Discount Factor $\gamma < 1$) | $O(1)$ ✓ |
| **BarrierScat** | Representation Collapse | **VQ-VAE** | **Grounding Loss** | Symbol channel loses grounding; macrostates become noise-like. | $\mathrm{ReLU}(\epsilon-I(X;K))^2 + \mathrm{ReLU}(H(K)-(\log|\mathcal{K}|-\epsilon))^2$ (Window Penalty) | $O(B)$ ✓ |
| **BarrierTypeII** | Type II Exclusion | **Critic/Policy** | **Scaling Mismatch** | $\beta>\alpha$ (Policy update scale outruns critic signal). | $\max(0, \beta - \alpha)$ (Scaling Penalty) | $O(P)$ ⚡ |
| **BarrierVac** | Model Stability Limit | **World Model** | **Regime Stability** | Operational mode is metastable; WM predicts collapse. | $\Vert \nabla^2 V(s) \Vert$ (Hessian Regularization) | $O(BZ^2)$ ✗ |
| **BarrierCap** | Capacity | **Policy** | **Fundamental Uncontrollability** | "Bad" region is too large for Policy to steer around. | $V(s) \to \infty$ for $s \in \text{Bad}$ (Safe RL) | $O(B)$ ⚡ |
| **BarrierGap** | Spectral Gap | **Critic** | **Convergence Stagnation** | Error surface is too flat ($\nabla V \approx 0$). | $\max(0, \epsilon - \Vert \nabla V \Vert)$ (Stiffness) | $O(BZ)$ ✓ |
| **BarrierAction** | Action Gap | **Critic** | **Cost Prohibitive** | Correct move requires more cost budget ($V$) than affordable. | $\Vert \nabla_\pi V(s, \pi) \Vert$ (Action Gradient) | $O(BAZ)$ ⚡ |
| **BarrierOmin** | O-Minimal | **World Model** | **Model Mismatch** | World exhibits non-smooth or non-stationary structure outside the WM class. | $\Vert \nabla S_t \Vert$ for O-Minimality (Lipschitz) | $O(ZP_{WM})$ ⚡ |
| **BarrierMix** | Mixing | **Policy** | **Exploration Trap** | Policy gets stuck in a local loop. | $-H(\pi)$ (Entropy Bonus) | $O(BA)$ ✓ |
| **BarrierEpi** | Epistemic | **VQ-VAE/WM** | **Information Overload** | Environment complexity exceeds $\log|\mathcal{K}|$ and/or WM class; closure breaks. | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{Sync}_{K-W}}$ (Distortion + Closure) | $O(BD)$ ✓ |
| **BarrierFreq** | Frequency | **World Model** | **Loop Instability** | Positive feedback causes oscillation amplification. | $\Vert J_{WM} \Vert < 1$ (Jacobian Spectral Norm) | $O(Z^2)$ ✗ |
| **BarrierBode** | Bode Sensitivity | **Policy** | **Waterbed Effect** | Suppressing error in one domain increases it in another. | $\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.}$ (Bode sensitivity integral) | FFT ✗ |
| **BarrierInput** | Input Stability | **All** | **Resource Exhaustion** | Agent runs out of battery/compute/tokens. | $\text{Cost}(s) > \text{Budget}$ (Resource Penalty) | $O(B)$ ✓ |
| **BarrierVariety** | Requisite Variety | **Policy** | **Ashby's Deficit** | Policy states < Disturbance states. | $\dim(Z) \ge \dim(\mathcal{X})$ (Width Penalty) | $O(1)$ ✓ |
| **BarrierLock** | Exclusion | **World Model** | **Hard-Coded Safety** | Safety interlock successfully prevents illegal state. | $\mathbb{I}(s \in \text{Forbidden}) \cdot \infty$ | $O(B)$ ✓ |

**Compute Legend:** ✓ Easy (<10% overhead) | ⚡ Medium (10-50% overhead) | ✗ Hard (>50% overhead or infeasible)

### 4.1 Barrier Implementation Details

Implementing these barriers requires rigorous cybernetic engineering. We divide them into **Single-Barrier Limits** and **Cross-Barrier Dilemmas**.

#### A. Single-Barrier Enforcement (Hard Constraints)

1.  **BarrierSat (Actuator Limit):**
    *   *Constraint:* $\|\pi(s)\| \le F_{max}$.
    *   *Implementation:* **Squashing Function**. Use `tanh` on the policy mean: $\mu(s) = F_{max} \cdot \tanh(f_\theta(s))$. Do not rely on clipping losses alone; the architecture must be incapable of exceeding limits.

2.  **BarrierTypeII (Scaling Mismatch):**
    *   *Constraint:* $\alpha > \beta$ (Critic is steeper than Policy).
    *   *Implementation:* **Two-Time-Scale Updating**.
        *   If $\text{Scale}(\text{Critic}) \le \text{Scale}(\text{Policy})$, **skip** the Policy update step ($k_\pi = 0$).
        *   Resume policy updates only when the Critic has re-established a valid gradient (restored a usable value landscape).

3.  **BarrierOmin (Tameness):**
    *   *Constraint:* $\|S\|_{Lip} \le K$.
    *   *Implementation:* **Spectral normalization** bounds the operator norm of each linear layer. With 1-Lipschitz activations, this upper-bounds the network Lipschitz constant by the product of per-layer spectral norms; choose per-layer caps so the implied global bound is $\le K$ {cite}`miyato2018spectral`.

4.  **BarrierGap (Spectral Gap):**
    *   *Constraint:* $\|\nabla V\| \ge \epsilon$ (No flat plateaus).
    *   *Implementation:* **Gradient Penalty**.
        $$ \mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} [(\|\nabla_{\hat{s}} V(\hat{s})\| - K)^2] $$
        Gradient-norm penalties discourage vanishing gradients on sampled points and help avoid large flat regions; they do not provide a global guarantee without additional assumptions {cite}`gulrajani2017improved`.

#### B. Cross-Barrier Regularization (Cybernetic Dilemmas)

The most dangerous failures occur when barriers conflict. We model these as **Trade-off Functionals**:

1.  **The Information-Control Tradeoff (BarrierScat vs BarrierCap):**
    *   *Classes:* **Rate-Distortion Optimization.**
    *   *Conflict:* High compression (anti-collapse) removes details needed for fine control (capacity/controllability).
    *   *Regularization:*
        $$
        \mathcal{L}_{\text{InfoControl}}
        =
        \underbrace{\beta_K\,\mathbb{E}[-\log p_\psi(K)] + \beta_\mu D_{KL}(q(z_\mu \mid x)\Vert p(z_\mu))}_{\text{Compression (Rate)}}
        +
        \underbrace{\gamma\,\mathbb{E}[\mathfrak{D}(Z,A)]}_{\text{Control Effort}}
        $$
        where $\mathfrak{D}$ is an actuation cost (e.g. KL-control to a prior $\pi_0$, or a calibrated norm/penalty on actions).
    *   *Mechanism:* Use Lagrange multipliers to find the Pareto frontier. If control performance drops, decrease $\beta_K,\beta_\mu$ (allocate more bits to the shutter).

2.  **The Stability-Plasticity Dilemma (BarrierVac vs BarrierPZ):**
    *   *Conflict:* A stable World Model (model stability limit) resists updating to new dynamics (plasticity / Zeno).
    *   *Regularization:* **Elastic Weight Consolidation (EWC)**.
        $$ \mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,old})^2 $$
    *   *Mechanism:* The Fisher Information Matrix $F_i$ measures "stiffness". We allow plastic changes in unimportant weights, but rigidly enforce stability in structural weights.

3.  **The Sensitivity Integral (BarrierBode):**
    *   *Conflict:* Suppressing error in one frequency band amplifies it in another (Bode sensitivity integral constraint: $\int_{0}^{\infty} \log |S(j\omega)| d\omega = \text{const.}$; equal to $0$ under standard stable/minimum-phase assumptions).
    *   *Regularization:* **Frequency-Weighted Cost**.
        $$ \mathcal{L}_{\text{Bode}} = \| \mathcal{F}(e_t) \cdot W(\omega) \|^2 $$
    *   *Mechanism:* Explicitly decide *where* to be blind. We penalize high-frequency errors heavily (instability) while accepting low-frequency drift (steady-state error), or vice versa.

---

## 5. Failure Modes (Observed Pathologies)

When Limits are breached or Interfaces fail, the agent exhibits specific pathologies.

| Mode | Standard Name | Failed Component | Fragile (Pathology) Name | Description |
|------|---------------|-----------------|--------------------------|-------------|
| **D.D** | Dispersion-Decay | **All (Optimal)** | **Success (Convergence)** | Agent solves task; error drops to a stable floor. |
| **S.E** | Subcritical-Equilib | **Policy** | **Curriculum Stumble** | Difficulty ramps up too fast for adaptation. |
| **C.D** | Conc-Dispersion | **Policy/Shutter** | **Mode Collapse / Obsession** | Agent over-focuses on one aspect, ignores rest. |
| **C.E** | Conc-Escape | **Policy/Critic** | **Divergence / Blow-up** | Gradients/activations diverge; optimization becomes unstable. |
| **T.E** | Topo-Extension | **Shutter/WM** | **Wrong Paradigm** | Architecture is topologically insufficient. |
| **S.D** | Struct-Dispersion | **Shutter** | **Symmetry Blindness** | Fails to exploit available symmetries. |
| **C.C** | Event Accumulation | **Policy/WM** | **Decision Paralysis** | Input happens faster than decision loop (Zeno). |
| **T.D** | Glassy Freeze | **Policy** | **Learned Helplessness** | Agent finds suboptimal safe spot, refuses to move. |
| **D.E** | Oscillatory | **Policy** | **Pilot-Induced Oscillation** | Overcorrection causes increasing instability. |
| **T.C** | Labyrinthine | **World Model** | **Overfitting to Noise** | WM models noise instead of signal. |
| **D.C** | Semantic Horizon | **Shutter/WM** | **Ungrounded inference** | Distribution shift causes internal rollouts to decouple from boundary evidence. |
| **B.E** | Sensitivity Expl. | **Critic** | **Fragility** | Optimization for one condition makes agent ultra-fragile. |
| **B.D** | Resource Depletion | **Boundary/Shutter** | **Starvation** | Running out of inputs/power. |
| **B.C** | Control Deficit | **Policy** | **Overwhelmed** | Disturbance more complex than controller (Ashby). |

---

## 6. Interventions (Mitigations)

Interventions are external mitigations to restore stability, re-ground the representation, or reduce unsafe update rates.

| Surgery ID | Target Mode | Target Component | Fragile (Upgrade) Translation | Mechanism |
|------------|-------------|------------------|-------------------------------|-----------|
| **SurgCE** | C.E (Divergence) | **Policy/Critic** | **Limiter / trust region** | **Gradient Clipping / Trust Region:** Clamp outputs; enforce $\Vert \pi_{new} - \pi_{old} \Vert < \delta$. |
| **SurgCC** | C.C (Zeno) | **WM/Policy** | **Time-boxing / Rate Limit** | **Skip-Frame / Latency:** Force fixed $\Delta t$; ignore inputs during cool-down. |
| **SurgCD_Alt**| C.D (Obsession)| **Policy** | **Reset / Reshuffling** | **Re-initialization:** Reset parameters of the obsession-locked sub-module to random. |
| **SurgSE** | S.E (Stumble) | **World Model** | **Curriculum Ease-off** | **Curriculum Learning:** Reduce Task Difficulty or Rewind to earlier level. |
| **SurgSC** | S.C (Instability)| **Critic** | **Parameter Freezing** | **Target Network Freeze:** Stop updating Target V; switch to slower exponential moving average. |
| **SurgCD** | C.D (Collapse) | **Shutter** | **Feature Pruning** | **Dead Code Pruning:** Identify and excise unused macro symbols / dead fibres. |
| **SurgSD** | S.D (Blindness) | **Shutter** | **Augmentation / Ghost Vars** | **Domain Randomization:** Inject noise into $x$ to force the shutter to learn robust macrostates. |
| **SurgTE** | T.E (Paradigm) | **Shutter/WM** | **Architecture Search** | **Neural Architecture Search (NAS):** Modify shutter+WM class to match topology (e.g., add hierarchy / memory). |
| **SurgTC** | T.C (Overfit) | **WM** | **Regularization** | **Weight Decay / Dropout:** Increase $\lambda \|\theta\|^2$ penalty. |
| **SurgTD** | T.D (Helplessness)| **Policy** | **Noise Injection** | **Parameter Space Noise:** Add $\xi \sim \mathcal{N}(0, \Sigma)$ to Policy weights. |
| **SurgDC** | D.C (Ungrounded) | **Shutter/WM** | **Smoothing / fallback** | **OOD rejection:** if micro surprisal spikes (e.g. $D_{KL}(q(z_\mu\mid x)\Vert p(z_\mu))>\tau$) and/or macro surprisal spikes (e.g. $-\log p_\psi(K)>\tau_K$), trigger fallback (safe stop). |
| **SurgDE** | D.E (Oscillate) | **Policy** | **Damping** | **Momentum Reduction:** Decrease Adam $\beta_1$ or increase batch size. |
| **SurgBE** | B.E (Fragile) | **Critic** | **Saturation / Anti-Windup** | **Spectral Normalization:** Constrain Lipschitz constant of $V(s)$. |
| **SurgBD** | B.D (Starve) | **Boundary/Shutter** | **Replay Buffer / Reservoir** | **Experience Replay:** Train on historical buffers to prevent catastrophic forgetting. |
| **SurgBC** | B.C (Deficit) | **Policy** | **Controller Expansion** | **Width Expansion:** Dynamically add neurons to the Policy network (Net2Net). |

---

## 7. Computational Considerations

This section provides rigorous cost analysis for implementing the Fragile Agent regulation framework, enabling practitioners to make informed trade-offs between safety coverage and computational overhead.

### 7.1 Interface Cost Summary

| Tier | Interfaces | Total Overhead | Failure Modes Prevented |
|------|-----------|----------------|-------------------------|
| **Essential** | CostBoundCheck, ZenoCheck, CompactCheck, ErgoCheck, ComplexCheck, StiffnessCheck | ~10% | 6/14 |
| **Important** | ScaleCheck, GeomCheck, OscillateCheck, ParamCheck | ~25% | 9/14 |
| **Advanced** | TopoCheck, TameCheck, BifurcateCheck, AlignCheck | ~60%+ | 13/14 |

### 7.2 Barrier Cost Summary

| Tier | Barriers | Implementation | Notes |
|------|----------|----------------|-------|
| **Architectural** | BarrierSat, BarrierVariety | Built-in (tanh, dim) | Zero runtime cost |
| **Standard RL** | BarrierMix, BarrierCausal, BarrierScat, BarrierEpi | Standard losses | Already in most implementations |
| **Specialized** | BarrierTypeII, BarrierOmin, BarrierGap | Medium cost | Requires auxiliary computation |
| **Infeasible** | BarrierBode, BarrierFreq, BarrierVac | See Section 8 | Need replacements |

### 7.3 Synchronization Loss Costs

| Sync Pair | Formula | Time Complexity | Implementation |
|-----------|---------|-----------------|----------------|
| **Shutter ↔ WM** | $\mathrm{CE}(K_{t+1},\hat{p}_\phi(K_{t+1}\mid K_t,A_t))$ | $O(B|\mathcal{K}|)$ | Easy - closure cross-entropy |
| **Critic ↔ Policy** | TD-Error + $\Delta A = \lvert A^\pi - A^{\text{Buffer}} \rvert$ | $O(B)$ | Easy - track advantage gap |
| **WM ↔ Policy** | $\mathbb{E}_{z \sim \pi}[\mathcal{L}_{\text{pred}}(z)]$ | $O(HBZ)$ | Medium - requires on-policy rollouts |

### 7.4 Implementation Tiers

#### Tier 1: Core Fragile Agent (~15% overhead)
For production systems with tight compute budgets.

**Loss Function:**
$$
\mathcal{L}_{\text{Fragile}}^{\text{core}} = \mathcal{L}_{\text{task}} + \lambda_{\text{shutter}} \mathcal{L}_{\text{shutter}} + \lambda_{\text{ent}} (-H(\pi)) + \lambda_{\text{zeno}} D_{KL}(\pi_t \Vert \pi_{t-1}) + \lambda_{\text{stiff}} \max(0, \epsilon - \Vert \nabla V \Vert)^2
$$

**Coverage:** Prevents Mode C.E (Blow-up), C.C (Zeno), C.D (Collapse), D.C (Ungrounded inference), T.D (Freeze), S.D (Blindness)

**Implementation:**
```python
def compute_fragile_core_loss(
    task_loss: torch.Tensor,
    shutter_loss: torch.Tensor,
    policy_logits: torch.Tensor,
    prev_policy_logits: torch.Tensor,
    critic_values: torch.Tensor,
    states: torch.Tensor,
    lambda_shutter: float = 1.0,
    lambda_ent: float = 0.01,
    lambda_zeno: float = 0.1,
    lambda_stiff: float = 0.01,
    stiff_eps: float = 0.1,
) -> torch.Tensor:
    """Core Fragile Agent loss with ~15% overhead."""

    # CompactCheck + ComplexCheck: shutter regularization (macro code + micro nuisance)
    rep_loss = shutter_loss.mean()

    # ErgoCheck + SymCheck: Policy entropy
    policy_dist = torch.distributions.Categorical(logits=policy_logits)
    entropy_loss = -policy_dist.entropy().mean()

    # ZenoCheck: Policy smoothness
    prev_dist = torch.distributions.Categorical(logits=prev_policy_logits.detach())
    zeno_loss = torch.distributions.kl_divergence(policy_dist, prev_dist).mean()

    # StiffnessCheck: Gradient penalty on critic
    states.requires_grad_(True)
    v = critic_values if critic_values.requires_grad else critic_values.detach()
    grad_v = torch.autograd.grad(
        v.sum(), states, create_graph=True, retain_graph=True
    )[0]
    grad_norm = grad_v.norm(dim=-1)
    stiff_loss = torch.relu(stiff_eps - grad_norm).pow(2).mean()

    total = (
        task_loss
        + lambda_shutter * rep_loss
        + lambda_ent * entropy_loss
        + lambda_zeno * zeno_loss
        + lambda_stiff * stiff_loss
    )
    return total
```

#### Tier 2: Standard Fragile Agent (~40% overhead)
For research and safety-conscious applications.

**Additional Terms:**
$$
\mathcal{L}_{\text{Fragile}}^{\text{std}} = \mathcal{L}_{\text{Fragile}}^{\text{core}} + \lambda_{\text{scale}} \max(0, \beta - \alpha) + \lambda_{\text{sync}}\,\mathcal{L}_{\text{Sync}_{K-W}} + \lambda_{\text{osc}} \Vert z_t - z_{t-2} \Vert
$$

**Additional Implementation (Diagnostics Only):**
```python
class ScalingExponentTracker:
    """
    Track α (curvature proxy) and β (policy-change proxy) for diagnostics.

    Note: this estimates parameter-space proxies for monitoring training health.
    It is not the state-space metric G; compute G via compute_state_space_fisher()
    in state space (Section 2.6).
    """
    def __init__(self, ema_decay: float = 0.99):
        self.alpha_ema = 2.0  # Default quadratic
        self.beta_ema = 2.0
        self.ema_decay = ema_decay
        self.log_losses = []
        self.log_param_norms = []

    def update(self, loss: float, model: nn.Module, grad_norm: float = None):
        # α estimation: log-linear regression of loss vs param norm
        param_norm = sum(p.pow(2).sum() for p in model.parameters()).sqrt().item()

        if loss > 0 and param_norm > 0:
            self.log_losses.append(np.log(loss))
            self.log_param_norms.append(np.log(param_norm))

        if len(self.log_losses) >= 20:
            # Fit α via least squares
            x = np.array(self.log_param_norms[-100:])
            y = np.array(self.log_losses[-100:])
            alpha_raw = np.polyfit(x - x.mean(), y - y.mean(), 1)[0]
            self.alpha_ema = self.ema_decay * self.alpha_ema + (1 - self.ema_decay) * alpha_raw

            # β from gradient scaling (if provided)
            if grad_norm is not None and grad_norm > 0:
                beta_raw = 2.0  # Approximate, could refine
                self.beta_ema = self.ema_decay * self.beta_ema + (1 - self.ema_decay) * beta_raw

        return self.alpha_ema, self.beta_ema

    def get_barrier_loss(self) -> float:
        """BarrierTypeII: max(0, β - α)"""
        return max(0.0, self.beta_ema - self.alpha_ema)


def compute_shutter_wm_sync_loss(
    K_next: torch.Tensor,          # shutter(x_{t+1}) → LongTensor macro code
    K_next_logits: torch.Tensor,   # WM(K_t, a_t) → logits over codes
) -> torch.Tensor:
    """Shutter ↔ WM synchronization loss (macro closure)."""
    return F.cross_entropy(K_next_logits, K_next)


def compute_oscillation_loss(
    z_t: torch.Tensor,
    z_history: List[torch.Tensor],  # [z_{t-1}, z_{t-2}, ...]
) -> torch.Tensor:
    """OscillateCheck: Period-2 oscillation penalty."""
    if len(z_history) < 2:
        return torch.tensor(0.0, device=z_t.device)
    z_t_minus_2 = z_history[-2]
    return (z_t - z_t_minus_2).pow(2).mean()
```

#### Tier 3: Full Fragile Agent (~80% overhead)
For safety-critical applications with verification requirements.

**Additional Terms:**
$$
\mathcal{L}_{\text{Fragile}}^{\text{full}} = \mathcal{L}_{\text{Fragile}}^{\text{std}} + \lambda_{\text{lip}} \mathcal{L}_{\text{Lipschitz}} + \lambda_{\text{geo}} \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{gain}} \mathcal{L}_{\text{gain}}
$$

See Section 8 for efficient implementations of the expensive terms.

#### Tier 4: Riemannian Fragile Agent (Covariant Updates)

This tier implements a **Riemannian / information-geometric** view, replacing Euclidean losses with geometry-aware equivalents. This approach is inspired by Natural Gradient methods {cite}`amari1998natural,martens2015kfac,martens2020natural` and Safe RL literature {cite}`chow2018lyapunov,kolter2019safe`.

**Key Insight (State-Space Fisher):** The Covariant Regulator uses the **State-Space Fisher Information** to scale the Lie Derivative. This measures how sensitively the policy responds to changes in the latent state $z$—NOT how the parameters $\theta$ affect the policy (which is what TRPO/PPO use). See Section 2.6 for the critical distinction between these geometries.

**A. compute_natural_gradient_loss(): Geometry-Aware Value Decrease**

```python
def compute_natural_gradient_loss(
    regulator: HypostructureRegulator,  # Agent with policy and critic
    state: torch.Tensor,                # z_t (latent state)
    policy_action: torch.Tensor,        # a_t from Policy(z_t)
    next_state: torch.Tensor,           # z_{t+1}
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Computes a geometry-aware value-decrease objective connecting Policy and Value.

    EUCLIDEAN (Standard RL):
        L = -log_prob * advantage  # Ignores geometry entirely

    RIEMANNIAN (This function):
        L = -<grad_V, velocity>_G  # Inner product under sensitivity metric

    The key difference: geometry-aware updates scale by the inverse local
    sensitivity/conditioning. In high-sensitivity regions (large G), steps shrink;
    in low-sensitivity regions, steps can be larger.

    Important: the metric G here is a state-space sensitivity metric (∂log π/∂z),
    not the parameter-space Fisher (∂log π/∂θ). See Section 2.6 for the distinction.
    """
    # 1. Compute State-Space Fisher Metric G (Diagonal Approximation)
    # G_ii = E[(∂log π/∂z_i)²] — measures control authority at each state dim
    fisher_diag = compute_state_space_fisher(regulator, state, include_value_hessian=False)
    metric_inv = 1.0 / (fisher_diag + epsilon)  # G^{-1}

    # 2. Compute the Value Gradient (nabla_z V)
    state_grad = state.detach().clone().requires_grad_(True)
    value_est = regulator.critic(state_grad)
    grad_v = torch.autograd.grad(
        outputs=value_est.sum(),
        inputs=state_grad,
        create_graph=True,
    )[0]  # [Batch, Latent_Dim]

    # 3. Compute State Velocity (z_dot)
    state_velocity = next_state - state  # [Batch, Latent_Dim]

    # 4. Compute the Natural Inner Product (Covariant Derivative)
    # EUCLIDEAN would be: (grad_v * state_velocity).sum()
    # RIEMANNIAN: weight by inverse metric
    natural_decrease = (grad_v * state_velocity * metric_inv).sum(dim=-1)

    # 5. The Loss: maximize value decrease (make V decrease fast)
    return -natural_decrease.mean()
```

**B. compute_control_theory_loss(): Neural Lyapunov with Sensitivity Metric**

```python
def compute_control_theory_loss(
    regulator: HypostructureRegulator,  # Agent with policy and critic
    states: torch.Tensor,               # z_t (latent state)
    next_states: torch.Tensor,          # z_{t+1}
    lambda_lyapunov: float = 1.0,
    target_decay: float = 0.1,          # alpha in Lyapunov constraint
    metric_mode: str = "state_fisher",  # Sensitivity metric (Fisher + optional value Hessian)
) -> torch.Tensor:
    """
    Implements Neural Lyapunov Control with a state-space sensitivity metric.

    Combines two constraints:
    1. Geometry-aware value decrease: policy loss scaled by geometry
    2. Lyapunov stability: critic enforces V_dot <= -alpha * V

    Important: the metric G is computed in state space (∂log π/∂z), not
    parameter space. See Section 2.6 for the distinction.
    """
    # 1. Compute state-space metric G (Fisher + optional value Hessian)
    if metric_mode == "state_fisher":
        g_metric = compute_state_space_fisher(regulator, states, include_value_hessian=True)
    else:
        g_metric = compute_state_space_fisher(regulator, states, include_value_hessian=False)
    metric_inv = 1.0 / (g_metric + 1e-6)

    # 2. Compute Time-Derivative of Value (V_dot)
    states_grad = states.detach().clone().requires_grad_(True)
    critic_values = regulator.critic(states_grad)
    grad_v = torch.autograd.grad(
        critic_values.sum(), states_grad, create_graph=True
    )[0]

    # 3. Geometry-aware value decrease (Policy Loss)
    # EUCLIDEAN: value_change = (grad_v * dynamics).sum()
    # RIEMANNIAN: scale by inverse metric
    dynamics = next_states - states
    value_change_geo = (grad_v * dynamics * metric_inv).sum(dim=-1)
    loss_policy = -value_change_geo.mean()

    # 4. Lyapunov Constraint (Critic Loss)
    # Ensure V_dot <= -alpha * V (Exponential Stability)
    # Penalize violations: ReLU(V_dot + alpha * V)^2
    v_dot = (grad_v * dynamics).sum(dim=-1)
    violation = torch.relu(v_dot + target_decay * critic_values.squeeze())
    loss_critic_lyapunov = violation.pow(2).mean()

    return loss_policy + lambda_lyapunov * loss_critic_lyapunov
```

**C. GeometryAwareLearner: Complete Training Loop**

```python
class GeometryAwareLearner:
    """
    Complete training loop implementing geometry-aware control updates.

    Three-phase update:
    1. Critic update: learn the value / Lyapunov landscape
    2. Metric estimate: compute state-space Fisher sensitivity
    3. Actor update: maximize value decrease under the metric

    Difference from Standard RL:
    - Standard: Maximize Q(s,a) (scalar value)
    - Geometry-aware: Maximize value decrease <grad_V, velocity>_G (metric-weighted dot product)

    Important: the metric G is computed in state space (∂log π/∂z), not
    parameter space. See Section 2.6 for the distinction.
    """

    def __init__(self, actor, critic, world_model, config):
        self.actor = actor
        self.critic = critic
        self.world_model = world_model

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)

        self.gamma = config.gamma
        self.device = config.device

    def train_step(self, batch):
        """
        Performs one Cybernetic Update step.
        Batch: (state, action, reward, next_state, done)
        """
        s, a, r, s_next, d = [x.to(self.device) for x in batch]

        # Phase 1: Critic update (learn cost/value)
        # Convert reward to cost (we minimize risk/cost)
        cost = -r

        # TD-Learning (Bellman Update)
        with torch.no_grad():
            target_v = cost + self.gamma * self.critic(s_next) * (1 - d)

        current_v = self.critic(s)
        critic_loss = nn.MSELoss()(current_v, target_v)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Phase 2: Metric estimation (state-space Fisher sensitivity)
        metric_g = self._compute_state_fisher(s)

        # Phase 3: Actor update (geometry-aware)
        # Goal: maximize value decrease under the metric

        for p in self.critic.parameters():
            p.requires_grad = False

        s.requires_grad_(True)
        val = self.critic(s)
        grad_v = torch.autograd.grad(val.sum(), s, create_graph=True)[0]

        pred_action = self.actor(s)
        s_velocity = self.world_model(s, pred_action) - s

        # Geometry-aware: value change = <Grad_V, Velocity>_G (weighted by sensitivity)
        # EUCLIDEAN would be: value_change = (grad_v * s_velocity).sum()
        value_change_geo = (grad_v * s_velocity / (metric_g + 1e-6)).sum(dim=-1)

        actor_loss = -value_change_geo.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _compute_state_fisher(self, state):
        """
        Computes the State-Space Fisher Information G.

        G_ii = E[(∂log π/∂z_i)²]

        Important: this is the state-space Fisher (how the policy changes with state),
        not the parameter-space Fisher (how the policy changes with weights).
        See Section 2.6 for the distinction.
        """
        state_grad = state.detach().clone().requires_grad_(True)
        action_mean = self.actor(state_grad)
        # Assuming Gaussian policy with fixed std
        action_std = torch.ones_like(action_mean) * 0.5
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        grad_z = torch.autograd.grad(log_prob.sum(), state_grad, create_graph=False)[0]
        fisher_diag = grad_z.pow(2).mean(dim=0)
        return fisher_diag + 1e-6
```

**D. RiemannianFragileAgent (Algorithm 3): The Complete Specification**

```python
class RiemannianFragileAgent(nn.Module):
    """
    Algorithm 3: The Riemannian Fragile Agent

    Notation:
    - Z: Latent state space (statistical manifold)
    - G: State-space sensitivity metric (Fisher + optional value Hessian)
    - z_macro, z_micro: Macro (predictive) and Micro (nuisance) coordinates
    - Ω: Regime indicator (from monitors)

    This algorithm combines macro/micro separation (Section 2.2b), the Sieve
    monitors (Sections 3–6), and Lyapunov-constrained control in a single loop.

    Key differences from standard RL:
    1. Explicit objectives: auxiliary terms are tied to measurable constraints/regularizers
    2. Metric-aware step control: update magnitudes can be scaled by a state-space metric
    3. Grounding checks: boundary-coupling and enclosure monitors limit ungrounded rollouts
    4. Geometry-aware representation: latent space may be treated as a manifold

    Important: the metric G here is a state-space sensitivity metric (e.g. Fisher/Hessian in z),
    not the parameter-space Fisher in θ. See Section 2.6 for the distinction.
    """

    def train_step(self, batch, trackers):
        # === PHASE I: SIEVE (Pre-Computation) ===
        # Before any gradient update, diagnose the regime

        # 1. Compute Levin Complexity (The Horizon)
        # K_L(τ) = -log P(τ | U) where U is universal machine
        # If K_L > C_observer (observer compute budget): stop / fallback
        regime = self.sieve.diagnose_regime(batch.trace)
        if regime == "UNDECIDABLE":
            return "STOP"
        if regime == "NOISE":
            return "REJECT"

        # === PHASE II: METRIC EXTRACTION ===
        # Compute state-space Fisher information (not optimizer statistics)
        # G_inv acts as a trust-region / step-size limit for the update
        with torch.no_grad():
            fisher_diag = compute_state_space_fisher(self, batch.obs)
            G_inv = 1.0 / (fisher_diag + 1e-8)

        # === PHASE III: SHUTTER UPDATE (VQ-VAE) ===
        # Enforce Causal Enclosure: discrete macro K carries the predictive signal,
        # while z_micro is treated as nuisance/residual information.
        K_t, z_macro, z_micro = self.shutter(batch.obs)  # z_macro := e_{K_t}

        # SYMBOLIC: Closure is cross-entropy / conditional entropy on the macro symbols.
        # (See Section 2.8: I(K_{t+1}; Z_t | K_t, A_t)=0 and H(K_{t+1}|K_t,A_t) small.)
        closure_loss = self._compute_closure_loss(K_t, z_micro)
        self.shutter_opt.step(self.shutter_loss + closure_loss)

        # Phase IV: Lyapunov update (critic)
        # Enforce Exponential Stability constraint on V
        V = self.critic(z_macro)
        V_next = self.critic(z_macro_next)
        V_dot = V_next - V

        # RIEMANNIAN: Lyapunov constraint: V_dot <= -zeta * V
        # EUCLIDEAN would just minimize TD error
        lyap_loss = torch.relu(V_dot + self.zeta * V).pow(2).mean()
        self.critic_opt.step(self.td_loss + lyap_loss)

        # === PHASE V: COVARIANT UPDATE (Policy) ===
        # Check Scaling Hierarchy (BarrierTypeII)
        alpha = trackers.get_scale('critic')  # Curvature scale (critic)
        beta = trackers.get_scale('actor')    # Exploration/update scale (actor)

        if alpha > beta:  # Critic is steeper than policy update scale
            # Calculate Natural Gradient Direction
            grad_V = torch.autograd.grad(V, z_macro)[0]
            velocity = self.world_model(z_macro, self.policy(z_macro)) - z_macro

            # Geometry-aware: value decrease weighted by sensitivity
            # EUCLIDEAN would be: L = -(grad_V * velocity).mean()
            L_nat = -torch.mean((grad_V * velocity) * G_inv)

            # Geodesic Stiffness (Zeno Constraint)
            L_zeno = self._geodesic_dist(self.policy_new, self.policy_old, G_inv)

            self.actor_opt.step(L_nat + L_zeno)
        else:
            # Policy updates too aggressive relative to critic certainty:
            # pause adaptation to let estimation catch up (wait state)
            pass

    def _compute_closure_loss(self, K_t, z_micro):
        """
        Causal Enclosure (symbolic form).

        With a discrete macro register K∈𝒦, enclosure is the pair of conditions:
        1) Predictability: H(K_{t+1} | K_t, a_t) is small (law-like macro dynamics).
        2) No leak: I(K_{t+1}; Z_{μ,t} | K_t, a_t)=0 (micro does not inform the law).

        Implementation sketch:
        - (1) is a cross-entropy loss over code indices (a Shannon quantity).
        - (2) is a conditional-independence penalty (HSIC/adversary/MINE), treated as
          a proxy for conditional mutual information.
        """
        logits_next = self.world_model.predict_code_logits(K_t, self.action)  # [B, |𝒦|]
        K_next = self.shutter.encode_code(self.next_obs)                      # [B]

        predict_loss = nn.CrossEntropyLoss()(logits_next, K_next)

        # Choose one MI proxy for the independence term:
        #   - HSIC(z_micro, one_hot(K_next))
        #   - adversary with gradient reversal predicting K_next from z_micro
        #   - variational estimator of I(K_next; z_micro | K_t, a_t)
        independence_loss = estimate_conditional_mi(K_next, z_micro, K_t, self.action)

        return predict_loss + self.lambda_ind * independence_loss

    def _geodesic_dist(self, policy_new, policy_old, G_inv):
        """
        Geodesic distance under the sensitivity metric.

        EUCLIDEAN: ||π_new - π_old||²
        RIEMANNIAN: ||π_new - π_old||²_G = (π_new - π_old)ᵀ G⁻¹ (π_new - π_old)
        """
        diff = policy_new - policy_old
        return (diff * diff * G_inv).sum(dim=-1).mean()
```

### 7.5 Cost-Benefit Decision Matrix

| Compute Budget | Recommended Tier | Key Trade-offs |
|----------------|------------------|----------------|
| **Tight (<20% overhead)** | Tier 1 | Covers basic stability; may miss scaling issues |
| **Moderate (20-50%)** | Tier 2 | Good coverage; catches most failure modes |
| **Generous (>50%)** | Tier 3 | Near-complete coverage; suitable for safety-critical |
| **Unlimited (offline)** | Full + verification | Complete formal verification possible |

### 7.6 Defect Functional Costs (from metalearning.md)

For training-time defect minimization:

| Defect | Formula | Per-Sample Cost | Batched Cost |
|--------|---------|-----------------|--------------|
| $K_C$ (Compatibility) | $\Vert S_t(u(s)) - u(s+t) \Vert$ | $O(Z)$ | $O(BZ)$ |
| $K_D$ (Value Decrease) | $\int \max(0, \partial_t \Phi + \mathfrak{D}) dt$ | $O(TZ)$ | $O(TBZ)$ |
| $K_{SC}$ (Symmetry) | $\sup_g d(g \cdot u(t), S_t(g \cdot u(0)))$ | $O(\lvert G \rvert TZ)$ | Often intractable |
| $K_{Cap}$ (Capacity) | $\int \lvert \text{cap}(\{u\}) - \mathfrak{D}(u) \rvert dt$ | $O(T)$ | $O(TB)$ |
| $K_{LS}$ (Local Structure) | Metric/norm deviations | $O(Z^2)$ | $O(BZ^2)$ |
| $K_{TB}$ (Information Bounds) | DPI violations | $O(B^2)$ | Quadratic in batch |

**Recommendation:** Use expected defect $\mathcal{R}_A(\theta) = \mathbb{E}[K_A^{(\theta)}(u)]$ with Monte Carlo sampling for tractability.

### 7.7 Tier 5: Atlas-Based Fragile Agent (Multi-Chart Architecture)

This tier introduces **manifold atlas** architecture—a principled approach for handling topologically complex latent spaces that cannot be covered by a single coordinate chart.

#### 7.7.1 Manifold Atlas Theory: Why Single Charts Fail

**The Fundamental Problem:**
A single neural network encoder defines a single coordinate chart on the latent manifold. However, many manifolds **cannot** be covered by a single chart {cite}`whitney1936differentiable,lee2012smooth`:

| Manifold | Minimum Charts | Why |
|----------|----------------|-----|
| **Sphere $S^2$** | 2 | No global flat coordinates (Hairy Ball Theorem) |
| **Torus $T^2$** | 4 | Non-trivial first homology |
| **Klein Bottle** | ∞ | Non-orientable |
| **Swiss Roll** | 1 | Topologically trivial but geometrically challenging |

**Symptoms of Single-Chart Failure:**
- Representation collapse (everything maps to one region)
- Discontinuities at chart boundaries
- Poor generalization to unseen topology
- Gradient instabilities near singularities

**The Atlas Solution:**
An **atlas** $\mathcal{A} = \{(U_i, \phi_i)\}_{i=1}^K$ is a collection of charts where:
- Each $U_i \subset M$ is an open set (region of the manifold)
- Each $\phi_i: U_i \to \mathbb{R}^d$ is a homeomorphism (local embedding)
- $\bigcup_i U_i = M$ (charts cover the entire manifold)
- Transition functions $\tau_{ij} = \phi_j \circ \phi_i^{-1}$ are smooth

**Neural Atlas Architecture:**
Replace a single encoder with a **Mixture of Experts** structure {cite}`jacobs1991adaptive`:
- **Router** (Atlas Topology): Learns which chart covers each input
- **Experts** (Local Charts): Each expert is a local encoder $\phi_i$
- **Blending** (Transition Functions): Soft mixing via router weights

#### 7.7.2 Orthonormal Constraints for Atlas Charts

To ensure each chart preserves local geometric structure, we enforce an **orthogonality/isometry constraint** via semi-orthogonal weight regularization.

**OrthogonalLinear Layer Implementation:**

```python
import torch
import torch.nn as nn

class OrthogonalLinear(nn.Module):
    """Linear layer with an orthogonality (approximate isometry) regularizer.

    Constraint: W^T W ≈ I (semi-orthogonality for rectangular W).
    Effect: Better conditioning and approximate distance preservation in the chart.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def orth_defect(self) -> torch.Tensor:
        """Compute the orthogonality defect.

        Returns ||W^T W - I||²_F where W is the weight matrix.
        This encourages W to be orthogonal (or semi-orthogonal).
        """
        W = self.linear.weight  # [out_features, in_features]

        # Handle rectangular matrices: use smaller dimension
        if W.shape[0] >= W.shape[1]:
            gram = torch.matmul(W.t(), W)  # [in, in]
            target = torch.eye(W.shape[1], device=W.device)
        else:
            gram = torch.matmul(W, W.t())  # [out, out]
            target = torch.eye(W.shape[0], device=W.device)

        return torch.norm(gram - target) ** 2
```

**Why Orthogonality?**

| Property | Orthogonal $W$ | Arbitrary $W$ |
|----------|----------------|---------------|
| **Singular values** | All = 1 | Can be 0 or ∞ |
| **Gradient flow** | Preserved | Explodes or vanishes |
| **Distance preservation** | $\|Wx\| = \|x\|$ | $\|Wx\| \neq \|x\|$ |
| **Inverse stability** | $W^{-1} = W^T$ | May not exist |
| **Information loss** | None | Possible |

#### 7.7.3 VICReg: Geometric Collapse Prevention

Each chart must produce non-degenerate embeddings. We enforce this via **VICReg** {cite}`bardes2022vicreg`.

```python
def compute_vicreg_loss(
    z: torch.Tensor,       # [B, Z] - embeddings from chart
    z_prime: torch.Tensor, # [B, Z] - embeddings from augmented view
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    gamma: float = 1.0,    # Target standard deviation
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """VICReg loss: Variance-Invariance-Covariance Regularization.

    Prevents representation collapse without negative samples.

    Components:
    - Invariance: Embeddings stable under perturbations
    - Variance: Each dimension has sufficient spread
    - Covariance: Dimensions are decorrelated

    Args:
        z: Embeddings from original input
        z_prime: Embeddings from augmented input
        lambda_inv, lambda_var, lambda_cov: Loss weights
        gamma: Target standard deviation per dimension
        eps: Numerical stability

    Returns:
        Total loss and dict of component losses
    """
    B, Z = z.shape

    # 1. Invariance Loss: z ≈ z' (metric stability)
    loss_inv = nn.functional.mse_loss(z, z_prime)

    # 2. Variance Loss: std(z_d) >= gamma (non-collapse)
    # Compute std per dimension, penalize if below gamma
    std_z = torch.sqrt(z.var(dim=0) + eps)  # [Z]
    std_z_prime = torch.sqrt(z_prime.var(dim=0) + eps)
    loss_var = torch.mean(nn.functional.relu(gamma - std_z)) + \
               torch.mean(nn.functional.relu(gamma - std_z_prime))

    # 3. Covariance Loss: Cov(z_i, z_j) → 0 for i ≠ j (decorrelation)
    z_centered = z - z.mean(dim=0)
    z_prime_centered = z_prime - z_prime.mean(dim=0)

    cov_z = (z_centered.T @ z_centered) / (B - 1)  # [Z, Z]
    cov_z_prime = (z_prime_centered.T @ z_prime_centered) / (B - 1)

    # Extract off-diagonal elements
    def off_diagonal(x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    loss_cov = off_diagonal(cov_z).pow(2).sum() / Z + \
               off_diagonal(cov_z_prime).pow(2).sum() / Z

    # Combined loss
    total = lambda_inv * loss_inv + lambda_var * loss_var + lambda_cov * loss_cov

    return total, {
        'invariance': loss_inv.item(),
        'variance': loss_var.item(),
        'covariance': loss_cov.item()
    }
```

#### 7.7.4 The Universal Loss Functional

The **Universal Loss** combines four components, each with a geometric interpretation:

$$\mathcal{L}_{\text{universal}} = \mathcal{L}_{\text{vicreg}} + \mathcal{L}_{\text{topology}} + \mathcal{L}_{\text{separation}} + \mathcal{L}_{\text{orth}}$$

**Component Breakdown:**

| Component | Formula | Interpretation | Coefficient |
|-----------|---------|------------------|-------------|
| **VICReg** | $\mathcal{L}_{\text{inv}} + \mathcal{L}_{\text{var}} + \mathcal{L}_{\text{cov}}$ | Data manifold structure | 25 / 25 / 1 |
| **Entropy** | $-\mathbb{E}[\sum w_i \log w_i]$ | Sharp chart boundaries | 2.0 |
| **Balance** | $\|\text{usage} - 1/K\|^2$ | Atlas completeness | 100.0 |
| **Separation** | $\sum_{i<j} \text{ReLU}(m - \|c_i - c_j\|)$ | Chart separation | 10.0 |
| **Orthogonality** | $\sum_l \|W_l^T W_l - I\|^2$ | Approx. isometry / conditioning | 0.01 |

**Topology Loss (Atlas Structure):**
```python
def compute_topology_loss(
    weights: torch.Tensor,  # [B, K] - router weights (softmax output)
    num_charts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Topology loss: Enforces atlas structure via information constraints.

    Two components:
    1. Entropy: Low entropy → sharp chart assignments
    2. Balance: Equal usage → all charts contribute

    Args:
        weights: Router output probabilities [B, K]
        num_charts: Number of charts K

    Returns:
        (entropy_loss, balance_loss)
    """
    # 1. Entropy loss: Encourage sharp assignments (low entropy)
    # H(w) = -Σ w_i log w_i → want this small
    entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=1)
    loss_entropy = entropy.mean()

    # 2. Balance loss: All charts should be used equally
    # usage_i = E[w_i] → want this close to 1/K
    mean_usage = weights.mean(dim=0)  # [K]
    target_usage = torch.ones(num_charts, device=weights.device) / num_charts
    loss_balance = torch.norm(mean_usage - target_usage) ** 2

    return loss_entropy, loss_balance
```

**Separation Loss (Chart separation):**
```python
def compute_separation_loss(
    chart_outputs: list[torch.Tensor],  # List of [B, Z] per chart
    weights: torch.Tensor,               # [B, K] router weights
    margin: float = 4.0,
) -> torch.Tensor:
    """Separation loss: Force chart centers apart.

    This enforces a margin between chart centers to encourage distinct regions
    covered by different charts.

    Args:
        chart_outputs: List of embeddings from each expert
        weights: Router attention weights
        margin: Minimum distance between chart centers

    Returns:
        Scalar loss penalizing overlapping charts
    """
    # Compute weighted center for each chart
    centers = []
    for i, z_i in enumerate(chart_outputs):
        w_i = weights[:, i:i+1]  # [B, 1]
        if w_i.sum() > 0:
            # Weighted mean of this chart's embeddings
            center = (z_i * w_i).sum(dim=0) / (w_i.sum() + 1e-6)  # [Z]
            centers.append(center)

    # Penalize charts that are too close
    loss_sep = torch.tensor(0.0, device=weights.device)
    if len(centers) > 1:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = torch.norm(centers[i] - centers[j])
                # Hinge loss: penalize if dist < margin
                loss_sep = loss_sep + torch.relu(margin - dist)

    return loss_sep
```

#### 7.7.5 HypoUniversal: Complete Atlas Architecture

```python
class HypoUniversal(nn.Module):
    """Universal Hypostructure Network with Atlas Architecture.

    This implements a multi-chart latent space where:
    - Router (Axiom TB): Learns chart assignments via soft attention
    - Experts (Axiom LS): Each chart is an orthogonality-constrained encoder
    - Output: Weighted blend of chart embeddings

    Theoretical Foundation:
    - Manifold Atlas: Complex manifolds need multiple charts
    - Orthonormal constraints: Each chart is well-conditioned / approximately isometric
    - VICReg: Prevents collapse within each chart
    - Separation: Forces charts to cover different regions

    Example:
        model = HypoUniversal(input_dim=3, latent_dim=2, num_charts=4)
        z, weights, chart_outputs = model(x)
        loss = universal_loss(z, x, weights, chart_outputs, model)
    """

    def __init__(self, input_dim: int, latent_dim: int, num_charts: int = 3):
        super().__init__()
        self.num_charts = num_charts

        # A. The Router (Topology / Axiom TB)
        # Learns which chart covers each input region
        # Standard layers (no orthogonality needed—cuts don't need isometry)
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_charts),
            nn.Softmax(dim=1)
        )

        # B. The Experts (Geometry / Axiom LS)
        # Each chart uses orthogonality-regularized layers for conditioning
        self.charts = nn.ModuleList()
        for _ in range(num_charts):
            expert = nn.Sequential(
                OrthogonalLinear(input_dim, 128),
                nn.ReLU(),
                OrthogonalLinear(128, 128),
                nn.ReLU(),
                OrthogonalLinear(128, latent_dim)
            )
            self.charts.append(expert)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Forward pass through atlas architecture.

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            z: Blended latent embedding [B, latent_dim]
            weights: Router attention [B, num_charts]
            chart_outputs: List of per-chart embeddings [B, latent_dim]
        """
        # Get chart selection weights
        weights = self.router(x)  # [B, num_charts]

        # Compute each chart's embedding
        chart_outputs = []
        z = torch.zeros(x.size(0), self.charts[0][-1].linear.out_features,
                       device=x.device)

        for i in range(self.num_charts):
            z_i = self.charts[i](x)  # [B, latent_dim]
            chart_outputs.append(z_i)
            # Weighted contribution
            z = z + weights[:, i:i+1] * z_i

        return z, weights, chart_outputs

    def compute_orth_loss(self) -> torch.Tensor:
        """Compute total orthogonality defect across all charts."""
        total_defect = torch.tensor(0.0)
        for chart in self.charts:
            for layer in chart:
                if isinstance(layer, OrthogonalLinear):
                    total_defect = total_defect + layer.orth_defect()
        return total_defect


def universal_loss(
    z: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
    chart_outputs: list[torch.Tensor],
    model: HypoUniversal,
    # VICReg weights
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    # Topology weights
    lambda_entropy: float = 2.0,
    lambda_balance: float = 100.0,
    # Separation
    lambda_sep: float = 10.0,
    margin: float = 4.0,
    # Orthogonality
    lambda_orth: float = 0.01,
) -> torch.Tensor:
    """Unified loss for atlas-based latent representations.

    Combines four loss families:
    1. VICReg: Data manifold structure (no collapse)
    2. Topology: Atlas structure (sharp, balanced charts)
    3. Separation: chart separation (distinct regions)
    4. Orthogonality: Conditioning / approximate isometry

    Args:
        z: Blended output [B, Z]
        x: Original input [B, D]
        weights: Router weights [B, K]
        chart_outputs: Per-chart embeddings
        model: The HypoUniversal model
        lambda_*: Loss component weights
        margin: Chart separation margin

    Returns:
        Total scalar loss
    """
    # 1. VICReg (Data Manifold)
    # Create augmented view via small noise
    x_aug = x + torch.randn_like(x) * 0.05
    z_prime, _, _ = model(x_aug)
    loss_vicreg, _ = compute_vicreg_loss(z, z_prime, lambda_inv, lambda_var, lambda_cov)

    # 2. Topology (Router Constraints)
    loss_entropy, loss_balance = compute_topology_loss(weights, model.num_charts)

    # 3. Separation (Chart Surgery)
    loss_sep = compute_separation_loss(chart_outputs, weights, margin)

    # 4. Orthogonality (Internal Conditioning)
    loss_orth = model.compute_orth_loss()

    # Combine all components
    return (loss_vicreg +
            lambda_entropy * loss_entropy +
            lambda_balance * loss_balance +
            lambda_sep * loss_sep +
            lambda_orth * loss_orth)
```

#### 7.7.6 Training the Atlas-Based Fragile Agent

```python
def train_atlas_model(
    model: HypoUniversal,
    data: torch.Tensor,
    epochs: int = 8000,
    lr: float = 1e-3,
) -> HypoUniversal:
    """Train the atlas-based model.

    Example usage:
        model = HypoUniversal(input_dim=3, latent_dim=2, num_charts=4)
        model = train_atlas_model(model, X, epochs=8000)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        z, weights, chart_outputs = model(data)
        loss = universal_loss(z, data, weights, chart_outputs, model)

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            usage = weights.mean(dim=0).detach().cpu().numpy()
            print(f"Epoch {epoch}: Loss={loss.item():.4f} | "
                  f"Chart Usage={usage}")

    return model
```

**Expected Behavior:**
- Charts should specialize to different topological regions
- Usage should be roughly balanced (each chart ~25% for K=4)
- Orthogonality defect should decrease over training
- Separation should increase to margin value

---

## 8. Infeasible Implementation Replacements

Several regularization terms from the theoretical framework are computationally infeasible for standard training. This section provides practical alternatives with full PyTorch implementations.

### 8.1 BarrierBode → Temporal Gain Margin

**Original (Infeasible):**
$$
\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.} \quad \text{(Bode sensitivity integral)}
$$

**Problem:** Requires frequency-domain analysis of the closed-loop transfer function $S(j\omega)$. Neural policies don't have closed-form transfer functions, and FFT requires long stationary trajectories.

**Replacement: Temporal Gain Margin**
$$
\mathcal{L}_{\text{gain}} = \sum_{k=1}^{K} \max\left(0, \frac{\Vert e_{t+k} \Vert}{\Vert e_t \Vert + \epsilon} - G_{\max}\right)^2
$$

This detects the same failure mode (error amplification / oscillatory instability) without LTI assumptions.

```python
def compute_gain_margin_loss(
    errors: torch.Tensor,  # Shape: [B, T] - tracking errors over time
    G_max: float = 2.0,     # Maximum allowed gain
    K: int = 5,             # Lookahead horizon
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    BarrierBode replacement: Temporal gain margin constraint.

    Penalizes trajectories where errors amplify over time,
    which indicates loop instability (same failure mode as Bode sensitivity).

    Args:
        errors: [B, T] tensor of error magnitudes at each timestep
        G_max: Maximum allowed amplification ratio
        K: Number of steps to check ahead
        eps: Numerical stability

    Returns:
        Scalar loss penalizing gain violations
    """
    B, T = errors.shape
    if T <= K:
        return torch.tensor(0.0, device=errors.device)

    total_violation = 0.0
    for k in range(1, min(K + 1, T)):
        # Gain at lag k: ||e_{t+k}|| / ||e_t||
        e_t = errors[:, :-k]  # [B, T-k]
        e_t_plus_k = errors[:, k:]  # [B, T-k]

        gain = e_t_plus_k / (e_t + eps)
        violation = torch.relu(gain - G_max).pow(2)
        total_violation = total_violation + violation.mean()

    return total_violation / K


# Alternative: Peak gain detection (simpler)
def compute_peak_gain_loss(
    errors: torch.Tensor,  # [B, T]
    G_max: float = 2.0,
) -> torch.Tensor:
    """Simpler version: just penalize max gain."""
    B, T = errors.shape
    e_ratios = errors[:, 1:] / (errors[:, :-1] + 1e-6)
    max_gain = e_ratios.max(dim=-1).values  # [B]
    return torch.relu(max_gain - G_max).pow(2).mean()
```

### 8.2 BifurcateCheck → Stochastic Jacobian Probing

**Original (Infeasible):**
$$
\det(J_{S_t}) \quad \text{where } J_{S_t} = \frac{\partial S_t(z)}{\partial z}
$$

**Problem:** Computing the full Jacobian is $O(Z^3)$. For $Z = 256$, this is ~16M operations per sample.

**Replacement: Hutchinson-style Jacobian Probing**
$$
\mathcal{L}_{\text{bifurcate}} = \text{Var}_v\left[\Vert J_{S_t} v \Vert^2\right] \quad \text{where } v \sim \mathcal{N}(0, I)
$$

High variance in the Jacobian-vector product norm indicates instability (eigenvalue spread).

```python
def compute_bifurcation_loss(
    world_model: nn.Module,
    z: torch.Tensor,           # [B, Z] - current latent states
    a: torch.Tensor,           # [B, A] - actions (if needed)
    n_probes: int = 5,
    instability_threshold: float = 1.0,
) -> torch.Tensor:
    """
    BifurcateCheck replacement: Stochastic Jacobian probing.

    Uses Hutchinson trace estimator principle: instead of computing
    full Jacobian, probe with random vectors. High variance in
    ||J @ v|| indicates eigenvalue spread → bifurcation sensitivity.

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        n_probes: Number of random direction probes
        instability_threshold: Variance threshold for penalty

    Returns:
        Scalar loss penalizing high Jacobian variance
    """
    B, Z = z.shape
    z = z.requires_grad_(True)

    # Forward through world model
    z_next = world_model(z, a)  # [B, Z]

    jvp_norms = []
    for _ in range(n_probes):
        # Random probe direction
        v = torch.randn_like(z)  # [B, Z]

        # Jacobian-vector product via autodiff (efficient: O(Z))
        jvp = torch.autograd.grad(
            outputs=z_next,
            inputs=z,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, Z]

        jvp_norm = jvp.norm(dim=-1)  # [B]
        jvp_norms.append(jvp_norm)

    # Stack and compute variance across probes
    jvp_norms = torch.stack(jvp_norms, dim=0)  # [n_probes, B]
    variance = jvp_norms.var(dim=0).mean()  # Average variance across batch

    # Penalize high variance (indicates instability)
    loss = torch.relu(variance - instability_threshold).pow(2)

    return loss
```

### 8.3 TameCheck → Lipschitz Gradient Proxy

**Original (Infeasible):**
$$
\Vert \nabla^2 S_t \Vert \quad \text{(Hessian norm)}
$$

**Problem:** Full Hessian is $O(Z^2 \times P_{WM})$ — prohibitive for large world models.

**Replacement: Lipschitz of Gradient**
$$
\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}
$$

Bounded gradient Lipschitz constant implies bounded Hessian (by definition).

```python
def compute_tame_loss(
    world_model: nn.Module,
    z: torch.Tensor,         # [B, Z]
    a: torch.Tensor,         # [B, A]
    perturbation_scale: float = 0.01,
    lipschitz_target: float = 1.0,
) -> torch.Tensor:
    """
    TameCheck replacement: Lipschitz gradient constraint.

    Instead of computing full Hessian, we estimate the Lipschitz
    constant of the gradient via finite differences. This bounds
    the Hessian spectral norm (tameness).

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        perturbation_scale: Size of random perturbation
        lipschitz_target: Target Lipschitz constant

    Returns:
        Scalar loss penalizing non-tame dynamics
    """
    B, Z = z.shape

    # Two nearby points
    z1 = z.requires_grad_(True)
    delta = torch.randn_like(z) * perturbation_scale
    z2 = (z + delta).requires_grad_(True)

    # Forward passes
    z1_next = world_model(z1, a)
    z2_next = world_model(z2, a)

    # Compute gradients at both points
    # Sum over output dims to get [B, Z] gradient
    grad1 = torch.autograd.grad(
        z1_next.sum(), z1, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    grad2 = torch.autograd.grad(
        z2_next.sum(), z2, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    # Lipschitz estimate: ||grad1 - grad2|| / ||z1 - z2||
    grad_diff = (grad1 - grad2).norm(dim=-1)  # [B]
    z_diff = delta.norm(dim=-1) + 1e-6  # [B]

    lipschitz_estimate = grad_diff / z_diff  # [B]

    # Penalize exceeding target Lipschitz constant
    loss = torch.relu(lipschitz_estimate - lipschitz_target).pow(2).mean()

    return loss
```

### 8.4 TopoCheck → Value Gradient Alignment

**Original (Infeasible):**
$$
T_{\text{reach}}(s_{\text{goal}}) \quad \text{(Reachability time)}
$$

**Problem:** Requires multi-step planning through world model: $O(H \times B \times Z)$ with potentially large horizon $H$.

**Replacement: Value Gradient Alignment**
$$
\mathcal{L}_{\text{topo}} = -\left\langle \nabla_s V(s), \frac{s_{\text{goal}} - s}{\Vert s_{\text{goal}} - s \Vert} \right\rangle
$$

If the Critic's gradient points toward the goal, gradient descent reaches it.

```python
def compute_topo_loss(
    critic: nn.Module,
    states: torch.Tensor,      # [B, Z] - current states
    goal_states: torch.Tensor,  # [B, Z] or [Z] - goal states
) -> torch.Tensor:
    """
    TopoCheck replacement: Value gradient alignment.

    Instead of computing multi-step reachability, we check if
    the critic's value gradient points toward the goal. This is
    a necessary condition for gradient-based reachability.

    Args:
        critic: V(s) -> scalar value
        states: Current states [B, Z]
        goal_states: Target states [B, Z] or [Z]

    Returns:
        Scalar loss (negative = gradient points toward goal)
    """
    B, Z = states.shape
    states = states.requires_grad_(True)

    # Compute value and its gradient
    values = critic(states)  # [B]
    grad_v = torch.autograd.grad(
        values.sum(), states, create_graph=True
    )[0]  # [B, Z]

    # Direction to goal
    if goal_states.dim() == 1:
        goal_states = goal_states.unsqueeze(0).expand(B, -1)

    to_goal = goal_states - states  # [B, Z]
    to_goal_normalized = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    # Alignment: should be negative (V decreases toward goal)
    alignment = (grad_v * to_goal_normalized).sum(dim=-1)  # [B]

    # Loss: penalize positive alignment (wrong direction)
    loss = torch.relu(alignment).mean()

    return loss
```

### 8.5 GeomCheck → Efficient InfoNCE

**Original (Expensive):**
$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum_{j=1}^{B} \exp(\text{sim}(z_t, z_j))}
$$

**Problem:** Full pairwise computation is $O(B^2 \times Z)$.

**Replacement: Sampled InfoNCE**
$$
\mathcal{L}_{\text{InfoNCE}}^{\text{eff}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\exp(\text{sim}(z_t, z_{t+k})) + \sum_{j=1}^{K} \exp(\text{sim}(z_t, z_{\text{neg},j}))}
$$

Use $K \ll B$ sampled negatives instead of full batch.

```python
class EfficientInfoNCE(nn.Module):
    """
    GeomCheck replacement: Efficient contrastive loss.

    Uses K sampled negatives instead of full batch pairwise.
    Reduces O(B²Z) to O(KBZ) where K << B.
    """

    def __init__(
        self,
        latent_dim: int,
        n_negatives: int = 128,
        tau: float = 0.1,  # softmax scale
    ):
        super().__init__()
        self.n_negatives = n_negatives
        self.tau = tau

        # Projection head (optional, improves quality)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        z_anchor: torch.Tensor,   # [B, Z] - z_t
        z_positive: torch.Tensor,  # [B, Z] - z_{t+k} (temporally close)
        z_bank: torch.Tensor = None,  # [M, Z] - memory bank for negatives
    ) -> torch.Tensor:
        """
        Compute efficient InfoNCE loss.

        Args:
            z_anchor: Anchor embeddings [B, Z]
            z_positive: Positive pairs [B, Z]
            z_bank: Optional memory bank for negatives [M, Z]

        Returns:
            Scalar contrastive loss
        """
        B, Z = z_anchor.shape

        # Project
        anchor = F.normalize(self.projector(z_anchor), dim=-1)  # [B, Z]
        positive = F.normalize(self.projector(z_positive), dim=-1)  # [B, Z]

        # Sample negatives
        if z_bank is not None and z_bank.shape[0] >= self.n_negatives:
            # Sample from memory bank
            indices = torch.randperm(z_bank.shape[0])[:self.n_negatives]
            negatives = z_bank[indices]  # [K, Z]
            negatives = F.normalize(self.projector(negatives), dim=-1)
        else:
            # Use other batch elements as negatives (in-batch)
            K = min(self.n_negatives, B - 1)
            # Shuffle and take first K (excluding self)
            perm = torch.randperm(B, device=z_anchor.device)
            negatives = anchor[perm[:K]]  # [K, Z]

        # Positive similarity: [B]
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Negative similarities: [B, K]
        neg_sim = torch.mm(anchor, negatives.T) / self.tau

        # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        # = pos - logsumexp([pos, neg1, neg2, ...])

        # Combine for logsumexp: [B, K+1]
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # Loss: -pos + logsumexp(all)
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)

        return loss.mean()


# Usage example:
def compute_geom_loss(
    vae_encoder: nn.Module,
    x_t: torch.Tensor,      # [B, D] - observation at time t
    x_t_plus_k: torch.Tensor,  # [B, D] - observation at time t+k
    info_nce: EfficientInfoNCE,
) -> torch.Tensor:
    """GeomCheck: Contrastive anchoring for latent space."""
    z_t = vae_encoder(x_t)
    z_t_k = vae_encoder(x_t_plus_k)
    return info_nce(z_t, z_t_k)
```

### 8.6 Summary: Replacement Mapping

| Original | Replacement | Speedup | Preserved Property |
|----------|-------------|---------|-------------------|
| BarrierBode (FFT) | Temporal Gain | ~100× | Detects oscillatory instability |
| BifurcateCheck ($O(Z^3)$) | Jacobian Probing | ~$Z^2/K$ | Detects eigenvalue spread |
| TameCheck ($O(Z^2 P)$) | Lipschitz Gradient | ~$ZP$ | Bounds Hessian norm |
| TopoCheck ($O(HBZ)$) | Value Alignment | ~$H$ | Ensures goal reachability |
| GeomCheck ($O(B^2 Z)$) | Sampled NCE | ~$B/K$ | Preserves slow features |

---

## 9. The Disentangled Variational Architecture: Hierarchical Latent Separation

This section provides a practical guide to implementing a **split-latent** architecture that separates a *predictive* macro register from a *nuisance* micro residual. This targets **BarrierEpi** (information overload) and **BarrierOmin** (model mismatch) by preventing the World Model from being forced to “explain” fundamentally unpredictable variation.

### 9.1 The Core Concept: Split-Brain Architecture

**Standard Agent:** Encodes the state into a single vector $z$.

**Disentangled Agent:** Encodes the state into a **discrete macro-symbol** plus a **continuous micro-residual** (Section 2.2b):

1. **$K_t \in \mathcal{K}$ (The Macro Symbol / Law Register):** Low-frequency, causal, predictable. This is the *quantized* macrostate (a code index). For downstream continuous networks we also use its embedding $z_{\text{macro}}:=e_{K_t}\in\mathbb{R}^{d_m}$, but the *information-carrying* object is the discrete symbol $K_t$.

2. **$z_{\mu,t}$ (The Micro Residual / Nuisance Stream):** High-frequency, hard-to-predict variation. A continuous latent used for reconstruction/texture but excluded from macro dynamics (treated as nuisance/noise, not state).

**The Golden Rule of Causal Enclosure:**

$$
P(K_{t+1}\mid K_t,a_t)\ \text{is sharply concentrated (ideally deterministic), and}\ I(K_{t+1};Z_{\mu,t}\mid K_t,a_t)=0.
$$

The macro symbol must be predictable **solely from its own history** (plus action). If the World Model needs micro-residuals to predict the next macro symbol, then $K_t$ is not a sufficient macro statistic (Section 2.8).

### 9.2 Architecture: The Disentangled VQ-VAE-RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DisentangledConfig:
    """Configuration for split-latent (macro/micro) agent."""
    obs_dim: int = 64 * 64 * 3      # Observation dimension
    hidden_dim: int = 256            # Encoder hidden dimension
    macro_embed_dim: int = 32        # Macro embedding dim (code vectors e_k)
    codebook_size: int = 512         # Number of discrete macrostates |𝒦|
    micro_dim: int = 128             # Micro (nuisance) latent dimension
    action_dim: int = 4              # Action dimension
    rnn_hidden_dim: int = 256        # Dynamics model RNN hidden

    # Loss weights
    lambda_closure: float = 1.0      # Causal enclosure weight
    lambda_slowness: float = 0.1     # Temporal smoothness weight
    lambda_dispersion: float = 0.01  # Micro KL weight
    lambda_vq: float = 1.0           # VQ codebook + commitment weight
    lambda_recon: float = 1.0        # Reconstruction weight

    # Training
    info_dropout_prob: float = 0.5   # Probability of dropping micro
    warmup_steps: int = 1000         # Warmup for closure loss


class Encoder(nn.Module):
    """Shared encoder backbone."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        # For image observations, use CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size (for 64x64 input: 256 * 4 * 4 = 4096)
        self.fc = nn.Linear(4096, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, T, C, H, W]
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.view(B * T, *x.shape[2:])
            h = F.relu(self.fc(self.conv(x)))
            return h.view(B, T, -1)
        return F.relu(self.fc(self.conv(x)))


class VectorQuantizer(nn.Module):
    """
    VQ layer: maps a continuous encoder output z_e to a discrete code K and
    its corresponding embedding e_K using a straight-through estimator.
    """

    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.beta = beta
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, D]
        # Compute squared L2 distances to codebook vectors
        codebook = self.codebook.weight  # [|𝒦|, D]
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )
        K = torch.argmin(distances, dim=-1)          # [B]
        z_q = self.codebook(K)                       # [B, D]

        # VQ-VAE losses
        codebook_loss = (z_q - z_e.detach()).pow(2).mean()
        commit_loss = self.beta * (z_e - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator: pass gradients to encoder as if identity
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, K, vq_loss

    def embed(self, K: torch.Tensor) -> torch.Tensor:
        return self.codebook(K)


class Decoder(nn.Module):
    """Decoder using both macro and micro latents."""

    def __init__(self, macro_dim: int, micro_dim: int, obs_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(macro_dim + micro_dim, 4096)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, z_macro: torch.Tensor, z_micro: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_macro, z_micro], dim=-1)
        h = F.relu(self.fc(z))
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class MacroDynamicsModel(nn.Module):
    """
    Macro dynamics model (micro-blind).

    Important: this module only sees z_macro (it is blind to z_micro). This
    forces causally/predictively relevant information into the macro channel.
    """

    def __init__(self, macro_embed_dim: int, action_dim: int, hidden_dim: int, codebook_size: int):
        super().__init__()
        self.macro_embed_dim = macro_embed_dim
        self.codebook_size = codebook_size

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(macro_embed_dim + action_dim, hidden_dim)

        # Project hidden state to a distribution over next macro symbol K_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_size),  # logits over 𝒦
        )

        # Uncertainty estimation (optional but recommended)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_macro: torch.Tensor,       # [B, macro_embed_dim] (= code embedding e_K)
        action: torch.Tensor,        # [B, action_dim]
        h_prev: torch.Tensor,        # [B, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the next macro symbol distribution from the current macro embedding only.

        Returns:
            logits_next: Logits for K_{t+1} over 𝒦
            h_next: Updated hidden state
            uncertainty: Scalar uncertainty proxy
        """
        # Concatenate macro state and action (NO micro!)
        x = torch.cat([z_macro, action], dim=-1)

        # Update hidden state
        h_next = self.gru(x, h_prev)

        # Predict next macro symbol (logits over 𝒦)
        logits_next = self.predictor(h_next)
        uncertainty = self.uncertainty(h_next)

        return logits_next, h_next, uncertainty


class DisentangledAgent(nn.Module):
    """
    Split-latent VQ-VAE + macro dynamics model.

    Separates:
    - K_t (macro): a discrete symbol in 𝒦 (predictive/controllable state)
    - z_mu (micro): a continuous residual (nuisance variation for reconstruction)
    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Macro: continuous pre-quantization → discrete code K via VQ
        self.head_macro = nn.Linear(config.hidden_dim, config.macro_embed_dim)
        self.vq = VectorQuantizer(config.codebook_size, config.macro_embed_dim)

        # Micro: Gaussian nuisance residual
        self.head_micro_mean = nn.Linear(config.hidden_dim, config.micro_dim)
        self.head_micro_logvar = nn.Linear(config.hidden_dim, config.micro_dim)

        # Macro dynamics model (blind to micro)
        self.macro_dynamics = MacroDynamicsModel(
            config.macro_embed_dim,
            config.action_dim,
            config.rnn_hidden_dim,
            config.codebook_size,
        )

        # Decoder (uses both)
        self.decoder = Decoder(config.macro_embed_dim, config.micro_dim)

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Encode observation into a discrete macro symbol and a continuous micro residual.

        Returns:
            K: Macro code index in 𝒦
            z_macro: Quantized macro embedding e_K (straight-through)
            z_micro: Micro latent (noise)
            micro_dist: (mean, logvar) for micro
            vq_loss: codebook + commitment loss
        """
        features = self.encoder(x)

        # Macro: quantize into a discrete symbol K and embedding z_macro := e_K
        z_e = self.head_macro(features)
        z_macro, K, vq_loss = self.vq(z_e)

        # Micro: High variance (entropic)
        micro_mean = self.head_micro_mean(features)
        micro_logvar = self.head_micro_logvar(features)
        # Allow higher variance for micro
        micro_logvar = torch.clamp(micro_logvar, min=-5, max=2)

        # Reparameterization trick
        z_micro = self._reparameterize(micro_mean, micro_logvar)

        return K, z_macro, z_micro, (micro_mean, micro_logvar), vq_loss

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_t: torch.Tensor,           # [B, C, H, W] current observation
        action: torch.Tensor,         # [B, action_dim]
        h_prev: torch.Tensor,         # [B, rnn_hidden_dim]
        training: bool = True,
    ) -> dict:
        """
        Full forward pass with information dropout.

        Returns dict with all intermediate values for loss computation.
        """
        # 1. Encode current observation
        K_t, z_macro, z_micro, micro_dist, vq_loss = self.encode(x_t)

        # 2. Macro dynamics step (blind to micro)
        logits_next, h_next, uncertainty = self.macro_dynamics(z_macro, action, h_prev)
        K_pred = torch.argmax(logits_next, dim=-1)
        z_macro_pred = self.vq.embed(K_pred)

        # 3. Information Dropout for reconstruction
        if training and torch.rand(1).item() < self.config.info_dropout_prob:
            # Drop micro — force decoder to use only macro
            z_micro_for_decode = torch.zeros_like(z_micro)
        else:
            z_micro_for_decode = z_micro

        # 4. Reconstruct
        x_recon = self.decoder(z_macro, z_micro_for_decode)

        return {
            'K_t': K_t,
            'z_macro': z_macro,
            'z_micro': z_micro,
            'z_macro_logits': logits_next,
            'K_pred': K_pred,
            'z_macro_pred': z_macro_pred,
            'micro_dist': micro_dist,
            'vq_loss': vq_loss,
            'h_next': h_next,
            'uncertainty': uncertainty,
            'x_recon': x_recon,
            'info_dropped': z_micro_for_decode.sum() == 0,
        }

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.rnn_hidden_dim, device=device)


```

### 9.3 Loss Function: Enforcing Macro/Micro Separation

The Disentangled Agent cannot be trained with a reconstruction loss alone. It requires a compound loss that enforces the **learnability threshold**: the macro channel must be predictive/closed, and the micro channel must be treated as nuisance rather than state.

$$
\mathcal{L}_{\text{total}}
=
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
\;+\;\lambda_{\text{vq}}\mathcal{L}_{\text{vq}}
\;+\;\lambda_{\text{closure}}\mathcal{L}_{\text{closure}}
\;+\;\lambda_{\text{slowness}}\mathcal{L}_{\text{slowness}}
\;+\;\lambda_{\text{dispersion}}\mathcal{L}_{\text{dispersion}}
$$
where $\mathcal{L}_{\text{closure}}$ is a cross-entropy on the discrete macro symbols (an estimator of $H(K_{t+1}\mid K_t,a_t)$) and $\mathcal{L}_{\text{vq}}$ is the codebook+commitment term from the VQ layer.

```python
class DisentangledLoss(nn.Module):
    """
    Compound loss for training the split-latent agent.

    Implements the five key constraints:
    1. Closure: Macro must predict itself (causal enclosure)
    2. Slowness: Macro should change slowly (prevents symbol churn)
    3. Micro prior: Micro should stay close to a simple prior (nuisance regularization)
    4. VQ: Macro must remain quantized (symbolic)
    5. Reconstruction: Both channels needed for full reconstruction
    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict,              # From DisentangledAgent.forward()
        x_target: torch.Tensor,     # Target observation (x_{t+1} for recon)
        K_next: torch.Tensor,       # Target macro code at t+1 (LongTensor)
        z_macro_prev: Optional[torch.Tensor] = None,  # e_{K_{t-1}} (for slowness)
        step: int = 0,              # Training step for warmup
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all loss components.

        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Individual losses for logging
        """
        losses = {}

        # === A. RECONSTRUCTION LOSS ===
        # Standard pixel-wise reconstruction
        losses['recon'] = F.mse_loss(outputs['x_recon'], x_target)

        # === B. CAUSAL ENCLOSURE LOSS (SYMBOLIC) ===
        # Predict the next macro symbol K_{t+1} from the current macro only.
        losses['closure'] = F.cross_entropy(outputs['z_macro_logits'], K_next)

        # Warmup: gradually increase closure weight
        closure_weight = min(1.0, step / self.config.warmup_steps)

        # === C. SLOWNESS LOSS ===
        # Penalize rapid changes in the macro embedding e_K (proxy for symbol churn)
        if z_macro_prev is not None:
            losses['slowness'] = (outputs['z_macro'] - z_macro_prev).pow(2).mean()
        else:
            losses['slowness'] = torch.tensor(0.0, device=outputs['z_macro'].device)

        # === D. MICRO PRIOR LOSS ===
        # Regularize micro toward a simple prior (we do NOT use it for macro prediction)
        # Micro KL: KL(q(z_mu|x) || N(0,I))
        micro_mean, micro_logvar = outputs['micro_dist']
        losses['dispersion'] = self._kl_divergence(micro_mean, micro_logvar)

        # === E. VQ LOSS (CODEBOOK + COMMITMENT) ===
        losses['vq'] = outputs['vq_loss']

        # === TOTAL LOSS ===
        total = (
            self.config.lambda_recon * losses['recon']
            + self.config.lambda_vq * losses['vq']
            + self.config.lambda_closure * closure_weight * losses['closure']
            + self.config.lambda_slowness * losses['slowness']
            + self.config.lambda_dispersion * losses['dispersion']
        )

        losses['total'] = total
        losses['closure_weight'] = closure_weight

        return total, losses

    def _kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mean, var) || N(0, I))"""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

### 9.4 The Complete Training Loop

```python
class DisentangledTrainer:
    """
    Complete training loop for the split-latent agent.

    Handles:
    - Temporal sequence processing
    - Gradient isolation between macro/micro paths
    - Warmup schedules
    - Diagnostic monitoring
    """

    def __init__(
        self,
        agent: DisentangledAgent,
        learning_rate: float = 3e-4,
        device: str = 'cuda',
    ):
        self.agent = agent.to(device)
        self.device = device
        self.loss_fn = DisentangledLoss(agent.config)

        # Separate optimizers for different components (optional)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

        # Diagnostics
        self.closure_history = []
        self.dispersion_history = []
        self.step = 0

    def train_step(
        self,
        observations: torch.Tensor,  # [B, T, C, H, W] sequence
        actions: torch.Tensor,        # [B, T-1, action_dim]
    ) -> dict:
        """
        Train on a sequence of observations.

        Args:
            observations: Temporal sequence of observations
            actions: Actions taken between observations

        Returns:
            Dictionary of losses and diagnostics
        """
        B, T = observations.shape[:2]
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        self.optimizer.zero_grad()

        # Initialize hidden state
        h = self.agent.init_hidden(B, self.device)

        total_loss = 0.0
        all_losses = {k: 0.0 for k in ['recon', 'vq', 'closure', 'slowness', 'dispersion']}

        z_macro_prev = None
        K_nexts = []          # Targets K_{t+1}
        logits_nexts = []     # Predicted logits for K_{t+1}

        for t in range(T - 1):
            x_t = observations[:, t]
            x_t1 = observations[:, t + 1]
            a_t = actions[:, t]

            # Forward pass at time t
            outputs_t = self.agent(x_t, a_t, h, training=True)
            h = outputs_t['h_next']

            # Encode t+1 for closure target
            K_t1, _, _, _, _ = self.agent.encode(x_t1)

            # Compute loss
            loss, losses = self.loss_fn(
                outputs_t,
                x_t1,
                K_t1,
                z_macro_prev,
                self.step,
            )

            total_loss = total_loss + loss
            for k in all_losses:
                if k in losses:
                    all_losses[k] = all_losses[k] + losses[k].item()

            z_macro_prev = outputs_t['z_macro'].detach()
            K_nexts.append(K_t1.detach())
            logits_nexts.append(outputs_t['z_macro_logits'].detach())

        # Backprop
        total_loss = total_loss / (T - 1)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.step += 1

        # Compute diagnostics
        avg_losses = {k: v / (T - 1) for k, v in all_losses.items()}
        avg_losses['total'] = total_loss.item()

        # Closure ratio (see 9.5): symbolic cross-entropy gain vs baseline
        closure_ratio = self._compute_closure_ratio(K_nexts, logits_nexts)
        avg_losses['closure_ratio'] = closure_ratio

        self.closure_history.append(closure_ratio)

        return avg_losses

    def _compute_closure_ratio(
        self,
        K_nexts: List[torch.Tensor],
        logits_nexts: List[torch.Tensor],
    ) -> float:
        """
        Closure ratio diagnostic (discrete macro).

        Ratio < 1  : predictive macro model beats a baseline (law-like)
        Ratio ≈ 1  : macro is no more predictable than baseline (no learned law)
        Ratio > 1  : predictor is worse than baseline (bug/degenerate training)
        """
        if not K_nexts:
            return float('nan')

        logits = torch.cat(logits_nexts, dim=0)  # [(T-1)B, |𝒦|]
        K_next = torch.cat(K_nexts, dim=0)       # [(T-1)B]

        # Model conditional cross entropy ≈ H(K_{t+1}|K_t,a_t)
        ce_model = F.cross_entropy(logits, K_next, reduction='mean')

        # Baseline: marginal code model (no conditioning)
        K_size = self.agent.config.codebook_size
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_baseline = (-torch.log(p[K_next])).mean()

        return (ce_model / (ce_baseline + 1e-8)).item()
```

### 9.5 Runtime Diagnostics: The Closure Ratio

The key diagnostic for macro/micro separation is the **Closure Ratio**:

$$
\text{Closure Ratio}
=
\frac{\mathbb{E}\big[-\log p_\psi(K_{t+1}\mid K_t,a_t)\big]}{\mathbb{E}\big[-\log p_{\text{base}}(K_{t+1})\big]}.
$$
With $p_{\text{base}}$ chosen as the marginal symbol model, the numerator estimates $H(K_{t+1}\mid K_t,a_t)$ and the denominator estimates $H(K_{t+1})$, so the *gap* is a direct estimate of predictive information $I(K_{t+1};K_t,a_t)$.

| Closure Ratio | Interpretation | Action |
|---------------|----------------|--------|
| $\approx 1$ | **No predictive law** — macro dynamics no better than marginal | Increase model capacity or improve macro encoder; check that actions are provided |
| $\ll 1$ | **Success** — conditional entropy collapses vs marginal | Sufficient macro statistic learned |
| $> 1$ | **Worse than baseline** | Bug/degeneracy (labels, codebook collapse, optimizer instability) |

```python
class ClosureMonitor:
    """
    Monitor for split-latent training.

    Integrates with Sieve nodes to detect failure modes.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.closure_ratios = []
        self.macro_predictability = []
        self.micro_entropy = []

    def update(
        self,
        logits_next: torch.Tensor,   # [B, |𝒦|]
        K_next: torch.Tensor,        # [B]
        micro_logvar: torch.Tensor,  # [B, d_μ]
    ):
        """Update diagnostics with new batch."""
        # Macro predictability: cross entropy (nats)
        ce_model = F.cross_entropy(logits_next, K_next, reduction='mean').item()

        # Baseline: marginal code model (per-batch)
        K_size = logits_next.shape[-1]
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_base = (-torch.log(p[K_next])).mean().item()

        # Closure ratio
        ratio = ce_model / (ce_base + 1e-8)
        self.closure_ratios.append(ratio)

        # Micro entropy proxy (Gaussian differential entropy)
        import math
        micro_ent = (0.5 * (micro_logvar + math.log(2 * math.pi * math.e))).sum(dim=-1).mean().item()

        # Individual metrics
        self.macro_predictability.append(ce_model)
        self.micro_entropy.append(micro_ent)

        # Keep window
        if len(self.closure_ratios) > self.window_size:
            self.closure_ratios.pop(0)
            self.macro_predictability.pop(0)
            self.micro_entropy.pop(0)

    def get_diagnostics(self) -> dict:
        """Get current diagnostic summary."""
        import numpy as np

        if not self.closure_ratios:
            return {}

        ratios = np.array(self.closure_ratios)
        macro_pred = np.array(self.macro_predictability)
        micro_ent = np.array(self.micro_entropy)

        return {
            'closure_ratio_mean': ratios.mean(),
            'closure_ratio_std': ratios.std(),
            'macro_predictability': macro_pred.mean(),
            'micro_entropy': micro_ent.mean(),
            'macro_model_learned': ratios.mean() < 0.5,  # Success threshold
            'recommendation': self._get_recommendation(ratios.mean()),
        }

    def _get_recommendation(self, ratio: float) -> str:
        if ratio < 0.3:
            return "Excellent: Clear separation of predictive macro and nuisance"
        elif ratio < 0.7:
            return "Good: Macro closure partially learned, consider increasing closure weight"
        elif ratio < 1.0:
            return "Warning: Macro/micro entanglement detected, increase lambda_closure"
        else:
            return "Error: Micro residual more predictive than macro, check architecture"

    def check_sieve_nodes(self) -> dict:
        """
        Map diagnostics to Sieve node checks.

        Returns status for relevant nodes.
        """
        diag = self.get_diagnostics()
        if not diag:
            return {}

        return {
            # TameCheck: Is the world model interpretable?
            'TameCheck': 'PASS' if diag['macro_predictability'] < 1.0 else 'WARN',

            # ComplexCheck: Is model capacity appropriate?
            'ComplexCheck': 'PASS' if diag['micro_entropy'] > -2.0 else 'WARN',

            # GeomCheck: Are latent spaces well-separated?
            'GeomCheck': 'PASS' if diag['closure_ratio_mean'] < 0.5 else 'WARN',

            # ParamCheck: Is the macro dynamics stable?
            'ParamCheck': 'PASS' if diag['closure_ratio_std'] < 0.5 else 'WARN',
        }
```

### 9.6 Advanced: Hierarchical Multi-Scale Latents

For complex environments, a single macro/micro split may be insufficient. A macro hierarchy extends the split-latent idea to multiple discrete scales:

$$
Z_t = (K_t^{(1)}, K_t^{(2)}, \ldots, K_t^{(L)}, Z_{\mu,t}),
\qquad
z_{\text{macro}}^{(i)} := e^{(i)}_{K_t^{(i)}}\in\mathbb{R}^{d_i}.
$$

Where $K^{(1)}$ is the slowest (most abstract) and $K^{(L)}$ is the fastest (most detailed) macro symbol; $z_{\text{macro}}^{(i)}$ denotes its code embedding.

```python
class HierarchicalDisentangled(nn.Module):
    """
    Multi-scale split-latent architecture.

    Each level operates at a different timescale:
    - Level 1: Slowest (global game state, long-term goals)
    - Level 2: Medium (object positions, velocities)
    - Level 3: Fast (fine motor control, reactions)
    - Micro: Noise (textures, particles)

    Inspired by Clockwork VAE (Saxena et al.) and
    Hierarchical World Models (Hafner et al.)
    """

    def __init__(
        self,
        config: DisentangledConfig,
        n_levels: int = 3,
        level_dims: List[int] = [8, 16, 32],
        level_codebook_sizes: List[int] = [64, 128, 256],
        level_update_freqs: List[int] = [8, 4, 1],  # Update every N steps
    ):
        super().__init__()
        self.n_levels = n_levels
        self.level_dims = level_dims
        self.level_codebook_sizes = level_codebook_sizes
        self.update_freqs = level_update_freqs

        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Separate heads for each macro level (pre-quantization)
        self.macro_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, dim) for dim in level_dims
        ])

        # VQ quantizers and macro dynamics models at each level
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(level_codebook_sizes[i], level_dims[i])
            for i in range(n_levels)
        ])
        self.macro_dynamics_models = nn.ModuleList([
            MacroDynamicsModel(
                level_dims[i],
                config.action_dim,
                config.rnn_hidden_dim // n_levels,
                level_codebook_sizes[i],
            )
            for i in range(n_levels)
        ])

        # Cross-level connections (top-down modulation)
        self.cross_level = nn.ModuleList([
            nn.Linear(level_dims[i], level_dims[i + 1])
            for i in range(n_levels - 1)
        ])

        # Micro head
        self.micro_head = nn.Linear(config.hidden_dim, config.micro_dim)

        # Decoder uses all levels
        total_dim = sum(level_dims) + config.micro_dim
        self.decoder = Decoder(total_dim, 0)  # No separate micro in decoder

        self.step_counter = 0

    def forward(
        self,
        x_t: torch.Tensor,
        action: torch.Tensor,
        h_prevs: List[torch.Tensor],  # Hidden state for each level
        training: bool = True,
    ) -> dict:
        """
        Hierarchical forward pass with clockwork updates.
        """
        features = self.encoder(x_t)

        z_macros = []
        z_macro_preds = []
        h_nexts = []

        top_down_context = None

        for i in range(self.n_levels):
            # Encode this level (pre-quantization)
            z_e_i = self.macro_heads[i](features)

            # Add top-down modulation from slower levels
            if top_down_context is not None:
                z_e_i = z_e_i + self.cross_level[i - 1](top_down_context)

            # Quantize into a discrete macro symbol K^{(i)} and embedding z_macro^{(i)} := e_{K^{(i)}}
            z_macro_i, K_i, _ = self.vq_levels[i](z_e_i)

            # Update macro dynamics only at appropriate frequency
            if self.step_counter % self.update_freqs[i] == 0:
                logits_i, h_next_i, _ = self.macro_dynamics_models[i](
                    z_macro_i, action, h_prevs[i]
                )
                K_pred_i = torch.argmax(logits_i, dim=-1)
                z_pred_i = self.vq_levels[i].embed(K_pred_i)
            else:
                # Hold state
                z_pred_i = z_macro_i
                h_next_i = h_prevs[i]

            z_macros.append(z_macro_i)
            z_macro_preds.append(z_pred_i)
            h_nexts.append(h_next_i)

            top_down_context = z_macro_i

        # Micro (always updates)
        z_micro = self.micro_head(features)

        # Information dropout on micro
        if training and torch.rand(1).item() < 0.5:
            z_micro = torch.zeros_like(z_micro)

        # Decode from all levels
        z_all = torch.cat(z_macros + [z_micro], dim=-1)
        x_recon = self.decoder(z_all, torch.zeros_like(z_micro))

        self.step_counter += 1

        return {
            'z_macros': z_macros,
            'z_macro_preds': z_macro_preds,
            'z_micro': z_micro,
            'h_nexts': h_nexts,
            'x_recon': x_recon,
        }
```

### 9.7 Literature Connections (Mapping + Differences)

This framework is largely a **recomposition** of known ideas. What differs is the insistence on (i) a *discrete* macro state used for prediction/control, and (ii) explicit, online-auditable contracts tying representation, dynamics, and safety.

| Construct in this document | Closest literature | What is different here | Representative references |
|---|---|---|---|
| **Discrete macro register $K_t$** | discrete latents / vector quantization | $K_t$ is treated as the *control-relevant* state so closure/capacity checks are well-typed, not just a compression trick | {cite}`oord2017vqvae` |
| **Causal enclosure / closure loss** | predictive state representations, state abstraction, bisimulation-style sufficiency | closure is used as an explicit defect functional to certify “macro sufficiency” for predicting macro dynamics | {cite}`littman2001predictive,singh2004predictive,li2006towards,ferns2004metrics` |
| **Micro residual $z_{\mu}$ as nuisance** | rate–distortion / information bottleneck views of representation learning | micro is kept out of macro dynamics (micro-blind prediction) and treated as reconstruction-only nuisance | {cite}`tishby2015deep` |
| **Micro-blind macro dynamics $\bar P$** | latent world models / predictive models | macro dynamics is constrained to depend on $K$ (and action) only; violation is diagnosed as enclosure failure | {cite}`hafner2019dreamer,ha2018worldmodels,lecun2022path` |
| **Gate Nodes + Barriers** | CMDPs and safe RL constraints | constraints include *representation* and *interface* diagnostics (grounding, mixing, saturation, switching), not only expected cost | {cite}`altman1999constrained,achiam2017constrained,chow2018lyapunov` |
| **MaxEnt/KL control + path-entropy exploration** | entropy-regularized RL, KL-control, linearly-solvable control | exploration is defined on the discrete macro register and tied to capacity/grounding diagnostics | {cite}`haarnoja2018soft,todorov2009efficient,kappen2005path` |
| **State-space sensitivity metric $G$** | information geometry / natural gradient | emphasizes **state-space** sensitivity as a runtime regulator (in addition to parameter-space natural gradients) | {cite}`amari1998natural,schulman2015trpo,martens2015kfac` |
| **Belief evolution as filter + projection** | Bayesian filtering + constrained inference | maps Sieve events to explicit belief-space projections/reweightings (predict → update → project) | {cite}`rabiner1989tutorial` |
| **Entropic OT bridge (optional)** | entropic optimal transport / Schrödinger bridge | used only as a unifying *view* of KL-regularized path measures, not as a required ontology | {cite}`cuturi2013sinkhorn,leonard2014schrodinger` |

**Key pointers.**
- Causal emergence / macro closure as a modeling advantage: {cite}`hoel2017map,rosas2020reconciling`
- Information bottleneck perspective: {cite}`tishby2015deep`

### 9.8 Computational Costs

| Loss Component | Formula | Time Complexity | Overhead |
|----------------|---------|-----------------|----------|
| $\mathcal{L}_{\text{recon}}$ | $\Vert x - \hat{x} \Vert^2$ | $O(BD)$ | Baseline |
| $\mathcal{L}_{\text{vq}}$ | $\| \operatorname{sg}[z_e]-e_{K}\|^2 + \beta\|z_e-\operatorname{sg}[e_K]\|^2$ | $O(B|\mathcal{K}|)$ | +3-8% |
| $\mathcal{L}_{\text{closure}}$ | $-\log p_\psi(K_{t+1}\mid K_t,a_t)$ | $O(B|\mathcal{K}|)$ | +5% |
| $\mathcal{L}_{\text{slowness}}$ | $\Vert e_{K_t} - e_{K_{t-1}} \Vert^2$ | $O(Bd_m)$ | +2% |
| $\mathcal{L}_{\text{dispersion}}$ | $D_{KL}(q_{\text{micro}} \Vert \mathcal{N}(0,I))$ | $O(BZ_\mu)$ | +3% |
| **Total Overhead** | | | **~12-18%** |

**When to Use Split-Latent vs Standard:**

| Scenario | Recommendation |
|----------|----------------|
| High-frequency texture (games, video) | Use split-latent — separate predictive state from nuisance texture |
| Low-noise simulation (MuJoCo) | Standard may suffice — dynamics are already clean |
| Real-world robotics | Use split-latent — sensor noise is significant |
| Long-horizon planning | Use hierarchical split-latent — multiple timescales |
| Compute-constrained | Standard — split-latent adds ~12-18% overhead |

### 9.9 Control Theory Translation: Dictionary

To ensure rigorous connections to the established literature, we explicitly map Hypostructure components to their Control Theory and optimization equivalents.

| Hypostructure Component | Control / Optimization Term | Role |
|:------------------------|:------------------------------|:-----|
| **Critic** | **Lyapunov / value function** ($V$) | Defines stability regions and cost contours. |
| **Policy** | **Lie Derivative Controller** ($\mathcal{L}_f V$) | Actuator that maximizes negative definiteness of $\dot{V}$. |
| **World Model** | **System Dynamics** ($f(x, u)$) | The vector field governing the flow. |
| **Fragile Index** | **State-space sensitivity metric** ($G_{ij}$) | Local conditioning from Hessian/Fisher structure. |
| **StiffnessCheck** | **LaSalle's Invariance Principle** | Guarantee that the system does not get stuck in limit cycles. |
| **BarrierAction** | **Controllability Gramian** | Measure of whether the actuator can affect the state. |
| **Scaling coefficients** ($\alpha, \beta, \gamma, \delta$) | **Learning-dynamics coefficients** | Relative update/volatility scales per component. |
| **BarrierTypeII** | **Scaling Hierarchy** | Ensures faster components don't outrun slower ones. |

**Related Work:**
- {cite}`chang2019neural` — Neural Lyapunov Control
- {cite}`berkenkamp2017safe` — Safe Model-Based RL with Stability Guarantees
- {cite}`lasalle1960extent` — The Extent of Asymptotic Stability

### 9.10 Differential-Geometry View (No Physics): Curvature as Conditioning

The Fragile Agent uses differential geometry as a **regulation tool**: the metric $G$ (from Hessian curvature and/or Fisher information) defines a local notion of distance/conditioning in latent space, and therefore how aggressively the controller should update.

**Core relationship.** Objective curvature defines $G$; $G$ defines how updates are scaled:
$$
\theta_{t+1} = \theta_t + \eta\,G^{-1}\nabla_\theta \mathcal{L}.
$$

**Dictionary (geometry → optimization).**

| Differential geometry / optimization | Fragile Agent | Interpretation |
|:--|:--|:--|
| Metric tensor | $G$ | Local sensitivity / conditioning |
| Metric distance | $d_G(\cdot,\cdot)$ | Trust-region measure in latent space |
| Natural gradient | $G^{-1}\nabla$ | Preconditioned update direction |
| Ill-conditioning | $\lambda_{\max}(G)\gg 1$ | Updates should shrink to maintain stability |

**Adaptive conditioning (“freeze-out”).** When $G$ becomes extremely ill-conditioned, geometry-aware updates shrink ($G^{-1}\to 0$ along stiff directions), preventing destabilizing steps and signaling that the current regime is hard to model/control with available capacity.

**Related Work:**
- {cite}`bronstein2021geometric` — Geometric Deep Learning (geometry for inductive bias)
- {cite}`amari1998natural` — Natural Gradient (information geometry)

**Key Distinction from Geometric Deep Learning:**

Geometric Deep Learning uses geometry to design **architectures** (equivariant neural networks). The Fragile Agent uses geometry for **runtime regulation**: curvature is estimated from the critic/policy sensitivity and used to adapt update magnitudes online.

### 9.11 The Entropy-Regularized Objective Functional

We use a regularized objective (in nats) that trades off task cost, representation complexity, and control effort.

Define an instantaneous regularized objective:
$$
F_t
:=
V(Z_t)
+ \beta_K\big(-\log p_\psi(K_t)\big)
+ \beta_\mu D_{KL}\!\left(q(z_{\mu,t}\mid x_t)\ \Vert\ p(z_\mu)\right)
+ T_c D_{KL}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),
$$
where $Z_t=(K_t,z_{\mu,t})$ and all terms are measured in nats.

A practical monotonicity surrogate is:
$$
\mathcal{L}_{\downarrow F}
:=
\mathbb{E}\!\left[\mathrm{ReLU}\!\left(F_{t+1}-F_t\right)^2\right].
$$

**Interpretation.**
- $V(Z_t)$: task-aligned cost-to-go (critic).
- $\beta_K(-\log p_\psi(K_t))$: macro codelength penalty (MDL / rate).
- $\beta_\mu D_{KL}(q(z_{\mu,t}\mid x_t)\Vert p(z_\mu))$: micro residual penalty (treat $z_\mu$ as nuisance).
- $T_c D_{KL}(\pi\Vert\pi_0)$: KL-regularized control effort (deviation from prior/constraints).

This is the standard structure behind information bottlenecks/MDL (representation) and KL-regularized control / MaxEnt RL (policy), stated without additional metaphors.

### 9.12 Atlas-Manifold Dictionary: From Topology to Neural Networks

This section provides a translation dictionary connecting **manifold theory** to the **neural network implementations** described in Section 7.7.

#### Core Correspondences

| Manifold Theory | Neural Implementation | Role | Section Reference |
|-----------------|----------------------|------|-------------------|
| **Manifold $M$** | Input data distribution | The space to be embedded | — |
| **Chart $(U_i, \phi_i)$** | Expert network $i$ | Local embedding function | 7.7.1 |
| **Atlas $\mathcal{A} = \{U_i\}$** | Router + Experts ensemble | Global coverage | 7.7.5 |
| **Transition function $\tau_{ij}$** | Weighted soft blending | Chart overlap handling | 7.7.5 |
| **Riemannian metric $g$** | Orthogonality regularizer $\|W^TW - I\|^2$ | Distance preservation | 7.7.2 |
| **Geodesic $\gamma(t)$** | Latent space trajectory | Optimal path | 2.4 |
| **Curvature $R$** | Hessian of loss landscape | Local complexity | 2.5 |
| **Chart separation** | Separation loss | Chart partitioning | 7.7.4 |

#### Self-Supervised Learning Correspondences

| SSL Concept | VICReg Term | Geometric Interpretation | Failure Mode Prevented |
|-------------|-------------|-------------------------|------------------------|
| **Augmentation invariance** | $\mathcal{L}_{\text{inv}}$ | Metric tensor stability | Sensitivity to noise |
| **Non-collapse** | $\mathcal{L}_{\text{var}}$ | Non-degenerate metric | Trivial constant solution |
| **Decorrelation** | $\mathcal{L}_{\text{cov}}$ | Coordinate independence | Redundant dimensions |
| **Negative sampling** | (Not needed in VICReg) | Contrastive boundary | — |

#### Mixture of Experts Correspondences

| MoE Concept {cite}`jacobs1991adaptive` | Atlas Concept | Implementation |
|-----------------------------------|---------------|----------------|
| **Gating network** | Chart selector | Router with softmax |
| **Expert networks** | Local charts $\phi_i$ | Orthogonality-constrained encoders |
| **Expert specialization** | Chart coverage $U_i$ | Learned via separation loss |
| **Load balancing** | Atlas completeness | Balance loss $\|\text{usage} - 1/K\|^2$ |
| **Expert capacity** | Chart dimension | Latent dimension $d$ |

#### Loss Function Decomposition

The **Universal Loss** (Section 7.7.4) decomposes into geometric objectives:

| Loss Component | Geometric Objective | Manifold Property Enforced |
|----------------|--------------------|-----------------------------|
| $\mathcal{L}_{\text{inv}}$ | Metric stability | Local isometry |
| $\mathcal{L}_{\text{var}}$ | Non-degeneracy | Full rank Jacobian |
| $\mathcal{L}_{\text{cov}}$ | Orthonormality | Riemannian normal coordinates |
| $\mathcal{L}_{\text{entropy}}$ | Sharp boundaries | Distinct chart domains |
| $\mathcal{L}_{\text{balance}}$ | Complete coverage | Atlas covers all of $M$ |
| $\mathcal{L}_{\text{sep}}$ | Disjoint interiors | $U_i \cap U_j$ minimal |
| $\mathcal{L}_{\text{orth}}$ | Isometric embedding | $\|Wx\| = \|x\|$ |

#### When to Use Atlas Architecture

| Data Topology | Single Chart | Atlas Required | Why |
|---------------|--------------|----------------|-----|
| Euclidean $\mathbb{R}^n$ | ✓ | — | Trivially covered |
| Sphere $S^2$ | ✗ | ≥2 charts | Hairy Ball Theorem |
| Torus $T^2$ | ✗ | ≥4 charts | Non-trivial $H_1$ |
| Swiss Roll | ✓* | — | Topologically trivial |
| Disconnected components | ✗ | ≥k charts | k components |
| Mixed topology | ✗ | Adaptive | Data-dependent |

*Swiss Roll is topologically trivial but may benefit from multiple charts for geometric reasons (unrolling).

#### Key Citations

| Concept | Citation | Contribution |
|---------|----------|--------------|
| **Manifold Atlas** | {cite}`lee2012smooth` | *Smooth Manifolds* textbook |
| **Embedding Theorem** | {cite}`whitney1936differentiable` | Any $n$-manifold embeds in $\mathbb{R}^{2n}$ |
| **Mixture of Experts** | {cite}`jacobs1991adaptive` | Gated expert networks |
| **VICReg** | {cite}`bardes2022vicreg` | Collapse prevention without negatives |
| **Barlow Twins** | {cite}`zbontar2021barlow` | Redundancy reduction |
| **InfoNCE** | {cite}`oord2018cpc` | Contrastive predictive coding |
| **Information Geometry** | {cite}`saxe2019information` | Fisher information in NNs |

---

## 10. Intrinsic Motivation: Maximum-Entropy Exploration

The previous layers define representation ($K,z_\mu$), predictive dynamics ($\bar{P}$), and stability/value constraints ($V,G$, Sieve checks). This layer formalizes an **intrinsic exploration pressure** on the discrete macro register: prefer policies that keep the set of reachable future macrostates diverse, which supports reachability/controllability and reduces brittle overcommitment to narrow paths.

### 10.1 Path Entropy and Exploration Gradients

We work on the **macro model** (the discrete register). Assume a macro Markov kernel
$$
\bar{P}(k'\mid k,a),\qquad k,k'\in\mathcal{K},\ a\in\mathcal{A},
$$
which is the learned effective dynamics demanded by Causal Enclosure (Section 2.8).

**Definition 10.1.1 (Macro Path Distribution).** Fix a horizon $H\in\mathbb{N}$ and a (possibly stochastic) policy $\pi(a\mid k)$. The induced distribution over length-$H$ macro trajectories
$$
\tau := (K_{t+1},\dots,K_{t+H}) \in \mathcal{K}^H
$$
conditioned on $K_t=k$ is
$$
P_\pi(\tau\mid k)
:=
\sum_{a_{t:t+H-1}}
\prod_{h=0}^{H-1}\pi(a_{t+h}\mid K_{t+h})\ \bar{P}(K_{t+h+1}\mid K_{t+h},a_{t+h}).
$$
(For continuous $\mathcal{A}$, replace the sum by an integral with respect to the action reference measure.)

**Definition 10.1.2 (Causal Path Entropy).** The causal path entropy at $(k,H)$ under $\pi$ is the Shannon entropy of the path distribution:
$$
S_c(k,H;\pi) := H\!\left(P_\pi(\cdot\mid k)\right)
= -\sum_{\tau\in\mathcal{K}^H} P_\pi(\tau\mid k)\log P_\pi(\tau\mid k).
$$
This quantity is well-typed precisely because the macro register is discrete: there is no differential-entropy ambiguity.

**Definition 10.1.3 (Exploration Gradient, metric form).** Let $z_{\text{macro}}=e_k\in\mathbb{R}^{d_m}$ denote the code embedding of $k$ (Section 2.2b), and let $G$ be the relevant metric on the macro chart (Section 2.5). Define the exploration gradient as the metric gradient of path entropy:
$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi),
$$
where $T_c>0$ is an entropy-regularization coefficient. Operationally, gradients are taken through the continuous pre-quantization coordinates (straight-through VQ estimator); in the strictly symbolic limit, the gradient becomes a discrete preference ordering induced by $S_c(k,H;\pi)$.

**Interpretation (Exploration / Reachability).** $S_c(k,H;\pi)$ measures how many future macro-trajectories remain plausible from $k$ under $\pi$ and $\bar{P}$. Increasing $S_c$ preserves **future reachability**: the agent stays inside regions with many reachable, non-absorbing macrostates.

### 10.2 MaxEnt Duality: Utility + Entropy Regularization

The same object appears from a variational principle.

**Definition 10.2.1 (MaxEnt RL objective on macrostates).** Let $\mathcal{R}(k,a)$ be an instantaneous reward/cost-rate term (Section 1.1.2, Section 2.7) and let $\gamma\in(0,1)$ be the discount factor (dimensionless). The maximum-entropy objective is
$$
J_{T_c}(\pi)
:=
\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,A_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right],
$$
where $\mathcal{H}$ is Shannon entropy. This is the standard “utility + entropy regularization” objective.

**Regimes.**
- $T_c\to 0$: $\pi$ collapses toward determinism; behavior can be brittle under distribution shift.
- $T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat).
- The useful regime is intermediate: enough entropy to remain robust, enough utility to remain directed.

**Proposition 10.2.2 (Soft Bellman form, discrete actions).** Assume finite $\mathcal{A}$. Define the soft state value
$$
V^*(k) := \max_{\pi} \ \mathbb{E}\Big[\sum_{t\ge 0}\gamma^t(\mathcal{R}+T_c\mathcal{H})\ \Big|\ K_0=k\Big].
$$
Then $V^*$ satisfies the entropic Bellman fixed point
$$
V^*(k)
=
T_c \log \sum_{a\in\mathcal{A}}
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right),
$$
and the corresponding optimal policy is the softmax policy
$$
\pi^*(a\mid k)\propto
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}[V^*(k')]\right)\right).
$$
*Proof sketch.* Standard convex duality / log-sum-exp variational identity: maximizing expected reward plus entropy yields a softmax (exponential-family) distribution; substituting back produces the log-partition recursion. (This is the “soft”/MaxEnt Bellman equation used in SAC-like methods.)

**Consequence.** The same mathematics can be read as:
1) maximize reward while retaining policy entropy (MaxEnt RL), or
2) maximize reachability/diversity of future macro trajectories (intrinsic motivation).

---

## 11. Belief Dynamics: Prediction, Update, Projection

Sections 2–9 describe geometry, metrics, and effective macro dynamics. What they do *not* yet encode is the irreversibility of online learning: boundary observations and constraint enforcement are not invertible operations. This section states the belief-evolution template directly as **filtering + projection** on the discrete macro register.

**Relation to prior work.** The predict–update recursion below is standard Bayesian filtering for discrete latent states (HMM/POMDP belief updates) {cite}`rabiner1989tutorial,kaelbling1998planning`. The additional ingredient emphasized here is the explicit **projection/reweighting layer** induced by safety and consistency checks (Section 3): belief updates are not just “Bayes + dynamics”, but “Bayes + dynamics + constraints”.

### 11.1 Why Purely Closed Simulators Are Insufficient

A purely closed internal simulator can roll forward hypotheses, but it cannot *incorporate new boundary information* without a non-invertible update. Two irreversibilities are unavoidable:
1. **Assimilation:** boundary observations $x_{t+1}$ update the macro belief (Bayesian correction).
2. **Constraint enforcement:** the Sieve applies online projections/reweightings that remove unsafe/inconsistent mass (Gate Nodes / Barriers).

Both operations are information projections: they reduce uncertainty and/or discard parts of state-space mass in a way that cannot be undone from the post-update state alone.

### 11.2 Filtering Template on the Discrete Macro Register

Let $p_t\in\Delta^{|\mathcal{K}|-1}$ be the macro belief over $K_t$.

**Prediction (model step).** Given the learned macro kernel $\bar{P}(k'\mid k,a_t)$ (Section 2.8), define the one-step predicted belief
$$
\tilde p_{t+1}(k') := \sum_{k\in\mathcal{K}} p_t(k)\,\bar{P}(k'\mid k,a_t).
$$

**Update (observation step).** Given an emission/likelihood model $L_{t+1}(k'):=p(x_{t+1}\mid k')$ (or any calibrated score proportional to likelihood), the posterior belief is
$$
p_{t+1}(k')
:=
\frac{L_{t+1}(k')\,\tilde p_{t+1}(k')}{\sum_{j\in\mathcal{K}} L_{t+1}(j)\,\tilde p_{t+1}(j)}.
$$

This is the standard Bayesian filtering recursion for a discrete latent state (HMM/POMDP belief update) {cite}`rabiner1989tutorial,kaelbling1998planning`. Units: probabilities are dimensionless; log-likelihoods and entropies are measured in nats.

### 11.3 Sieve Events as Projections / Reweightings

When a check triggers, we apply a *projection-like* operator to the belief state. Two common forms are:

- **Hard projection (mask + renormalize):**
  $$
  p'_{t}(k)\propto p_t(k)\cdot \mathbb{I}\!\left[\text{feasible}(k)\right].
  $$
  Example: feasibility defined by a cost budget $V(k)\le V_{\max}$ (CostBoundCheck).

- **Soft reweighting (exponential tilt):**
  $$
  p'_t(k)\propto p_t(k)\,\exp\!\left(-\lambda\cdot \text{penalty}(k)\right),
  $$
  which implements a differentiable “push away” from unstable regions.

These are classical constrained-inference moves (mirror descent / I-projection style), and they are the belief-space counterpart of the Gate Nodes.

### 11.4 Over/Under Coupling as Forgetting vs Ungrounded Inference

The coupling window in Theorem 15.1.3 reflects a trade-off:
- **Over-coupling:** noisy or overly aggressive updates drive mixing; the macro register loses stable structure (forgetting / symbol dispersion).
- **Under-coupling:** insufficient boundary information causes internal rollouts to dominate (model drift / ungrounded inference; Mode D.C).

The Sieve (Sections 3–6) is the control layer that keeps the agent inside the regime where macrostates remain stable *and* grounded.

---

## 12. Correspondence Table: Filtering / Control Template

The table below is a dictionary from standard **filtering and constrained inference** to the Fragile Agent components. It is purely classical: belief evolution is “predict → update → project”.

| Filtering / Control Object | Fragile Agent Equivalent | Role |
| :--- | :--- | :--- |
| Belief state $p_t(k)$ | Macro belief over $\mathcal{K}$ | Summary statistic for control |
| Prediction $\tilde p_{t+1}=\bar{P}^\top p_t$ | Macro dynamics model $\bar{P}(k'\mid k,a)$ | One-step forecast |
| Likelihood $L_{t+1}(k)=p(x_{t+1}\mid k)$ | Shutter/emission score for macrostates | Boundary grounding signal |
| Bayes update $p_{t+1}\propto L_{t+1}\odot \tilde p_{t+1}$ | Assimilation step | Incorporate observations |
| Projection / reweighting $p'_t$ | Sieve checks (CostBoundCheck, CompactCheck, …) | Enforce feasibility/stability |
| Entropy $H(p_t)$ | Macro uncertainty / symbol mixing | Detect collapse vs dispersion |
| KL-control $D_{KL}(\pi\Vert\pi_0)$ | Control-effort regularizer | Penalize deviation from prior |

---

## 13. Duality of Exploration and Soft Optimality

This section makes the exploration layer precise: the exploration gradient (Section 10.1.3) is dual to entropy-regularized (soft) optimal control once the macro channel is discrete.

### 13.1 Formal Definitions (Path Space, Causal Entropy, Exploration Gradient)

**Definition 13.1.1 (Causal Path Space).** For a macrostate $k\in\mathcal{K}$ and horizon $H$, define the future macro path space
$$
\Gamma_H(k) := \mathcal{K}^H.
$$

**Definition 13.1.2 (Path Probability).** $P_\pi(\tau\mid k)$ is the induced path probability from Definition 10.1.1.

**Definition 13.1.3 (Causal Entropy).** $S_c(k,H;\pi)$ is the Shannon entropy of $P_\pi(\cdot\mid k)$ (Definition 10.1.2).

**Definition 13.1.4 (Exploration gradient, covariant form).** On a macro chart with metric $G$ (Section 2.5),
$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,\nabla_G S_c(k,H;\pi).
$$

### 13.2 The Equivalence Theorem (Duality of Causal Regulation)

We state the equivalence in the setting where it is unambiguous.

**Theorem 13.2.1 (Equivalence of Entropy-Regularized Control Forms; discrete macro).** Assume:
1. finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$,
2. an enclosure-consistent macro kernel $\bar{P}(k'\mid k,a)$,
3. bounded reward flux $\mathcal{R}(k,a)$.

Then the following are equivalent characterizations of the same optimal control law:

1. **MaxEnt control (utility + freedom):** $\pi^*$ maximizes $J_{T_c}(\pi)$ from Definition 10.2.1.
2. **Exponentially tilted trajectory measure (KL-regularization).** Fix a reference (prior) policy $\pi_0(a\mid k)$ with full support (uniform when $\mathcal{A}$ is finite). For the finite-horizon trajectory
   $$
   \omega := (A_t,\dots,A_{t+H-1},K_{t+1},\dots,K_{t+H}),
   $$
   the optimal controlled path law admits an exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar{P}$:
   $$
   P^*(\omega\mid K_t=k)\ \propto\
   \Big[\prod_{h=0}^{H-1}\pi_0(A_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h})\Big]\,
   \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}(K_{t+h},A_{t+h})\right),
   $$
   where the normalizer is the (state-dependent) path-space normalizing constant.
3. **Soft Bellman optimality:** the optimal value function $V^*$ satisfies the soft Bellman recursion of Proposition 10.2.2, and $\pi^*$ is the corresponding softmax policy.

Moreover, the path-space log-normalizer is (up to scaling) the soft value. Gradients of the log-normalizer therefore induce a well-defined exploration direction in any differentiable macro coordinate system. The link between soft optimality and path entropy is cleanest when stated as a KL-regularized variational identity: if $P_0(\omega\mid k)$ denotes the reference trajectory measure induced by $\pi_0$ and $\bar{P}$, then
$$
\log Z(k)
=
\sup_{P(\cdot\mid k)}
\left\{
\frac{1}{T_c}\,\mathbb{E}_{P}\!\left[\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}\right]
-D_{KL}(P(\cdot\mid k)\Vert P_0(\cdot\mid k))
\right\},
$$
and the optimizer is exactly the exponentially tilted law $P^*$. In the special case where $P_0$ is uniform (or treated as constant), the KL term differs from Shannon path entropy by an additive constant, recovering the standard “maximize entropy subject to expected reward” view.

*Proof sketch.* Set up the constrained variational problem “maximize path entropy subject to an expected reward constraint.” The Euler–Lagrange condition yields an exponential-family distribution on paths. The normalizer obeys dynamic programming and equals the soft value. Differentiating the log-normalizer yields the corresponding exploration-gradient direction.

---

## 14. Implementation Note: Entropy-Regularized Optimal Transport Bridge

This is optional machinery, but it provides a clean path-space view of KL-regularized control and filtering.

**Entropic bridge (Schrödinger bridge) viewpoint.** Given a reference dynamics (e.g. the macro kernel $\bar{P}$, or a continuous diffusion on $\mathcal{Z}_\mu$) and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals. This is the rigorous “most likely flow under entropy regularization” principle (entropic optimal transport) {cite}`cuturi2013sinkhorn,leonard2014schrodinger`.

In the Fragile Agent:
- the reference process is the world model’s internal rollout,
- the boundary observations induce marginal constraints (via the shutter),
- the Sieve imposes feasibility constraints (via projections),
so each training update can be read as an entropic optimal transport step on belief trajectories.

---

## 15. Theorem: The Information–Stability Threshold (Coupling Window)

The coupling-window view implies a necessary **window condition**: coupling must be strong enough to remain grounded (BoundaryCheck) but not so strong that the macro register loses coherence (dispersion/mixing).

We state this as a rate balance rather than an ill-typed scalar comparison.

**Definition 15.1.1 (Grounding rate).** Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step:
$$
\lambda_{\text{in}} := \mathbb{E}[G_t].
$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

**Definition 15.1.2 (Mixing rate).** Let $S_t:=H(K_t)$ be the macro entropy. The *mixing rate* is the expected entropy growth not attributable to purposeful exploration:
$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].
$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

**Theorem 15.1.3 (Information–stability window; operational).** A necessary condition for stable, grounded macrostates is the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that, along typical trajectories,
$$
\epsilon \le I(X_t;K_t) \quad\text{and}\quad H(K_t)\le \log|\mathcal{K}|-\epsilon,
$$
and the net entropy balance satisfies
$$
\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}.
$$
Violations correspond to identifiable barrier modes:
- If $I(X;K)\approx 0$: under-coupling → ungrounded inference / decoupling (Mode D.C).
- If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion → symbol dispersion (BarrierScat).

*Remark.* This theorem is intentionally stated at the level of measurable information quantities (Gate Nodes) so it can be audited online; strengthening it to a sufficient condition requires specifying the macro kernel class and a contraction inequality (e.g. log-Sobolev / Doeblin-type conditions).

---

## 16. Summary: Unified Information-Theoretic Control View

| Level | Formalism | Law |
| :--- | :--- | :--- |
| Geometry | Riemannian $(\mathcal{Z},G)$ | Distance measured by a sensitivity metric (Fisher/Hessian) |
| Boundary | Markov blanket $B_t$ | Environment = boundary law $P_{\partial}$ (Section 1.1) |
| Exploration | Causal entropy / MaxEnt RL | Reachability pressure via path entropy on $\mathcal{K}$ (Section 10) |
| Belief Dynamics | Filtering + projection | Predict → update → project (Section 11) |
| Optimality | Soft Bellman / log-normalizer | Soft value = log-normalizer; exploration gradient from path entropy (Section 13) |

**Fragile conclusion.** The agent is a controller with explicit information and stability constraints. Macro symbols remain meaningful only inside the coupling window (Theorem 15.1.3); outside it, the system either exhibits ungrounded inference (under-coupling) or loses macro structure through excessive mixing/dispersion (over-coupling).

---

## 17. Capacity-Constrained Metric Law: Geometry from Interface Limits

Section 9.10 used a “gravity” analogy to motivate curvature as a regulator. This section removes the analogy: the curvature law is derived as a structural response to **information-theoretic constraints** induced by the agent’s finite-bandwidth boundary (Markov blanket).

The key idea is operational: **the representational complexity of the internal state is bounded by the capacity of the interface channel.** When the agent operates near this bound, curvature appears as the geometric mechanism that prevents internal information volume from exceeding what can be grounded at the interface.

### 17.1 The Boundary–Bulk Information Inequality

**Definition 17.1.1 (DPI / boundary-capacity constraint).** Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition 1.1.1). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:
$$
I_{\text{bulk}} \;\le\; C_{\partial},
$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is the amount of information the agent can stably maintain in $\mathcal{Z}$ without violating Causal Enclosure (no internal source term $\sigma$; Definition 2.11.7).
Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$.

**Definition 17.1.2 (Bulk information volume).** Let $\rho_I(z,t)\ge 0$ be an information density on $\mathcal{Z}$ with units of nats per unit Riemannian volume $d\mu_G=\sqrt{|G|}\,dz^n$ ($n=\dim\mathcal{Z}$). Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by
$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \rho_I(z,t)\, d\mu_G.
$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in Section 2.11; here the integral measures grounded structure in nats.

**Definition 17.1.3 (Boundary capacity: area law at finite resolution).** Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity bound is an area law:
$$
C_{\partial}(\partial\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G,
$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface).
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, the most conservative computable proxy is
$$
C_{\partial}\ \approx\ \mathbb{E}[I(X_t;K_t)]\ \le\ \log|\mathcal{K}|,
$$
which is exactly Node 13 (BoundaryCheck) and Theorem 15.1.3’s grounding condition.

### 17.2 Main Result (Capacity-Saturated Metric Law)

The detailed variational construction is recorded in Appendix A. The main consequence is an Euler–Lagrange identity that ties curvature of the latent geometry to a risk-induced tensor under a finite-capacity boundary.

**Theorem 17.2.1 (Capacity-constrained metric law).** Under the regularity and boundary-clamping hypotheses stated in Appendix A, and under the soundness condition that bulk structure is boundary-grounded (no internal source term $\sigma$ on $\operatorname{int}(\mathcal{Z})$; Definition 2.11.7), stationarity of a capacity-constrained curvature functional implies
$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},
$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the loss-gradient (risk) tensor induced by a chosen risk Lagrangian density $\mathcal{L}_{\text{risk}}(V;G)$. Units: $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* Curvature is the geometric mechanism that prevents the internal information volume (Definition 17.1.2) from exceeding the boundary’s information bandwidth (Definition 17.1.3) while remaining grounded.

**Implementation hook.** The squared residual of this identity defines a capacity-consistency regularizer $\mathcal{L}_{\text{cap-metric}}$; see Appendix B for the consolidated list of loss definitions and naming conventions.

---

## 18. Conclusion

The Fragile Agent is a capacity- and stability-constrained control specification: the environment is described by an observation generator $P_{\partial}$, the agent’s internal state is a split macro/micro bundle, and stability is enforced by explicit defect functionals rather than implicit optimizer heuristics.

1. Discretizing the macro channel $K$ turns enclosure, capacity, and grounding into well-typed information constraints ($I(X;K)$, $H(K)$, closure cross-entropy).
2. The critic induces a Fisher/Hessian sensitivity geometry; the policy becomes a regulated flow on a curved manifold, with stability and coupling audited by Gate Nodes and Barriers.
3. Exploration and control are expressed via MaxEnt/KL-control and causal path entropy on $\mathcal{K}$; belief evolution is filtering + projection (Section 11).
4. When representational complexity is constrained by finite interface capacity, the latent metric obeys a capacity-constrained consistency law; deviations yield computable consistency defects (Section 17).

Appendix A records the full derivations. Appendix B consolidates notation and all regularization losses.

---

## Appendix A: Full Derivations (Capacity-Constrained Curvature Functional)

### A.1 Capacity-Constrained Curvature Functional (Variational Principle)

If the agent attempts to maintain internal structure such that $I_{\text{bulk}}(\mathcal{Z})>C_{\partial}(\partial\mathcal{Z})$, it has exceeded what can be grounded at the boundary; this is an enclosure violation and must be rejected by the Sieve (Section 3, Node 13).

In the optimal / sound regime the constraint is active (saturated):
$$
I_{\text{bulk}}(\mathcal{Z}) = C_{\partial}(\partial\mathcal{Z}).
$$

We now encode this as a variational constraint to obtain a *metric law* from information limits.

**Definition A.1.1 (Boundary capacity form).** Define the boundary capacity $(n\!-\!1)$-form
$$
\omega_{\partial} := \frac{1}{\eta_\ell}\, dA_G,
$$
so that $C_{\partial}(\partial\mathcal{Z})=\oint_{\partial\mathcal{Z}}\omega_{\partial}$ (Definition 17.1.3).

**Definition A.1.2 (Boundary-capacity constraint functional).** Define the saturation functional
$$
\mathcal{C}[G,V]
:=
\underbrace{\int_{\mathcal{Z}} \rho_I(G,V)\, d\mu_G}_{I_{\text{bulk}}}
\;-\;
\underbrace{\oint_{\partial\mathcal{Z}}\omega_{\partial}}_{C_{\partial}},
$$
where $\rho_I(G,V)$ is an *information density* (nats per unit $d\mu_G$) compatible with the agent’s representation scheme (Definition 17.1.2). This $\rho_I$ is distinct from the belief density $p$ used in Section 2.11. When $\rho_I$ is instantiated via the split shutter, the most conservative computable proxy is a global one, $I_{\text{bulk}}\approx \mathbb{E}[I(X;K)]$ (Node 13), and the window theorem (Theorem 15.1.3) supplies the admissible operating range.

**Definition A.1.3 (Risk Lagrangian density).** Fix a smooth potential $V\in C^\infty(\mathcal{Z})$. A canonical risk Lagrangian density is the scalar-field functional
$$
\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V),
$$
where $U:\mathbb{R}\to\mathbb{R}$ is a (possibly learned) on-site potential capturing non-gradient costs. (The sign convention is chosen for a Riemannian metric; see e.g. Lee, *Riemannian Manifolds*, 2018, for the variational identities used below.)

**Definition A.1.4 (Capacity-constrained curvature functional).** Let $R(G)$ be the scalar curvature of $G$ and let $\Lambda\in\mathbb{R}$ be a constant. Define the constrained functional
$$
\mathcal{S}[G,V]
:=
\int_{\mathcal{Z}}\left(R(G)-2\Lambda + 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G
\;-\;
2\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial},
$$
with coupling $\kappa\in\mathbb{R}$. The last term is the explicit boundary capacity penalty, and $\Lambda$ is a bulk capacity offset that remains once the boundary is clamped at finite resolution.

*Remark (why $\Lambda$ is allowed).* A constant term in the integrand is the simplest coordinate-invariant scalar density and produces a $\Lambda G_{ij}$ term in the metric Euler–Lagrange equation. Here $\Lambda$ plays the role of a baseline curvature / capacity offset.

### A.2 First Variation (Expanded Derivation)

We work in the standard calculus of variations on Riemannian manifolds with boundary. Assume:
1) $G$ is $C^2$ and $V$ is $C^2$ (so curvature and gradients are well-defined),
2) variations $\delta G^{ij}$ are smooth, symmetric, and compactly supported in $\mathcal{Z}$ or satisfy Dirichlet boundary conditions $\delta G^{ij}\vert_{\partial\mathcal{Z}}=0$ (the boundary is clamped by the sensorium; cf. Definition 2.11.9 / Theorem 2.11.10).

Under these hypotheses, the first variation of $\mathcal{S}$ is well-defined as a distribution; the standard identities below can be found in standard differential-geometry references (e.g. {cite}`lee2018riemannian`).

#### A.2.1 Variation of the volume form

Let $d\mu_G=\sqrt{|G|}\,dz^n$. The determinant identity gives
$$
\delta \sqrt{|G|}
=
-\frac{1}{2}\sqrt{|G|}\,G_{ij}\,\delta G^{ij},
$$
equivalently $\delta d\mu_G = -\tfrac12\,G_{ij}\,\delta G^{ij}\, d\mu_G$.

#### A.2.2 Variation of the curvature term

Write the curvature functional as $\mathcal{S}_{\text{geo}}[G]:=\int_{\mathcal{Z}}R(G)\,d\mu_G$. The variation splits as
$$
\delta(R\,d\mu_G)
=
(\delta R)\,d\mu_G + R\,\delta d\mu_G.
$$
For the scalar curvature, use
$$
R = G^{ij}R_{ij},
$$
hence
$$
\delta R
=
R_{ij}\,\delta G^{ij} + G^{ij}\,\delta R_{ij}.
$$
The Palatini identity gives
$$
\delta R_{ij} = \nabla_k(\delta \Gamma^k_{ij})-\nabla_j(\delta\Gamma^k_{ik}),
$$
and the Christoffel variation is
$$
\delta\Gamma^k_{ij}
=
\frac12\,G^{k\ell}\left(\nabla_i \delta G_{j\ell}+\nabla_j \delta G_{i\ell}-\nabla_\ell \delta G_{ij}\right),
$$
where $\delta G_{ij} = -G_{ia}G_{jb}\,\delta G^{ab}$.

Substituting and collecting terms yields the standard decomposition
$$
\delta\mathcal{S}_{\text{geo}}
=
\int_{\mathcal{Z}}\left(R_{ij}-\frac12 R\,G_{ij}\right)\delta G^{ij}\,d\mu_G
\;+\;
\oint_{\partial\mathcal{Z}} \mathcal{B}_{\text{curv}}(\delta G,\nabla\delta G),
$$
where $\mathcal{B}_{\text{curv}}$ is an explicit boundary $(n\!-\!1)$-form built from $\delta\Gamma$ (equivalently from $\delta G$ and its first derivatives). For a well-posed Dirichlet variational problem one can add an appropriate boundary term to cancel $\mathcal{B}_{\text{curv}}$. In our setting the boundary is clamped, so we impose $\delta G\vert_{\partial\mathcal{Z}}=0$ and the boundary term vanishes.

#### A.2.3 Variation of the risk term

Let $\mathcal{S}_{\text{risk}}[G,V] := \int_{\mathcal{Z}}\mathcal{L}_{\text{risk}}(V;G)\,d\mu_G$. Define the (Riemannian-signature) risk tensor by
$$
T_{ij}
:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\text{risk}})}{\delta G^{ij}}.
$$
Holding $V$ fixed under $\delta G$ and using $\delta d\mu_G = -\tfrac12 G_{ij}\delta G^{ij} d\mu_G$ gives the standard identity
$$
\delta \mathcal{S}_{\text{risk}}
=
-\frac12 \int_{\mathcal{Z}} T_{ij}\,\delta G^{ij}\,d\mu_G.
$$
For the risk Lagrangian
$$
\mathcal{L}_{\text{risk}}=\tfrac12 G^{ab}\nabla_a V\nabla_b V + U(V),
$$
the explicit computation yields
$$
T_{ij}
=
\nabla_i V\,\nabla_j V
-G_{ij}\left(\frac12\,G^{ab}\nabla_a V\nabla_b V + U(V)\right).
$$

#### A.2.4 Capacity (boundary) term and the emergence of $\Lambda$

The explicit boundary penalty $-\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial}$ depends only on the induced boundary metric through $dA_G$. Under the clamped boundary condition $\delta G\vert_{\partial\mathcal{Z}}=0$, its first variation vanishes.

The remaining constant $\Lambda$ in Definition A.1.4 plays the role of the bulk Lagrange multiplier for finite boundary capacity: it is the only diffeomorphism-invariant way to represent an additive “grounding floor” required for non-degenerate macrostates (cf. Theorem 15.1.3). Formally, varying $-2\Lambda\int_{\mathcal{Z}} d\mu_G$ gives
$$
\delta\left(-2\Lambda\int_{\mathcal{Z}} d\mu_G\right)
=
\int_{\mathcal{Z}} \Lambda G_{ij}\,\delta G^{ij}\,d\mu_G.
$$

### A.3 Recovery of the Metric Stationarity Condition

**Lemma A.3.1 (Divergence-to-boundary conversion).** For any sufficiently regular information flux field $\mathbf{j}$ on $\mathcal{Z}$,
$$
\int_{\mathcal{Z}} \operatorname{div}_G(\mathbf{j})\, d\mu_G
=
\oint_{\partial \mathcal{Z}} \langle \mathbf{j}, \mathbf{n}\rangle\, dA_G,
$$
which is the Riemannian divergence theorem underlying the global balance equation in Theorem 2.11.10.

**Theorem A.3.2 (Capacity-consistency identity; proof of Theorem 17.2.1).** Under the hypotheses of Section A.2, stationarity of $\mathcal{S}[G,V]$ with respect to arbitrary variations $\delta G^{ij}$ that vanish on $\partial\mathcal{Z}$ implies the Euler–Lagrange equation
$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},
$$
with $T_{ij}$ given by Section A.2.3.

*Proof.* Combine Sections A.2.1–A.2.4:
$$
\delta\mathcal{S}
=
\int_{\mathcal{Z}}\left[
\left(R_{ij}-\frac12 R\,G_{ij}\right)
\;+\;
\Lambda G_{ij}
\;-\;
\kappa T_{ij}
\right]\delta G^{ij}\,d\mu_G
\;+\;
\text{(boundary terms)}.
$$
Boundary terms vanish under the clamped boundary condition (or after adding an appropriate boundary term). Because $\delta G^{ij}$ is arbitrary in the interior, the fundamental lemma of the calculus of variations implies the bracketed tensor must vanish pointwise almost everywhere, yielding the stated identity (see e.g. Evans, *Partial Differential Equations*, 2010, for the functional-analytic lemma).

*Interpretation.* The Ricci curvature governs local volume growth; enforcing a boundary-limited bulk information volume forces the metric to stretch/compress coordinates so that information-dense regions (large $\|\nabla V\|$ and/or large $U(V)$) do not generate bulk structure that cannot be grounded at the boundary.

*Remark (regularizer).* The squared residual of this identity defines the capacity-consistency loss $\mathcal{L}_{\text{cap-metric}}$; see Appendix B.

---

## Appendix B: Units, Parameters, and Coefficients (Audit Table)

### B.1 Base Units (Information + Steps)

We use a purely information-theoretic unit system:
- **Information / cost:** nats ($\mathrm{nat}$).
- **Time:** discrete interaction steps ($\mathrm{step}$).

Conventions:
- Entropies $H(\cdot)$, mutual information $I(\cdot;\cdot)$, and divergences $D_{KL}$ are measured in $\mathrm{nat}$.
- Value/cost scalars ($V$, $F_t$, budgets, thresholds) are measured in $\mathrm{nat}$.
- Per-step rates (HJB terms, $\Delta V$, any “cost rate”) are measured in $\mathrm{nat/step}$.
- Latent coordinates ($z$, $z_\mu$, code embeddings $e_k$) are treated as **normalized/dimensionless**; any physical units should be absorbed into preprocessing and encoder normalization.

### B.2 Parameter / Coefficient Units (by Role)

| Symbol | Meaning (context) | Units |
|---|---|---|
| $t$ | step index | $\mathrm{step}$ |
| $\Delta t$ | optional mapping from steps to wall-clock | $\mathrm{s/step}$ |
| $r_t$ | reward/cost per step | $\mathrm{nat}$ |
| $\mathcal{R}$ | reward/cost rate (when written as a rate) | $\mathrm{nat/step}$ |
| $V$ | value / cost-to-go | $\mathrm{nat}$ |
| $\Delta V$ | value change per step | $\mathrm{nat/step}$ |
| $\mathfrak{D}$ | control-effort / regularization rate term | $\mathrm{nat/step}$ |
| $\lambda$ | Lyapunov rate in $\dot V\le -\lambda V$ (continuous-time form) | $\mathrm{step^{-1}}$ |
| $\gamma$ | discount factor (MaxEnt RL) | dimensionless |
| $H$ | horizon / planning depth | $\mathrm{step}$ |
| $T_c$ | entropy-regularization coefficient (MaxEnt RL) | dimensionless |
| $\beta$ | exponential-family scale in $\exp(-\beta V)$ | dimensionless |
| $\Theta$ | local conditioning proxy (Definition 2.11.1) | dimensionless (when $z$ normalized) |
| $\epsilon$ | numeric stabilizer / threshold | inherits compared quantity |
| $\eta$ | step size / learning-rate symbol | dimensionless (in normalized coordinates) |
| $\beta$ | VQ-VAE commitment weight (Section 2.2b / 3.3) | dimensionless |
| $\beta_\mu$ | micro KL weight (VQ-VAE) | dimensionless |
| $\beta_K$ | macro codelength weight (rate term) | dimensionless |
| $\lambda_{\text{use}}$ | codebook usage regularizer weight | dimensionless |
| $\lambda_{\text{*}}$ | composite-loss weights (e.g. $\lambda_{\text{shutter}},\lambda_{\text{ent}},\lambda_{\text{zeno}}$) | dimensionless |
| $\lambda,\mu,\nu$ | VICReg component weights | dimensionless |
| $\alpha,\beta,\gamma,\delta$ | scaling coefficients (Section 3.2) | dimensionless |
| $V_{\text{max}},V_{\text{limit}},V_{\text{proxy}},V_{\text{true}},B_{\text{switch}}$ | risk/cost budgets and thresholds | $\mathrm{nat}$ |
| $\lambda_{\text{in}},\lambda_{\text{mix}}$ | grounding/mixing information rates | $\mathrm{nat/step}$ |
| $q_{k\to k'}$ | macro transition probabilities | dimensionless |
| $C_{\partial}$ | boundary information capacity | $\mathrm{nat}$ |
| $I_{\text{bulk}}$ | bulk information volume | $\mathrm{nat}$ |
| $\ell$ | boundary resolution scale | boundary-length units (chosen) |
| $\eta_\ell$ | boundary area-per-nat at resolution $\ell$ | $[dA_G]/\mathrm{nat}$ |
| $\Lambda$ | curvature/capacity offset constant (metric law) | $[z]^{-2}$ |
| $\kappa$ | coupling in $R_{ij}-\\tfrac12R G_{ij}+\\Lambda G_{ij}=\\kappa T_{ij}$ | chosen so $\kappa T_{ij}$ has units $[z]^{-2}$ |

### B.3 Symbol Overload (Important)

Some Greek letters are intentionally overloaded in different submodels:
- $\beta$ appears as (i) the exponential-family scale in $\exp(-\beta V)$, (ii) the VQ-VAE commitment weight, and (iii) the softmax/logit scale in contrastive losses (often written $\tau^{-1}$); treat each by its local definition and units above.
- $\gamma$ appears as (i) discount factor, and (ii) the World Model volatility scaling coefficient $\gamma$ (Section 3.2).
- $\lambda$ appears as (i) Lyapunov rate ($\mathrm{step^{-1}}$), (ii) generic loss weights (dimensionless), and (iii) other Lagrange multipliers (units stated locally).

---

## References

```{bibliography}
```
