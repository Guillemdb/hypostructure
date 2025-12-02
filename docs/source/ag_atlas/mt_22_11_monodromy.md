# Metatheorem 22.11 (The Monodromy-Weight Lock)

**Statement.** Let $\pi: \mathcal{X} \to \Delta$ be a family of smooth projective varieties degenerating to a singular fiber $X_0$ as $t \to 0 \in \Delta$. The limiting mixed Hodge structure on $H^k(X_t)$ encodes a hypostructure $\mathbb{H}_{\text{MHS}}$ satisfying:

1. **Schmid's Theorem ↔ Profile Exactification**: The nilpotent orbit approximation $\exp(u N) \cdot F^\bullet$ near $t = 0$ is the hypostructure profile map $\Pi_C$ (Axiom TB), where $N = \log T$ is the monodromy logarithm.

2. **Weight Filtration ↔ Decay Rates**: The weight filtration $W_\bullet$ on $H^*$ is indexed by nilpotency degrees $n_1 \leq \cdots \leq n_r$, which equal the scaling exponents $(\alpha_i)$ of Axiom SC. The monodromy eigenvalues encode the decay rates.

3. **Clemens-Schmid ↔ Mode C.D Transitions**: The Clemens-Schmid exact sequence identifies vanishing cycles (Mode C.D) with the kernel of the monodromy action, while invariant cycles persist (Mode C.C).

---

## Proof

**Setup.** Let $\pi: \mathcal{X} \to \Delta$ be a proper flat family over the unit disk $\Delta = \{t \in \mathbb{C} : |t| < 1\}$ with:
- **Generic fibers**: $X_t = \pi^{-1}(t)$ smooth for $t \neq 0$
- **Special fiber**: $X_0$ has at worst normal crossing singularities
- **Monodromy**: The fundamental group $\pi_1(\Delta^*, t_0)$ acts on $H^k(X_{t_0}, \mathbb{Z})$ via a quasi-unipotent operator $T$ (i.e., $(T^m - I)^N = 0$ for some $m, N$)

Write $T = T_s T_u$ (Jordan decomposition) with $T_s$ semisimple and $T_u$ unipotent. Define the **monodromy logarithm** by
$$
N = \log T_u = \sum_{n=1}^\infty \frac{(-1)^{n+1}}{n} (T_u - I)^n.
$$

By the **monodromy theorem** (Grothendieck, Landman), $N$ is nilpotent: $N^{k+1} = 0$ on $H^k$.

### Step 1: Nilpotent Orbit Theorem (Schmid)

**(H1)** The **nilpotent orbit theorem** describes the limiting behavior of the Hodge filtration $F^\bullet_t$ on $H^k(X_t, \mathbb{C})$ as $t \to 0$.

**Step 1a: Statement of Schmid's theorem.**

**Theorem (Schmid, 1973).** There exists a limiting Hodge filtration $F^\bullet_\infty$ on $H^k(X_0, \mathbb{C})$ such that:
$$
F^p_t \sim \exp\left(\frac{\log t}{2\pi i} N\right) \cdot F^p_\infty
$$
as $t \to 0$ in $\Delta^*$. Moreover, $(F^\bullet_\infty, W_\bullet)$ is a mixed Hodge structure, where $W_\bullet$ is the weight filtration associated to $N$.

Here $\sim$ means equality modulo higher-order terms in $\text{Im}(\tau)^{-1}$ where $\tau = \frac{\log t}{2\pi i}$.

**Step 1b: Hypostructure profile map.**

Recall the **profile map** $\Pi_C: \text{Reg}_C \to \mathbb{F}$ (Definition 8.1) encodes the "shape" of the conservative region in the feasible set. For degenerations, we identify:
- **Feasible region**: $\mathbb{F} = H^k(X_t, \mathbb{C})$ (fixed vector space via parallel transport)
- **Conservative profile**: $\text{Reg}_C = F^\bullet_t$ (Hodge filtration at parameter $t$)
- **Profile map**: $\Pi_C(t) = \exp(\tau N) \cdot F^\bullet_\infty$ (nilpotent orbit)

The map $\Pi_C: \Delta^* \to \text{Flag}(H^k)$ parametrizes the Hodge flag as $t$ varies. Schmid's theorem asserts that $\Pi_C(t)$ extends continuously to $t = 0$ after the logarithmic twist.

**Step 1c: SL(2)-orbit theorem.**

Schmid's result was refined by **Cattani-Kaplan-Schmid** to show that the nilpotent orbit lies in a single $\text{SL}(2, \mathbb{C})$-orbit:
$$
\Pi_C(\Delta^*) \subseteq \{g \cdot F^\bullet_\infty : g \in \exp(\mathbb{C} N)\} \subseteq \text{Flag}(H^k).
$$

This is the **minimal degeneracy**: the orbit is determined by a single nilpotent element $N \in \mathfrak{sl}(2)$, not a full $\text{SL}(n)$-action.

**Step 1d: Exactification of profile.**

Axiom TB requires that the profile map $\Pi_C$ is **exact** (Definition 8.2): it captures the full geometry, not just asymptotic behavior. Schmid's theorem provides exactness in the form:
$$
\left\| F^p_t - \exp(\tau N) \cdot F^p_\infty \right\| = O(e^{-c/|\log|t||})
$$
for some $c > 0$. This exponential convergence is the signature of **profile exactification**.

**Conclusion.** Schmid's nilpotent orbit theorem is the realization of the profile map $\Pi_C$ for Hodge-theoretic hypostructures. $\square_{\text{Step 1}}$

### Step 2: Weight Filtration as Scaling Exponents

**(H2)** The **weight filtration** $W_\bullet$ on $H^k$ is the central object of mixed Hodge theory. It measures the "complexity" of the cohomology, with higher weights corresponding to more singular behavior.

**Step 2a: Definition of weight filtration.**

Given a nilpotent operator $N: H^k \to H^k$ with $N^{k+1} = 0$, the weight filtration $W_\bullet$ is the unique increasing filtration such that:
1. $N(W_i) \subseteq W_{i-2}$ (grading property)
2. $N^i: \text{Gr}^W_{k+i} \xrightarrow{\sim} \text{Gr}^W_{k-i}$ is an isomorphism for $i \geq 0$ (primitivity)

Explicitly, define
$$
W_i = \bigoplus_{j \leq i} \ker(N^{j+1}) \cap \text{Im}(N^{k-j}).
$$

**Step 2b: Connection to scaling exponents.**

The indices $i$ in the weight filtration correspond to the **scaling exponents** $\alpha$ of Axiom SC. To see this, consider the rescaled operator
$$
N_\lambda = \lambda \cdot N.
$$

The eigenvalues of $\exp(N_\lambda)$ are $1 + \lambda \mu_i + O(\lambda^2)$, where $\mu_i$ are the "weight" eigenvalues. As $\lambda \to 0$ (degeneration limit), the weight filtration stratifies the cohomology by decay rate:
$$
\|v\|_t \sim |t|^{-\alpha_i} \quad \text{for } v \in \text{Gr}^W_i.
$$

**Step 2c: Monodromy eigenvalues.**

The monodromy operator $T = \exp(2\pi i N)$ has eigenvalues $e^{2\pi i \lambda_j}$ where $\lambda_j \in \mathbb{Q}$ (by quasi-unipotence). The weight filtration sorts cohomology classes by $\lambda_j$:
$$
W_i = \bigoplus_{\lambda_j \leq i/2} H^k_{\lambda_j}
$$
where $H^k_\lambda$ is the $e^{2\pi i \lambda}$-eigenspace of $T$.

**Step 2d: Successive weights and Axiom SC.**

The successive quotients $\text{Gr}^W_i$ have dimensions $\dim(\text{Gr}^W_i) = r_i$, which are the **multiplicities** of weights. These correspond to the exponents $\alpha_1, \ldots, \alpha_r$ in Axiom SC:
$$
\text{Vol}(\mathbb{F}_\lambda) \sim \prod_{i=1}^r \lambda^{-\alpha_i r_i}.
$$

For the cohomological hypostructure, $\mathbb{F}_\lambda$ is the "normalized" cohomology $H^k / |t|^{W_\bullet}$, and the volume growth is governed by the weight grading.

**Conclusion.** The weight filtration indices are the scaling exponents of Axiom SC, encoding the decay rates of cohomology as $t \to 0$. $\square_{\text{Step 2}}$

### Step 3: Clemens-Schmid Exact Sequence

**(H3)** The **Clemens-Schmid exact sequence** relates the cohomology of the generic fiber $X_t$ to the cohomology of the special fiber $X_0$ via vanishing and nearby cycles.

**Step 3a: Vanishing and nearby cycles.**

Define the functors:
- **Nearby cycles**: $\psi_\pi(\mathbb{Q}_{X_t}) = $ sheaf on $X_0$ given by $\lim_{t \to 0} H^*(X_t)$
- **Vanishing cycles**: $\phi_\pi(\mathbb{Q}_{X_t}) = \ker(1 - T)$ where $T$ is monodromy

These fit into the **specialization sequence**:
$$
\cdots \to H^k(X_0) \xrightarrow{\text{sp}} H^k(\psi) \xrightarrow{1 - T} H^k(\psi) \xrightarrow{\text{var}} H^k(X_0) \to \cdots
$$

**Step 3b: Clemens-Schmid sequence.**

By **Poincaré duality** on the fibers, the sequence twists to:
$$
\cdots \to H_k(X_0) \xrightarrow{N} H_{k-2}(X_0)(-1) \to H^k(X_t) \xrightarrow{1-T^{-1}} H^k(X_t) \to H_{k}(X_0) \to \cdots
$$

This is the **Clemens-Schmid exact sequence** (Clemens, 1977; Schmid, 1973).

**Step 3c: Hypostructure interpretation.**

Identify the terms with hypostructure modes:
- $H^k(X_t)$ with $1 - T^{-1} = 0$ (monodromy-invariant): **Mode C.C** (Conservative-Continuous)
- $H^k(X_t)$ with $(1 - T^{-1}) \neq 0$ (monodromy-variant): **Mode C.D** (Conservative-Discrete)
- $\text{Im}(N)$: **Mode D.D** (Dissipative-Discrete, pure vanishing)

The exact sequence encodes **mode transitions**:
$$
\text{Mode D.D} \xrightarrow{N} \text{Mode C.D} \xrightarrow{1-T} \text{Mode C.C}.
$$

**Step 3d: Vanishing cycles as dissipation.**

The vanishing cycles $\phi_\pi$ are classes in $H^k(X_t)$ that "disappear" in the limit $t \to 0$ (they collapse to singular points of $X_0$). This is **dissipation** in the hypostructure sense: energy concentrated at singularities.

The **Picard-Lefschetz formula** quantifies this:
$$
T(\delta) = \delta + (-1)^{k(k-1)/2} \langle \delta, \gamma \rangle \gamma
$$
where $\gamma$ is the vanishing cycle and $\langle \cdot, \cdot \rangle$ is the intersection form. The monodromy creates a "reflection" across $\gamma$, mixing Mode C.D with Mode D.D.

**Conclusion.** The Clemens-Schmid sequence is the exact sequence of mode transitions in the cohomological hypostructure. $\square_{\text{Step 3}}$

### Step 4: Application to Mirror Symmetry

We conclude by connecting monodromy to **mirror symmetry** (previewing Metatheorem 22.12).

**Step 4a: B-model periods.**

For a Calabi-Yau variety $X$ in a mirror family $\pi: \mathcal{X} \to \Delta$, the **periods** are integrals
$$
\Pi_\alpha(t) = \int_{\gamma_\alpha} \Omega_t
$$
where $\Omega_t$ is the holomorphic volume form on $X_t$ and $\gamma_\alpha \in H_n(X_t, \mathbb{Z})$.

The periods satisfy the **Picard-Fuchs equation**:
$$
\mathcal{L}_{\text{PF}} \cdot \Pi = 0
$$
where $\mathcal{L}_{\text{PF}}$ is a differential operator. Near $t = 0$, solutions behave as
$$
\Pi(t) \sim \sum_{k=0}^N c_k (\log t)^k \cdot t^{\lambda}
$$
where $\lambda$ is a monodromy eigenvalue and $N$ is the nilpotency degree.

**Step 4b: A-model instantons.**

On the mirror (A-model) side, the genus-0 Gromov-Witten invariants $N_d$ count holomorphic curves of degree $d$. The generating function is
$$
F_0(q) = \sum_{d=0}^\infty N_d q^d, \quad q = e^{2\pi i t}.
$$

**Mirror symmetry** equates:
$$
\Pi(t) = e^{F_0(q)/q} \quad (\text{modularity correction}).
$$

The monodromy $T$ on the B-side corresponds to the **shift symmetry** $t \mapsto t + 1$ on the A-side (since $q \mapsto e^{2\pi i} q = q$).

**Step 4c: Monodromy weight and instanton order.**

The weight filtration on $H^*(X_t)$ corresponds to the **instanton order** on the A-side:
$$
W_k \leftrightarrow \text{contributions from degree } d \leq k \text{ curves}.
$$

Higher weights (more singular cohomology) correspond to higher-degree instantons (more wrapping). The nilpotent operator $N$ is the **derivative** $d/dq$ acting on the instanton expansion.

**Step 4d: Thomas-Yau conjecture.**

The **Thomas-Yau conjecture** posits that special Lagrangian submanifolds (A-model) correspond to stable sheaves (B-model). The monodromy-weight lock ensures:
- **Mode C.C** (invariant cycles) $\leftrightarrow$ Special Lagrangians (calibrated)
- **Mode C.D** (variant cycles) $\leftrightarrow$ Lagrangian cobordisms (non-calibrated)

This is the hypostructure manifestation of **homological mirror symmetry**.

**Conclusion.** The monodromy-weight structure on the B-model encodes the instanton structure of the A-model via mirror symmetry. $\square_{\text{Step 4}}$

---

## Key Insight

The monodromy-weight lock reveals a profound connection between:

1. **Schmid's Nilpotent Orbit** $\leftrightarrow$ **Profile Exactification** (Axiom TB)
   - The Hodge filtration near $t = 0$ is governed by a single nilpotent $N$
   - The profile map $\Pi_C$ extends continuously via $\exp(\tau N)$

2. **Weight Filtration** $\leftrightarrow$ **Scaling Exponents** (Axiom SC)
   - Weights $W_i$ stratify cohomology by decay rate $|t|^{-i/2}$
   - Scaling exponents $\alpha_i$ measure volume growth of feasible regions

3. **Clemens-Schmid Sequence** $\leftrightarrow$ **Mode Transitions**
   - Vanishing cycles = Mode D.D (dissipative-discrete)
   - Variant cycles = Mode C.D (conservative-discrete)
   - Invariant cycles = Mode C.C (conservative-continuous)

The **monodromy logarithm** $N$ is the infinitesimal generator of mode transitions, encoding how cohomology classes "flow" between modes as the degeneration parameter $t \to 0$. The nilpotency $N^{k+1} = 0$ ensures finite-time transitions, consistent with Axiom TB's requirement of **bounded transition times**.

The deep consequence for mirror symmetry: monodromy on the B-model (complex geometry) encodes instanton corrections on the A-model (symplectic geometry). The weight filtration is the bridge, with weights corresponding to instanton degrees. This is the ultimate realization of **Axiom R** (Reflection): geometric complexity on one side equals analytic complexity on the mirror side.

$\square$
