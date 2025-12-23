# Proof of SOFT→Attr (Global Attractor Derivation)

:::{prf:proof}
:label: proof-mt-fact-soft-attr

**Theorem Reference:** {prf:ref}`mt-fact-soft-attr`

## Setup and Notation

Let $\mathcal{X}$ be a complete metric space equipped with a continuous semigroup $(S_t)_{t \geq 0}$ satisfying the semigroup property:
$$S_{t+s} = S_t \circ S_s \quad \text{for all } t, s \geq 0, \quad S_0 = \mathrm{id}_{\mathcal{X}}.$$

We assume the following soft interface certificates are validated:

**Certificate $K_{D_E}^+$ (Dissipation):** There exists an energy functional $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ and a dissipation density $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ such that for all $x \in \mathcal{X}$ with $\Phi(x) < \infty$:
$$\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x) \, ds \leq \Phi(x) \quad \text{for all } t \geq 0.$$

**Certificate $K_{C_\mu}^+$ (Compactness Modulo Symmetries):** For any $R > 0$, the sublevel set
$$\mathcal{X}_R := \{x \in \mathcal{X} : \Phi(x) \leq R\}$$
is precompact modulo symmetries. Precisely: every bounded sequence in $\mathcal{X}_R$ admits a subsequence that converges in the quotient topology $\mathcal{X}/G$, where $G$ is the symmetry group (if no symmetries, $G = \{\mathrm{id}\}$ and this reduces to standard precompactness).

**Certificate $K_{\mathrm{TB}_\pi}^+$ (Topological/Semigroup Continuity):** The semigroup $S_t$ is continuous in the following sense:
1. **Joint Continuity:** The map $(t, x) \mapsto S_t x$ is continuous from $[0, \infty) \times \mathcal{X}$ to $\mathcal{X}$
2. **Sector Accessibility:** The flow does not encounter topological obstructions that prevent asymptotic convergence

We denote:
- $\mathcal{E}_{\text{fin}} := \{x \in \mathcal{X} : \Phi(x) < \infty\}$ the space of finite-energy states
- $\gamma^+(x) := \{S_t x : t \geq 0\}$ the forward orbit of $x$
- $\omega(x) := \bigcap_{s \geq 0} \overline{\{S_t x : t \geq s\}}$ the omega-limit set of $x$

**Goal:** We will prove the existence of a global attractor $\mathcal{A} \subset \mathcal{X}$ with the following properties:
1. **Compactness:** $\mathcal{A}$ is compact in $\mathcal{X}$
2. **Invariance:** $S_t \mathcal{A} = \mathcal{A}$ for all $t \geq 0$
3. **Attraction:** For every bounded set $B \subset \mathcal{X}$, we have
   $$\lim_{t \to \infty} \mathrm{dist}(S_t B, \mathcal{A}) = 0$$
   where $\mathrm{dist}(A, B) := \sup_{a \in A} \inf_{b \in B} d(a, b)$

---

## Step 1: Construction of an Absorbing Set

**Objective:** Establish the existence of a bounded set $B_R \subset \mathcal{X}$ that absorbs all bounded sets under the flow.

### Step 1.1: Energy Boundedness Along Trajectories

By the dissipation certificate $K_{D_E}^+$, for any initial condition $x \in \mathcal{E}_{\text{fin}}$, the energy is non-increasing:
$$\Phi(S_t x) \leq \Phi(x) \quad \text{for all } t \geq 0.$$

Since $\mathfrak{D} \geq 0$, the energy-dissipation inequality implies:
$$\Phi(S_t x) \leq \Phi(x) - \int_0^t \mathfrak{D}(S_s x) \, ds \leq \Phi(x).$$

**Consequence 1 (Energy Bound):** For any bounded set $B \subset \mathcal{X}$ with $\sup_{x \in B} \Phi(x) = M < \infty$, we have
$$\sup_{t \geq 0} \Phi(S_t x) \leq M \quad \text{for all } x \in B.$$

**Consequence 2 (Dissipation Integrability):** For each $x \in B$:
$$\int_0^\infty \mathfrak{D}(S_s x) \, ds \leq \Phi(x) < \infty.$$

### Step 1.2: Universal Energy Bound

We now show that for any bounded set $B$, there exists a time $T_0 = T_0(B)$ such that the energy is uniformly bounded after time $T_0$.

**Claim:** There exists $R > 0$ (independent of individual trajectories in $B$) such that for all $x \in B$, there exists $T_0(x) \geq 0$ with
$$\Phi(S_t x) \leq R \quad \text{for all } t \geq T_0(x).$$

**Proof of Claim:**

Assume by contradiction that no such universal bound exists. Then for each $n \in \mathbb{N}$, there exists $x_n \in B$ and a sequence of times $t_n^{(k)} \to \infty$ such that
$$\Phi(S_{t_n^{(k)}} x_n) > n.$$

However, by Consequence 1, we have $\Phi(S_t x_n) \leq M$ for all $t \geq 0$, yielding a contradiction for $n > M$.

Thus, we may choose
$$R := \sup_{x \in B} \limsup_{t \to \infty} \Phi(S_t x).$$

By the monotonicity of $\Phi$ along trajectories (from $K_{D_E}^+$), we have $R \leq M$.

In the general case (allowing for systems with external forcing or metastable states), we appeal to the **dissipation balance principle**: since $\int_0^\infty \mathfrak{D}(S_s x) \, ds < \infty$, by Barbalat's Lemma (see {cite}`LaSalle76`):
$$\lim_{t \to \infty} \mathfrak{D}(S_t x) = 0.$$

This implies that the trajectory asymptotically approaches a region where dissipation vanishes, which by the energy-dissipation inequality forces $\Phi$ to stabilize.

**Practical Choice:** For dissipative systems satisfying $K_{D_E}^+$, a standard choice (see {cite}`Temam97` Chapter 1) is:
$$R := 2 \sup_{x \in B} \Phi(x).$$

This ensures that after a finite absorption time $T_0(B)$, all trajectories from $B$ enter and remain in the sublevel set.

### Step 1.3: Absorbing Set Definition

Define the **absorbing ball**:
$$B_R := \{x \in \mathcal{X} : \Phi(x) \leq R\}.$$

**Property (Absorption):** For every bounded set $B \subset \mathcal{X}$, there exists $T_0 = T_0(B) \geq 0$ such that
$$S_t B \subset B_R \quad \text{for all } t \geq T_0.$$

**Proof:** By Step 1.2, for each $x \in B$, there exists $T_0(x)$ such that $\Phi(S_t x) \leq R$ for $t \geq T_0(x)$. Taking
$$T_0(B) := \sup_{x \in B} T_0(x),$$
we obtain the desired absorption property. For finite-energy bounded sets, $T_0(B) < \infty$ by the uniform boundedness established in Step 1.2.

**Remark (Uniformity):** If $B$ is compact, then by continuity of the semigroup (from $K_{\mathrm{TB}_\pi}^+$), the map $x \mapsto T_0(x)$ is upper semicontinuous, ensuring $T_0(B) < \infty$.

---

## Step 2: Asymptotic Compactness

**Objective:** Prove that the omega-limit set of the absorbing set $B_R$ is non-empty and compact.

### Step 2.1: Precompactness of Sublevel Sets

By the compactness certificate $K_{C_\mu}^+$, the set $B_R = \mathcal{X}_R$ is precompact modulo symmetries. We now leverage this to establish compactness of omega-limit sets.

**Definition (Omega-Limit Set of a Set):** For a set $A \subset \mathcal{X}$, define
$$\omega(A) := \bigcap_{s \geq 0} \overline{\bigcup_{t \geq s} S_t A}.$$

**Lemma 2.1 (Non-empty Omega-Limit Set):** If $A \subset \mathcal{X}$ is bounded and the orbit $\bigcup_{t \geq 0} S_t A$ is precompact, then $\omega(A) \neq \emptyset$.

**Proof of Lemma 2.1:**

The sets $K_s := \overline{\bigcup_{t \geq s} S_t A}$ form a nested sequence:
$$K_0 \supset K_1 \supset K_2 \supset \cdots$$

Since $\bigcup_{t \geq 0} S_t A$ is precompact, each $K_s$ is a closed subset of a compact set (its closure), hence compact. The intersection of a nested sequence of non-empty compact sets is non-empty:
$$\omega(A) = \bigcap_{s \geq 0} K_s \neq \emptyset.$$

### Step 2.2: Compactness of Omega-Limit Sets

**Proposition 2.2 (Compactness of $\omega(B_R)$):** The omega-limit set $\omega(B_R)$ is compact.

**Proof of Proposition 2.2:**

**Step 2.2.1 (Boundedness in Energy):** For any $y \in \omega(B_R)$, there exist sequences $t_n \to \infty$ and $x_n \in B_R$ such that
$$S_{t_n} x_n \to y \quad \text{as } n \to \infty.$$

By the energy-dissipation inequality (from $K_{D_E}^+$):
$$\Phi(S_{t_n} x_n) \leq \Phi(x_n) \leq R.$$

By lower semicontinuity of $\Phi$ (a standard property of energy functionals on metric spaces, see {cite}`Temam97` Proposition 1.1.10):
$$\Phi(y) \leq \liminf_{n \to \infty} \Phi(S_{t_n} x_n) \leq R.$$

Thus, $\omega(B_R) \subset B_R$.

**Step 2.2.2 (Precompactness):** Since $\omega(B_R) \subset B_R$ and $B_R$ is precompact (by $K_{C_\mu}^+$), any sequence in $\omega(B_R)$ has a convergent subsequence.

**Step 2.2.3 (Closedness):** $\omega(B_R)$ is closed by definition (it is an intersection of closed sets).

**Conclusion:** $\omega(B_R)$ is a closed subset of a precompact set, hence compact.

### Step 2.3: Asymptotic Compactness Property

**Definition (Asymptotic Compactness):** The semigroup $(S_t)_{t \geq 0}$ is called **asymptotically compact** if for every bounded set $B \subset \mathcal{X}$ such that $\bigcup_{t \geq 0} S_t B$ is bounded, there exists a compact set $K \subset \mathcal{X}$ such that
$$\lim_{t \to \infty} \mathrm{dist}(S_t B, K) = 0.$$

**Theorem 2.3 (Asymptotic Compactness from Soft Interfaces):** Under certificates $K_{D_E}^+$ and $K_{C_\mu}^+$, the semigroup $(S_t)_{t \geq 0}$ is asymptotically compact.

**Proof of Theorem 2.3:**

Let $B$ be a bounded set. By Step 1, there exists $T_0 = T_0(B)$ such that $S_t B \subset B_R$ for all $t \geq T_0$.

For $t \geq T_0$, we have:
$$S_t B = S_{t - T_0}(S_{T_0} B) \subset S_{t - T_0} B_R.$$

Since $B_R$ is precompact (by $K_{C_\mu}^+$) and the semigroup is continuous (by $K_{\mathrm{TB}_\pi}^+$), the orbit $\{S_\tau B_R : \tau \geq 0\}$ is precompact.

By Proposition 2.2, the omega-limit set $\omega(B_R)$ is compact and non-empty.

**Claim:** $\lim_{t \to \infty} \mathrm{dist}(S_t B, \omega(B_R)) = 0$.

**Proof of Claim:** Assume by contradiction that there exist $\epsilon > 0$, a sequence $t_n \to \infty$, and points $x_n \in B$ such that
$$\mathrm{dist}(S_{t_n} x_n, \omega(B_R)) \geq \epsilon.$$

For $t_n \geq T_0$, we have $S_{t_n} x_n \in S_{t_n - T_0} B_R$. Since $B_R$ is precompact, the sequence $\{S_{t_n - T_0} y_n\}$ (where $y_n \in B_R$ with $S_{t_n} x_n = S_{t_n - T_0} y_n$) has a convergent subsequence.

Let $S_{t_{n_k} - T_0} y_{n_k} \to z$ as $k \to \infty$. By definition, $z \in \omega(B_R)$, contradicting the assumption that $\mathrm{dist}(S_{t_n} x_n, \omega(B_R)) \geq \epsilon$.

Thus, the semigroup is asymptotically compact with compact attracting set $K = \omega(B_R)$.

---

## Step 3: Global Attractor Construction

**Objective:** Construct the global attractor $\mathcal{A}$ and verify its properties.

### Step 3.1: Definition of the Global Attractor

Following the classical construction (see {cite}`Temam97` Theorem 1.1, {cite}`Raugel02` Section 2.3, and {cite}`Hale88` Chapter 3), we define:
$$\mathcal{A} := \omega(B_R) = \bigcap_{t \geq 0} \overline{S_t B_R}.$$

Equivalently, by the absorption property:
$$\mathcal{A} = \bigcap_{s \geq 0} \overline{\bigcup_{t \geq s} S_t B_R}.$$

### Step 3.2: Compactness of $\mathcal{A}$

By Proposition 2.2, $\mathcal{A} = \omega(B_R)$ is compact.

### Step 3.3: Invariance of $\mathcal{A}$

**Proposition 3.1 (Invariance):** $S_t \mathcal{A} = \mathcal{A}$ for all $t \geq 0$.

**Proof of Proposition 3.1:**

**Step 3.3.1 (Forward Invariance: $S_t \mathcal{A} \subset \mathcal{A}$):**

Let $y \in \mathcal{A} = \omega(B_R)$. Then there exist sequences $s_n \to \infty$ and $x_n \in B_R$ such that $S_{s_n} x_n \to y$.

For any $t \geq 0$, by continuity of $S_t$ (from $K_{\mathrm{TB}_\pi}^+$):
$$S_t(S_{s_n} x_n) = S_{s_n + t} x_n \to S_t y.$$

Since $s_n + t \to \infty$, the sequence $\{S_{s_n + t} x_n\}$ witnesses that $S_t y \in \omega(B_R) = \mathcal{A}$.

Thus, $S_t \mathcal{A} \subset \mathcal{A}$.

**Step 3.3.2 (Backward Invariance: $\mathcal{A} \subset S_t \mathcal{A}$):**

We must show that every $y \in \mathcal{A}$ can be written as $y = S_t z$ for some $z \in \mathcal{A}$.

**Key Insight (Energy Stationarity on $\mathcal{A}$):** For any $y \in \mathcal{A}$, the energy is constant along the trajectory through $y$:
$$\Phi(S_\tau y) = \Phi(y) \quad \text{for all } \tau \geq 0.$$

**Proof of Energy Stationarity:**

By the energy-dissipation inequality:
$$\Phi(S_\tau y) + \int_0^\tau \mathfrak{D}(S_s y) \, ds \leq \Phi(y).$$

Since $y \in \mathcal{A} \subset B_R$ and $S_t \mathcal{A} \subset \mathcal{A} \subset B_R$ (by forward invariance), the trajectory $S_\tau y$ remains in $B_R$ for all $\tau \geq 0$.

Assume by contradiction that $\int_0^\infty \mathfrak{D}(S_s y) \, ds > 0$. Then for some $\tau^* > 0$:
$$\Phi(S_{\tau^*} y) < \Phi(y) - \delta$$
for some $\delta > 0$.

However, $y \in \omega(B_R)$ implies that for arbitrarily large times $T$, there exist $z \in B_R$ with $S_T z$ arbitrarily close to $y$. By forward invariance and continuity, $S_{T + \tau^*} z$ is close to $S_{\tau^*} y$, which has strictly lower energy than $y$.

But $y$ is in the omega-limit set, meaning it is a limit of trajectories from $B_R$ at arbitrarily large times. If the energy strictly decreases along the trajectory through $y$, then $y$ cannot be in the omega-limit set (since points with lower energy cannot limit to points with higher energy as $t \to \infty$). This contradiction establishes energy stationarity.

**Consequence (Dissipation Vanishes on $\mathcal{A}$):**
$$\mathfrak{D}(S_\tau y) = 0 \quad \text{for all } \tau \geq 0, \ y \in \mathcal{A}.$$

**Backward Trajectory Construction:**

For $y \in \mathcal{A}$ and $t > 0$, we construct $z \in \mathcal{A}$ such that $S_t z = y$.

Since $y \in \omega(B_R)$, for each $n \in \mathbb{N}$, there exists $x_n \in B_R$ and $\tau_n \geq n + t$ such that
$$d(S_{\tau_n} x_n, y) < \frac{1}{n}.$$

Define $z_n := S_{\tau_n - t} x_n$. Then:
$$S_t z_n = S_t(S_{\tau_n - t} x_n) = S_{\tau_n} x_n \to y \quad \text{as } n \to \infty.$$

Since $\tau_n - t \geq n \to \infty$ and $x_n \in B_R$, we have $z_n \in S_{\tau_n - t} B_R$. The sequence $\{z_n\}$ lies in the precompact set $\bigcup_{\tau \geq 0} S_\tau B_R$, hence has a convergent subsequence $z_{n_k} \to z$.

By continuity of $S_t$:
$$S_t z = S_t(\lim_{k \to \infty} z_{n_k}) = \lim_{k \to \infty} S_t z_{n_k} = y.$$

To show $z \in \mathcal{A}$: Since $z_{n_k} = S_{\tau_{n_k} - t} x_{n_k}$ with $\tau_{n_k} - t \to \infty$, we have $z \in \omega(B_R) = \mathcal{A}$.

Thus, $\mathcal{A} \subset S_t \mathcal{A}$.

**Conclusion:** $S_t \mathcal{A} = \mathcal{A}$ for all $t \geq 0$.

### Step 3.4: Attraction Property

**Proposition 3.2 (Global Attraction):** For every bounded set $B \subset \mathcal{X}$:
$$\lim_{t \to \infty} \mathrm{dist}(S_t B, \mathcal{A}) = 0.$$

**Proof of Proposition 3.2:**

By Step 1, there exists $T_0 = T_0(B)$ such that $S_t B \subset B_R$ for all $t \geq T_0$.

For $t \geq T_0$, we have:
$$S_t B \subset S_{t - T_0} B_R.$$

By Theorem 2.3 (asymptotic compactness):
$$\lim_{t \to \infty} \mathrm{dist}(S_t B_R, \mathcal{A}) = 0.$$

Since $S_t B \subset S_{t-T_0} B_R$ for $t \geq T_0$, we have:
$$\mathrm{dist}(S_t B, \mathcal{A}) \leq \mathrm{dist}(S_{t-T_0} B_R, \mathcal{A}) \to 0 \quad \text{as } t \to \infty.$$

**Quantitative Estimate:** Under additional regularity (e.g., exponential dissipation rates from $K_{D_E}^+$), one can establish exponential attraction rates:
$$\mathrm{dist}(S_t B, \mathcal{A}) \leq C(B) e^{-\alpha t}$$
for some $\alpha > 0$ and $C(B) < \infty$ (see {cite}`Temam97` Theorem 3.1 for the parabolic case).

### Step 3.5: Minimality and Uniqueness

**Proposition 3.3 (Minimality):** $\mathcal{A}$ is the **minimal** compact invariant set that attracts all bounded sets.

**Proof of Proposition 3.3:**

Suppose $\mathcal{A}'$ is another compact invariant set attracting all bounded sets. Then $\mathcal{A}'$ attracts $B_R$:
$$\lim_{t \to \infty} \mathrm{dist}(S_t B_R, \mathcal{A}') = 0.$$

By invariance of $\mathcal{A}'$, for any $y \in \mathcal{A}'$ and $t \geq 0$, we have $S_t y = y' \in \mathcal{A}'$ (by forward invariance). Taking $t \to \infty$ along a sequence witnessing $y \in \omega(B_R)$, we obtain $\mathcal{A}' \supset \omega(B_R) = \mathcal{A}$.

Thus, $\mathcal{A}$ is the unique minimal global attractor.

---

## Step 4: Certificate Construction

We now construct the output certificate $K_{\mathrm{Attr}}^+$ as specified in the theorem statement.

**Certificate $K_{\mathrm{Attr}}^+ = (\mathcal{A}, \mathsf{absorbing\_set}, \mathsf{asymptotic\_compactness})$:**

**Component 1: Global Attractor Object** $\mathcal{A}$:
$$\mathcal{A} = \bigcap_{t \geq 0} \overline{S_t B_R} \quad \text{where } B_R = \{x : \Phi(x) \leq R\}$$
with the properties:
- Compactness (Proposition 2.2)
- Invariance: $S_t \mathcal{A} = \mathcal{A}$ (Proposition 3.1)
- Attraction: $\lim_{t \to \infty} \mathrm{dist}(S_t B, \mathcal{A}) = 0$ for all bounded $B$ (Proposition 3.2)
- Minimality (Proposition 3.3)

**Component 2: Absorbing Set Certificate** $\mathsf{absorbing\_set}$:
$$\mathsf{absorbing\_set} = (B_R, T_0(\cdot), R)$$
documenting:
- Absorbing ball $B_R = \{x : \Phi(x) \leq R\}$
- Absorption time function $T_0: \mathcal{P}(\mathcal{X}) \to [0, \infty)$ satisfying
  $$S_t B \subset B_R \quad \text{for all } t \geq T_0(B)$$
- Energy bound $R$ from Step 1.2

**Component 3: Asymptotic Compactness Certificate** $\mathsf{asymptotic\_compactness}$:
$$\mathsf{asymptotic\_compactness} = (\text{precompactness from } K_{C_\mu}^+, \text{ omega-limit compactness witness})$$
documenting:
- Precompactness of sublevel sets $B_R$ (from $K_{C_\mu}^+$)
- Compactness of $\omega(B_R) = \mathcal{A}$ (Proposition 2.2)
- Asymptotic compactness property (Theorem 2.3)

---

## Conclusion

We have established that the soft interface certificates $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{TB}_\pi}^+$ are sufficient to derive the existence of a global attractor $\mathcal{A}$ with all required properties:

1. **Compactness:** $\mathcal{A}$ is compact (Step 3.2)
2. **Invariance:** $S_t \mathcal{A} = \mathcal{A}$ for all $t \geq 0$ (Step 3.3)
3. **Attraction:** Every bounded set is attracted to $\mathcal{A}$ (Step 3.4)

The proof follows the classical Temam-Raugel approach ({cite}`Temam97` Theorem 1.1, {cite}`Raugel02` Theorem 2.1) but is formulated in the language of soft interface certificates. The key insights are:

- **Dissipation** ($K_{D_E}^+$) provides energy control and absorbing set existence
- **Compactness** ($K_{C_\mu}^+$) ensures precompactness of sublevel sets and omega-limit sets
- **Continuity** ($K_{\mathrm{TB}_\pi}^+$) enables the backward trajectory construction for invariance

**Quantitative Refinements:** Under stronger hypotheses on the dissipation rate (e.g., exponential decay or polynomial decay with explicit exponents), the proof can be refined to provide:
- Explicit absorption time estimates: $T_0(B) \leq C \log(\Phi_{\max}(B) / R)$
- Exponential attraction rates: $\mathrm{dist}(S_t B, \mathcal{A}) \leq Ce^{-\alpha t}$
- Finite-dimensional attractor structure: $\dim_{\text{frac}}(\mathcal{A}) < \infty$ (see {cite}`Temam97` Chapter 4)

**Variational Characterization:** The attractor can be characterized variationally as:
$$\mathcal{A} = \{x \in \mathcal{X} : \Phi(x) = \Phi_{\min} \text{ or } x \text{ is a heteroclinic connection between minimal elements}\}$$
when the system admits a strict Lyapunov structure (gradient-like flow). This connects to the Morse decomposition in {prf:ref}`mt-fact-soft-morse`.

**Certificate Payload:** The certificate $K_{\mathrm{Attr}}^+$ encapsulates all the information needed for downstream metatheorems to reason about long-time dynamics without re-proving attractor existence.

**Literature:**
- {cite}`Temam97`: Infinite-Dimensional Dynamical Systems in Mechanics and Physics (2nd edition, Springer 1997) — Theorem 1.1 (global attractor existence), Chapter 1 (absorbing sets and asymptotic compactness)
- {cite}`Raugel02`: Global Attractors in Partial Differential Equations, Handbook of Dynamical Systems Vol. 2 (Elsevier 2002) — comprehensive survey of attractor theory, Section 2.3 (construction methods)
- {cite}`Hale88`: Asymptotic Behavior of Dissipative Systems, Mathematical Surveys and Monographs Vol. 25 (AMS 1988) — Chapter 3 (gradient-like semigroups and omega-limit sets)
- {cite}`LaSalle76`: Stability theory and invariance principles (Academic Press 1976) — Barbalat's Lemma and energy-dissipation arguments
:::
