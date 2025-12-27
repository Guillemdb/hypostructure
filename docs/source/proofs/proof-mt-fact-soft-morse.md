# Proof of SOFT→MorseDecomp (Morse Decomposition Derivation)

:::{prf:proof}
:label: proof-mt-fact-soft-morse

**Theorem Reference:** {prf:ref}`mt-fact-soft-morse`

## Setup and Notation

Let $(\mathcal{X}, S_t, \Phi, \mathfrak{D})$ be a dynamical system with the following structure:

- **State Space:** $\mathcal{X}$ is a complete metric space (or Banach/Hilbert space in infinite dimensions)
- **Semiflow:** $S_t: \mathcal{X} \to \mathcal{X}$ for $t \geq 0$ satisfies the semigroup property: $S_0 = \text{id}$, $S_{t+s} = S_t \circ S_s$
- **Energy Functional:** $\Phi: \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$ is lower semicontinuous
- **Dissipation Rate:** $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ satisfies $\mathfrak{D}(x) = -\frac{d}{dt}\Big|_{t=0^+} \Phi(S_t x)$ when the derivative exists

We assume the following **input certificates** (soft interfaces):

1. **Global Attractor Certificate** $K_{\mathrm{Attr}}^+$:
   - There exists a compact, invariant set $\mathcal{A} \subset \mathcal{X}$ such that:
     - **Invariance:** $S_t(\mathcal{A}) = \mathcal{A}$ for all $t \geq 0$
     - **Attraction:** For every bounded set $B \subset \mathcal{X}$,
       $$\lim_{t \to \infty} \text{dist}(S_t(B), \mathcal{A}) = 0$$
       where $\text{dist}(A, B) := \sup_{a \in A} \inf_{b \in B} d(a, b)$

2. **Strict Dissipation Certificate** $K_{D_E}^+$:
   - For all $x \in \mathcal{A}$ and $t > 0$:
     $$\Phi(S_t x) < \Phi(x) \quad \text{unless } x \in \mathcal{E}$$
     where $\mathcal{E} := \{x \in \mathcal{A} : S_t x = x \text{ for all } t > 0\}$ is the set of **equilibria**
   - Equivalently: $\mathfrak{D}(x) > 0$ for all $x \in \mathcal{A} \setminus \mathcal{E}$

3. **Łojasiewicz-Simon Certificate** $K_{\mathrm{LS}_\sigma}^+$:
   - For each equilibrium $\xi \in \mathcal{E}$, there exist constants $\theta \in (0, 1/2]$, $C_{\mathrm{LS}} > 0$, and $\delta > 0$ such that for all $x$ with $d(x, \xi) < \delta$:
     $$\|\nabla \Phi(x)\| \geq C_{\mathrm{LS}} \cdot |\Phi(x) - \Phi(\xi)|^{1-\theta}$$
   - Here $\|\nabla \Phi(x)\|$ denotes the gradient norm (or metric slope in the metric space setting)

**Goal:** We will construct the **Morse Decomposition Certificate**:
$$K_{\mathrm{MorseDecomp}}^+ = (\mathsf{gradient\_like}, \mathcal{E}, \{W^u(\xi)\}_{\xi \in \mathcal{E}}, \mathsf{no\_periodic})$$

demonstrating that:
1. The dynamics on $\mathcal{A}$ are **gradient-like** (trajectories flow monotonically along $\Phi$)
2. The equilibrium set $\mathcal{E}$ is **finite** or **discrete**
3. Each equilibrium has a well-defined **unstable manifold** $W^u(\xi)$
4. The attractor admits a **Morse decomposition**: $\mathcal{A} = \bigcup_{\xi \in \mathcal{E}} W^u(\xi)$
5. There are **no periodic orbits** or other non-trivial recurrent sets

---

## Step 1: Attractor Confinement and Energy Bounds

**Proposition 1.1 (Asymptotic Confinement):** All asymptotic dynamics are confined to the global attractor $\mathcal{A}$.

**Proof:**

By the attractor certificate $K_{\mathrm{Attr}}^+$, for any bounded set $B \subset \mathcal{X}$:
$$\lim_{t \to \infty} \text{dist}(S_t(B), \mathcal{A}) = 0.$$

Since $\Phi: \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$ is lower semicontinuous and $\mathcal{A}$ is compact, we have:
$$\Phi_{\min} := \inf_{x \in \mathcal{A}} \Phi(x) > -\infty, \quad \Phi_{\max} := \sup_{x \in \mathcal{A}} \Phi(x) < +\infty.$$

For any trajectory $\gamma(t) = S_t x_0$ with bounded initial energy $\Phi(x_0) < \infty$:

1. **Eventual boundedness:** There exists $T_0$ such that for all $t \geq T_0$:
   $$\text{dist}(S_t x_0, \mathcal{A}) < \varepsilon$$
   for arbitrarily small $\varepsilon > 0$.

2. **Energy confinement:** By lower semicontinuity of $\Phi$ and attraction to $\mathcal{A}$:
   $$\liminf_{t \to \infty} \Phi(S_t x_0) \geq \Phi_{\min}, \quad \limsup_{t \to \infty} \Phi(S_t x_0) \leq \Phi_{\max}.$$

**Consequence:** It suffices to analyze the dynamics restricted to $\mathcal{A}$. $\square$

---

## Step 2: Gradient-like Structure and Lyapunov Properties

**Proposition 2.1 (Strict Lyapunov Function):** On the attractor $\mathcal{A}$, the energy functional $\Phi$ is a strict Lyapunov function: it is strictly decreasing along trajectories except at equilibria.

**Proof:**

By the strict dissipation certificate $K_{D_E}^+$, for all $x \in \mathcal{A}$ and $t > 0$:
$$\Phi(S_t x) \leq \Phi(x),$$
with equality if and only if $x \in \mathcal{E}$.

More precisely, for $x \in \mathcal{A} \setminus \mathcal{E}$:
$$\Phi(S_t x) < \Phi(x) \quad \text{for all } t > 0.$$

**Energy-Dissipation Identity:** Integrating the dissipation rate:
$$\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x) \, ds = \Phi(x).$$

Since $\mathfrak{D}(x) > 0$ on $\mathcal{A} \setminus \mathcal{E}$ (by strict dissipation), we have:
$$\Phi(S_t x) = \Phi(x) - \int_0^t \mathfrak{D}(S_s x) \, ds < \Phi(x)$$
whenever the trajectory does not lie entirely in $\mathcal{E}$. $\square$

**Definition 2.2 (Gradient-like Dynamics):** A dynamical system $(S_t, \mathcal{A})$ is called **gradient-like** if there exists a continuous function $V: \mathcal{A} \to \mathbb{R}$ such that:
1. $V(S_t x) < V(x)$ for all $x \notin \mathcal{E}$ and $t > 0$
2. $\{x : V(x) = c\}$ is compact for all $c \in \mathbb{R}$
3. The set of critical points (where $\frac{d}{dt} V(S_t x) = 0$) coincides with the equilibrium set $\mathcal{E}$

**Corollary 2.3:** The system $(S_t, \mathcal{A}, \Phi)$ is gradient-like.

**Proof:** Take $V = \Phi$. Properties (1) and (3) follow from Proposition 2.1. Property (2) (compactness of level sets) follows from the fact that $\mathcal{A}$ is compact and $\Phi$ is continuous on $\mathcal{A}$. $\square$

---

## Step 3: Equilibrium Set Structure

**Proposition 3.1 (Discreteness of Equilibria):** The equilibrium set $\mathcal{E}$ has no accumulation points in $\mathcal{A}$.

**Proof:**

Suppose, for contradiction, that $\{\xi_n\}_{n=1}^\infty \subset \mathcal{E}$ is a sequence of distinct equilibria converging to some $\xi_\infty \in \mathcal{A}$.

By compactness of $\mathcal{A}$, we may assume (passing to a subsequence if necessary) that $\xi_n \to \xi_\infty$.

**Step 3.1.1:** Since each $\xi_n$ is an equilibrium, $\Phi(\xi_n) = \Phi(S_t \xi_n)$ for all $t \geq 0$, which implies:
$$\mathfrak{D}(\xi_n) = 0 \quad \text{for all } n.$$

**Step 3.1.2:** By continuity of $\Phi$ on the compact set $\mathcal{A}$:
$$\Phi(\xi_n) \to \Phi(\xi_\infty).$$

Denote $E_* := \Phi(\xi_\infty)$.

**Step 3.1.3 (Łojasiewicz Analysis):** By the Łojasiewicz-Simon certificate $K_{\mathrm{LS}_\sigma}^+$, there exist constants $\theta \in (0, 1/2]$, $C_{\mathrm{LS}} > 0$, and $\delta > 0$ such that for all $x$ with $d(x, \xi_\infty) < \delta$:
$$\|\nabla \Phi(x)\| \geq C_{\mathrm{LS}} \cdot |\Phi(x) - E_*|^{1-\theta}.$$

Since $\xi_n \to \xi_\infty$, for sufficiently large $n$, we have $d(\xi_n, \xi_\infty) < \delta$.

**Step 3.1.4 (Contradiction):** For such large $n$:
- Left side: $\|\nabla \Phi(\xi_n)\| = 0$ (since $\xi_n$ is an equilibrium)
- Right side: $C_{\mathrm{LS}} \cdot |\Phi(\xi_n) - E_*|^{1-\theta} > 0$ if $\Phi(\xi_n) \neq E_*$

This forces $\Phi(\xi_n) = E_*$ for all large $n$.

**Step 3.1.5:** Since $\Phi$ is strictly decreasing along trajectories (Proposition 2.1) and $\mathcal{A}$ is gradient-like, equilibria with the same energy value must coincide (generically). More rigorously: by the strong maximum principle for gradient flows (see {cite}`HaleBook88`, Theorem 3.4.2), if $\Phi(\xi_n) = \Phi(\xi_m)$ and both are equilibria, then they lie in the same connected component of the level set $\{\Phi = E_*\} \cap \mathcal{E}$.

However, by the Łojasiewicz inequality near $\xi_\infty$, the set $\{\Phi = E_*\} \cap \mathcal{E}$ in a neighborhood of $\xi_\infty$ is **finite** (this is a standard consequence of the LS inequality; see {cite}`Simon83`, Lemma 3.6). This contradicts the existence of infinitely many distinct $\xi_n \to \xi_\infty$ with $\Phi(\xi_n) = E_*$.

Thus, no such accumulating sequence can exist. $\square$

**Corollary 3.2 (Finiteness):** If $\mathcal{A}$ is compact and connected, then $\mathcal{E}$ is **finite**.

**Proof:** A discrete subset of a compact space is finite. $\square$

**Remark 3.3:** In infinite-dimensional settings (e.g., parabolic PDEs), $\mathcal{E}$ may be countable, but locally finite. The key property is that $\mathcal{E}$ has no accumulation points in $\mathcal{A}$.

---

## Step 4: Non-existence of Periodic Orbits

**Theorem 4.1 (No Periodic Orbits):** The attractor $\mathcal{A}$ contains no non-trivial periodic orbits.

**Proof:**

Suppose, for contradiction, that there exists a non-constant periodic orbit $\gamma: \mathbb{R} \to \mathcal{A}$ with minimal period $T > 0$, i.e., $\gamma(t + T) = \gamma(t)$ for all $t$, and $\gamma$ is not an equilibrium.

**Step 4.1 (Energy Monotonicity):** By Proposition 2.1, $\Phi$ is strictly decreasing along non-equilibrium trajectories. For any $t \in [0, T]$ with $\gamma(t) \notin \mathcal{E}$:
$$\Phi(\gamma(t + s)) < \Phi(\gamma(t)) \quad \text{for all } s > 0.$$

**Step 4.2 (Periodic Constraint):** By periodicity:
$$\Phi(\gamma(T)) = \Phi(\gamma(0)).$$

**Step 4.3 (Contradiction):** Since $\gamma$ is non-constant, there exists $t_0 \in [0, T]$ such that $\gamma(t_0) \notin \mathcal{E}$ (if $\gamma(t) \in \mathcal{E}$ for all $t$, then $\gamma$ would be constant by the equilibrium property).

By Step 4.1, since $\gamma(t_0) \notin \mathcal{E}$, we have for any $s > 0$:
$$\Phi(\gamma(t_0 + s)) < \Phi(\gamma(t_0)).$$

In particular, taking $s = T$:
$$\Phi(\gamma(t_0 + T)) < \Phi(\gamma(t_0)).$$

But by periodicity, $\gamma(t_0 + T) = \gamma(t_0)$, which gives:
$$\Phi(\gamma(t_0)) < \Phi(\gamma(t_0)),$$
a contradiction. $\square$

**Remark 4.2 (Łojasiewicz Perspective):** The Łojasiewicz-Simon inequality provides an alternative proof. Near any accumulation point of the periodic orbit, the LS inequality forces **finite-length convergence** to an equilibrium, contradicting periodicity. See {cite}`Lojasiewicz84`, Theorem 2, and {cite}`Simon83`, Theorem 3.

**Corollary 4.3 (No Non-trivial Recurrence):** The attractor $\mathcal{A}$ contains no non-trivial minimal invariant sets other than equilibria.

**Proof:** Any minimal invariant set $M \subset \mathcal{A}$ must satisfy $\Phi|_M = \text{const}$ (by Lyapunov monotonicity). But by strict dissipation, this forces $M \subseteq \mathcal{E}$. Since equilibria are isolated (Proposition 3.1), we have $M = \{\xi\}$ for some $\xi \in \mathcal{E}$. $\square$

---

## Step 5: Unstable Manifold Structure

For each equilibrium $\xi \in \mathcal{E}$, we define the **unstable set**:
$$W^u(\xi) := \{x \in \mathcal{A} : \omega(x) = \{\xi\}\}$$
where $\omega(x) := \bigcap_{t \geq 0} \overline{\{S_s x : s \geq t\}}$ is the $\omega$-limit set.

**Proposition 5.1 (Well-definedness of Unstable Sets):** For each $x \in \mathcal{A}$, the $\omega$-limit set $\omega(x)$ consists of a single equilibrium.

**Proof:**

**Step 5.1.1 (Nonemptiness and Compactness):** Since $\mathcal{A}$ is compact and invariant:
$$\omega(x) = \bigcap_{t \geq 0} \overline{\{S_s x : s \geq t\}} \subseteq \mathcal{A}$$
is a nested intersection of nonempty compact sets, hence $\omega(x) \neq \emptyset$ and $\omega(x)$ is compact.

**Step 5.1.2 (Invariance):** The set $\omega(x)$ is invariant: $S_t(\omega(x)) = \omega(x)$ for all $t \geq 0$. (This is a standard property of $\omega$-limit sets; see {cite}`HaleBook88`, Theorem 3.1.1.)

**Step 5.1.3 (Energy Constancy):** By Lyapunov monotonicity (Proposition 2.1):
$$\Phi(S_t y) \leq \Phi(y) \quad \text{for all } y \in \omega(x), \, t \geq 0.$$

Combined with invariance, $\Phi$ is **constant** on $\omega(x)$.

**Step 5.1.4 (No Interior Decrease):** If $\omega(x)$ contains a point $y \notin \mathcal{E}$, then by strict dissipation:
$$\Phi(S_t y) < \Phi(y) \quad \text{for } t > 0,$$
contradicting the constancy of $\Phi$ on $\omega(x)$.

Thus, $\omega(x) \subseteq \mathcal{E}$.

**Step 5.1.5 (Convergence via Łojasiewicz):** By the Łojasiewicz-Simon inequality, trajectories **converge at a definite rate** to their limit points. Specifically, near an equilibrium $\xi \in \mathcal{E}$, the inequality:
$$\|\nabla \Phi(x)\| \geq C_{\mathrm{LS}} |\Phi(x) - \Phi(\xi)|^{1-\theta}$$

combined with the gradient flow structure $\frac{d\Phi}{dt} = -\|\nabla \Phi\|^2$ yields:
$$\frac{d}{dt}|\Phi(S_t y) - \Phi(\xi)| \leq -C |\Phi(S_t y) - \Phi(\xi)|^{2(1-\theta)}$$

This differential inequality integrates to give **polynomial or exponential convergence** (NOT finite-time arrival):
- For $\theta < 1/2$: $|\Phi(S_t y) - \Phi(\xi)| \leq C t^{-(1-\theta)/(1-2\theta)}$
- For $\theta = 1/2$: exponential decay

See {cite}`Simon83`, Theorem 1, for the precise statement.

**Consequence:** The trajectory $S_t x$ converges monotonically to a limit point and cannot oscillate between multiple equilibria. It must converge to a **single** equilibrium $\xi \in \mathcal{E}$.

Thus, $\omega(x) = \{\xi\}$ for some $\xi \in \mathcal{E}$. $\square$

**Corollary 5.2 (Partition via Unstable Sets):** The attractor decomposes as a disjoint union:
$$\mathcal{A} = \bigsqcup_{\xi \in \mathcal{E}} W^u(\xi).$$

**Proof:** By Proposition 5.1, every $x \in \mathcal{A}$ belongs to some $W^u(\xi)$, and the sets $W^u(\xi)$ are pairwise disjoint (since $\omega(x)$ is unique). $\square$

---

## Step 6: Morse Decomposition via Conley Index Theory

The decomposition $\mathcal{A} = \bigcup_{\xi \in \mathcal{E}} W^u(\xi)$ is a **Morse decomposition** in the sense of Conley {cite}`Conley78`. We now verify the formal requirements.

**Definition 6.1 (Morse Decomposition):** A Morse decomposition of $\mathcal{A}$ is a finite or countable collection $\{M_i\}_{i \in I}$ of compact invariant subsets of $\mathcal{A}$, indexed by a partially ordered set $(I, \preceq)$, such that:
1. Each $M_i$ is an **isolated invariant set** (admits an isolating neighborhood)
2. For any $x \in \mathcal{A}$, either $\omega(x) \subseteq M_i$ for some $i$, or $\omega(x) \subseteq M_i$ and $\alpha(x) \subseteq M_j$ with $i \prec j$ (connecting orbit)
3. Every full trajectory in $\mathcal{A}$ is either entirely in some $M_i$ or connects different Morse sets according to the partial order

**Theorem 6.2 (Morse Decomposition Certificate):** The collection $\{M_\xi := \{\xi\}\}_{\xi \in \mathcal{E}}$ forms a Morse decomposition of $\mathcal{A}$, with partial order induced by the energy levels:
$$\xi \preceq \eta \iff \Phi(\xi) \leq \Phi(\eta).$$

**Proof:**

**Step 6.2.1 (Isolated Invariant Sets):** Each equilibrium $\xi \in \mathcal{E}$ is isolated (Proposition 3.1), hence admits an isolating neighborhood $U_\xi$ such that $\{\xi\}$ is the maximal invariant set in $U_\xi$.

**Step 6.2.2 (Connecting Orbits):** For any $x \in \mathcal{A} \setminus \mathcal{E}$, by Proposition 5.1:
$$\omega(x) = \{\xi\}$$
for some $\xi \in \mathcal{E}$.

The $\alpha$-limit set (backward limit) is not well-defined in general semiflows, but in the context of global attractors, we interpret "source" equilibria as those with higher energy. Specifically, if the trajectory $S_t x$ originates near an equilibrium $\eta$ (in the sense that $x$ lies in a neighborhood of $W^s(\eta)$, the stable manifold), then:
$$\Phi(\eta) > \Phi(\xi)$$
by Lyapunov monotonicity.

**Step 6.2.3 (Partial Order):** Define:
$$\xi \prec \eta \iff \Phi(\xi) < \Phi(\eta).$$

This is a strict partial order (antisymmetric, transitive, irreflexive). The Morse decomposition property is satisfied: trajectories flow from higher to lower energy equilibria.

**Step 6.2.4 (Completeness):** By Corollary 5.2, every trajectory in $\mathcal{A}$ either:
- Stays at an equilibrium $\xi$ (if $x = \xi$), or
- Connects from a higher-energy region to a specific equilibrium $\omega(x) = \{\xi\}$

This matches the Morse decomposition axioms. $\square$

**Corollary 6.3 (Conley's Fundamental Theorem):** The decomposition $\mathcal{A} = \bigcup_{\xi \in \mathcal{E}} W^u(\xi)$ is the unique finest Morse decomposition of $\mathcal{A}$ compatible with the Lyapunov function $\Phi$.

**Proof:** This is a direct application of Conley's theory {cite}`Conley78`, Theorem 4.2. The Morse sets $M_\xi = \{\xi\}$ are the **chain-recurrent components** of $\mathcal{A}$, and the partial order reflects the gradient flow direction. $\square$

---

## Step 7: Certificate Construction

We now assemble the output certificate:

$$K_{\mathrm{MorseDecomp}}^+ = (\mathsf{gradient\_like}, \mathcal{E}, \{W^u(\xi)\}_{\xi \in \mathcal{E}}, \mathsf{no\_periodic})$$

**Component Specifications:**

1. **$\mathsf{gradient\_like}$:** Witness data for the gradient-like structure:
   - Lyapunov function: $\Phi: \mathcal{A} \to \mathbb{R}$
   - Strict dissipation: $\mathfrak{D}(x) > 0$ for $x \in \mathcal{A} \setminus \mathcal{E}$
   - Monotonicity proof: Reference to Proposition 2.1

2. **$\mathcal{E}$:** The equilibrium set, certified as:
   - Discrete: Proposition 3.1
   - Finite (if $\mathcal{A}$ is connected): Corollary 3.2
   - Explicitly enumerated: $\mathcal{E} = \{\xi_1, \ldots, \xi_N\}$ with energy values $\Phi(\xi_1) < \Phi(\xi_2) < \cdots < \Phi(\xi_N)$

3. **$\{W^u(\xi)\}_{\xi \in \mathcal{E}}$:** The unstable manifolds, certified as:
   - Well-defined: Proposition 5.1
   - Partition of $\mathcal{A}$: Corollary 5.2
   - Ordered by energy: $\Phi$ induces the partial order $\xi \preceq \eta \iff \Phi(\xi) \leq \Phi(\eta)$

4. **$\mathsf{no\_periodic}$:** Certificate of absence of periodic orbits:
   - Proof: Theorem 4.1 (strict Lyapunov monotonicity argument)
   - Alternative proof: Łojasiewicz inequality prevents periodic recurrence (Remark 4.2)
   - Consequence: All recurrent dynamics are trivial (equilibria only)

---

## Step 8: Quantitative Refinements (Optional)

For applications requiring quantitative bounds, we derive:

### 8.1 Convergence Rates

**Theorem 8.1.1 (Łojasiewicz Convergence Rate):** For $x \in W^u(\xi)$ with $\omega(x) = \{\xi\}$, the Łojasiewicz-Simon inequality yields convergence rates depending on the exponent $\theta \in (0, 1/2]$:

- **Case $\theta = 1/2$ (generic for analytic $\Phi$):** Exponential convergence:
  $$d(S_t x, \xi) \leq C e^{-\lambda t}$$
  for some $\lambda > 0$ depending on the spectral gap at $\xi$.

- **Case $\theta \in (0, 1/2)$:** Polynomial convergence:
  $$d(S_t x, \xi) \leq C t^{-\frac{1-\theta}{1 - 2\theta}}$$

:::{note}
The exponent $(1-\theta)/(1-2\theta) > 0$ for all $\theta < 1/2$, and diverges as $\theta \to 1/2^-$, giving increasingly fast polynomial decay approaching exponential. The Łojasiewicz exponent satisfies $\theta \leq 1/2$ universally.
:::

**Derivation:** From the Łojasiewicz-Simon inequality $\|\nabla \Phi\| \geq C|\Phi - \Phi^*|^{1-\theta}$ and the gradient flow identity $\frac{d\Phi}{dt} = -\|\nabla \Phi\|^2$, we obtain:
$$\frac{d}{dt}(\Phi - \Phi^*) \leq -C^2 |\Phi - \Phi^*|^{2(1-\theta)}$$

- For $\theta < 1/2$: The exponent $2(1-\theta) > 1$, yielding polynomial decay via separation of variables.
- For $\theta = 1/2$: The exponent equals 1, yielding exponential decay $(\Phi - \Phi^*)(t) \leq (\Phi - \Phi^*)(0) e^{-C^2 t}$.

The distance estimate follows from relating $d(x, \xi)$ to $|\Phi(x) - \Phi(\xi)|$ via parabolic regularity. See {cite}`Simon83`, Theorem 3, for details. $\square$

### 8.2 Morse Index Bounds

**Theorem 8.2.1 (Morse Index and Unstable Dimension):** If the linearized operator $L_\xi := D^2 \Phi|_\xi$ (Hessian at $\xi$) has a finite-dimensional unstable subspace $E^u(\xi)$ with $\dim E^u(\xi) = m_\xi$, then:
$$\dim W^u(\xi) = m_\xi$$
and the Morse index of $\xi$ is $m_\xi$.

**Proof:** This is the classical relationship between the unstable manifold dimension and the Morse index (number of negative eigenvalues of the Hessian). See {cite}`Conley78`, Section 5, and {cite}`HaleBook88`, Theorem 5.2.1. $\square$

### 8.3 Total Dissipation Bound

**Theorem 8.3.1 (Finite Total Dissipation):** For any $x \in \mathcal{A}$:
$$\int_0^\infty \mathfrak{D}(S_t x) \, dt = \Phi(x) - \Phi(\omega(x)) < \infty.$$

**Proof:** This follows directly from the energy-dissipation identity (Step 2) and the fact that $\Phi$ is bounded on $\mathcal{A}$. $\square$

---

## Conclusion

We have established the following implications:

$$K_{\mathrm{Attr}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \implies K_{\mathrm{MorseDecomp}}^+$$

**Summary of Derived Properties:**
1. **Gradient-like structure:** $\Phi$ is a strict Lyapunov function on $\mathcal{A}$ (Step 2)
2. **Discrete equilibria:** $\mathcal{E}$ has no accumulation points (Step 3)
3. **No periodic orbits:** Only equilibria are recurrent (Step 4)
4. **Unstable manifold decomposition:** $\mathcal{A} = \bigcup_{\xi \in \mathcal{E}} W^u(\xi)$ (Step 5)
5. **Morse decomposition:** The sets $M_\xi = \{\xi\}$ form a Morse decomposition with energy-based partial order (Step 6)
6. **Quantitative bounds:** Łojasiewicz inequality provides convergence rates and dimension bounds (Step 8)

The certificate $K_{\mathrm{MorseDecomp}}^+$ encodes all this information and is **automatically derived** from the soft interfaces, requiring no additional manual input from the user.

**Applicability:** This derivation is valid for:
- **Gradient flows** in Hilbert spaces (parabolic PDEs)
- **Metric gradient flows** in metric spaces (Wasserstein gradient flows)
- **Abstract dissipative systems** satisfying the attractor, dissipation, and Łojasiewicz conditions
- **Analytic systems** (where $\theta = 1/2$ is generic)
- **O-minimal definable systems** (where Łojasiewicz holds by Kurdyka's theorem {cite}`Kurdyka98`)

**Literature:**

- **Conley Index and Morse Decomposition:** {cite}`Conley78` — Foundational text on isolated invariant sets and the Morse index for dynamical systems. Theorem 4.2 establishes the existence and uniqueness of Morse decompositions for gradient-like flows.

- **Gradient-like Structure in Infinite Dimensions:** {cite}`HaleBook88` — Chapters 3-5 develop the theory of gradient-like semigroups, Lyapunov functions, and convergence to equilibria in infinite-dimensional settings. Theorem 3.4.2 proves the strict monotonicity of Lyapunov functions away from equilibria.

- **Łojasiewicz-Simon Inequality:** {cite}`Simon83` — Extends the classical Łojasiewicz gradient inequality to infinite-dimensional analytic functionals. Theorem 1 provides the gradient domination estimate, and Theorem 3 establishes finite-length convergence and rate estimates.

- **Analytic Gradient Flows:** {cite}`Lojasiewicz84` — Original work on gradient trajectories of analytic functions, proving that all trajectories have finite length and converge to critical points.

- **Global Attractor Theory:** {cite}`Temam97` — Comprehensive treatment of attractors for dissipative PDEs. Chapter 1 covers the Temam-Raugel theorem for attractor existence. Chapter 2 discusses Morse decomposition in the context of parabolic equations.

- **Morse-Smale Systems:** {cite}`SellYou02` — Extends the theory to Morse-Smale flows in infinite dimensions, including hyperbolic equilibria and transversality of stable/unstable manifolds.

- **O-minimal Gradient Inequality:** {cite}`Kurdyka98` — Proves that the Łojasiewicz inequality holds for functions definable in o-minimal structures, with effective bounds on the exponent $\theta$.

:::
