# FACT-SoftMorse: Soft→MorseDecomp Compilation — GMT Translation

## Original Statement (Hypostructure)

The soft permits compile to a Morse decomposition: the flow decomposes into gradient-like dynamics between isolated invariant sets.

## GMT Setting

**Flow:** $\varphi_t: \mathbf{I}_k(M) \to \mathbf{I}_k(M)$ — gradient flow of $\Phi$

**Morse Decomposition:** $\mathcal{M} = \{M_1, \ldots, M_m\}$ — finite collection of disjoint compact invariant sets

**Morse Ordering:** $M_i < M_j$ if $\exists$ connecting orbit from $M_j$ to $M_i$

## GMT Statement

**Theorem (Soft→MorseDecomp Compilation).** Under soft permits, the gradient flow admits a Morse decomposition:

1. **Finite Collection:** $\mathcal{M} = \{M_1, \ldots, M_m\}$ with $m \leq N(\Lambda, n, \theta)$

2. **Invariance:** Each $M_i$ is compact and invariant: $\varphi_t(M_i) = M_i$

3. **Gradient-Like:** The flow is gradient-like between Morse sets:
$$T \notin \bigcup_i M_i \implies \Phi(\varphi_t(T)) < \Phi(T) \text{ for } t > 0$$

4. **Conley Index:** Each $M_i$ has well-defined Conley index $h(M_i)$

## Proof Sketch

### Step 1: Conley's Fundamental Theorem

**Conley's Theorem (1978):** Every flow on a compact metric space admits a finest Morse decomposition.

**Reference:** Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. CBMS Regional Conference Series 38, AMS.

**Application:** The compact attractor $\mathcal{A}$ (from FACT-SoftAttr) admits Morse decomposition.

### Step 2: Morse Sets as Critical Levels

**For Gradient Flows:** The Morse sets are:
$$M_c := \{T \in \mathcal{A} : \Phi(T) = c, \, \nabla \Phi(T) = 0\}$$

for critical values $c$.

**Łojasiewicz Implies Isolation:** By $K_{\text{LS}_\sigma}^+$, critical points are isolated, so Morse sets are discrete unions of equilibria.

**Reference:** Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations. *Ann. of Math.*, 118, 525-571.

### Step 3: Finiteness of Morse Sets

**Use of $K_{C_\mu}^+$:** Compactness + isolation implies finitely many equilibria.

**Energy Bound:** Each equilibrium $T_*$ has $\Phi(T_*) \leq \Lambda$.

**Counting (Simon, 1983):** The number of critical points is bounded:
$$|\text{Crit}(\Phi) \cap \mathcal{A}| \leq N(\Lambda, n, \theta)$$

where $\theta$ is the Łojasiewicz exponent.

### Step 4: Connecting Orbits

**Definition:** A connecting orbit from $M_j$ to $M_i$ is a complete trajectory $\gamma: \mathbb{R} \to \mathbf{I}_k(M)$ with:
$$\alpha(\gamma) \subset M_j, \quad \omega(\gamma) \subset M_i$$

where $\alpha, \omega$ are alpha/omega limit sets.

**Energy Ordering:** If $\gamma$ connects $M_j$ to $M_i$:
$$\Phi(M_j) > \Phi(M_i)$$

by strict dissipation between equilibria.

**Partial Order:** The relation $M_i < M_j$ is a partial order (no cycles by energy monotonicity).

### Step 5: Conley Index Construction

**Index Pairs (Conley-Zehnder, 1984):** For isolated invariant set $S$, an index pair $(N, L)$ satisfies:
1. $N, L$ are compact, $L \subset N$
2. $S \subset \text{int}(N \setminus L)$
3. $L$ is an exit set for $N$

**Reference:** Conley, C., Zehnder, E. (1984). Morse-type index theory for flows and periodic solutions of Hamiltonian equations. *Comm. Pure Appl. Math.*, 37, 207-253.

**Conley Index:** $h(S) := [N/L]$ (pointed homotopy type)

**Index Invariance:** $h(S)$ is independent of choice of index pair.

### Step 6: Computation of Indices

**For Hyperbolic Equilibria:** If $T_*$ is hyperbolic with unstable dimension $k$:
$$h(T_*) \simeq S^k$$

**Morse Inequality (Floer, 1989):**
$$\sum_i (-1)^{\dim(M_i)} \chi(h(M_i)) = \chi(\mathcal{A})$$

**Reference:** Floer, A. (1989). Witten's complex and infinite-dimensional Morse theory. *J. Diff. Geom.*, 30, 207-221.

### Step 7: Gradient-Like Structure

**Definition:** A flow is **gradient-like** if:
1. Fixed points are isolated
2. Between fixed points, a Lyapunov function strictly decreases

**Theorem:** Under soft permits, $\varphi_t$ is gradient-like with Lyapunov function $\Phi$.

*Proof:* By $K_{D_E}^+$: $\frac{d}{dt}\Phi(\varphi_t(T)) = -|\nabla \Phi|^2 < 0$ unless $\nabla \Phi = 0$.

### Step 8: Stable and Unstable Manifolds

**Stable Manifold (Palis-de Melo, 1982):**
$$W^s(M_i) := \{T : \omega(T) \subset M_i\}$$

**Unstable Manifold:**
$$W^u(M_i) := \{T : \alpha(T) \subset M_i\}$$

**Reference:** Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.

**Decomposition:** The attractor is:
$$\mathcal{A} = \bigcup_i M_i \cup \bigcup_{i<j} W^u(M_j) \cap W^s(M_i)$$

### Step 9: Compilation Theorem

**Theorem (Soft→MorseDecomp):** The compilation:
$$(K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+) \to \text{MorseDecomp}$$

produces:
- Finite Morse decomposition $\mathcal{M} = \{M_1, \ldots, M_m\}$
- Partial order via connecting orbits
- Conley indices $h(M_i)$ for each Morse set
- Stable/unstable manifold structure

**Constructive Content:**
1. Algorithm to find equilibria
2. Algorithm to detect connecting orbits
3. Algorithm to compute Conley indices (via discretization)

**Reference:** Mischaikow, K., Mrozek, M. (2002). Conley index. *Handbook of Dynamical Systems*, Vol. 2, 393-460.

### Step 10: GMT Examples

**Example 1: Geodesic Flow on Energy Surface**
- Morse sets: closed geodesics
- Connecting orbits: heteroclinic connections
- Conley index: related to Morse index of geodesic

**Example 2: Mean Curvature Flow**
- Morse sets: self-similar solutions (spheres, cylinders)
- Connecting orbits: neck-pinch trajectories
- Applications: surgery description

**Example 3: Ricci Flow**
- Morse sets: Einstein metrics
- Connecting orbits: Ricci flow trajectories
- Hamilton-Perelman surgery as Morse theory

## Key GMT Inequalities Used

1. **Conley Decomposition:**
   $$\mathcal{A} = \bigcup_i M_i \cup \text{connections}$$

2. **Morse Inequality:**
   $$\sum_i (-1)^{\dim M_i} \chi(h(M_i)) = \chi(\mathcal{A})$$

3. **Energy Ordering:**
   $$M_i < M_j \implies \Phi(M_j) > \Phi(M_i)$$

4. **Finiteness:**
   $$|\mathcal{M}| \leq N(\Lambda, n, \theta)$$

## Literature References

- Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. AMS.
- Conley, C., Zehnder, E. (1984). Morse-type index theory. *Comm. Pure Appl. Math.*, 37.
- Floer, A. (1989). Witten's complex. *J. Diff. Geom.*, 30.
- Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
- Mischaikow, K., Mrozek, M. (2002). Conley index. *Handbook of Dynamical Systems*, Vol. 2.
- Salamon, D. (1990). Morse theory, the Conley index and Floer homology. *Bull. LMS*, 22, 113-140.
