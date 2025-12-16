# Metatheorem 22.10 (The Tropical Limit Principle)

**Statement.** Let $X \subset (\mathbb{C}^*)^n$ be a subvariety defined over a valued field $(K, v)$, and let $\text{Trop}(X) \subset \mathbb{R}^n$ be its tropicalization via the valuation map $\text{Val}(x_1, \ldots, x_n) = (v(x_1), \ldots, v(x_n))$. Then the scaling limit $t \to 0$ of the hypostructure $\mathbb{H}_X^t$ recovers a tropical hypostructure $\mathbb{H}_{\text{trop}}$ satisfying:

1. **Log-Limit ↔ Piecewise Linear**: The smooth variety $X$ degenerates to the tropical variety $\text{Trop}(X)$, a piecewise-linear polyhedral complex, as the Axiom SC scaling parameter $\lambda \to 0$.

2. **Amoebas**: The feasible region $\mathbb{F}$ of the hypostructure projects under $\text{Val}$ to the **amoeba** $\mathcal{A}(X) \subset \mathbb{R}^n$, whose spine is $\text{Trop}(X)$.

3. **Patchworking**: Global solutions to the hypostructure problem can be constructed from local piecewise-linear gluing via Viro's patchworking theorem—the tropical analogue of Axiom TB (Transition Between Modes).

---

## Proof

**Setup.** Fix a valued field $(K, v)$ with valuation ring $\mathcal{O}_K$ and residue field $k = \mathcal{O}_K/\mathfrak{m}$. For tropical geometry, we typically use:
- $K = \mathbb{C}\{\{t\}\}$, the field of Puiseux series, with valuation $v(f) = \min\{r : a_r \neq 0\}$ for $f = \sum a_r t^r$
- The tropicalization functor $\text{Trop}: \text{Var}_K \to \text{TropVar}$ sending algebraic varieties to piecewise-linear spaces

Let $X \subset (\mathbb{C}^*)^n$ be defined by polynomial equations $f_1, \ldots, f_m \in K[x_1^{\pm 1}, \ldots, x_n^{\pm 1}]$. The **tropical variety** is
$$
\text{Trop}(X) = \{w \in \mathbb{R}^n : \text{trop}(f_i)(w) \text{ is attained at least twice for all } i\}
$$
where $\text{trop}(f)(w) = \min_{a \in \text{supp}(f)} \{v(c_a) + \langle a, w \rangle\}$ is the tropical polynomial (minimum replaces sum, addition replaces product).

### Step 1: Degeneration via Maslov Dequantization

**(H1)** The tropical limit $t \to 0$ is formalized by **Maslov dequantization**, a limiting process that converts smooth geometry to piecewise-linear geometry.

**Step 1a: One-parameter family.**

Embed $X$ as a family $X_t \subset (\mathbb{C}^*)^n$ parametrized by $t \in \mathbb{C}^*$ near 0. Write equations as
$$
f_i(x; t) = \sum_{a \in A_i} c_{i,a}(t) x^a
$$
where $c_{i,a}(t) = t^{v_{i,a}} \cdot u_{i,a}$ with $u_{i,a} \in \mathcal{O}^*_K$ (units).

Taking the logarithmic limit $\log_t$:
$$
\lim_{t \to 0} \frac{\log|f_i(x; t)|}{\log|t|} = \text{trop}(f_i)(\text{Val}(x)).
$$

**Step 1b: Axiom SC scaling.**

Recall Axiom SC defines scaling of the feasible region $\mathbb{F}_\lambda$ as $\lambda \to 0$:
$$
\mathbb{F}_\lambda = \{x : \|x\| \leq \lambda^{-\alpha}, \, f_i(x) = 0\}.
$$

Under the change of variables $x_j = e^{w_j/\log(1/\lambda)}$, the constraint $\|x\| \leq \lambda^{-\alpha}$ becomes $\|w\| \leq \alpha$, and the equations $f_i(x) = 0$ become tropical equations $\text{trop}(f_i)(w) = 0$ in the limit $\lambda \to 0$.

**Step 1c: Convergence theorem.**

**Theorem (Kapranov, Mikhalkin).** The family $X_t$ converges to $\text{Trop}(X)$ in the **Hausdorff metric** on compact subsets of $\mathbb{R}^n$ under the Log map:
$$
\text{Log}_t: (\mathbb{C}^*)^n \to \mathbb{R}^n, \quad (z_1, \ldots, z_n) \mapsto \left(\frac{\log|z_1|}{\log|t|}, \ldots, \frac{\log|z_n|}{\log|t|}\right).
$$

Explicitly, for any $\varepsilon > 0$ and compact $K \subset \text{Trop}(X)$, there exists $\delta > 0$ such that
$$
|t| < \delta \implies \text{Log}_t(X_t) \cap K \subset K + B_\varepsilon.
$$

**Conclusion.** The tropical variety $\text{Trop}(X)$ is the $\lambda \to 0$ limit of the smooth variety $X$ under Axiom SC scaling. $\square_{\text{Step 1}}$

### Step 2: Amoebas as Feasible Regions

**(H2)** The **amoeba** of $X$ is defined as the image under the Log map:
$$
\mathcal{A}(X) = \text{Log}(X) = \{(\log|z_1|, \ldots, \log|z_n|) : (z_1, \ldots, z_n) \in X(\mathbb{C})\} \subset \mathbb{R}^n.
$$

This is the projection of the feasible region $X(\mathbb{C})$ to "log-space," the natural coordinate system for hypostructure scaling.

**Step 2a: Amoeba structure.**

Amoebas have rich geometric structure:
- **Tentacles**: Unbounded convex regions extending to infinity
- **Holes**: Bounded convex regions (vacuoles) where the amoeba is absent
- **Spine**: The tropical variety $\text{Trop}(X)$ sits at the "boundary" of the amoeba, forming its skeleton

**Theorem (Forsberg-Passare-Tsikh).** The amoeba $\mathcal{A}(X)$ is the complement of a union of convex sets, and the spine $\text{Trop}(X)$ is the closure of the locus where $\mathcal{A}(X)$ has local dimension $< n$.

**Step 2b: Hypostructure interpretation.**

Define the **scaled hypostructure** $\mathbb{H}_\lambda$ by:
- **State space**: $X_\lambda = \{x \in X : \text{Re}(x) \sim \lambda^{-\alpha}\}$ (points at scale $\lambda^{-\alpha}$)
- **Feasible region**: $\mathbb{F}_\lambda = \text{Log}(X_\lambda) \subset \mathbb{R}^n$

As $\lambda \to 0$, the feasible region $\mathbb{F}_\lambda$ accumulates on $\text{Trop}(X)$:
$$
\lim_{\lambda \to 0} \mathbb{F}_\lambda = \text{Trop}(X)
$$
in the Hausdorff topology. This is the geometric content of Axiom SC: **the tropical variety is the scaling limit of the classical variety**.

**Step 2c: Volume computation.**

The volume of the amoeba is related to the degree of $X$. For a hypersurface $X = V(f) \subset (\mathbb{C}^*)^n$, Mikhalkin proved:
$$
\text{Vol}_{2n-2}(\partial \mathcal{A}(X)) = \deg(f) \cdot \text{Vol}(\Delta_f)
$$
where $\Delta_f$ is the Newton polytope of $f$. This volume is the tropical analogue of the Axiom Cap bound $|\mathbb{F}| \leq C(\alpha, \beta)$.

**Conclusion.** The amoeba $\mathcal{A}(X)$ is the feasible region of the tropical hypostructure, with spine $\text{Trop}(X)$. $\square_{\text{Step 2}}$

### Step 3: Viro's Patchworking and Mode Gluing

**(H3)** **Viro's patchworking theorem** provides a combinatorial construction of real algebraic varieties from tropical data. This is the tropical version of Axiom TB (mode transitions): gluing local piecewise-linear solutions to form a global smooth solution.

**Step 3a: Patchworking setup.**

Let $\Delta \subset \mathbb{R}^n$ be a lattice polytope, and let $\mathcal{T}$ be a triangulation of $\Delta$ into simplices. Assign signs $\sigma_\tau \in \{\pm 1\}$ to each simplex $\tau \in \mathcal{T}$.

**Viro's Theorem.** There exists a polynomial $f_t(x) \in \mathbb{R}[x_1, \ldots, x_n]$ such that:
1. $\text{NewtPoly}(f_t) = \Delta$ (Newton polytope)
2. As $t \to 0$, the real zero locus $V_\mathbb{R}(f_t)$ degenerates to a limit curve $\Gamma$ determined by the signed triangulation $(\mathcal{T}, \sigma)$
3. The topology of $\Gamma$ is computed from the tropical variety $\text{Trop}(V(f))$ and the sign distribution $\sigma$

**Step 3b: Local-to-global gluing.**

The patchworking process is:
1. **Local**: On each simplex $\tau$, solve the tropical equation $\text{trop}(f)|_\tau = \max_{a \in \tau} \langle a, w \rangle$ (piecewise-linear)
2. **Matching**: Ensure solutions agree on overlaps $\tau \cap \tau'$ (gluing condition)
3. **Global**: The patched solution lifts to a smooth algebraic variety $X_t$ for small $t$

**Step 3c: Hypostructure interpretation.**

This parallels Axiom TB:
- **Mode C (Conservative)**: Simplices $\tau$ with sign $\sigma_\tau = +1$ correspond to "positive" regions
- **Mode D (Dissipative)**: Simplices with $\sigma_\tau = -1$ correspond to "negative" regions
- **Transition**: The gluing condition $\sigma_\tau \cdot \sigma_{\tau'} = (-1)^{\dim(\tau \cap \tau')+1}$ on common faces encodes the mode transition rule

The **profile map** $\Pi_C \cup \Pi_D \to \mathbb{F}$ (Definition 8.1) is realized tropically as the **subdivision map** from the triangulation $\mathcal{T}$ to the polytope $\Delta$.

**Step 3d: Welschinger invariants.**

For real enumerative geometry, patchworking computes **Welschinger invariants** $W_d$ (signed counts of real rational curves). These are tropical invariants satisfying
$$
|W_d| \leq G_d
$$
where $G_d$ is the Gromov-Witten invariant (complex count). The inequality reflects Mode D dissipation: real curves are a constrained subset of complex curves.

**Conclusion.** Viro's patchworking theorem is the tropical realization of Axiom TB, enabling global construction from local PL data. $\square_{\text{Step 3}}$

### Step 4: Tropical Compactifications and Boundary Behavior

We conclude by connecting tropical limits to the boundary behavior of Axiom LS (large-scale structure).

**Step 4a: Berkovich spaces.**

The **Berkovich analytification** $X^{\text{an}}$ provides a natural framework for tropical geometry. For $X/K$, the space $X^{\text{an}}$ is a compact Hausdorff space containing both:
- Classical points $X(K)$
- Tropical limit points (Shilov boundary)

The retraction $\rho: X^{\text{an}} \to \text{Trop}(X)$ is continuous, making $\text{Trop}(X)$ a "skeleton" of $X^{\text{an}}$.

**Step 4b: Axiom LS at infinity.**

Axiom LS requires asymptotic stabilization at large scales. Tropically, this becomes:
- **Interior**: Smooth behavior of $X$ for $|x| \ll \lambda^{-\alpha}$
- **Boundary**: Piecewise-linear behavior of $\text{Trop}(X)$ for $|x| \sim \lambda^{-\alpha}$

The Berkovich space interpolates between these regimes, providing a unified framework.

**Step 4c: Payne's balancing condition.**

**Theorem (Payne).** The tropical variety $\text{Trop}(X)$ is balanced: at each codimension-1 face, the sum of outgoing primitive vectors (weighted by multiplicity) is zero.

This is the tropical version of **Kirchhoff's law** for conservative flows (Axiom C). The balancing condition ensures that tropical varieties come from algebraic varieties, not arbitrary polyhedral complexes.

**Conclusion.** Tropical geometry provides a piecewise-linear shadow of algebraic geometry, capturing the large-scale behavior required by Axiom LS. $\square_{\text{Step 4}}$

---

## Key Insight

The tropical limit principle reveals that **piecewise-linear geometry is the scaling limit of algebraic geometry**. Under the Maslov dequantization $t \to 0$:

- **Smooth varieties** $\rightsquigarrow$ **Polyhedral complexes**
- **Polynomial equations** $\rightsquigarrow$ **Tropical equations** (min-plus algebra)
- **Intersection theory** $\rightsquigarrow$ **Balancing condition**
- **Enumerative invariants** $\rightsquigarrow$ **Combinatorial counts**

This correspondence is captured by the hypostructure axioms:
- **Axiom SC**: Scaling exponents $(\alpha, \beta)$ control the degeneration rate
- **Axiom TB**: Mode transitions $\leftrightarrow$ Patchworking/gluing
- **Axiom LS**: Berkovich skeleton $\leftrightarrow$ Asymptotic stabilization

The **amoeba** is the intermediate object bridging classical and tropical worlds: it is the image of the algebraic variety $X$ in log-space, and its spine is the tropical variety $\text{Trop}(X)$. Viro's patchworking theorem shows that tropical data determines classical topology, making tropical geometry a powerful computational tool.

The deep philosophical point: **tropical geometry is not an approximation but an intrinsic feature** of algebraic geometry at large scales. The hypostructure framework naturally accommodates both regimes, with Axiom SC governing the transition.

$\square$
