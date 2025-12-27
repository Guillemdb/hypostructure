# RESOLVE-Expansion: Thin-to-Full Expansion — GMT Translation

## Original Statement (Hypostructure)

Given thin objects (state space, energy, dissipation, symmetry), the framework automatically constructs full hypostructure data including topological structure, singularity detection, and profile classification.

## GMT Setting

**Thin Objects:**
- $(X, d)$ — complete metric space
- $\Phi: X \to [0, \infty]$ — energy functional
- $\mathfrak{D}: X \to [0, \infty]$ — dissipation rate
- $G$ — isometry group

**Full Structure:**
- $\mathbf{I}_k(X)$ — integral currents on $X$
- $\mathcal{P}(X)$ — profile space (tangent measures)
- $\mathcal{S}(X)$ — surgery operators
- $\text{sing}(X)$ — singular stratification

## GMT Statement

**Theorem (Automatic Structure Expansion).** Let $(X, d, \mu)$ be a complete metric measure space with:

1. **(Thin Energy)** $\Phi: X \to [0, \infty]$ proper lower semicontinuous

2. **(Thin Dissipation)** $|\partial \Phi|: X \to [0, \infty]$ the metric slope

3. **(Thin Symmetry)** $G \subset \text{Isom}(X, d)$ closed subgroup

Then there exist canonical constructions:

**(A) Topological Structure:**
$$\pi_0(X) = \{\text{path components}\}, \quad H_k(X; \mathbb{Z}) \text{ via singular homology}$$

**(B) Singularity Detection:**
$$\text{sing}(\Phi) := \{x : |\partial \Phi|(x) = \infty \text{ or } x \in \text{acc}(\{|\partial \Phi| \geq n\})\}$$

**(C) Profile Classification:**
$$\mathcal{P} := \{\text{Tan}(\mu, x) : x \in X\} / G \quad \text{(tangent measures mod symmetry)}$$

**(D) Surgery Construction:**
$$\mathcal{O}_S: X \dashrightarrow X' \quad \text{via metric gluing along isometric boundaries}$$

## Proof Sketch

### Step 1: Topological Structure via Čech Theory

**Path Components:** For metric spaces:
$$\pi_0(X) = X / \sim$$
where $x \sim y$ iff there exists a continuous path $\gamma: [0, 1] \to X$ with $\gamma(0) = x$, $\gamma(1) = y$.

**Čech Homology:** The Čech homology $\check{H}_k(X; \mathbb{Z})$ is well-defined for all compact metric spaces and agrees with singular homology for ANRs.

**Reference:** Eilenberg, S., Steenrod, N. (1952). *Foundations of Algebraic Topology*. Princeton University Press.

### Step 2: Singular Set Construction

**Metric Singular Set:** Define layers:
$$S_n := \{x \in X : |\partial \Phi|(x) \geq n\}$$

The singular set is:
$$\text{sing}(\Phi) := \bigcap_{n \geq 1} \overline{S_n} = \{x : \limsup_{y \to x} |\partial \Phi|(y) = \infty\}$$

**Properties (Ambrosio-Tilli, 2004):**
1. $\text{sing}(\Phi)$ is closed
2. $\text{sing}(\Phi) \subset \{x : \Phi(x) = \sup \Phi\}$ if $\Phi$ is bounded
3. $\text{sing}(\Phi)$ has measure zero for $\lambda$-convex $\Phi$ (Alexandrov theorem)

**Reference:** Ambrosio, L., Tilli, P. (2004). *Topics on Analysis in Metric Spaces*. Oxford University Press.

### Step 3: Tangent Measures and Profile Extraction

**Tangent Measures (Preiss, 1987):** For a Radon measure $\mu$ on $\mathbb{R}^n$ and $x \in \text{spt}(\mu)$, define:
$$\text{Tan}(\mu, x) := \left\{ \nu : \nu = \lim_{r_j \to 0} \frac{(T_{x, r_j})_\# \mu}{\mu(B_{r_j}(x))} \text{ in weak-* topology} \right\}$$

where $T_{x,r}(y) = (y - x)/r$.

**Existence (Preiss):** $\text{Tan}(\mu, x) \neq \emptyset$ for $\mu$-a.e. $x$.

**Reference:** Preiss, D. (1987). Geometry of measures in $\mathbb{R}^n$: Distribution, rectifiability, and densities. *Ann. of Math.*, 125, 537-643.

### Step 4: Profile Classification Modulo Symmetry

**$G$-Quotient:** The profile space is:
$$\mathcal{P} := \text{Tan}(\mu, \cdot) / G$$

where $G$ acts on tangent measures by:
$$(g \cdot \nu)(A) := \nu(g^{-1}(A))$$

**Finiteness (under energy bounds):** By compactness theorems:

**Theorem (De Lellis-Spadaro, 2011):** For area-minimizing currents in $\mathbb{R}^{n+k}$, the set of tangent cones at a singular point is finite.

**Reference:** De Lellis, C., Spadaro, E. (2011). Q-valued functions revisited. *Mem. Amer. Math. Soc.*, 211.

### Step 5: Surgery via Metric Gluing

**Metric Gluing (Burago-Burago-Ivanov, 2001):** Given:
- $(X_1, d_1)$ and $(X_2, d_2)$ complete metric spaces
- $A_1 \subset X_1$, $A_2 \subset X_2$ closed
- $\phi: A_1 \to A_2$ isometry

The **glued space** is:
$$X_1 \sqcup_\phi X_2 := (X_1 \sqcup X_2) / (a \sim \phi(a))$$

with the **gluing metric**:
$$d([x], [y]) := \inf_{a \in A_1} (d_1(x, a) + d_2(\phi(a), y'))$$

**Reference:** Burago, D., Burago, Y., Ivanov, S. (2001). *A Course in Metric Geometry*. AMS.

### Step 6: Surgery Operators

**Excision-Gluing:** For $\Sigma \subset X$ the singular set:
1. **Excise:** Remove tubular neighborhood $N_\varepsilon(\Sigma)$
2. **Cap:** Attach a "cap" space $C$ along $\partial N_\varepsilon(\Sigma) \cong \partial C$
3. **Glue:** Form $X' := (X \setminus N_\varepsilon(\Sigma)) \sqcup_{\text{id}} C$

**Pushout Construction:** In the category of metric spaces:
$$X' = (X \setminus \Sigma) \sqcup_{\partial N_\varepsilon(\Sigma)} C$$

This is the categorical pushout along the inclusion $\partial N_\varepsilon(\Sigma) \hookrightarrow C$.

**Reference:**
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5, 1-92.
- Perelman, G. (2003). Ricci flow with surgery on three-manifolds. *arXiv:math/0303109*.

### Step 7: Canonical Expansion Functor

**Functor Construction:** Define:
$$\mathcal{F}: \mathbf{Thin} \to \mathbf{Full}$$

where $\mathcal{F}(X, d, \Phi, G) := (X, d, \Phi, G, \pi_*(X), H_*(X), \text{sing}, \mathcal{P}, \mathcal{O}_S)$.

**Naturality:** For morphisms $f: (X_1, \ldots) \to (X_2, \ldots)$ in $\mathbf{Thin}$:
$$\mathcal{F}(f): \mathcal{F}(X_1) \to \mathcal{F}(X_2)$$

preserves all derived structures.

**Reference:** Mac Lane, S. (1998). *Categories for the Working Mathematician*. 2nd ed., Springer.

## Key GMT Inequalities Used

1. **Tangent Measure Existence (Preiss):**
   $$\text{Tan}(\mu, x) \neq \emptyset \quad \text{for } \mu\text{-a.e. } x$$

2. **Rectifiability Criterion:**
   $$\Theta^*(\mu, x) < \infty \text{ for } \mu\text{-a.e. } x \implies \mu \text{ is rectifiable}$$

3. **Metric Gluing Distance:**
   $$d_{X_1 \sqcup_\phi X_2}([x], [y]) = \inf_{a \in A} (d_1(x, a) + d_2(\phi(a), y'))$$

4. **Energy Lower Semicontinuity under Gluing:**
   $$\Phi(X') \leq \Phi(X) - \Phi(\text{excised}) + \Phi(\text{cap})$$

## Literature References

- Preiss, D. (1987). Geometry of measures in $\mathbb{R}^n$. *Ann. of Math.*, 125, 537-643.
- Ambrosio, L., Kirchheim, B. (2000). Currents in metric spaces. *Acta Math.*, 185, 1-80.
- Burago, D., Burago, Y., Ivanov, S. (2001). *A Course in Metric Geometry*. AMS.
- De Lellis, C., Spadaro, E. (2016). Regularity of area-minimizing currents I-III. *Geom. Funct. Anal.* and *Ann. of Math.*
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Perelman, G. (2002-2003). The entropy formula / Ricci flow with surgery / Finite extinction time. arXiv.
