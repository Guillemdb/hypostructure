# ACT-Compactify: Lyapunov Compactification Principle — GMT Translation

## Original Statement (Hypostructure)

The Lyapunov compactification principle shows how to compactify configuration spaces using Lyapunov functions, adding ideal points at infinity where the Lyapunov function tends to its extremes.

## GMT Setting

**Lyapunov Function:** $\Phi: X \to \mathbb{R}$ monotone along flow

**Compactification:** Add ideal points at $\Phi \to \pm\infty$

**Control:** Lyapunov structure controls behavior at infinity

## GMT Statement

**Theorem (Lyapunov Compactification).** For flow space $\mathbf{I}_k(M)$ with Lyapunov $\Phi$:

1. **Compactification:** $\overline{\mathbf{I}_k(M)} = \mathbf{I}_k(M) \cup \partial_\infty$

2. **Lyapunov Extension:** $\Phi$ extends continuously to $\overline{\mathbf{I}_k(M)}$

3. **Boundary:** $\partial_\infty = \Phi^{-1}(\pm\infty)$ consists of limiting configurations

4. **Flow Extension:** Flow extends to compactification with fixed points at boundary

## Proof Sketch

### Step 1: Lyapunov Functions

**Definition:** $\Phi: X \to \mathbb{R}$ is Lyapunov for flow $\varphi_t$ if:
$$\frac{d}{dt}\Phi(\varphi_t(x)) \leq 0$$

**Strict Lyapunov:** Inequality strict except at equilibria.

**Reference:** LaSalle, J. P. (1976). *The Stability of Dynamical Systems*. SIAM.

### Step 2: Sublevel Compactness

**GMT Setting:** For area functional $\mathbf{M}$ on $\mathbf{I}_k(M)$:
$$\{T : \mathbf{M}(T) + \mathbf{M}(\partial T) \leq C\}$$

is precompact in flat topology.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.

**Lyapunov Sublevel:** $\{\Phi \leq c\}$ is compact for each $c$.

### Step 3: End Compactification

**Definition:** The end compactification adds points at infinity:
$$\bar{X} = X \cup \mathcal{E}(X)$$

where $\mathcal{E}(X) = $ ends of $X$.

**Reference:** Freudenthal, H. (1931). Über die Enden topologischer Räume und Gruppen. *Math. Z.*, 33, 692-713.

**Lyapunov Ends:** Ends correspond to directions where $\Phi \to \pm\infty$.

### Step 4: Metric Compactification

**Gromov Boundary:** For proper geodesic space $X$:
$$\partial_\infty X = \{\text{equivalence classes of rays}\}$$

**Reference:** Bridson, M., Haefliger, A. (1999). *Metric Spaces of Non-Positive Curvature*. Springer.

**Lyapunov Ray:** Ray along which $\Phi$ is monotone.

### Step 5: Flow Extension

**Theorem:** Flow $\varphi_t$ extends to $\bar{X}$:
$$\bar{\varphi}_t: \bar{X} \to \bar{X}$$

with $\bar{\varphi}_t|_{\partial_\infty} = \text{id}$ (boundary is fixed).

*Proof:*
- $\Phi$ monotone implies trajectories have limits at $\pm\infty$
- Limits define points in $\partial_\infty$
- Extended flow fixes these limits

### Step 6: Attractor-Repeller Decomposition

**Conley Decomposition:** Using Lyapunov function:
$$X = A \cup R \cup C$$

where $A$ = attractor, $R$ = repeller, $C$ = connecting orbits.

**Reference:** Conley, C. (1978). *Isolated Invariant Sets*. AMS.

**Compactification:** $A, R$ may include ideal points.

### Step 7: One-Point Compactification

**Simple Case:** If $\Phi \to +\infty$ uniformly at infinity:
$$\bar{X} = X \cup \{\infty\}$$

one-point compactification.

**Lyapunov:** $\Phi(\infty) = +\infty$ defines boundary value.

### Step 8: Mass Compactification for Currents

**For $\mathbf{I}_k(M)$:** Compactify by:
$$\overline{\mathbf{I}_k(M)} = \{T \in \mathbf{I}_k(M) : \mathbf{M}(T) \leq C\} \cup \partial_\infty$$

where $\partial_\infty$ = limits of sequences with $\mathbf{M} \to \infty$.

**Varifold Compactification:** Integral varifolds with bounded mass are precompact.

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.

### Step 9: Lyapunov at Infinity

**Limiting Lyapunov:** For $T_n \to T_\infty \in \partial_\infty$:
$$\Phi(T_\infty) = \lim_{n \to \infty} \Phi(T_n)$$

when limit exists.

**Well-Defined:** Lyapunov value on boundary is well-defined.

### Step 10: Compilation Theorem

**Theorem (Lyapunov Compactification):**

1. **Compactification:** Add ideal points at Lyapunov extremes

2. **Extension:** $\Phi$ and flow extend to boundary

3. **Fixed Points:** Boundary consists of equilibria of extended flow

4. **Attractor-Repeller:** Decomposition extends to compactification

**Applications:**
- Asymptotic analysis of flows
- Compactification of moduli spaces
- Boundary behavior of variational problems

## Key GMT Inequalities Used

1. **Lyapunov Monotonicity:**
   $$\frac{d}{dt}\Phi(\varphi_t(x)) \leq 0$$

2. **Precompactness:**
   $$\{\mathbf{M} + \mathbf{M}\partial \leq C\} \text{ precompact}$$

3. **Limit Extension:**
   $$\Phi(T_\infty) = \lim \Phi(T_n)$$

4. **Boundary Fixed:**
   $$\bar{\varphi}_t|_{\partial_\infty} = \text{id}$$

## Literature References

- LaSalle, J. P. (1976). *Stability of Dynamical Systems*. SIAM.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Freudenthal, H. (1931). Enden topologischer Räume. *Math. Z.*, 33.
- Bridson, M., Haefliger, A. (1999). *Non-Positive Curvature*. Springer.
- Conley, C. (1978). *Isolated Invariant Sets*. AMS.
- Allard, W. K. (1972). First variation of a varifold. *Ann. of Math.*, 95.
