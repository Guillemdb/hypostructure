---
title: "KRNL-HamiltonJacobi - Complexity Theory Translation"
---

# KRNL-HamiltonJacobi: Hamilton-Jacobi Characterization

## Original Statement (Hypostructure)

Under interface permits $D_E$ (dissipation-energy), $\mathrm{LS}_\sigma$ (Lojasiewicz-Simon), and $\mathrm{GC}_\nabla$ (gradient consistency), the Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static Hamilton-Jacobi equation:

$$\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)$$

subject to the boundary condition $\mathcal{L}(x) = \Phi_{\min}$ for $x \in M$.

The Lyapunov functional $\mathcal{L}$ encodes accumulated cost from $x$ to the minimum-energy manifold $M$, and the Hamilton-Jacobi PDE characterizes $\mathcal{L}$ as the distance function under the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$.

## Complexity Theory Statement

**Theorem (Continuous-Limit Bellman Equation).** Let $G = (V, E, w)$ be a weighted graph (state space) with:
- **States:** $V$ represents configurations
- **Transitions:** $E$ represents allowed moves between configurations
- **Costs:** $w: E \to \mathbb{R}^+$ assigns cost to each transition

Let $M \subseteq V$ be the set of **accepting states** (goal configurations). Define the **optimal cost function** $L: V \to \mathbb{R}^+$ by:

$$L(x) = \inf_{\gamma: x \rightsquigarrow M} \sum_{e \in \gamma} w(e)$$

where the infimum is over all paths from $x$ to $M$.

**Discrete Optimality (Bellman Equation):** The optimal cost function satisfies:

$$L(x) = \min_{y \sim x} \{w(x,y) + L(y)\} \quad \text{for } x \notin M$$

with boundary condition $L(x) = 0$ for $x \in M$.

**Continuum Limit (Eikonal/Hamilton-Jacobi):** As the graph becomes dense (mesh size $h \to 0$) with $w(x,y) \approx c(x) \cdot d(x,y)$ for local cost rate $c(x)$, the discrete Bellman equation converges to the **eikonal equation**:

$$|\nabla L(x)| = c(x)$$

equivalently written as $|\nabla L|^2 = c(x)^2 = D(x)$ where $D = c^2$ is the dissipation rate.

**Complexity Interpretation:** At every point in configuration space, the rate of change of optimal cost equals the local transition cost. The gradient direction indicates the optimal escape route.

## Terminology Translation Table

| Hypostructure | Complexity Theory | Interpretation |
|---------------|-------------------|----------------|
| Hamilton-Jacobi PDE $\|\nabla \mathcal{L}\|^2 = \mathfrak{D}$ | Bellman equation $L(x) = \min_y\{w(x,y) + L(y)\}$ | Optimal substructure principle |
| Viscosity solution | Well-defined at non-smooth points | Uniqueness via comparison principle |
| Lyapunov functional $\mathcal{L}(x)$ | Optimal cost-to-go $L(x)$ | Value function in dynamic programming |
| Minimum manifold $M$ | Accepting states | Goal configurations with zero cost |
| Boundary condition $\mathcal{L}\|_M = \Phi_{\min}$ | Terminal cost $L(x) = 0$ for $x \in M$ | Free exit at goal states |
| $\|\nabla \mathcal{L}\| = \sqrt{\mathfrak{D}}$ | $\|$gradient$\|$ = local cost rate $c(x)$ | Marginal cost of progress |
| Jacobi metric $g_{\mathfrak{D}}$ | Weighted graph metric | Cost-weighted distances |
| Gradient flow $\dot{x} = -\nabla \mathcal{L}$ | Greedy local optimization | Following steepest descent |
| Dissipation $\mathfrak{D}(x)$ | Local computational cost | Resources consumed per step |
| Conformal factor | Edge weight scaling | Inhomogeneous cost landscape |

## Proof Sketch

### Setup: The Discrete and Continuous Frameworks

**Discrete Setting.** Consider a weighted graph $G = (V, E, w)$ representing configuration space:
- Vertices $V$: configurations (states)
- Edges $E$: transitions (allowed operations)
- Weights $w(x,y)$: cost of transition from $x$ to $y$
- Terminal set $M \subseteq V$: accepting states

The **optimal cost function** $L: V \to \mathbb{R}^+ \cup \{\infty\}$ assigns to each state the minimum total cost to reach $M$.

**Continuous Setting.** A Riemannian manifold $(X, g)$ with:
- Cost rate function $c: X \to \mathbb{R}^+$ (local transition cost per unit distance)
- Terminal manifold $M \subset X$ with zero cost
- Dissipation $D(x) = c(x)^2$

The correspondence is: as the graph becomes a fine mesh approximating a continuous space, discrete path costs converge to curve integrals.

### Step 1: Discrete Bellman Equation (Dynamic Programming)

**Claim.** The optimal cost function $L$ satisfies the Bellman optimality principle:

$$L(x) = \min_{y: (x,y) \in E} \{w(x,y) + L(y)\} \quad \text{for } x \notin M$$

with $L(x) = 0$ for $x \in M$.

*Proof.* Let $\gamma^* = (x = x_0, x_1, \ldots, x_k \in M)$ be an optimal path from $x$ to $M$ with cost $L(x) = \sum_{i=0}^{k-1} w(x_i, x_{i+1})$.

Decompose: $L(x) = w(x, x_1) + \sum_{i=1}^{k-1} w(x_i, x_{i+1})$.

The tail $(x_1, \ldots, x_k)$ must be an optimal path from $x_1$ to $M$, otherwise we could improve $\gamma^*$ by substituting a better tail. Hence:

$$L(x) = w(x, x_1) + L(x_1) \geq \min_{y \sim x}\{w(x,y) + L(y)\}$$

Equality holds by taking $y = x_1$.

**Rearranged Form.** The Bellman equation can be rewritten as a discrete gradient condition:

$$L(x) - L(y) = w(x,y) - \epsilon(x,y)$$

where $\epsilon(x,y) \geq 0$ is the slack (zero when $y$ is on the optimal path from $x$). Along optimal paths: $L(x) - L(\text{next}(x)) = w(x, \text{next}(x))$.

### Step 2: Continuum Limit and the Eikonal Equation

**Mesh Approximation.** Consider a sequence of graphs $G_h = (V_h, E_h, w_h)$ with mesh size $h \to 0$:
- $V_h = X \cap (h\mathbb{Z})^n$ (lattice points in configuration space)
- Edges connect nearest neighbors: $(x,y) \in E_h$ iff $|x - y| = h$
- Edge weights: $w_h(x,y) = c(x) \cdot h + O(h^2)$

Let $L_h: V_h \to \mathbb{R}$ be the discrete optimal cost function.

**Claim.** As $h \to 0$, we have $L_h \to L$ uniformly on compact sets, where $L$ satisfies the eikonal equation $|\nabla L| = c$.

*Proof Sketch.* Along a direction $e$ with $|e| = 1$:

$$\frac{L_h(x) - L_h(x + he)}{h} \approx c(x)$$

by the Bellman equation $L_h(x) = w_h(x, x+he) + L_h(x+he) = c(x)h + L_h(x+he)$.

Taking $h \to 0$:

$$-\nabla L(x) \cdot e = c(x)$$

for the optimal direction $e$. Maximizing over unit directions:

$$|\nabla L(x)| = \max_{|e|=1} (-\nabla L \cdot e) = c(x)$$

**Eikonal Form.** Squaring both sides: $|\nabla L|^2 = c(x)^2 = D(x)$, which is the Hamilton-Jacobi equation.

### Step 3: The Eikonal Equation as Shortest-Path Characterization

**Claim.** The eikonal equation $|\nabla L| = c$ characterizes $L$ as the geodesic distance to $M$ under the **weighted metric** $ds = c(x) |dx|$.

*Proof.* Define the Jacobi-transformed metric $\tilde{g} = c^2 \cdot g$ (in complexity terms: edge weights incorporate local cost). The geodesic distance is:

$$\tilde{d}(x, M) = \inf_{\gamma: x \to M} \int_\gamma c(\gamma(t)) |\dot{\gamma}(t)| \, dt$$

This is exactly the continuum limit of the discrete path cost $\sum_e w(e)$.

**Standard Result.** The distance function $d_M(x) = \tilde{d}(x, M)$ satisfies the eikonal equation $|\nabla d_M|_{\tilde{g}} = 1$ in the weighted metric, which transforms to $|\nabla d_M|_g = c(x)$ in the original metric.

**Identification.** Since $L(x) = \tilde{d}(x, M)$ by construction, we have $|\nabla L| = c$, equivalently $|\nabla L|^2 = D$.

### Step 4: Uniqueness via Comparison Principle (Viscosity Theory)

The eikonal equation $|\nabla L| = c$ may have non-smooth solutions (the distance function has kinks at the "cut locus" where multiple optimal paths meet). **Viscosity solutions** provide the correct uniqueness framework.

**Definition (Viscosity Solution).** A continuous function $L: X \to \mathbb{R}$ is a viscosity solution of $|\nabla L| = c$ if:

1. **(Subsolution)** For every smooth function $\phi$ with $L - \phi$ having a local maximum at $x_0$:
   $$|\nabla \phi(x_0)| \leq c(x_0)$$

2. **(Supersolution)** For every smooth function $\phi$ with $L - \phi$ having a local minimum at $x_0$:
   $$|\nabla \phi(x_0)| \geq c(x_0)$$

**Interpretation.** At points where $L$ is non-differentiable, we test using smooth "probe functions" $\phi$ that touch $L$ from above or below. This captures the correct one-sided derivatives.

**Comparison Principle (Crandall-Lions).** If $L_1$ is a viscosity subsolution and $L_2$ is a viscosity supersolution with $L_1 \leq L_2$ on $\partial \Omega \cup M$, then $L_1 \leq L_2$ on $\Omega$.

**Uniqueness Corollary.** There is a unique viscosity solution $L$ to $|\nabla L| = c$ with boundary condition $L|_M = 0$. This solution equals the geodesic distance function.

**Complexity Interpretation.** The comparison principle is the **optimality verification**: any proposed cost function $\tilde{L}$ satisfying local consistency ($|\nabla \tilde{L}| \leq c$ subsolution) with correct boundary conditions must equal $L$. There is a unique optimal cost-to-go function.

### Certificate Construction

The complexity theory certificate consists of:

**Certificate:** $K_{\text{HJ}}^+ = (\text{solution}, \text{local\_optimality}, \text{uniqueness})$

1. **Solution Function:** $L: V \to \mathbb{R}^+$ mapping each configuration to its optimal cost-to-go
   - Computed via dynamic programming (discrete) or PDE solver (continuous)
   - Satisfies boundary condition $L|_M = 0$

2. **Local Optimality Proof:** Verification that for each $x \notin M$:
   - *Discrete:* $L(x) = \min_{y \sim x}\{w(x,y) + L(y)\}$ (Bellman equation)
   - *Continuous:* $|\nabla L(x)| = c(x)$ (eikonal equation)
   - This certifies that $L$ represents a consistent assignment of costs

3. **Uniqueness Proof:** Demonstration via comparison principle:
   - Any other function $\tilde{L}$ satisfying local optimality and boundary conditions equals $L$
   - The optimal cost function is unique

**Explicit Certificate Construction (Discrete):**

```
Algorithm: Compute-Optimal-Cost
Input: Graph G = (V, E, w), terminal set M
Output: Optimal cost function L and optimal policy pi

1. Initialize: L(x) = 0 for x in M, L(x) = infinity otherwise
2. Priority queue Q with M, keyed by L-values
3. While Q non-empty:
     x = extract-min(Q)
     For each predecessor y with (y, x) in E:
       If L(y) > w(y,x) + L(x):
         L(y) = w(y,x) + L(x)
         pi(y) = x
         Update Q with (y, L(y))
4. Return (L, pi)

Certificate verification:
- For each x not in M: check L(x) = min_{y: (x,y) in E} {w(x,y) + L(y)}
- For each x in M: check L(x) = 0
- Output: (L, verification_trace, policy_pi)
```

## Connections to Classical Results

### Eikonal Equation and Geometric Optics

The eikonal equation $|\nabla L| = c$ arises in geometric optics where $L$ is the phase of a wave and $c(x)$ is the local wave speed. Light rays are orthogonal to level sets of $L$ (wavefronts). In complexity theory:
- **Wavefronts** = iso-cost surfaces (all states with same optimal cost)
- **Rays** = optimal paths (trajectories of steepest descent in cost)
- **Refraction** = path bending due to inhomogeneous costs

### Viscosity Solutions (Crandall-Lions 1983)

The viscosity solution framework, developed by Crandall and Lions, provides:
1. **Well-posedness** for Hamilton-Jacobi equations with discontinuous Hamiltonians
2. **Uniqueness** via the comparison principle
3. **Stability** under approximation (the discrete Bellman solutions converge to the continuous viscosity solution)

**Key Insight.** At non-smooth points (cut locus, shock formation), multiple gradients are possible. The viscosity framework selects the physically/computationally meaningful one: the limit of smooth approximations.

### Optimal Control Theory (Bellman 1957)

The Hamilton-Jacobi-Bellman equation is the continuous-time limit of dynamic programming:

$$\partial_t V + H(x, \nabla_x V) = 0$$

where $V(x,t)$ is the cost-to-go at time $t$, and $H$ is the Hamiltonian. The static version (infinite horizon, discounted or minimum-time) reduces to:

$$H(x, \nabla V) = 0$$

For minimum-cost problems with running cost $c(x)$ and unit speed:
$$H(x, p) = |p| - c(x)$$
giving $|\nabla V| = c(x)$.

### Dijkstra's Algorithm as Bellman Propagation

Dijkstra's algorithm for shortest paths is exactly discrete Bellman propagation:
- Initialize distances from source
- Greedily extend shortest-path tree
- Bellman equation: $d(v) = \min_{u \sim v}\{w(u,v) + d(u)\}$

The algorithm complexity $O(|E| + |V| \log |V|)$ reflects the cost of certifying optimality.

### Connection to the Hypostructure Framework

The hypostructure Hamilton-Jacobi theorem states $\|\nabla_g \mathcal{L}\|^2 = \mathfrak{D}$ under permits $D_E$, $\mathrm{LS}_\sigma$, $\mathrm{GC}_\nabla$. In complexity terms:

| Permit | Complexity Condition |
|--------|---------------------|
| $D_E$ (Dissipation-Energy) | Finite total cost: $\sum_{\text{path}} w(e) < \infty$ |
| $\mathrm{LS}_\sigma$ (Lojasiewicz-Simon) | Convergence guarantee: no cycles of zero cost |
| $\mathrm{GC}_\nabla$ (Gradient Consistency) | Local cost = gradient norm: $w(x,y)/d(x,y) \to c(x)$ |

The Lyapunov functional $\mathcal{L}$ is the optimal cost-to-go, the minimum manifold $M$ is the set of accepting states, and the gradient flow $\dot{x} = -\nabla \mathcal{L}$ is the policy of always taking the locally optimal action.

## Literature References

- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. [Foundation of optimal cost-to-go]
- Crandall, M.G., Lions, P.-L. (1983). Viscosity solutions of Hamilton-Jacobi equations. *Trans. Amer. Math. Soc.*, 277, 1-42. [Uniqueness theory]
- Evans, L.C. (2010). *Partial Differential Equations*. AMS. [Chapter 10: Hamilton-Jacobi equations]
- Sethian, J.A. (1999). *Level Set Methods and Fast Marching Methods*. Cambridge University Press. [Computational eikonal solvers]
- Dijkstra, E.W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269-271. [Discrete optimal path algorithm]
- Bertsekas, D.P. (2017). *Dynamic Programming and Optimal Control*. Athena Scientific. [Modern treatment]
