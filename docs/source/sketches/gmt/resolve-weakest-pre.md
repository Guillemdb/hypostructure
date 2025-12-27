# RESOLVE-WeakestPre: Weakest Precondition Principle — GMT Translation

## Original Statement (Hypostructure)

Users provide only definitions (state space, energy, dissipation, symmetry) and interface implementations. The Sieve automatically determines regularity.

## GMT Setting

**Input:** $(X, d, \mu, \Phi, G)$ — thin objects
**Output:** Regularity verdict $\mathcal{V} \in \{\text{REG}, \text{SING}, \text{INC}\}$

**Interface Predicates:** Computable functions $\mathcal{P}_i: X \to \{\text{YES}, \text{NO}\}$

**Weakest Precondition:** Minimal assumptions for regularity guarantee

## GMT Statement

**Theorem (Weakest Precondition for Regularity).** Let $(X, d)$ be a complete metric space with energy $\Phi: X \to [0, \infty]$. The following are the **minimal sufficient conditions** for global regularity:

**(WP1) Finite Energy:** $\mathbf{M}(T) := \sup_x \Phi(x) < \infty$ on bounded sets

**(WP2) Dissipation Bound:** $\int_0^T |\partial \Phi|^2(S_t x) \, dt \leq \Phi(x)$

**(WP3) Compactness Modulo Symmetry:** $\{x : \Phi(x) \leq c\} / G$ is compact

**(WP4) Stiffness:** $|\partial \Phi|(x) \geq C |\Phi(x) - \Phi_{\min}|^{1-\theta}$ near critical set

**(WP5) Capacity Bound:** $\text{Cap}(\text{sing}) = 0$

These are **weakest** in the sense that removing any condition allows counterexamples.

## Proof Sketch

### Step 1: Dijkstra's Weakest Precondition Calculus

**Program Logic Framework:** In Dijkstra's framework (1976), for a program $S$ and postcondition $R$:
$$\text{wp}(S, R) := \text{weakest predicate } P \text{ such that } \{P\} S \{R\}$$

**Translation to GMT:** The "program" is the gradient flow evolution; the "postcondition" is regularity.

$$\text{wp}(\text{GradFlow}, \text{Regularity}) = \text{WP1} \land \text{WP2} \land \text{WP3} \land \text{WP4} \land \text{WP5}$$

**Reference:** Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice-Hall.

### Step 2: Necessity of WP1 (Finite Energy)

**Counterexample without WP1:** Let $X = \mathbb{R}$, $\Phi(x) = e^x$. Then $|\partial \Phi|(x) = e^x \to \infty$ as $x \to \infty$, and the gradient flow $\dot{x} = -e^x$ blows up in finite time:
$$x(t) = -\log(e^{-x_0} + t) \to -\infty \text{ as } t \to e^{-x_0}$$

**Necessity:** Without energy bounds, finite-time blow-up is possible.

### Step 3: Necessity of WP2 (Dissipation)

**Counterexample without WP2:** Let $X = \mathbb{R}^2$, $\Phi(x, y) = x^2 - y^2$ (saddle). The flow:
$$\dot{x} = -2x, \quad \dot{y} = 2y$$

has $y(t) \to \infty$ (blow-up along unstable direction) despite bounded energy on some trajectories.

**Energy-Dissipation Identity (AGS):** The EDI ensures:
$$\Phi(S_t x) + \int_0^t |\partial \Phi|^2 \, ds = \Phi(x)$$

Without this, energy can transfer rather than dissipate.

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.

### Step 4: Necessity of WP3 (Compactness)

**Counterexample without WP3:** Let $X = \mathbb{R}^n$, $\Phi(x) = |x|^2 / (1 + |x|^2)$. Energy is bounded but:
- Sublevel sets $\{\Phi \leq c\}$ are non-compact for $c < 1$
- Trajectories can escape to infinity without concentrating

**Lions' Concentration-Compactness:** Compactness modulo symmetry is precisely the hypothesis of Lions (1984) ensuring profile extraction.

**Reference:** Lions, P.-L. (1984). The concentration-compactness principle. *Ann. Inst. H. Poincaré*, 1, 109-145.

### Step 5: Necessity of WP4 (Stiffness)

**Counterexample without WP4:** Let $\Phi(x) = |x|^\alpha$ with $\alpha < 2$. Near 0:
$$|\partial \Phi|(x) = \alpha |x|^{\alpha - 1}$$

The Łojasiewicz inequality requires $|\partial \Phi| \geq C |\Phi|^{1-\theta}$, i.e.:
$$|x|^{\alpha - 1} \geq C |x|^{\alpha(1-\theta)}$$

This fails for $\alpha < 2$ and small $\theta$.

**Consequence:** Without stiffness, convergence can be arbitrarily slow (only polynomial, not exponential), and pathological accumulation can occur.

**Reference:** Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES preprint.

### Step 6: Necessity of WP5 (Capacity)

**Counterexample without WP5:** The Cantor set $C \subset [0,1]$ has:
- $\mathcal{H}^s(C) = 1$ for $s = \log 2 / \log 3$
- $\text{Cap}_{1,2}(C) > 0$ in dimension 1

A flow with singularities on $C$ is geometrically regular (codimension > 0) but analytically singular.

**Federer's Theorem:** Capacity zero is necessary for removability:

**Theorem (Federer, 1969):** A closed set $E \subset \mathbb{R}^n$ is removable for $W^{1,p}$ functions if and only if $\text{Cap}_{1,p}(E) = 0$.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 2.10.19]

### Step 7: Sufficiency (Combined Conditions)

**Theorem:** WP1-WP5 together imply global regularity.

*Proof Sketch:*
1. **WP1 + WP2:** Energy-dissipation gives finite-time bounds
2. **WP3:** Compactness extracts convergent subsequences
3. **WP4:** Łojasiewicz-Simon ensures convergence to equilibria
4. **WP5:** Capacity zero singular set is removable

**Key Combination (Simon, 1983):** The interplay of compactness + stiffness yields:
$$d(S_t x, \mathcal{M}) \leq C(1 + t)^{-1/(1-2\theta)}$$

for some $\theta \in (0, 1/2)$, ensuring asymptotic regularity.

**Reference:** Simon, L. (1983). Asymptotics for a class of non-linear evolution equations. *Ann. of Math.*, 118, 525-571.

### Step 8: Automation via Predicate Evaluation

**Sieve Algorithm:** Given thin objects, evaluate predicates:
1. Check WP1: $\sup_{\text{bounded}} \Phi < \infty$
2. Check WP2: Verify EDI from flow equations
3. Check WP3: Test compactness via covering arguments
4. Check WP4: Compute Łojasiewicz exponent
5. Check WP5: Estimate capacity of potential singular set

**Output:**
- All YES $\Rightarrow$ $\mathcal{V} = \text{REG}$
- Some NO $\Rightarrow$ $\mathcal{V} = \text{SING}$ with mode classification
- Some INC $\Rightarrow$ $\mathcal{V} = \text{INC}$ with missing premises

## Key GMT Inequalities Used

1. **Energy-Dissipation Inequality:**
   $$\Phi(S_t x) + \int_0^t |\partial \Phi|^2 \leq \Phi(x)$$

2. **Łojasiewicz-Simon Gradient Inequality:**
   $$|\partial \Phi|(x) \geq C |\Phi(x) - \Phi_*|^{1-\theta}$$

3. **Capacity-Removability:**
   $$\text{Cap}_{1,p}(E) = 0 \implies E \text{ removable for } W^{1,p}$$

4. **Compactness Modulo Symmetry:**
   $$\{x_n\} \text{ bounded} \implies \exists g_n \in G: g_n \cdot x_{n_k} \to x_\infty$$

## Literature References

- Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice-Hall.
- Back, R.-J., von Wright, J. (1998). *Refinement Calculus*. Springer.
- Lions, P.-L. (1984). Concentration-compactness I & II. *Ann. Inst. H. Poincaré*, 1.
- Simon, L. (1983). Asymptotics for non-linear evolution equations. *Ann. of Math.*, 118, 525-571.
- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
