# KRNL-Subsystem: Subsystem Inheritance — GMT Translation

## Original Statement (Hypostructure)

If a parent system is globally regular (Lock blocked), then every invariant subsystem inherits regularity. Singularities cannot emerge in restricted dynamics.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — complete Riemannian manifold

**Parent Current:** $T \in \mathbf{I}_k(M)$ — integral current representing the full system

**Subsystem:** $S \subset T$ — integral current with $\text{spt}(S) \subset \text{spt}(T)$

**Invariance:** $\phi_t(S) = S$ for all $t$ under the flow $\phi_t$

**Regularity:** $\text{sing}(T) = \emptyset$ (no singular points)

## GMT Statement

**Theorem (Regularity Inheritance for Invariant Subsystems).** Let $T \in \mathbf{I}_k(M)$ be a regular integral current (smooth embedded submanifold) satisfying:

1. **(Global Regularity)** $\text{sing}(T) = \emptyset$

2. **(Invariant Subsystem)** $S \in \mathbf{I}_j(M)$ with $j \leq k$ satisfies:
   - $\text{spt}(S) \subset \text{spt}(T)$
   - $S$ is invariant under the flow: $(\phi_t)_\# S = S$

3. **(Energy Control)** $\mathbf{M}(S) \leq \mathbf{M}(T)$

Then $S$ is also regular: $\text{sing}(S) = \emptyset$.

**Corollary:** If $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(T)) = \emptyset$ (Lock blocked for $T$), then $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(S)) = \emptyset$ for any invariant $S \subset T$.

## Proof Sketch

### Step 1: Invariant Manifold Theory Background

**Fenichel's Theorem (1971):** Let $M_0 \subset M$ be a compact normally hyperbolic invariant manifold for a flow $\phi_t$. Then:
1. $M_0$ persists under small $C^1$ perturbations
2. Nearby invariant manifolds are $C^r$-diffeomorphic to $M_0$
3. The stable/unstable manifolds $W^{s/u}(M_0)$ are $C^r$ submanifolds

**Reference:** Fenichel, N. (1971). Persistence and smoothness of invariant manifolds for flows. *Indiana Univ. Math. J.*, 21, 193-226.

### Step 2: Current Restriction

**Restriction Current:** For $S \subset T$, define the restriction as:
$$T|_S := T \llcorner \text{spt}(S)$$

**Mass Inequality:** $\mathbf{M}(T|_S) \leq \mathbf{M}(T)$

**First Variation Inheritance:** If $\delta T = H_T \cdot \|T\|$, then:
$$\delta(T|_S) = (H_T)|_S \cdot \|T|_S\|$$

The mean curvature restricts to the subsystem.

### Step 3: Singularity Obstruction Transfer

**Contrapositive Argument:** Suppose $\text{sing}(S) \neq \emptyset$. Then there exists $x_0 \in \text{spt}(S)$ with:
$$\Theta^j(S, x_0) \geq 1 + \varepsilon_0$$

(density exceeds regularity threshold).

**Embedding into Parent:** Since $\text{spt}(S) \subset \text{spt}(T)$, we have $x_0 \in \text{spt}(T)$. The tangent cone at $x_0$ satisfies:
$$\text{Tan}(S, x_0) \subset \text{Tan}(T, x_0)$$

### Step 4: Tangent Cone Analysis

**Tangent Cone of Subsystem:** The tangent cone $C_S := \text{Tan}(S, x_0)$ is a $j$-dimensional cone in $\mathbb{R}^n$.

**Parent Tangent Cone:** The tangent cone $C_T := \text{Tan}(T, x_0)$ is a $k$-dimensional cone with $C_S \subset C_T$.

**Regularity of $C_T$:** Since $T$ is regular, $C_T = T_{x_0}(\text{spt}(T))$ is a $k$-dimensional linear subspace (the tangent plane).

**Consequence for $C_S$:** A cone contained in a linear subspace must itself be a linear subspace:
$$C_S \subset C_T \cong \mathbb{R}^k \implies C_S \cong \mathbb{R}^j$$

Hence $C_S$ is regular (a $j$-plane), contradicting $\text{sing}(S) \ni x_0$.

### Step 5: Hirsch-Pugh-Shub Theory

**Normally Hyperbolic Invariant Manifolds:** The theory of Hirsch-Pugh-Shub (1977) provides:

**Theorem (HPS):** Let $\Lambda \subset M$ be a compact invariant set for a $C^r$ flow, $r \geq 1$. If $\Lambda$ is normally hyperbolic:
$$T_\Lambda M = T\Lambda \oplus E^s \oplus E^u$$
with uniform expansion/contraction in $E^{s/u}$, then $\Lambda$ is a $C^r$ submanifold.

**Reference:** Hirsch, M., Pugh, C., Shub, M. (1977). *Invariant Manifolds*. Lecture Notes in Math. 583, Springer.

### Step 6: Energy Monotonicity Transfer

**Monotonicity for Subsystem:** The monotonicity formula for $S$:
$$\frac{\|S\|(B_r(x))}{r^j} \leq e^{Cr} \frac{\|S\|(B_R(x))}{R^j}$$

is inherited from the ambient current $T$, with the same constant $C$ (depending on $\|H_T\|$).

**Density Bound Transfer:** Since $S \subset T$ and $T$ is regular:
$$\Theta^j(S, x) \leq \Theta^k(T, x) = 1$$

(no density accumulation in the subsystem).

### Step 7: Categorical Inheritance

**Morphism Obstruction:** If $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(T)) = \emptyset$, then for any subsystem $S$:

The inclusion $\iota: S \hookrightarrow T$ induces:
$$\iota^*: \text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(T)) \to \text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(S))$$

If a singularity morphism $\phi: \mathbb{H}_{\text{bad}} \to \mathbb{H}(S)$ existed, then $\iota \circ \phi: \mathbb{H}_{\text{bad}} \to \mathbb{H}(T)$ would be a singularity morphism for $T$, contradicting the Lock.

**Reference:** Wiggins, S. (2003). *Introduction to Applied Nonlinear Dynamical Systems and Chaos*. 2nd ed., Springer. [Chapter 3: Invariant Manifolds]

## Key GMT Inequalities Used

1. **Mass Restriction:**
   $$\mathbf{M}(T|_S) \leq \mathbf{M}(T)$$

2. **Tangent Cone Inclusion:**
   $$\text{Tan}(S, x) \subset \text{Tan}(T, x)$$

3. **Density Inheritance:**
   $$\Theta^j(S, x) \leq \Theta^k(T, x) \cdot \binom{k}{j}$$

4. **Monotonicity Transfer:**
   $$\frac{\|S\|(B_r(x))}{r^j} \text{ bounded if } \frac{\|T\|(B_r(x))}{r^k} \text{ bounded}$$

## Literature References

- Fenichel, N. (1971). Persistence and smoothness of invariant manifolds for flows. *Indiana Univ. Math. J.*, 21, 193-226.
- Hirsch, M., Pugh, C., Shub, M. (1977). *Invariant Manifolds*. Lecture Notes in Math. 583, Springer.
- Wiggins, S. (2003). *Introduction to Applied Nonlinear Dynamical Systems and Chaos*. Springer.
- Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.2: Restriction and Slicing]
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU. [Chapter 5: Monotonicity]
- Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95, 417-491.
