# FACT-SoftWP: Soft→WP Compilation — GMT Translation

## Original Statement (Hypostructure)

The soft permit $K_{D_E}^+ \land K_{C_\mu}^+ \land K_{\text{SC}_\lambda}^+ \land K_{\text{LS}_\sigma}^+$ compiles to a well-posedness theorem for the associated evolution equation.

## GMT Setting

**Evolution:** $\partial_t T = -\nabla \Phi(T)$ — gradient flow on currents

**Well-Posedness (Hadamard):**
1. Existence of solution
2. Uniqueness of solution
3. Continuous dependence on initial data

**Soft Certificates:** $K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+$

## GMT Statement

**Theorem (Soft→WP Compilation).** If soft certificates hold, then:

1. **Existence:** For any $T_0 \in \mathbf{I}_k(M)$ with $\mathbf{M}(T_0) \leq \Lambda$, there exists a solution $T: [0, \infty) \to \mathbf{I}_k(M)$

2. **Uniqueness:** Solutions are unique in the class of dissipative solutions

3. **Stability:** $d_{\text{flat}}(T_t, S_t) \leq e^{Ct} \cdot d_{\text{flat}}(T_0, S_0)$

## Proof Sketch

### Step 1: Existence via Minimizing Movements

**De Giorgi's Minimizing Movements (1993):** Construct discrete approximation:
$$T_\tau^{n+1} := \arg\min_{S} \left\{ \Phi(S) + \frac{d^2(S, T_\tau^n)}{2\tau} \right\}$$

**Reference:** De Giorgi, E. (1993). New problems on minimizing movements. *Boundary Value Problems for PDE and Applications*, 81-98.

**Convergence (Ambrosio-Gigli-Savaré, 2008):** As $\tau \to 0$:
$$T_\tau(t) := T_\tau^{\lfloor t/\tau \rfloor} \to T(t)$$

in metric sense.

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces*. Birkhäuser.

### Step 2: Compactness Ensures Limit Exists

**Use of $K_{C_\mu}^+$:** The mass bound implies:
$$\sup_{n} \mathbf{M}(T_\tau^n) \leq \Lambda$$

By Federer-Fleming compactness, subsequences converge.

**Diagonal Argument:** For $\tau_j \to 0$, extract convergent subsequence $T_{\tau_j}(t) \to T(t)$ for all $t$.

### Step 3: Energy-Dissipation Identity

**Use of $K_{D_E}^+$:** The limit satisfies:
$$\Phi(T(t)) + \int_0^t |\partial \Phi|^2(T(s)) \, ds = \Phi(T_0)$$

**Proof:** Pass to limit in discrete energy identity:
$$\Phi(T_\tau^n) + \sum_{k=0}^{n-1} \frac{d^2(T_\tau^{k+1}, T_\tau^k)}{2\tau} \leq \Phi(T_\tau^0)$$

### Step 4: Uniqueness via Łojasiewicz-Simon

**Use of $K_{\text{LS}_\sigma}^+$:** The Łojasiewicz inequality implies convergence to equilibrium:
$$|\nabla \Phi|(T) \geq c |\Phi(T) - \Phi_*|^{1-\theta}$$

**Simon's Uniqueness (1983):** Near equilibrium, solutions are unique.

**Reference:** Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations. *Ann. of Math.*, 118, 525-571.

**Global Uniqueness:** Combine with energy-dissipation identity. If $T, S$ are two solutions with $T_0 = S_0$:
$$\frac{d}{dt} d^2(T_t, S_t) \leq C \cdot d^2(T_t, S_t)$$

By Gronwall, $d(T_t, S_t) = 0$ for all $t$.

### Step 5: Stability via Contraction

**λ-Convexity:** If $\Phi$ is $\lambda$-convex on $(\mathbf{I}_k(M), d_{\text{flat}})$:
$$\Phi(\gamma_s) \leq (1-s)\Phi(\gamma_0) + s\Phi(\gamma_1) - \frac{\lambda}{2}s(1-s)d^2(\gamma_0, \gamma_1)$$

**Contraction (Ambrosio-Gigli-Savaré):** For $\lambda \geq 0$:
$$d(T_t, S_t) \leq e^{-\lambda t} d(T_0, S_0)$$

For $\lambda < 0$:
$$d(T_t, S_t) \leq e^{|\lambda| t} d(T_0, S_0)$$

**Use of $K_{\text{SC}_\lambda}^+$:** Scale coherence provides control on the convexity constant $\lambda$.

### Step 6: Regularity of Solutions

**Theorem:** Under soft permits, the solution $T(t)$ satisfies:

1. $T(t) \in \mathbf{I}_k(M)$ for all $t \geq 0$
2. $\text{sing}(T(t)) \subset \text{sing}(T_0)$ for $t > 0$ (regularity improvement)
3. $T(t) \to T_\infty$ as $t \to \infty$ (convergence to equilibrium)

**Proof of (3):** By $K_{\text{LS}_\sigma}^+$ and Simon (1983):
$$\int_0^\infty \|T'(t)\| \, dt < \infty$$

Hence $T(t)$ has finite length and converges.

### Step 7: Compilation Theorem

**Theorem (Soft→WP):** The compilation map:
$$\text{Compile}: (K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+) \to \text{WP}(\partial_t T = -\nabla\Phi)$$

is well-defined and:
- **Sound:** If soft certificates hold, WP holds
- **Complete:** WP implies existence of soft certificates (converse direction)

**Soundness Proof:** Steps 1-6 above.

**Completeness Proof:** Given WP:
- Existence implies bounded orbits ($K_{C_\mu}^+$)
- Uniqueness implies dissipation structure ($K_{D_E}^+$)
- Stability implies scale control ($K_{\text{SC}_\lambda}^+$)
- Convergence to equilibrium implies Łojasiewicz ($K_{\text{LS}_\sigma}^+$)

### Step 8: Examples

**Example 1: Mean Curvature Flow**

- **Soft permits:** Huisken's monotonicity, Brakke compactness, Schulze's LS
- **WP Result:** Brakke flow exists, is unique among unit-density flows

**Reference:** Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.

**Example 2: Ricci Flow**

- **Soft permits:** Perelman's entropy, Hamilton's compactness, Perelman's no local collapsing
- **WP Result:** Ricci flow with surgery is well-posed

**Reference:** Perelman, G. (2002-2003). The entropy formula / Ricci flow with surgery. arXiv.

**Example 3: Harmonic Map Heat Flow**

- **Soft permits:** Energy monotonicity, Struwe compactness, Simon's LS
- **WP Result:** Weak solutions exist, are unique, converge to harmonic map

**Reference:** Struwe, M. (1985). On the evolution of harmonic maps in higher dimensions. *J. Diff. Geom.*, 28, 485-502.

## Key GMT Inequalities Used

1. **Minimizing Movements:**
   $$T^{n+1} = \arg\min \left\{ \Phi(S) + \frac{d^2(S, T^n)}{2\tau} \right\}$$

2. **Energy-Dissipation:**
   $$\Phi(T_t) + \int_0^t |\partial\Phi|^2 = \Phi(T_0)$$

3. **Contraction:**
   $$d(T_t, S_t) \leq e^{-\lambda t} d(T_0, S_0)$$

4. **Simon's Convergence:**
   $$\int_0^\infty \|T'(t)\| < \infty \implies T_t \to T_\infty$$

## Literature References

- De Giorgi, E. (1993). New problems on minimizing movements. *Boundary Value Problems*.
- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
- Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.
- Struwe, M. (1985). Evolution of harmonic maps. *J. Diff. Geom.*, 28.
- Perelman, G. (2002-2003). Ricci flow papers. arXiv.
