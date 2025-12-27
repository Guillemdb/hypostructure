# LOCK-UniqueAttractor: Unique-Attractor Lock — GMT Translation

## Original Statement (Hypostructure)

The unique-attractor lock shows that when a flow has a unique attractor in a configuration class, all trajectories must converge to it, blocking alternative limiting behaviors.

## GMT Setting

**Attractor:** Invariant set attracting all nearby trajectories

**Uniqueness:** Single attractor in given class

**Lock:** Alternative limits are blocked by attractor basin

## GMT Statement

**Theorem (Unique-Attractor Lock).** For gradient flow on $\mathbf{I}_k(M)$:

1. **Unique Attractor:** If $T_\infty$ is the unique critical point with $\Phi(T_\infty) \leq \Phi_0$ and $[T_\infty] = \alpha$

2. **Basin:** All $T$ with $\Phi(T) \leq \Phi_0$ and $[T] = \alpha$ converge to $T_\infty$

3. **Lock:** No other limiting behavior exists in this class

## Proof Sketch

### Step 1: Critical Points and Attractors

**Critical Point:** $T_* \in \mathbf{I}_k(M)$ is critical for $\Phi$ if:
$$\delta\Phi(T_*) = 0$$

**Stable Critical Point:** If Hessian $D^2\Phi(T_*)$ is positive definite.

**Reference:** Milnor, J. (1963). *Morse Theory*. Princeton University Press.

### Step 2: Łojasiewicz-Simon Convergence

**Theorem (Simon, 1983):** If $\Phi$ is analytic and $T_t$ is gradient flow near critical $T_*$:
$$\|T_t - T_*\| \leq C t^{-\beta}$$

for some $\beta > 0$.

**Reference:** Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations. *Ann. of Math.*, 118, 525-571.

**Consequence:** Gradient flows converge to critical points.

### Step 3: Uniqueness Condition

**Assumption:** In homology class $\alpha$ with energy bound $\Phi_0$:
$$\{T : T \text{ critical}, [T] = \alpha, \Phi(T) \leq \Phi_0\} = \{T_\infty\}$$

**Isolation:** $T_\infty$ is the unique minimizer in its class.

### Step 4: Sublevel Set Structure

**Sublevel Set:** $\Phi^{\leq c} = \{T \in \mathbf{I}_k(M) : \Phi(T) \leq c\}$

**Connected Component:** $\mathcal{C}_\alpha(c) = $ connected component of $\Phi^{\leq c}$ containing class $\alpha$

**Contraction:** Flow contracts $\mathcal{C}_\alpha(c)$ toward critical set.

### Step 5: Conley Index Theory

**Conley Index:** For isolated invariant set $S$:
$$h(S) = [N/L]$$

where $(N, L)$ is index pair.

**Reference:** Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. AMS.

**Unique Attractor:** If $h(T_\infty) = \Sigma^0$ (0-sphere), then $T_\infty$ attracts.

### Step 6: Basin of Attraction

**Definition:** Basin of attraction:
$$\mathcal{B}(T_\infty) = \{T : \omega(T) = T_\infty\}$$

where $\omega(T)$ is the $\omega$-limit set.

**Theorem:** If $T_\infty$ unique critical in class, then:
$$\mathcal{B}(T_\infty) \supset \{T : [T] = \alpha, \Phi(T) \leq \Phi_0\}$$

### Step 7: No Alternative Limits

**$\omega$-Limit Analysis:** For any trajectory $T_t$:
- $\omega(T)$ is non-empty, compact, connected
- $\omega(T)$ consists of critical points (gradient flow)
- By uniqueness, $\omega(T) = \{T_\infty\}$

**Reference:** Hirsch, M., Smale, S., Devaney, R. (2012). *Differential Equations, Dynamical Systems, and an Introduction to Chaos*. Academic Press.

### Step 8: Stability Under Perturbation

**Structural Stability:** Unique attractor persists under small perturbation:

If $\Phi_\epsilon = \Phi + \epsilon \Psi$ with $\|\Psi\|$ small, then $\Phi_\epsilon$ has unique attractor $T_\infty^\epsilon$ near $T_\infty$.

**Reference:** Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.

### Step 9: Lyapunov Function

**Strict Lyapunov:** $\Phi$ is strict Lyapunov for flow toward $T_\infty$:
$$\frac{d}{dt}\Phi(T_t) = -\|\nabla\Phi(T_t)\|^2 < 0$$

unless $T_t = T_\infty$.

**Consequence:** $\Phi$ decreases strictly along non-stationary trajectories.

### Step 10: Compilation Theorem

**Theorem (Unique-Attractor Lock):**

1. **Unique Critical:** $T_\infty$ sole critical point in class with bounded energy

2. **Global Basin:** All trajectories in class converge to $T_\infty$

3. **Lock:** Alternative limiting behaviors blocked

4. **Stability:** Uniqueness persists under perturbation

**Applications:**
- Guaranteed convergence of variational sequences
- Selection principle for minimizers
- Obstruction to bifurcation

## Key GMT Inequalities Used

1. **Łojasiewicz-Simon:**
   $$\|T_t - T_*\| \leq C t^{-\beta}$$

2. **Lyapunov Decay:**
   $$\frac{d}{dt}\Phi(T_t) = -\|\nabla\Phi\|^2$$

3. **Basin Containment:**
   $$\{[T] = \alpha, \Phi(T) \leq \Phi_0\} \subset \mathcal{B}(T_\infty)$$

4. **Conley Index:**
   $$h(T_\infty) = \Sigma^0 \implies \text{attractor}$$

## Literature References

- Milnor, J. (1963). *Morse Theory*. Princeton.
- Simon, L. (1983). Asymptotics for nonlinear evolution. *Ann. of Math.*, 118.
- Conley, C. (1978). *Isolated Invariant Sets*. AMS.
- Hirsch, M., Smale, S., Devaney, R. (2012). *Differential Equations and Dynamical Systems*. Academic Press.
- Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
