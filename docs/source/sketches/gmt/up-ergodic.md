# UP-Ergodic: Ergodic-Sat Theorem — GMT Translation

## Original Statement (Hypostructure)

The ergodic-saturation theorem shows that time averages equal space averages for ergodic flows, allowing statistical characterization of long-time behavior.

## GMT Setting

**Ergodicity:** Flow visits all accessible states with correct frequency

**Time Average:** $\bar{f} = \lim_{T \to \infty} \frac{1}{T} \int_0^T f(\varphi_t(x)) \, dt$

**Space Average:** $\langle f \rangle = \int f \, d\mu$ for invariant measure $\mu$

## GMT Statement

**Theorem (Ergodic-Saturation).** If the flow $\varphi_t$ is ergodic with respect to invariant measure $\mu$:

1. **Birkhoff:** Time average = space average a.e.:
$$\lim_{T \to \infty} \frac{1}{T} \int_0^T f(\varphi_t(x)) \, dt = \int f \, d\mu$$

2. **Saturation:** The measure $\mu$ is saturated (supported on attractor)

3. **Uniqueness:** Ergodic measure is unique (if exists)

## Proof Sketch

### Step 1: Invariant Measure

**Definition:** Measure $\mu$ is **invariant** under $\varphi_t$ if:
$$\mu(\varphi_t^{-1}(A)) = \mu(A) \quad \forall A, t$$

**Existence (Krylov-Bogoliubov):** For continuous flow on compact space, invariant measures exist.

**Reference:** Krylov, N., Bogoliubov, N. (1937). La théorie générale de la mesure dans son application à l'étude des systèmes dynamiques de la mécanique non linéaire. *Ann. of Math.*, 38, 65-113.

### Step 2: Ergodicity

**Definition:** $(X, \varphi_t, \mu)$ is **ergodic** if:
$$\varphi_t^{-1}(A) = A \implies \mu(A) \in \{0, 1\}$$

(only trivial invariant sets).

**Equivalently:** $\mu$ cannot be written as non-trivial convex combination of invariant measures.

### Step 3: Birkhoff Ergodic Theorem

**Theorem (Birkhoff, 1931):** For $f \in L^1(\mu)$:
$$\bar{f}(x) := \lim_{T \to \infty} \frac{1}{T} \int_0^T f(\varphi_t(x)) \, dt$$

exists $\mu$-a.e. and:
$$\int \bar{f} \, d\mu = \int f \, d\mu$$

If ergodic: $\bar{f} = \int f \, d\mu$ a.e. (constant).

**Reference:** Birkhoff, G. D. (1931). Proof of the ergodic theorem. *Proc. Natl. Acad. Sci.*, 17, 656-660.

### Step 4: Physical Interpretation

**Time Average:** Long-time measurement of observable $f$ along trajectory.

**Space Average:** Ensemble average over all states with weight $\mu$.

**Ergodic Hypothesis:** For ergodic systems, single trajectory samples entire space.

### Step 5: Saturation on Attractor

**Support of $\mu$:** $\text{supp}(\mu) = \overline{\{x : \mu(U) > 0 \text{ for all } U \ni x\}}$

**Attractor Saturation:** For dissipative systems:
$$\text{supp}(\mu) \subset \mathcal{A}$$

(invariant measure concentrated on attractor).

**Reference:** Ruelle, D. (1989). *Elements of Differentiable Dynamics and Bifurcation Theory*. Academic Press.

### Step 6: SRB Measures

**Sinai-Ruelle-Bowen Measure:** For hyperbolic attractors:
$$\mu_{\text{SRB}} = \lim_{T \to \infty} \frac{1}{T} \int_0^T \delta_{\varphi_t(x)} \, dt$$

for Lebesgue-a.e. $x$ in basin of attraction.

**Reference:** Young, L.-S. (2002). What are SRB measures, and which dynamical systems have them? *J. Stat. Phys.*, 108, 733-754.

### Step 7: Gradient Flow Ergodicity

**Gradient Flows:** For $\partial_t T = -\nabla \Phi(T)$:
- No periodic orbits (energy strictly decreases)
- Trajectories converge to equilibria
- Ergodic measure = weighted sum of Dirac masses on equilibria

**Consequence:** For gradient flows, ergodicity is trivial (converges to critical point).

### Step 8: Mixing and Decay

**Mixing:** $(X, \varphi_t, \mu)$ is **mixing** if:
$$\lim_{t \to \infty} \mu(A \cap \varphi_t^{-1}(B)) = \mu(A) \mu(B)$$

**Correlation Decay:** For observables $f, g$:
$$C_{fg}(t) := \int f \cdot (g \circ \varphi_t) \, d\mu - \int f \, d\mu \int g \, d\mu \to 0$$

**Rate:** Exponential mixing gives $|C_{fg}(t)| \leq C e^{-\lambda t}$.

### Step 9: Applications in GMT

**Averaging for Currents:** For current-valued observable:
$$\bar{T} = \lim_{T \to \infty} \frac{1}{T} \int_0^T T_t \, dt$$

**Ergodic Current:** $\bar{T}$ is stationary if flow is ergodic.

**Statistical Equilibrium:** Long-time average gives effective equilibrium state.

### Step 10: Compilation Theorem

**Theorem (Ergodic-Saturation):**

1. **Birkhoff:** Time averages = space averages

2. **Saturation:** Measure supported on attractor

3. **Uniqueness:** Ergodic measure unique in each ergodic component

4. **Statistical:** Long-time behavior characterized by invariant measure

**Applications:**
- Statistical mechanics of geometric flows
- Long-time averages of observables
- Effective equilibrium states

## Key GMT Inequalities Used

1. **Birkhoff:**
   $$\bar{f} = \lim \frac{1}{T}\int_0^T f(\varphi_t(x)) \, dt = \int f \, d\mu$$

2. **Support:**
   $$\text{supp}(\mu) \subset \mathcal{A}$$

3. **Mixing Decay:**
   $$|C_{fg}(t)| \to 0$$

4. **Gradient Convergence:**
   $$T_t \to T_*$$ (for gradient flows)

## Literature References

- Birkhoff, G. D. (1931). Proof of the ergodic theorem. *Proc. Natl. Acad. Sci.*, 17.
- Krylov, N., Bogoliubov, N. (1937). La théorie générale de la mesure. *Ann. of Math.*, 38.
- Ruelle, D. (1989). *Elements of Differentiable Dynamics*. Academic Press.
- Young, L.-S. (2002). What are SRB measures? *J. Stat. Phys.*, 108.
- Walters, P. (1982). *An Introduction to Ergodic Theory*. Springer.
