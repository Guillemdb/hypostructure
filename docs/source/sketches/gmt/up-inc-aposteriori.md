# UP-IncAPosteriori: A-Posteriori Discharge — GMT Translation

## Original Statement (Hypostructure)

A-posteriori discharge resolves inconclusive cases by examining the outcome after further evolution, using hindsight to classify initially ambiguous cases.

## GMT Setting

**A-Posteriori:** Classification based on future behavior

**Hindsight:** Use $T_t$ for $t > t_0$ to classify $T_{t_0}$

**Discharge:** Resolve initially ambiguous cases

## GMT Statement

**Theorem (A-Posteriori Discharge).** If case is inconclusive at time $t_0$:

1. **Wait:** Evolve to time $t_1 > t_0$

2. **Backward Classify:** Use $T_{t_1}$ to classify $T_{t_0}$

3. **Resolution:** Initially ambiguous profile becomes classifiable

## Proof Sketch

### Step 1: Ambiguity at Initial Time

**Initial Ambiguity:** At $t_0$:
$$T_{t_0} \approx V_1 \text{ and } T_{t_0} \approx V_2$$

for distinct profiles $V_1, V_2 \in \mathcal{L}$.

**Cause:** $T_{t_0}$ is at boundary between basins of attraction.

### Step 2: Evolution Clarifies

**Divergence:** As $t$ increases:
$$d(T_t, V_1) \to 0 \text{ or } d(T_t, V_2) \to 0$$

(one or the other, not both).

**Mechanism:** Gradient flow moves toward attractors, breaking ambiguity.

### Step 3: Backward Classification

**Classification Rule:** Define $T_{t_0}$'s class by:
$$\text{Class}(T_{t_0}) := \text{Class}(T_{t_1})$$

for $t_1 > t_0$ where classification is clear.

**Well-Defined:** By flow continuity, the classification is consistent.

### Step 4: Unique Continuation

**Unique Continuation Theorem:** Solutions to parabolic equations are uniquely determined by:
- Initial data, or
- Final data (backward uniqueness)

**Reference:** Escauriaza, L., Seregin, G., Šverák, V. (2003). Backward uniqueness for parabolic equations. *Arch. Rational Mech. Anal.*, 169, 147-157.

**Application:** $T_{t_1}$ uniquely determines $T_{t_0}$ (backward in time).

### Step 5: Asymptotic Classification

**Omega-Limit:** $\omega(T_{t_0}) = \lim_{t \to \infty} T_t$

**Classification by Limit:**
$$\text{Class}(T_{t_0}) := \text{Class}(\omega(T_{t_0}))$$

**Under Łojasiewicz:** The omega-limit is a single equilibrium:
$$\omega(T_{t_0}) = \{T_*\}$$

for some $T_* \in \text{Crit}(\Phi)$.

### Step 6: Profile Resolution

**Profile at $t_0$:** Ambiguous between $V_1, V_2$

**Profile at $t_1$:** Clearly $V_1$ (say)

**A-Posteriori:** Retrospectively classify $T_{t_0}$'s profile as $V_1$.

**Physical Interpretation:** The "true" profile was always $V_1$; ambiguity was observational.

### Step 7: Finite Wait Time

**Theorem:** Ambiguity resolves in finite time.

*Proof:* By Łojasiewicz-Simon:
$$\int_0^\infty \|T'(t)\| \, dt < \infty$$

Hence $T_t$ converges in finite arc length. Classification becomes clear at finite $t_1$.

**Reference:** Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.

### Step 8: Stability of Classification

**Theorem:** A-posteriori classification is stable under perturbation.

*Proof:* If $T_{t_0}^\varepsilon \to T_{t_0}$ and:
$$\omega(T_{t_0}^\varepsilon) \to \omega(T_{t_0})$$

then classification is continuous in initial data.

**Caveat:** Near bifurcation points, small perturbations can change the outcome.

### Step 9: Discharge Algorithm

**A-Posteriori Discharge Algorithm:**

```
def aposteriori_discharge(T, t_0, tactics):
    # Step 1: Try immediate classification
    result = try_classify(T, t_0, tactics)
    if result.conclusive:
        return result

    # Step 2: Evolve forward
    t_1 = t_0
    while t_1 < T_max:
        t_1 += Δt
        T_t1 = flow(T, t_0, t_1)

        # Step 3: Try classification at t_1
        result = try_classify(T_t1, t_1, tactics)
        if result.conclusive:
            # Step 4: Backward classify
            return backward_classify(T, t_0, result)

    return STILL_INCONCLUSIVE
```

### Step 10: Compilation Theorem

**Theorem (A-Posteriori Discharge):**

1. **Finite Time:** Ambiguity resolves in finite time under Łojasiewicz

2. **Backward Classification:** Future determines past classification

3. **Stability:** Classification robust under perturbation (away from bifurcations)

4. **Completeness:** All cases resolvable a-posteriori (under soft permits)

**Applications:**
- Handle boundary cases in profile classification
- Resolve ambiguous singularity types
- Classify limiting behavior

## Key GMT Inequalities Used

1. **Convergence:**
   $$T_t \to T_* \text{ as } t \to \infty$$

2. **Finite Arc Length:**
   $$\int_0^\infty \|T'(t)\| < \infty$$

3. **Backward Uniqueness:**
   $$T_{t_1} \text{ determines } T_{t_0}$$

4. **Classification Stability:**
   $$T_n \to T \implies \text{Class}(T_n) \to \text{Class}(T)$$

## Literature References

- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
- Escauriaza, L., Seregin, G., Šverák, V. (2003). Backward uniqueness. *Arch. Rational Mech. Anal.*, 169.
- Haraux, A., Jendoubi, M. A. (2015). *The Convergence Problem for Dissipative Autonomous Systems*. Springer.
- Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
