# UP-TypeII: Type II Suppression — GMT Translation

## Original Statement (Hypostructure)

Type II singularities (slow-forming, non-self-similar) are suppressed under soft permits, leaving only Type I (self-similar, controlled) singularities.

## GMT Setting

**Type I:** $|A|(x, t) \leq C(T - t)^{-1/2}$ — self-similar blowup rate

**Type II:** $\limsup_{t \to T} (T-t)^{1/2} |A|(x, t) = \infty$ — faster than self-similar

**Suppression:** Type II cannot occur under soft permits

## GMT Statement

**Theorem (Type II Suppression).** Under soft permits $K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+$:

1. **Type I Bound:** All singularities satisfy Type I rate:
$$|A|(x, t) \leq C(T - t)^{-1/2}$$

2. **Self-Similarity:** Blow-up limits are self-similar solutions

3. **Type II Exclusion:** No Type II singularities form

## Proof Sketch

### Step 1: Singularity Classification

**Type I Singularity:** At singular time $T$:
$$\limsup_{t \to T} (T - t)^{1/2} \sup_M |A|(\cdot, t) < \infty$$

**Type II Singularity:**
$$\limsup_{t \to T} (T - t)^{1/2} \sup_M |A|(\cdot, t) = \infty$$

**Reference:** Hamilton, R. S. (1995). Formation of singularities in the Ricci flow. *Surveys in Diff. Geom.*, 2, 7-136.

### Step 2: Huisken's Monotonicity

**Monotonicity Formula (Huisken, 1990):** For MCF:
$$\frac{d}{dt} \int_M \Phi_{(x_0, t_0)}(x, t) \, d\mu_t = -\int_M \left|H + \frac{(x - x_0)^\perp}{2(t_0 - t)}\right|^2 \Phi \, d\mu_t$$

where $\Phi_{(x_0, t_0)}(x, t) = (4\pi(t_0 - t))^{-n/2} e^{-|x-x_0|^2/4(t_0-t)}$.

**Reference:** Huisken, G. (1990). Asymptotic behavior for singularities of MCF. *J. Diff. Geom.*, 31, 285-299.

**Consequence:** Density $\Theta(x_0, t_0) = \lim_{t \to t_0^-} \int_M \Phi_{(x_0, t_0)} \, d\mu_t$ exists.

### Step 3: Density Lower Bound

**Type I Implies Density Bound:** For Type I singularities:
$$\Theta(x_0, T) \geq \Theta_0 > 0$$

(density is uniformly positive at singular points).

**Type II Density:** Type II singularities can have:
$$\Theta(x_0, T) = 0 \text{ or undefined}$$

### Step 4: Scale Coherence Excludes Type II

**Use of $K_{\text{SC}_\lambda}^+$:** Scale coherence requires blow-up limits to be cones.

**Type I Blow-Up:** Rescaling $\lambda_j = (T - t_j)^{-1/2}$ gives:
$$M_j := \lambda_j (M_{t_j} - x_0) \to C_\infty$$

where $C_\infty$ is a self-similar solution.

**Type II Blow-Up:** Rescaling at Type II rate gives:
$$\lambda_j = |A|_{\max}(t_j)$$

which grows faster than $(T-t_j)^{-1/2}$. The limit is **not** scale-coherent.

### Step 5: Łojasiewicz Argument

**Use of $K_{\text{LS}_\sigma}^+$:** Near singularity, Łojasiewicz inequality implies:
$$\int_t^T |\nabla \Phi| \, ds \leq C |\Phi(t) - \Phi(T)|^\theta$$

**Type I Consequence:** Energy approach is power-law:
$$\Phi(T) - \Phi(t) \sim (T - t)^\alpha$$

**Type II Impossibility:** Type II would require:
$$\Phi(T) - \Phi(t) \sim e^{-c/(T-t)}$$

or similar, violating Łojasiewicz.

### Step 6: Compactness Argument

**Use of $K_{C_\mu}^+$:** Type I sequences are precompact:
$$\{(T - t_j)^{1/2} \cdot M_{t_j}\}_j \text{ has convergent subsequence}$$

**Type II Failure:** Type II sequences have:
$$|A|_{\max} (T - t_j)^{1/2} \to \infty$$

which violates compactness of blow-up sequences.

### Step 7: Examples

**Type I Examples:**
- Round shrinking sphere in MCF
- Round shrinking soliton in Ricci flow
- Smooth extinction

**Type II Examples (Excluded):**
- Degenerate neckpinch (Angenent-Velázquez)
- Pancake singularity

**Reference:** Angenent, S. B., Velázquez, J. J. L. (1997). Degenerate neckpinches in mean curvature flow. *J. reine angew. Math.*, 482, 15-66.

### Step 8: Quantitative Type I Bound

**Theorem:** Under soft permits, the Type I constant is bounded:
$$\sup_{(x, t)} (T - t)^{1/2} |A|(x, t) \leq C(\Lambda, n, \theta)$$

where $\Lambda$ is energy bound, $n$ is dimension, $\theta$ is Łojasiewicz exponent.

*Proof:* Combine:
1. Energy bound $\Phi \leq \Lambda$
2. Monotonicity formula
3. Łojasiewicz decay rate

### Step 9: Blow-Up Limit Classification

**Self-Similar Solutions:** Type I blow-up limits are:
- Shrinking solitons: $\partial_t M = H$ with $H = -\frac{x^\perp}{2(T-t)}$
- Classified: spheres, cylinders (Colding-Minicozzi)

**Reference:** Colding, T., Minicozzi, W. (2012). Generic mean curvature flow I: Generic singularities. *Ann. of Math.*, 175, 755-833.

**Consequence:** Only finitely many blow-up types under Type I.

### Step 10: Compilation Theorem

**Theorem (Type II Suppression):**

1. **Type I Only:** All singularities are Type I under soft permits

2. **Self-Similar:** Blow-up limits are self-similar solutions

3. **Quantitative:** Type I constant bounded by $C(\Lambda, n, \theta)$

4. **Classification:** Blow-up limits belong to finite list

**Applications:**
- MCF surgery: only Type I singularities need handling
- Ricci flow: ancient solutions classified (Perelman)
- Regularity: Type I implies better control

## Key GMT Inequalities Used

1. **Huisken Monotonicity:**
   $$\frac{d}{dt} \int \Phi \, d\mu \leq 0$$

2. **Type I Rate:**
   $$|A| \leq C(T - t)^{-1/2}$$

3. **Łojasiewicz Decay:**
   $$\Phi(T) - \Phi(t) \sim (T - t)^\alpha$$

4. **Scale Coherence:**
   $$(\eta_{0,\lambda})_\# C = C$$

## Literature References

- Hamilton, R. S. (1995). Formation of singularities in Ricci flow. *Surveys in Diff. Geom.*, 2.
- Huisken, G. (1990). Asymptotic behavior for MCF singularities. *J. Diff. Geom.*, 31.
- Colding, T., Minicozzi, W. (2012). Generic mean curvature flow. *Ann. of Math.*, 175.
- Angenent, S. B., Velázquez, J. J. L. (1997). Degenerate neckpinches. *J. reine angew. Math.*, 482.
- Perelman, G. (2002-2003). Ricci flow papers. arXiv.
