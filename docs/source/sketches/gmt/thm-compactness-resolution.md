# THM-CompactnessResolution: Compactness Resolution — GMT Translation

## Original Statement (Hypostructure)

Every bounded sequence in the state space has a convergent subsequence, and the limit admits a complete resolution into profiles plus remainder.

## GMT Setting

**State Space:** $(\mathbf{I}_k(M), d_{\text{flat}})$ — integral currents with flat metric

**Bounded Sequence:** $\{T_j\} \subset \mathbf{I}_k(M)$ with $\sup_j \mathbf{M}(T_j) \leq \Lambda$

**Resolution:** Decomposition into profiles and remainder

## GMT Statement

**Theorem (Compactness Resolution).** Let $\{T_j\} \subset \mathbf{I}_k(M)$ satisfy:
$$\sup_j (\mathbf{M}(T_j) + \mathbf{M}(\partial T_j)) \leq \Lambda$$

Then:

1. **Compactness:** There exists subsequence $T_{j_l} \to T_\infty$ in flat norm

2. **Profile Resolution:**
$$T_j = \sum_{i=1}^N V^i_j + w_j$$

with profiles $V^i$ and vanishing remainder $w_j$

3. **Mass Identity:**
$$\lim_{j \to \infty} \mathbf{M}(T_j) = \sum_{i=1}^N \mathbf{M}(V^i) + \lim_{j \to \infty} \mathbf{M}(w_j)$$

## Proof Sketch

### Step 1: Federer-Fleming Compactness

**Theorem (Federer-Fleming, 1960):** If $\{T_j\} \subset \mathbf{I}_k(M)$ satisfies:
$$\sup_j (\mathbf{M}(T_j) + \mathbf{M}(\partial T_j)) \leq \Lambda$$

then there exists a subsequence converging in flat norm:
$$T_{j_l} \to T_\infty \in \mathbf{I}_k(M)$$

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

**Proof Sketch:** The flat norm $\mathbb{F}(T) = \inf_{R} (\mathbf{M}(T - \partial R) + \mathbf{M}(R))$ metrizes weak convergence. The mass bound provides precompactness.

### Step 2: Concentration Analysis

**Defect Measure (Lions, 1984):** For weakly converging $T_j \rightharpoonup T_\infty$, the defect is:
$$\nu := \lim_{j \to \infty} \|T_j\| - \|T_\infty\|$$

as measures (in weak-* topology).

**Reference:** Lions, P.-L. (1984). The concentration-compactness principle. *Ann. Inst. H. Poincaré*, 1.

**Concentration Points:** $\nu$ is supported on the concentration set:
$$\Sigma := \{x : \nu(\{x\}) > 0\}$$

### Step 3: Profile Extraction

**Iterative Extraction (Bahouri-Gérard, 1999):**

```
R_j^0 = T_j, n = 0
While defect measure ν^n ≠ 0:
    n = n + 1
    Find concentration point x^n ∈ supp(ν^{n-1})
    Find concentration scale λ_j^n → 0
    Extract profile: V^n = lim_{j→∞} (η_{x^n, λ_j^n})_# R_j^{n-1}
    Update remainder: R_j^n = R_j^{n-1} - (η_{x^n, λ_j^n})^{-1}_# V^n
```

**Reference:** Bahouri, H., Gérard, P. (1999). High frequency approximation. *Amer. J. Math.*, 121, 131-175.

### Step 4: Orthogonality of Profiles

**Scale Separation:** Profiles concentrate at asymptotically separated scales:
$$\frac{\lambda_j^i}{\lambda_j^m} + \frac{\lambda_j^m}{\lambda_j^i} + \frac{|x_j^i - x_j^m|^2}{\lambda_j^i \lambda_j^m} \to \infty$$

for $i \neq m$.

**Mass Orthogonality:**
$$\lim_{j \to \infty} \mathbf{M}(T_j) = \sum_{i=1}^N \mathbf{M}(V^i) + \lim_{j \to \infty} \mathbf{M}(w_j)$$

### Step 5: Remainder Vanishing

**Critical Norm Decay:** The remainder $w_j$ satisfies:
$$\|w_j\|_{L^{p^*}} \to 0 \quad \text{as } N \to \infty$$

where $p^* = nk/(n-k)$ is critical Sobolev exponent.

**Proof:** Each profile extraction removes a quantum of critical norm:
$$\|R_j^{n-1}\|_{L^{p^*}} - \|R_j^n\|_{L^{p^*}} \geq c \|V^n\|_{L^{p^*}}$$

### Step 6: Finite Profile Count

**Energy Bound:** Each non-trivial profile has mass $\geq \varepsilon_0 > 0$:
$$V^i \neq 0 \implies \mathbf{M}(V^i) \geq \varepsilon_0$$

**Profile Count:**
$$N \leq \Lambda / \varepsilon_0$$

**Reference:** Struwe, M. (1984). A global compactness result for elliptic boundary value problems. *Math. Z.*, 187, 511-517.

### Step 7: Resolution Uniqueness

**Theorem:** The profile decomposition is unique up to:
1. Reordering of profiles
2. Symmetry transformations within each orbit

*Proof:* The concentration set $\Sigma$ and concentration scales $\{\lambda_j^i\}$ are determined by the sequence. Profiles are uniquely determined as blow-up limits.

### Step 8: Resolution Structure

**Complete Resolution:**
$$T_j = \underbrace{\sum_{i=1}^N V^i_j}_{\text{profiles}} + \underbrace{w_j}_{\text{remainder}}$$

where:
- $V^i_j = (\eta_{x_j^i, \lambda_j^i})^{-1}_\# V^i$ — rescaled profiles
- $w_j$ — remainder with $\|w_j\|_{L^{p^*}} \to 0$

**Reconstruction:** Given profiles and concentration data, reconstruct original sequence.

### Step 9: Stratification of Limit

**Singular Set of Limit:** The limit $T_\infty$ may have singular set:
$$\text{sing}(T_\infty) \subset \Sigma \cup \text{sing}(V^i)$$

**Dimension Bound (Simon, 1983):**
$$\dim_{\mathcal{H}}(\text{sing}(T_\infty)) \leq k - 2$$

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

### Step 10: Compilation Theorem

**Theorem (Compactness Resolution):**

1. **Existence:** Every bounded sequence has convergent subsequence

2. **Profile Resolution:** Limit decomposes into profiles plus remainder

3. **Mass Identity:** Mass splits among profiles and remainder

4. **Finite Complexity:** Profile count bounded by $\Lambda / \varepsilon_0$

**Applications:**
- Blow-up analysis at singularities
- Regularity theory for variational problems
- Well-posedness via compactness methods

## Key GMT Inequalities Used

1. **Federer-Fleming:**
   $$\sup_j \mathbf{M}(T_j) < \infty \implies T_{j_k} \to T_\infty$$

2. **Mass Lower Bound:**
   $$V^i \neq 0 \implies \mathbf{M}(V^i) \geq \varepsilon_0$$

3. **Mass Splitting:**
   $$\mathbf{M}(T_j) = \sum_i \mathbf{M}(V^i_j) + \mathbf{M}(w_j) + o(1)$$

4. **Remainder Decay:**
   $$\|w_j\|_{L^{p^*}} \to 0$$

## Literature References

- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Lions, P.-L. (1984). Concentration-compactness. *Ann. Inst. H. Poincaré*, 1.
- Bahouri, H., Gérard, P. (1999). High frequency approximation. *Amer. J. Math.*, 121.
- Struwe, M. (1984). Global compactness result. *Math. Z.*, 187.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Kenig, C., Merle, F. (2006). Global well-posedness. *Acta Math.*, 201.
