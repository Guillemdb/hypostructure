# FACT-Gate: Gate Evaluator Factory — GMT Translation

## Original Statement (Hypostructure)

The gate evaluator factory constructs decision procedures that test whether flow can pass through a given configuration, based on local geometric/analytic conditions.

## GMT Setting

**Gate:** $G \subset \mathbf{I}_k(M)$ — subset of currents through which flow must pass

**Gate Evaluator:** $\text{eval}_G: \mathbf{I}_k(M) \to \{\text{PASS}, \text{BLOCK}, \text{UNKNOWN}\}$

**Factory:** Constructs $\text{eval}_G$ from geometric data

## GMT Statement

**Theorem (Gate Evaluator Factory).** There exists a factory $\mathcal{F}_{\text{gate}}$ that, given:
- Gate specification $G = \{T : P(T)\}$ for decidable predicate $P$
- Soft certificates $K^+$

produces a gate evaluator $\text{eval}_G$ with:

1. **Soundness:** $\text{eval}_G(T) = \text{PASS} \implies T \in G$

2. **Completeness:** $T \in G$ with margin $\delta \implies \text{eval}_G(T) = \text{PASS}$

3. **Efficiency:** Evaluation uses $O(|\text{mesh}|^k)$ operations

## Proof Sketch

### Step 1: Gate Predicates

**Geometric Predicates:** Common gate conditions:
1. **Mass bound:** $\mathbf{M}(T) \leq \Lambda$
2. **Density bound:** $\Theta_k(T, x) \leq \theta_0$ for all $x$
3. **Curvature bound:** $\|A\|_{L^\infty} \leq \kappa_0$
4. **Regularity:** $\text{sing}(T) = \emptyset$
5. **Topological:** $H_*(T) \cong H_*(T_0)$ (fixed homology)

**Decidability:** Each predicate is decidable given sufficient approximation.

### Step 2: Evaluation via Discretization

**Mesh Approximation:** Given current $T$, construct discrete approximation $T_h$ with:
$$d_{\text{flat}}(T, T_h) \leq C h^{\alpha}$$

**Predicate Evaluation on Mesh:**
$$P_h(T_h) \approx P(T) \text{ with error } O(h^\beta)$$

**Reference:** Brakke, K. (1992). *Surface Evolver*. Software and documentation.

### Step 3: Mass Gate

**Gate:** $G_{\Lambda} := \{T : \mathbf{M}(T) \leq \Lambda\}$

**Evaluator:**
```
eval_mass(T):
    m = compute_mass(T)
    if m < Λ - ε: return PASS
    if m > Λ + ε: return BLOCK
    return UNKNOWN
```

**Soundness:** Numerical mass computation with error $\pm \varepsilon$.

### Step 4: Regularity Gate

**Gate:** $G_{\text{reg}} := \{T : T \text{ is regular}\}$

**ε-Regularity Test (Allard, 1972):** If:
$$\int_{B_r(x)} |A|^2 \, d\|T\| < \varepsilon_0$$

then $T$ is regular in $B_{r/2}(x)$.

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95, 417-491.

**Evaluator:**
```
eval_regularity(T):
    for each ball B_r(x) covering spt(T):
        if tilt_excess(T, B_r(x)) > ε_0:
            return BLOCK
    return PASS
```

### Step 5: Density Gate

**Gate:** $G_{\theta} := \{T : \Theta_k(T, x) \leq \theta_0 \text{ for all } x\}$

**Monotonicity Formula (Almgren, 1979):**
$$\frac{d}{dr} \frac{\mathbf{M}(T \cap B_r(x))}{r^k} \geq 0$$

**Reference:** Almgren, F. J. (1979). Dirichlet's problem for multiple valued functions. *Arch. Rational Mech. Anal.*, 72, 275-369.

**Evaluator:**
```
eval_density(T, θ_0):
    for each x in grid:
        θ = lim_{r→0} M(T ∩ B_r(x)) / (ω_k r^k)
        if θ > θ_0 + ε: return BLOCK
    return PASS
```

### Step 6: Topological Gate

**Gate:** $G_{H_*} := \{T : H_*(T) \cong H_0\}$ for fixed $H_0$

**Homology Computation (Edelsbrunner-Harer, 2010):** Compute persistent homology from discrete approximation.

**Reference:** Edelsbrunner, H., Harer, J. (2010). *Computational Topology*. AMS.

**Evaluator:**
```
eval_homology(T, H_0):
    H = compute_homology(T)
    if H ≅ H_0: return PASS
    if H ≇ H_0 robustly: return BLOCK
    return UNKNOWN
```

### Step 7: Composite Gates

**Conjunction:** $G_1 \cap G_2$ evaluated by:
```
eval_and(T, G_1, G_2):
    r_1 = eval_{G_1}(T)
    r_2 = eval_{G_2}(T)
    if r_1 = BLOCK or r_2 = BLOCK: return BLOCK
    if r_1 = PASS and r_2 = PASS: return PASS
    return UNKNOWN
```

**Disjunction:** $G_1 \cup G_2$ evaluated by:
```
eval_or(T, G_1, G_2):
    r_1 = eval_{G_1}(T)
    r_2 = eval_{G_2}(T)
    if r_1 = PASS or r_2 = PASS: return PASS
    if r_1 = BLOCK and r_2 = BLOCK: return BLOCK
    return UNKNOWN
```

### Step 8: Factory Construction

**Factory Algorithm:**

```
GateFactory(specification):
    Parse specification into predicate tree P
    For each atomic predicate p in P:
        Select evaluator eval_p from library
    Compose evaluators according to tree structure
    Return composite evaluator eval_P
```

**Library of Atomic Evaluators:**
- `eval_mass`: Mass bound check
- `eval_density`: Density bound check
- `eval_regularity`: ε-regularity check
- `eval_curvature`: Second fundamental form bound
- `eval_homology`: Topological invariant check
- `eval_boundary`: Boundary matching check

### Step 9: Error Analysis

**False Positives:** $\text{eval}(T) = \text{PASS}$ but $T \notin G$

**False Negatives:** $\text{eval}(T) = \text{BLOCK}$ but $T \in G$

**Theorem:** With mesh size $h$ and margin $\delta$:
- False positive rate: $O(h^\alpha / \delta)$
- False negative rate: $O(h^\alpha / \delta)$

**Robust Evaluation:** Use margins: if $P(T)$ with margin $\delta$, then $\text{eval}(T) = \text{PASS}$ with high confidence.

### Step 10: Compilation Theorem

**Theorem (Gate Factory):** The factory $\mathcal{F}_{\text{gate}}$:

1. **Inputs:** Gate specification $(P, \text{params})$
2. **Outputs:** Evaluator $\text{eval}_G$
3. **Guarantees:**
   - Sound: no false positives with high probability
   - Complete: no false negatives for $\delta$-interior points
   - Efficient: polynomial in mesh complexity

**Constructive Content:**
- Given specification, produce executable evaluator
- Evaluator can be run on any approximation $T_h$
- Error bounds computable from mesh parameters

## Key GMT Inequalities Used

1. **Mass Computation:**
   $$\mathbf{M}(T) = \int_T 1 \, d\|T\|$$

2. **ε-Regularity:**
   $$\int_{B_r} |A|^2 < \varepsilon_0 \implies \text{regular in } B_{r/2}$$

3. **Monotonicity:**
   $$r \mapsto \frac{\mathbf{M}(T \cap B_r)}{r^k} \text{ non-decreasing}$$

4. **Approximation:**
   $$d_{\text{flat}}(T, T_h) \leq C h^\alpha$$

## Literature References

- Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.
- Almgren, F. J. (1979). Dirichlet's problem for multiple valued functions. *Arch. Rational Mech. Anal.*, 72.
- Brakke, K. (1992). *Surface Evolver*. Experimental Mathematics.
- Edelsbrunner, H., Harer, J. (2010). *Computational Topology*. AMS.
- Sullivan, J. M. (1990). A crystalline approximation theorem for hypersurfaces. *Princeton PhD thesis*.
