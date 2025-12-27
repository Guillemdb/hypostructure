# UP-Surgery: Surgery Promotion — GMT Translation

## Original Statement (Hypostructure)

The surgery promotion shows that local surgical repairs extend to global resolution, with each surgery improving regularity.

## GMT Setting

**Local Surgery:** Repair singularity in $B_\varepsilon(\Sigma)$

**Global Surgery:** All singularities repaired systematically

**Promotion:** Local repairs combine to global resolution

## GMT Statement

**Theorem (Surgery Promotion).** If local surgery is admissible at each singular point:

1. **Finite Surgery Count:** Total surgeries bounded: $N_{\text{surg}} \leq \Lambda/\epsilon_T$

2. **Global Resolution:** Combined surgeries resolve all singularities

3. **Topology Control:** Surgery preserves essential topology

## Proof Sketch

### Step 1: Surgery Separation

**Separation Condition:** Singularities $\{x_1, \ldots, x_m\}$ satisfy:
$$d(x_i, x_j) > 2\varepsilon_{\text{surg}}$$

for $i \neq j$, where $\varepsilon_{\text{surg}}$ is surgery scale.

**Simultaneous Surgery:** When separated, surgeries at different points can be performed simultaneously:
$$T' = \bigcap_i \mathcal{O}_{S_i}(T)$$

**Reference:** Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.

### Step 2: Energy Drop Per Surgery

**Minimum Drop (Hamilton, 1997):**
$$\Phi(T) - \Phi(T') \geq \epsilon_T = c(n) \cdot r_{\text{surg}}^n$$

where $r_{\text{surg}}$ is surgery radius.

**Reference:** Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.

**Finite Surgery Count:**
$$N_{\text{surg}} \leq \frac{\Phi(T_0) - \Phi_{\min}}{\epsilon_T}$$

### Step 3: Connected Sum Decomposition

**Topological Effect:** Each surgery performs a connected sum:
$$M_{\text{after}} = (M_{\text{before}} \setminus B_\varepsilon(\Sigma)) \cup_{\partial} V$$

**Connected Sum:** If $V = S^{n-1} \times B^1$:
$$M_{\text{after}} \cong M_1 \# M_2$$

(manifold splits).

**Reference:** Milnor, J. (1965). *Topology from the Differentiable Viewpoint*. Princeton.

### Step 4: Inductive Resolution

**Resolution by Induction on Complexity:**

**Base Case:** Manifolds with no singularities — done.

**Inductive Step:** If $M$ has singularities:
1. Perform surgery at all singularities
2. Resulting pieces have lower complexity
3. Apply induction to each piece

**Complexity:** $\chi(M) + \text{Vol}(M)$ or similar invariant.

### Step 5: No Accumulation

**Zeno Prevention:** Surgery times $\{t_i\}$ cannot accumulate:
$$\inf_i (t_{i+1} - t_i) \geq t_{\min} > 0$$

*Proof:* After surgery:
1. Curvature is bounded by $C/r_{\text{surg}}^2$
2. Time to next surgery $\geq c \cdot r_{\text{surg}}^2$
3. This is positive

### Step 6: Canonical Neighborhood Theorem

**Theorem (Perelman, 2003):** Near high-curvature points, the geometry is close to canonical:
$$|Rm| \geq r^{-2} \implies B_r(x) \approx \text{canonical neighborhood}$$

**Canonical Neighborhoods:**
1. Round neck: $S^{n-1} \times I$
2. Cap: hemisphere
3. Round component: $S^n / \Gamma$

**Reference:** Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.

**Consequence:** All surgeries are of standard type.

### Step 7: Surgery Compatibility

**Global Compatibility:** Surgeries at different locations must be compatible:
- Scales $\varepsilon_i$ satisfy separation
- Profiles $V_i$ are from same library $\mathcal{L}$
- Orientations match

**Theorem:** Under soft permits, surgeries are always compatible.

*Proof:* Scale coherence ($K_{\text{SC}_\lambda}^+$) ensures profiles are from library. Separation is ensured by curvature threshold.

### Step 8: Persistence of Permits

**Theorem:** Soft permits persist through surgery:
$$K^+(T) \implies K^+(T')$$

for each permit type.

*Proof by type:*
- $K_{D_E}^+$: Energy decreases
- $K_{C_\mu}^+$: Mass decreases
- $K_{\text{SC}_\lambda}^+$: Scale structure preserved
- $K_{\text{LS}_\sigma}^+$: Łojasiewicz exponent unchanged or improved

### Step 9: Terminal State

**Theorem:** After finitely many surgeries, one of:
1. **Extinction:** $M$ disappears (e.g., shrinks to point)
2. **Equilibrium:** $M$ converges to steady state
3. **Classification:** $M$ decomposes into classified pieces

**Geometrization (Perelman, 2003):** For 3-manifolds:
$$M = (\text{hyperbolic}) \# (\text{spherical}) \# (\text{Seifert fibered}) \# \cdots$$

**Reference:** Perelman, G. (2003). Finite extinction time. arXiv:math/0307245.

### Step 10: Compilation Theorem

**Theorem (Surgery Promotion):**

1. **Local-to-Global:** Local surgeries combine to global resolution

2. **Finite Count:** At most $\Lambda/\epsilon_T$ surgeries

3. **Topology:** Connected sum decomposition

4. **Terminal:** Resolution terminates with classified state

**Applications:**
- Geometrization of 3-manifolds
- Classification of MCF singularities
- General singularity resolution

## Key GMT Inequalities Used

1. **Energy Drop:**
   $$\Phi(T) - \Phi(T') \geq \epsilon_T$$

2. **Surgery Count:**
   $$N_{\text{surg}} \leq \Lambda/\epsilon_T$$

3. **Separation:**
   $$d(x_i, x_j) > 2\varepsilon_{\text{surg}}$$

4. **Canonical Neighborhood:**
   $$|Rm| \geq r^{-2} \implies B_r \approx \text{standard}$$

## Literature References

- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Perelman, G. (2003). Finite extinction time. arXiv:math/0307245.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12.
