# RESOLVE-AutoSurgery: Automatic Surgery — GMT Translation

## Original Statement (Hypostructure)

Surgery is automatically performed when admissibility is verified. The surgery operator applies the canonical library profile to the singular region, producing a regular extension.

## GMT Setting

**Input Current:** $T \in \mathbf{I}_k(M)$ with singular set $\Sigma = \text{sing}(T)$

**Surgery Operator:** $\mathcal{O}_S: \mathbf{I}_k(M) \times \mathcal{L} \to \mathbf{I}_k(M)$

**Output Current:** $T' = \mathcal{O}_S(T, V)$ — surgically modified current

## GMT Statement

**Theorem (Automatic Surgery).** Given admissible surgery data $(\Sigma, V, \mathcal{O}_S)$ with $V \in \mathcal{L}$, the surgery operator produces:

$$T' = \mathcal{O}_S(T, V) := (T \llcorner (M \setminus B_\varepsilon(\Sigma))) \cup_{\partial B_\varepsilon(\Sigma)} V_\varepsilon$$

where $V_\varepsilon$ is the scaled canonical profile matching boundary data. The result satisfies:

1. **Regularity Improvement:** $\text{sing}(T') \subset \text{sing}(T) \setminus \Sigma$
2. **Mass Control:** $\mathbf{M}(T') \leq \mathbf{M}(T)$
3. **Boundary Preservation:** $\partial T' = \partial T$ on $M \setminus B_\varepsilon(\Sigma)$

## Proof Sketch

### Step 1: Boundary Matching Problem

**Setup:** Given $T$ with singularity at $\Sigma$, we must find $V_\varepsilon$ matching:
$$\partial V_\varepsilon = \langle T, d_\Sigma, \varepsilon \rangle$$

where $d_\Sigma$ is distance to $\Sigma$ and $\langle T, d_\Sigma, \varepsilon \rangle$ is the slice.

**Plateau Problem (Douglas-Rado, 1930s):** Given a Jordan curve $\Gamma$, there exists area-minimizing surface $\Sigma$ with $\partial \Sigma = \Gamma$.

**Reference:** Douglas, J. (1931). Solution of the problem of Plateau. *Trans. AMS*, 33, 263-321.

**GMT Extension (Federer-Fleming, 1960):** For any $(k-1)$-current $S$ with $\partial S = 0$, there exists $T \in \mathbf{I}_k$ with $\partial T = S$ minimizing mass.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

### Step 2: Canonical Profile Insertion

**Library Profile:** For $V \in \mathcal{L}$, the canonical form is:
- Ricci flow: round $S^3$, $S^2 \times \mathbb{R}$, Bryant soliton (Perelman, 2003)
- MCF: $S^n$, $S^k \times \mathbb{R}^{n-k}$ (Huisken-Sinestrari, 2009)
- Minimal surfaces: planes, catenoids, Scherk surfaces

**Reference:**
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175, 137-221.

**Scaling:** The profile $V$ is scaled to match the surgery scale $\varepsilon$:
$$V_\varepsilon(x) = \varepsilon^k V(x/\varepsilon)$$

### Step 3: Gluing Procedure

**Interpolation Region:** Define neck region $N_\varepsilon = B_{2\varepsilon}(\Sigma) \setminus B_\varepsilon(\Sigma)$.

**Cutoff Function:** Let $\chi_\varepsilon$ be smooth with:
- $\chi_\varepsilon = 0$ on $B_\varepsilon(\Sigma)$
- $\chi_\varepsilon = 1$ on $M \setminus B_{2\varepsilon}(\Sigma)$
- $|\nabla \chi_\varepsilon| \leq C/\varepsilon$

**Glued Current (White, 1991):**
$$T' = \chi_\varepsilon \cdot T + (1 - \chi_\varepsilon) \cdot V_\varepsilon$$

interpreted as currents via:
$$T'(\omega) = T(\chi_\varepsilon \omega) + V_\varepsilon((1-\chi_\varepsilon)\omega) + \text{correction}$$

**Reference:** White, B. (1991). Existence of least-energy configurations of immiscible fluids. *J. Geom. Anal.*, 1, 169-192.

### Step 4: Mass Control

**Mass Comparison:** By the canonical profile optimality:
$$\mathbf{M}(V_\varepsilon) \leq \mathbf{M}(T \llcorner B_\varepsilon(\Sigma)) + C \varepsilon^{k+1}$$

**Proof:** The canonical profiles $V \in \mathcal{L}$ are (locally) mass-minimizing among currents with the same boundary. The error term comes from boundary mismatch.

**Hamilton's Energy Drop (1997):** Each surgery decreases total energy:
$$\mathbf{M}(T') \leq \mathbf{M}(T) - c(n) \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

**Reference:** Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5, 1-92.

### Step 5: Regularity Improvement

**Singular Set Reduction:** After surgery:
$$\text{sing}(T') = \text{sing}(T) \setminus \Sigma$$

since:
1. $V_\varepsilon$ is smooth (canonical profile)
2. $T \llcorner (M \setminus B_{2\varepsilon}(\Sigma))$ retains original singularities
3. The gluing region is handled by smooth interpolation

**Allard Regularity (1972):** If $T'$ is close to a plane in $B_r(x)$ with small tilt excess:
$$\int_{B_r(x)} |A|^2 \, d\|T'\| < \varepsilon$$

then $T'$ is regular at $x$.

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95, 417-491.

### Step 6: Uniqueness of Surgery

**Theorem (Surgery Uniqueness):** The surgery outcome $T'$ is unique up to:
1. Choice of scale $\varepsilon$ (within admissible range)
2. Gauge transformations from symmetry group $G$

*Proof:* The canonical profile $V \in \mathcal{L}$ is determined by profile classification (RESOLVE-Profile). The gluing is canonical by mass minimization.

**Kleiner-Lott (2008):** Detailed verification that Perelman's surgery is well-defined and canonical.

**Reference:** Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12, 2587-2855.

### Step 7: Surgery Algorithm

**Automatic Surgery Procedure:**

```
Input: T ∈ I_k(M) with singular set Σ
Output: T' = O_S(T, V)

1. Identify singularities: Σ = sing(T)
2. For each connected component Σ_i of Σ:
   a. Compute blow-up profile: V_i = lim_{λ→0} (η_{x_i,λ})_# T
   b. Classify: V_i ∈ L or V_i ∈ F or V_i wild
   c. If V_i ∈ L: select canonical replacement
   d. If V_i ∈ F: apply equivalence move, then select
   e. If wild: HALT (not admissible)
3. Choose surgery scale ε satisfying:
   - ε < d(Σ_i, Σ_j)/10 for all i ≠ j (separation)
   - ε > ε_min(Λ, n) (non-degeneracy)
4. Perform gluing: T' = (T ∖ ⋃_i B_ε(Σ_i)) ∪ ⋃_i V_{i,ε}
5. Return T'
```

### Step 8: Iteration and Termination

**Finite Surgery Count (Perelman, 2003):**
$$N_{\text{surg}} \leq \frac{\Phi(T_0) - \Phi_{\min}}{\epsilon_T}$$

**Proof:** Each surgery decreases energy by $\geq \epsilon_T$. Total energy drop bounded.

**Termination:** After finitely many surgeries, either:
- All singularities removed: $\text{sing}(T^{(N)}) = \emptyset$
- Remaining singularities non-removable (wild) — contradiction to soft permits

## Key GMT Inequalities Used

1. **Plateau Existence:**
   $$\partial S = 0 \implies \exists T : \partial T = S, \, \mathbf{M}(T) = \min$$

2. **Mass-Minimizing Profile:**
   $$V \in \mathcal{L} \implies \mathbf{M}(V) \leq \mathbf{M}(T) \text{ for } \partial T = \partial V$$

3. **Surgery Energy Drop:**
   $$\mathbf{M}(T') \leq \mathbf{M}(T) - c \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

4. **Surgery Count:**
   $$N_{\text{surg}} \leq \Phi_0 / \epsilon_T$$

## Literature References

- Douglas, J. (1931). Solution of the problem of Plateau. *Trans. AMS*, 33.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12.
