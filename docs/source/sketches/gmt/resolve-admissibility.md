# RESOLVE-Admissibility: Surgery Admissibility Trichotomy — GMT Translation

## Original Statement (Hypostructure)

Before surgery, profiles must pass admissibility checks: canonical library membership, codimension bounds, and capacity estimates. Three outcomes: Admissible, Admissible-up-to-equivalence, or Not-Admissible.

## GMT Setting

**Surgery Data:** $(\Sigma, V, \mathcal{O}_S)$ — singular set, profile, surgery operator

**Canonicity:** $V \in \mathcal{L}$ — profile in canonical library

**Codimension:** $\text{codim}(\Sigma) := n - \dim_{\mathcal{H}}(\Sigma) \geq 2$

**Capacity:** $\text{Cap}_{1,2}(\Sigma) \leq \varepsilon_{\text{adm}}$ — removability threshold

## GMT Statement

**Theorem (Surgery Admissibility Trichotomy).** For a proposed surgery with data $(\Sigma, V, \mathcal{O}_S)$, exactly one holds:

**Case 1 (Admissible):** All conditions satisfied:
- $V \in \mathcal{L}$ (canonical profile)
- $\text{codim}(\Sigma) \geq 2$
- $\text{Cap}_{1,2}(\Sigma) \leq \varepsilon_{\text{adm}}$
- $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ (positive progress)

**Case 2 (Admissible up to Equivalence):** After symmetry/gauge transformation $g \in G$:
$$g \cdot (\Sigma, V) \text{ satisfies Case 1}$$

**Case 3 (Not Admissible):** At least one condition fails with explicit witness.

## Proof Sketch

### Step 1: Canonical Library Check

**Library Definition (Perelman, 2003):** For Ricci flow, the canonical library is:
$$\mathcal{L}_{\text{Ricci}} = \{\text{round } S^3, \text{round } S^2 \times \mathbb{R}, \text{Bryant soliton}\}$$

**Reference:** Perelman, G. (2003). Ricci flow with surgery on three-manifolds. arXiv:math/0303109.

**MCF Library (Huisken-Sinestrari, 2009):**
$$\mathcal{L}_{\text{MCF}} = \{S^n, S^k \times \mathbb{R}^{n-k} : 1 \leq k \leq n\}$$

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries of two-convex hypersurfaces. *Invent. Math.*, 175, 137-221.

**Membership Test:** Compare profile $V$ against library elements:
$$\min_{C \in \mathcal{L}} d_{\text{GH}}(V, C) < \varepsilon_{\text{lib}}$$

using Gromov-Hausdorff distance.

### Step 2: Codimension Estimate

**Federer's Dimension Reduction (1969):** For area-minimizing currents:
$$\dim_{\mathcal{H}}(\text{sing}(T)) \leq n - 2$$

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 5.4]

**Codimension Computation:** Using Hausdorff dimension:
$$\text{codim}(\Sigma) = n - \sup\{s : \mathcal{H}^s(\Sigma) > 0\}$$

**Minkowski Content:** Alternatively:
$$\text{codim}(\Sigma) \geq 2 \iff \lim_{r \to 0} \frac{\mathcal{L}^n(B_r(\Sigma))}{r^2} < \infty$$

### Step 3: Capacity Bound

**Sobolev Capacity (Adams-Hedberg, 1996):**
$$\text{Cap}_{1,2}(\Sigma) := \inf\left\{ \int_{\mathbb{R}^n} |\nabla u|^2 : u \geq 1 \text{ on } \Sigma, u \in C_c^\infty \right\}$$

**Reference:** Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.

**Capacity-Dimension Relation (Maz'ya, 1985):**
$$\dim_{\mathcal{H}}(\Sigma) \leq n - 2 \implies \text{Cap}_{1,2}(\Sigma) = 0$$

**Reference:** Maz'ya, V. G. (1985). *Sobolev Spaces*. Springer.

**Threshold Verification:**
$$\text{Cap}_{1,2}(\Sigma) \leq \varepsilon_{\text{adm}}(n, \Lambda)$$

where $\varepsilon_{\text{adm}}$ depends on ambient dimension and energy bound.

### Step 4: Progress Measure

**Energy Drop (Hamilton, 1997):** Each surgery drops energy by:
$$\Delta\Phi_{\text{surg}} \geq c(n) \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

**Reference:** Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5, 1-92.

**Isoperimetric Bound:** By the isoperimetric inequality:
$$\text{Vol}(\Sigma)^{(n-2)/n} \geq c_{\text{iso}} \cdot \text{Area}(\partial B_\Sigma)^{(n-2)/(n-1)}$$

**Positive Progress:** The condition $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ ensures:
- No infinitesimal surgeries (Zeno prevention)
- Finite total surgery count

### Step 5: Equivalence Moves (Case 2)

**Symmetry Group Action:** For $g \in G = \text{Isom}(M) \times \text{Scale} \times \text{Gauge}$:
$$g \cdot \Sigma = \{g(x) : x \in \Sigma\}$$
$$g \cdot V = g_\# V$$

**Equivalence Moves (Kleiner-Lott, 2008):**
1. Translation: $V \mapsto V - x_0$
2. Scaling: $V \mapsto \lambda V$
3. Rotation: $V \mapsto R \cdot V$
4. Gauge: $V \mapsto e^{i\phi} V$ (for gauge theories)

**Reference:** Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12, 2587-2855.

**Admissibility After Equivalence:** If $\exists g \in G$ such that $g \cdot (\Sigma, V)$ satisfies Case 1, then the original surgery is admissible up to equivalence.

### Step 6: Failure Witnesses (Case 3)

**Failure Modes:**

1. **Non-Canonical Profile:**
   $$V \notin \mathcal{L} \text{ and } V \notin g \cdot \mathcal{L} \text{ for all } g \in G$$
   Witness: $d_{\text{GH}}(V, \mathcal{L}) > \varepsilon_{\text{lib}}$

2. **Insufficient Codimension:**
   $$\dim_{\mathcal{H}}(\Sigma) > n - 2$$
   Witness: $\mathcal{H}^{n-2+\delta}(\Sigma) > 0$ for some $\delta > 0$

3. **Excessive Capacity:**
   $$\text{Cap}_{1,2}(\Sigma) > \varepsilon_{\text{adm}}$$
   Witness: Capacity test function $u$ with $\int |\nabla u|^2 > \varepsilon_{\text{adm}}$

4. **Horizon Profile:**
   $$V \in \text{Case 3 of Profile Trichotomy (Wild)}$$
   Witness: Wildness certificate (positive Lyapunov, fractal dimension, etc.)

### Step 7: Surgery Count Bound

**Theorem (Perelman, 2003):** Under admissibility conditions, the total surgery count is bounded:
$$N_{\text{surg}} \leq \frac{\Phi(x_0) - \Phi_{\min}}{\epsilon_T}$$

*Proof:* Each surgery drops energy by $\geq \epsilon_T$. Total energy drop $\leq \Phi(x_0) - \Phi_{\min}$ (bounded). Hence $N_{\text{surg}} \cdot \epsilon_T \leq \Phi(x_0) - \Phi_{\min}$.

**Reference:** Perelman, G. (2003). Finite extinction time for the solutions to the Ricci flow on certain three-manifolds. arXiv:math/0307245.

## Key GMT Inequalities Used

1. **Federer Dimension Bound:**
   $$\dim_{\mathcal{H}}(\text{sing}(T)) \leq n - 2$$

2. **Capacity-Dimension:**
   $$\dim_{\mathcal{H}}(\Sigma) \leq n - 2 \iff \text{Cap}_{1,2}(\Sigma) = 0$$

3. **Isoperimetric Energy Drop:**
   $$\Delta\Phi \geq c(n) \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

4. **Surgery Count:**
   $$N_{\text{surg}} \leq \Phi_0 / \epsilon_T$$

## Literature References

- Perelman, G. (2002-2003). The entropy formula / Ricci flow with surgery / Finite extinction time. arXiv.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.
- Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12.
- Evans, L. C., Gariepy, R. F. (2015). *Measure Theory and Fine Properties of Functions*. CRC Press.
