# RESOLVE-AutoAdmit: Automatic Admissibility — GMT Translation

## Original Statement (Hypostructure)

Surgery admissibility is automatically verified via soft certificate checking. If admissibility fails, an explicit witness to the failure is provided.

## GMT Setting

**Surgery Data:** $(\Sigma, V, \mathcal{O}_S)$ — singular set, replacement profile, surgery operator

**Soft Certificates:** $K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+$

**Admissibility Predicate:** $\text{Adm}(\Sigma, V) \in \{\text{True}, \text{False}\}$

## GMT Statement

**Theorem (Automatic Admissibility Verification).** Given surgery data $(\Sigma, V, \mathcal{O}_S)$ and soft certificates, admissibility can be automatically determined:

**Case 1 (Admissible):** All checks pass:
- $V \in \mathcal{L}$ verified via Gromov-Hausdorff proximity
- $\text{codim}(\Sigma) \geq 2$ verified via Hausdorff dimension
- $\text{Cap}_{1,2}(\Sigma) \leq \varepsilon_{\text{adm}}$ verified via capacity computation
- $\Delta\Phi \geq \epsilon_T$ verified via energy comparison

**Case 2 (Not Admissible):** Explicit witness provided:
- Distance witness: $d_{\text{GH}}(V, \mathcal{L}) > \varepsilon_{\text{lib}}$
- Dimension witness: $\mathcal{H}^{n-2+\delta}(\Sigma) > 0$
- Capacity witness: $\text{Cap}_{1,2}(\Sigma) > \varepsilon_{\text{adm}}$
- Progress witness: $\Delta\Phi < \epsilon_T$

## Proof Sketch

### Step 1: Library Membership via Gromov-Hausdorff

**Gromov's Compactness (1981):** The space of compact metric spaces with diameter $\leq D$ and doubling constant $\leq C$ is precompact in GH topology.

**Reference:** Gromov, M. (1981). Groups of polynomial growth and expanding maps. *Publ. Math. IHES*, 53, 53-78.

**Library Enumeration:** For finite canonical library $\mathcal{L} = \{C_1, \ldots, C_N\}$:
$$\text{Membership}(V) := \min_{i=1}^N d_{\text{GH}}(V, C_i) < \varepsilon_{\text{lib}}$$

**Algorithmic Computation (Mémoli, 2007):** The GH distance can be approximated:
$$d_{\text{GH}}(X, Y) = \frac{1}{2} \inf_R \text{dis}(R)$$

where $R \subset X \times Y$ is a correspondence and $\text{dis}(R) = \sup\{|d_X(x,x') - d_Y(y,y')| : (x,y), (x',y') \in R\}$.

**Reference:** Mémoli, F. (2007). On the use of Gromov-Hausdorff distances for shape comparison. *Eurographics Symp. Point-Based Graphics*.

### Step 2: Codimension via Hausdorff Dimension

**Hausdorff Dimension Computation:** For $\Sigma \subset M^n$:
$$\dim_{\mathcal{H}}(\Sigma) = \inf\{s : \mathcal{H}^s(\Sigma) = 0\} = \sup\{s : \mathcal{H}^s(\Sigma) = \infty\}$$

**Box-Counting Approximation (Falconer, 1990):**
$$\dim_{\text{box}}(\Sigma) = \lim_{r \to 0} \frac{\log N(r)}{-\log r}$$

where $N(r)$ is the minimal number of $r$-balls covering $\Sigma$.

**Reference:** Falconer, K. (1990). *Fractal Geometry: Mathematical Foundations and Applications*. Wiley.

**Codimension Test:**
$$\text{codim}(\Sigma) \geq 2 \iff \dim_{\mathcal{H}}(\Sigma) \leq n - 2$$

### Step 3: Capacity via Sobolev Test Functions

**Capacity Definition (Adams-Hedberg, 1996):**
$$\text{Cap}_{1,2}(\Sigma) = \inf\left\{\int_M |\nabla u|^2 + u^2 \, d\mu : u \geq 1 \text{ on } \Sigma, u \in W^{1,2}(M)\right\}$$

**Reference:** Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.

**Numerical Approximation:** Discretize $M$ and solve the variational problem:
$$\min_{u_h \in V_h} \int_M |\nabla u_h|^2 + u_h^2 \, dx \quad \text{s.t. } u_h|_{\Sigma_h} \geq 1$$

**Witness Construction:** If $\text{Cap}_{1,2}(\Sigma) > \varepsilon_{\text{adm}}$, the minimizing $u$ with $\int |\nabla u|^2 > \varepsilon_{\text{adm}}$ is the witness.

### Step 4: Progress via Energy Comparison

**Energy Drop Computation:** Before surgery: $\Phi_{\text{pre}}$. After surgery: $\Phi_{\text{post}}$.
$$\Delta\Phi = \Phi_{\text{pre}} - \Phi_{\text{post}}$$

**Positive Progress Check:**
$$\Delta\Phi \geq \epsilon_T$$

**Witness Construction:** If $\Delta\Phi < \epsilon_T$, the pair $(\Phi_{\text{pre}}, \Phi_{\text{post}})$ with $\Delta\Phi < \epsilon_T$ is the witness.

### Step 5: Decidability and Termination

**Theorem (Decidability of Admissibility):** Under soft permits with computable bounds, admissibility is decidable.

*Proof:* Each sub-check is decidable:
1. GH distance to finite library: finite minimum
2. Hausdorff dimension: approximable via box counting
3. Capacity: variational problem with finite discretization
4. Energy drop: explicit computation

**Termination:** The algorithm terminates in $O(N \cdot |\text{mesh}|^k)$ time where $N = |\mathcal{L}|$ is library size.

### Step 6: Witness Certification

**Certification Protocol:**

1. **Negative Witness Verification:** Given claimed witness $w$, verify:
   - For distance: check $d_{\text{GH}}(V, C_i) > \varepsilon_{\text{lib}}$ for all $C_i$
   - For dimension: verify $\mathcal{H}^{n-2+\delta}(\Sigma) > 0$ via covering argument
   - For capacity: verify $\int |\nabla u_w|^2 > \varepsilon_{\text{adm}}$ for witness function $u_w$

2. **Positive Witness Verification:** Given claimed admissibility, verify all four conditions hold.

**Reference:** Blum, L., Cucker, F., Shub, M., Smale, S. (1998). *Complexity and Real Computation*. Springer.

### Step 7: Error Bounds

**Approximation Error (Cheeger-Colding, 1997):** For GH approximation:
$$|d_{\text{GH}}^{\text{approx}}(V, C) - d_{\text{GH}}(V, C)| \leq \epsilon_{\text{approx}}$$

**Reference:** Cheeger, J., Colding, T. (1997). On the structure of spaces with Ricci curvature bounded below I. *J. Diff. Geom.*, 46, 406-480.

**Robust Admissibility:** With margin $\delta > 0$:
$$\text{Adm}_\delta := \text{Adm with thresholds relaxed by } \delta$$

ensures stability under approximation errors.

## Key GMT Inequalities Used

1. **GH Compactness:**
   $$\{(X, d) : \text{diam}(X) \leq D, \text{doubling}(X) \leq C\} \text{ is precompact}$$

2. **Dimension-Capacity:**
   $$\dim_{\mathcal{H}}(\Sigma) \leq n - 2 \implies \text{Cap}_{1,2}(\Sigma) = 0$$

3. **Box-Counting Bound:**
   $$\dim_{\mathcal{H}}(\Sigma) \leq \underline{\dim}_{\text{box}}(\Sigma)$$

4. **Capacity Estimate:**
   $$\text{Cap}_{1,2}(B_r(x)) \asymp r^{n-2}$$

## Literature References

- Gromov, M. (1981). Groups of polynomial growth. *Publ. Math. IHES*, 53, 53-78.
- Adams, D., Hedberg, L. (1996). *Function Spaces and Potential Theory*. Springer.
- Falconer, K. (1990). *Fractal Geometry*. Wiley.
- Cheeger, J., Colding, T. (1997). Structure of spaces with Ricci curvature bounded below. *J. Diff. Geom.*, 46.
- Mémoli, F. (2007). Gromov-Hausdorff distances. *Eurographics Symposium*.
- Blum, L. et al. (1998). *Complexity and Real Computation*. Springer.
