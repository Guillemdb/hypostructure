# FACT-Surgery: Surgery Schema Factory — GMT Translation

## Original Statement (Hypostructure)

The surgery factory constructs surgical modification procedures that remove singularities by excision and replacement with canonical profiles.

## GMT Setting

**Surgery Schema:** $\mathcal{S} = (\Sigma, V, \mathcal{O}_S, \varepsilon)$ — singular set, profile, operator, scale

**Factory:** Constructs $\mathcal{S}$ from singularity data

**Output:** Surgically modified current $T'$

## GMT Statement

**Theorem (Surgery Schema Factory).** There exists a factory $\mathcal{F}_{\text{surg}}$ that, given:
- Singular current $T \in \mathbf{I}_k(M)$ with $\Sigma = \text{sing}(T)$
- Canonical library $\mathcal{L}$
- Soft certificates $K^+$

produces surgery schema $\mathcal{S}$ with:

1. **Profile Selection:** $V \in \mathcal{L}$ matching tangent cone at $\Sigma$

2. **Scale Selection:** $\varepsilon \in (\varepsilon_{\min}, \varepsilon_{\max})$ satisfying separation constraints

3. **Operator Construction:** $\mathcal{O}_S$ implementing cut-and-paste

4. **Output:** $T' = \mathcal{O}_S(T)$ with $\text{sing}(T') \subsetneq \text{sing}(T)$

## Proof Sketch

### Step 1: Singularity Detection

**Singular Set (Federer, 1970):** The singular set is:
$$\text{sing}(T) := \{x \in \text{spt}(T) : T \text{ is not a smooth } k\text{-manifold near } x\}$$

**Stratification (White, 1997):**
$$\text{sing}(T) = S^{(0)} \cup S^{(1)} \cup \cdots \cup S^{(k-2)}$$

where $\dim(S^{(j)}) \leq j$.

**Reference:** White, B. (1997). Stratification of minimal surfaces. *J. reine angew. Math.*, 488, 1-35.

**Detection Algorithm:** For each $x \in \text{spt}(T)$:
1. Compute tilt excess $E(T, B_r(x))$
2. If $E < \varepsilon_{\text{reg}}$: $x$ is regular
3. Else: $x \in \text{sing}(T)$

### Step 2: Tangent Cone Extraction

**Blow-Up Sequence:** At $x \in \text{sing}(T)$:
$$T_{x,\lambda} := (\eta_{x,\lambda})_\# T, \quad \eta_{x,\lambda}(y) = (y-x)/\lambda$$

**Tangent Cone (Simon, 1983):**
$$C_x := \lim_{\lambda \to 0} T_{x,\lambda}$$

exists by compactness.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

**Extraction Algorithm:**
1. Compute $T_{x,\lambda_j}$ for $\lambda_j = 2^{-j}$
2. Extract convergent subsequence
3. Identify limit cone $C_x$

### Step 3: Library Matching

**Library:** $\mathcal{L} = \{C_1, \ldots, C_N\}$ — classified cones

**Matching (Gromov-Hausdorff):**
$$\text{match}(C_x) := \arg\min_{C_i \in \mathcal{L}} d_{\text{GH}}(C_x \cap B_1, C_i \cap B_1)$$

**Threshold:** Accept match if $d_{\text{GH}} < \varepsilon_{\text{lib}}$.

**Examples (MCF, Huisken-Sinestrari, 2009):**
- $C_1 = S^{n-1}$ (round sphere)
- $C_2 = S^{k-1} \times \mathbb{R}^{n-k}$ (cylinder)

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175, 137-221.

### Step 4: Scale Selection

**Separation Constraint:** Surgery scales must satisfy:
$$\varepsilon < \frac{1}{10} \min_{x \neq y \in \text{sing}(T)} d(x, y)$$

**Non-Degeneracy:** Scale must be large enough:
$$\varepsilon > \varepsilon_{\min}(\Lambda, n)$$

to ensure positive energy drop.

**Optimal Scale (Hamilton, 1997):**
$$\varepsilon_{\text{opt}} = c(n) \cdot \min\left(\text{sing separation}, \text{inj radius}\right)$$

**Reference:** Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5, 1-92.

### Step 5: Cut Operation

**Excision:** Remove neighborhood of singularity:
$$T_{\text{cut}} := T \llcorner (M \setminus B_\varepsilon(\Sigma))$$

**Boundary:** The cut creates boundary:
$$\partial_{\text{new}} := \langle T, d_\Sigma, \varepsilon \rangle$$

(slice of $T$ at distance $\varepsilon$ from $\Sigma$).

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.3]

### Step 6: Paste Operation

**Profile Insertion:** Scale library profile to match boundary:
$$V_\varepsilon := \varepsilon^k \cdot V(x/\varepsilon)$$

**Boundary Matching:** Verify:
$$\partial V_\varepsilon = \partial_{\text{new}}$$

(up to orientation).

**Gluing (White, 1989):**
$$T' := T_{\text{cut}} \cup_{\partial_{\text{new}}} V_\varepsilon$$

**Reference:** White, B. (1989). A new proof of the compactness theorem. *Comment. Math. Helv.*, 64, 207-220.

### Step 7: Operator Construction

**Surgery Operator:**
$$\mathcal{O}_S: T \mapsto T' = (T \llcorner (M \setminus B_\varepsilon(\Sigma))) \cup V_\varepsilon$$

**Properties:**
1. **Mass:** $\mathbf{M}(T') \leq \mathbf{M}(T)$
2. **Boundary:** $\partial T' = \partial T$ (on $M \setminus B_{2\varepsilon}(\Sigma)$)
3. **Regularity:** $\text{sing}(T') \cap B_\varepsilon(\Sigma) = \emptyset$

### Step 8: Factory Algorithm

**Surgery Schema Factory:**

```
SurgeryFactory(T, L, soft_certs):
    # Step 1: Detect singularities
    Σ = detect_singularities(T)

    # Step 2: For each singular point
    for x in Σ:
        # Extract tangent cone
        C_x = extract_tangent_cone(T, x)

        # Match to library
        V_x = match_to_library(C_x, L)
        if V_x is None:
            return SURGERY_FAILED (wild profile)

    # Step 3: Select surgery scale
    ε = select_scale(Σ, T)

    # Step 4: Construct operator
    O_S = construct_operator(Σ, {V_x}, ε)

    # Return schema
    return SurgerySchema(Σ, {V_x}, O_S, ε)
```

### Step 9: Correctness Verification

**Schema Verification:**

1. **Profile Check:** Each $V_x \in \mathcal{L}$ (library membership)

2. **Scale Check:** $\varepsilon$ satisfies bounds

3. **Energy Drop:** $\Phi(T') \leq \Phi(T) - \epsilon_T$

4. **Regularity Improvement:** $\text{sing}(T') \subsetneq \text{sing}(T)$

**Certificate:** Output $(V_x, \varepsilon, \Delta\Phi)$ as proof.

### Step 10: Compilation Theorem

**Theorem (Surgery Factory):** The factory $\mathcal{F}_{\text{surg}}$:

1. **Inputs:** Singular current $T$, library $\mathcal{L}$, soft certificates
2. **Outputs:** Surgery schema $\mathcal{S}$
3. **Guarantees:**
   - If tangent cones are in $\mathcal{L}$, surgery succeeds
   - Surgery reduces singular set
   - Energy drops by at least $\epsilon_T$

**Perelman's Surgery (2003):** For Ricci flow:
- Library: $\{S^3, S^2 \times \mathbb{R}, \text{Bryant}\}$
- Scale: determined by curvature threshold
- Energy: Perelman's entropy

**Reference:** Perelman, G. (2003). Ricci flow with surgery on three-manifolds. arXiv:math/0303109.

## Key GMT Inequalities Used

1. **Stratification:**
   $$\dim(\text{sing}(T)) \leq k-2$$

2. **Tangent Cone Existence:**
   $$C_x = \lim_{\lambda \to 0} T_{x,\lambda}$$

3. **Library Matching:**
   $$d_{\text{GH}}(C_x, \mathcal{L}) < \varepsilon_{\text{lib}}$$

4. **Energy Drop:**
   $$\Phi(T') \leq \Phi(T) - c \cdot \text{Vol}(\Sigma)^{(n-2)/n}$$

## Literature References

- White, B. (1997). Stratification of minimal surfaces. *J. reine angew. Math.*, 488.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
