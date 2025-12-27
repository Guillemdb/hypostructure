# ACT-Lift: Regularity Lift Principle — GMT Translation

## Original Statement (Hypostructure)

The regularity lift principle shows how to lift regularity from a lower-dimensional slice or projection to the full current, propagating smoothness from partial data.

## GMT Setting

**Lift:** Regularity of slice/projection implies regularity of current

**Slicing:** Current $T$ sliced by function $f$ gives family $\langle T, f, y \rangle$

**Propagation:** Slice regularity lifts to ambient regularity

## GMT Statement

**Theorem (Regularity Lift Principle).** For $T \in \mathbf{I}_k(M)$:

1. **Slicing:** If $\langle T, f, y \rangle$ is regular for a.e. $y$

2. **Lift:** Then $T$ is regular at corresponding points

3. **Dimension:** Regularity propagates from $(k-1)$-slices to $k$-current

## Proof Sketch

### Step 1: Slicing Theory

**Definition:** For $T \in \mathbf{I}_k(M)$ and Lipschitz $f: M \to \mathbb{R}^m$:
$$\langle T, f, y \rangle \in \mathbf{I}_{k-m}(f^{-1}(y))$$

for a.e. $y \in \mathbb{R}^m$.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §4.3.

### Step 2: Coarea Formula

**Theorem:** For $T \in \mathbf{I}_k(M)$ and Lipschitz $f: M \to \mathbb{R}$:
$$T = \int_{\mathbb{R}} \langle T, f, y \rangle \, dy$$

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §3.2.

**Consequence:** Current reconstructed from slices.

### Step 3: Regularity of Slices

**Slice Regularity:** For area-minimizing $T$, slices $\langle T, f, y \rangle$ are area-minimizing in fiber.

**Reference:** Federer, H. (1970). The singular sets of area minimizing rectifiable currents. *Bull. AMS*, 76.

**Dimension Reduction:** Regularity analysis reduces to lower dimension.

### Step 4: Lifting Argument

**Theorem:** If $\langle T, f, y \rangle$ is smooth for a.e. $y \in U$:

Then $T$ is smooth on $f^{-1}(U)$ away from measure-zero set.

*Sketch:*
1. Smooth slices have smooth tangent planes
2. Tangent planes of $T$ are products: $\text{Tan}(\langle T, f, y \rangle) \times \mathbb{R}$
3. Product of smooth tangent planes is smooth

### Step 5: Almgren's Stratification

**Theorem (Almgren):** Singular set stratifies:
$$\text{sing}(T) = S_0 \cup S_1 \cup \cdots \cup S_{k-2}$$

with $\dim(S_j) \leq j$.

**Reference:** Almgren, F. J. (2000). *Almgren's Big Regularity Paper*. World Scientific.

**Lift:** Regularity of generic slice implies regularity outside codimension-2.

### Step 6: Unique Continuation

**Theorem:** If $T$ is smooth on open set $U$ and $T$ is area-minimizing:

Then $T$ is smooth on connected component of regular set containing $U$.

**Reference:** Simon, L. (1996). Theorems on regularity and singularity of energy minimizing maps. Birkhäuser.

**Lift:** Local smoothness propagates.

### Step 7: Product Structure

**Product Lift:** If $T = S \times \llbracket I \rrbracket$ locally:
$$\text{reg}(T) = \text{reg}(S) \times I$$

**Consequence:** Regularity lifts from factor to product.

### Step 8: Carleman Estimates

**Unique Continuation via Carleman:** For elliptic systems:
$$\int e^{2\tau\phi}|u|^2 \leq C\int e^{2\tau\phi}|Lu|^2$$

**Reference:** Hörmander, L. (1985). *Analysis of Linear PDO*. Springer.

**Lift:** Carleman implies regularity propagation.

### Step 9: GMT Regularity Theory

**De Giorgi-Nash-Moser:** Solutions of elliptic equations are Hölder continuous.

**Reference:** De Giorgi, E. (1957). Sulla differenziabilità e l'analiticità delle estremali degli integrali multipli. *Mem. Accad. Sci. Torino*, 3, 25-43.

**Lift:** Interior regularity from boundary regularity.

### Step 10: Compilation Theorem

**Theorem (Regularity Lift Principle):**

1. **Slice:** $\langle T, f, y \rangle$ regular for a.e. $y$

2. **Lift:** $T$ regular on $f^{-1}(\text{regular values})$

3. **Propagation:** Local regularity spreads via unique continuation

4. **Stratification:** Singular set has codimension $\geq 2$

**Applications:**
- Dimension reduction for regularity
- Product regularity
- Inductive proofs on dimension

## Key GMT Inequalities Used

1. **Coarea:**
   $$T = \int \langle T, f, y \rangle \, dy$$

2. **Slice Regularity:**
   $$\text{sing}(\langle T, f, y \rangle) \subset \text{sing}(T) \cap f^{-1}(y)$$

3. **Lift:**
   $$\text{reg}(\langle T, f, \cdot \rangle) \text{ a.e.} \implies \text{reg}(T) \text{ generic}$$

4. **Stratification:**
   $$\dim(\text{sing}(T)) \leq k - 2$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Federer, H. (1970). Singular sets of area minimizing currents. *Bull. AMS*, 76.
- Almgren, F. J. (2000). *Big Regularity Paper*. World Scientific.
- Simon, L. (1996). Regularity and singularity. Birkhäuser.
- De Giorgi, E. (1957). Differenziabilità delle estremali. *Mem. Accad. Sci. Torino*, 3.
- Hörmander, L. (1985). *Analysis of Linear PDO*. Springer.
