# ACT-Projective: Projective Extension Principle — GMT Translation

## Original Statement (Hypostructure)

The projective extension principle shows how to extend configurations to projective completions, adding points at infinity to compactify and regularize the structure.

## GMT Setting

**Projective Completion:** $M \hookrightarrow \overline{M}$ adding ideal points

**Extension:** Current $T$ extends to $\bar{T}$ on completion

**Regularization:** Behavior at infinity controlled by projective structure

## GMT Statement

**Theorem (Projective Extension).** For $T \in \mathbf{I}_k(\mathbb{R}^n)$:

1. **Completion:** Embed $\mathbb{R}^n \hookrightarrow \mathbb{P}^n$

2. **Extension:** $T$ extends to $\bar{T} \in \mathbf{I}_k(\mathbb{P}^n)$ with controlled mass at infinity

3. **Regularity:** Projective structure constrains behavior at infinity

## Proof Sketch

### Step 1: Projective Space

**Real Projective Space:** $\mathbb{RP}^n = (\mathbb{R}^{n+1} \setminus \{0\})/\mathbb{R}^*$

**Complex Projective Space:** $\mathbb{CP}^n = (\mathbb{C}^{n+1} \setminus \{0\})/\mathbb{C}^*$

**Embedding:** $\mathbb{R}^n \hookrightarrow \mathbb{RP}^n$ via $x \mapsto [x:1]$

**Reference:** Harris, J. (1992). *Algebraic Geometry: A First Course*. Springer.

### Step 2: Hyperplane at Infinity

**Definition:** $H_\infty = \{[x_0:\cdots:x_n] : x_n = 0\} \cong \mathbb{RP}^{n-1}$

**Compactification:** $\mathbb{RP}^n = \mathbb{R}^n \cup H_\infty$

**Metric:** Fubini-Study metric makes $\mathbb{RP}^n$ compact.

### Step 3: Extension of Currents

**Bounded Mass:** If $\mathbf{M}(T) < \infty$ in $\mathbb{R}^n$:
$$\bar{T} = T + T_\infty$$

where $T_\infty$ is current at infinity.

**Closure:** $\bar{T} = $ closure of $T$ in $\mathbb{RP}^n$.

**Reference:** Harvey, R., Lawson, H. B. (1983). An intrinsic characterization of Kähler manifolds. *Invent. Math.*, 74, 169-198.

### Step 4: Growth Control

**Polynomial Growth:** If $|T|_{B_R} \leq C R^k$:

Then $T$ extends to current in projective space with:
$$\mathbf{M}(\bar{T}) < \infty$$

**Exponential Growth:** Faster growth may not extend.

### Step 5: Algebraic Currents

**Theorem (Bishop):** Positive closed currents of finite mass on projective manifolds are given by analytic cycles.

**Reference:** Bishop, E. (1964). Conditions for the analyticity of certain sets. *Michigan Math. J.*, 11, 289-304.

**Extension:** Algebraic subvarieties of $\mathbb{C}^n$ extend to $\mathbb{CP}^n$.

### Step 6: Chow's Theorem

**Theorem (Chow):** Every analytic subvariety of $\mathbb{CP}^n$ is algebraic.

**Reference:** Chow, W. L. (1949). On compact complex analytic varieties. *Amer. J. Math.*, 71, 893-914.

**Consequence:** Projective structure implies algebraic structure.

### Step 7: Degree at Infinity

**Degree:** For algebraic variety $V \subset \mathbb{CP}^n$:
$$\deg(V) = \#(V \cap L)$$

for generic linear subspace $L$ of complementary dimension.

**Control:** Degree bounds complexity of $V \cap H_\infty$.

### Step 8: Compactification of Moduli

**Deligne-Mumford:** Moduli space of curves compactifies:
$$\mathcal{M}_g \subset \overline{\mathcal{M}}_g$$

**Reference:** Deligne, P., Mumford, D. (1969). The irreducibility of the space of curves. *Publ. Math. IHES*, 36, 75-109.

**Boundary:** Points at infinity = stable nodal curves.

### Step 9: Projective Regularity

**Regularity at Infinity:** If $T$ extends to $\bar{T}$ with:
$$\bar{T} \text{ regular at } H_\infty$$

then $T$ has controlled behavior at infinity in $\mathbb{R}^n$.

**Constraint:** Projective extension constrains asymptotic behavior.

### Step 10: Compilation Theorem

**Theorem (Projective Extension):**

1. **Completion:** $\mathbb{R}^n \hookrightarrow \mathbb{P}^n$ adds infinity

2. **Extension:** Finite mass currents extend

3. **Algebraicity:** Closed positive currents are algebraic (Chow)

4. **Regularity:** Projective structure constrains infinity

**Applications:**
- Compactification of moduli spaces
- Algebraic geometry of currents
- Asymptotic analysis

## Key GMT Inequalities Used

1. **Mass Extension:**
   $$\mathbf{M}(\bar{T}) = \mathbf{M}(T) + \mathbf{M}(T_\infty)$$

2. **Growth Bound:**
   $$|T|_{B_R} \leq CR^k \implies \text{extends}$$

3. **Degree:**
   $$\deg(V) = \#(V \cap L_{\text{generic}})$$

4. **Bishop:**
   $$\partial T = 0, T \geq 0, \mathbf{M}(T) < \infty \implies T \text{ algebraic}$$

## Literature References

- Harris, J. (1992). *Algebraic Geometry: A First Course*. Springer.
- Harvey, R., Lawson, H. B. (1983). Intrinsic characterization of Kähler manifolds. *Invent. Math.*, 74.
- Bishop, E. (1964). Analyticity of certain sets. *Michigan Math. J.*, 11.
- Chow, W. L. (1949). Compact complex analytic varieties. *Amer. J. Math.*, 71.
- Deligne, P., Mumford, D. (1969). Irreducibility of curve space. *Publ. Math. IHES*, 36.
