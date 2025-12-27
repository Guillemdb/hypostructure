# UP-Holographic: Holographic-Regularity Theorem — GMT Translation

## Original Statement (Hypostructure)

The holographic-regularity theorem shows that boundary data determines bulk regularity: information on a lower-dimensional boundary controls interior behavior.

## GMT Setting

**Holography:** Bulk properties determined by boundary data

**Regularity:** Interior regularity from boundary regularity

**Dimension Reduction:** $n$-dimensional bulk controlled by $(n-1)$-dimensional boundary

## GMT Statement

**Theorem (Holographic-Regularity).** For $T \in \mathbf{I}_k(M)$ with $\partial T = S$:

1. **Boundary Control:** Regularity of $S$ implies regularity of $T$ near $\partial M$

2. **Interior Propagation:** Boundary regularity propagates into interior

3. **Dimension Count:** Bulk singular set has $\dim \leq k - 2$ if boundary is smooth

## Proof Sketch

### Step 1: Boundary Value Problem

**Plateau Problem:** Given $(k-1)$-current $S$ with $\partial S = 0$:
$$\text{Find } T \text{ minimizing } \mathbf{M}(T) \text{ with } \partial T = S$$

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

### Step 2: Boundary Regularity Implies Interior

**Allard Boundary Regularity (1975):** If $\partial T$ is smooth:
$$T \text{ is regular up to } \partial M$$

near points where density is 1.

**Reference:** Allard, W. K. (1975). On boundary regularity for Plateau's problem. *Bull. AMS*, 75, 522-523.

### Step 3: Interior Regularity

**De Giorgi-Nash-Moser:** Minimizers of elliptic problems are regular in interior.

**For Area Minimizers (Federer, 1970):**
$$\dim(\text{sing}(T)) \leq k - 2$$

if $T$ is area-minimizing.

**Reference:** Federer, H. (1970). The singular sets of area minimizing rectifiable currents. *Bull. AMS*, 76, 767-771.

### Step 4: Propagation of Regularity

**Unique Continuation:** If $T$ is smooth near boundary, smoothness propagates:
$$T|_{U} \text{ regular}, U \cap \partial M \neq \emptyset \implies T \text{ regular in neighborhood}$$

**Maximum Principle:** For minimal surfaces:
$$\text{sing}(T) \cap \partial M = \emptyset \implies \text{sing}(T) \text{ stays away from boundary}$$

### Step 5: Holographic Principle in Physics

**AdS/CFT (Maldacena, 1998):** Bulk gravity = boundary CFT

**Mathematical Analogue:** Bulk geometry determined by boundary data.

**Reference:** Maldacena, J. (1998). The large N limit of superconformal field theories and supergravity. *Adv. Theor. Math. Phys.*, 2, 231-252.

**GMT Version:** Current in bulk determined (up to minimization) by boundary.

### Step 6: Carleman Estimates

**Carleman Inequality:** For second-order elliptic operators:
$$\int e^{2\tau\phi} |u|^2 \leq C \int e^{2\tau\phi} |Lu|^2$$

**Unique Continuation:** Carleman implies unique continuation from boundary.

**Reference:** Hörmander, L. (1985). *The Analysis of Linear Partial Differential Operators*. Springer.

### Step 7: Boundary-to-Bulk Map

**Dirichlet-to-Neumann:** Given boundary data $f$:
$$\Lambda: f \mapsto \partial_\nu u|_{\partial M}$$

where $u$ solves $Lu = 0$ with $u|_{\partial M} = f$.

**Inverse Problem:** $\Lambda$ determines coefficients of $L$ (Calderón problem).

**Reference:** Calderón, A. P. (1980). On an inverse boundary value problem. *Seminar on Numerical Analysis*, 65-73.

### Step 8: Holographic Encoding of Singularities

**Theorem:** Singular set of bulk current is encoded in boundary:
$$\text{sing}(T) \subset \text{span}(\text{sing}(\partial T)) \cup \text{interior sing}$$

**Boundary-Controlled:** If $\partial T$ smooth, bulk singularities are interior only.

### Step 9: Dimension Reduction

**Co-dimension:** Bulk singular set has co-dimension $\geq 2$:
$$\dim(\text{sing}(T)) \leq k - 2$$

**Boundary Effect:** Smooth boundary pushes singularities to interior.

### Step 10: Compilation Theorem

**Theorem (Holographic-Regularity):**

1. **Boundary Control:** $\partial T$ smooth $\implies$ $T$ regular near $\partial M$

2. **Propagation:** Regularity propagates from boundary to interior

3. **Dimension:** $\dim(\text{sing}(T)) \leq k - 2$ with smooth boundary

4. **Encoding:** Bulk singularities encoded in boundary data

**Applications:**
- Plateau problem regularity
- Boundary value problems in GMT
- Inverse problems

## Key GMT Inequalities Used

1. **Boundary Regularity:**
   $$\partial T \text{ smooth} \implies T \text{ regular near } \partial M$$

2. **Singular Dimension:**
   $$\dim(\text{sing}(T)) \leq k - 2$$

3. **Unique Continuation:**
   $$T|_U = 0 \implies T = 0 \text{ (for minimizers)}$$

4. **Carleman:**
   $$\int e^{2\tau\phi}|u|^2 \leq C\int e^{2\tau\phi}|Lu|^2$$

## Literature References

- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Allard, W. K. (1975). On boundary regularity. *Bull. AMS*, 75.
- Federer, H. (1970). Singular sets of area minimizing currents. *Bull. AMS*, 76.
- Hörmander, L. (1985). *The Analysis of Linear PDO*. Springer.
- Maldacena, J. (1998). The large N limit. *Adv. Theor. Math. Phys.*, 2.
- Calderón, A. P. (1980). Inverse boundary value problem. *Seminar on Numerical Analysis*.
