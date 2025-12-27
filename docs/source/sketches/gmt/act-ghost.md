# ACT-Ghost: Derived Extension / BRST Principle — GMT Translation

## Original Statement (Hypostructure)

The derived extension (BRST) principle shows how to extend structures to derived/ghost sectors, adding auxiliary fields that encode gauge symmetry and constraints cohomologically.

## GMT Setting

**Derived Structure:** Chain complexes and cohomology

**Ghost Fields:** Auxiliary variables encoding symmetry

**BRST:** Cohomological formulation of constraints

## GMT Statement

**Theorem (Derived Extension).** For constrained currents:

1. **Constraint:** $T$ satisfies $\Phi(T) = 0$ for constraint functional $\Phi$

2. **BRST Extension:** Extend to $(T, c, \bar{c})$ with ghost fields

3. **Cohomology:** Physical states = $H^0(Q)$ where $Q$ is BRST operator

4. **Equivalence:** Constrained system ≅ BRST cohomology

## Proof Sketch

### Step 1: BRST Formalism in Physics

**Origin:** Becchi-Rouet-Stora-Tyutin formalism for gauge theories.

**BRST Operator:** $Q$ with $Q^2 = 0$, encodes gauge symmetry.

**Reference:** Henneaux, M., Teitelboim, C. (1992). *Quantization of Gauge Systems*. Princeton.

### Step 2: Homological Algebra Setting

**Chain Complex:** $(C_\bullet, \partial)$ with $\partial^2 = 0$

**Cohomology:** $H_*(C) = \ker(\partial)/\text{im}(\partial)$

**Reference:** Weibel, C. (1994). *An Introduction to Homological Algebra*. Cambridge.

### Step 3: Koszul Complex

**Definition:** For functions $f_1, \ldots, f_m$ on $X$:
$$K_\bullet(f_1, \ldots, f_m) = \Lambda^\bullet(\mathbb{R}^m) \otimes \mathcal{O}_X$$

with differential $\partial = \sum f_i \cdot \iota_{\partial/\partial\xi_i}$.

**Cohomology:** $H^0(K) = \mathcal{O}_X/(f_1, \ldots, f_m)$ = functions on constraint locus.

**Reference:** Eisenbud, D. (1995). *Commutative Algebra*. Springer.

### Step 4: Ghost Variables

**Definition:** Ghost $c^i$ and antighost $\bar{c}_i$ are auxiliary variables:
- Ghost: degree +1
- Antighost: degree -1

**BRST Operator:**
$$Q = c^i \frac{\partial}{\partial x^i} \text{ (schematically)}$$

### Step 5: Derived Constraint Locus

**Classical Constraint:** $Z = \{x : f_1(x) = \cdots = f_m(x) = 0\}$

**Derived Enhancement:** Replace $Z$ by derived scheme:
$$Z^{\text{der}} = \text{Spec}(K_\bullet(f_1, \ldots, f_m))$$

**Reference:** Lurie, J. (2004). *Derived Algebraic Geometry*. Thesis.

### Step 6: Current Constraints

**GMT Setting:** Constraint $\partial T = S$ is homological:
$$T \in \partial^{-1}(S) = \{T : \partial T = S\}$$

**Derived:** Consider chain complex of currents:
$$\cdots \to \mathbf{I}_{k+1}(M) \xrightarrow{\partial} \mathbf{I}_k(M) \xrightarrow{\partial} \mathbf{I}_{k-1}(M) \to \cdots$$

### Step 7: BRST for Gauge Symmetry

**Gauge Symmetry:** For currents with symmetry group $G$:
- Ghost: $c \in \mathfrak{g}[1]$ (Lie algebra shifted)
- Antighost: $\bar{c} \in \mathfrak{g}^*[-1]$

**BRST Operator:** $Q = \delta_{\text{gauge}} + c^a c^b f_{ab}^c \partial/\partial c^c + \ldots$

### Step 8: Physical States

**Theorem:** Physical states are BRST cohomology:
$$\mathcal{H}_{\text{phys}} = H^0(Q)$$

*Proof (sketch):*
1. $Q$-closed states satisfy constraints + gauge invariance
2. $Q$-exact states are gauge trivial
3. Cohomology = gauge-invariant constrained states

### Step 9: Derived Categories

**Derived Category:** $D(X) = $ derived category of coherent sheaves

**Currents as Objects:** Current $T$ → object in derived category via:
$$T \mapsto \mathcal{O}_{\text{spt}(T)}$$

**Reference:** Kontsevich, M. (1995). Homological algebra of mirror symmetry. *Proc. ICM Zürich*.

### Step 10: Compilation Theorem

**Theorem (Derived Extension):**

1. **Classical:** Constraint $\Phi(T) = 0$

2. **Derived:** Extend by ghost fields $(c, \bar{c})$

3. **BRST:** $Q^2 = 0$ encodes constraints cohomologically

4. **Equivalence:** Constrained currents ≅ $H^0(Q)$

**Applications:**
- Gauge-invariant formulations
- Derived geometry in GMT
- Cohomological constraints

## Key GMT Inequalities Used

1. **Chain Complex:**
   $$\partial^2 = 0$$

2. **Koszul:**
   $$H^0(K) = R/(f_1, \ldots, f_m)$$

3. **BRST:**
   $$Q^2 = 0, \quad \mathcal{H}_{\text{phys}} = H^0(Q)$$

4. **Current Homology:**
   $$H_k(M) = \ker(\partial)/\text{im}(\partial)$$

## Literature References

- Henneaux, M., Teitelboim, C. (1992). *Quantization of Gauge Systems*. Princeton.
- Weibel, C. (1994). *Homological Algebra*. Cambridge.
- Eisenbud, D. (1995). *Commutative Algebra*. Springer.
- Lurie, J. (2004). *Derived Algebraic Geometry*. Thesis.
- Kontsevich, M. (1995). Homological algebra of mirror symmetry. *Proc. ICM*.
