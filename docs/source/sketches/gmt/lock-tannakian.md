# LOCK-Tannakian: Tannakian Recognition Lock — GMT Translation

## Original Statement (Hypostructure)

The Tannakian recognition lock shows that categorical structure (tensor category with fiber functor) determines group structure, locking representations to underlying symmetry group.

## GMT Setting

**Tannakian Category:** Tensor category equivalent to representations of group

**Fiber Functor:** Forgetful functor to vector spaces

**Recognition:** Category determines group up to isomorphism

## GMT Statement

**Theorem (Tannakian Recognition Lock).** For suitable tensor category $\mathcal{C}$:

1. **Structure:** $\mathcal{C}$ is rigid tensor category with fiber functor $\omega: \mathcal{C} \to \text{Vect}_k$

2. **Recognition:** $\mathcal{C} \cong \text{Rep}_k(G)$ for affine group scheme $G$

3. **Reconstruction:** $G = \underline{\text{Aut}}^\otimes(\omega)$

4. **Lock:** Categorical data uniquely determines symmetry

## Proof Sketch

### Step 1: Tensor Categories

**Definition:** A tensor category is:
- Abelian category $\mathcal{C}$
- Bifunctor $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$
- Unit object $\mathbf{1}$
- Associativity and unit constraints

**Reference:** Deligne, P., Milne, J. (1982). Tannakian categories. *Hodge Cycles, Motives, and Shimura Varieties*. Springer LNM 900.

### Step 2: Rigid Structure

**Duality:** Object $X$ is rigid if there exists $X^\vee$ with:
$$\text{ev}: X^\vee \otimes X \to \mathbf{1}, \quad \text{coev}: \mathbf{1} \to X \otimes X^\vee$$

satisfying zigzag identities.

**Rigid Category:** All objects are rigid.

### Step 3: Fiber Functor

**Definition:** A fiber functor is exact faithful tensor functor:
$$\omega: \mathcal{C} \to \text{Vect}_k$$

**Neutralized:** Category with chosen fiber functor is neutralized Tannakian.

### Step 4: Tannaka Reconstruction

**Theorem (Tannaka-Krein-Deligne):** If $\mathcal{C}$ is rigid abelian tensor category over field $k$ with fiber functor $\omega$, then:
$$\mathcal{C} \cong \text{Rep}_k(G)$$

where $G = \underline{\text{Aut}}^\otimes(\omega)$ is affine group scheme.

**Reference:** Deligne, P. (1990). Catégories tannakiennes. *The Grothendieck Festschrift*. Birkhäuser.

### Step 5: Affine Group Scheme

**Definition:** $G = \underline{\text{Aut}}^\otimes(\omega)$ is the functor:
$$R \mapsto \{\text{natural tensor automorphisms of } \omega \otimes R\}$$

**Hopf Algebra:** Coordinate ring $\mathcal{O}(G)$ is Hopf algebra:
$$\mathcal{O}(G) = \text{End}^\otimes(\omega)^\vee$$

### Step 6: GMT Interpretation

**Currents with Symmetry:** For $T \in \mathbf{I}_k(M)$ with $G$-symmetry:
- Symmetry group $G$ acts on $T$
- Orbit $G \cdot T$ forms family

**Tensor Structure:** Intersection, union of currents give tensor operations.

### Step 7: Lock Mechanism

**Uniqueness:** The group $G$ is uniquely determined by $\mathcal{C}$:

*Proof:*
1. Two fiber functors $\omega_1, \omega_2$ are related by isomorphism (torsor)
2. Automorphism groups are inner conjugate
3. Therefore $G$ unique up to isomorphism

**Lock:** Categorical structure fixes symmetry group.

### Step 8: Motivic Galois Group

**Motives:** The category of motives $\mathcal{M}_k$ is (conjecturally) Tannakian.

**Motivic Galois:** $G_{\text{mot}} = \underline{\text{Aut}}^\otimes(\omega_B)$ where $\omega_B$ is Betti realization.

**Reference:** André, Y. (2004). *Une Introduction aux Motifs*. Société Mathématique de France.

**Lock:** Motivic Galois group encodes all symmetries of algebraic varieties.

### Step 9: Differential Galois Theory

**Differential Equations:** Solutions of linear ODE form tensor category.

**Differential Galois Group:** Picard-Vessiot theory:
$$G = \text{Aut}(K/k)$$

where $K$ is Picard-Vessiot extension.

**Reference:** van der Put, M., Singer, M. (2003). *Galois Theory of Linear Differential Equations*. Springer.

### Step 10: Compilation Theorem

**Theorem (Tannakian Recognition Lock):**

1. **Tensor Category:** Rigid abelian with fiber functor

2. **Reconstruction:** $\mathcal{C} \cong \text{Rep}(G)$

3. **Uniqueness:** $G$ determined up to isomorphism

4. **Lock:** Categorical structure locks symmetry

**Applications:**
- Motivic Galois theory
- Differential Galois theory
- Symmetry recognition in geometry

## Key GMT Inequalities Used

1. **Tannaka Duality:**
   $$\mathcal{C} \cong \text{Rep}(G)$$

2. **Hopf Algebra:**
   $$\mathcal{O}(G) = \text{End}^\otimes(\omega)^\vee$$

3. **Fiber Functor Torsor:**
   $$\text{Isom}^\otimes(\omega_1, \omega_2) = G\text{-torsor}$$

4. **Uniqueness:**
   $$G_1 \cong G_2 \text{ canonically}$$

## Literature References

- Deligne, P., Milne, J. (1982). Tannakian categories. *LNM* 900.
- Deligne, P. (1990). Catégories tannakiennes. *Grothendieck Festschrift*.
- André, Y. (2004). *Une Introduction aux Motifs*. SMF.
- van der Put, M., Singer, M. (2003). *Galois Theory of Linear ODE*. Springer.
- Saavedra Rivano, N. (1972). *Catégories Tannakiennes*. Springer LNM 265.
