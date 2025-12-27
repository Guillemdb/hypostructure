# KRNL-Openness: Openness of Regularity

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-openness*

The regularity locus is open in appropriate topology.

---

## Arithmetic Formulation

### Setup

"Openness of regularity" in arithmetic means:
- Good reduction is an open condition
- Regular families stay regular under small deformation
- Smoothness is stable under perturbation

### Statement (Arithmetic Version)

**Theorem (Arithmetic Openness of Regularity).** Regularity is open:

1. **Good reduction:** The locus of good reduction is open in moduli
2. **Smoothness:** Smooth fibers form an open set in families
3. **Non-vanishing:** $L(E, 1) \neq 0$ is open in families (generically)

---

### Proof

**Step 1: Good Reduction Openness**

For family of elliptic curves $\mathcal{E} \to S$:

**Good reduction locus:**
$$U = \{s \in S : \mathcal{E}_s \text{ has good reduction at all } p\}$$

**Openness proof:**
- Bad reduction at $p$ requires $p | \Delta_s$
- $\Delta: S \to \mathbb{Z}$ is continuous
- $\{s : p \nmid \Delta_s\}$ is open (preimage of open)
- $U = \bigcap_p \{s : p \nmid \Delta_s\}$ is open for finitely many $p$

**For infinite families:** Good reduction outside $S$ is open where $S$ is the set of bad primes.

**Step 2: Smoothness Openness**

For morphism $f: X \to S$:

**Smooth locus:**
$$U = \{s \in S : X_s \text{ is smooth}\}$$

**Openness [EGA IV]:**
- Smoothness is equivalent to: $\Omega_{X/S}$ locally free + fibers geometrically regular
- Both conditions are open
- Hence $U$ is open

**Arithmetic version:** For $\mathcal{E} \to \text{Spec}(\mathbb{Z})$:
- Smooth locus is open (complement of finitely many closed primes)

**Step 3: Non-Vanishing Openness**

For family $\mathcal{E}_t$ parametrized by $t$:

**Non-vanishing locus:**
$$U = \{t : L(\mathcal{E}_t, 1) \neq 0\}$$

**Generic openness:**
- $L(\mathcal{E}_t, 1)$ varies analytically in $t$
- Zeros of analytic function are isolated (generically)
- Hence $U$ is open and dense (conjecturally)

**Density result [Kolyvagin, average rank]:**
- Positive proportion of $E$ have $L(E, 1) \neq 0$
- These form an open dense set in moduli

**Step 4: Regularity in Families**

**General principle:** For algebraic family $\{X_t\}_{t \in T}$:

**Regular fiber locus:**
$$U = \{t : X_t \text{ is regular}\}$$

**Openness:** By upper semicontinuity of singularity dimension:
- $\dim \text{Sing}(X_t)$ is upper semicontinuous
- $\{\dim \text{Sing} = -\infty\} = \{X_t \text{ smooth}\}$ is open

**Step 5: Conductor Openness**

**Conductor function:** $N: \mathcal{M} \to \mathbb{Z}_{>0}$ on moduli space

**Level sets:**
$$\mathcal{M}_{\leq N} = \{E : N_E \leq N\}$$

**Openness:** $\mathcal{M}_{\leq N}$ is Zariski open in $\mathcal{M}$ (finite union of components with bounded conductor).

**Step 6: Openness Certificate**

The openness certificate:
$$K_{\text{Open}}^+ = (\text{regularity condition}, \text{open set}, \text{proof of openness})$$

**Components:**
- **Condition:** (good reduction, smoothness, non-vanishing)
- **Set:** Explicit description of open locus
- **Proof:** Semicontinuity argument or explicit bound

**Examples:**
- (Good reduction at $p$): $\{E : p \nmid \Delta_E\}$ — open
- (Rank 0): $\{E : L(E, 1) \neq 0\}$ — open and dense (conjecturally)
- (Smooth family): Generic fiber smooth — open

---

### Key Arithmetic Ingredients

1. **EGA IV** [Grothendieck 1966]: Openness of smoothness.
2. **Semicontinuity** [Hartshorne]: Upper semicontinuity of fiber dimension.
3. **Average Rank** [Goldfeld, Katz-Sarnak]: Density of rank 0.
4. **Discriminant Continuity** [Silverman]: Discriminant varies continuously.

---

### Arithmetic Interpretation

> **Arithmetic regularity is open. Good reduction, smoothness, and L-function non-vanishing define open conditions in moduli space. Small perturbations of regular objects remain regular. This openness principle ensures stability of arithmetic properties under deformation.**

---

### Literature

- [Grothendieck 1966] A. Grothendieck, *EGA IV: Étude locale des schémas*
- [Hartshorne 1977] R. Hartshorne, *Algebraic Geometry*
- [Katz-Sarnak 1999] N. Katz, P. Sarnak, *Random Matrices, Frobenius Eigenvalues, and Monodromy*
- [Silverman 1994] J. Silverman, *Advanced Topics in the Arithmetic of Elliptic Curves*
