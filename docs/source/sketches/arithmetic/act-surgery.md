# ACT-Surgery: Structural Surgery Principle

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-surgery*

Singular structures can be surgically resolved while preserving essential flow properties.

---

## Arithmetic Formulation

### Setup

"Structural surgery" in arithmetic means:
- **Singularity:** Bad reduction, ramification, poles
- **Resolution:** Semistable reduction, desingularization
- **Preservation:** L-function, rational points, Galois representation

### Statement (Arithmetic Version)

**Theorem (Arithmetic Surgery Principle).** For arithmetic object $X$ with singularities:

1. **Resolution exists:** $\tilde{X} \to X$ resolves singularities
2. **Invariant preservation:** $L(\tilde{X}, s)$ relates to $L(X, s)$ by explicit factors
3. **Rational points:** $\tilde{X}(\mathbb{Q}) \to X(\mathbb{Q})$ is well-controlled

---

### Proof

**Step 1: Semistable Reduction (Grothendieck)**

For abelian variety $A/K$:

**Theorem [Grothendieck 1972]:** There exists finite extension $K'/K$ such that $A_{K'}$ has semistable reduction at all primes.

**Proof sketch:**
1. For each prime $\mathfrak{p}$ of bad reduction, extend $K$ to tame $\mathfrak{p}$
2. Combine extensions: $K' = K(\sqrt[n]{a_1}, \ldots, \sqrt[n]{a_r})$
3. Over $K'$, reduction is either good or multiplicative (semistable)

**Step 2: Resolution of Singularities (Hironaka)**

For variety $X/\mathbb{Q}$:

**Theorem [Hironaka 1964]:** There exists $\pi: \tilde{X} \to X$ with:
- $\tilde{X}$ is smooth
- $\pi$ is birational
- $\pi$ is isomorphism over smooth locus

**Arithmetic refinement:** $\tilde{X}$ can be defined over $\mathbb{Q}$ if $X$ is.

**Step 3: L-function Surgery**

**Before surgery:** $L(X, s) = \prod_p L_p(X, s)$

**After surgery:** $L(\tilde{X}, s) = \prod_p L_p(\tilde{X}, s)$

**Relation:** At good primes (unchanged):
$$L_p(X, s) = L_p(\tilde{X}, s)$$

At bad primes (surgery effect):
$$L_p(\tilde{X}, s) = L_p(X, s) \cdot (\text{correction factor})_p$$

**Explicit formula [Serre 1970]:**
$$\frac{L(\tilde{X}, s)}{L(X, s)} = \prod_{p \in S} \frac{L_p(\tilde{X}, s)}{L_p(X, s)}$$

where $S$ is the singular locus support.

**Step 4: Rational Point Surgery**

**Claim:** Surgery preserves rational point structure.

**For blowup $\tilde{X} \to X$:**
$$\tilde{X}(\mathbb{Q}) \to X(\mathbb{Q}) \text{ is surjective}$$

**Proof:** If $P \in X(\mathbb{Q})$ is smooth, it lifts uniquely. If singular, the exceptional fiber over $P$ contains rational points iff $P$ is rational.

**For base change $X_{K'} \to X$:**
$$X(\mathbb{Q}) = X_{K'}(\mathbb{Q}) \cap X(K)$$

by descent.

**Step 5: Galois Representation Surgery**

**Before:** $\rho_X: G_\mathbb{Q} \to \text{GL}(H^i_{\text{ét}}(X_{\bar{\mathbb{Q}}}, \mathbb{Q}_\ell))$

**After surgery:**
$$\rho_{\tilde{X}} \cong \rho_X \oplus \bigoplus_{E \text{ exceptional}} \rho_E$$

where $\rho_E$ are explicit representations from exceptional divisors.

**Preservation:** The "essential" part $\rho_X$ is preserved; surgery adds known pieces.

**Step 6: Surgery Certificate**

The surgery certificate:
$$K_{\text{Surgery}}^+ = (\pi: \tilde{X} \to X, L\text{-factor corrections}, \text{point map})$$

Components:
- **Resolution map:** $\pi$
- **L-function relation:** Explicit factors at bad primes
- **Rational points:** Descent data

---

### Key Arithmetic Ingredients

1. **Grothendieck's Theorem** [SGA 7]: Semistable reduction for AVs.
2. **Hironaka's Resolution** [Hironaka 1964]: Resolution of singularities.
3. **Serre's L-function Theory** [Serre 1970]: Local factors and products.
4. **Descent Theory** [Weil 1956]: Rational points under field extension.

---

### Arithmetic Interpretation

> **Arithmetic singularities (bad reduction) can be surgically resolved. Semistable reduction, desingularization, and base change are the main tools. L-functions, rational points, and Galois representations are preserved up to explicit, computable corrections.**

---

### Literature

- [Grothendieck 1972] A. Grothendieck, *SGA 7: Groupes de Monodromie*
- [Hironaka 1964] H. Hironaka, *Resolution of singularities*
- [Serre 1970] J.-P. Serre, *Facteurs locaux des fonctions zêta*
- [Weil 1956] A. Weil, *The field of definition of a variety*
