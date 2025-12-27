# LOCK-Reconstruction: Structural Reconstruction

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-reconstruction*

Certain invariants uniquely determine structure: reconstruction from invariants locks against ambiguity.

---

## Arithmetic Formulation

### Setup

"Structural reconstruction" in arithmetic means:
- **Invariants:** L-function, Galois representation, height
- **Reconstruction:** Invariants determine object uniquely
- **Lock:** Unique determination eliminates ambiguity

### Statement (Arithmetic Version)

**Theorem (Arithmetic Reconstruction Lock).** Arithmetic objects are determined by invariants:

1. **Faltings:** $E_1 \sim E_2$ (isogenous) $\iff$ $L(E_1, s) = L(E_2, s)$
2. **Strong multiplicity one:** Automorphic representation determined by almost all Hecke eigenvalues
3. **Lock:** Invariants lock isomorphism class

---

### Proof

**Step 1: Faltings Isogeny Theorem**

**Theorem [Faltings 1983]:** For abelian varieties $A, B$ over number field $K$:
$$\text{Hom}_K(A, B) \otimes \mathbb{Q}_\ell \cong \text{Hom}_{G_K}(T_\ell(A), T_\ell(B))$$

**Consequence:** Isogeny class determined by Galois representation.

**L-function version:** $L(A, s) = L(B, s) \iff A \sim B$.

**Step 2: Strong Multiplicity One**

**Theorem [Jacquet-Shalika]:** For cuspidal automorphic representations $\pi, \pi'$ of $GL_n$:
$$a_p(\pi) = a_p(\pi') \text{ for almost all } p \implies \pi \cong \pi'$$

**Reconstruction:** Almost all Hecke eigenvalues determine the form.

**Reference:** [Jacquet-Shalika 1981]

**Step 3: Modularity and Reconstruction**

**Modularity [Wiles et al.]:** $E/\mathbb{Q}$ corresponds to weight-2 newform $f_E$.

**Reconstruction chain:**
$$E \to \rho_E \to L(E, s) \to f_E \to E$$

**Lock:** Any step determines all others (up to isogeny/twist).

**Step 4: Serre's Conjecture (Proved)**

**Theorem [Khare-Wintenberger 2009]:** Every odd irreducible $\bar{\rho}: G_\mathbb{Q} \to GL_2(\bar{\mathbb{F}}_p)$ is modular.

**Reconstruction:** Mod $p$ representation $\to$ modular form $\to$ geometric object.

**Step 5: Height Reconstruction**

**For CM elliptic curves:** Heights of special values reconstruct curve.

**Gross-Zagier:** $L'(E, 1) \neq 0 \implies$ Heegner point non-torsion.

**Reconstruction:** L-value $\to$ height $\to$ point $\to$ curve (via moduli).

**Step 6: Reconstruction Certificate**

$$K_{\text{Recon}}^+ = (\text{invariant set}, \text{reconstruction theorem}, \text{object class})$$

Examples:
- (Galois rep, Faltings, isogeny class of AV)
- (Hecke eigenvalues, strong mult. one, automorphic rep)

---

### Key Arithmetic Ingredients

1. **Faltings' Theorem** [Faltings 1983]: Tate conjecture for abelian varieties.
2. **Strong Multiplicity One** [Jacquet-Shalika]: Uniqueness of automorphic forms.
3. **Modularity** [Wiles 1995]: Elliptic curves are modular.
4. **Serre's Conjecture** [Khare-Wintenberger 2009]: Mod $p$ representations are modular.

---

### Arithmetic Interpretation

> **Arithmetic invariants lock objects. Faltings' theorem says Galois representations determine abelian varieties up to isogeny. Strong multiplicity one says Hecke eigenvalues determine modular forms. These reconstruction theorems lock against ambiguity: knowing the invariants is knowing the object.**

---

### Literature

- [Faltings 1983] G. Faltings, *Endlichkeitssätze für abelsche Varietäten*
- [Jacquet-Shalika 1981] H. Jacquet, J. Shalika, *On Euler products and the classification of automorphic forms*
- [Wiles 1995] A. Wiles, *Modular elliptic curves and Fermat's last theorem*
- [Khare-Wintenberger 2009] C. Khare, J.-P. Wintenberger, *Serre's modularity conjecture*
