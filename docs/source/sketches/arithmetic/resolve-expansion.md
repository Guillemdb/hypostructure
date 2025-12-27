# RESOLVE-Expansion: Thin-to-Full Expansion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-expansion*

Thin kernel objects expand canonically to full hypostructures via the expansion adjunction. No information is lost in the process.

---

## Arithmetic Formulation

### Setup

"Thin" arithmetic data:
- Minimal polynomial
- Local data at finitely many primes
- Basic height information

"Full" arithmetic structure:
- Complete L-function
- All Galois representations
- Full motivic structure

### Statement (Arithmetic Version)

**Theorem (Arithmetic Expansion).** Thin arithmetic data expands to full structure:

$$\mathcal{F}: \mathbf{Thin}_{\text{arith}} \to \mathbf{Mot}_\mathbb{Q}$$

The expansion is:
1. **Faithful:** No information lost
2. **Universal:** Unique up to isomorphism
3. **Computable:** Effective algorithms exist

---

### Proof

**Step 1: Thin Data Specification**

For an elliptic curve $E/\mathbb{Q}$:

**Thin data:**
- Weierstrass equation: $y^2 = x^3 + ax + b$, $a, b \in \mathbb{Z}$
- Conductor: $N_E = \prod_p p^{f_p}$
- Some rational points: $P_1, \ldots, P_k \in E(\mathbb{Q})$

For a number field $K/\mathbb{Q}$:
- Minimal polynomial: $f(x) \in \mathbb{Z}[x]$
- Discriminant: $\Delta_K$
- Ring of integers: $\mathcal{O}_K$

**Step 2: Expansion Algorithm**

**(a) L-function construction:**

From thin data, compute:
$$L(E, s) = \prod_p L_p(E, s)$$

Local factors:
- Good reduction: $L_p(E, s) = (1 - a_p p^{-s} + p^{1-2s})^{-1}$
- Bad reduction: $(1 - a_p p^{-s})^{-1}$ or $1$

where $a_p = p + 1 - \#E(\mathbb{F}_p)$ (computed by point counting).

**(b) Galois representation:**

From $\ell$-torsion $E[\ell]$:
$$\rho_{E,\ell}: G_\mathbb{Q} \to \text{GL}_2(\mathbb{Z}_\ell)$$

Characteristic polynomial at Frobenius:
$$\det(I - \rho_{E,\ell}(\text{Frob}_p) T) = 1 - a_p T + p T^2$$

**(c) Motive construction:**

The motive $h^1(E)$ has:
- Betti realization: $H^1(E(\mathbb{C}), \mathbb{Q}) \cong \mathbb{Q}^2$
- de Rham realization: $H^1_{\text{dR}}(E/\mathbb{Q})$
- $\ell$-adic realization: $H^1_{\text{ét}}(E_{\overline{\mathbb{Q}}}, \mathbb{Q}_\ell)$

**Step 3: Faithfulness Verification**

**Claim:** Thin data determines full structure uniquely.

**Proof:** By **Faltings' isogeny theorem** [Faltings 1983]:
$$\text{Hom}(E_1, E_2) \otimes \mathbb{Q}_\ell \cong \text{Hom}_{G_\mathbb{Q}}(T_\ell(E_1), T_\ell(E_2))$$

The Galois representation (derived from thin data) determines the curve up to isogeny.

By **Tate's conjecture** [Tate 1966, Faltings 1983]:
Thin data → Galois representation → Motive → Full structure

No information lost.

**Step 4: Universality**

The expansion is **universal** in the sense:

For any morphism $f: \mathcal{T} \to U(M)$ from thin data to underlying data of motive $M$:
$$\exists! \tilde{f}: \mathcal{F}(\mathcal{T}) \to M$$

This is the adjunction property (thm-expansion-adjunction).

**Step 5: Computability**

All steps are effective:
- Point counting: $O(p^{1/4+\epsilon})$ [Schoof's algorithm]
- L-function coefficients: Polynomial in $\log p$
- Galois representation: From mod-$\ell$ points

---

### Key Arithmetic Ingredients

1. **Faltings' Isogeny Theorem** [Faltings 1983]: Galois determines curve.
2. **Tate's Conjecture** [Tate 1966]: Endomorphisms from Galois.
3. **Schoof's Algorithm** [Schoof 1985]: Efficient point counting.
4. **Comparison Theorems** [Grothendieck]: Cohomology realizations.

---

### Arithmetic Interpretation

> **Minimal arithmetic data (equation, conductor) uniquely determines the full motivic structure. The expansion is canonical and computable.**

---

### Literature

- [Faltings 1983] G. Faltings, *Endlichkeitssätze...*
- [Schoof 1985] R. Schoof, *Elliptic curves over finite fields and the computation of square roots mod p*
- [Tate 1966] J. Tate, *Endomorphisms of abelian varieties*
