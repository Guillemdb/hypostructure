# ACT-Align: Adjoint Surgery

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-align*

Adjoint operations align structures for surgery.

---

## Arithmetic Formulation

### Setup

"Adjoint surgery" in arithmetic means:
- Dual operations that reverse direction
- Adjoint functors relating arithmetic categories
- Alignment via adjunction before surgery

### Statement (Arithmetic Version)

**Theorem (Arithmetic Adjoint Alignment).** Adjoint operations align arithmetic:

1. **Dual isogeny:** $\phi: A \to B$ has dual $\hat{\phi}: B \to A$
2. **Restriction-induction:** $(f^*, f_*)$ adjoint pair for morphisms
3. **Verdier duality:** Aligns cohomology for surgery

---

### Proof

**Step 1: Dual Isogeny Alignment**

**Setup:** Isogeny $\phi: A \to B$ of abelian varieties.

**Dual isogeny:** $\hat{\phi}: \hat{B} \to \hat{A}$ where $\hat{A} = \text{Pic}^0(A)$.

**Adjoint property:**
$$\langle \phi(a), b \rangle_B = \langle a, \hat{\phi}(b) \rangle_A$$

for Weil pairings.

**Alignment for surgery:**
- $\phi \circ \hat{\phi} = [\deg \phi]$ on $B$
- $\hat{\phi} \circ \phi = [\deg \phi]$ on $A$
- Dual aligns source and target for composition

**Step 2: Restriction-Induction Adjunction**

**Setup:** Field extension $L/K$.

**Restriction:** $\text{Res}_{L/K}: \text{Rep}(G_L) \to \text{Rep}(G_K)$

**Induction:** $\text{Ind}_{L/K}: \text{Rep}(G_K) \to \text{Rep}(G_L)$

**Adjunction (Frobenius reciprocity):**
$$\text{Hom}_{G_K}(\text{Ind}_{L/K} V, W) \cong \text{Hom}_{G_L}(V, \text{Res}_{L/K} W)$$

**Alignment:** Before base change surgery, align via restriction-induction.

**Step 3: Verdier Duality**

**Setup:** Derived category $D^b_c(X, \mathbb{Q}_\ell)$.

**Dualizing functor:** $\mathbb{D} = R\mathcal{H}om(-, \omega_X)$

**Adjunction:**
$$\text{Hom}(A, \mathbb{D}B) \cong \text{Hom}(B, \mathbb{D}A)^*$$

**Alignment for cohomology surgery:**
- Poincaré duality: $H^i(X) \cong H^{2n-i}(X)^*$
- Lefschetz duality for pairs
- Enables cutting/gluing in cohomology

**Step 4: Height Pairing Alignment**

**Néron-Tate pairing:** $\langle \cdot, \cdot \rangle: E(\mathbb{Q}) \times E(\mathbb{Q}) \to \mathbb{R}$

**Self-adjoint:** $\langle P, Q \rangle = \langle Q, P \rangle$

**Alignment for height surgery:**
- Height is quadratic form
- Pairing diagonalizes on MW lattice
- Aligned basis enables regulator computation

**Step 5: Langlands Duality**

**Dual group:** $G \mapsto {}^L G$ (Langlands dual)

**Example:** $\text{SL}_n \leftrightarrow \text{PGL}_n$

**Adjunction in Langlands:**
$$L(s, \rho) = L(s, {}^L\rho)$$

**Alignment:** Dual groups align automorphic and Galois sides.

**Step 6: Adjoint Alignment Certificate**

The adjoint alignment certificate:
$$K_{\text{Adj}}^+ = (\text{operation}, \text{adjoint}, \text{alignment achieved})$$

**Components:**
- **Operation:** Forward map (isogeny, induction, pullback)
- **Adjoint:** Reverse map (dual isogeny, restriction, pushforward)
- **Alignment:** How adjunction prepares for surgery

**Examples:**
| Operation | Adjoint | Alignment |
|-----------|---------|-----------|
| $\phi: A \to B$ | $\hat{\phi}: B \to A$ | $\phi \hat{\phi} = [n]$ |
| $\text{Ind}_{L/K}$ | $\text{Res}_{L/K}$ | Frobenius reciprocity |
| $f^*$ | $f_*$ | Projection formula |

---

### Key Arithmetic Ingredients

1. **Dual Isogeny** [Mumford 1970]: Duality for abelian varieties.
2. **Frobenius Reciprocity** [Frobenius 1898]: Induction-restriction adjunction.
3. **Verdier Duality** [Verdier 1967]: Derived category duality.
4. **Langlands Dual** [Langlands 1967]: Dual group correspondence.

---

### Arithmetic Interpretation

> **Adjoint operations align arithmetic structures for surgery. Dual isogenies provide canonical reversals, restriction-induction adjunction aligns Galois actions, Verdier duality aligns cohomology. This alignment is the preparation step before performing controlled surgery on arithmetic objects.**

---

### Literature

- [Mumford 1970] D. Mumford, *Abelian Varieties*
- [Frobenius 1898] F.G. Frobenius, *Über Relationen zwischen den Charakteren einer Gruppe*
- [Verdier 1967] J.-L. Verdier, *Catégories dérivées*
- [Langlands 1967] R.P. Langlands, *Letter to A. Weil*
