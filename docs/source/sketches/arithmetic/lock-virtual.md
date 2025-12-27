# LOCK-Virtual: Virtual Cycle Correspondence

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-virtual*

Virtual fundamental classes provide invariants even for singular/obstructed moduli: they lock the count against deformation.

---

## Arithmetic Formulation

### Setup

"Virtual cycle" in arithmetic means:
- **Moduli:** Space of arithmetic objects (curves, sheaves)
- **Virtual class:** Corrected fundamental class for obstructed moduli
- **Locked invariant:** Deformation-invariant counts

### Statement (Arithmetic Version)

**Theorem (Arithmetic Virtual Lock).** For moduli of arithmetic objects:

1. **Virtual class:** $[\mathcal{M}]^{\text{vir}}$ exists and has expected dimension
2. **Invariance:** Integrals over $[\mathcal{M}]^{\text{vir}}$ are deformation-invariant
3. **Lock:** Virtual invariants lock enumerative counts

---

### Proof

**Step 1: Moduli of Elliptic Curves**

**Moduli space:** $\mathcal{M}_{1,1}$ — moduli of elliptic curves with one marked point.

**Dimension:** Expected dimension matches actual (no obstruction).

**Virtual = actual:** $[\mathcal{M}_{1,1}]^{\text{vir}} = [\mathcal{M}_{1,1}]$.

**Step 2: Compactification**

**Deligne-Mumford:** $\overline{\mathcal{M}}_{g,n}$ — stable curves.

**Boundary:** Nodal curves added at infinity.

**Virtual class:** For $g > 1$, obstruction bundle gives correction:
$$[\overline{\mathcal{M}}_{g,n}]^{\text{vir}} = c_{\text{top}}(\text{Obs}) \cap [\overline{\mathcal{M}}_{g,n}]$$

**Step 3: Gromov-Witten Invariants**

**Definition:** For target $X$ and class $\beta \in H_2(X, \mathbb{Z})$:
$$\langle \gamma_1, \ldots, \gamma_n \rangle_{g,\beta} = \int_{[\overline{\mathcal{M}}_{g,n}(X, \beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i)$$

**Lock:** GW invariants are deformation invariants of $X$.

**Reference:** [Kontsevich-Manin 1994]

**Step 4: Donaldson-Thomas Invariants**

**Moduli of sheaves:** $\mathcal{M}_\beta(X)$ — stable sheaves on CY 3-fold.

**DT invariant:**
$$DT_\beta = \int_{[\mathcal{M}_\beta]^{\text{vir}}} 1$$

**Virtual dimension:** $\text{vd} = 0$ for CY 3-fold.

**Reference:** [Thomas 2000]

**Step 5: Lock via Deformation Invariance**

**Theorem:** For proper family $X_t$:
$$\int_{[\mathcal{M}(X_t)]^{\text{vir}}} \alpha_t = \text{constant}$$

**Proof:** Virtual class constructed from perfect obstruction theory, which is functorial.

**Lock:** Virtual invariants are locked against deformation of target.

**Step 6: Arithmetic Application**

**Moduli of abelian varieties:** $\mathcal{A}_g$ — principally polarized AVs.

**Siegel modular forms:** Sections of automorphic bundles on $\mathcal{A}_g$.

**Virtual aspect:** Intersection numbers on $\overline{\mathcal{A}}_g$ (compactified).

---

### Key Arithmetic Ingredients

1. **Virtual Fundamental Class** [Behrend-Fantechi 1997]: Construction.
2. **Gromov-Witten Theory** [Kontsevich-Manin 1994]: Curve counting.
3. **Donaldson-Thomas Theory** [Thomas 2000]: Sheaf counting.
4. **MNOP Conjecture** [Maulik-Nekrasov-Okounkov-Pandharipande]: GW/DT correspondence.

---

### Arithmetic Interpretation

> **Virtual classes lock arithmetic counts. When moduli spaces are obstructed or singular, the virtual fundamental class provides the "correct" cycle for integration. Virtual invariants (Gromov-Witten, Donaldson-Thomas) are deformation-invariant, locking enumerative geometry against variation of parameters.**

---

### Literature

- [Behrend-Fantechi 1997] K. Behrend, B. Fantechi, *The intrinsic normal cone*
- [Kontsevich-Manin 1994] M. Kontsevich, Y. Manin, *Gromov-Witten classes*
- [Thomas 2000] R. Thomas, *A holomorphic Casson invariant*
- [MNOP 2006] D. Maulik et al., *Gromov-Witten theory and Donaldson-Thomas theory*
