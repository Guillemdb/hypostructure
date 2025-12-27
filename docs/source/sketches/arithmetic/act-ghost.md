# ACT-Ghost: Derived Extension / BRST

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-ghost*

Ghost/derived structures extend the framework to handle singularities.

---

## Arithmetic Formulation

### Setup

"Ghost/derived extension" in arithmetic means:
- Derived categories handle non-smooth situations
- Tate-Shafarevich as "ghost" points
- Cohomological extensions beyond geometric objects

### Statement (Arithmetic Version)

**Theorem (Arithmetic Derived Extension).** Derived structures extend arithmetic:

1. **Sha as ghost:** Tate-Shafarevich group represents "ghost" rational points
2. **Derived category:** $D^b(\text{Coh}(X))$ extends geometry
3. **Motivic extension:** Motives with modulus handle ramification

---

### Proof

**Step 1: Tate-Shafarevich as Ghost Points**

**Definition:**
$$\text{Ш}(A/K) = \ker\left(H^1(K, A) \to \prod_v H^1(K_v, A)\right)$$

**Ghost interpretation:**
- Elements of Ш are "locally trivial but globally non-trivial"
- They represent torsors that "look like points locally" but aren't globally
- Ghost points: exist in cohomology but not geometrically

**BSD connection:**
$$|E(\mathbb{Q})| \cdot |\text{Ш}(E)| = \text{(both contribute to } L\text{-value)}$$

**Step 2: Derived Category Extension**

**Setup:** Variety $X$ with singularities.

**Problem:** Cohomology of singular $X$ is poorly behaved.

**Extension:** Work in $D^b(\text{Coh}(X))$ (bounded derived category).

**Arithmetic benefits:**
- Grothendieck duality works in derived setting
- Intersection theory extends via derived
- Fourier-Mukai transforms work for singular varieties

**Step 3: Motivic Ghost Extension**

**Problem:** Standard motives don't see ramification.

**Extension:** Motives with modulus [Kahn-Saito-Yamazaki].

**Ghost modulus:** Extra data encoding ramification:
$$h_0^{S}(X, D)$$

where $D$ is an effective divisor (the "ghost" boundary).

**Arithmetic benefit:** Captures wild ramification information.

**Step 4: Derived Galois Cohomology**

**Standard:** $H^i(G_K, M)$ for Galois module $M$.

**Derived extension:** $R\Gamma(G_K, M)$ as object in derived category.

**Ghost classes:** Higher derived functors capture:
- Obstruction to lifting
- Extension classes
- Derived tensor products

**Step 5: BRST-like Structure**

**Physical BRST:** Ghost fields ensure gauge invariance.

**Arithmetic analog:** Descent obstructions ensure Galois invariance.

**Structure:**
- "Physical" = rational points
- "Ghost" = Sha elements
- "BRST cohomology" = Mordell-Weil modulo ghosts

**Exact sequence:**
$$0 \to E(K) \to \text{Sel}(E) \to \text{Ш}(E) \to 0$$

Sel = "total" (physical + ghost), Ш = "pure ghost".

**Step 6: Derived/Ghost Certificate**

The derived/ghost certificate:
$$K_{\text{Ghost}}^+ = (\text{derived object}, \text{ghost contribution}, \text{physical extraction})$$

**Components:**
- **Derived:** Object in derived category or Selmer group
- **Ghost:** Sha or derived correction
- **Physical:** Actual geometric object (MW group, variety)

---

### Key Arithmetic Ingredients

1. **Tate-Shafarevich** [Tate 1958]: Ghost rational points.
2. **Derived Categories** [Verdier 1967]: Extending coherent sheaves.
3. **Motives with Modulus** [KSY 2016]: Ghost ramification data.
4. **Selmer Groups** [Selmer 1951]: Physical + ghost combined.

---

### Arithmetic Interpretation

> **Derived/ghost structures extend arithmetic to handle singularities and obstructions. Sha represents ghost points—locally present but globally absent. Derived categories handle singular varieties. Motives with modulus capture ramification ghosts. This derived extension ensures arithmetic works even in non-smooth situations.**

---

### Literature

- [Tate 1958] J. Tate, *WC-groups over p-adic fields*
- [Verdier 1967] J.-L. Verdier, *Catégories dérivées*
- [Kahn-Saito-Yamazaki 2016] B. Kahn, S. Saito, T. Yamazaki, *Motives with modulus*
- [Selmer 1951] E. Selmer, *The Diophantine equation...*
