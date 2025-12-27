# RESOLVE-Tower: Soft Local Tower Globalization

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-tower*

Local resolution data can be globalized via tower constructions: if each local piece admits a resolution, and compatibility conditions are satisfied, then a global resolution exists.

---

## Arithmetic Formulation

### Setup

"Tower globalization" in arithmetic means:
- **Local data:** Resolution at each prime $\mathfrak{p}$
- **Tower:** Extension tower $K \subset K_1 \subset K_2 \subset \cdots$
- **Globalization:** Local-to-global principle for resolution

### Statement (Arithmetic Version)

**Theorem (Arithmetic Tower Globalization).** Let $A/K$ be an abelian variety with bad reduction at $S$. Then:

1. **Local resolution:** For each $\mathfrak{p} \in S$, there exists $K_\mathfrak{p}/K$ local extension such that $A_{K_\mathfrak{p}}$ has semistable reduction
2. **Tower compatibility:** The extensions $K_\mathfrak{p}$ can be chosen compatibly
3. **Global resolution:** There exists finite $K'/K$ with $A_{K'}$ semistable everywhere

---

### Proof

**Step 1: Néron-Ogg-Shafarevich Criterion**

**Theorem [Serre-Tate 1968]:** $A/K$ has good reduction at $\mathfrak{p}$ iff:
$$\text{Gal}(\bar{K}_\mathfrak{p}/K_\mathfrak{p}^{\text{unr}}) \text{ acts trivially on } T_\ell(A)$$

for $\ell \neq \text{char}(\kappa_\mathfrak{p})$.

**Step 2: Local Semistable Reduction**

**Theorem [Grothendieck 1972]:** For each $\mathfrak{p}$, there exists finite extension $K_\mathfrak{p}/K$ (tamely ramified of degree dividing $n!$ where $n = \dim A$) such that $A_{K_\mathfrak{p}}$ has semistable reduction.

**Local tower:**
$$K \subset K_\mathfrak{p}^{\text{tame}} \subset K_\mathfrak{p}$$

**Step 3: Compatibility via Weak Approximation**

**Claim:** Local extensions can be chosen compatibly.

**Proof:** For each $\mathfrak{p} \in S$:
- Choose $K_\mathfrak{p}$ minimal with semistable reduction
- By class field theory, these embed in a common extension
- Use weak approximation to find global $K'$ approximating all $K_\mathfrak{p}$

**Step 4: Composite Extension**

**Global extension:**
$$K' = \prod_{\mathfrak{p} \in S} K_\mathfrak{p}$$

(compositum of local extensions)

**Degree bound:**
$$[K' : K] \leq \prod_{\mathfrak{p} \in S} [K_\mathfrak{p} : K] \leq (n!)^{|S|}$$

**Step 5: Tower Structure**

**Resolution tower:**
$$K = K_0 \subset K_1 \subset \cdots \subset K_r = K'$$

where each $K_i/K_{i-1}$ resolves one prime.

**Height control:** Heights in tower satisfy:
$$h_{K_i}(\alpha) = [K_i : K_{i-1}] \cdot h_{K_{i-1}}(\alpha)$$

**Step 6: Globalization Certificate**

The globalization certificate:
$$K_{\text{Tower}}^+ = (K'/K, \{K_\mathfrak{p}\}_{\mathfrak{p} \in S}, \text{semistability})$$

---

### Key Arithmetic Ingredients

1. **Néron-Ogg-Shafarevich** [Serre-Tate 1968]: Good reduction criterion.
2. **Grothendieck's Semistable Reduction** [SGA 7]: Local resolution.
3. **Weak Approximation** [Artin-Whaples]: Global from local.
4. **Class Field Theory** [Artin]: Abelian extensions.

---

### Arithmetic Interpretation

> **Local arithmetic resolutions can be globalized. Semistable reduction at each bad prime can be achieved by local extensions, and these local extensions combine into a global tower $K'/K$ where all singularities are resolved. This is the arithmetic analogue of resolving singularities patch by patch.**

---

### Literature

- [Serre-Tate 1968] J.-P. Serre, J. Tate, *Good reduction of abelian varieties*
- [Grothendieck 1972] A. Grothendieck, *SGA 7: Groupes de Monodromie*
- [Artin-Whaples 1945] E. Artin, G. Whaples, *Axiomatic characterization of fields*
- [Silverman 1994] J. Silverman, *Advanced Topics in the Arithmetic of Elliptic Curves*
