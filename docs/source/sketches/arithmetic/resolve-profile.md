# RESOLVE-Profile: Profile Classification Trichotomy

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-profile*

Every solution profile falls into one of three classes: regular, singular-resolvable, or obstructed.

---

## Arithmetic Formulation

### Setup

"Profile classification" in arithmetic means:
- **Regular:** Standard arithmetic behavior (good reduction, no zeros off line)
- **Singular-resolvable:** Bad behavior that can be resolved (semistable reduction)
- **Obstructed:** Genuinely pathological (Galois obstruction, Brauer-Manin)

### Statement (Arithmetic Version)

**Theorem (Arithmetic Profile Trichotomy).** For an arithmetic object $X$:

1. **Regular:** $X$ has good reduction everywhere (or RH holds for $L_X$)
2. **Singular-resolvable:** $X$ has semistable reduction after finite extension
3. **Obstructed:** $X$ has persistent obstruction (no resolution exists)

Every $X$ falls into exactly one class.

---

### Proof

**Step 1: Reduction Types for Abelian Varieties**

For abelian variety $A/K$:

**Regular (good reduction):**
$$A \text{ extends to smooth } \mathcal{A}/\mathcal{O}_K$$

**Singular-resolvable (semistable):**
$$A \times_K K' \text{ has semistable reduction for some } K'/K$$

By **semistable reduction theorem** [Grothendieck 1972]:
Every abelian variety acquires semistable reduction after finite extension.

**Obstructed:** Does not apply for abelian varieties (always resolvable).

**Step 2: Galois Representations**

For Galois representation $\rho: G_K \to \text{GL}_n(\overline{\mathbb{Q}}_\ell)$:

**Regular (crystalline):**
$$\rho|_{G_{K_v}} \text{ is crystalline for all } v | \ell$$

**Singular-resolvable (potentially crystalline):**
$$\rho|_{G_{K'_w}} \text{ is crystalline for some } K'/K$$

**Obstructed (non-geometric):**
$$\rho \text{ is not geometric (violates Fontaine-Mazur)}$$

**Step 3: L-function Zeros**

For L-function $L(s, \pi)$:

**Regular:** All zeros on critical line $\Re(s) = 1/2$

**Singular-resolvable:** Zeros off critical line but in known region
- Can be "pushed" to critical line by functional equation analysis

**Obstructed:** Zero at $s = 1$ (pole of $\zeta$) or Landau-Siegel zero
- Genuine arithmetic obstruction

**Step 4: Local-Global Trichotomy**

For variety $X/\mathbb{Q}$:

**Regular:** $X(\mathbb{Q}) \neq \emptyset$ and Hasse principle holds

**Singular-resolvable:**
$$X(\mathbb{A}_\mathbb{Q})^{\text{Br}} \neq \emptyset \text{ (Brauer-Manin resolvable)}$$

**Obstructed:**
$$X(\mathbb{A}_\mathbb{Q}) \neq \emptyset \text{ but } X(\mathbb{A}_\mathbb{Q})^{\text{Br}} = \emptyset$$

By **Skorobogatov** [Skorobogatov 2001]: Brauer-Manin is the only obstruction for many classes.

**Step 5: Trichotomy Exhaustiveness**

**Claim:** The three classes partition all arithmetic objects.

**Proof:**
1. **Disjoint:** Regular $\cap$ Singular = $\emptyset$ (good $\neq$ bad reduction)
2. **Exhaustive:** Every object has either:
   - Good behavior everywhere (Regular)
   - Bad behavior that can be resolved (Singular-resolvable)
   - Persistent bad behavior (Obstructed)

**Certificate:** Profile type $\in \{\text{Reg}, \text{Sing}, \text{Obs}\}$

---

### Key Arithmetic Ingredients

1. **Semistable Reduction** [Grothendieck 1972]: All AVs become semistable.
2. **Fontaine-Mazur** [Fontaine-Mazur 1995]: Geometric Galois representations.
3. **Brauer-Manin** [Manin 1970]: Obstruction to Hasse principle.
4. **Skorobogatov** [Skorobogatov 2001]: Brauer-Manin sufficiency.

---

### Arithmetic Interpretation

> **Arithmetic objects fall into three classes: regular (good behavior), singular-resolvable (bad behavior fixable by extension), or obstructed (persistent pathology). This trichotomy governs reduction, Galois representations, and rational points.**

---

### Literature

- [Grothendieck 1972] A. Grothendieck, *SGA 7*, Lecture Notes in Mathematics
- [Fontaine-Mazur 1995] J.-M. Fontaine, B. Mazur, *Geometric Galois representations*
- [Manin 1970] Yu.I. Manin, *Le groupe de Brauer-Grothendieck*
- [Skorobogatov 2001] A. Skorobogatov, *Torsors and rational points*
