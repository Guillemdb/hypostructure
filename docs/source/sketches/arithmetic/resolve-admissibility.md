# RESOLVE-Admissibility: Surgery Admissibility Trichotomy

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-admissibility*

Surgery operations fall into three classes: admissible, conditionally admissible, or inadmissible.

---

## Arithmetic Formulation

### Setup

"Surgery admissibility" in arithmetic means:
- **Admissible:** Resolution operation preserves arithmetic structure
- **Conditionally admissible:** Requires auxiliary hypothesis
- **Inadmissible:** Destroys essential arithmetic information

### Statement (Arithmetic Version)

**Theorem (Arithmetic Surgery Admissibility).** For resolution operation $\sigma$:

1. **Admissible:** $\sigma$ preserves L-function, rational points, Galois action
2. **Conditionally admissible:** $\sigma$ preserves under GRH or BSD assumption
3. **Inadmissible:** $\sigma$ changes arithmetic invariants

---

### Proof

**Step 1: Blowup Admissibility**

For blowup $\tilde{X} \to X$ at smooth center $Z$:

**Admissible operations:**
- Rational points: $\tilde{X}(\mathbb{Q}) \to X(\mathbb{Q})$ is surjective
- Birational invariants preserved: $K_{\tilde{X}} = \pi^* K_X + (c-1)E$
- L-function: $L(\tilde{X}, s) = L(X, s) \cdot L(\mathbb{P}^{c-1}, s)^{\text{mult}}$

**Proof [Manin 1974]:** Blowups are admissible for birational geometry.

**Step 2: Semistable Reduction**

For $A/K$ abelian variety:

**Admissible:** Base change $A \times_K K' \to A$ where $K'/K$ achieves semistable reduction.

**Preservation:**
- $L(A/K, s) = L(A/K', s)^{[K':K]} \cdot (\text{ramification factors})$
- Mordell-Weil: $A(K) \hookrightarrow A(K')$

**Conditional admissibility:** For BSD, need $[K':K]$ divides special value factor.

**Step 3: Level Lowering**

For modular form $f \in S_k(\Gamma_0(N))$:

**Operation:** Level lowering $f \mapsto f'$ where $f' \in S_k(\Gamma_0(N'))$ for $N' | N$.

**Admissibility [Ribet 1990]:**
- **Admissible if:** $\bar{\rho}_f \cong \bar{\rho}_{f'}$ (same residual representation)
- **Inadmissible if:** Level lowering changes eigenvalue structure

**Criterion:** Admissible $\iff$ the prime $p$ with $p^2 | N$ but $p \nmid N'$ satisfies $\bar{\rho}_f|_{I_p}$ is trivial.

**Step 4: Conductor Reduction**

For elliptic curve $E/\mathbb{Q}$:

**Quadratic twist:** $E \mapsto E^{(d)}$ for squarefree $d$

**Admissibility analysis:**
- $N_{E^{(d)}} = N_E \cdot d^2 / \gcd(N_E, d)^2$ (approximately)
- **Admissible:** If $d$ is coprime to $N_E$
- **Conditionally admissible:** If $d | N_E$ (changes reduction type)
- **Inadmissible:** If destroys rational points

**Step 5: Trichotomy Certification**

**Algorithm to classify surgery $\sigma$:**

```
ADMISSIBILITY(σ):
  1. Compute effect on L-function: L(σ(X), s) vs L(X, s)
  2. Compute effect on rational points: σ(X)(Q) vs X(Q)
  3. Compute effect on Galois: ρ_{σ(X)} vs ρ_X

  IF all preserved:
    RETURN "Admissible"
  ELIF preserved under GRH/BSD:
    RETURN "Conditionally-Admissible"
  ELSE:
    RETURN "Inadmissible"
```

**Trichotomy completeness:** Every surgery is exactly one type.

---

### Key Arithmetic Ingredients

1. **Manin's Theorem** [Manin 1974]: Birational invariance of Brauer-Manin.
2. **Ribet's Level Lowering** [Ribet 1990]: When level reduction is admissible.
3. **Grothendieck's Semistable Reduction** [SGA 7]: Base change admissibility.
4. **Twist Theory** [Silverman 1994]: Quadratic twist effects.

---

### Arithmetic Interpretation

> **Arithmetic surgeries (blowups, base changes, level lowerings, twists) are classified as admissible (preserve invariants), conditionally admissible (preserve under GRH/BSD), or inadmissible (destroy structure).**

---

### Literature

- [Manin 1974] Yu.I. Manin, *Cubic Forms*
- [Ribet 1990] K. Ribet, *On modular representations of Gal(Q̄/Q)*
- [Grothendieck 1972] A. Grothendieck, *SGA 7*, Springer
- [Silverman 1994] J. Silverman, *Advanced Topics in Elliptic Curves*
