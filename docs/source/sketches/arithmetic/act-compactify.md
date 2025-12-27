# ACT-Compactify: Lyapunov Compactification

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-compactify*

Lyapunov functions compactify configuration spaces: adding ideal points at infinity where the Lyapunov function tends to its extremes.

---

## Arithmetic Formulation

### Setup

"Lyapunov compactification" in arithmetic means:
- **Lyapunov:** Height function as energy
- **Compactification:** Adding cusps/boundary to moduli
- **Control:** Height controls behavior at infinity

### Statement (Arithmetic Version)

**Theorem (Arithmetic Lyapunov Compactification).** Height functions compactify arithmetic spaces:

1. **Height Lyapunov:** $h: X(\bar{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$ bounded below
2. **Compactification:** $\{h \leq B\}$ is finite (Northcott)
3. **Boundary:** Points at "$h = \infty$" form cusp/boundary divisor

---

### Proof

**Step 1: Height as Lyapunov Function**

**Properties:** For $h: X(\bar{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$:
- $h(P) \geq 0$ (non-negative)
- $h(P) = 0 \iff P$ is torsion/special
- Finiteness of sublevel sets (Northcott)

**Lyapunov analogy:** Height measures "distance from equilibrium" (algebraic points).

**Step 2: Northcott Compactness**

**Theorem [Northcott]:**
$$\#\{P \in X(\bar{\mathbb{Q}}) : h(P) \leq B, [\mathbb{Q}(P):\mathbb{Q}] \leq d\} < \infty$$

**Compactness:** Sublevel sets are "compact" in arithmetic sense.

**Step 3: Borel-Serre Compactification**

**Modular curves:** $\mathcal{H}/\Gamma$ compactified by adding cusps.

**Height at cusps:** $h \to \infty$ as $\tau \to$ cusp.

**Compactification:**
$$\overline{X_\Gamma} = X_\Gamma \cup \{\text{cusps}\}$$

**Reference:** [Borel-Serre 1973]

**Step 4: Satake Compactification**

**Siegel modular varieties:** $\mathcal{A}_g$ compactified.

**Boundary:** Degenerate abelian varieties at infinity.

**Height:** Faltings height $h_F(A) \to \infty$ at boundary.

**Reference:** [Satake 1960]

**Step 5: Toroidal Compactification**

**Arithmetic variety:** $\bar{X}$ smooth compactification of $X$.

**Boundary divisor:** $D = \bar{X} \setminus X$

**Height control:**
$$h(P) \to \infty \text{ as } P \to D$$

**Compactification certificate:** $(\bar{X}, D, h)$

**Step 6: Lyapunov Compactification Certificate**

$$K_{\text{Compact}}^+ = (X, h, \bar{X}, \text{boundary behavior})$$

**Components:**
- Original space $X$
- Height function $h$
- Compactification $\bar{X}$
- Boundary = $\{h = \infty\}$

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finite sublevel sets.
2. **Borel-Serre Compactification** [1973]: Adding cusps.
3. **Satake Compactification** [1960]: Siegel modular varieties.
4. **Toroidal Compactification** [AMRT 1975]: Smooth completion.

---

### Arithmetic Interpretation

> **Height functions compactify arithmetic spaces. The Northcott property ensures finiteness of bounded-height sets, analogous to compact sublevel sets of a Lyapunov function. Cusps, boundary divisors, and degenerate varieties form the "infinity" added in compactification. The height blows up at these boundaries, completing the analogy with Lyapunov compactification in dynamics.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic*
- [Borel-Serre 1973] A. Borel, J.-P. Serre, *Corners and arithmetic groups*
- [Satake 1960] I. Satake, *On compactifications of the quotient spaces*
- [AMRT 1975] A. Ash, D. Mumford, M. Rapoport, Y.-S. Tai, *Smooth Compactifications of Locally Symmetric Varieties*
