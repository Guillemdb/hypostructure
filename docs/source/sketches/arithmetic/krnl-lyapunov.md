# KRNL-Lyapunov: Canonical Lyapunov Functional

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-lyapunov*

Given validated interface permits for dissipation, compactness, and local stiffness, there exists a canonical Lyapunov functional with explicit construction via optimal transport cost to equilibrium.

---

## Arithmetic Formulation

### Setup

The arithmetic Lyapunov functional is the **canonical height**:
- **State space:** $X(\overline{\mathbb{Q}})$ = algebraic points on a variety $X/\mathbb{Q}$
- **Equilibrium set:** $M$ = special points (torsion, CM points, rational points)
- **Lyapunov functional:** $\hat{h}: X(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$ = canonical (Néron-Tate) height

### Statement (Arithmetic Version)

**Theorem (Canonical Height as Lyapunov Functional).** Let $A/\mathbb{Q}$ be an abelian variety with Néron-Tate height $\hat{h}: A(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$. Given:

1. **Dissipation ($D_E$):** $\hat{h}(P) \geq 0$ with $\hat{h}(P) = 0 \Leftrightarrow P \in A_{\text{tors}}$
2. **Compactness ($C_\mu$):** For any $B > 0$, the set $\{P : \hat{h}(P) \leq B, [k(P):\mathbb{Q}] \leq d\}$ is finite
3. **Stiffness ($\mathrm{LS}_\sigma$):** The height pairing $\langle \cdot, \cdot \rangle: A(\overline{\mathbb{Q}}) \times A(\overline{\mathbb{Q}}) \to \mathbb{R}$ is non-degenerate on $A(\mathbb{Q})/A_{\text{tors}}$

Then:
1. **Monotonicity:** Under any height-decreasing operation, $\hat{h}$ decreases strictly outside $A_{\text{tors}}$
2. **Stability:** $\hat{h}$ attains its minimum precisely on $M = A_{\text{tors}}$
3. **Height Equivalence:** $\hat{h}(P) - 0 \asymp h(P)$ for Weil height $h$
4. **Uniqueness:** Any other quadratic form with these properties equals $\hat{h}$ (up to scaling)

---

### Proof

**Step 1: Construction of Canonical Height**

For an abelian variety $A/\mathbb{Q}$ with symmetric ample line bundle $L$, the **Néron-Tate height** is defined by:
$$\hat{h}_L(P) = \lim_{n \to \infty} \frac{h_L([n]P)}{n^2}$$

where $[n]: A \to A$ is multiplication by $n$ and $h_L$ is the Weil height associated to $L$.

By **Néron's theorem** [Néron 1965] and **Tate's construction** [Tate 1965]:
- The limit exists and is independent of $L$ up to bounded function
- $\hat{h}_L: A(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$ is a quadratic form

**Step 2: Verification of Monotonicity**

Define the "flow" as the multiplication-by-$n$ map. For $P \notin A_{\text{tors}}$:
$$\hat{h}([n]P) = n^2 \cdot \hat{h}(P)$$

**Interpretation:** Under the inverse operation $[n]^{-1}$ (when defined):
$$\hat{h}([n]^{-1}P) = \frac{\hat{h}(P)}{n^2} < \hat{h}(P)$$

This is the arithmetic analogue of energy dissipation: repeatedly taking $n$-th roots decreases height.

More precisely, under **reduction modulo $p$**:
$$\hat{h}(P) = h(P) + O(1)$$
and the local height at $p$ measures arithmetic "dissipation":
$$\lambda_p(P) = -\log |P \mod p|_p$$

**Step 3: Stability at Torsion Points**

**Claim:** $\hat{h}(P) = 0 \Leftrightarrow P \in A_{\text{tors}}$

**Proof:**
$(\Rightarrow)$ If $\hat{h}(P) = 0$, then $\hat{h}([n]P) = n^2 \cdot 0 = 0$ for all $n$. The set $\{[n]P : n \in \mathbb{Z}\}$ has bounded Weil height (since $h = \hat{h} + O(1)$). By Northcott, this set is finite. Hence the orbit is finite, so $P$ is torsion.

$(\Leftarrow)$ If $P \in A_{\text{tors}}$, say $[m]P = 0$, then:
$$m^2 \cdot \hat{h}(P) = \hat{h}([m]P) = \hat{h}(0) = 0$$
Hence $\hat{h}(P) = 0$. $\square$

**Equilibrium:** $M = A_{\text{tors}} = \ker(\hat{h})$

**Step 4: Height Equivalence**

By the **Néron-Tate limit construction**:
$$\hat{h}(P) = h_L(P) + O(1)$$

where the $O(1)$ is uniform over $A(\overline{\mathbb{Q}})$. Hence:
$$\hat{h}(P) \asymp h_L(P) - h_L(0)$$

for the normalized Weil height.

**Step 5: Uniqueness (Non-degeneracy)**

The height pairing:
$$\langle P, Q \rangle = \frac{1}{2}\left(\hat{h}(P+Q) - \hat{h}(P) - \hat{h}(Q)\right)$$

is a **symmetric bilinear form** on $A(\overline{\mathbb{Q}}) \otimes \mathbb{R}$.

By the **Mordell-Weil theorem** [Mordell 1922, Weil 1928]:
$$A(\mathbb{Q}) \cong \mathbb{Z}^r \oplus A(\mathbb{Q})_{\text{tors}}$$

By **Néron's theorem** [Néron 1965]:
$$\langle \cdot, \cdot \rangle \text{ is non-degenerate on } A(\mathbb{Q}) / A(\mathbb{Q})_{\text{tors}}$$

**Uniqueness:** Any other quadratic form $\hat{h}'$ satisfying the same properties must equal $\hat{h}$ on $A(\mathbb{Q})/A_{\text{tors}}$ by non-degeneracy. Extension to $A(\overline{\mathbb{Q}})$ is unique by the Galois equivariance principle (KRNL-Equivariance).

---

### Explicit Construction: Optimal Transport Interpretation

The hypostructure formula:
$$\mathcal{L}(x) = \inf\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\}$$

translates arithmetically to:

$$\hat{h}(P) = \inf\left\{h(Q) + d_{\text{height}}(P, Q) : Q \in A_{\text{tors}}\right\}$$

where $d_{\text{height}}$ is the **height distance**:
$$d_{\text{height}}(P, Q) = \hat{h}(P - Q)$$

**Verification:** For $Q \in A_{\text{tors}}$:
$$h(Q) + \hat{h}(P - Q) = 0 + \hat{h}(P) = \hat{h}(P)$$

The infimum is achieved at any torsion point, giving $\mathcal{L}(P) = \hat{h}(P)$.

---

### Key Arithmetic Ingredients

1. **Néron-Tate Height** [Néron 1965, Tate 1965]: Canonical quadratic height on abelian varieties.

2. **Mordell-Weil Theorem** [Mordell 1922]: Finite generation of rational points.

3. **Northcott's Theorem** [Northcott 1950]: Finiteness from bounded height.

4. **Height Pairing Non-degeneracy** [Néron 1965]: Ensures uniqueness.

---

### Arithmetic Interpretation

> **The Néron-Tate height is the canonical "energy" functional in arithmetic: it measures distance from torsion, is quadratic, Galois-invariant, and uniquely determined by these properties.**

The Lyapunov perspective reveals:
- **Torsion = equilibria:** Zero height = zero energy
- **Height descent:** Arithmetic operations that decrease height are "dissipative"
- **Regularity:** Points of bounded height satisfy arithmetic compactness

---

### Literature

- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs sur les variétés abéliennes*, Ann. Math.
- [Tate 1965] J. Tate, *On the conjectures of Birch and Swinnerton-Dyer* (height construction)
- [Mordell 1922] L.J. Mordell, *On the rational solutions of the indeterminate equations of the third and fourth degrees*
- [Hindry-Silverman 2000] M. Hindry, J. Silverman, *Diophantine Geometry*, Ch. B (heights)
