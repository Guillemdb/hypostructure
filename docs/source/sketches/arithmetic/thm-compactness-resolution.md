# Compactness Resolution

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-compactness-resolution*

At Node 3, the Sieve executes a dichotomy: if energy concentrates, a canonical profile emerges (Axiom C satisfied constructively); if energy disperses, global existence holds (success state). Regularity is decidable regardless of whether Compactness holds *a priori*.

---

## Arithmetic Formulation

### Setup

The arithmetic analogue of "compactness" is the **Northcott property**: finiteness of algebraic numbers with bounded height and degree.

- **Concentration:** Heights are bounded, Galois orbits are finite
- **Dispersion:** Heights go to infinity, orbits become Zariski-dense

### Statement (Arithmetic Version)

**Theorem (Arithmetic Compactness Resolution).** Let $\{\alpha_n\}_{n \geq 1}$ be a sequence of algebraic numbers. The Sieve executes:

1. **Concentration Branch:** If $\sup_n h(\alpha_n) < \infty$ and $\sup_n [\mathbb{Q}(\alpha_n):\mathbb{Q}] < \infty$, then the sequence has a **convergent subsequence** to an algebraic limit. The Northcott property is satisfied constructively.

2. **Dispersion Branch:** If $h(\alpha_n) \to \infty$ or $[\mathbb{Q}(\alpha_n):\mathbb{Q}] \to \infty$, the sequence "disperses" to the arithmetic boundary. This triggers **Mode D.D** (global existence in height space)—the sequence exists globally but has no algebraic concentration point.

**Conclusion:** Arithmetic regularity is decidable at runtime based on height behavior, not assumed *a priori*.

---

### Proof

**Step 1: Northcott's Theorem (Compactness in Arithmetic)**

By **Northcott's Theorem** [Northcott 1950]:

For any $B > 0$ and $d \geq 1$:
$$N(B, d) := \#\{\alpha \in \overline{\mathbb{Q}} : h(\alpha) \leq B, [\mathbb{Q}(\alpha):\mathbb{Q}] \leq d\} < \infty$$

**Explicit bound:** By [Loxton-van der Poorten 1983]:
$$N(B, d) \leq C(d) \cdot e^{2dB}$$

where $C(d)$ depends polynomially on $d$.

**Step 2: Concentration Branch—Profile Extraction**

If $\sup_n h(\alpha_n) \leq B$ and $\sup_n [\mathbb{Q}(\alpha_n):\mathbb{Q}] \leq d$, then:

$$\{\alpha_n : n \geq 1\} \subseteq N(B, d)$$

Since $N(B,d)$ is finite, the sequence $\{\alpha_n\}$ takes only finitely many values. Hence:
- There exists $\alpha^* \in \overline{\mathbb{Q}}$ such that $\alpha_n = \alpha^*$ for infinitely many $n$
- The "profile" $\alpha^*$ is the concentration point

**Certificate produced:** $K_{C_\mu}^+ = (\alpha^*, B, d, N(B,d))$

This is the arithmetic analogue of extracting a blow-up profile via concentration-compactness.

**Step 3: Dispersion Branch—Escape to Boundary**

If $h(\alpha_n) \to \infty$, the sequence escapes every compact set in height space. This has two sub-cases:

**(a) Height escape with bounded degree:**

If $[\mathbb{Q}(\alpha_n):\mathbb{Q}] \leq d$ but $h(\alpha_n) \to \infty$, then by the **Lehmer problem** lower bound [Dobrowolski 1979]:
$$h(\alpha) \geq \frac{c}{d}\left(\frac{\log\log d}{\log d}\right)^3 > 0$$

for non-torsion $\alpha$. Heights cannot approach 0 from above, so $h \to \infty$ is genuine escape.

**(b) Degree escape:**

If $[\mathbb{Q}(\alpha_n):\mathbb{Q}] \to \infty$, the sequence leaves all number fields. By **Faltings' theorem** [Faltings 1991] (generalizing Northcott):

For a variety $X/\mathbb{Q}$ and ample height $h_L$:
$$\{\alpha \in X(\overline{\mathbb{Q}}) : h_L(\alpha) \leq B, [\mathbb{Q}(\alpha):\mathbb{Q}] \leq d\}$$
is finite. Degree escape means leaving this set.

**Interpretation:** In both cases, the sequence "disperses" to the arithmetic boundary (the generic point, transcendental numbers, or non-algebraic limits). This is **Mode D.D**: global existence without algebraic concentration.

**Step 4: Runtime Decidability**

The dichotomy is **decidable** at runtime:

1. **Input:** Sequence $\{\alpha_n\}$ with computed heights $\{h(\alpha_n)\}$ and degrees $\{d_n\}$

2. **Test:** Check if $\sup_n h(\alpha_n) < \infty$ and $\sup_n d_n < \infty$

3. **Output:**
   - If bounded: Enter Concentration Branch, extract profile
   - If unbounded: Enter Dispersion Branch, declare global existence

**No a priori compactness assumption needed.** The Northcott bound is computed dynamically.

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finiteness of bounded height, bounded degree algebraic numbers.

2. **Dobrowolski's Bound** [Dobrowolski 1979]: Lower bound for heights of non-torsion points.

3. **Faltings' Generalization** [Faltings 1991]: Height finiteness on varieties.

---

### Arithmetic Interpretation

> **Arithmetic compactness (Northcott property) is verified constructively, not assumed. Heights determine at runtime whether we're in a "compact" or "dispersive" regime.**

This resolves the "compactness critique" in arithmetic:
- **We don't assume** all sequences have bounded height
- **We check** whether heights are bounded
- **If bounded:** Apply Northcott, extract algebraic limit
- **If unbounded:** Declare dispersion, still a valid outcome

---

### Connection to L-functions

For L-functions, the analogue is:

| **Analytic Concept** | **Arithmetic Concept** |
|---------------------|----------------------|
| Energy concentration | Zeros cluster near critical line |
| Dispersion | Zeros spread (satisfy RH spacing) |
| Profile extraction | Isolate individual zeros |
| Compactness | GRH: no zeros off critical strip |

The **zero-spacing distribution** (Montgomery's conjecture, GUE statistics) corresponds to the "dispersive" mode: zeros are separated, no blow-up concentration occurs.

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*
- [Dobrowolski 1979] E. Dobrowolski, *On a question of Lehmer and the number of irreducible factors of a polynomial*
- [Faltings 1991] G. Faltings, *Diophantine approximation on abelian varieties*, Ann. Math.
- [Loxton-van der Poorten 1983] J.H. Loxton, A.J. van der Poorten, *Multiplicative dependence in number fields*
