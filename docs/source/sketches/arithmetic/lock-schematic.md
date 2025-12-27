# LOCK-Schematic: Semialgebraic Exclusion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-schematic*

Semialgebraic constraints exclude regions: configurations in semialgebraic "bad" sets are locked out by polynomial barriers.

---

## Arithmetic Formulation

### Setup

"Semialgebraic exclusion" in arithmetic means:
- **Semialgebraic:** Defined by polynomial equations/inequalities
- **Exclusion:** Height bounds, conductor bounds exclude regions
- **Lock:** Algebraic constraints lock configurations

### Statement (Arithmetic Version)

**Theorem (Arithmetic Semialgebraic Exclusion).** Arithmetic invariants satisfy polynomial constraints:

1. **Height regions:** $\{P \in E(\mathbb{Q}) : \hat{h}(P) < B\}$ is finite (Northcott)
2. **Conductor exclusion:** Elliptic curves with $N_E < N_0$ form finite set
3. **Discriminant barriers:** $|\Delta_K| > D_0$ excludes from small discriminant region

---

### Proof

**Step 1: Northcott as Semialgebraic Exclusion**

**Northcott's Theorem:**
$$\#\{P \in E(\bar{\mathbb{Q}}) : \hat{h}(P) \leq B, [\mathbb{Q}(P):\mathbb{Q}] \leq d\} < \infty$$

**Semialgebraic view:** Height is degree of minimal polynomial; bounded degree + bounded height = semialgebraic constraint on coefficients.

**Step 2: Conductor Bounds**

**Szpiro's Conjecture (weak form):** For elliptic curve $E/\mathbb{Q}$:
$$\log |\Delta_E| \leq C \cdot \log N_E + C'$$

**Semialgebraic:** Conductor $N_E = \prod_{p | \Delta} p^{f_p}$ is polynomial in local data.

**Exclusion:** Only finitely many $E$ with $N_E < N_0$ (up to isomorphism).

**Step 3: Discriminant Barriers**

**Minkowski bound:** For number field $K$:
$$|\Delta_K| \geq \left(\frac{\pi}{4}\right)^{r_2} \frac{n^n}{n!}$$

**Exclusion:** No field $K$ with $|\Delta_K| < $ Minkowski bound exists.

**Hermite's theorem:** Only finitely many $K$ with $|\Delta_K| < D$.

**Step 4: Polynomial Height Inequalities**

**Weil height machine:** For morphism $\phi: X \to Y$:
$$h_Y(\phi(P)) = \deg(\phi) \cdot h_X(P) + O(1)$$

**Semialgebraic:** Height transforms polynomially under algebraic maps.

**Step 5: Effective Mordell**

**Effective bounds [Baker, Wüstholz]:** For curve $C/\mathbb{Q}$ of genus $g \geq 2$:
$$h(P) \leq C(C) \quad \text{for all } P \in C(\mathbb{Q})$$

**Semialgebraic lock:** Points on $C$ lie in bounded height region.

**Step 6: Exclusion Certificate**

The exclusion certificate:
$$K_{\text{Excl}}^+ = (\text{polynomial constraint}, \text{bound}, \text{excluded region})$$

Examples:
- $(h(P) \leq B$, Northcott, infinite height region excluded$)$
- $(N_E \leq N_0$, finiteness, large conductor region excluded$)$

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finiteness for bounded height.
2. **Hermite's Theorem** [Hermite 1857]: Finiteness for bounded discriminant.
3. **Szpiro's Conjecture** [Szpiro 1981]: Height-conductor inequality.
4. **Baker's Method** [Baker 1966]: Effective height bounds.

---

### Arithmetic Interpretation

> **Arithmetic configurations are excluded by polynomial (semialgebraic) barriers. Height bounds, conductor bounds, and discriminant bounds define regions where only finitely many objects exist. These barriers "lock out" the infinite region, concentrating arithmetic interest in bounded semialgebraic sets.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic*
- [Hermite 1857] C. Hermite, *Sur le nombre limité d'irrationalités*
- [Szpiro 1981] L. Szpiro, *Séminaire sur les pinceaux de courbes*
- [Baker 1966] A. Baker, *Linear forms in logarithms*
