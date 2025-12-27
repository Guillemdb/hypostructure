# ACT-Lift: Regularity Lift Principle

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-lift*

Regularity lifts from lower levels to higher levels.

---

## Arithmetic Formulation

### Setup

"Regularity lift" in arithmetic means:
- Properties lift from residue fields to local fields to global
- Good reduction lifts to semistable to potential good
- Regularity propagates upward in extension towers

### Statement (Arithmetic Version)

**Theorem (Arithmetic Regularity Lift).** Regularity lifts through levels:

1. **Hensel lift:** Mod $p$ solutions lift to $p$-adic
2. **Local-global lift:** Local regularity contributes to global
3. **Extension lift:** Regularity over $K$ lifts to extensions $L/K$

---

### Proof

**Step 1: Hensel Lifting**

**Setup:** $f(x) \equiv 0 \pmod{p}$ with simple root $a$.

**Hensel's Lemma:** If $f(a) \equiv 0 \pmod{p}$ and $f'(a) \not\equiv 0 \pmod{p}$:
$$\exists \alpha \in \mathbb{Z}_p: f(\alpha) = 0, \alpha \equiv a \pmod{p}$$

**Regularity lift:** The mod-$p$ regularity (simple root) lifts to $p$-adic regularity (actual root).

**Iteration:** $a_0 \to a_1 \to \cdots \to \alpha$ with $a_n \equiv \alpha \pmod{p^n}$.

**Step 2: Local-to-Global Lift**

**Setup:** $X/\mathbb{Q}$ with $X(\mathbb{Q}_v) \neq \emptyset$ for all $v$.

**Lift (when possible):** Under Hasse principle:
$$X(\mathbb{Q}_v) \neq \emptyset \text{ for all } v \Rightarrow X(\mathbb{Q}) \neq \emptyset$$

**Regularity interpretation:**
- Local smoothness (at each $v$)
- Lifts to global existence

**When lift fails:** Brauer-Manin obstruction blocks the lift.

**Step 3: Extension Lift**

**Setup:** $E/K$ elliptic curve, $L/K$ extension.

**Rank lift:**
$$E(K) \hookrightarrow E(L)$$

**Regularity lift:**
- If $E$ has good reduction at $\mathfrak{p}$, and $\mathfrak{P} | \mathfrak{p}$ unramified
- Then $E$ has good reduction at $\mathfrak{P}$

**Mordell-Weil lift:** Rank can only increase:
$$\text{rank } E(K) \leq \text{rank } E(L)$$

**Step 4: Cohomology Lift**

**Inflation:**
$$H^n(G/H, M^H) \xrightarrow{\text{inf}} H^n(G, M)$$

**Regularity lift:** Cohomology over quotient lifts to full group.

**Hochschild-Serre:**
$$0 \to H^1(G/H, M^H) \to H^1(G, M) \to H^1(H, M)^{G/H} \to \cdots$$

**Step 5: Deformation Lift**

**Setup:** Galois representation $\bar{\rho}: G_K \to \text{GL}_n(\mathbb{F}_\ell)$.

**Lift to characteristic 0:**
$$\bar{\rho} \leadsto \rho: G_K \to \text{GL}_n(\mathcal{O})$$

where $\mathcal{O}$ is a DVR with residue field $\mathbb{F}_\ell$.

**Regularity lift:** Mod-$\ell$ representation lifts to $\ell$-adic.

**Obstruction:** $H^2$ measures obstruction to lifting.

**Step 6: Regularity Lift Certificate**

The regularity lift certificate:
$$K_{\text{Lift}}^+ = (\text{base level}, \text{target level}, \text{lift mechanism})$$

**Components:**
- **Base:** Mod $p$, local, or lower field
- **Target:** $p$-adic, global, or extension
- **Mechanism:** Hensel, Hasse, inflation, deformation

**Examples:**
| Base | Target | Mechanism |
|------|--------|-----------|
| $\mathbb{F}_p$ root | $\mathbb{Z}_p$ root | Hensel |
| $\mathbb{Q}_v$ point | $\mathbb{Q}$ point | Hasse |
| $E(K)$ | $E(L)$ | Injection |
| $\bar{\rho}$ | $\rho$ | Deformation |

---

### Key Arithmetic Ingredients

1. **Hensel's Lemma** [Hensel 1908]: Local root lifting.
2. **Hasse Principle** [Hasse 1923]: Local-global lifting.
3. **Inflation-Restriction** [Hochschild-Serre]: Cohomology lifting.
4. **Deformation Theory** [Mazur 1989]: Galois representation lifting.

---

### Arithmetic Interpretation

> **Arithmetic regularity lifts through levels. Hensel lifts mod-$p$ solutions to $p$-adic. Hasse principle lifts local to global. Mordell-Weil lifts through extensions. Deformation theory lifts mod-$\ell$ representations. Each lift carries regularity from lower to higher structure.**

---

### Literature

- [Hensel 1908] K. Hensel, *Theorie der algebraischen Zahlen*
- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen*
- [Hochschild-Serre 1953] G. Hochschild, J.-P. Serre, *Cohomology of group extensions*
- [Mazur 1989] B. Mazur, *Deforming Galois representations*
