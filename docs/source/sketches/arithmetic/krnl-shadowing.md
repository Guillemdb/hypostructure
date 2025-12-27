# KRNL-Shadowing: Shadowing Metatheorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-shadowing*

Approximate solutions shadow true solutions.

---

## Arithmetic Formulation

### Setup

"Shadowing" in arithmetic means:
- Approximate rational points are close to true rational points
- Numerical L-function zeros approximate actual zeros
- p-adic approximations shadow global solutions

### Statement (Arithmetic Version)

**Theorem (Arithmetic Shadowing).** Approximate solutions shadow exact solutions:

1. **Rational shadowing:** Points close to $X(\mathbb{Q})$ in adelic topology are shadowed
2. **Zero shadowing:** Numerical zeros of L-functions shadow true zeros
3. **Height shadowing:** Approximate minimal points shadow actual minimal points

---

### Proof

**Step 1: Adelic Shadowing**

**Setup:** $X/\mathbb{Q}$ with $X(\mathbb{A}_\mathbb{Q}) \neq \emptyset$

**Shadowing principle:** If $(x_v) \in \prod_v X(\mathbb{Q}_v)$ is close to $X(\mathbb{Q})$:
$$d_{\mathbb{A}}((x_v), X(\mathbb{Q})) < \epsilon$$

then there exists $x \in X(\mathbb{Q})$ shadowing $(x_v)$.

**When it holds [Brauer-Manin]:**
- If Brauer-Manin obstruction vanishes
- If $X$ satisfies weak approximation

**Quantitative:** Distance to shadow bounded by reciprocal of height.

**Step 2: L-function Zero Shadowing**

**Setup:** Compute $\zeta(\sigma + it)$ numerically with error $< \epsilon$.

**Shadowing theorem [Odlyzko]:**
If numerical computation shows $|\zeta(\sigma + it)| < \epsilon$ for $\sigma$ near 1/2:
- There exists true zero $\rho$ with $|\rho - (\sigma + it)| < C\epsilon^{1/2}$

**Proof:** L-functions don't oscillate too rapidly; small values imply nearby zeros.

**Rigorous shadowing [Turing]:**
The Turing method verifies zeros lie on critical line by shadowing argument.

**Step 3: Height Shadowing**

**Setup:** Search for points on $E(\mathbb{Q})$ with height $\leq B$.

**Shadowing:** If search finds approximate point $P'$ with:
$$|P' - P| < \delta \text{ in real embedding}$$

then $P \in E(\mathbb{Q})$ shadows $P'$.

**Quantitative [Silverman]:**
$$h(P) \geq h(P') - C\delta$$

The approximate minimum shadows the true minimum.

**Step 4: p-adic Shadowing (Hensel)**

**Hensel's Lemma:** If $f(a) \equiv 0 \pmod{p^k}$ and $f'(a) \not\equiv 0 \pmod{p}$:
- There exists unique $\alpha \in \mathbb{Z}_p$ with $f(\alpha) = 0$
- $\alpha \equiv a \pmod{p^k}$

**Shadowing:** The approximate root $a$ shadows the true root $\alpha$.

**Lifting:** Each approximation refines: $a_k \to a_{k+1} \to \cdots \to \alpha$

**Step 5: Descent Shadowing**

**Setup:** Descent computes upper bound on rank.

**Shadowing:** Elements of Selmer group shadow:
- Either actual points in $E(\mathbb{Q})$
- Or elements of $\text{Ш}$

**Shadowing criterion:**
$$\xi \in \text{Sel}^{(n)}(E) \text{ shadows } P \in E(\mathbb{Q}) \iff \xi \in \text{Im}(\kappa: E(\mathbb{Q}) \to \text{Sel})$$

**Step 6: Shadowing Certificate**

The shadowing certificate:
$$K_{\text{Shad}}^+ = (\text{approximate solution}, \text{shadow tolerance}, \text{true solution})$$

**Components:**
- **Approximate:** $(x_v)$, numerical zero, or p-adic approximation
- **Tolerance:** $\epsilon$ or $p^k$ precision
- **Shadow:** Proof that true solution exists within tolerance

---

### Key Arithmetic Ingredients

1. **Hensel's Lemma** [Hensel 1908]: p-adic root shadowing.
2. **Weak Approximation** [Hasse]: Adelic shadowing for quadrics.
3. **Odlyzko's Methods** [Odlyzko 1987]: Rigorous zero location.
4. **Height Bounds** [Silverman 1990]: Height gap for shadowing.

---

### Arithmetic Interpretation

> **Arithmetic shadowing connects approximations to exact solutions. Hensel lifting shadows p-adic roots, adelic approximations shadow rational points (when Brauer-Manin allows), numerical L-function zeros shadow true zeros. This shadowing principle validates computational approaches to arithmetic.**

---

### Literature

- [Hensel 1908] K. Hensel, *Theorie der algebraischen Zahlen*
- [Hasse 1931] H. Hasse, *Über ternäre quadratische Formen*
- [Odlyzko 1987] A.M. Odlyzko, *On the distribution of spacings between zeros of the zeta function*
- [Silverman 1990] J. Silverman, *The difference between the Weil height and the canonical height*
