# KRNL-WeakStrong: Weak-Strong Uniqueness

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-weak-strong*

Weak solutions satisfying additional regularity are unique and equal strong solutions.

---

## Arithmetic Formulation

### Setup

"Weak-strong uniqueness" in arithmetic means:
- Weak solutions (cohomological) become unique when they lift to strong (geometric)
- Selmer elements with extra properties are unique (represent actual points)
- Local solutions with global properties are unique

### Statement (Arithmetic Version)

**Theorem (Arithmetic Weak-Strong).** Weak solutions with regularity are unique:

1. **Selmer to Mordell-Weil:** Selmer elements lifting to rational points are unique
2. **Local to global:** Local solutions with Brauer compatibility are unique
3. **Cohomology to geometry:** Cohomology classes with algebraic representatives are unique

---

### Proof

**Step 1: Selmer Weak-Strong**

**Weak solution:** $\xi \in \text{Sel}^{(n)}(E)$ (cohomological)

**Strong solution:** $P \in E(\mathbb{Q})$ with $\kappa(P) = \xi$

**Weak-strong uniqueness:**
If $\xi = \kappa(P)$ for some $P$, then $P$ is unique up to $nE(\mathbb{Q})$.

**Proof:**
- $\kappa: E(\mathbb{Q}) \to H^1(\mathbb{Q}, E[n])$ is injective mod $n$
- If $\kappa(P) = \kappa(Q)$, then $P - Q \in nE(\mathbb{Q})$
- The "strong" lift is unique mod n-torsion

**Step 2: Local-Global Weak-Strong**

**Weak solution:** $(x_v) \in X(\mathbb{A}_\mathbb{Q})$ (adelic)

**Strong solution:** $x \in X(\mathbb{Q})$ (rational)

**Weak-strong uniqueness:**
If $(x_v)$ is in the Brauer-Manin set and lifts to $x \in X(\mathbb{Q})$:
- The rational point $x$ is determined by $(x_v)$ (in discrete topology)

**Proof:** $X(\mathbb{Q})$ embeds discretely in $X(\mathbb{A}_\mathbb{Q})$.

**Step 3: Hodge Weak-Strong**

**Weak solution:** $\alpha \in H^{2p}(X, \mathbb{Q}) \cap H^{p,p}(X)$ (Hodge class)

**Strong solution:** $[Z]$ for algebraic cycle $Z$ (algebraic class)

**Weak-strong uniqueness:**
If $\alpha = [Z]$ for some $Z$, then $Z$ is unique up to:
- Rational equivalence
- Algebraic equivalence (in Chow group)

**Proof [Deligne]:** The cycle class map has controlled kernel (related to Griffiths group).

**Step 4: Galois Weak-Strong**

**Weak solution:** $\rho: G_\mathbb{Q} \to \text{GL}_n(\overline{\mathbb{Q}}_\ell)$ (representation)

**Strong solution:** $\rho = \rho_X$ for variety $X$ (geometric)

**Weak-strong uniqueness [Fontaine-Mazur]:**
If $\rho$ is geometric (comes from variety), it's determined by:
- Frobenius traces at finitely many primes
- Hence "unique" among representations with same traces

**Step 5: Height Weak-Strong**

**Weak solution:** $P'$ with $\hat{h}(P') \approx \hat{h}_{\min}$ (approximate minimizer)

**Strong solution:** $P$ with $\hat{h}(P) = \hat{h}_{\min}$ (true minimizer)

**Weak-strong uniqueness:**
- If gap $\hat{h}_{\min+1} - \hat{h}_{\min} > \epsilon$
- And $\hat{h}(P') < \hat{h}_{\min} + \epsilon/2$
- Then $P' = P$ (unique minimizer)

**Step 6: Weak-Strong Certificate**

The weak-strong certificate:
$$K_{\text{WS}}^+ = (\text{weak solution}, \text{regularity}, \text{strong solution}, \text{uniqueness proof})$$

**Components:**
- **Weak:** Cohomological or adelic object
- **Regularity:** Additional property (lifting, Brauer, algebraicity)
- **Strong:** Geometric lift
- **Uniqueness:** Why the lift is unique

---

### Key Arithmetic Ingredients

1. **Selmer-to-MW** [Cassels 1962]: Kummer sequence exactness.
2. **Discrete Embedding** [Weil]: $X(\mathbb{Q}) \hookrightarrow X(\mathbb{A}_\mathbb{Q})$.
3. **Cycle Class** [Deligne 1974]: Kernel is algebraic equivalence.
4. **Fontaine-Mazur** [FM 1995]: Geometric representations are determined.

---

### Arithmetic Interpretation

> **Arithmetic weak solutions (cohomological, adelic, Hodge classes) become unique when they have strong regularity (lift to points, cycles, varieties). Selmer elements lifting to Mordell-Weil are unique, adelic points in Brauer-Manin set lifting to rational are unique. This weak-strong principle connects cohomology to geometry.**

---

### Literature

- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Weil 1967] A. Weil, *Basic Number Theory*
- [Deligne 1974] P. Deligne, *Théorie de Hodge III*
- [Fontaine-Mazur 1995] J.-M. Fontaine, B. Mazur, *Geometric Galois representations*
