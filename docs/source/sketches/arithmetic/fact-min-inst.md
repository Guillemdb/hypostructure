# FACT-MinInst: Minimal Instantiation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-min-inst*

The minimal (thin) instantiation requires only uncontroversial physical/mathematical definitions, with all other properties derived by the Sieve.

---

## Arithmetic Formulation

### Setup

Minimal arithmetic input:
- Field/variety specification
- Standard height function
- Natural Galois action

All advanced properties (BSD, Hodge, etc.) are **derived**, not assumed.

### Statement (Arithmetic Version)

**Theorem (Minimal Arithmetic Burden).** To apply the arithmetic framework, one needs only:

1. **Minimal data:**
   - Object $X$ (field, curve, variety)
   - Standard height $h$ (Weil/Néron-Tate)
   - Galois group $G$ (absolute Galois group)

2. **Derived properties:**
   - Finiteness of rational points (from Northcott + height)
   - L-function properties (from Galois representation)
   - Special value formulas (from functional equation)

3. **Not assumed:**
   - BSD conjecture
   - Riemann Hypothesis
   - Hodge conjecture

---

### Proof

**Step 1: What Is Assumed (Minimal Input)**

**(a) Object specification:**
- Elliptic curve: $E: y^2 = x^3 + ax + b$ with $a, b \in \mathbb{Q}$
- Number field: $K = \mathbb{Q}[x]/(f(x))$ with $f \in \mathbb{Z}[x]$ irreducible
- Variety: Scheme $X$ of finite type over $\mathbb{Q}$

These are **definitions**, not conjectures.

**(b) Height specification:**
- Weil height: $h(\alpha) = \frac{1}{[K:\mathbb{Q}]} \sum_v \log^+ |\alpha|_v$
- Néron-Tate: $\hat{h}(P) = \lim_{n \to \infty} h([n]P)/n^2$

These are **constructions**, not assumptions.

**(c) Galois action:**
- $G_\mathbb{Q} = \text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$ acts on $X(\overline{\mathbb{Q}})$
- Well-defined by algebraic closure construction

**Step 2: What Is Derived**

**(a) Finiteness (Northcott):**
$$\#\{P \in X(\overline{\mathbb{Q}}) : h(P) \leq B, [k(P):\mathbb{Q}] \leq d\} < \infty$$

This is a **theorem** [Northcott 1950], derived from height properties.

**(b) Mordell-Weil:**
$$E(\mathbb{Q}) \cong \mathbb{Z}^r \oplus E(\mathbb{Q})_{\text{tors}}$$

This is a **theorem** [Mordell 1922], derived from descent.

**(c) L-function existence:**
$$L(E, s) = \prod_p L_p(E, s)$$

The Euler product is a **construction** from the Galois representation.

**(d) Functional equation:**
$$\Lambda(E, s) = w_E \cdot \Lambda(E, 2-s)$$

This is a **theorem** [Wiles-Taylor 1995], derived from modularity.

**Step 3: What Is NOT Assumed**

The framework does **not** assume the conjecture it aims to verify:

**(a) BSD conjecture:**
- Statement: $\text{ord}_{s=1} L(E, s) = \text{rank } E(\mathbb{Q})$
- Status: **Target**, not hypothesis

**(b) Riemann Hypothesis:**
- Statement: Zeros of $\zeta(s)$ lie on $\Re(s) = 1/2$
- Status: **Target**, not hypothesis

**(c) Hodge conjecture:**
- Statement: Hodge classes are algebraic
- Status: **Target**, not hypothesis

**Step 4: Separation of Input and Output**

| **Input (assumed)** | **Output (derived/targeted)** |
|--------------------|------------------------------|
| Object exists | Conjecture holds |
| Height defined | Special values computed |
| Galois acts | Galois obstruction verified |
| L-function converges | Zeros located |

The derivation chain is:
$$\text{Minimal Input} \xrightarrow{\text{Sieve}} \text{Certificates} \xrightarrow{\text{Lock}} \text{Conjecture}$$

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Derived finiteness.
2. **Mordell-Weil Theorem** [Mordell 1922]: Structure of rational points.
3. **Modularity** [Wiles 1995]: Functional equation source.
4. **Separation Principle**: Input ≠ Output.

---

### Arithmetic Interpretation

> **The arithmetic framework requires only standard definitions (object, height, Galois). All deep properties are derived, and the conjecture is the target—never the hypothesis.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in arithmetic*
- [Mordell 1922] L.J. Mordell, *On the rational solutions...*
- [Wiles 1995] A. Wiles, *Modular elliptic curves and Fermat's Last Theorem*
