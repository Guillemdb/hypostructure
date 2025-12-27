# FACT-ValidInst: Valid Instantiation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-valid-inst*

Given thin kernel objects satisfying minimal conditions, a valid hypostructure instantiation exists and the Sieve can be applied.

---

## Arithmetic Formulation

### Setup

An arithmetic "instantiation" specifies:
- The number field or variety
- The L-function or height function
- The relevant Galois group or symmetry

### Statement (Arithmetic Version)

**Theorem (Arithmetic Valid Instantiation).** Given arithmetic thin data:
1. **Arena:** Number field $K/\mathbb{Q}$ or variety $X/\mathbb{Q}$
2. **Height:** Weil height $h: X(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$
3. **Dissipation:** Conductor or discriminant
4. **Symmetry:** Galois group $G_K = \text{Gal}(\overline{K}/K)$

A valid arithmetic framework instantiation exists if:
- $K/\mathbb{Q}$ is a finite extension (or $X$ is of finite type)
- $h$ satisfies the Northcott property
- The Galois action is well-defined

---

### Proof

**Step 1: Arena Verification**

For number fields $K/\mathbb{Q}$:
- $[K:\mathbb{Q}] = n < \infty$ (finite degree)
- $\mathcal{O}_K$ is a Dedekind domain [Neukirch, Ch. I]
- $\text{Spec}(\mathcal{O}_K)$ is a well-defined scheme

For varieties $X/\mathbb{Q}$:
- $X$ is of finite type over $\mathbb{Q}$ [Hartshorne, Ch. II]
- $X(\overline{\mathbb{Q}})$ is the set of algebraic points

**Step 2: Height Verification**

The Weil height $h: X(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$ satisfies:

**(a) Northcott property:** By [Northcott 1950]:
$$\#\{P \in X(\overline{\mathbb{Q}}) : h(P) \leq B, [k(P):\mathbb{Q}] \leq d\} < \infty$$

**(b) Galois invariance:** By [Weil 1929]:
$$h(\sigma(P)) = h(P) \quad \forall \sigma \in G_\mathbb{Q}$$

**(c) Functoriality:** For morphisms $f: X \to Y$:
$$h_Y(f(P)) \leq C \cdot h_X(P) + O(1)$$

**Step 3: Dissipation (Conductor)**

The conductor $\mathfrak{f}$ measures arithmetic complexity:

For number fields: $\mathfrak{f}_K = \prod_\mathfrak{p} \mathfrak{p}^{f_\mathfrak{p}}$

For varieties: $N_X = \prod_p p^{f_p(X)}$ [Ogg-Saito formula]

Properties:
- $\mathfrak{f}$ is an ideal in $\mathcal{O}_K$ (or integer for varieties)
- Bounded below: $N(\mathfrak{f}) \geq 1$
- Finite for any fixed object

**Step 4: Galois Action**

The Galois group acts on:
- Points: $\sigma: X(\overline{\mathbb{Q}}) \to X(\overline{\mathbb{Q}})$
- Functions: $\sigma: \overline{\mathbb{Q}}(X) \to \overline{\mathbb{Q}}(X)$
- Cohomology: $\sigma: H^*(X, \mathbb{Q}_\ell) \to H^*(X, \mathbb{Q}_\ell)$

By [Grothendieck SGA 1], this action is well-defined and continuous.

**Step 5: Instantiation Certificate**

The valid instantiation produces certificate:
$$K_{\text{inst}}^+ = (K, \mathcal{O}_K, h, \mathfrak{f}, G_K, \text{verifications})$$

This enables the arithmetic Sieve to proceed.

---

### Key Arithmetic Ingredients

1. **Dedekind Domains** [Neukirch 1999]: Structure of $\mathcal{O}_K$.
2. **Weil Height** [Weil 1929]: Functorial height theory.
3. **Conductor Formula** [Ogg 1967]: Arithmetic complexity measure.
4. **Galois Theory** [Grothendieck SGA 1]: Group action on schemes.

---

### Literature

- [Neukirch 1999] J. Neukirch, *Algebraic Number Theory*, Springer
- [Northcott 1950] D.G. Northcott, *An inequality in arithmetic*
- [Grothendieck SGA 1] A. Grothendieck, *Revêtements étales et groupe fondamental*
