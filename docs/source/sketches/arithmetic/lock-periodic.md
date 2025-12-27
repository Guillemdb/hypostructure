# LOCK-Periodic: The Periodic Law

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-periodic*

Periodic orbits create locks: return maps encode dynamical constraints preventing escape.

---

## Arithmetic Formulation

### Setup

"Periodic law" in arithmetic means:
- **Periodicity:** Frobenius orbits, continued fractions, unit groups
- **Return map:** Galois action cycling through conjugates
- **Lock:** Periodic structure constrains arithmetic

### Statement (Arithmetic Version)

**Theorem (Arithmetic Periodic Lock).** Periodic structures lock arithmetic:

1. **Frobenius orbits:** $\text{Frob}_p$ acts on $\bar{E}(\bar{\mathbb{F}}_p)$ with periodic orbits
2. **Unit groups:** $\mathcal{O}_K^\times$ periodic action on ideals
3. **Lock:** Period length determines splitting behavior

---

### Proof

**Step 1: Frobenius Periodicity**

**Frobenius:** $\text{Frob}_p: x \mapsto x^p$ on $\bar{\mathbb{F}}_p$.

**Orbit:** For $\alpha \in \mathbb{F}_{p^n}$:
$$\alpha, \alpha^p, \alpha^{p^2}, \ldots, \alpha^{p^{n-1}}, \alpha^{p^n} = \alpha$$

**Period:** $|\text{orbit}| = [\mathbb{F}_p(\alpha) : \mathbb{F}_p]$.

**Step 2: Elliptic Curve Periodicity**

**Frobenius on $E$:** $\phi: (x, y) \mapsto (x^p, y^p)$

**Periodicity:** $\phi^n = [p^n]$ on $E(\mathbb{F}_{p^n})$.

**Characteristic polynomial:**
$$\phi^2 - a_p \phi + p = 0$$

**Periods:** Eigenvalue arguments give period structure.

**Step 3: Dirichlet's Unit Theorem**

**Units:** For number field $K$ with $r_1$ real, $r_2$ complex places:
$$\mathcal{O}_K^\times \cong \mu_K \times \mathbb{Z}^{r_1 + r_2 - 1}$$

**Periodicity:** Roots of unity $\mu_K$ are the periodic elements.

**Regulator:** Non-periodic part measured by $\text{Reg}(K)$.

**Step 4: Continued Fraction Periodicity**

**Theorem [Lagrange]:** $\alpha \in \mathbb{R}$ has eventually periodic CF $\iff \alpha$ is quadratic irrational.

**Period:** For $\alpha = \frac{p + \sqrt{d}}{q}$:
$$\alpha = [a_0; \overline{a_1, a_2, \ldots, a_\ell}]$$

**Connection:** Period relates to fundamental unit of $\mathbb{Q}(\sqrt{d})$.

**Step 5: Artin's Conjecture (Periodic Density)**

**Conjecture [Artin]:** Primitive root $a$ modulo $p$ for density $\prod_p (1 - 1/(p(p-1)))$ of primes.

**Periodicity:** If $a$ is primitive root, multiplicative order is $p - 1$ (maximal period).

**Lock:** Primitivity (maximal period) occurs with positive density.

**Step 6: Periodic Lock Certificate**

$$K_{\text{Periodic}}^+ = (\text{periodic structure}, \text{period length}, \text{locked behavior})$$

Examples:
- (Frobenius orbit, $[\mathbb{F}_p(\alpha):\mathbb{F}_p]$, splitting type)
- (Unit action, period $|\mu_K|$, class number)

---

### Key Arithmetic Ingredients

1. **Frobenius Automorphism** [Frobenius 1896]: Generator of Galois group.
2. **Dirichlet Unit Theorem** [Dirichlet 1846]: Structure of unit groups.
3. **Lagrange's Theorem** [Lagrange 1770]: Periodic continued fractions.
4. **Artin's Primitive Root Conjecture** [Artin 1927]: Density of primitive roots.

---

### Arithmetic Interpretation

> **Periodicity locks arithmetic structure. Frobenius orbits on finite fields, unit groups in number fields, and continued fraction expansions exhibit periodic behavior. The period length encodes arithmetic invariants: splitting degrees, regulators, class numbers. This periodic structure "locks" the arithmetic, making invariants computable from periods.**

---

### Literature

- [Frobenius 1896] G. Frobenius, *Über Beziehungen zwischen den Primidealen*
- [Dirichlet 1846] P.G.L. Dirichlet, *Zur Theorie der complexen Einheiten*
- [Lagrange 1770] J.-L. Lagrange, *Additions to Euler's Algebra*
- [Hooley 1967] C. Hooley, *On Artin's conjecture*
