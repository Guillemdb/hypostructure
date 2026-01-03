# ACT-Projective: Projective Extension

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-projective*

Projective extensions complete partial structures.

---

## Arithmetic Formulation

### Setup

"Projective extension" in arithmetic means:
- Completing affine to projective
- Extending local to global
- Compactifying moduli spaces

### Statement (Arithmetic Version)

**Theorem (Arithmetic Projective Extension).** Projective extensions complete arithmetic:

1. **Affine-to-projective:** $X \hookrightarrow \bar{X}$ with controlled boundary
2. **Local-to-global:** Adelic completion via all places
3. **Moduli compactification:** Adding degenerate objects at boundary

---

### Proof

**Step 1: Affine to Projective Completion**

**Setup:** Affine variety $X \subset \mathbb{A}^n$.

**Projective closure:** $\bar{X} \subset \mathbb{P}^n$ obtained by homogenizing equations.

**Boundary:** $\bar{X} \setminus X = $ points at infinity.

**Arithmetic control:**
- Rational points: $X(\mathbb{Q}) \subset \bar{X}(\mathbb{Q})$
- Height: Weil height on $\bar{X}$ extends to $X$
- L-function: Same (birational invariant for smooth $X$)

**Step 2: Adelic Projective Extension**

**Setup:** Number field $K$ with places $v$.

**Adelic completion:**
$$\mathbb{A}_K = \prod_v' K_v$$

**Projective extension:**
$$K \hookrightarrow \mathbb{A}_K$$

diagonally embeds $K$ into its completion.

**Global-to-local:** Every global object extends to local at all places.

**Step 3: Moduli Compactification**

**Setup:** Moduli space $\mathcal{M}_g$ of smooth curves.

**Deligne-Mumford compactification:** $\overline{\mathcal{M}}_g$

**Boundary:** Stable curves with nodes (controlled degenerations).

**Arithmetic extension:**
- Every family of smooth curves extends to stable family
- L-function extends (with extra factors at boundary)
- Height extends continuously

**Step 4: Néron Model Extension**

**Setup:** Abelian variety $A/K$.

**Néron model:** $\mathcal{A}/\mathcal{O}_K$ extending $A$.

**Projective extension:**
$$A(K) = \mathcal{A}(\mathcal{O}_K)$$

**Boundary:** Special fiber $\mathcal{A}_\mathfrak{p}$ at bad primes.

**Control:** Component group $\Phi_\mathfrak{p} = \mathcal{A}_\mathfrak{p}/\mathcal{A}_\mathfrak{p}^0$ measures boundary.

**Step 5: L-function Extension**

**Setup:** Euler product $L(s) = \prod_p L_p(s)$ converging for $\Re(s) > 1$.

**Projective extension:** Analytic continuation to all $s \in \mathbb{C}$.

**Boundary:** Poles and zeros (controlled singularities).

**Functional equation:** Relates $s$ to $1-s$ (symmetry of extended object).

**Step 6: Projective Extension Certificate**

The projective extension certificate:
$$K_{\text{Proj}}^+ = (\text{base}, \text{extension}, \text{boundary description})$$

**Components:**
- **Base:** Affine variety, local field, open moduli
- **Extension:** Projective closure, adeles, compactification
- **Boundary:** Points at infinity, archimedean places, degenerations

---

### Key Arithmetic Ingredients

1. **Projective Closure** [Hartshorne]: Completing affine varieties.
2. **Adeles** [Tate 1950]: Global-local completion.
3. **Deligne-Mumford** [DM 1969]: Moduli compactification.
4. **Néron Models** [Néron 1964]: Extending abelian varieties.

---

### Arithmetic Interpretation

> **Projective extension completes arithmetic structures. Affine varieties complete to projective, local fields combine to adeles, moduli spaces compactify by adding degenerations. The extension controls arithmetic at infinity and enables global analysis.**

---

### Literature

- [Hartshorne 1977] R. Hartshorne, *Algebraic Geometry*
- [Tate 1950] J. Tate, *Fourier analysis in number fields and Hecke's zeta-functions*
- [Deligne-Mumford 1969] P. Deligne, D. Mumford, *The irreducibility of the space of curves of given genus*
- [Néron 1964] A. Néron, *Modèles minimaux des variétés abéliennes*
