# FACT-SoftWP: Soft→WP Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-wp*

Soft interface permits compile to Well-Posedness (WP) permits: local existence and uniqueness.

---

## Arithmetic Formulation

### Setup

"Well-Posedness" in arithmetic means:
- Objects are well-defined
- Constructions give unique results
- Local-to-global principles apply

### Statement (Arithmetic Version)

**Theorem (Arithmetic Well-Posedness).** Given soft arithmetic permits:
1. **Height bounded:** $h(P) \leq B$
2. **Degree bounded:** $[k(P):\mathbb{Q}] \leq d$
3. **Conductor bounded:** $N \leq C$

Then arithmetic constructions are well-posed:
- L-function converges absolutely for $\Re(s) > 3/2$
- Galois representation is well-defined
- Height pairing is non-degenerate

---

### Proof

**Step 1: L-function Convergence**

For conductor $N \leq C$, the L-function:
$$L(E, s) = \sum_{n=1}^\infty \frac{a_n}{n^s}$$

**Absolute convergence:** By Deligne's bound [Deligne 1974]:
$$|a_p| \leq 2\sqrt{p}$$

Hence for $\Re(s) = \sigma > 3/2$:
$$\sum_n \frac{|a_n|}{n^\sigma} \leq \prod_p \left(1 + \frac{2\sqrt{p}}{p^\sigma} + \frac{p}{p^{2\sigma}} + \cdots\right) < \infty$$

**Certificate:** $K_{\text{WP}}^+ = (L(E,s), \text{convergence for } \sigma > 3/2)$

**Step 2: Galois Representation Well-Definedness**

For $E/\mathbb{Q}$ with bounded conductor:

The $\ell$-adic Tate module:
$$T_\ell(E) = \varprojlim_n E[\ell^n]$$

is a free $\mathbb{Z}_\ell$-module of rank 2.

**Well-definedness:**
- Galois action is continuous [Serre 1968]
- Image is contained in $\text{GL}_2(\mathbb{Z}_\ell)$
- Independent of choices (base point, isomorphism)

**Certificate:** $K_{\rho}^+ = (\rho_{E,\ell}, T_\ell(E), \text{continuity proof})$

**Step 3: Height Pairing Non-Degeneracy**

The Néron-Tate height pairing:
$$\langle \cdot, \cdot \rangle: E(\mathbb{Q}) \times E(\mathbb{Q}) \to \mathbb{R}$$

**Non-degeneracy:** By [Néron 1965]:
$$\langle P, Q \rangle = 0 \; \forall Q \in E(\mathbb{Q}) \Rightarrow P \in E(\mathbb{Q})_{\text{tors}}$$

For bounded height, the pairing matrix on generators has:
$$\det(\langle P_i, P_j \rangle) \neq 0$$

**Certificate:** $K_{\langle\rangle}^+ = (\langle \cdot, \cdot \rangle, \text{non-degeneracy on } E(\mathbb{Q})/\text{tors})$

**Step 4: Local-Global Compatibility**

Height decomposes:
$$\hat{h}(P) = \sum_v n_v \lambda_v(P)$$

**Well-posedness:**
- Each local height $\lambda_v$ is well-defined [Silverman 1986]
- Sum converges (finitely many non-zero terms for any $P$)
- Result independent of decomposition choice

**Step 5: Compilation**

Soft permits → WP permits:

| **Soft Permit** | **WP Permit** |
|----------------|---------------|
| Height bounded | L-function converges |
| Degree bounded | Galois representation finite image |
| Conductor bounded | Local factors well-defined |

The compilation is automatic given the bounds.

---

### Key Arithmetic Ingredients

1. **Deligne's Bound** [Deligne 1974]: $|a_p| \leq 2\sqrt{p}$.
2. **Néron's Theorem** [Néron 1965]: Height pairing properties.
3. **Serre's Galois Theory** [Serre 1968]: Continuity of representations.
4. **Local Heights** [Silverman 1986]: Decomposition well-defined.

---

### Arithmetic Interpretation

> **Bounded arithmetic complexity (height, degree, conductor) ensures all standard constructions are well-defined. This is the arithmetic "well-posedness" guarantee.**

---

### Literature

- [Deligne 1974] P. Deligne, *La conjecture de Weil. I*
- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs*
- [Serre 1968] J.-P. Serre, *Abelian ℓ-adic representations*
- [Silverman 1986] J. Silverman, *The Arithmetic of Elliptic Curves*
