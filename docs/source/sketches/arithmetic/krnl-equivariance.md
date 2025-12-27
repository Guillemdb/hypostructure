# KRNL-Equivariance: Equivariance Principle

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-equivariance*

If a symmetry group $G$ acts on the system distribution and parameter space compatibly, then risk minimizers lie in $G$-orbits, gradient flow preserves equivariance, and learned structures inherit all symmetries.

---

## Arithmetic Formulation

### Setup

In arithmetic, the fundamental symmetry group is the **absolute Galois group**:
$$G_\mathbb{Q} = \text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$$

For a number field $K/\mathbb{Q}$, the relevant group is $G_K = \text{Gal}(\overline{\mathbb{Q}}/K)$.

The "parameter space" consists of:
- **Galois representations:** $\rho: G_\mathbb{Q} \to \text{GL}_n(\overline{\mathbb{Q}}_\ell)$
- **L-functions:** $L(s, \rho)$ attached to $\rho$
- **Motives:** Objects in $\mathbf{Mot}_\mathbb{Q}$

### Statement (Arithmetic Version)

**Theorem (Galois Equivariance Principle).** Let $G = G_\mathbb{Q}$ act on:
- The category of Galois representations $\mathbf{Rep}_\ell(G_\mathbb{Q})$
- The space of L-functions $\mathcal{L}$
- The parameter space $\Theta$ of arithmetic invariants

Under the compatibility assumptions:

1. **(Galois-Covariant Distribution):** If $\rho$ is a Galois representation, then $\sigma \cdot \rho := \rho \circ \text{Ad}_\sigma$ is also a representation.

2. **(Equivariant Parametrization):** $L(s, \sigma \cdot \rho) = L(s, \rho)$ for inner automorphisms.

3. **(Invariant Height):** $h(\sigma(\alpha)) = h(\alpha)$ for all $\sigma \in G_\mathbb{Q}$.

Then:
- **Arithmetic invariants are Galois-invariant:** $\text{ht}(\rho) = \text{ht}(\sigma \cdot \rho)$
- **L-function zeros are preserved:** $L(\rho, \rho) = 0 \Leftrightarrow L(\rho, \sigma \cdot \rho) = 0$
- **Motives inherit Galois symmetry:** $M \mapsto \sigma \cdot M$ preserves motivic structure

---

### Proof

**Step 1: Galois Invariance of Heights**

By the **Weil height machine** [Hindry-Silverman, Thm. B.2.1], for any $\alpha \in \overline{\mathbb{Q}}$ and $\sigma \in G_\mathbb{Q}$:
$$h(\sigma(\alpha)) = h(\alpha)$$

**Proof:** The height decomposes as:
$$h(\alpha) = \frac{1}{[K:\mathbb{Q}]} \sum_{v} [K_v : \mathbb{Q}_v] \cdot \log^+ |\alpha|_v$$

For $\sigma \in G_\mathbb{Q}$, the action permutes places $v \mapsto \sigma(v)$ with:
$$|\sigma(\alpha)|_{\sigma(v)} = |\alpha|_v$$

The sum is invariant under permutation. $\square$

**Step 2: Galois Invariance of L-functions**

For a Galois representation $\rho: G_\mathbb{Q} \to \text{GL}_n(\overline{\mathbb{Q}}_\ell)$, define:
$$L(s, \rho) = \prod_p \det(I - \rho(\text{Frob}_p) \cdot p^{-s})^{-1}$$

**Claim:** $L(s, \sigma \cdot \rho) = L(s, \rho)$ for inner automorphisms $\sigma \in G_\mathbb{Q}$.

**Proof:** For inner automorphisms (conjugation), we have:
$$(\sigma \cdot \rho)(\text{Frob}_p) = \sigma \cdot \rho(\text{Frob}_p) \cdot \sigma^{-1}$$

The determinant is invariant under conjugation:
$$\det(I - (\sigma \cdot \rho)(\text{Frob}_p) \cdot p^{-s}) = \det(I - \rho(\text{Frob}_p) \cdot p^{-s})$$

Hence $L(s, \sigma \cdot \rho) = L(s, \rho)$. $\square$

**Step 3: Equivariance of Motivic Operations**

The category $\mathbf{Mot}_\mathbb{Q}$ of motives over $\mathbb{Q}$ carries a $G_\mathbb{Q}$-action via:
$$\sigma \cdot M = M \otimes_\mathbb{Q} \overline{\mathbb{Q}} \xrightarrow{\sigma^*} M \otimes_\mathbb{Q} \overline{\mathbb{Q}}$$

By **Jannsen's theorem** [Jannsen 1992]:
$$\text{End}_{\mathbf{Mot}}(M) \otimes \mathbb{Q}_\ell \cong \text{End}_{G_\mathbb{Q}}(H_\ell(M))$$

The endomorphism algebra is $G_\mathbb{Q}$-invariant.

**Consequence:** Motivic structures (weight filtration, Hodge structure) are preserved by $G_\mathbb{Q}$-action:
$$W_\bullet(\sigma \cdot M) = \sigma \cdot W_\bullet(M)$$

**Step 4: Preservation of Arithmetic "Defects"**

In the hypostructure framework, "defects" measure deviation from permits. Arithmetically:

**Defect functional:** For a Galois representation $\rho$, define:
$$\mathcal{D}(\rho) = \sum_p |\text{Tr}(\rho(\text{Frob}_p)) - \text{expected}|^2 \cdot p^{-1}$$

**Equivariance:** By Step 2 (trace invariance under conjugation):
$$\mathcal{D}(\sigma \cdot \rho) = \mathcal{D}(\rho)$$

The defect is $G_\mathbb{Q}$-invariant.

**Step 5: Gradient Flow Preservation**

Consider "gradient flow" on the space of representations, minimizing the defect:
$$\frac{d\rho}{dt} = -\nabla_\rho \mathcal{D}(\rho)$$

**Claim:** If $\rho_0$ is $G_\mathbb{Q}$-equivariant, so is $\rho_t$ for all $t > 0$.

**Proof:** By $G_\mathbb{Q}$-invariance of $\mathcal{D}$:
$$\nabla_\rho \mathcal{D}(\sigma \cdot \rho) = \sigma \cdot \nabla_\rho \mathcal{D}(\rho)$$

The flow commutes with $G_\mathbb{Q}$-action:
$$\sigma \cdot \rho_t = (\sigma \cdot \rho)_t$$

Equivariance is preserved. $\square$

---

### Key Arithmetic Ingredients

1. **Galois Invariance of Heights** [Weil 1929]: $h(\sigma(\alpha)) = h(\alpha)$.

2. **Conjugacy Invariance of Trace** [Linear Algebra]: $\text{Tr}(ABA^{-1}) = \text{Tr}(B)$.

3. **Jannsen's Theorem** [Jannsen 1992]: Motivic endomorphisms = Galois-equivariant maps.

4. **Noether's Theorem** (Arithmetic Analogue): Galois symmetry → conserved arithmetic invariants.

---

### Arithmetic Interpretation

> **Arithmetic structures that are Galois-invariant remain so under all natural operations; the Galois group is the "symmetry group" of number theory.**

This principle ensures:
- **Well-defined L-functions:** Independent of base field extension
- **Canonical heights:** Galois-invariant height pairings
- **Motivic descent:** Structures over $\overline{\mathbb{Q}}$ descend to $\mathbb{Q}$ when $G_\mathbb{Q}$-invariant

---

### Literature

- [Weil 1929] A. Weil, *L'arithmétique sur les courbes algébriques*
- [Hindry-Silverman 2000] M. Hindry, J. Silverman, *Diophantine Geometry*, Thm. B.2.1
- [Jannsen 1992] U. Jannsen, *Motives, numerical equivalence, and semi-simplicity*, Invent. Math.
- [Noether 1918] E. Noether, *Invariante Variationsprobleme* (conceptual analogue)
