# KRNL-Consistency: The Fixed-Point Principle

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-consistency*

In the hypostructure framework, this theorem states that for a structural flow with strict dissipation, the following are equivalent: (1) the system satisfies hypostructure axioms, (2) trajectories are asymptotically self-consistent, (3) persistent states are fixed points.

---

## Arithmetic Formulation

### Setup

Let $K/\mathbb{Q}$ be a number field with ring of integers $\mathcal{O}_K$. Consider:

- **State space:** $\mathcal{X} = \text{Spec}(\mathcal{O}_K)$ with the Zariski topology
- **Height functional:** $h: \mathcal{X} \to \mathbb{R}_{\geq 0}$ the logarithmic Weil height
- **Frobenius action:** For each prime $\mathfrak{p} \subset \mathcal{O}_K$, let $\text{Frob}_\mathfrak{p}$ denote the Frobenius automorphism
- **Dissipation:** $\mathfrak{D}(x) := \sum_{\mathfrak{p}} \log N(\mathfrak{p}) \cdot \mathbf{1}_{x \in V(\mathfrak{p})}$ (contribution from ramified primes)

The "flow" is the iteration of Galois conjugation: $\sigma^n: K \to K$ for $\sigma \in \text{Gal}(K/\mathbb{Q})$.

### Statement (Arithmetic Version)

**Theorem (Arithmetic Fixed-Point Principle).** Let $K/\mathbb{Q}$ be a Galois extension with $G = \text{Gal}(K/\mathbb{Q})$. For any $\alpha \in K$ with bounded height $h(\alpha) < \infty$, the following are equivalent:

1. **Axiom Satisfaction:** The orbit $\{\sigma(\alpha) : \sigma \in G\}$ satisfies the arithmetic height bounds (Northcott property).

2. **Asymptotic Self-Consistency:** The sequence of Galois conjugates stabilizes: for any sequence $\sigma_n \in G$, the heights $h(\sigma_n(\alpha))$ converge.

3. **Fixed-Point Characterization:** The only elements with $G$-invariant height are the fixed points: $\alpha \in K^G = \mathbb{Q}$.

---

### Proof

**Step 1 (1 ⇒ 2): Height Boundedness Implies Convergence**

Assume the orbit $\{\sigma(\alpha)\}_{\sigma \in G}$ satisfies the arithmetic axioms, meaning:
$$\sup_{\sigma \in G} h(\sigma(\alpha)) \leq h(\alpha) < \infty$$

By **Northcott's Theorem** [Northcott 1950], there are only finitely many algebraic numbers of bounded height and bounded degree:
$$\#\{\beta \in \overline{\mathbb{Q}} : h(\beta) \leq B, [\mathbb{Q}(\beta):\mathbb{Q}] \leq d\} < \infty$$

Since $[\mathbb{Q}(\sigma(\alpha)):\mathbb{Q}] = [K:\mathbb{Q}]$ for all $\sigma$, and heights are bounded, the orbit is finite. Any sequence $(\sigma_n(\alpha))$ therefore has a convergent subsequence (in fact, is eventually periodic).

**Step 2 (2 ⇒ 3): Convergent Orbits Are Fixed**

If the orbit $\{\sigma(\alpha)\}_{\sigma \in G}$ is self-consistent (heights converge), then by the finiteness from Step 1, there exists $\alpha^* \in K$ such that:
$$\lim_{n \to \infty} \sigma_n(\alpha) = \alpha^* \quad \text{(stabilization)}$$

The Galois group $G$ acts continuously, so for any $\tau \in G$:
$$\tau(\alpha^*) = \tau\left(\lim_{n} \sigma_n(\alpha)\right) = \lim_{n} \tau\sigma_n(\alpha) = \alpha^*$$

The last equality holds because $\{\tau\sigma_n : n \in \mathbb{N}\}$ is another enumeration of the group action, producing the same limit. Thus $\alpha^* \in K^G$.

By **Galois theory** [Lang, Algebra, Ch. VI], $K^G = \mathbb{Q}$ for a Galois extension. Hence $\alpha^* \in \mathbb{Q}$.

**Step 3 (3 ⇒ 1): Rational Elements Satisfy All Axioms**

If $\alpha \in \mathbb{Q}$, then:
- The orbit is trivial: $\sigma(\alpha) = \alpha$ for all $\sigma \in G$
- Height is preserved: $h(\sigma(\alpha)) = h(\alpha)$
- All arithmetic axioms (bounded height, finite orbits, Northcott property) are trivially satisfied

Conversely, if $\alpha \notin \mathbb{Q}$, then $\alpha$ has a non-trivial conjugate $\sigma(\alpha) \neq \alpha$. The "trajectory" $\alpha \mapsto \sigma(\alpha) \mapsto \sigma^2(\alpha) \mapsto \cdots$ either:
- (a) Cycles back (finite orbit, satisfies axioms), or
- (b) Has unbounded height (violates axioms, analogous to "blow-up")

Case (b) is excluded by the finite-energy hypothesis $h(\alpha) < \infty$ combined with the **Weil height machine** [Hindry-Silverman, Thm. B.2.1]:
$$h(\sigma(\alpha)) = h(\alpha) \quad \text{for all } \sigma \in \text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$$

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finiteness of algebraic numbers with bounded height and degree.

2. **Galois Invariance of Height** [Weil 1929]: $h(\sigma(\alpha)) = h(\alpha)$ for all $\sigma \in \text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$.

3. **Galois Correspondence** [Artin 1944]: $K^G = \mathbb{Q}$ for Galois extensions.

---

### Arithmetic Interpretation

The Fixed-Point Principle translates to:

> **Arithmetic elements that are "stable" under Galois action are precisely the rationals.**

This is the arithmetic analogue of the dynamical principle: "structures persisting under evolution are equilibria." In number theory:
- "Evolution" = Galois conjugation
- "Equilibrium" = Rational number (fixed by all automorphisms)
- "Singularity" = Algebraic number with wild Galois orbit (height blow-up)

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*, Proc. Cambridge Phil. Soc.
- [Weil 1929] A. Weil, *L'arithmétique sur les courbes algébriques*, Acta Math.
- [Hindry-Silverman 2000] M. Hindry, J. Silverman, *Diophantine Geometry: An Introduction*, GTM 201
- [Lang 2002] S. Lang, *Algebra*, GTM 211, Ch. VI (Galois Theory)
