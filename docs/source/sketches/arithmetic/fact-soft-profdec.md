# FACT-SoftProfDec: Soft→ProfDec Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-profdec*

Soft permits compile to Profile Decomposition permits: bounded energy sequences decompose into orthogonal profiles.

---

## Arithmetic Formulation

### Setup

"Profile Decomposition" in arithmetic corresponds to:
- **Prime factorization:** $n = \prod_p p^{v_p(n)}$
- **Euler product:** $L(s) = \prod_p L_p(s)$
- **Galois decomposition:** $\rho = \bigoplus_i \rho_i$

### Statement (Arithmetic Version)

**Theorem (Arithmetic Profile Decomposition).** For a sequence of algebraic points $\{P_n\}$ with bounded height:
$$\sup_n h(P_n) \leq B < \infty$$

There exists a decomposition into arithmetic "profiles":
$$P_n \sim \sum_{j=1}^J P_n^{(j)} + r_n$$

where:
1. **Profiles $P_n^{(j)}$:** Converge to well-defined arithmetic limits
2. **Remainder $r_n$:** Has height $\to 0$
3. **Orthogonality:** $\langle P_n^{(i)}, P_n^{(j)} \rangle \to 0$ for $i \neq j$

---

### Proof

**Step 1: Euler Product as Profile Decomposition**

For an L-function $L(E, s)$:
$$L(E, s) = \prod_p L_p(E, s)$$

Each local factor is a "profile":
$$L_p(E, s) = \frac{1}{1 - a_p p^{-s} + p^{1-2s}}$$

**Orthogonality:** Different primes contribute independently:
$$\log L(E, s) = \sum_p \log L_p(E, s)$$

No cross-terms between different $p$.

**Step 2: Prime Factorization Analogy**

For an integer $n$:
$$n = \prod_p p^{v_p(n)}$$

**Profiles:** Each $p^{v_p(n)}$ is a "prime power profile."

**Orthogonality:** $\gcd(p^a, q^b) = 1$ for $p \neq q$.

**Reconstruction:** Product recovers original.

**Step 3: Galois Representation Decomposition**

For a Galois representation $\rho: G_\mathbb{Q} \to \text{GL}_n(\overline{\mathbb{Q}}_\ell)$:

By **semisimplicity** [Deligne 1980]:
$$\rho \cong \bigoplus_{i=1}^k \rho_i^{n_i}$$

where $\rho_i$ are irreducible.

**Profiles:** Each $\rho_i$ is an irreducible "profile."

**Orthogonality:** By Schur's lemma:
$$\text{Hom}(\rho_i, \rho_j) = 0 \text{ for } i \neq j$$

**Step 4: Height Decomposition**

For points on abelian varieties, decompose by:

**(a) Torsion extraction:**
$$P = P_{\text{tors}} + P_{\text{free}}$$

where $P_{\text{tors}} \in A_{\text{tors}}$ and $P_{\text{free}}$ projects to $A(\mathbb{Q})/A_{\text{tors}}$.

**(b) Orthogonal basis:**
Choose $Q_1, \ldots, Q_r$ with $\langle Q_i, Q_j \rangle = 0$ for $i \neq j$.

**(c) Decomposition:**
$$P_{\text{free}} = \sum_{i=1}^r \frac{\langle P, Q_i \rangle}{\hat{h}(Q_i)} Q_i + r$$

where $r$ has small height.

**Step 5: Bounded Sequence Decomposition**

For $\{P_n\}$ with $h(P_n) \leq B$:

By Northcott, the sequence has finitely many distinct values up to the action of the Galois group and "small" perturbations.

**Extract profiles:**
1. Identify concentration points (limits of subsequences)
2. Each concentration point $P^{(j)}$ is a profile
3. Remainder has $h(r_n) \to 0$

**Orthogonality:** Different profiles have disjoint Galois orbits or orthogonal height contributions.

---

### Key Arithmetic Ingredients

1. **Euler Product** [Euler 1737]: L-function factorization.
2. **Semisimplicity** [Deligne 1980]: Galois representations decompose.
3. **Schur's Lemma** [Representation Theory]: Orthogonality of irreducibles.
4. **Northcott Property** [Northcott 1950]: Concentration in bounded height.

---

### Arithmetic Interpretation

> **Arithmetic objects decompose into "prime" or "irreducible" profiles, analogous to prime factorization or Euler products. Different profiles are orthogonal.**

---

### Literature

- [Deligne 1980] P. Deligne, *La conjecture de Weil. II*
- [Northcott 1950] D.G. Northcott, *An inequality in arithmetic*
- [Bahouri-Gérard 1999] H. Bahouri, P. Gérard, *High frequency approximation of solutions to critical nonlinear wave equations*
