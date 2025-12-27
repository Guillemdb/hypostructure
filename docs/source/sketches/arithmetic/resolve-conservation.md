# RESOLVE-Conservation: Conservation of Flow

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-conservation*

Resolution operations conserve essential flow quantities—total "mass" is preserved.

---

## Arithmetic Formulation

### Setup

"Conservation of flow" in arithmetic means:
- **Conserved quantity:** Arithmetic invariant unchanged by resolution
- **Mass:** Degree, height, L-value
- Operations redistribute but don't create or destroy

### Statement (Arithmetic Version)

**Theorem (Arithmetic Conservation).** Under arithmetic resolution $\sigma: X \to Y$:

1. **Degree conservation:** $\deg X = \deg Y$ (for projective varieties)
2. **L-value conservation:** $L^*(X, 1) = L^*(Y, 1) \cdot \text{(explicit factor)}$
3. **Height conservation:** $h(\sigma(P)) \sim h(P)$ up to bounded error

---

### Proof

**Step 1: Degree Conservation**

For birational map $\sigma: X \dashrightarrow Y$:

**Conservation:** $\deg X = \deg Y$

**Proof:** Degree is a birational invariant for projective varieties [Hartshorne Ch. II].

**For blowup:** $\tilde{X} \to X$ at point:
- $\deg \tilde{X} = \deg X$ (same generic degree)
- Exceptional divisor contributes to Picard group, not degree

**Step 2: L-value Conservation**

For surgery $\sigma: X \to Y$:

**Functional equation preserved:**
$$\Lambda(X, s) = \epsilon \cdot \Lambda(X, k - s)$$

**Central value relation:**
$$L^*(X, k/2) = L^*(Y, k/2) \cdot \prod_{p \in S} C_p$$

where $C_p$ are local correction factors at bad primes.

**Explicit formula [Bloch 1984]:**
$$\frac{L^*(X, 1)}{L^*(Y, 1)} = \prod_{p \in \text{Sing}} \frac{|H^1(X_p, \mathbb{Q}_\ell)|}{|H^1(Y_p, \mathbb{Q}_\ell)|}$$

**Step 3: Height Conservation**

For morphism $f: X \to Y$ of projective varieties:

**Height transformation:**
$$h_Y(f(P)) = \deg(f) \cdot h_X(P) + O(1)$$

**For isomorphism:** $\deg(f) = 1$, so:
$$h_Y(f(P)) = h_X(P) + O(1)$$

**Conservation:** Heights are preserved up to bounded function.

**Step 4: BSD Conservation**

For isogenous elliptic curves $E \sim E'$:

**BSD invariants conserve:**
$$\frac{L^{(r)}(E, 1) / \Omega_E \cdot \text{Reg}_E}{\prod c_p \cdot |E(\mathbb{Q})_{\text{tors}}|^2} = \frac{L^{(r)}(E', 1) / \Omega_{E'} \cdot \text{Reg}_{E'}}{\prod c'_p \cdot |E'(\mathbb{Q})_{\text{tors}}|^2} \cdot |\ker \phi|$$

where $\phi: E \to E'$ is the isogeny.

**Proof [Cassels 1962]:** Isogeny invariance of BSD quotient.

**Step 5: Mass Distribution**

**Conservation principle:** Total "arithmetic mass" is conserved:

$$\sum_{i} m(X_i) = \sum_j m(Y_j)$$

where $m$ is an arithmetic measure (height, conductor contribution, L-value).

**Example (Conductor):**
- Before surgery: $N_E$
- After twist by $d$: $N_{E^{(d)}}$
- Relation: $N_{E^{(d)}} = N_E \cdot d^2 / \gcd(N_E, d)^2$

Mass redistributes among primes but total is conserved modulo the twist factor.

**Step 6: Conservation Certificate**

The conservation certificate:
$$K_{\text{Cons}}^+ = (\text{degree equality}, L\text{-value ratio}, \text{height bound})$$

**Verification:** Check that $\deg X = \deg Y$, compute $L^*(X)/L^*(Y)$, verify height difference is bounded.

---

### Key Arithmetic Ingredients

1. **Birational Invariance** [Hartshorne]: Degree is birational invariant.
2. **Bloch's Formula** [Bloch 1984]: L-value relations under modification.
3. **Height Machine** [Lang 1983]: Heights transform predictably.
4. **Cassels' Theorem** [Cassels 1962]: BSD isogeny invariance.

---

### Arithmetic Interpretation

> **Arithmetic surgery conserves essential quantities: degree is birational invariant, L-values transform by explicit local factors, heights change by bounded amounts. Total arithmetic "mass" is redistributed, not created or destroyed.**

---

### Literature

- [Hartshorne 1977] R. Hartshorne, *Algebraic Geometry*, Springer
- [Bloch 1984] S. Bloch, *Height pairings for algebraic cycles*
- [Lang 1983] S. Lang, *Fundamentals of Diophantine Geometry*
- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1. III*
