# LOCK-Antichain: Antichain-Surface Correspondence

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-antichain*

Antichains correspond to codimension-one surfaces.

---

## Arithmetic Formulation

### Setup

"Antichain-surface correspondence" in arithmetic means:
- Independent elements in posets correspond to hypersurfaces
- Maximal independent sets of primes correspond to divisors
- Antichain structure reflects geometric codimension

### Statement (Arithmetic Version)

**Theorem (Arithmetic Antichain-Surface).** Antichains correspond to arithmetic hypersurfaces:

1. **Prime antichain:** Independent primes correspond to divisor
2. **Selmer antichain:** Independent Selmer elements correspond to curve
3. **Height antichain:** Independent points correspond to regulator

---

### Proof

**Step 1: Prime Antichain to Divisor**

**Setup:** $S = \{p_1, \ldots, p_k\}$ set of primes (antichain in divisibility order).

**Divisor construction:**
$$D_S = \sum_{i=1}^k [p_i] \in \text{Div}(\text{Spec}(\mathbb{Z}))$$

**Correspondence:**
- Antichain (no $p_i | p_j$ for $i \neq j$) ↔ reduced divisor
- Codimension 1 in Spec(Z)

**Support:** $\text{supp}(D_S) = S$

**Step 2: Selmer Antichain to Descent**

**Setup:** $\{\xi_1, \ldots, \xi_r\} \subset \text{Sel}^{(n)}(E)$ independent elements.

**Antichain:** No $\xi_i$ in span of others (independence in $\mathbb{F}_n$-vector space).

**Surface correspondence:**
- Each $\xi_i$ corresponds to $n$-covering $C_i \to E$
- Independent elements ↔ fiber product $C_1 \times_E \cdots \times_E C_r$
- Dimension $= r$ = size of antichain

**Step 3: Height Antichain to Regulator**

**Setup:** $\{P_1, \ldots, P_r\} \subset E(\mathbb{Q})$ independent points (mod torsion).

**Antichain:** No $P_i$ in $\mathbb{Z}$-span of others.

**Surface (regulator):**
$$\text{Reg}_E = \det(\langle P_i, P_j \rangle)$$

**Correspondence:**
- $r$ independent points ↔ $r$-dimensional lattice
- Regulator = covolume = "area" of fundamental domain
- Antichain size = rank = "dimension" of Mordell-Weil

**Step 4: Galois Antichain to Cohomology**

**Setup:** Independent Galois cohomology classes $\{c_1, \ldots, c_k\} \subset H^1(G, M)$.

**Antichain:** Linearly independent in cohomology group.

**Surface:** Correspond to $k$-dimensional subspace of $H^1$.

**Arithmetic interpretation:** Independent local obstructions ↔ hypersurface in obstruction space.

**Step 5: Conductor Antichain**

**Setup:** Prime factorization $N = p_1^{e_1} \cdots p_k^{e_k}$.

**Antichain:** $\{p_1, \ldots, p_k\}$ (distinct prime factors).

**Surface:** Bad reduction locus = union of $k$ codimension-1 strata in Spec(Z).

**Correspondence:**
- $k = \omega(N)$ = number of antichain elements
- Each $p_i$ contributes codimension-1 "surface"

**Step 6: Antichain-Surface Certificate**

The antichain-surface certificate:
$$K_{\text{AC}}^+ = (\text{antichain}, \text{surface}, \text{correspondence proof})$$

**Components:**
- **Antichain:** Independent elements in poset
- **Surface:** Codimension-1 object
- **Correspondence:** How antichain determines surface

**Examples:**
| Antichain | Surface |
|-----------|---------|
| $\{p_1, \ldots, p_k\}$ primes | Divisor on Spec(Z) |
| $\{\xi_1, \ldots, \xi_r\}$ Selmer | n-descent variety |
| $\{P_1, \ldots, P_r\}$ MW | Regulator lattice |

---

### Key Arithmetic Ingredients

1. **Divisor Theory** [Hartshorne]: Codimension-1 cycles.
2. **Selmer Groups** [Cassels 1962]: Cohomological descent.
3. **Regulator** [Néron 1965]: Height pairing determinant.
4. **Galois Cohomology** [Serre 1964]: Obstruction theory.

---

### Arithmetic Interpretation

> **Antichains in arithmetic posets correspond to codimension-1 surfaces. Independent primes give divisors, independent Selmer elements give descent varieties, independent Mordell-Weil generators give the regulator lattice. This antichain-surface correspondence connects combinatorics to geometry.**

---

### Literature

- [Hartshorne 1977] R. Hartshorne, *Algebraic Geometry*
- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs*
- [Serre 1964] J.-P. Serre, *Cohomologie Galoisienne*
