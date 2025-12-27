# DAG Structure Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-dag*

The sieve diagram is a directed acyclic graph (DAG). All edges point forward in topological ordering. Consequently: no backward edges exist, each epoch visits at most $|V|$ nodes, and the sieve terminates.

---

## Arithmetic Formulation

### Setup

The arithmetic analogue is the **partial order on arithmetic invariants**:
- **Nodes:** Arithmetic tests (height bounds, degree checks, Galois conditions)
- **Edges:** Logical implications between conditions
- **DAG property:** No circular dependencies in arithmetic verification

### Statement (Arithmetic Version)

**Theorem (Well-Ordering of Arithmetic Verification).** The verification protocol for an arithmetic conjecture forms a directed acyclic graph where:

1. **Height hierarchy:** Tests are ordered by height complexity: $h_1 < h_2 \Rightarrow$ Test($h_1$) precedes Test($h_2$)
2. **Degree hierarchy:** Within fixed height, tests ordered by field degree
3. **Galois hierarchy:** Within fixed degree, tests ordered by Galois group structure

Consequently:
- No circular dependencies in verification
- Each verification chain has length at most $|V|$ (finite node count)
- Verification terminates

---

### Proof

**Step 1: Height as Primary Well-Ordering**

By **Northcott's theorem** [Northcott 1950], for fixed $B$ and $d$:
$$\#\{\alpha \in \overline{\mathbb{Q}} : h(\alpha) \leq B, [\mathbb{Q}(\alpha):\mathbb{Q}] \leq d\} < \infty$$

Order verification by increasing height: $h_0 < h_1 < h_2 < \cdots$

**No backward edges:** If Test($h_i$) depends on Test($h_j$) with $h_j > h_i$, we would have a backward edge. But arithmetic implications flow from:
- Lower height → higher height (specialization)
- Not the reverse (generalization requires new input)

**Step 2: Degree as Secondary Ordering**

For fixed height $h = B$, order by degree: $d_0 < d_1 < d_2 < \cdots$

By the **degree formula** for field extensions [Lang, Algebra]:
$$[\mathbb{Q}(\alpha):\mathbb{Q}] = \deg(\min_\alpha)$$

Degree is a natural number, hence well-ordered. Within height level $B$:
- Degree-$d$ tests precede degree-$(d+1)$ tests
- No circular dependencies

**Step 3: Galois Structure as Tertiary Ordering**

For fixed $(B, d)$, Galois groups are ordered by:
$$G_1 \leq G_2 \iff G_1 \text{ is a quotient of } G_2$$

The lattice of subgroups of $S_d$ is finite, providing a well-founded partial order.

**Step 4: DAG Verification**

Combining the three orderings lexicographically:
$$(h_1, d_1, G_1) < (h_2, d_2, G_2)$$
iff $h_1 < h_2$, or ($h_1 = h_2$ and $d_1 < d_2$), or ($h_1 = h_2$, $d_1 = d_2$, and $G_1 < G_2$).

This is a **well-order** on triples (product of well-orders).

**Edges in the verification DAG:**
- Test$(h, d, G)$ → Test$(h', d', G')$ only if $(h, d, G) < (h', d', G')$

No backward edges possible. DAG property verified.

**Step 5: Termination**

Each path through the DAG visits nodes in increasing order. Since:
- Heights are bounded (for any given problem)
- Degrees are bounded (for any given problem)
- Galois groups are finite

The path length is bounded by:
$$|V| \leq N(B, d) \cdot |\text{Subgroups}(S_d)| < \infty$$

Verification terminates.

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finiteness of bounded height/degree algebraic numbers.

2. **Well-Ordering of $\mathbb{N}$** [Peano Axioms]: Degrees form a well-order.

3. **Finite Galois Groups** [Galois Theory]: For degree-$d$ extensions, $\text{Gal} \leq S_d$.

4. **Lexicographic Product** [Order Theory]: Product of well-orders is well-ordered.

---

### Arithmetic Interpretation

> **Arithmetic verification is naturally acyclic: conditions at lower height/degree cannot depend on conditions at higher height/degree. This ensures termination of any verification procedure.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*
- [Lang 2002] S. Lang, *Algebra*, GTM 211
- [Kahn 1962] A.B. Kahn, *Topological sorting of large networks*, Comm. ACM
