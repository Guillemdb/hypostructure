# Non-Circularity Rule

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-non-circularity*

The barrier/surgery specifications are non-circular: no barrier depends on the very property it is supposed to establish.

---

## Arithmetic Formulation

### Setup

In arithmetic proofs, non-circularity means:
- Definitions don't reference themselves
- Proofs don't assume what they're proving
- Dependencies form a DAG (no cycles)

### Statement (Arithmetic Version)

**Theorem (Non-Circularity of Arithmetic Verification).** The arithmetic verification protocol satisfies:

1. **Definition non-circularity:** No arithmetic invariant is defined in terms of itself
2. **Proof non-circularity:** No theorem's proof assumes the theorem
3. **Dependency acyclicity:** The implication graph of arithmetic facts is a DAG

Specifically: the verification that $\text{Hom}(\mathfrak{B}_T, Z) = \emptyset$ does not assume the conjecture it aims to prove.

---

### Proof

**Step 1: Stratification of Arithmetic Invariants**

Arithmetic invariants are stratified by **logical complexity**:

| **Level** | **Invariants** | **Dependencies** |
|-----------|---------------|------------------|
| 0 | Field degree, minimal polynomial | None |
| 1 | Height, discriminant | Level 0 |
| 2 | Galois group, ramification | Levels 0-1 |
| 3 | L-function, conductor | Levels 0-2 |
| 4 | Special values, BSD invariants | Levels 0-3 |

**Non-circularity:** Each level depends only on lower levels.

**Step 2: Lock Verification is Level-Bounded**

The Lock check "$\text{Hom}(\mathfrak{B}_T, Z) = \emptyset$" proceeds by:

1. **Construct $\mathfrak{B}_T$:** Uses only the bad pattern definition (Level 2-3)
2. **Construct $\mathbb{H}(Z)$:** Uses height, Galois, L-function data (Levels 1-3)
3. **Check morphism existence:** Uses homological algebra (Level 4)

**Critical observation:** The conjecture $\text{Conj}(T, Z)$ is at Level 5 (conclusion). The Lock check uses only Levels 0-4. No circularity.

**Step 3: Barrier Specifications**

Each barrier checks a precondition:

**(a) Height barrier:**
$$\text{BarrierHeight}: h(Z) < B$$

This uses only the height definition (Level 1), not the conjecture (Level 5).

**(b) Galois barrier:**
$$\text{BarrierGalois}: \text{Gal}(K/\mathbb{Q}) \text{ solvable}$$

This uses Galois group computation (Level 2), not L-function properties (Level 3+).

**(c) L-function barrier:**
$$\text{BarrierL}: L(1/2, \pi) \neq 0$$

This uses explicit L-value computation (Level 3), independent of the conjecture.

**Step 4: Induction on Proof Depth**

**Claim:** No proof of depth $n$ assumes facts of depth $\geq n$.

**Proof by strong induction:**
- **Base:** Depth-0 facts (definitions) have no dependencies
- **Inductive:** A depth-$n$ fact uses only depth-$(n-1)$ facts (by stratification)

Hence no proof is circular.

**Step 5: Independence from Conjecture**

The conjecture $\text{Conj}(T, Z)$ (e.g., RH, Hodge, BSD) is at the **top level**.

**All intermediate certificates** are computed without assuming the conjecture:
- Heights: computed from minimal polynomials
- Galois groups: computed from factorization
- L-values: computed from Euler products (numerical or rigorous)
- Morphism obstructions: computed from dimension/weight mismatches

The final step: "$\text{Hom} = \emptyset \Rightarrow \text{Conj}$" is a **logical implication**, not a circular assumption.

---

### Example: BSD Conjecture

**Conjecture:** $\text{ord}_{s=1} L(E, s) = \text{rank } E(\mathbb{Q})$

**Verification levels:**
1. Compute $E(\mathbb{Q})$: Heights of generators (Level 1-2)
2. Compute $L(E, s)$: Euler product (Level 3)
3. Compute order of vanishing: Numerical approximation (Level 3)
4. Compare rank and order: Equality check (Level 4)

**Non-circularity:** At no point does the computation of rank or $L$-value assume BSD. The comparison is the final step.

---

### Key Arithmetic Ingredients

1. **Stratification of Logic** [Tarski 1933]: Truth predicates are stratified.

2. **Well-Founded Definitions** [Set Theory]: No self-referential definitions.

3. **Proof-Theoretic Ordinals** [Gentzen 1936]: Proofs have well-ordered depth.

4. **Independence Proofs** [Gödel 1931]: Consistency vs. provability.

---

### Arithmetic Interpretation

> **Arithmetic verification never assumes what it's trying to prove. Each step uses only previously established facts, ensuring logical soundness.**

---

### Literature

- [Tarski 1933] A. Tarski, *The concept of truth in formalized languages*
- [Gentzen 1936] G. Gentzen, *Die Widerspruchsfreiheit der reinen Zahlentheorie*
- [Gödel 1931] K. Gödel, *Über formal unentscheidbare Sätze*
