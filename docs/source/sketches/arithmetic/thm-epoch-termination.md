# Epoch Termination Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-epoch-termination*

Each epoch terminates in finite time, visiting finitely many nodes.

---

## Arithmetic Formulation

### Setup

An "epoch" in arithmetic verification corresponds to:
- Processing all algebraic numbers of a given height/degree
- Checking a finite collection of Galois orbits
- Verifying finitely many local conditions

### Statement (Arithmetic Version)

**Theorem (Arithmetic Epoch Termination).** Let $\mathcal{A}(B, d)$ be the set of verification tasks for algebraic numbers with:
- Height $h(\alpha) \leq B$
- Degree $[\mathbb{Q}(\alpha):\mathbb{Q}] \leq d$

Then each verification epoch terminates in finite time:
$$T_{\text{epoch}} \leq C(B, d) < \infty$$

where $C(B, d)$ depends only on the height bound and degree bound.

---

### Proof

**Step 1: Finiteness of Task Set**

By **Northcott's theorem** [Northcott 1950]:
$$|\mathcal{A}(B, d)| = N(B, d) < \infty$$

The number of algebraic numbers to verify is finite.

**Step 2: Bounded Verification per Element**

For each $\alpha \in \mathcal{A}(B, d)$, verification involves:

**(a) Minimal polynomial computation:**
$$\min_\alpha \in \mathbb{Z}[x], \quad \deg(\min_\alpha) = [\mathbb{Q}(\alpha):\mathbb{Q}] \leq d$$

Coefficients bounded by: $|a_i| \leq C_0(h(\alpha), d) \leq C_0(B, d)$

By **Mahler's inequality** [Mahler 1964]:
$$h(\alpha) \geq \frac{1}{d}\log |a_d| - \log 2$$

Hence coefficient size is polynomially bounded in $e^{dB}$.

**(b) Galois group computation:**
By factoring $\min_\alpha$ over finite extensions. Number of steps bounded by:
$$S_{\text{Galois}} \leq d! \cdot \text{poly}(\log |a_i|) \leq d! \cdot \text{poly}(dB)$$

**(c) Local condition checks:**
For each prime $p \leq P(B, d)$ (effective bound from verification protocol):
- Reduction mod $p$: $O(\log |a_i|)$ operations
- Local Galois group: $O(d^2)$ operations

**Step 3: Total Epoch Time**

Summing over all elements:
$$T_{\text{epoch}} \leq N(B, d) \cdot \left(S_{\text{poly}} + S_{\text{Galois}} + S_{\text{local}}\right)$$

By [Loxton-van der Poorten 1983]:
$$N(B, d) \leq C \cdot (2d+1)^{d+1} \cdot e^{2dB}$$

Each factor is finite for fixed $(B, d)$. Hence:
$$T_{\text{epoch}} \leq C(B, d) < \infty$$

**Step 4: Independence from Previous Epochs**

The epoch for $(B, d)$ processes only elements in $\mathcal{A}(B, d)$.

Elements from previous epochs $(B', d') < (B, d)$ are already processed. By the DAG structure (thm-dag-structure), no revisits occur.

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finite task set.

2. **Mahler's Inequality** [Mahler 1964]: Coefficient bounds from height.

3. **Galois Group Algorithms** [Cohen 1993]: Polynomial-time Galois computation.

4. **Local-Global Principle** [Hasse 1923]: Local checks are finite.

---

### Arithmetic Interpretation

> **Each "level" of arithmetic complexity (fixed height and degree) contains finitely many objects and can be fully processed in finite time.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*
- [Mahler 1964] K. Mahler, *An inequality for the discriminant of a polynomial*
- [Cohen 1993] H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138
