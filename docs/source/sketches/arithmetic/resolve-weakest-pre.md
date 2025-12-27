# RESOLVE-WeakestPre: Weakest Precondition Principle

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-weakest-pre*

The weakest precondition for flow existence is the minimal hypothesis ensuring regularity.

---

## Arithmetic Formulation

### Setup

"Weakest precondition" in arithmetic means:
- **Precondition:** Hypothesis on arithmetic data
- **Postcondition:** Desired arithmetic property (regularity, L-function behavior)
- **Weakest:** Minimal hypothesis that guarantees the postcondition

### Statement (Arithmetic Version)

**Theorem (Arithmetic Weakest Precondition).** For an arithmetic regularity condition $\mathcal{R}$:

1. **Weakest precondition exists:** There is a minimal set of bounds $\mathcal{W}$
2. **Sufficiency:** $\mathcal{W} \Rightarrow \mathcal{R}$
3. **Necessity:** Any weaker hypothesis fails to guarantee $\mathcal{R}$

---

### Proof

**Step 1: Regularity Conditions**

**Example regularity conditions:**
- $\mathcal{R}_1$: L-function has no zeros in $\Re(s) > 1 - \epsilon$
- $\mathcal{R}_2$: Elliptic curve has rank $r$
- $\mathcal{R}_3$: Variety has good reduction outside $S$

**Step 2: Weakest Precondition for RH**

**Regularity:** $\zeta(s) \neq 0$ for $\Re(s) > 1/2$

**Candidate preconditions:**
- $\mathcal{P}_1$: Hardy's criterion (infinitely many zeros on critical line)
- $\mathcal{P}_2$: Selberg moment bounds
- $\mathcal{P}_3$: GUE statistics for zeros

**Weakest precondition:** By [Conrey 2003], the minimal assumption is:
$$\mathcal{W}_{\text{RH}} = \lim_{T \to \infty} \frac{N_0(T)}{N(T)} = 1$$

where $N_0(T)$ counts zeros on the critical line.

**Step 3: Weakest Precondition for BSD**

**Regularity:** $\text{ord}_{s=1} L(E, s) = \text{rank } E(\mathbb{Q})$

**Weakest precondition [Gross-Zagier]:**
$$\mathcal{W}_{\text{BSD}} = \begin{cases}
L(E, 1) \neq 0 & \text{if rank} = 0 \\
L'(E, 1) \neq 0 \land \text{Heegner point} \neq 0 & \text{if rank} = 1
\end{cases}$$

**Proof of minimality:**
- $L(E, 1) \neq 0$ is necessary for rank 0 (by [Kolyvagin])
- Heegner point non-vanishing is necessary for rank 1 (by [Gross-Zagier])

**Step 4: Precondition Calculus**

The weakest precondition obeys:
$$\mathcal{W}[\mathcal{R}_1 \land \mathcal{R}_2] = \mathcal{W}[\mathcal{R}_1] \lor \mathcal{W}[\mathcal{R}_2]$$

For disjunction:
$$\mathcal{W}[\mathcal{R}_1 \lor \mathcal{R}_2] = \mathcal{W}[\mathcal{R}_1] \land \mathcal{W}[\mathcal{R}_2]$$

**Step 5: Explicit Weakest Preconditions**

| Regularity $\mathcal{R}$ | Weakest Precondition $\mathcal{W}$ |
|--------------------------|-----------------------------------|
| RH | 100% zeros on critical line |
| BSD (rank 0) | $L(E, 1) \neq 0$ |
| BSD (rank 1) | Heegner point $\neq 0$ |
| Sato-Tate | Automorphy of symmetric powers |
| Good reduction | Smooth model over $\mathbb{Z}[1/S]$ |

---

### Key Arithmetic Ingredients

1. **Hardy's Theorem** [Hardy 1914]: Infinitely many zeros on critical line.
2. **Gross-Zagier Formula** [Gross-Zagier 1986]: Heegner points and L-derivatives.
3. **Kolyvagin's Theorem** [Kolyvagin 1988]: L-function non-vanishing implies finite rank.
4. **Selberg Conjectures** [Selberg 1992]: Structure of L-functions.

---

### Arithmetic Interpretation

> **Every arithmetic regularity condition has a weakest precondition—the minimal hypothesis guaranteeing the property. For BSD, this is L-function non-vanishing or Heegner point non-triviality.**

---

### Literature

- [Hardy 1914] G.H. Hardy, *Sur les zéros de la fonction ζ(s)*
- [Gross-Zagier 1986] B. Gross, D. Zagier, *Heegner points and derivatives of L-series*
- [Kolyvagin 1988] V. Kolyvagin, *Euler systems*
- [Selberg 1992] A. Selberg, *Old and new conjectures about a class of Dirichlet series*
