# LOCK-UniqueAttractor: Unique-Attractor Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-unique-attractor*

The dynamical system has a unique global attractor.

---

## Arithmetic Formulation

### Setup

"Unique attractor" in arithmetic means:
- All orbits converge to a single special locus
- Torsion points form the unique attractor for height dynamics
- Special points attract general points

### Statement (Arithmetic Version)

**Theorem (Arithmetic Unique Attractor).** Arithmetic dynamics has unique attractor:

1. **Height attractor:** Torsion subgroup $A_{\text{tors}}$ is the unique attractor for Néron-Tate dynamics
2. **Galois attractor:** Fixed field $K^G$ is the unique attractor for Galois descent
3. **Moduli attractor:** CM points attract in moduli space dynamics

---

### Proof

**Step 1: Torsion as Unique Attractor**

For abelian variety $A/K$:

**Height functional:** $\hat{h}: A(\bar{K}) \to \mathbb{R}_{\geq 0}$

**Attractor:** $\mathcal{A} = \ker(\hat{h}) = A_{\text{tors}}$

**Uniqueness proof:**
1. $\hat{h}(P) = 0 \iff P$ is torsion [Néron-Tate]
2. Under multiplication $[n]: A \to A$:
   $$\hat{h}([n]P) = n^2 \hat{h}(P)$$
3. Division by $n$ decreases height:
   $$\hat{h}([n^{-1}]P) = \frac{\hat{h}(P)}{n^2} \to 0$$
4. Limit is in $A_{\text{tors}}$

**Uniqueness:** Any other attractor would have $\hat{h} > 0$ elements, contradicting decay.

**Step 2: Galois Descent Attractor**

For Galois extension $K/k$:

**Dynamical system:** $G = \text{Gal}(K/k)$ acts on $X(K)$

**Attractor:** $X(k) = X(K)^G$ (fixed points)

**Uniqueness:**
- $X(k)$ is the maximal $G$-invariant subset
- Any point not in $X(k)$ has non-trivial orbit
- $X(k)$ is the unique global fixed point set

**Step 3: CM Attractor in Moduli**

**Moduli space:** $\mathcal{A}_g$ (moduli of principally polarized AVs)

**Special points:** CM points (AVs with complex multiplication)

**Attractor property [André-Oort]:**
- Sequences of CM points equidistribute to special subvarieties
- Non-CM points don't accumulate in the same way
- CM locus is the "attractor" for arithmetic sequences

**Step 4: Uniqueness via Bogomolov**

**Bogomolov conjecture** [Ullmo-Zhang 1998]:
For subvariety $V \subset A$ not containing translate of abelian subvariety:
$$\{P \in V(\bar{K}) : \hat{h}(P) < \epsilon\}$$
is not Zariski-dense for small $\epsilon$.

**Consequence:** The unique attractor (where $\hat{h} = 0$) is exactly $A_{\text{tors}}$.

**Uniqueness:** No other locus attracts—non-torsion points stay bounded away.

**Step 5: Attractor Basin**

**Basin of attraction:** All of $A(\bar{K})$

**Proof:** For any $P \in A(\bar{K})$:
$$\lim_{n \to \infty} \hat{h}([m^{-n}]P) = 0$$

for any $m \geq 2$ (assuming division exists).

**Global attractor:** $A_{\text{tors}}$ attracts everything.

**Step 6: Unique Attractor Certificate**

The unique attractor certificate:
$$K_{\text{UA}}^+ = (A_{\text{tors}}, \text{uniqueness proof}, \text{basin} = A(\bar{K}))$$

**Components:**
- **Attractor:** $A_{\text{tors}}$
- **Uniqueness:** $\ker(\hat{h}) = A_{\text{tors}}$ and Bogomolov gap
- **Basin:** All algebraic points

---

### Key Arithmetic Ingredients

1. **Néron-Tate Height** [Néron 1965]: Quadratic form with $\ker = A_{\text{tors}}$.
2. **Bogomolov-Ullmo-Zhang** [1998]: Height gap off special loci.
3. **André-Oort** [Pila 2011]: CM points in moduli.
4. **Galois Descent** [Weil 1956]: Fixed points under Galois.

---

### Arithmetic Interpretation

> **Arithmetic dynamics has a unique global attractor: the torsion subgroup. Néron-Tate height decreases under division, and all points flow toward torsion. Bogomolov-Ullmo-Zhang ensures no other locus can be an attractor. This unique attractor theorem is the arithmetic analog of convergence to equilibrium.**

---

### Literature

- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs*
- [Ullmo 1998] E. Ullmo, *Positivité et discrétion des points algébriques*
- [Zhang 1998] S.-W. Zhang, *Equidistribution of small points on abelian varieties*
- [Pila 2011] J. Pila, *O-minimality and the André-Oort conjecture*
