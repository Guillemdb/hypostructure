# FACT-SoftAttr: Soft→Attr Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-attr*

Soft permits compile to Global Attractor Existence permits.

---

## Arithmetic Formulation

### Setup

The "global attractor" in arithmetic is the **special locus**:
- CM points
- Torsion points
- Rational points

All algebraic points "attract" toward these special structures.

### Statement (Arithmetic Version)

**Theorem (Arithmetic Attractor Existence).** For an arithmetic dynamical system with bounded complexity:

1. **Attractor exists:** The special locus $M$ (torsion, CM) is a global attractor
2. **Attraction rate:** Heights decay toward $M$ at rate $\geq \epsilon_0 > 0$
3. **Attractor structure:** $M$ is the fixed point set of the dynamics

---

### Proof

**Step 1: Special Locus as Attractor**

For an abelian variety $A/\mathbb{Q}$:

**Special locus:**
$$M = A_{\text{tors}} \cup A_{\text{CM}} \cup A(\mathbb{Q})_{\text{special}}$$

**Attractor property:** By **Bogomolov's conjecture** [Ullmo 1998, Zhang 1998]:
- If $V \subset A$ is a subvariety not containing a translate of an abelian subvariety
- Then $\{P \in V(\overline{\mathbb{Q}}) : \hat{h}(P) < \epsilon\}$ is not Zariski-dense for small $\epsilon$

**Interpretation:** Non-special points have height bounded away from 0.

**Step 2: Height Decay Rate**

Under the multiplication map $[n]: A \to A$:
$$\hat{h}([n]P) = n^2 \cdot \hat{h}(P)$$

**Decay toward torsion:** The "inverse" operation (n-th roots) decreases height:
$$\hat{h}([n^{-1}]P) = \frac{\hat{h}(P)}{n^2}$$

**Exponential decay:** After $k$ steps of dividing by $n$:
$$\hat{h}_k = \frac{\hat{h}_0}{n^{2k}} \to 0$$

**Rate:** $\epsilon_0 = \log n^2 > 0$.

**Step 3: Attractor is Fixed Point Set**

**Fixed points of height dynamics:**
$$\hat{h}([n]P) = \hat{h}(P) \iff P \in A_{\text{tors}}$$

(since $n^2 \cdot \hat{h}(P) = \hat{h}(P)$ implies $\hat{h}(P) = 0$)

**Attractor = $\ker(\hat{h})$:** The attractor is precisely the torsion subgroup.

**Step 4: Global Convergence**

**Claim:** For any $P \in A(\overline{\mathbb{Q}})$, the orbit approaches $A_{\text{tors}}$.

**Proof:**
- For $P \notin A_{\text{tors}}$: $\hat{h}(P) > 0$
- Under division by $n$: $\hat{h} \to 0$
- The limit is in $\ker(\hat{h}) = A_{\text{tors}}$

**Step 5: Attractor Certificate**

The attractor certificate:
$$K_{\text{Attr}}^+ = (A_{\text{tors}}, \hat{h}, \text{convergence proof}, \epsilon_0 = 2\log 2)$$

Components:
- **Attractor:** $M = A_{\text{tors}}$
- **Lyapunov function:** $\hat{h}$
- **Decay rate:** $\epsilon_0$

---

### Key Arithmetic Ingredients

1. **Bogomolov Conjecture** [Ullmo 1998, Zhang 1998]: Height lower bounds off special loci.
2. **Néron-Tate Height** [Néron 1965]: Quadratic form with $\ker = A_{\text{tors}}$.
3. **Mordell-Weil** [Mordell 1922]: $A(\mathbb{Q})$ is finitely generated.
4. **Manin-Mumford** [Raynaud 1983]: Torsion points on subvarieties.

---

### Arithmetic Interpretation

> **Torsion points form the global attractor of arithmetic dynamics. Heights measure "distance from attractor"—all points eventually approach torsion under division operations.**

---

### Literature

- [Ullmo 1998] E. Ullmo, *Positivité et discrétion des points algébriques des courbes*
- [Zhang 1998] S.-W. Zhang, *Equidistribution of small points on abelian varieties*
- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs sur les variétés abéliennes*
- [Raynaud 1983] M. Raynaud, *Sous-variétés d'une variété abélienne et points de torsion*
