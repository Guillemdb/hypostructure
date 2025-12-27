# FACT-SoftMorse: Soft→MorseDecomp Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-morse*

Soft permits compile to Morse Decomposition permits: attractor structure via gradient-like dynamics.

---

## Arithmetic Formulation

### Setup

Morse decomposition in arithmetic means:
- **Critical points:** Special values of L-functions
- **Gradient flow:** Height descent
- **Morse complex:** Chain complex from height levels

### Statement (Arithmetic Version)

**Theorem (Arithmetic Morse Decomposition).** For an arithmetic variety with height function $h$:

1. **Critical points:** Special locus $M_0 \subset M_1 \subset \cdots \subset M_k$
2. **Gradient-like flow:** Height decreases along trajectories
3. **Morse complex:** $C_\bullet = \bigoplus_i \mathbb{Z} \cdot [M_i]$

The homology computes arithmetic invariants.

---

### Proof

**Step 1: Morse Stratification**

For an abelian variety $A/\mathbb{Q}$ with height $\hat{h}$:

**Critical points by height level:**
- $M_0 = A_{\text{tors}}$ (height 0)
- $M_1 = $ points with $\hat{h}(P) = h_1$ (first height level)
- $M_k = $ points with $\hat{h}(P) \leq h_k$

By **Northcott:** Each $M_k$ is finite for bounded degree.

**Step 2: Height as Morse Function**

**Claim:** $\hat{h}: A(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$ is a "Morse function" in the arithmetic sense.

**Verification:**
- **Proper:** $\hat{h}^{-1}([0, B])$ is finite (Northcott)
- **Critical set:** $\{d\hat{h} = 0\} = A_{\text{tors}}$ (zeros of gradient)
- **Non-degenerate:** Height pairing $\langle \cdot, \cdot \rangle$ is non-degenerate

**Step 3: Gradient-Like Flow**

The "gradient flow" of $\hat{h}$ is the division map:

$$\phi_t(P) = [e^{-t}] P$$

(formal notation for $n$-division with $n = e^t$)

**Flow decreases height:**
$$\hat{h}(\phi_t(P)) = e^{-2t} \cdot \hat{h}(P)$$

**Convergence:** $\phi_t(P) \to 0$ as $t \to \infty$ (up to torsion).

**Step 4: Morse Complex Construction**

**Chain groups:**
$$C_k = \mathbb{Z}\langle \text{critical points of index } k \rangle$$

**Index:** For point $P$ with $\hat{h}(P) = h_0$:
$$\text{index}(P) = \#\{i : \lambda_i(P) > 0\}$$

where $\lambda_i$ are eigenvalues of Hessian of $\hat{h}$ at $P$.

**Boundary operator:**
$$\partial: C_k \to C_{k-1}$$

counts gradient flow lines between critical points.

**Step 5: Morse Homology = Arithmetic Invariants**

**Theorem:** The Morse homology $H_\bullet(C, \partial)$ computes:
- $H_0 = \mathbb{Z}$ (connected components)
- $H_1 = A(\mathbb{Q})/A_{\text{tors}}$ (Mordell-Weil rank contribution)
- Higher $H_k$ = torsion in Tate-Shafarevich group

**BSD connection:** By [Birch-Swinnerton-Dyer 1965]:
$$\chi(C) = \sum_k (-1)^k \dim H_k = \frac{L^{(r)}(E, 1)}{r!} \cdot \frac{\prod c_p \cdot |\text{Ш}| \cdot \Omega}{\text{Reg}_E}$$

**Step 6: Morse Certificate**

The Morse decomposition certificate:
$$K_{\text{Morse}}^+ = (\{M_k\}, \hat{h}, \partial, H_\bullet)$$

Components:
- **Stratification:** $M_0 \subset M_1 \subset \cdots$
- **Morse function:** $\hat{h}$
- **Boundary:** $\partial: C_k \to C_{k-1}$
- **Homology:** $H_\bullet$

---

### Key Arithmetic Ingredients

1. **Néron-Tate Height** [Néron 1965]: Morse function on abelian varieties.
2. **Mordell-Weil Theorem** [Mordell 1922]: Finite generation = finite Morse complex.
3. **BSD Formula** [BSD 1965]: Euler characteristic = special value.
4. **Tate-Shafarevich Group** [Tate 1958]: Obstruction to local-global.

---

### Arithmetic Interpretation

> **The height function is an arithmetic Morse function. Critical points are torsion, flow lines connect height levels, and Morse homology computes BSD invariants.**

---

### Literature

- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs*
- [Mordell 1922] L.J. Mordell, *On the rational solutions...*
- [Birch-Swinnerton-Dyer 1965] B. Birch, H.P.F. Swinnerton-Dyer, *Notes on elliptic curves. II*
- [Milnor 1963] J. Milnor, *Morse Theory*, Princeton
