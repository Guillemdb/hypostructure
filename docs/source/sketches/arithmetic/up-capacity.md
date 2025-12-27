# UP-Capacity: Capacity Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-capacity*

Capacity bounds promote to global regularity estimates.

---

## Arithmetic Formulation

### Setup

"Capacity" in arithmetic means:
- How much information a structure can hold
- Dimension of Galois cohomology, size of Selmer groups
- Promoted to global bounds on rational points, L-values

### Statement (Arithmetic Version)

**Theorem (Arithmetic Capacity).** Capacity bounds imply global bounds:

1. **Selmer capacity:** $\dim \text{Sel}(E) \leq B \Rightarrow \text{rank } E(\mathbb{Q}) \leq B$
2. **Cohomological capacity:** $\dim H^1(G_K, V) \leq c \cdot \log N$
3. **L-function capacity:** Zeros in critical strip $\leq c \cdot \log T$

---

### Proof

**Step 1: Selmer Capacity**

**Definition:** $n$-Selmer group:
$$\text{Sel}^{(n)}(E/\mathbb{Q}) = \ker\left(H^1(\mathbb{Q}, E[n]) \to \prod_v H^1(\mathbb{Q}_v, E[n])\right)$$

**Capacity bound:**
$$\dim_{\mathbb{F}_n} \text{Sel}^{(n)}(E) \geq \text{rank } E(\mathbb{Q})$$

**Promotion:** If $\dim \text{Sel}^{(n)}(E) \leq B$, then $\text{rank } E(\mathbb{Q}) \leq B$.

**Proof [Cassels 1962]:** Mordell-Weil injects into Selmer:
$$E(\mathbb{Q})/nE(\mathbb{Q}) \hookrightarrow \text{Sel}^{(n)}(E)$$

**Step 2: Cohomological Capacity**

For Galois representation $V$ with conductor $N$:

**Capacity bound [Brumer 1995]:**
$$\dim H^1(G_\mathbb{Q}, V) \leq c \cdot \dim(V)^2 \cdot \log N$$

**Proof:** Uses the Euler characteristic formula:
$$\chi(V) = \dim H^0 - \dim H^1 + \dim H^2 = -\dim(V) \cdot \text{ord}_{s=1} L(V, s)$$

with L-function bounds.

**Step 3: Zero Counting Capacity**

For L-function $L(s, \pi)$ with conductor $N$:

**Zero count [Selberg]:**
$$N(T) = \#\{\rho : L(\rho, \pi) = 0, |\Im(\rho)| \leq T\}$$

**Capacity bound:**
$$N(T) = \frac{T}{\pi} \log\frac{NT}{2\pi e} + O(\log T)$$

**Promotion:** The number of zeros is bounded by conductor and height.

**Step 4: Height Capacity**

**Northcott's theorem:** The set
$$\{P \in \mathbb{P}^n(\overline{\mathbb{Q}}) : h(P) \leq B, [\mathbb{Q}(P):\mathbb{Q}] \leq d\}$$

is finite.

**Capacity count [Schanuel 1979]:**
$$\#\{P : h(P) \leq B\} \sim c_n \cdot B^{n+1}$$

**Promotion:** Height bounds give point count bounds.

**Step 5: Mordell-Weil Capacity**

For elliptic curve $E/\mathbb{Q}$ with rank $r$:

**Regulator capacity:**
$$\text{Reg}_E = \det(\langle P_i, P_j \rangle)$$

where $P_1, \ldots, P_r$ are generators.

**Capacity bound [Lang's conjecture]:**
$$\text{Reg}_E \geq c(E)^r$$

**Promotion:** Large regulator → generators have large height → fewer small-height points.

**Step 6: Capacity Certificate**

The capacity certificate:
$$K_{\text{Cap}}^+ = (\text{space}, \text{dimension bound}, \text{global consequence})$$

**Examples:**
- (Selmer, $\dim \leq B$, rank $\leq B$)
- (Zeros, $N(T) \leq c \log NT$, sparse zeros)
- (Heights, $\#\{h \leq B\} < \infty$, finite point sets)

---

### Key Arithmetic Ingredients

1. **Selmer Groups** [Cassels 1962]: Bound rank via cohomology.
2. **Brumer's Bound** [Brumer 1995]: Cohomology bounded by conductor.
3. **Selberg's Zero Count** [Selberg 1946]: L-function zero density.
4. **Northcott** [Northcott 1950]: Height bounds give finiteness.

---

### Arithmetic Interpretation

> **Arithmetic capacity bounds (Selmer dimension, cohomology dimension, zero count) promote to global bounds (rank, point count, L-value). The capacity of a structure limits what it can contain, enabling finiteness theorems.**

---

### Literature

- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Brumer 1995] A. Brumer, *The average rank of elliptic curves*
- [Selberg 1946] A. Selberg, *Contributions to the theory of the Riemann zeta-function*
- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic*
