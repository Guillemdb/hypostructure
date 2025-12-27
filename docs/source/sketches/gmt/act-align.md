# ACT-Align: Adjoint Surgery Principle — GMT Translation

## Original Statement (Hypostructure)

The adjoint surgery principle shows how surgery operations have adjoints: if one surgery connects configurations A→B, the adjoint connects B→A, providing reversibility in the resolution process.

## GMT Setting

**Adjoint:** Dual operation reversing direction

**Surgery Pair:** Forward surgery $\sigma$ has adjoint $\sigma^*$

**Reversibility:** Adjoint surgery reverses effect (up to control)

## GMT Statement

**Theorem (Adjoint Surgery).** For surgery operations on $\mathbf{I}_k(M)$:

1. **Forward Surgery:** $\sigma: T \mapsto T'$ resolves singularity

2. **Adjoint Surgery:** $\sigma^*: T' \mapsto T''$ with $T'' \approx T$

3. **Correspondence:** $(\sigma \circ \sigma^*)^n \to \text{id}$ in appropriate sense

4. **Energy:** $\mathbf{M}(\sigma(T)) - \mathbf{M}(T) = -(\mathbf{M}(\sigma^*(T')) - \mathbf{M}(T'))$

## Proof Sketch

### Step 1: Surgery as Cobordism

**Cobordism View:** Surgery $\sigma$ defines cobordism:
$$W: \partial W = T \sqcup (-T')$$

**Adjoint Cobordism:** $W^*: \partial W^* = T' \sqcup (-T'')$

**Reference:** Milnor, J. (1965). *Lectures on the h-Cobordism Theorem*. Princeton.

### Step 2: Adjoint in Hilbert Space

**Operator Adjoint:** For bounded $A: H_1 \to H_2$:
$$\langle Ax, y \rangle_2 = \langle x, A^*y \rangle_1$$

**Reference:** Reed, M., Simon, B. (1980). *Methods of Modern Mathematical Physics*. Academic Press.

**Surgery Analog:** $\sigma^*$ is adjoint in appropriate inner product on currents.

### Step 3: Surgery Inner Product

**Mass Pairing:** For $T, S \in \mathbf{I}_k(M)$:
$$\langle T, S \rangle = \int \langle T, S \rangle_x \, d\mathcal{H}^k(x)$$

when supports coincide.

**Adjoint Condition:**
$$\langle \sigma(T), S \rangle = \langle T, \sigma^*(S) \rangle$$

### Step 4: Cut-Paste Duality

**Forward:** Cut along $S^{k-1}$, paste $D^k \times S^{n-k-1}$

**Adjoint:** Cut along $S^{n-k-1}$, paste $S^{k-1} \times D^{n-k}$

**Reversal:** Adjoint surgery undoes forward surgery topologically.

### Step 5: Energy Balance

**Theorem:** Surgery energy satisfies:
$$\Delta E(\sigma) + \Delta E(\sigma^*) = 0$$

*Proof:*
- $\sigma$: removes neck, adds caps — mass change $\Delta_\sigma$
- $\sigma^*$: adds neck, removes caps — mass change $-\Delta_\sigma$
- Total: zero energy change for $\sigma \circ \sigma^*$

### Step 6: Handle Decomposition

**Handles:** $n$-manifold decomposes as:
$$M = h_0 \cup h_1 \cup \cdots \cup h_n$$

where $h_k$ is $k$-handle.

**Duality:** $k$-handle in $M$ ↔ $(n-k)$-handle in $M^*$ (dual).

**Reference:** Gompf, R., Stipsicz, A. (1999). *4-Manifolds and Kirby Calculus*. AMS.

### Step 7: Adjoint in Categories

**Categorical Adjunction:** Functors $L \dashv R$ satisfy:
$$\text{Hom}(L(A), B) \cong \text{Hom}(A, R(B))$$

**Reference:** MacLane, S. (1971). *Categories for the Working Mathematician*. Springer.

**Surgery Adjunction:** $\sigma \dashv \sigma^*$ in category of currents with cobordisms.

### Step 8: Reversibility

**Theorem:** $\sigma \circ \sigma^* \simeq \text{id}$ up to homotopy:

*Sketch:*
1. $\sigma$ creates standard piece
2. $\sigma^*$ removes standard piece
3. Composition returns to original (modulo small corrections)

**Control:** $\|\sigma \circ \sigma^*(T) - T\| \leq C\epsilon$ for $\epsilon$-surgery.

### Step 9: Symplectic Structure

**Symplectic Analog:** For symplectic manifolds, surgery preserves symplectic structure if:
$$\omega|_{\text{gluing region}} \text{ matches}$$

**Adjoint:** Symplectic adjoint surgery reverses orientation.

**Reference:** Weinstein, A. (1991). Contact surgery and symplectic handlebodies. *Hokkaido Math. J.*, 20, 241-251.

### Step 10: Compilation Theorem

**Theorem (Adjoint Surgery):**

1. **Existence:** Every surgery $\sigma$ has adjoint $\sigma^*$

2. **Energy:** $\Delta E(\sigma) = -\Delta E(\sigma^*)$

3. **Composition:** $\sigma \circ \sigma^* \approx \text{id}$

4. **Categorical:** Surgery forms adjoint pairs

**Applications:**
- Reversible resolution
- Energy-conserving deformations
- Cobordism theory

## Key GMT Inequalities Used

1. **Energy Balance:**
   $$\Delta E(\sigma) + \Delta E(\sigma^*) = 0$$

2. **Adjoint Condition:**
   $$\langle \sigma(T), S \rangle = \langle T, \sigma^*(S) \rangle$$

3. **Reversibility:**
   $$\|\sigma \circ \sigma^*(T) - T\| \leq C\epsilon$$

4. **Handle Duality:**
   $$h_k^M \leftrightarrow h_{n-k}^{M^*}$$

## Literature References

- Milnor, J. (1965). *h-Cobordism Theorem*. Princeton.
- Reed, M., Simon, B. (1980). *Mathematical Physics*. Academic Press.
- Gompf, R., Stipsicz, A. (1999). *4-Manifolds and Kirby Calculus*. AMS.
- MacLane, S. (1971). *Categories*. Springer.
- Weinstein, A. (1991). Contact surgery. *Hokkaido Math. J.*, 20.
