# LOCK-Antichain: Antichain-Surface Correspondence Lock — GMT Translation

## Original Statement (Hypostructure)

The antichain-surface correspondence lock shows that antichains in the resolution poset correspond to co-dimension-one surfaces, creating geometric barriers between resolution stages.

## GMT Setting

**Antichain:** Set of mutually incomparable elements in a poset

**Surface:** Co-dimension one rectifiable set

**Correspondence:** Antichains ↔ separating surfaces in flow space

## GMT Statement

**Theorem (Antichain-Surface Correspondence Lock).** For resolution flow on $\mathbf{I}_k(M)$:

1. **Resolution Poset:** $(\mathcal{P}, \leq)$ where $T_1 \leq T_2$ if $T_2$ resolves from $T_1$

2. **Antichain:** $A \subset \mathcal{P}$ with no $T_1, T_2 \in A$ satisfying $T_1 \leq T_2$

3. **Surface:** Each maximal antichain corresponds to separating hypersurface in $\mathbf{I}_k(M)$

4. **Lock:** Trajectories must cross antichain surfaces exactly once

## Proof Sketch

### Step 1: Resolution Poset Structure

**Definition:** The resolution poset $(\mathcal{P}, \leq)$:
- Elements: Currents $T \in \mathbf{I}_k(M)$
- Order: $T_1 \leq T_2$ if flow from $T_1$ reaches $T_2$

**Reference:** Stanley, R. (2012). *Enumerative Combinatorics*. Cambridge.

### Step 2: Antichains in Posets

**Definition:** An antichain $A \subset \mathcal{P}$ satisfies:
$$T_1, T_2 \in A, T_1 \neq T_2 \implies T_1 \not\leq T_2 \text{ and } T_2 \not\leq T_1$$

**Maximal Antichain:** Every element of $\mathcal{P}$ is comparable to some element of $A$.

**Dilworth's Theorem:** Width of poset equals minimum number of chains covering it.

**Reference:** Dilworth, R. P. (1950). A decomposition theorem for partially ordered sets. *Ann. of Math.*, 51, 161-166.

### Step 3: Geometric Realization

**Flow Space:** Space of all flow trajectories $\gamma: [0, T] \to \mathbf{I}_k(M)$

**Level Sets:** For functional $\Phi$, the level set:
$$\Phi^{-1}(c) = \{T : \Phi(T) = c\}$$

is co-dimension one in the flow direction.

### Step 4: Antichain as Cross-Section

**Theorem:** Maximal antichain $A$ corresponds to cross-section of flow:

*Proof:*
1. Maximal antichain intersects every chain (trajectory) exactly once
2. Each flow trajectory is a chain in $\mathcal{P}$
3. Therefore $A$ is a "time slice" of resolution process
4. Geometrically: co-dimension one surface transverse to flow

### Step 5: Separating Surfaces

**Definition:** Surface $\Sigma \subset \mathbf{I}_k(M)$ separates $T_-$ from $T_+$ if:
- $T_-$ and $T_+$ lie in different components of $\mathbf{I}_k(M) \setminus \Sigma$
- Any path from $T_-$ to $T_+$ crosses $\Sigma$

**Antichain Surface:** The geometric realization of antichain $A$ separates earlier from later states.

### Step 6: Slicing by Energy

**Energy Slicing:** For gradient flow with functional $\Phi$:
$$A_c = \{T : \Phi(T) = c, T \text{ on a flow line}\}$$

is an antichain (level sets are incomparable in flow order).

**Rectifiability:** Level sets of Lipschitz function are rectifiable:
$$\mathcal{H}^{n-1}(\Phi^{-1}(c)) < \infty$$

for almost every $c$.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §3.2.

### Step 7: Mirsky's Theorem

**Theorem (Mirsky):** Height of poset equals minimum number of antichains covering it.

**Reference:** Mirsky, L. (1971). A dual of Dilworth's decomposition theorem. *Amer. Math. Monthly*, 78, 876-877.

**Geometric:** Height bounds number of required surface crossings.

### Step 8: Lock Mechanism

**Single Crossing:** Each trajectory crosses antichain surface exactly once:

*Proof:*
1. Flow trajectory $\gamma$ is a chain in $\mathcal{P}$
2. Maximal antichain $A$ intersects every chain exactly once
3. Surface $\Sigma_A$ is crossed exactly once by $\gamma$

**Lock:** Cannot revisit states before antichain surface (no backtracking in resolution).

### Step 9: Morse-Theoretic View

**Morse Function:** If $\Phi$ is Morse, level sets are smooth except at critical values.

**Handle Decomposition:** Passing critical value changes topology by handle attachment.

**Reference:** Milnor, J. (1963). *Morse Theory*. Princeton.

**Antichain Surfaces:** Regular level sets between critical values form antichain surfaces.

### Step 10: Compilation Theorem

**Theorem (Antichain-Surface Correspondence Lock):**

1. **Poset:** Resolution order on $\mathbf{I}_k(M)$

2. **Antichain:** Mutually incomparable configurations

3. **Surface:** Co-dimension one geometric realization

4. **Lock:** Trajectories cross antichain surfaces exactly once

**Applications:**
- Stage separation in resolution algorithms
- Barrier construction for flows
- Decomposition of configuration space

## Key GMT Inequalities Used

1. **Dilworth:**
   $$\text{width}(\mathcal{P}) = \min |\text{chain cover}|$$

2. **Mirsky:**
   $$\text{height}(\mathcal{P}) = \min |\text{antichain cover}|$$

3. **Level Set Rectifiability:**
   $$\mathcal{H}^{n-1}(\Phi^{-1}(c)) < \infty \text{ a.e. } c$$

4. **Single Crossing:**
   $$|\gamma \cap \Sigma_A| = 1$$

## Literature References

- Stanley, R. (2012). *Enumerative Combinatorics*. Cambridge.
- Dilworth, R. P. (1950). Decomposition of partially ordered sets. *Ann. of Math.*, 51.
- Mirsky, L. (1971). Dual of Dilworth's theorem. *Amer. Math. Monthly*, 78.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Milnor, J. (1963). *Morse Theory*. Princeton.
