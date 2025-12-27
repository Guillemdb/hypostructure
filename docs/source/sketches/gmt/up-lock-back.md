# UP-LockBack: Lock-Back Theorem — GMT Translation

## Original Statement (Hypostructure)

The lock-back theorem shows that barriers blocking forward evolution also prevent backward escape, creating trapped regions.

## GMT Setting

**Lock:** Forward barrier preventing transition to bad states

**Lock-Back:** Backward barrier preventing escape from good states

**Trapped Region:** Invariant under both forward and backward flow

## GMT Statement

**Theorem (Lock-Back).** If barrier $B$ blocks forward transition $T \to T_{\text{bad}}$:

1. **Backward Lock:** $B$ also blocks backward transition from $T_{\text{good}}$

2. **Trapping:** The region behind $B$ is invariant

3. **Symmetry:** Energy-based locks are automatically symmetric

## Proof Sketch

### Step 1: Energy Barrier Symmetry

**Forward Barrier:** $\Phi(T) < E_0$ blocks transition to $\{S : \Phi(S) \geq E_1\}$ (since $\Phi$ decreases).

**Backward Barrier:** $\Phi(S) \geq E_1$ blocks transition to $\{T : \Phi(T) < E_0\}$ (since $\Phi$ increases backward).

**Symmetry:** Energy monotonicity gives two-sided barrier.

### Step 2: Topological Barrier Symmetry

**Forward Barrier:** $\pi_k(T) \not\cong \pi_k(B)$ blocks $T \to B$.

**Backward Barrier:** Same obstruction blocks $B \to T$.

**Symmetry:** Topological obstructions are symmetric (homotopy is symmetric relation).

### Step 3: Capacity Barrier Symmetry

**Forward:** $\text{Cap}(B) > 0$ requires energy to cross from $A$ to $F$.

**Backward:** Same capacity barrier requires energy to cross from $F$ to $A$.

**Symmetry:** Capacity is symmetric measure.

### Step 4: Trapping Region

**Definition:** Region $R$ is **trapped** if:
$$\varphi_t(R) = R \quad \forall t \in \mathbb{R}$$

**Construction:** Given barrier $B$ separating $A$ from $F$:
$$R = A \cup B \cup F$$

where flow stays in each component.

### Step 5: Invariant Manifold Structure

**Stable Manifold:** $W^s(M_i) = \{T : \omega(T) \subset M_i\}$

**Unstable Manifold:** $W^u(M_i) = \{T : \alpha(T) \subset M_i\}$

**Lock-Back:** $M_i$ is locked from both directions:
- Forward: can reach $M_i$ but not escape
- Backward: cannot reach $M_i$ from outside

**Reference:** Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.

### Step 6: Energy Level Lock-Back

**Energy Shell:** $\Phi^{-1}([a, b])$ is trapped:
- Forward: $\Phi$ decreases, cannot exceed $b$
- Backward: $\Phi$ increases, cannot go below $a$

**Consequence:** Flow in $\Phi^{-1}([a, b])$ stays in $\Phi^{-1}([a, b])$.

### Step 7: Attractor Basin Lock

**Basin of Attraction:** $\mathcal{B}(\mathcal{A}) = \{T : \omega(T) \subset \mathcal{A}\}$

**Lock-Back Property:** $\mathcal{B}(\mathcal{A})$ is positively invariant (trapped forward).

**Repeller Basin:** $\mathcal{B}^*(\mathcal{R}) = \{T : \alpha(T) \subset \mathcal{R}\}$ is negatively invariant (trapped backward).

### Step 8: Conley Decomposition

**Conley's Theory (1978):** Flow decomposes into:
$$M = \bigsqcup_i \mathcal{B}(M_i)$$

where $M_i$ are Morse sets.

**Reference:** Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. AMS.

**Lock-Back Structure:** Each basin is forward-trapped; connections between basins are one-way.

### Step 9: Time-Reversal

**Gradient Flow Reversal:** $\partial_t T = -\nabla \Phi$ reverses to $\partial_t T = +\nabla \Phi$.

**Lock Reversal:** Forward lock becomes backward lock under time reversal.

**Symmetry:** For gradient flows, lock-back is automatic by time-reversal symmetry of energy functional.

### Step 10: Compilation Theorem

**Theorem (Lock-Back):**

1. **Symmetry:** Energy/topological/capacity locks are symmetric

2. **Trapping:** Regions behind barriers are invariant

3. **Conley Structure:** Flow decomposes into trapped basins

4. **Two-Sided:** Forward barriers automatically give backward barriers

**Applications:**
- Invariant manifold theory
- Conley index computations
- Stability analysis

## Key GMT Inequalities Used

1. **Energy Monotonicity:**
   $$\Phi(\varphi_t(T)) \leq \Phi(T)$$ (forward), $$\Phi(\varphi_t(T)) \geq \Phi(T)$$ (backward)

2. **Topological Symmetry:**
   $$\pi_k(T) \not\cong \pi_k(S) \iff \pi_k(S) \not\cong \pi_k(T)$$

3. **Capacity Symmetry:**
   $$\text{Cap}(A \to B) = \text{Cap}(B \to A)$$

4. **Trapping:**
   $$\varphi_t(R) \subset R$$ (forward and backward)

## Literature References

- Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. AMS.
- Palis, J., de Melo, W. (1982). *Geometric Theory of Dynamical Systems*. Springer.
- Mischaikow, K., Mrozek, M. (2002). Conley index. *Handbook of Dynamical Systems*, Vol. 2.
- Salamon, D. (1990). Morse theory, the Conley index and Floer homology. *Bull. LMS*, 22.
