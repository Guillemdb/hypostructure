# UP-ShadowRetro: Shadow-Sector Retroactive — GMT Translation

## Original Statement (Hypostructure)

The shadow-sector retroactive theorem shows that topological obstructions detected later can be traced back to constrain earlier evolution.

## GMT Setting

**Shadow Sector:** Topological invariants carried by the flow

**Retroactive:** Future obstructions constrain past states

**Constraint Propagation:** Backward-in-time propagation of constraints

## GMT Statement

**Theorem (Shadow-Sector Retroactive).** If topological obstruction $\omega \in H^*(M)$ is detected at time $t_1$:

1. **Backward Propagation:** $\omega$ existed at all earlier times $t < t_1$

2. **Initial Constraint:** The initial data $T_0$ must have carried $\omega$

3. **Classification:** $\omega$ constrains the class of admissible initial data

## Proof Sketch

### Step 1: Topological Invariants

**Cohomology Classes:** For $T \in \mathbf{I}_k(M)$:
$$[T] \in H_k(M; \mathbb{Z})$$

**Pairing:** For $\omega \in H^k(M)$:
$$\langle [T], \omega \rangle = \int_T \omega$$

**Reference:** Spanier, E. H. (1966). *Algebraic Topology*. McGraw-Hill.

### Step 2: Invariance Under Flow

**Theorem (Topological Invariance):** Homotopy type is preserved by gradient flow:
$$\varphi_t: T \mapsto T' \implies [T] = [T'] \in H_k(M)$$

*Proof:* Gradient flow is homotopy. Homotopic maps induce same map on homology.

**Consequence:** If $\omega(T_1) \neq 0$ at $t_1$, then $\omega(T_0) \neq 0$ at $t_0 < t_1$.

### Step 3: Backward Propagation

**Backward Chain:** For $t_0 < t_1 < t_2 < \cdots < t_n$:
$$\omega(T_{t_n}) \neq 0 \implies \omega(T_{t_{n-1}}) \neq 0 \implies \cdots \implies \omega(T_{t_0}) \neq 0$$

**Retroactive Constraint:** Detecting $\omega$ at $t_n$ retroactively tells us about $T_{t_0}$.

### Step 4: Surgery and Topology

**Surgery Effect:** Surgery may change homology:
$$H_k(M_{\text{after}}) \neq H_k(M_{\text{before}})$$

**Connected Sum:**
$$H_k(M_1 \# M_2) \cong H_k(M_1) \oplus H_k(M_2)$$

(for $0 < k < n-1$).

**Reference:** Milnor, J. (1965). *Topology from the Differentiable Viewpoint*. Princeton.

**Retroactive:** Pre-surgery topology constrains what surgeries were possible.

### Step 5: Characteristic Classes

**Chern/Pontryagin Classes:** For vector bundles:
$$c_k(E) \in H^{2k}(M; \mathbb{Z})$$

**Invariance:** Characteristic classes are topological invariants.

**Retroactive Application:** If $c_k(T_1) \neq 0$, then the initial bundle had $c_k(T_0) \neq 0$.

**Reference:** Milnor, J., Stasheff, J. (1974). *Characteristic Classes*. Princeton.

### Step 6: Obstruction Theory

**Obstruction Classes:** For lifting problems:
$$\mathcal{O} \in H^{n+1}(X; \pi_n(F))$$

**Obstruction Vanishing:** Lift exists iff $\mathcal{O} = 0$.

**Reference:** Steenrod, N. (1951). *The Topology of Fibre Bundles*. Princeton.

**Retroactive:** If lift exists at $t_1$, then obstruction was zero at $t_0$.

### Step 7: Shadow Sector Definition

**Shadow Sector:** The collection of topological invariants:
$$\mathcal{S}(T) := (H_*(T), \pi_*(T), c_*(T), \ldots)$$

**Shadow Conservation:** $\mathcal{S}$ is conserved under continuous deformation.

**Retroactive Information:** $\mathcal{S}(T_1)$ fully determines $\mathcal{S}(T_0)$ (without surgery).

### Step 8: Constraint on Initial Data

**Admissible Initial Data:** Given target topology $\mathcal{S}_{\text{target}}$:
$$\text{Adm}(T_0) := \{T_0 : \mathcal{S}(T_0) = \mathcal{S}_{\text{target}}\}$$

**Retroactive Classification:** Knowing final topology determines initial topology class.

### Step 9: Applications

**Example 1: Genus Conservation**
- Genus $g$ is preserved under MCF (until singularity)
- If $g(T_1) = 2$, then $g(T_0) = 2$

**Example 2: Knot Invariants**
- Knot type preserved under homotopy
- Alexander polynomial, Jones polynomial retroactive

**Example 3: Characteristic Numbers**
- $\chi(M)$, signature $\sigma(M)$ retroactive
- Determine cobordism class

### Step 10: Compilation Theorem

**Theorem (Shadow-Sector Retroactive):**

1. **Backward Propagation:** Topological invariants propagate backward in time

2. **Initial Constraint:** Final topology constrains initial topology

3. **Surgery Tracking:** Topology changes at surgery are recorded

4. **Complete Information:** Shadow sector $\mathcal{S}$ provides complete topological record

**Applications:**
- Constrain initial data from final state
- Track topological changes through flow
- Classify admissible initial conditions

## Key GMT Inequalities Used

1. **Homology Invariance:**
   $$\varphi_t(T) \simeq T \implies [T] = [\varphi_t(T)]$$

2. **Pairing:**
   $$\langle [T], \omega \rangle = \int_T \omega$$

3. **Connected Sum:**
   $$H_k(M_1 \# M_2) = H_k(M_1) \oplus H_k(M_2)$$

4. **Shadow Conservation:**
   $$\mathcal{S}(T_1) = \mathcal{S}(T_0)$$ (no surgery)

## Literature References

- Spanier, E. H. (1966). *Algebraic Topology*. McGraw-Hill.
- Milnor, J. (1965). *Topology from the Differentiable Viewpoint*. Princeton.
- Milnor, J., Stasheff, J. (1974). *Characteristic Classes*. Princeton.
- Steenrod, N. (1951). *The Topology of Fibre Bundles*. Princeton.
- Hatcher, A. (2002). *Algebraic Topology*. Cambridge.
