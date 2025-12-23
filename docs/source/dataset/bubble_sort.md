# Termination of Bubble Sort Algorithm

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Termination and polynomial-time complexity of the bubble sort algorithm |
| **System Type** | $T_{\text{algorithmic}}$ (Discrete Combinatorial Dynamics) |
| **Target Claim** | Regular Termination Confirmed |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2025-12-23 |
| **Status** | Final |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{algorithmic}}$ is a **good type** (finite stratification + constructible caps).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** (Definition {prf:ref}`def-automation-guarantee`), hence profile extraction, admissibility, and surgery are computed automatically by the framework factories.

**Certificate:**
$$K_{\mathrm{Auto}}^+ = (T_{\text{algorithmic}}\ \text{good},\ \text{AutomationGuarantee holds},\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, RESOLVE-AutoSurgery})$$

---

## Abstract

This document presents a **machine-checkable proof object** for the **termination and polynomial-time complexity of bubble sort** using the Hypostructure framework.

**Approach:** We instantiate the algorithmic hypostructure with bubble sort on permutations of $n$ elements. The potential function $\Phi(\sigma) = $ number of inversions serves as a strict Lyapunov function. Each swap operation decreases inversions by exactly 1, and the system reaches equilibrium (sorted array) in at most $\binom{n}{2}$ steps.

**Result:** The Lock is blocked via strict monotonic descent on a finite discrete potential. All certificates are positive; the proof is unconditional. This instance serves as the **Family I (Stable) baseline** in the Hypostructure problem taxonomy.

---

## Theorem Statement

::::{prf:theorem} Termination of Bubble Sort
:label: thm-bubble-sort

**Given:**
- State space: $\mathcal{X} = S_n$, the symmetric group on $n$ elements
- Dynamics: For each $i = 1, \ldots, n-1$: if $\sigma(i) > \sigma(i+1)$, swap $\sigma(i) \leftrightarrow \sigma(i+1)$
- Initial data: Any permutation $\sigma_0 \in S_n$

**Claim (REG-BubbleSort):** The algorithm terminates in finite time with the sorted permutation. The total number of swaps is at most $\binom{n}{2}$, hence the time complexity is $O(n^2)$.

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\mathcal{X}$ | State space $S_n$ (permutations of $\{1,\ldots,n\}$) |
| $\Phi(\sigma)$ | Inversion count $\|\{(i,j): i < j, \sigma(i) > \sigma(j)\}\|$ |
| $\mathfrak{D}$ | Swap cost (1 per swap) |
| $\sigma_*$ | Sorted permutation (identity): $\sigma_*(i) = i$ |
| $S$ | Safe sector: all permutations $S_n$ |

::::

---

## Part 0: Interface Permit Implementation

### 0.1 Core Interface Permits (Nodes 1-12)

#### Template: $D_E$ (Energy Interface)
- [x] **Height Functional $\Phi$:** $\Phi(\sigma) = \text{inv}(\sigma) = \|\{(i,j): i < j, \sigma(i) > \sigma(j)\}\|$
- [x] **Dissipation Rate $\mathfrak{D}$:** $\mathfrak{D} = 1$ per swap operation
- [x] **Energy Inequality:** Each swap strictly decreases $\Phi$ by exactly 1
- [x] **Bound Witness:** $B = \Phi(\sigma_0) \le \binom{n}{2}$

#### Template: $\mathrm{Rec}_N$ (Recovery Interface)
- [x] **Bad Set $\mathcal{B}$:** Empty (no singularities in finite discrete system)
- [x] **Recovery Map $\mathcal{R}$:** Not needed
- [x] **Event Counter $\#$:** $N(T) \le \binom{n}{2}$ (bounded by initial inversions)
- [x] **Finiteness:** Trivially satisfied (strict monotonic descent on $\mathbb{N}$)

#### Template: $C_\mu$ (Compactness Interface)
- [x] **Symmetry Group $G$:** Trivial (no continuous symmetry exploited)
- [x] **Group Action $\rho$:** Identity
- [x] **Quotient Space:** $\mathcal{X}//G = S_n$
- [x] **Concentration Measure:** Discrete topology prevents concentration (finite state space)

#### Template: $\mathrm{SC}_\lambda$ (Scaling Interface)
- [x] **Scaling Action:** None (discrete system, no continuous scaling)
- [x] **Height Exponent $\alpha$:** Not applicable (no scaling)
- [x] **Dissipation Exponent $\beta$:** Not applicable (no scaling)
- [x] **Criticality:** Trivial (discrete dynamics)

#### Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)
- [x] **Parameter Space $\Theta$:** $\{n \in \mathbb{N}\}$ (problem size)
- [x] **Parameter Map $\theta$:** $\theta(\sigma) = n$ (fixed for given problem)
- [x] **Reference Point $\theta_0$:** $n$ (problem size)
- [x] **Stability Bound:** Parameter $n$ is constant

#### Template: $\mathrm{Cap}_H$ (Capacity Interface)
- [x] **Capacity Functional:** Cardinality $|\cdot|$
- [x] **Singular Set $\Sigma$:** Empty (all states are regular)
- [x] **Codimension:** $\text{codim}(\Sigma) = \infty$
- [x] **Capacity Bound:** $\text{Cap}(\Sigma) = 0$

#### Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)
- [x] **Gradient Operator $\nabla$:** Discrete gradient on state space
- [x] **Critical Set $M$:** Sorted permutation $\{\sigma_*\}$
- [x] **Łojasiewicz Exponent $\theta$:** $\theta = 1$ (discrete linear decrease)
- [x] **Łojasiewicz-Simon Inequality:** $\Delta\Phi = -1$ per swap (strict decrease)

#### Template: $\mathrm{TB}_\pi$ (Topology Interface)
- [x] **Topological Invariant $\tau$:** Permutation parity (even/odd)
- [x] **Sector Classification:** Single sector (adjacent swaps preserve parity class; algorithm works within each class)
- [x] **Sector Preservation:** Parity is preserved by adjacent transpositions
- [x] **Tunneling Events:** None (topology is discrete and rigid)

#### Template: $\mathrm{TB}_O$ (Tameness Interface)
- [x] **O-minimal Structure $\mathcal{O}$:** Finite discrete structure
- [x] **Definability $\text{Def}$:** All states and transitions are finitely describable
- [x] **Singular Set Tameness:** $\Sigma = \emptyset$
- [x] **Cell Decomposition:** Trivial (finite set)

#### Template: $\mathrm{TB}_\rho$ (Mixing Interface)
- [x] **Measure $\mathcal{M}$:** Counting measure on $S_n$
- [x] **Invariant Measure $\mu$:** Dirac mass on $\sigma_*$ (unique equilibrium)
- [x] **Mixing Time $\tau_{\text{mix}}$:** $\tau_{\text{mix}} \le \binom{n}{2}$ (finite time to equilibrium)
- [x] **Mixing Property:** Gradient flow to unique fixed point (no recurrence)

#### Template: $\mathrm{Rep}_K$ (Dictionary Interface)
- [x] **Language $\mathcal{L}$:** Sequence notation $[\sigma(1), \sigma(2), \ldots, \sigma(n)]$
- [x] **Dictionary $D$:** Permutation array representation
- [x] **Complexity Measure $K$:** $K(\sigma) = \Phi(\sigma)$ (inversion count)
- [x] **Faithfulness:** Polynomial-time computable and decidable

#### Template: $\mathrm{GC}_\nabla$ (Gradient Interface)
- [x] **Metric Tensor $g$:** Discrete metric (Hamming-type on permutations)
- [x] **Vector Field $v$:** Swap operation that decreases inversions
- [x] **Gradient Compatibility:** $\Phi$ is a strict Lyapunov function: $\Delta\Phi = -1$
- [x] **Resolution:** Pure gradient descent (monotonic, no oscillation)

### 0.2 Boundary Interface Permits (Nodes 13-16)
*System is closed (finite discrete state space with no external input/output). Boundary nodes skipped.*

### 0.3 The Lock (Node 17)
- [x] **Category $\mathbf{Hypo}_{T_{\text{algo}}}$:** Algorithmic hypostructures
- [x] **Universal Bad Pattern $\mathcal{H}_{\text{bad}}$:** Non-termination (infinite loop)
- [x] **Primary Tactic Selected:** E1 + E2
- [x] **Tactic Logic:**
    * $I(\mathcal{H}) = \Phi(t) \in [0, \binom{n}{2}] \cap \mathbb{N}$ (bounded discrete potential)
    * $I(\mathcal{H}_{\text{bad}}) = $ infinite execution (no termination)
    * Conclusion: Strict monotonic descent on finite $\mathbb{N}$ $\implies$ $\mathrm{Hom} = \emptyset$
- [x] **Exclusion Tactics:**
  - [x] E1 (Dimension): Finite state space $|S_n| = n!$ prevents infinite descent
  - [x] E2 (Invariant Mismatch): Strict decrease $\Delta\Phi = -1$ contradicts non-termination

---

## Part I: The Instantiation (Thin Object Definitions)

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**
* **State Space ($\mathcal{X}$):** $S_n$, the symmetric group on $n$ elements (permutations of $\{1, 2, \ldots, n\}$).
* **Metric ($d$):** Discrete metric: $d(\sigma, \tau) = |\{i : \sigma(i) \ne \tau(i)\}|$ (Hamming distance).
* **Measure ($\mu$):** Counting measure on $S_n$.

### **2. The Potential ($\Phi^{\text{thin}}$)**
* **Height Functional ($\Phi$):** $\Phi(\sigma) = \text{inv}(\sigma) = |\{(i,j): i < j, \sigma(i) > \sigma(j)\}|$ (inversion count).
* **Gradient/Slope ($\nabla$):** The discrete gradient points toward reducing inversions via adjacent swaps.
* **Scaling Exponent ($\alpha$):** Not applicable (discrete system).

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**
* **Dissipation Rate ($D$):** $\mathfrak{D} = 1$ per swap operation.
* **Dynamics:** Adjacent swap: if $\sigma(i) > \sigma(i+1)$, swap $\sigma(i) \leftrightarrow \sigma(i+1)$.

### **4. The Invariance ($G^{\text{thin}}$)**
* **Symmetry Group ($\text{Grp}$):** Trivial group $\{e\}$ (no symmetry reduction exploited by algorithm).
* **Scaling ($\mathcal{S}$):** None (discrete system).

---

## Part II: Sieve Execution

### Level 1: Conservation (Nodes 1-3)

#### Node 1: EnergyCheck ($D_E$)

**Question:** Is the height functional bounded along trajectories?

**Step-by-step execution:**
1. [x] Write the energy functional: $\Phi(\sigma) = \text{inv}(\sigma)$
2. [x] Bound the initial energy: $\Phi(\sigma_0) \le \binom{n}{2}$ (maximum inversions)
3. [x] Each swap operation: If $\sigma(i) > \sigma(i+1)$, swap removes this inversion
4. [x] Evaluate change: $\Delta\Phi = \Phi(\sigma') - \Phi(\sigma) = -1$ per swap
5. [x] Monotonicity: $\Phi(t+1) = \Phi(t) - 1$ whenever a swap occurs
6. [x] Lower bound: $\Phi(\sigma) \ge 0$ for all $\sigma$ (inversions are non-negative)
7. [x] Result: $\Phi$ is strictly decreasing and bounded below

**Certificate:**
* [x] $K_{D_E}^+ = (\Phi, \mathfrak{D}, \Delta\Phi = -1)$ → **Go to Node 2**

---

#### Node 2: ZenoCheck ($\mathrm{Rec}_N$)

**Question:** Are recovery events (surgeries) finite?

**Step-by-step execution:**
1. [x] Identify discrete events: Each swap is a discrete event
2. [x] Count maximum events: Total swaps $\le \Phi(\sigma_0) \le \binom{n}{2}$
3. [x] Since $\Phi$ decreases by 1 per swap and is bounded below by 0
4. [x] Maximum $\binom{n}{2}$ swaps possible
5. [x] No Zeno behavior: Finite discrete steps, each takes unit time

**Certificate:**
* [x] $K_{\mathrm{Rec}_N}^+ = (N \le \binom{n}{2})$ → **Go to Node 3**

---

#### Node 3: CompactCheck ($C_\mu$)

**Question:** Does energy concentrate into canonical profiles?

**Step-by-step execution:**
1. [x] State space is finite: $|S_n| = n!$
2. [x] Discrete topology: No concentration possible (finite set)
3. [x] Canonical profile: Sorted permutation $\sigma_* = \text{id}$
4. [x] All trajectories converge to $\sigma_*$ (unique equilibrium)
5. [x] Compactness: Trivial (finite space is compact)

**Certificate:**
* [x] $K_{C_\mu}^+ = (|S_n| = n!, \text{finite discrete})$ → **Go to Node 4**

---

### Level 2: Duality & Structure (Nodes 4-7)

#### Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)

**Question:** Is the blow-up profile subcritical?

**Step-by-step execution:**
1. [x] System is discrete: No continuous scaling parameter
2. [x] Problem size $n$ is fixed for any given instance
3. [x] No scaling dynamics: Discrete state transitions
4. [x] Subcriticality: Trivially satisfied (no criticality in discrete system)

**Certificate:**
* [x] $K_{\mathrm{SC}_\lambda}^+ = (\text{discrete, no scaling})$ → **Go to Node 5**

---

#### Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)

**Question:** Are system constants stable under perturbation?

**Step-by-step execution:**
1. [x] Identify parameters: Problem size $n$ (fixed)
2. [x] No external parameters: Algorithm is deterministic
3. [x] Stability: $n$ does not change during execution
4. [x] Result: Trivially stable (no perturbations)

**Certificate:**
* [x] $K_{\mathrm{SC}_{\partial c}}^+ = (n\ \text{fixed})$ → **Go to Node 6**

---

#### Node 6: GeomCheck ($\mathrm{Cap}_H$)

**Question:** Does the singular set have codimension $\ge 2$?

**Step-by-step execution:**
1. [x] Define singular set: $\Sigma = \emptyset$ (all states are regular)
2. [x] Every permutation is a valid state
3. [x] No singularities in discrete finite state space
4. [x] Codimension: $\text{codim}(\Sigma) = \infty$

**Certificate:**
* [x] $K_{\mathrm{Cap}_H}^+ = (\Sigma = \emptyset)$ → **Go to Node 7**

---

#### Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)

**Question:** Is there a spectral gap / Łojasiewicz inequality?

**Step-by-step execution:**
1. [x] Energy-dissipation relation: $\Delta\Phi = -1$ per swap
2. [x] Spectral gap: Discrete gap of 1 (strict integer decrease)
3. [x] Łojasiewicz inequality: $|\Delta\Phi| = 1 \ge 1 \cdot d(\sigma, \sigma_*)^1$
4. [x] Linear convergence: Distance to equilibrium decreases by at least 1 per step

**Certificate:**
* [x] $K_{\mathrm{LS}_\sigma}^+ = (\Delta\Phi = -1, \text{strict gap})$ → **Go to Node 8**

---

### Level 3: Topology (Nodes 8-9)

#### Node 8: TopoCheck ($\mathrm{TB}_\pi$)

**Question:** Is the topological sector preserved/simplified?

**Step-by-step execution:**
1. [x] Identify topological invariant: Permutation parity (even/odd)
2. [x] Adjacent transpositions preserve parity
3. [x] Note: Bubble sort only uses adjacent swaps
4. [x] However, the sorted state is reachable from any permutation of same parity
5. [x] Single sector: All even (or odd) permutations form connected component
6. [x] No tunneling: Parity is exactly preserved

**Certificate:**
* [x] $K_{\mathrm{TB}_\pi}^+ = (\text{parity preserved})$ → **Go to Node 9**

---

#### Node 9: TameCheck ($\mathrm{TB}_O$)

**Question:** Is the singular set definable in an o-minimal structure?

**Step-by-step execution:**
1. [x] Singular set: $\Sigma = \emptyset$
2. [x] State space: Finite discrete set $S_n$
3. [x] Definability: Every finite set is definable in any o-minimal structure
4. [x] Cell decomposition: Trivial (finite discrete points)

**Certificate:**
* [x] $K_{\mathrm{TB}_O}^+ = (\text{finite discrete})$ → **Go to Node 10**

---

### Level 4: Mixing & Complexity (Nodes 10-11)

#### Node 10: ErgoCheck ($\mathrm{TB}_\rho$)

**Question:** Does the flow exhibit dissipative/mixing behavior?

**Step-by-step execution:**
1. [x] Check monotonicity: $\Phi$ strictly decreases until equilibrium
2. [x] Check recurrence: No—system converges to unique fixed point $\sigma_*$
3. [x] Convergence: $\sigma(t) \to \sigma_*$ in at most $\binom{n}{2}$ steps
4. [x] Mixing time: $\tau_{\text{mix}} \le \binom{n}{2}$ (finite and explicit)

**Certificate:**
* [x] $K_{\mathrm{TB}_\rho}^+ = (\text{dissipative}, \tau_{\text{mix}} \le \binom{n}{2})$ → **Go to Node 11**

---

#### Node 11: ComplexCheck ($\mathrm{Rep}_K$)

**Question:** Is the description complexity bounded?

**Step-by-step execution:**
1. [x] Identify complexity measure: $K(\sigma) = \text{inv}(\sigma)$
2. [x] Computational complexity: Inversion count computable in $O(n^2)$
3. [x] Total operations: At most $\binom{n}{2} = O(n^2)$ swaps
4. [x] Time complexity: $O(n^2)$ (polynomial)
5. [x] Description length: Permutation requires $O(n \log n)$ bits

**Certificate:**
* [x] $K_{\mathrm{Rep}_K}^+ = (K = O(n^2), \text{polynomial-time})$ → **Go to Node 12**

---

### Level 5: Gradient Structure (Node 12)

#### Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)

**Question:** Is there oscillatory behavior in the dynamics?

**Step-by-step execution:**
1. [x] Energy is a strict Lyapunov function: $\Delta\Phi = -1 < 0$ (strict decrease)
2. [x] Monotonic descent: No backtracking, no oscillation
3. [x] Gradient flow: Each swap is a greedy descent step
4. [x] Result: **Monotonic** — no oscillation

**Certificate:**
* [x] $K_{\mathrm{GC}_\nabla}^- = (\Phi, \text{strict Lyapunov})$
→ **Go to Node 13**

---

### Level 6: Boundary (Node 13)

#### Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)

**Question:** Is the system open (external input/output coupling)?

**Step-by-step execution:**
1. [x] State space: Finite closed set $S_n$
2. [x] No external input: Algorithm operates on given permutation
3. [x] No external output coupling: Pure state transformation
4. [x] Therefore: Closed system

**Certificate:**
* [x] $K_{\mathrm{Bound}_\partial}^- = (\text{closed system})$ → **Go to Node 17**

---

### Level 7: The Lock (Node 17)

#### Node 17: LockCheck ($\mathrm{Cat}_{\mathrm{Hom}}$)

**Question:** Is $\text{Hom}(\mathcal{H}_{\text{bad}}, \mathcal{H}) = \emptyset$?

**Step-by-step execution:**
1. [x] Define $\mathcal{H}_{\text{bad}}$: Non-terminating trajectory (infinite loop)
2. [x] Apply Tactic E1 (Dimension/Finiteness):
   - State space is finite: $|S_n| = n!$
   - Potential $\Phi \in [0, \binom{n}{2}] \cap \mathbb{N}$ (finite discrete range)
   - Strict decrease: $\Delta\Phi = -1$ per swap
   - Lower bound: $\Phi \ge 0$
3. [x] Apply Tactic E2 (Invariant Mismatch):
   - Non-termination requires infinite steps
   - But $\Phi$ can decrease at most $\binom{n}{2}$ times
   - After $\le \binom{n}{2}$ swaps, $\Phi = 0$ (sorted state)
   - At $\Phi = 0$, no inversions remain, algorithm terminates
   - Contradiction: Cannot have infinite execution
4. [x] Verify: No bad pattern (non-termination) can embed

**Certificate:**
* [x] $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}} = (\text{E1+E2}, \text{non-termination excluded})$

**Lock Status:** **BLOCKED** ✓

---

## Part II-B: Upgrade Pass

### Inc-to-Positive Upgrades

| Original | Upgraded To | Mechanism | Reference |
|----------|-------------|-----------|-----------|
| — | — | — | — |

*Note: All certificates were positive on first pass. No inc certificates generated.*

---

## Part II-C: Breach/Surgery Protocol

*No breaches occurred during the sieve execution. The discrete dynamics on finite state space $S_n$ are inherently regular.*

**Breach Log:** EMPTY

---

## Part III-A: Lyapunov Reconstruction

*Not required: The inversion count $\Phi(\sigma) = \text{inv}(\sigma)$ already serves as a valid strict Lyapunov function with unit dissipation per swap. No ghost extension needed.*

---

## Part III-B: Metatheorem Extraction

### **1. Surgery Admissibility (RESOLVE-AutoAdmit)**
*Not applicable: No singularities occur in finite discrete algorithmic dynamics.*

### **2. Structural Surgery (RESOLVE-AutoSurgery)**
*Not applicable: No surgery needed.*

### **3. The Lock (Node 17)**
* **Question:** $\text{Hom}(\text{Bad}, \mathcal{H}) = \emptyset$?
* **Bad Pattern:** Non-termination (infinite execution)
* **Tactic E1 (Dimension/Finiteness):** Finite state space + strict monotonic descent prevents infinite trajectories
* **Tactic E2 (Invariant Mismatch):** Bounded discrete potential with strict decrease contradicts non-termination
* **Result:** **BLOCKED** ($K_{\mathrm{Lock}}^{\mathrm{blk}}$)

---

## Part III-C: Obligation Ledger

### Table 1: Introduced Obligations

| ID | Node | Certificate | Obligation | Missing | Status |
|----|------|-------------|------------|---------|--------|
| — | — | — | — | — | — |

*No obligations introduced.*

### Table 2: Discharge Events

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| — | — | — | — |

*No discharge events.*

### Table 3: Remaining Obligations

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| — | — | — |

**Ledger Validation:** $\mathsf{Obl}(\Gamma_{\mathrm{final}}) = \varnothing$ ✓

---

## Part IV: Final Certificate Chain

### Validity Checklist

1. [x] All required nodes executed with explicit certificates
2. [x] No barriers breached (all checks passed)
3. [x] No inc certificates (all positive)
4. [x] Lock certificate obtained: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
5. [x] No unresolved obligations
6. [x] No Lyapunov reconstruction needed
7. [x] No surgery protocol needed
8. [x] Result extraction completed

### Certificate Accumulation Trace

```
Node 1:  K_{D_E}^+ (strict inversion decrease)
Node 2:  K_{Rec_N}^+ (finite swaps ≤ C(n,2))
Node 3:  K_{C_μ}^+ (finite discrete space)
Node 4:  K_{SC_λ}^+ (no scaling)
Node 5:  K_{SC_∂c}^+ (n fixed)
Node 6:  K_{Cap_H}^+ (Σ = ∅)
Node 7:  K_{LS_σ}^+ (strict gap Δ=-1)
Node 8:  K_{TB_π}^+ (parity preserved)
Node 9:  K_{TB_O}^+ (finite definable)
Node 10: K_{TB_ρ}^+ (dissipative, τ_mix ≤ C(n,2))
Node 11: K_{Rep_K}^+ (polynomial O(n²))
Node 12: K_{GC_∇}^- (strict Lyapunov)
Node 13: K_{Bound_∂}^- (closed)
Node 17: K_{Cat_Hom}^{blk} (E1+E2)
```

### Final Certificate Set

$$\Gamma_{\mathrm{final}} = \{K_{D_E}^+, K_{\mathrm{Rec}_N}^+, K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{SC}_{\partial c}}^+, K_{\mathrm{Cap}_H}^+, K_{\mathrm{LS}_\sigma}^+, K_{\mathrm{TB}_\pi}^+, K_{\mathrm{TB}_O}^+, K_{\mathrm{TB}_\rho}^+, K_{\mathrm{Rep}_K}^+, K_{\mathrm{GC}_\nabla}^-, K_{\mathrm{Bound}_\partial}^-, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}\}$$

### Conclusion

**REGULAR TERMINATION CONFIRMED**

The bubble sort algorithm terminates in at most $\binom{n}{2} = O(n^2)$ steps for all initial permutations. The inversion count provides a strict Lyapunov function that guarantees convergence to the sorted state.

---

## Formal Proof

::::{prf:proof} Proof of Theorem {prf:ref}`thm-bubble-sort`

**Phase 1: Instantiation**
Instantiate the algorithmic hypostructure with:
- State space $\mathcal{X} = S_n$ (permutations)
- Dynamics: Adjacent swap when $\sigma(i) > \sigma(i+1)$
- Initial data: $\sigma_0 \in S_n$

**Phase 2: Potential Bounds**
The inversion count $\Phi(\sigma) = |\{(i,j): i < j, \sigma(i) > \sigma(j)\}|$ satisfies:
$$0 \le \Phi(\sigma) \le \binom{n}{2}$$

Each adjacent swap of out-of-order elements decreases $\Phi$ by exactly 1:
$$\Delta\Phi = -1 \text{ per swap}$$

**Phase 3: Termination**
Since $\Phi \in \mathbb{N}$ and $\Delta\Phi = -1 < 0$ whenever a swap occurs:
- The algorithm can perform at most $\Phi(\sigma_0) \le \binom{n}{2}$ swaps
- When $\Phi = 0$, no inversions remain, hence $\sigma = \sigma_*$ (sorted)
- At sorted state, no swaps occur, algorithm terminates

**Phase 4: Complexity**
Total swaps: $T \le \binom{n}{2} = \frac{n(n-1)}{2} = O(n^2)$

Each swap requires $O(1)$ operations, hence total time complexity is $O(n^2)$.

**Phase 5: Lock Exclusion**
By Tactics E1 (finite state space) and E2 (strict monotonic descent), no non-terminating trajectory can exist:
$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}: \quad \mathrm{Hom}(\mathcal{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset$$

**Phase 6: Conclusion**
Regular termination follows with polynomial-time complexity $O(n^2)$. $\square$

::::

---

## Verification Summary

| Component | Status | Certificate |
|-----------|--------|-------------|
| Energy Bound | Positive | $K_{D_E}^+$ |
| Surgery Finiteness | Positive | $K_{\mathrm{Rec}_N}^+$ |
| Compactness | Positive | $K_{C_\mu}^+$ |
| Scaling Analysis | Positive | $K_{\mathrm{SC}_\lambda}^+$ |
| Parameter Stability | Positive | $K_{\mathrm{SC}_{\partial c}}^+$ |
| Singular Codimension | Positive | $K_{\mathrm{Cap}_H}^+$ |
| Stiffness Gap | Positive | $K_{\mathrm{LS}_\sigma}^+$ |
| Topology Preservation | Positive | $K_{\mathrm{TB}_\pi}^+$ |
| Tameness | Positive | $K_{\mathrm{TB}_O}^+$ |
| Mixing/Dissipation | Positive | $K_{\mathrm{TB}_\rho}^+$ |
| Complexity Bound | Positive | $K_{\mathrm{Rep}_K}^+$ |
| Gradient Structure | Negative | $K_{\mathrm{GC}_\nabla}^-$ |
| Boundary | Negative | $K_{\mathrm{Bound}_\partial}^-$ |
| Lock | **BLOCKED** | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |
| Obligation Ledger | EMPTY | — |
| **Final Status** | **UNCONDITIONAL** | — |

---

## References

- D. E. Knuth, *The Art of Computer Programming, Vol. 3: Sorting and Searching*, Addison-Wesley (1973)
- R. Sedgewick, *Algorithms*, 4th ed., Addison-Wesley (2011)
- T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, MIT Press (2009)

---

## Appendix: Replay Bundle (Machine-Checkability)

This proof object is replayed by providing:
1. `trace.json`: ordered node outcomes
2. `certs/`: serialized certificates with payload hashes
3. `inputs.json`: thin objects and initial-state hash
4. `closure.cfg`: promotion/closure settings

**Replay acceptance criterion:** The checker recomputes the same $\Gamma_{\mathrm{final}}$ and emits `FINAL`.

---

## Executive Summary: The Proof Dashboard

### 1. System Instantiation (The Physics)

| Object | Definition | Role |
| :--- | :--- | :--- |
| **Arena ($\mathcal{X}$)** | $S_n$ (permutations) | State Space |
| **Potential ($\Phi$)** | $\text{inv}(\sigma) = |\{(i,j): i<j, \sigma(i)>\sigma(j)\}|$ | Lyapunov Functional |
| **Cost ($\mathfrak{D}$)** | $\mathfrak{D} = 1$ per swap | Discrete Dissipation |
| **Invariance ($G$)** | Trivial $\{e\}$ | No Symmetry Reduction |

### 2. Execution Trace (The Logic)

| Node | Check | Outcome | Certificate Payload | Γ (Certificate Accumulation) |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Energy Bound | YES | $\Delta\Phi = -1$, $\Phi \in [0,\binom{n}{2}]$ | $\{K_{D_E}^+\}$ |
| **2** | Zeno Check | YES | $N \le \binom{n}{2}$ swaps | $\Gamma_1 \cup \{K_{\mathrm{Rec}}^+\}$ |
| **3** | Compact Check | YES | Finite discrete space $|S_n|=n!$ | $\Gamma_2 \cup \{K_{C_\mu}^+\}$ |
| **4** | Scale Check | YES | No scaling (discrete) | $\Gamma_3 \cup \{K_{\mathrm{SC}_\lambda}^+\}$ |
| **5** | Param Check | YES | $n$ fixed | $\Gamma_4 \cup \{K_{\mathrm{SC}_{\partial c}}^+\}$ |
| **6** | Geom Check | YES | $\Sigma = \emptyset$ | $\Gamma_5 \cup \{K_{\mathrm{Cap}}^+\}$ |
| **7** | Stiffness Check | YES | $\Delta\Phi = -1$ (strict gap) | $\Gamma_6 \cup \{K_{\mathrm{LS}}^+\}$ |
| **8** | Topo Check | YES | Parity preserved | $\Gamma_7 \cup \{K_{\mathrm{TB}_\pi}^+\}$ |
| **9** | Tame Check | YES | Finite discrete | $\Gamma_8 \cup \{K_{\mathrm{TB}_O}^+\}$ |
| **10** | Ergo Check | YES | $\tau_{\text{mix}} \le \binom{n}{2}$ | $\Gamma_9 \cup \{K_{\mathrm{TB}_\rho}^+\}$ |
| **11** | Complex Check | YES | $O(n^2)$ polynomial-time | $\Gamma_{10} \cup \{K_{\mathrm{Rep}}^+\}$ |
| **12** | Oscillate Check | NO | Strict Lyapunov (monotonic) | $\Gamma_{11} \cup \{K_{\mathrm{GC}}^-\}$ |
| **13** | Boundary Check | CLOSED | No external coupling | $\Gamma_{12} \cup \{K_{\mathrm{Bound}}^-\}$ |
| **--** | **SURGERY** | **N/A** | — | $\Gamma_{13}$ |
| **--** | **RE-ENTRY** | **N/A** | — | $\Gamma_{13}$ |
| **17** | **LOCK** | **BLOCK** | E1+E2 | $\Gamma_{13} \cup \{K_{\mathrm{Lock}}^{\mathrm{blk}}\} = \Gamma_{\mathrm{final}}$ |

### 3. Lock Mechanism (The Exclusion)

| Tactic | Description | Status | Reason / Mechanism |
| :--- | :--- | :---: | :--- |
| **E1** | Dimension | PASS | Finite state space $|S_n| = n!$ + bounded potential |
| **E2** | Invariant | PASS | Strict decrease $\Delta\Phi = -1$ contradicts infinite loop |
| **E3** | Positivity | N/A | — |
| **E4** | Integrality | N/A | — |
| **E5** | Functional | N/A | — |
| **E6** | Causal | N/A | — |
| **E7** | Thermodynamic | N/A | — |
| **E8** | Holographic | N/A | — |
| **E9** | Ergodic | N/A | — |
| **E10** | Definability | N/A | — |

### 4. Final Verdict

* **Status:** UNCONDITIONAL
* **Obligation Ledger:** EMPTY
* **Singularity Set:** $\Sigma = \emptyset$
* **Primary Blocking Tactic:** E1+E2 (Finite discrete potential + Strict monotonic descent)

---

## Document Information

| Field | Value |
|-------|-------|
| Document Type | Proof Object |
| Framework | Hypostructure v1.0 |
| Problem Class | Algorithmic (Discrete Combinatorics) |
| System Type | $T_{\text{algorithmic}}$ |
| Verification Level | Machine-checkable |
| Inc Certificates | 0 introduced, 0 discharged |
| Final Status | **UNCONDITIONAL** |
| Generated | 2025-12-23 |

---
