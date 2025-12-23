# The Collatz Conjecture (3n+1 Problem)

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Does every positive integer eventually reach 1 under the Collatz map? |
| **System Type** | $T_{\text{discrete}}$ (Discrete Dynamical System) |
| **Target Claim** | Zeno Horizon (Node 2) |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2025-12-23 |
| **Status** | Final |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for diagnostic analysis via the Universal Singularity Modules, though key limitations prevent full automation.

- **Type witness:** $T_{\text{discrete}}$ is a **bounded type** (finite stratification exists but lacks crucial closure properties).
- **Automation limitation:** The Hypostructure satisfies partial automation guarantees but **fails at Node 2 (ZenoCheck)** due to the absence of uniform bounds on stopping times.

**Certificate:**
$$K_{\mathrm{Auto}}^{\mathrm{partial}} = (T_{\text{discrete}}\ \text{well-defined},\ \text{Node 1 automated},\ \text{Node 2+ conditional on termination})$$

---

## Abstract

This document presents a **machine-checkable diagnostic trace** for the **Collatz Conjecture** using the Hypostructure framework.

**Approach:** We instantiate the discrete hypostructure with the Collatz map $T: \mathbb{N} \to \mathbb{N}$ defined by $T(n) = n/2$ if $n$ even, $T(n) = 3n+1$ if $n$ odd. The trajectory from initial value $n_0$ generates a discrete sequence. We define energy $E(n) = \log_2(n)$ (roughly measuring "size"). Each iteration is a discrete event.

**Key Diagnostic Finding:** Node 2 (ZenoCheck) asks: "Are discrete events finite for all starting points?" We **cannot prove this**. No known local method provides a uniform bound on stopping time. The map is not monotonic, local induction fails, and the global structure is opaque.

**Result:** The Sieve issues $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ (inconclusive certificate) at Node 2, which persists through the entire diagnostic. The Lock receives this unresolved obligation and returns $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$ (HORIZON verdict). This is the framework's formal acknowledgment that the Collatz conjecture lies beyond its current capacity—a genuine **Zeno Horizon** at the very start of the Sieve.

---

## Theorem Statement

::::{prf:theorem} Collatz Conjecture (3n+1 Problem)
:label: thm-collatz

**Given:**
- State space: $\mathcal{X} = \mathbb{N}$ (positive integers)
- Dynamics: The Collatz map
  $$T(n) = \begin{cases} n/2 & \text{if } n \text{ even} \\ 3n+1 & \text{if } n \text{ odd} \end{cases}$$
- Initial data: $n_0 \in \mathbb{N}$

**Claim (Collatz Conjecture):** For all $n_0 \in \mathbb{N}$, there exists $k < \infty$ such that $T^k(n_0) = 1$.

Equivalently: The stopping time $\tau(n) = \min\{k \ge 0 : T^k(n) = 1\}$ is finite for all $n \in \mathbb{N}$.

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\mathcal{X}$ | State space $\mathbb{N}$ |
| $T(n)$ | Collatz map |
| $E(n)$ | Energy/height $\log_2(n)$ |
| $\mathfrak{D}$ | Cost per iteration: $\mathfrak{D} = 1$ |
| $\tau(n)$ | Stopping time to reach 1 |
| $N(n)$ | Number of iterations (event count) |

::::

---

## Part 0: Interface Permit Implementation

### 0.1 Core Interface Permits (Nodes 1-12)

#### Template: $D_E$ (Energy Interface)
- [x] **Height Functional $\Phi$:** $E(n) = \log_2(n)$ (tracks "size" on log scale)
- [x] **Dissipation Rate $\mathfrak{D}$:** $\mathfrak{D} = 1$ per iteration (discrete cost)
- [x] **Energy Inequality:** On trajectories that terminate: $E(T(n)) < E(n)$ on average (conditional)
- [x] **Bound Witness:** $E(n) \le E(n_0)$ is not uniformly guaranteed; trajectories can temporarily increase

#### Template: $\mathrm{Rec}_N$ (Recovery Interface)
- [ ] **Bad Set $\mathcal{B}$:** Unknown (cycles other than {1,4,2,1}? divergent trajectories?)
- [ ] **Recovery Map $\mathcal{R}$:** Not applicable
- [ ] **Event Counter $\#$:** $N(n) = \tau(n)$ (number of iterations to reach 1)
- [x] **Finiteness:** **UNKNOWN** — this is the entire conjecture

#### Template: $C_\mu$ (Compactness Interface)
- [x] **Symmetry Group $G$:** Trivial (no useful symmetry)
- [x] **Group Action $\rho$:** Identity
- [x] **Quotient Space:** $\mathcal{X}$ (no reduction via symmetry)
- [ ] **Concentration Measure:** Not applicable (state space is discrete and unbounded)

#### Template: $\mathrm{SC}_\lambda$ (Scaling Interface)
- [x] **Scaling Action:** $n \mapsto \lambda n$ (multiplicative scaling)
- [x] **Height Exponent $\alpha$:** $E(\lambda n) = \log_2(\lambda n) = \log_2(\lambda) + \log_2(n) = \alpha \log_2(\lambda) + E(n)$, $\alpha = 1$
- [ ] **Dissipation Exponent $\beta$:** Not well-defined (iteration count is not scale-invariant)
- [x] **Criticality:** Scaling structure does not yield subcriticality control

#### Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)
- [x] **Parameter Space $\Theta$:** None (map is fixed)
- [x] **Parameter Map $\theta$:** Constant
- [x] **Reference Point $\theta_0$:** $(3,1,2)$ (coefficients in map definition)
- [x] **Stability Bound:** Trivial (no parameter variation)

#### Template: $\mathrm{Cap}_H$ (Capacity Interface)
- [ ] **Capacity Functional:** Not applicable (discrete space)
- [ ] **Singular Set $\Sigma$:** Unknown (potential divergent orbits or cycles)
- [ ] **Codimension:** Not defined
- [ ] **Capacity Bound:** Not computable

#### Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)
- [ ] **Gradient Operator $\nabla$:** Not applicable (discrete system)
- [x] **Critical Set $M$:** $\{1, 2, 4\}$ (the known cycle)
- [ ] **Łojasiewicz Exponent $\theta$:** Not applicable
- [ ] **Łojasiewicz-Simon Inequality:** Not applicable

#### Template: $\mathrm{TB}_\pi$ (Topology Interface)
- [x] **Topological Invariant $\tau$:** Parity structure (odd/even)
- [x] **Sector Classification:** Odd vs even integers (but map crosses between them)
- [x] **Sector Preservation:** Not preserved (odd $\mapsto$ even, even $\mapsto$ may stay even or become odd)
- [x] **Tunneling Events:** Constant (each odd step transitions to even)

#### Template: $\mathrm{TB}_O$ (Tameness Interface)
- [x] **O-minimal Structure $\mathcal{O}$:** $\mathbb{R}_{\text{an}}$ (map is piecewise linear, hence definable)
- [x] **Definability $\text{Def}$:** Map is o-minimal definable
- [ ] **Singular Set Tameness:** Unknown
- [ ] **Cell Decomposition:** Unknown

#### Template: $\mathrm{TB}_\rho$ (Mixing Interface)
- [x] **Measure $\mathcal{M}$:** Counting measure on $\mathbb{N}$
- [x] **Invariant Measure $\mu$:** $\{1,2,4\}$ cycle
- [ ] **Mixing Time $\tau_{\text{mix}}$:** Not applicable (dissipative, not mixing)
- [x] **Mixing Property:** Trajectories converge to fixed cycle (conjectured)

#### Template: $\mathrm{Rep}_K$ (Dictionary Interface)
- [x] **Language $\mathcal{L}$:** Binary representation of $n$
- [x] **Dictionary $D$:** $n \mapsto \text{binary string}$
- [x] **Complexity Measure $K$:** $K(n) = \lceil \log_2(n) \rceil$ (bit length)
- [x] **Faithfulness:** Bit length bounded by energy

#### Template: $\mathrm{GC}_\nabla$ (Gradient Interface)
- [ ] **Metric Tensor $g$:** Not applicable (discrete system)
- [x] **Vector Field $v$:** Discrete map $T$
- [ ] **Gradient Compatibility:** Not applicable
- [ ] **Resolution:** Cannot determine (no Lyapunov function proven)

### 0.2 Boundary Interface Permits (Nodes 13-16)
*System is on $\mathbb{N}$ with no external coupling. Boundary nodes not applicable.*

### 0.3 The Lock (Node 17)
- [x] **Category $\mathbf{Hypo}_{T_{\text{discrete}}}$:** Discrete dynamical systems
- [ ] **Universal Bad Pattern $\mathcal{H}_{\text{bad}}$:** Infinite orbit or non-trivial cycle
- [ ] **Primary Tactic Selected:** None available
- [x] **Tactic Logic:**
    * We cannot construct invariants that exclude bad patterns
    * No dimension argument (state space is 0-dimensional at each point)
    * No monotonicity (trajectories can increase)
    * No computable bound on orbit length
- [x] **Exclusion Tactics:**
  - [ ] E1 (Dimension): Not applicable (discrete, no embedding theory helps)
  - [ ] E2 (Invariant Mismatch): No known global invariant
  - [ ] E3-E10: None applicable

---

## Part I: The Instantiation (Thin Object Definitions)

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**
* **State Space ($\mathcal{X}$):** $\mathbb{N} = \{1, 2, 3, \ldots\}$, the positive integers.
* **Metric ($d$):** $d(m, n) = |m - n|$ (standard discrete metric).
* **Measure ($\mu$):** Counting measure on $\mathbb{N}$.

### **2. The Potential ($\Phi^{\text{thin}}$)**
* **Height Functional ($\Phi$):** $E(n) = \log_2(n)$.
* **Gradient/Slope ($\nabla$):** Not defined (discrete system).
* **Scaling Exponent ($\alpha$):** Under $n \to \lambda n$, $E \to E + \log_2(\lambda)$. Additive, not power-law.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**
* **Dissipation Rate ($D$):** $\mathfrak{D} = 1$ per iteration.
* **Dynamics:**
  $$T(n) = \begin{cases} n/2 & \text{if } n \equiv 0 \pmod{2} \\ 3n+1 & \text{if } n \equiv 1 \pmod{2} \end{cases}$$

### **4. The Invariance ($G^{\text{thin}}$)**
* **Symmetry Group ($\text{Grp}$):** Trivial $\{e\}$ (no useful symmetries).
* **Scaling ($\mathcal{S}$):** Multiplicative scaling $n \mapsto \lambda n$ (non-affine action on discrete space).

---

## Part II: Sieve Execution

### Level 1: Conservation (Nodes 1-3)

#### Node 1: EnergyCheck ($D_E$)

**Question:** Is the height functional bounded along trajectories?

**Step-by-step execution:**
1. [x] Write the energy functional: $E(n) = \log_2(n)$
2. [x] Consider a trajectory: $n_0, n_1 = T(n_0), n_2 = T(n_1), \ldots$
3. [x] If trajectory reaches 1: Then eventually $E = 0$, so trajectory is bounded
4. [x] If trajectory diverges or cycles: Energy may be unbounded or bounded to cycle
5. [x] **Conditional on termination**: Each terminating trajectory has $E(n_k) \to 0$
6. [x] Empirical evidence: All tested $n < 2^{68}$ terminate
7. [x] Result: **Conditionally YES** — if trajectories terminate, energy is bounded

**Certificate:**
* [x] $K_{D_E}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove all trajectories terminate"}, \mathsf{missing}: K_{\text{term}}^+, \mathsf{code}: \text{OPEN\_CONJECTURE}, \mathsf{trace}: \text{Node 2})$ → **Go to Node 2**

---

#### Node 2: ZenoCheck ($\mathrm{Rec}_N$)

**Question:** Are recovery events (discrete iterations) finite?

**Step-by-step execution:**
1. [x] Identify discrete events: Each application of $T$ is one event
2. [x] Event counter: $N(n) = \tau(n) = \min\{k : T^k(n) = 1\}$
3. [x] Question: Is $N(n) < \infty$ for all $n \in \mathbb{N}$?
4. [x] **This is exactly the Collatz conjecture**
5. [x] Attempt local bounds:
   - No monotonicity: $T(27) = 82 > 27$
   - No inductive argument works
   - Longest known: $\tau(27) = 111$ steps
   - No closed-form bound on $\tau(n)$ as function of $n$
6. [x] Statistical evidence: All $n < 2^{68}$ verified to terminate
7. [x] Theoretical status: **OPEN PROBLEM**

**Certificate:**
* [x] $K_{\mathrm{Rec}_N}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove } \tau(n) < \infty \ \forall n", \mathsf{missing}: K_{\text{bound}}^+ \text{ (uniform stopping time bound)}, \mathsf{code}: \text{NON\_PAINLEVE}, \mathsf{trace}: \text{"No local induction; map non-monotonic; global structure unknown"})$

**Barrier Status:** **INCONCLUSIVE** at the SECOND NODE

**Routing:** This INC certificate is **mandatory** (Remark {prf:ref}`rem-mandatory-inc`). Cannot proceed without assumption. → **Conditional continuation to Node 3**

---

#### Node 3: CompactCheck ($C_\mu$)

**Question:** Does energy concentrate into canonical profiles?

**Step-by-step execution:**
1. [x] State space is discrete: $\mathbb{N}$
2. [x] No notion of "concentration" in continuous sense
3. [x] Profile: The only known attractor is the cycle $\{1, 2, 4\}$
4. [x] **Conditional on conjecture**: All trajectories reach this cycle
5. [x] Result: **Conditionally YES** (concentration to fixed cycle)

**Certificate:**
* [x] $K_{C_\mu}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove all orbits reach } \{1,2,4\}\text{ cycle"}, \mathsf{missing}: K_{\mathrm{Rec}_N}^+, \mathsf{code}: \text{DEPENDENT\_ON\_NODE\_2}, \mathsf{trace}: \text{Node 2})$ → **Go to Node 4**

---

### Level 2: Duality & Structure (Nodes 4-7)

#### Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)

**Question:** Is the blow-up profile subcritical?

**Step-by-step execution:**
1. [x] No blow-up in finite time (discrete system)
2. [x] Scaling: $T(\lambda n) \ne \lambda T(n)$ (map is not scale-invariant)
3. [x] Cannot define subcriticality index in standard sense
4. [x] Result: **Not applicable** (discrete system lacks scale structure)

**Certificate:**
* [x] $K_{\mathrm{SC}_\lambda}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Assess criticality"}, \mathsf{missing}: K_{\text{scale}}^+ \text{ (scale-invariance)}, \mathsf{code}: \text{NO\_SCALING\_SYMMETRY}, \mathsf{trace}: \text{"Discrete map; no continuous scaling"})$ → **Go to Node 5**

---

#### Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)

**Question:** Are system constants stable under perturbation?

**Step-by-step execution:**
1. [x] System has fixed parameters: $3n+1$ (not $an+b$ for variable $a,b$)
2. [x] No parameter variation to check
3. [x] Result: **Trivially stable** (no parameters)

**Certificate:**
* [x] $K_{\mathrm{SC}_{\partial c}}^+ = (\text{fixed map}, \text{no parameters})$ → **Go to Node 6**

---

#### Node 6: GeomCheck ($\mathrm{Cap}_H$)

**Question:** Does the singular set have codimension $\ge 2$?

**Step-by-step execution:**
1. [x] Singular set: Potential divergent orbits or unexpected cycles
2. [x] State space is discrete (0-dimensional at each point)
3. [x] Codimension not well-defined in discrete setting
4. [x] Result: **Not applicable**

**Certificate:**
* [x] $K_{\mathrm{Cap}_H}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Characterize singular set"}, \mathsf{missing}: K_{\text{geom}}^+ \text{ (geometric structure)}, \mathsf{code}: \text{DISCRETE\_TOPOLOGY}, \mathsf{trace}: \text{"No capacity theory for } \mathbb{N}\text{"})$ → **Go to Node 7**

---

#### Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)

**Question:** Is there a spectral gap / Łojasiewicz inequality?

**Step-by-step execution:**
1. [x] No gradient structure (discrete map)
2. [x] No Lyapunov function known
3. [x] No spectral theory (not a linear operator)
4. [x] Result: **Not applicable**

**Certificate:**
* [x] $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove Lyapunov decrease"}, \mathsf{missing}: K_{\text{Lyap}}^+ \text{ (Lyapunov function)}, \mathsf{code}: \text{NO\_GRADIENT\_STRUCTURE}, \mathsf{trace}: \text{"Discrete; non-monotonic; no known Lyapunov"})$ → **Go to Node 8**

---

### Level 3: Topology (Nodes 8-9)

#### Node 8: TopoCheck ($\mathrm{TB}_\pi$)

**Question:** Is the topological sector preserved/simplified?

**Step-by-step execution:**
1. [x] Topological structure: Parity (odd/even)
2. [x] Map behavior: Odd $\mapsto$ even (via $3n+1$), even $\mapsto$ even or odd (via $n/2$)
3. [x] No conserved topological invariant
4. [x] Result: **No useful sector preservation**

**Certificate:**
* [x] $K_{\mathrm{TB}_\pi}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Identify conserved topology"}, \mathsf{missing}: K_{\text{topo}}^+ \text{ (topological invariant)}, \mathsf{code}: \text{NO\_INVARIANT}, \mathsf{trace}: \text{"Parity not preserved"})$ → **Go to Node 9**

---

#### Node 9: TameCheck ($\mathrm{TB}_O$)

**Question:** Is the singular set definable in an o-minimal structure?

**Step-by-step execution:**
1. [x] The map $T$ is piecewise linear, hence o-minimal definable
2. [x] Singular set (non-terminating orbits): Unknown, but if finite or co-finite, would be definable
3. [x] Result: **Conditionally tame** (map is definable; singular set status unknown)

**Certificate:**
* [x] $K_{\mathrm{TB}_O}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove singular set is definable"}, \mathsf{missing}: K_{\mathrm{Rec}_N}^+, \mathsf{code}: \text{DEPENDENT\_ON\_NODE\_2}, \mathsf{trace}: \text{Node 2})$ → **Go to Node 10**

---

### Level 4: Mixing & Complexity (Nodes 10-11)

#### Node 10: ErgoCheck ($\mathrm{TB}_\rho$)

**Question:** Does the flow exhibit dissipative/mixing behavior?

**Step-by-step execution:**
1. [x] Conjectured behavior: All orbits eventually enter cycle $\{1,2,4\}$
2. [x] If true: System is dissipative (not mixing in ergodic sense, but convergent)
3. [x] Result: **Conditionally dissipative**

**Certificate:**
* [x] $K_{\mathrm{TB}_\rho}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove convergence to cycle"}, \mathsf{missing}: K_{\mathrm{Rec}_N}^+, \mathsf{code}: \text{DEPENDENT\_ON\_NODE\_2}, \mathsf{trace}: \text{Node 2})$ → **Go to Node 11**

---

#### Node 11: ComplexCheck ($\mathrm{Rep}_K$)

**Question:** Is the description complexity bounded?

**Step-by-step execution:**
1. [x] Complexity: $K(n) = \lceil \log_2(n) \rceil$ (bit length)
2. [x] For terminating trajectories: Complexity reaches $K(1) = 1$
3. [x] Result: **Conditionally bounded**

**Certificate:**
* [x] $K_{\mathrm{Rep}_K}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove complexity convergence"}, \mathsf{missing}: K_{\mathrm{Rec}_N}^+, \mathsf{code}: \text{DEPENDENT\_ON\_NODE\_2}, \mathsf{trace}: \text{Node 2})$ → **Go to Node 12**

---

### Level 5: Gradient Structure (Node 12)

#### Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)

**Question:** Is there oscillatory behavior in the dynamics?

**Step-by-step execution:**
1. [x] No Lyapunov function known
2. [x] Trajectories can increase: $T(27) = 82$, $T(82) = 41$, oscillatory descent
3. [x] Result: **Yes, oscillatory** (not monotone descent)

**Certificate:**
* [x] $K_{\mathrm{GC}_\nabla}^+ = (\text{oscillatory}, \text{no Lyapunov})$ → **Go to Node 13**

---

### Level 6: Boundary (Node 13)

#### Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)

**Question:** Is the system open (external input/output coupling)?

**Step-by-step execution:**
1. [x] Domain: $\mathbb{N}$ (no boundary in topological sense)
2. [x] Closed system (no external input)
3. [x] Result: **Closed**

**Certificate:**
* [x] $K_{\mathrm{Bound}_\partial}^- = (\text{closed system})$ → **Go to Node 17**

---

### Level 7: The Lock (Node 17)

#### Node 17: LockCheck ($\mathrm{Cat}_{\mathrm{Hom}}$)

**Question:** Is $\text{Hom}(\mathcal{H}_{\text{bad}}, \mathcal{H}) = \emptyset$?

**Step-by-step execution:**
1. [x] Define $\mathcal{H}_{\text{bad}}$: Divergent orbit or non-trivial cycle
2. [x] Attempt Tactic E1 (Dimension): Not applicable (discrete, no embedding)
3. [x] Attempt Tactic E2 (Invariant Mismatch): No known invariant to use
4. [x] Attempt Tactic E3 (Positivity): Not applicable
5. [x] Attempt Tactic E4 (Integrality): Not applicable (already integer-valued)
6. [x] Attempt Tactics E5-E10: None applicable
7. [x] **No tactic succeeds**
8. [x] Check obligation ledger: $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ unresolved
9. [x] Apply {prf:ref}`def-lock-breached-inc`: Lock barrier cannot be blocked

**Certificate:**
* [x] $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}} = (\mathsf{type}: \text{HORIZON}, \mathsf{stratum}: \text{Node 2 (ZenoCheck)}, \mathsf{reason}: \text{"Cannot prove finiteness of discrete events"}, \mathsf{dependency}: K_{\mathrm{Rec}_N}^{\mathrm{inc}}, \mathsf{obstruction}: \text{"No uniform bound on stopping time; non-monotonic map; local induction fails"})$

**Lock Status:** **HORIZON** (NOT blocked, NOT breached, but inconclusive)

---

## Part II-B: Upgrade Pass

### Inc-to-Positive Upgrades

| Original | Upgraded To | Mechanism | Reference |
|----------|-------------|-----------|-----------|
| $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ | — | No upgrade available | Unresolved |
| $K_{D_E}^{\mathrm{inc}}$ | — | Dependent on Node 2 | Unresolved |
| $K_{C_\mu}^{\mathrm{inc}}$ | — | Dependent on Node 2 | Unresolved |
| $K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}$ | — | No scale structure | Unresolved |
| $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | — | Discrete topology | Unresolved |
| $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | — | No gradient structure | Unresolved |
| $K_{\mathrm{TB}_\pi}^{\mathrm{inc}}$ | — | No conserved invariant | Unresolved |
| $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ | — | Dependent on Node 2 | Unresolved |
| $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | — | Dependent on Node 2 | Unresolved |
| $K_{\mathrm{Rep}_K}^{\mathrm{inc}}$ | — | Dependent on Node 2 | Unresolved |

**Note:** No metatheorems discharge the Node 2 certificate. The Zeno horizon persists.

---

## Part III-A: Lyapunov Reconstruction

*Not applicable: No Lyapunov function is known for the Collatz map. The framework cannot reconstruct one from the available permits.*

---

## Part III-B: Metatheorem Extraction

### **1. Surgery Admissibility (MT 15.1)**
*Not applicable: No singularities or surgery events in discrete map.*

### **2. Structural Surgery (MT 16.1)**
*Not applicable: No surgery needed.*

### **3. The Lock (Node 17)**
* **Question:** $\text{Hom}(\text{Bad}, \mathcal{H}) = \emptyset$?
* **Bad Pattern:** Divergent orbit or non-trivial cycle
* **Available Tactics:** None succeed
  - E1 (Dimension): Discrete system, no embedding theory
  - E2 (Invariant): No global invariant known
  - E3-E10: Not applicable
* **Unresolved Obligation:** $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ from Node 2
* **Result:** **HORIZON** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$)

---

## Part III-C: Obligation Ledger

### Table 1: Introduced Obligations

| ID | Node | Certificate | Obligation | Missing | Status |
|----|------|-------------|------------|---------|--------|
| OBL-1 | 1 | $K_{D_E}^{\mathrm{inc}}$ | Prove all trajectories terminate | $K_{\text{term}}^+$ | **UNRESOLVED** |
| OBL-2 | 2 | $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ | Prove $\tau(n) < \infty$ for all $n$ | $K_{\text{bound}}^+$ (uniform stopping time) | **UNRESOLVED** |
| OBL-3 | 3 | $K_{C_\mu}^{\mathrm{inc}}$ | Prove convergence to cycle | $K_{\mathrm{Rec}_N}^+$ | **UNRESOLVED** (depends on OBL-2) |
| OBL-4 | 4 | $K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}$ | Assess criticality | $K_{\text{scale}}^+$ | **UNRESOLVED** (no scaling) |
| OBL-5 | 6 | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | Characterize singular set | $K_{\text{geom}}^+$ | **UNRESOLVED** (discrete) |
| OBL-6 | 7 | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | Prove Lyapunov decrease | $K_{\text{Lyap}}^+$ | **UNRESOLVED** (no Lyapunov) |
| OBL-7 | 8 | $K_{\mathrm{TB}_\pi}^{\mathrm{inc}}$ | Identify conserved topology | $K_{\text{topo}}^+$ | **UNRESOLVED** (no invariant) |
| OBL-8 | 9 | $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ | Prove singular set definable | $K_{\mathrm{Rec}_N}^+$ | **UNRESOLVED** (depends on OBL-2) |
| OBL-9 | 10 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | Prove convergence to cycle | $K_{\mathrm{Rec}_N}^+$ | **UNRESOLVED** (depends on OBL-2) |
| OBL-10 | 11 | $K_{\mathrm{Rep}_K}^{\mathrm{inc}}$ | Prove complexity convergence | $K_{\mathrm{Rec}_N}^+$ | **UNRESOLVED** (depends on OBL-2) |

### Table 2: Discharge Events

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| — | — | — | — |

**No discharges occurred.**

### Table 3: Remaining Obligations

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| OBL-2 | Prove $\tau(n) < \infty$ for all $n \in \mathbb{N}$ | **This is the Collatz conjecture itself.** No local method provides uniform bound. Map is non-monotonic. No inductive argument works. Global structure unknown. |
| OBL-1, OBL-3, OBL-8, OBL-9, OBL-10 | (Various) | All depend on OBL-2 |
| OBL-4, OBL-5, OBL-6, OBL-7 | (Various) | Structural limitations (discrete system, no scaling, no gradient structure) |

**Ledger Validation:** $\mathsf{Obl}(\Gamma_{\mathrm{final}}) \ne \varnothing$ — **HORIZON**

---

## Part IV: Final Certificate Chain

### Validity Checklist

1. [x] All required nodes executed with explicit certificates
2. [x] Primary barrier: Node 2 (ZenoCheck) returns **INC**
3. [ ] No path to positive certificates (all conditional)
4. [ ] Lock certificate obtained: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$ (**HORIZON**, not BLOCKED)
5. [ ] Unresolved obligations remain
6. [ ] No Lyapunov reconstruction possible
7. [x] No surgery protocol needed
8. [x] Result extraction completed: **HORIZON**

### Certificate Accumulation Trace

```
Node 1:  K_{D_E}^{inc} (energy bounded conditional on termination)
Node 2:  K_{Rec_N}^{inc} (CANNOT PROVE finite stopping time) ← HORIZON ORIGIN
Node 3:  K_{C_μ}^{inc} (depends on Node 2)
Node 4:  K_{SC_λ}^{inc} (no scaling structure)
Node 5:  K_{SC_∂c}^+ (trivial, no parameters)
Node 6:  K_{Cap_H}^{inc} (discrete topology)
Node 7:  K_{LS_σ}^{inc} (no Lyapunov)
Node 8:  K_{TB_π}^{inc} (no conserved topology)
Node 9:  K_{TB_O}^{inc} (depends on Node 2)
Node 10: K_{TB_ρ}^{inc} (depends on Node 2)
Node 11: K_{Rep_K}^{inc} (depends on Node 2)
Node 12: K_{GC_∇}^+ (oscillatory, no gradient structure)
Node 13: K_{Bound_∂}^- (closed system)
Node 17: K_{Cat_Hom}^{hor} (HORIZON - cannot exclude bad patterns)
```

### Final Certificate Set

$$\Gamma_{\mathrm{final}} = \{K_{D_E}^{\mathrm{inc}}, K_{\mathrm{Rec}_N}^{\mathrm{inc}}, K_{C_\mu}^{\mathrm{inc}}, K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}, K_{\mathrm{SC}_{\partial c}}^+, K_{\mathrm{Cap}_H}^{\mathrm{inc}}, K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}, K_{\mathrm{TB}_\pi}^{\mathrm{inc}}, K_{\mathrm{TB}_O}^{\mathrm{inc}}, K_{\mathrm{TB}_\rho}^{\mathrm{inc}}, K_{\mathrm{Rep}_K}^{\mathrm{inc}}, K_{\mathrm{GC}_\nabla}^+, K_{\mathrm{Bound}_\partial}^-, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}\}$$

### Conclusion

**HORIZON (FAMILY VIII - ZENO HORIZON AT NODE 2)**

The Hypostructure framework correctly identifies that the Collatz conjecture lies beyond its current capacity. The diagnostic reaches a **Zeno Horizon** at Node 2, unable to prove that discrete events (iterations of the map) are finite for all starting values.

**Key Finding:** The framework's honest admission of limitation. The system cannot prove what mathematics has not yet proven. The INC certificate at Node 2 propagates through the entire diagnostic, culminating in a HORIZON verdict at the Lock.

**Why This Is Correct:**
- No local induction works (map is non-monotonic)
- No uniform bound on stopping time $\tau(n)$
- Global structure of iteration graph is unknown
- The question "are events finite?" IS the Collatz conjecture

This is the framework working as designed: identifying the precise locus of limitation (Node 2: ZenoCheck) and propagating that limitation honestly to the final verdict.

---

## Formal Diagnostic Report

::::{prf:theorem} Collatz Diagnostic Result
:label: thm-collatz-diagnostic

**Phase 1: Instantiation**
Instantiate the discrete hypostructure with:
- State space $\mathcal{X} = \mathbb{N}$
- Dynamics: Collatz map $T(n)$
- Initial data: $n_0 \in \mathbb{N}$

**Phase 2: Sieve Execution**
Proceed through nodes 1-17, issuing certificates at each step.

**Phase 3: Horizon Identification**
Node 2 (ZenoCheck) returns:
$$K_{\mathrm{Rec}_N}^{\mathrm{inc}} = (\text{"Prove } \tau(n) < \infty", K_{\text{bound}}^+, \text{NON\_PAINLEVE}, \text{"No local bound on stopping time"})$$

This certificate cannot be discharged by any available metatheorem.

**Phase 4: Lock Evaluation**
The Lock receives unresolved obligations and no applicable exclusion tactic. Returns:
$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}: \quad \text{HORIZON at Node 2 (ZenoCheck)}$$

**Phase 5: Conclusion**
The Collatz conjecture is identified as a **Zeno Horizon** problem. The framework correctly places it in **Family VIII (Horizon)**, **Primary Stratum: Node 2**.

**Verdict:** HORIZON — The system lacks the capacity to prove (or disprove) the conjecture with current methods.

::::

---

## Verification Summary

| Component | Status | Certificate |
|-----------|--------|-------------|
| Energy Bound | Inconclusive | $K_{D_E}^{\mathrm{inc}}$ |
| Surgery Finiteness | **INCONCLUSIVE** | $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ ← **HORIZON SOURCE** |
| Compactness | Inconclusive | $K_{C_\mu}^{\mathrm{inc}}$ |
| Scaling Analysis | Inconclusive | $K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}$ |
| Parameter Stability | Positive | $K_{\mathrm{SC}_{\partial c}}^+$ |
| Singular Codimension | Inconclusive | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ |
| Stiffness Gap | Inconclusive | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ |
| Topology Preservation | Inconclusive | $K_{\mathrm{TB}_\pi}^{\mathrm{inc}}$ |
| Tameness | Inconclusive | $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ |
| Mixing/Dissipation | Inconclusive | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ |
| Complexity Bound | Inconclusive | $K_{\mathrm{Rep}_K}^{\mathrm{inc}}$ |
| Gradient Structure | Positive (oscillatory) | $K_{\mathrm{GC}_\nabla}^+$ |
| Boundary | Negative (closed) | $K_{\mathrm{Bound}_\partial}^-$ |
| Lock | **HORIZON** | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$ |
| Obligation Ledger | **NON-EMPTY** | 10 unresolved obligations |
| **Final Status** | **HORIZON (FAMILY VIII, NODE 2)** | — |

---

## The Horizon Interpretation

### What This Diagnostic Means

**The Collatz conjecture is a Zeno Horizon problem.** Specifically:

1. **Location:** Node 2 (ZenoCheck) — "Are discrete events finite?"
2. **Question:** Does every starting value $n \in \mathbb{N}$ reach 1 in finitely many steps?
3. **Obstruction:** No local method can bound $\tau(n)$ uniformly
4. **Nature:** The map can temporarily increase values ($27 \to 82$), preventing monotone arguments
5. **Status:** Open problem in mathematics since 1937

### Why This Is a High-Profile Example

The Collatz conjecture is perhaps the **most famous Zeno Horizon problem**:
- Simple to state
- Easy to verify computationally for individual cases
- Utterly resistant to theoretical proof
- The horizon appears **immediately** at Node 2 (not deep in the Sieve)

### Framework Behavior

The Hypostructure framework:
1. Correctly identifies the locus of difficulty (Node 2)
2. Issues mandatory INC certificate (cannot proceed without it)
3. Propagates this limitation through conditional certificates at downstream nodes
4. Honestly reports HORIZON at the Lock
5. **Makes no false claims about regularity**

This is the intended behavior: the framework does not pretend to solve open problems. It identifies where its methods fail and reports this failure precisely.

---

## References

- L. Collatz, "On the motivation and origin of the (3n+1)-problem", Journal of Qufu Normal University 12 (1986)
- J. C. Lagarias, "The 3x+1 problem and its generalizations", American Mathematical Monthly 92 (1985)
- T. Tao, "Almost all orbits of the Collatz map attain almost bounded values", arXiv:1909.03562 (2019)

---

## Appendix: Why Node 2 Cannot Be Resolved

### The Zeno Question

Node 2 asks: "Are discrete events finite?" For the Collatz map:
- Each iteration is an event
- Question becomes: Does $\tau(n) = \min\{k : T^k(n) = 1\}$ exist for all $n$?
- **This is exactly the Collatz conjecture**

### Why Local Methods Fail

1. **No Monotonicity:**
   - Odd step: $n \mapsto 3n+1$ (increase)
   - Even step: $n \mapsto n/2$ (decrease)
   - Net effect: unpredictable
   - Example: $27 \to 82 \to 41 \to 124 \to 62 \to 31 \to 94 \to \ldots$ (111 steps total)

2. **No Inductive Structure:**
   - Cannot prove: "If $\tau(n) < \infty$, then $\tau(T^{-1}(n)) < \infty$"
   - Preimages are not well-controlled
   - Backward dynamics are multi-valued and complex

3. **No Global Invariant:**
   - Energy $E(n) = \log_2(n)$ is not monotone
   - No Lyapunov function known
   - Parity switches between odd and even
   - No conserved quantity to track

4. **No Scaling Control:**
   - The map $T$ is not scale-invariant
   - Cannot deduce $\tau(\lambda n)$ from $\tau(n)$
   - Each starting value is a separate problem

### What Would Be Needed

To resolve Node 2 (and thus the Collatz conjecture), one would need:
- **Global bound:** $\tau(n) \le f(n)$ for some computable $f$
- **OR:** Invariant measure/structure proving almost-sure termination
- **OR:** Algebraic/number-theoretic insight revealing hidden structure
- **OR:** Proof of impossibility (finding a divergent orbit or non-trivial cycle)

None of these are currently available.

---

## Executive Summary: The Diagnostic Dashboard

### 1. System Instantiation (The Physics)

| Object | Definition | Role |
| :--- | :--- | :--- |
| **Arena ($\mathcal{X}$)** | $\mathbb{N}$ | Positive Integers |
| **Potential ($\Phi$)** | $E(n) = \log_2(n)$ | Height/Size Measure |
| **Cost ($\mathfrak{D}$)** | $\mathfrak{D} = 1$ | Iteration Count |
| **Invariance ($G$)** | Trivial | No Symmetry |

### 2. Execution Trace (The Logic)

| Node | Check | Outcome | Certificate Payload | Γ (Certificate Accumulation) |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Energy Bound | INC | Conditional on termination | $\{K_{D_E}^{\mathrm{inc}}\}$ |
| **2** | Zeno Check | **INC** | **Cannot prove $\tau(n) < \infty$** | $\Gamma_1 \cup \{K_{\mathrm{Rec}_N}^{\mathrm{inc}}\}$ **← HORIZON** |
| **3** | Compact Check | INC | Depends on Node 2 | $\Gamma_2 \cup \{K_{C_\mu}^{\mathrm{inc}}\}$ |
| **4** | Scale Check | INC | No scaling structure | $\Gamma_3 \cup \{K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}\}$ |
| **5** | Param Check | YES | No parameters | $\Gamma_4 \cup \{K_{\mathrm{SC}_{\partial c}}^+\}$ |
| **6** | Geom Check | INC | Discrete topology | $\Gamma_5 \cup \{K_{\mathrm{Cap}_H}^{\mathrm{inc}}\}$ |
| **7** | Stiffness Check | INC | No Lyapunov | $\Gamma_6 \cup \{K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}\}$ |
| **8** | Topo Check | INC | No invariant | $\Gamma_7 \cup \{K_{\mathrm{TB}_\pi}^{\mathrm{inc}}\}$ |
| **9** | Tame Check | INC | Depends on Node 2 | $\Gamma_8 \cup \{K_{\mathrm{TB}_O}^{\mathrm{inc}}\}$ |
| **10** | Ergo Check | INC | Depends on Node 2 | $\Gamma_9 \cup \{K_{\mathrm{TB}_\rho}^{\mathrm{inc}}\}$ |
| **11** | Complex Check | INC | Depends on Node 2 | $\Gamma_{10} \cup \{K_{\mathrm{Rep}_K}^{\mathrm{inc}}\}$ |
| **12** | Oscillate Check | YES | Oscillatory | $\Gamma_{11} \cup \{K_{\mathrm{GC}_\nabla}^+\}$ |
| **13** | Boundary Check | CLOSED | No boundary | $\Gamma_{12} \cup \{K_{\mathrm{Bound}_\partial}^-\}$ |
| **--** | **SURGERY** | **N/A** | — | $\Gamma_{13}$ |
| **17** | **LOCK** | **HORIZON** | Node 2 unresolved | $\Gamma_{13} \cup \{K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}\} = \Gamma_{\mathrm{final}}$ |

### 3. Lock Mechanism (The Exclusion Attempt)

| Tactic | Description | Status | Reason / Mechanism |
| :--- | :--- | :---: | :--- |
| **E1** | Dimension | FAIL | Discrete system; no embedding theory applicable |
| **E2** | Invariant | FAIL | No global invariant known |
| **E3** | Positivity | N/A | Not applicable |
| **E4** | Integrality | N/A | Already integer-valued |
| **E5** | Functional | FAIL | No Lyapunov function |
| **E6** | Causal | N/A | Not applicable |
| **E7** | Thermodynamic | N/A | No entropy structure |
| **E8** | Holographic | N/A | No capacity theory |
| **E9** | Ergodic | FAIL | Cannot prove convergence |
| **E10** | Definability | PARTIAL | Map is definable, but singular set unknown |

**Result:** No tactic excludes bad patterns. Unresolved Node 2 obligation persists.

### 4. Final Verdict

* **Status:** HORIZON
* **Family:** VIII (Horizon)
* **Primary Stratum:** Node 2 (ZenoCheck)
* **Obligation Ledger:** 10 unresolved obligations (primary: OBL-2)
* **Singularity Set:** Unknown (potentially empty, potentially containing divergent orbits)
* **Critical Obstruction:** Cannot prove finite stopping time for all $n \in \mathbb{N}$

---

## Document Information

| Field | Value |
|-------|-------|
| Document Type | Diagnostic Trace (Horizon Report) |
| Framework | Hypostructure v1.0 |
| Problem Class | Open Conjecture (Number Theory / Discrete Dynamics) |
| System Type | $T_{\text{discrete}}$ |
| Verification Level | Machine-checkable diagnostic |
| Inc Certificates | 10 introduced, 0 discharged |
| Final Status | **HORIZON (FAMILY VIII, NODE 2)** |
| Generated | 2025-12-23 |

---
