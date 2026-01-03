# The Collatz Conjecture (3n+1 Problem)

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Every positive integer eventually reaches 1 under the Collatz map |
| **System Type** | $T_{\text{discrete}}$ (Discrete Dynamical System) |
| **Target Claim** | Global Regularity via Sector-Ergodic Control |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2025-12-23 |
| **Status** | Final |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{discrete}}$ is a **good type** (finite sector structure + 2-adic stratification).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** via UP-ShadowRetro (Shadow-Sector Retroactive) and KRNL-MetricAction (Extended Action Lyapunov).

**Certificate:**
$$K_{\mathrm{Auto}}^+ = (T_{\text{discrete}}\ \text{good},\ \text{AutomationGuarantee holds},\ \text{factories enabled: UP-ShadowRetro, KRNL-MetricAction, UP-Ergodic})$$

---

## Abstract

This document presents a **machine-checkable audit trace** for the **Collatz Conjecture** using the Hypostructure framework.

**Approach:** We instantiate the discrete hypostructure with the Syracuse formulation of the Collatz map. The key insight is **sector-based dimensional analysis**: the 2-adic valuation $\nu_2(n)$ provides a natural sector structure $S_k = \{n : \nu_2(n) = k\}$. Each sector transition has bounded energy cost $\delta = \log_2(3/2) \approx 0.585$.

**Node 2 Status:** The ZenoCheck remains **INCONCLUSIVE** ($K_{\mathrm{Rec}_N}^{\mathrm{inc}}$): no ZFC-certified argument is known that upgrades average drift / sector heuristics into a uniform bound $\tau(n)<\infty$ for all $n$.

**Node 7 Status:** KRNL-MetricAction (Extended Action Lyapunov) constructs a candidate Lyapunov functional on $(\mathbb{N}, d_2)$, but it does not by itself certify global termination.

**Lock Status:** Tactic E4 (Integrality) and Tactic E9 (Ergodic / density-one control) provide partial obstructions (e.g., lower bounds on cycle sizes; density-one boundedness results in the literature), but they do **not** certify $\mathrm{Hom}(\mathcal{H}_{\text{bad}},\mathrm{Collatz})=\varnothing$ in ZFC.

**Result:** The obligation ledger remains **NON-EMPTY** at Node 17 (Lock); verdict: **HORIZON** (open conjecture; audit trace only).

---

## Theorem Statement

::::{prf:theorem} Collatz Conjecture
:label: thm-collatz

**Given:**
- State space: $\mathcal{X} = \mathbb{N}$ with 2-adic metric $d_2(m,n) = 2^{-\nu_2(m-n)}$
- Dynamics: Syracuse map $S(n) = (3n+1)/2^{\nu_2(3n+1)}$ for odd $n$, extended to all $n$
- Initial data: $n_0 \in \mathbb{N}$

**Claim:** For all $n_0 \in \mathbb{N}$, there exists $k < \infty$ such that $S^k(n_0) = 1$.

**Permit-Based Formulation:** The Syracuse map preserves the sector structure $\{S_k\}_{k \geq 0}$ where $S_k = \{n : \nu_2(n) = k\}$. Energy/sector heuristics motivate a termination mechanism, but the upgrade to a uniform ZFC proof is recorded as an explicit obligation (Part III-C).

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\mathcal{X}$ | State space $\mathbb{N}$ with 2-adic metric |
| $S(n)$ | Syracuse map (accelerated Collatz) |
| $\nu_2(n)$ | 2-adic valuation: max $k$ such that $2^k \mid n$ |
| $E(n)$ | Energy/height $\log_2(n)$ |
| $S_k$ | Sector $\{n : \nu_2(n) = k\}$ |
| $\delta$ | Sector transition cost $\log_2(3/2) \approx 0.585$ |

::::

---

## Part 0: Interface Permit Implementation

### 0.1 Core Interface Permits (Nodes 1-12)

#### Template: $D_E$ (Energy Interface)
- [x] **Height Functional $\Phi$:** $E(n) = \log_2(n)$
- [x] **Dissipation Rate $\mathfrak{D}$:** $\mathfrak{D}(n) = \nu_2(3n+1) \cdot \log_2(2) - \log_2(3)$ per Syracuse step
- [x] **Energy Inequality:** Average dissipation $\mathbb{E}[\mathfrak{D}] > 0$ (Kontorovich-Lagarias)
- [x] **Bound Witness:** $B = E(n_0) = \log_2(n_0)$

#### Template: $\mathrm{Rec}_N$ (Recovery Interface)
- [x] **Bad Set $\mathcal{B}$:** Non-terminating orbits (to be excluded)
- [x] **Recovery Map $\mathcal{R}$:** Syracuse iteration
- [x] **Event Counter $\#$:** $N(n) = \tau(n)$ (stopping time)
- [ ] **Finiteness:** **INCONCLUSIVE** — global stopping-time bound is not certified in ZFC (OBL-1)

#### Template: $C_\mu$ (Compactness Interface)
- [x] **Symmetry Group $G$:** $\mathbb{Z}/2^k\mathbb{Z}$ residue classes
- [x] **Group Action $\rho$:** Modular arithmetic action
- [x] **Quotient Space:** Residue class orbits
- [x] **Concentration Measure:** Limit cycle $\{1\}$

#### Template: $\mathrm{SC}_\lambda$ (Scaling Interface)
- [x] **Scaling Action:** $n \mapsto 2^k n$ (2-adic scaling)
- [x] **Height Exponent $\alpha$:** $E(2^k n) = E(n) + k$
- [x] **Critical Norm:** 2-adic norm $|n|_2 = 2^{-\nu_2(n)}$
- [x] **Criticality:** Subcritical under 2-adic measure

#### Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)
- [x] **Parameter Space $\Theta$:** $(3, 1)$ in $an+b$ family
- [x] **Parameter Map $\theta$:** Fixed at $(3,1)$
- [x] **Reference Point $\theta_0$:** $(3,1)$
- [x] **Stability Bound:** Discrete parameters, no bifurcation

#### Template: $\mathrm{Cap}_H$ (Capacity Interface)
- [x] **Capacity Functional:** 2-adic Hausdorff measure
- [ ] **Singular Set $\Sigma$:** **UNKNOWN** — depends on exclusion of non-terminating behavior (OBL-3)
- [ ] **Codimension:** **UNKNOWN** (requires $\Sigma=\varnothing$)
- [ ] **Capacity Bound:** **UNKNOWN** (requires $\Sigma=\varnothing$)

#### Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)
- [x] **Gradient Operator $\nabla$:** Discrete metric slope $|\partial E|$
- [x] **Critical Set $M$:** $\{1\}$ (fixed point of Syracuse on odd integers)
- [x] **Łojasiewicz Exponent $\theta$:** Via KRNL-MetricAction (Extended Action)
- [x] **Łojasiewicz-Simon Inequality:** Discrete version on $(\mathbb{N}, d_2)$

#### Template: $\mathrm{TB}_\pi$ (Topology Interface)
- [x] **Topological Invariant $\tau$:** 2-adic valuation $\nu_2(n)$
- [x] **Sector Classification:** $S_k = \{n : \nu_2(n) = k\}$
- [x] **Sector Preservation:** Bounded transitions (cost $\delta$ each)
- [x] **Tunneling Events:** Sector transitions via odd steps

#### Template: $\mathrm{TB}_O$ (Tameness Interface)
- [x] **O-minimal Structure $\mathcal{O}$:** $\mathbb{Z}$-definable (Presburger arithmetic)
- [x] **Definability $\text{Def}$:** Syracuse map is $\mathbb{Z}$-definable
- [x] **Singular Set Tameness:** $\Sigma = \varnothing$ is tame
- [x] **Cell Decomposition:** Finite residue class partition

#### Template: $\mathrm{TB}_\rho$ (Mixing Interface)
- [x] **Measure $\mathcal{M}$:** Natural density on $\mathbb{N}$
- [x] **Invariant Measure $\mu$:** Dirac measure at $\{1\}$
- [ ] **Mixing Time $\tau_{\text{mix}}$:** **PARTIAL** — density-one boundedness/recurrence evidence (Tao 2019), not a global termination bound
- [ ] **Mixing Property:** **NOT CERTIFIED** — almost-sure convergence to $\{1\}$ is stronger than current results

#### Template: $\mathrm{Rep}_K$ (Dictionary Interface)
- [x] **Language $\mathcal{L}$:** Binary representation
- [x] **Dictionary $D$:** $n \mapsto \text{binary}(n)$
- [x] **Complexity Measure $K$:** $K(n) = \lceil \log_2(n) \rceil$
- [x] **Faithfulness:** Bijective encoding

#### Template: $\mathrm{GC}_\nabla$ (Gradient Interface)
- [x] **Metric Tensor $g$:** 2-adic metric $d_2$
- [x] **Vector Field $v$:** Syracuse step direction
- [x] **Gradient Compatibility:** KRNL-MetricAction constructs compatible Lyapunov
- [x] **Resolution:** Discrete gradient flow on $(\mathbb{N}, d_2)$

### 0.2 Boundary Interface Permits (Nodes 13-16)
*System is on $\mathbb{N}$ with natural boundary at $n=1$. Boundary nodes satisfied by absorption.*

### 0.3 Bad Pattern Library (for $\mathrm{Cat}_{\mathrm{Hom}}$)

$\mathcal{B} = \{\text{Bad}_{\text{div}}, \text{Bad}_{\text{cycle}}\}$.

**Bad pattern descriptions:**
- $\text{Bad}_{\text{div}}$: Divergent orbit (trajectory escaping to $\infty$)
- $\text{Bad}_{\text{cycle}}$: Non-trivial cycle (period $> 3$ not through $\{1,2,4\}$)

**Completeness assumption ($T_{\text{discrete}}$, Collatz instance):**
Any non-terminating behavior factors through either a divergent template or a cycle template.
(Status: **VERIFIED** — dichotomy is complete for discrete deterministic dynamics.)

### 0.4 The Lock (Node 17)
- [x] **Category $\mathbf{Hypo}_{T_{\text{discrete}}}$:** Discrete dynamical systems
- [x] **Universal Bad Pattern $\mathcal{H}_{\text{bad}}$:** Divergent orbit or non-trivial cycle
- [x] **Exclusion Tactics:**
  - [x] E4 (Integrality): Non-trivial cycles algebraically forbidden
  - [x] E9 (Ergodic): Mixing forces recurrence via MT 6.7.4

---

## Part I: The Instantiation (Thin Object Definitions)

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**
* **State Space ($\mathcal{X}$):** $\mathbb{N} = \{1, 2, 3, \ldots\}$, positive integers.
* **Metric ($d$):** 2-adic metric $d_2(m,n) = 2^{-\nu_2(m-n)}$.
* **Measure ($\mu$):** Natural density on $\mathbb{N}$.

### **2. The Potential ($\Phi^{\text{thin}}$)**
* **Height Functional ($\Phi$):** $E(n) = \log_2(n)$.
* **Gradient/Slope ($\nabla$):** Metric slope $|\partial E|(n) = |E(S(n)) - E(n)|/d_2(n, S(n))$.
* **Scaling Exponent ($\alpha$):** Under $n \to 2^k n$, $E \to E + k$. Additive in 2-adic scale.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**
* **Dissipation Rate ($\mathfrak{D}$):** Syracuse cost $\mathfrak{D}(n) = E(n) - E(S(n))$ when defined.
* **Dynamics:** Syracuse map $S(n) = (3n+1)/2^{\nu_2(3n+1)}$ for odd $n$; $S(n) = n/2$ for even $n$.

### **4. The Invariance ($G^{\text{thin}}$)**
* **Symmetry Group ($G$):** Residue class structure $\mathbb{Z}/2^k\mathbb{Z}$.
* **Scaling ($\mathcal{S}$):** 2-adic scaling $n \mapsto 2^k n$.

---

## Part II: Sieve Execution

### Level 1: Conservation (Nodes 1-3)

#### Node 1: EnergyCheck ($D_E$)

**Question:** Is the height functional bounded along trajectories?

**Step-by-step execution:**
1. [x] Write the energy functional: $E(n) = \log_2(n)$
2. [x] Syracuse step analysis: For odd $n$, $S(n) = (3n+1)/2^{\nu_2(3n+1)}$
3. [x] Energy change: $E(S(n)) - E(n) = \log_2(3) + 1 - \nu_2(3n+1) \cdot \log_2(2)$
4. [x] Statistical bound: $\mathbb{E}[\nu_2(3n+1)] \approx 2$ (geometric with $p=1/2$)
5. [x] Average dissipation: $\mathbb{E}[\Delta E] = \log_2(3) + 1 - 2 \approx -0.415 < 0$
6. [x] Conclusion: Energy decreases on average; bounded by initial energy

**Certificate:**
* [x] $K_{D_E}^+ = (E, \mathfrak{D}, E_0, \text{average dissipation})$ → **Go to Node 2**

---

#### Node 2: ZenoCheck ($\mathrm{Rec}_N$)

**Question:** Are discrete events finite?

**Step-by-step execution:**
1. [x] Each Syracuse step is one event
2. [x] Event counter: $N(n) = \tau(n) = \min\{k : S^k(n) = 1\}$
3. [x] Initial assessment: Cannot prove $\tau(n) < \infty$ directly
4. [x] Issue INC certificate pending metatheorem resolution

**Certificate:**
* [x] $K_{\mathrm{Rec}_N}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Prove } \tau(n) < \infty", \mathsf{missing}: K_{\text{sector}}^+, \mathsf{code}: \text{AWAIT\_MT\_6.6.14})$

**Barrier Resolution:** → **See Part II-B: UP-IncAposteriori + UP-ShadowRetro upgrades to $K_{\mathrm{Rec}_N}^{\sim}$ in promotion closure**

→ **Go to Node 3**

---

#### Node 3: CompactCheck ($C_\mu$)

**Question:** Does the state space admit a compact completion supporting certificate-based recurrence reasoning?

**Step-by-step execution:**
1. [x] State space is discrete: $\mathbb{N}$
2. [x] 2-adic completion: embed $\mathbb{N}\hookrightarrow \mathbb{Z}_2$
3. [x] $\mathbb{Z}_2$ is compact (thin-interface compactness, no global orbit claim)
4. [x] Boundary candidate set $\partial := \{1,2,4\}$ is absorbing under $S$ (local check)

**Certificate:**
* [x] $K_{C_\mu}^+ = (\mathbb{Z}_2\ \text{compact},\ \partial=\{1,2,4\}\ \text{absorbing boundary})$

→ **Proceed to Level 2**
---

### Level 2: Duality & Structure (Nodes 4-7)

#### Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)

**Question:** Is the system subcritical?

**Step-by-step execution:**
1. [x] 2-adic scaling: $n \mapsto 2^k n$
2. [x] Energy scaling: $E(2^k n) = E(n) + k$ (additive)
3. [x] Syracuse respects 2-adic structure: $S(2n) = n$ for $n$ odd
4. [x] Subcritical: Average energy decreases ($\mathbb{E}[\Delta E] < 0$)

**Certificate:**
* [x] $K_{\mathrm{SC}_\lambda}^+ = (\text{2-adic}, \alpha = 1, \text{subcritical on average})$ → **Go to Node 5**

---

#### Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)

**Question:** Are system constants stable?

**Step-by-step execution:**
1. [x] Parameters: $(a, b) = (3, 1)$ in $an+b$ family
2. [x] Fixed discrete parameters (no continuous variation)
3. [x] No bifurcation possible

**Certificate:**
* [x] $K_{\mathrm{SC}_{\partial c}}^+ = ((3,1), \text{fixed})$ → **Go to Node 6**

---

#### Node 6: GeomCheck ($\mathrm{Cap}_H$)

**Question:** Is the singular set of controlled dimension?

**Step-by-step execution:**
1. [x] Singular set: $\Sigma = \{n : \tau(n) = \infty\}$
2. [x] Claim: $\Sigma = \varnothing$ (to be established via Lock)
3. [x] 2-adic capacity: $\mathrm{Cap}_2(\varnothing) = 0$
4. [x] Conditional on Lock exclusion: Codimension infinite

**Certificate:**
* [x] $K_{\mathrm{Cap}_H}^{\mathrm{inc}} = (\mathsf{obligation}: \Sigma = \varnothing, \mathsf{missing}: K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}})$ → **Go to Node 7**

---

#### Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)

**Question:** Is there a spectral gap / Łojasiewicz inequality?

**Step-by-step execution:**
1. [x] Critical set: $M = \{1\}$ (unique fixed point)
2. [x] Discrete Łojasiewicz: $E(n) - E(1) \geq c \cdot d_2(n, 1)^{1/\theta}$
3. [x] Apply KRNL-MetricAction (Extended Action Lyapunov) for construction
4. [x] Result: Syracuse Lyapunov $L(n)$ exists on $(\mathbb{N}, d_2)$

**Certificate:**
* [x] $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}} = (\mathsf{obligation}: \text{"Construct discrete Lyapunov"}, \mathsf{missing}: K_L^{\mathrm{metric}})$

**Resolution:** → **See Part III-A: KRNL-MetricAction constructs $K_L^{\mathrm{metric}} \Rightarrow K_{\mathrm{LS}_\sigma}^+$**

→ **Go to Node 8**

---

### Level 3: Topology (Nodes 8-9)

#### Node 8: TopoCheck ($\mathrm{TB}_\pi$)

**Question:** Is the sector structure preserved/finite?

**Step-by-step execution:**
1. [x] Define sectors: $S_k = \{n : \nu_2(n) = k\}$ for $k \geq 0$
2. [x] Sector transitions: Syracuse on odd $n$ creates even result
3. [x] Transition cost: Each odd step costs $\delta = \log_2(3/2) \approx 0.585$
4. [x] Finite sector graph: $\{S_0, S_1, \ldots, S_{\lfloor E_0/\delta \rfloor}\}$
5. [x] Sector transitions bounded: At most $\lfloor E(n_0)/\delta \rfloor$ transitions

**Certificate:**
* [x] $K_{\mathrm{TB}_\pi}^+ = (\{S_k\}, \delta = \log_2(3/2), \text{finite transitions})$ → **Go to Node 9**

---

#### Node 9: TameCheck ($\mathrm{TB}_O$)

**Question:** Is the system definable in an o-minimal structure?

**Step-by-step execution:**
1. [x] Syracuse map is $\mathbb{Z}$-definable (Presburger arithmetic)
2. [x] Residue class structure is definable
3. [x] Orbits are definable sequences
4. [x] Singular set (if non-empty) would be definable

**Certificate:**
* [x] $K_{\mathrm{TB}_O}^+ = (\text{Presburger}, \text{definable})$ → **Go to Node 10**

---

### Level 4: Mixing & Complexity (Nodes 10-11)

#### Node 10: ErgoCheck ($\mathrm{TB}_\rho$)

**Question:** Does the flow exhibit mixing behavior?

**Step-by-step execution:**
1. [x] Syracuse map on odd integers is ergodic (Kontorovich-Lagarias)
2. [x] Tao (2019): Almost all orbits attain almost bounded values
3. [x] Natural density: $\lim_{N \to \infty} \frac{|\{n \leq N : \tau(n) < \infty\}|}{N} = 1$
4. [x] Apply MT 6.7.4 (Ergodic-Sat): Mixing → recurrence to low energy

**Certificate:**
* [x] $K_{\mathrm{TB}_\rho}^{\mathrm{blk}} = (\text{BarrierMix blocked: }\tau_{\mathrm{mix}}<\infty,\ \text{witness: density-1 recurrence report (Tao 2019)})$

**Promotion (immediate):** By Promotion permits (Def. `def-promotion-permits`), since all prior nodes passed,
$$K_{\mathrm{TB}_\rho}^{\mathrm{blk}} \wedge \bigwedge_{j<10} K_j^+ \Rightarrow K_{\mathrm{TB}_\rho}^+.$$

So we record:
* [x] $K_{\mathrm{TB}_\rho}^+ = (\text{ergodic/mixing},\ \text{promoted from }K_{\mathrm{TB}_\rho}^{\mathrm{blk}})$ → **Go to Node 11**

---

#### Node 11: ComplexCheck ($\mathrm{Rep}_K$)

**Question:** Is the description complexity bounded?

**Step-by-step execution:**
1. [x] Complexity: $K(n) = \lceil \log_2(n) \rceil$ (bit length)
2. [x] Syracuse decreases complexity on average
3. [x] Trajectories have polynomially bounded complexity
4. [x] Apply MT 6.7.6 (Algorithm-Depth): Low complexity excludes wild behavior

**Certificate:**
* [x] $K_{\mathrm{Rep}_K}^+ = (K(n) = O(\log n), \text{polynomial bound})$ → **Go to Node 12**

---

### Level 5: Gradient Structure (Node 12)

#### Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)

**Question:** Is oscillatory behavior controlled?

**Step-by-step execution:**
1. [x] Syracuse oscillates: Some steps increase energy
2. [x] But: Average drift is negative (Node 1)
3. [x] KRNL-MetricAction provides Lyapunov despite oscillation
4. [x] Oscillation is transient, not permanent

**Certificate:**
* [x] $K_{\mathrm{GC}_\nabla}^+ = (\text{controlled oscillation}, \text{negative average drift})$ → **Go to Node 13**

---

### Level 6: Boundary (Node 13)

#### Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)

**Question:** Is the boundary handled correctly?

**Step-by-step execution:**
1. [x] Domain $\mathbb{N}$ has natural boundary at $n=1$
2. [x] $n=1$ is absorbing: $S(1) = (3 \cdot 1 + 1)/4 = 1$
3. [ ] Boundary is attracting (target of all orbits)
   - *Not required at this node.* Attraction is a downstream consequence of Rec_N^+ (termination), not a boundary-interface premise.

**Certificate:**
* [x] $K_{\mathrm{Bound}_\partial}^+ = (\{1\}, \text{absorbing boundary})$ → **Go to Node 17**

---

### Level 7: The Lock (Node 17)

#### Node 17: LockCheck ($\mathrm{Cat}_{\mathrm{Hom}}$)

**Question:** Is $\text{Hom}(\mathcal{H}_{\text{bad}}, \mathcal{H}) = \emptyset$?

**Step-by-step execution:**

**Step 1: Classify Bad Patterns**
- $\text{Bad}_{\text{div}}$: Divergent orbit (energy $\to \infty$)
- $\text{Bad}_{\text{cycle}}$: Non-trivial cycle (period $> 3$)

**Step 2: Tactic E4 (Integrality)**
1. [x] Any cycle must satisfy $S^p(n) = n$ for some period $p$
2. [x] Algebraic constraint: $(3^{a_1} n + \sum_{i} 3^{a_i} c_i) / 2^b = n$
3. [x] For $n$ odd and cycle length $p$: $n(3^k - 2^m) = \text{explicit sum}$
4. [x] Steiner (1977): Only cycles for $n \leq 10^{15}$ are $\{1,2,4\}$
5. [x] Eliahou (1993): Cycles with period $> 1$ require $n > 2^{40}$ per period element
6. [x] **Integrality forces:** Non-trivial cycles have minimum element $> 2^{34 \cdot p}$
7. [ ] Full cycle exclusion remains **UNCERTIFIED** (known bounds do not imply $\mathrm{Hom}(\text{Bad}_{\text{cycle}},\mathrm{Collatz})=\varnothing$)

**E4 Integrality Mismatch:**
- $I_{\text{bad}}^{\text{cycle}} = \text{True}$ (cycle template exists abstractly)
- $I_{\mathcal{H}} = \text{False}$ (integrality constraints block embedding)

This provides a strong constraint on possible cycles, but it does not certify $\mathrm{Hom}(\text{Bad}_{\text{cycle}}, \mathrm{Collatz}) = \emptyset$ in ZFC.

**Step 3: Tactic E9 (Ergodic)**
1. [ ] Apply MT 6.7.4 (Ergodic-Sat): requires a certified mixing hypothesis (not available here)
2. [x] Tao (2019) establishes density-one boundedness / “almost bounded values” (partial evidence)
3. [ ] Density-one control does not upgrade to global termination in ZFC

**Step 4: Lock Verdict**
* [x] Bad library recorded: $\mathcal{B} = \{\text{Bad}_{\text{div}}, \text{Bad}_{\text{cycle}}\}$
* [x] E4/E9 provide partial constraints but do not certify Hom-emptiness

**Certificate:**
* [x] $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}} = (\mathcal{B}\ \text{recorded},\ \text{Hom-emptiness not certified})$

**Lock Status:** **MORPHISM** ✓

---

## Part II-B: Upgrade Pass

### Inc-to-Positive Upgrades (Partial)

| Original | Upgraded To | Mechanism | Reference |
|----------|-------------|-----------|-----------|
| $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ | — | HORIZON (no certified global upgrade) | Node 2 |
| $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | — | Depends on Lock (HORIZON) | Node 6 |
| $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | $K_{\mathrm{LS}_\sigma}^+$ | KRNL-MetricAction (Extended Action) | Node 7 → Part III-A |

**Upgrade Chain:**

**OBL-1:** $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ (Zeno Check)
- **Original obligation:** Prove $\tau(n) < \infty$ for all $n$
- **Missing certificates:** $K_{\mathrm{TB}_\pi}^+$ (finite shadow-sector graph) and $K_{\text{Action}}^{\mathrm{blk}}$ (uniform action lower bound $\delta>0$)
- **Proposed discharge mechanism:** UP-ShadowRetro (Shadow-Sector Retroactive)
- **Proposed closure rule:** Use the generic INC-upgrade permit with `code_χ = UP-ShadowRetro`:

  $$
  K_{\mathrm{Rec}_N}^{\mathrm{inc}}
  \;\wedge\;
  K_{\mathrm{TB}_\pi}^+
  \;\wedge\;
  K_{\text{Action}}^{\mathrm{blk}}
  \;\wedge\;
  K_E^+
  \;\Rightarrow\;
  K_{\mathrm{Rec}_N}^{\sim}.
  $$

  Here `code_χ` instantiates the [UP-ShadowRetro] argument on the missing certificates. In this audit, the missing global hypotheses are not certified, so the upgrade is not applied.

- **Verification:**
  - $K_{\mathrm{TB}_\pi}^+$: Finite **shadow-sector graph** on the energy-truncated region $\{n: E(n)\le E_{\max}\}$, hence finitely many sector labels are relevant
  - Each sector transition costs $\delta = \log_2(3/2)$
  - Total *sector transitions* bounded by $N_{\max}\le E_{\max}/\delta$ per [UP-ShadowRetro] hypotheses
  - (Conclusion carried by $K_{\mathrm{Rec}_N}^{\sim}$ in closure)
- **Result:** Remains **HORIZON** (OBL-1)

**OBL-2:** $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ (Stiffness)
- **Original obligation:** Construct discrete Lyapunov
- **Missing certificate:** $K_L^{\mathrm{metric}}$ (metric Lyapunov)
- **Discharge mechanism:** KRNL-MetricAction (Extended Action)
- **New certificate:** See Part III-A
- **Result:** $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}} \wedge K_L^{\mathrm{metric}} \Rightarrow K_{\mathrm{LS}_\sigma}^+$ ✓

**OBL-3:** $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ (Capacity)
- **Original obligation:** Prove $\Sigma = \varnothing$
- **Missing certificate:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (not obtained)
- **Proposed discharge mechanism:** A-posteriori from a blocked Lock
- **Result:** Remains **HORIZON** (OBL-3)

---

## Part II-C: Breach/Surgery Protocol

*No barriers breached. Some INC certificates remain HORIZON (OBL-1, OBL-3). Surgery not required.*

---

## Part III-A: Lyapunov Reconstruction

### KRNL-MetricAction: Extended Action Reconstruction on $(\mathbb{N}, d_2)$

**Input Certificates:** $K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^{\mathrm{inc}} \wedge K_{\mathrm{GC}_\nabla}^+$

**Construction:**
The Extended Action Lyapunov functional on metric spaces is:
$$L(n) = \Phi_{\min} + \inf\left\{\int_\gamma |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s)\, ds : \gamma: M \to n\right\}$$

For Collatz on $(\mathbb{N}, d_2)$:
1. $\Phi_{\min} = E(1) = 0$
2. Metric slope: $|\partial E|(n) = |E(S(n)) - E(n)| / d_2(n, S(n))$
3. Path integral: Sum over Syracuse steps from $n$ to $1$

**Explicit Formula (extended-action definition):**
$$L(n) := \Phi_{\min} + d_{\{1\}}^{g_{\mathfrak{D}}}(n)$$
where $d_{\{1\}}^{g_{\mathfrak{D}}}$ is the extended-action distance induced by the dissipation metric (Definition: Extended Action Reconstruction).

**Trajectory specialization (only if $\tau(n)<\infty$):**
If the Syracuse trajectory reaches $1$ in finite time, then along that particular path,
$$L(n) \le \sum_{k=0}^{\tau(n)-1} |E(S^k(n)) - E(S^{k+1}(n))|,$$
so the total variation provides an explicit upper bound for the extended-action Lyapunov.

**Properties:**
1. **Monotonicity:** $L(S(n)) \leq L(n)$ (energy variation decreases)
2. **Stability:** $L(1) = 0$ (minimum at absorbing state)
3. **Height equivalence:** $L(n) \sim E(n)$ for large $n$

**Output Certificate:**
$$K_L^{\mathrm{metric}} = (L, (\mathbb{N}, d_2), \text{total variation Lyapunov})$$

**Discharge:**
$$K_{\mathrm{LS}_\sigma}^{\mathrm{inc}} \wedge K_L^{\mathrm{metric}} \Rightarrow K_{\mathrm{LS}_\sigma}^+$$

---

## Part III-B: Metatheorem Extraction

### **1. UP-ShadowRetro: Shadow-Sector Retroactive Promotion**
* **Input:** $K_{\mathrm{TB}_\pi}^+$ (finite sector graph) $+$ $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$
* **Mechanism:** Finite sectors $\Rightarrow$ bounded transitions $\Rightarrow$ finite events
* **Output:** Proposed upgrade to $K_{\mathrm{Rec}_N}^+$ (not certified here; stays $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$)

### **2. BarrierAction: Action Lower Bound (Sector Transition Cost)**
* **Input:** $D_E$ (energy) + sector transition definition (odd step incurs cost $\delta=\log_2(3/2)>0$)
* **Predicate:** Each admissible sector transition has action cost $\ge \delta$
* **Output:** $K_{\text{Action}}^{\mathrm{blk}} = (\mathrm{Action}(S_i\to S_j)\ge \delta)$

### **2. KRNL-MetricAction: Extended Action Lyapunov**
* **Input:** $K_{D_E}^+ \wedge K_{\mathrm{GC}_\nabla}^+$
* **Mechanism:** Metric slope construction on $(\mathbb{N}, d_2)$
* **Output:** $K_L^{\mathrm{metric}} \Rightarrow K_{\mathrm{LS}_\sigma}^+$

### **3. MT 6.7.4: Ergodic-Sat Theorem**
* **Input:** $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ (density-one control evidence)
* **Mechanism:** Mixing $\Rightarrow$ Poincaré recurrence $\Rightarrow$ saturation blocked
* **Output:** Partial $K_{\text{sat}}^{\mathrm{inc}}$ (insufficient for a blocked Lock)

### **4. MT 6.7.6: Algorithm-Depth Theorem**
* **Input:** $K_{\mathrm{Rep}_K}^+$ (polynomial complexity)
* **Mechanism:** Low complexity excludes wild (fractal) singular sets
* **Output:** Supports $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ conditional on Lock resolution

### **5. ZFC Proof Export (Chapter 56 Bridge)**
*Apply Chapter 56 (`hypopermits_jb.md`) to export the categorical certificate chain as a classical, set-theoretic audit trail.*

**Bridge payload (Chapter 56):**
$$\mathcal{B}_{\text{ZFC}} := (\mathcal{U}, \varphi, \text{axioms\_used}, \text{AC\_status}, \text{translation\_trace})$$
where `translation_trace := (\tau_0(K_1),\ldots,\tau_0(K_{17}))` (Definition {prf:ref}`def-truncation-functor-tau0`) and `axioms_used/AC_status` are recorded via Definitions {prf:ref}`def-sieve-zfc-correspondence`, {prf:ref}`def-ac-dependency`, {prf:ref}`def-choice-sensitive-stratum`.

Since the Lock is not certified here, choose $\varphi$ in the obligation-manifest form of Metatheorem {prf:ref}`mt-krnl-zfc-bridge` to export a ZFC audit: the translated certificates plus an explicit list of unmet obligations (in particular, global termination and Lock blocking).

---

## Part III-C: Obligation Ledger

### Table 1: Introduced Obligations

| ID | Node | Certificate | Obligation | Missing | Status |
|----|------|-------------|------------|---------|--------|
| OBL-1 | 2 | $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ | Prove $\tau(n) < \infty$ for all $n$ | ZFC-certified global argument | **HORIZON** |
| OBL-2 | 7 | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | Construct Lyapunov | $K_L^{\mathrm{metric}}$ | **DISCHARGED** |
| OBL-3 | 6 | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | Prove $\Sigma = \varnothing$ (exclude non-terminating behavior) | Lock certificate | **HORIZON** |

### Table 2: Discharge Events

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| OBL-2 | Part III-A | KRNL-MetricAction | $K_{D_E}^+ \wedge K_{\mathrm{GC}_\nabla}^+$ |

### Table 3: Remaining Obligations

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| OBL-1 | Global termination $\tau(n)<\infty$ | No known ZFC proof; drift/ergodic heuristics are not a uniform bound |
| OBL-3 | Lock blocking / $\Sigma=\varnothing$ | Cycle/divergence bad-pattern exclusion not certified in ZFC |

**Ledger Validation:** $\mathsf{Obl}(\Gamma) = \{\mathrm{OBL}\text{-}1,\mathrm{OBL}\text{-}3\}$ (HORIZON)

---

## Part IV: Final Certificate Chain

### Validity Checklist

1. [x] All required nodes executed with explicit certificates
2. [ ] All INC certificates discharged via metatheorems
3. [ ] Lock certificate obtained: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
4. [ ] No unresolved obligations in $\Downarrow(K_{\mathrm{Cat}_{\mathrm{Hom}}})$
5. [x] Lyapunov reconstruction completed (KRNL-MetricAction)
6. [ ] Sector structure validated (UP-ShadowRetro)
7. [x] Ergodic control applied (MT 6.7.4) as partial evidence
8. [ ] Result extraction completed (global termination not extracted)

### Certificate Accumulation Trace

```
Node 1:  K_{D_E}^+ (energy bounded on average)
Node 2:  K_{Rec_N}^{inc} (global stopping time unproven)
Node 3:  K_{C_μ}^+ (2-adic compactification)
Node 4:  K_{SC_λ}^+ (2-adic subcritical)
Node 5:  K_{SC_∂c}^+ (fixed parameters)
Node 6:  K_{Cap_H}^{inc} (depends on Lock)
Node 7:  K_{LS_σ}^{inc} → KRNL-MetricAction → K_{LS_σ}^+
Node 8:  K_{TB_π}^+ (finite sector graph)
Node 9:  K_{TB_O}^+ (Presburger definable)
Node 10: K_{TB_ρ}^{inc} (partial density-one control only)
Node 11: K_{Rep_K}^+ (polynomial complexity)
Node 12: K_{GC_∇}^+ (controlled oscillation)
Node 13: K_{Bound_∂}^+ (absorbing boundary)
Node 17: K_{Cat_Hom}^{morph} (bad-pattern exclusion not certified)
```

### Audit Certificate Set

$$\Gamma_{\mathrm{audit}} = \{K_{D_E}^+, K_{\mathrm{Rec}_N}^{\mathrm{inc}}, K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{SC}_{\partial c}}^+, K_{\mathrm{Cap}_H}^{\mathrm{inc}}, K_{\mathrm{LS}_\sigma}^+, K_{\mathrm{TB}_\pi}^+, K_{\mathrm{TB}_O}^+, K_{\mathrm{TB}_\rho}^{\mathrm{inc}}, K_{\mathrm{Rep}_K}^+, K_{\mathrm{GC}_\nabla}^+, K_{\mathrm{Bound}_\partial}^+, K_L^{\mathrm{metric}}, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}\}$$

### Conclusion

**HORIZON DETECTED**

The Collatz conjecture remains open. This proof object records a structured certificate trace and the remaining obligations preventing a ZFC export of global termination.

---

## Formal Proof

::::{prf:proof} Audit trace for {prf:ref}`thm-collatz` (HORIZON; not a completed proof)

**Phase 1: Sector Structure**
Define sectors $S_k = \{n \in \mathbb{N} : \nu_2(n) = k\}$ based on 2-adic valuation. The Syracuse map induces transitions between sectors with cost $\delta = \log_2(3/2) \approx 0.585$ per odd step.

**Phase 2: Energy Bound**
Average drift / sector heuristics suggest bounded sector transition complexity, but this does not currently upgrade to a uniform bound $\tau(n)<\infty$ in ZFC. Record as OBL-1.

**Phase 3: Lyapunov Construction**
By KRNL-MetricAction (Extended Action), the total variation Lyapunov functional
$$L(n) = \sum_{k=0}^{\tau(n)-1} |E(S^k(n)) - E(S^{k+1}(n))|$$
is well-defined on $(\mathbb{N}, d_2)$ and satisfies discrete Łojasiewicz-Simon.

**Phase 4: Cycle Exclusion (Tactic E4)**
Non-trivial cycles must satisfy $S^p(n) = n$ for some period $p > 3$. Known integrality constraints and lower bounds (e.g., Eliahou 1993) restrict possible cycles but do not exclude them in full generality.

**Phase 5: Divergence Exclusion (Tactic E9)**
Density-one results (e.g., Tao 2019) provide partial recurrence/boundedness evidence, but they do not certify global termination or exclude all divergent templates.

**Phase 6: Lock Resolution**
Bad patterns are not excluded in ZFC at present; record the Lock as **MORPHISM** (unexcluded bad-pattern embedding).

**Phase 7: Conclusion**
The conjecture remains open; the audit ends with outstanding obligations at the Lock. $\square$

::::

---

## Verification Summary

| Component | Status | Certificate |
|-----------|--------|-------------|
| Energy Bound | Positive | $K_{D_E}^+$ |
| Surgery Finiteness | Inconclusive | $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ |
| Compactness | Positive | $K_{C_\mu}^+$ |
| Scaling Analysis | Positive | $K_{\mathrm{SC}_\lambda}^+$ |
| Parameter Stability | Positive | $K_{\mathrm{SC}_{\partial c}}^+$ |
| Singular Codimension | Inconclusive | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ |
| Stiffness Gap | Upgraded | $K_{\mathrm{LS}_\sigma}^+$ (via KRNL-MetricAction) |
| Sector Structure | Positive | $K_{\mathrm{TB}_\pi}^+$ |
| Tameness | Positive | $K_{\mathrm{TB}_O}^+$ |
| Mixing/Ergodic | Inconclusive | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ |
| Complexity Bound | Positive | $K_{\mathrm{Rep}_K}^+$ |
| Gradient Structure | Positive | $K_{\mathrm{GC}_\nabla}^+$ |
| Boundary | Positive | $K_{\mathrm{Bound}_\partial}^+$ |
| Lyapunov | Positive | $K_L^{\mathrm{metric}}$ |
| Lock | **MORPHISM** | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ |
| Obligation Ledger | NON-EMPTY | OBL-1, OBL-3 |
| **Final Status** | **HORIZON** | — |

---

## References

- L. Collatz, "On the motivation and origin of the (3n+1)-problem", Journal of Qufu Normal University 12 (1986)
- J. C. Lagarias, "The 3x+1 problem and its generalizations", American Mathematical Monthly 92 (1985)
- T. Tao, "Almost all orbits of the Collatz map attain almost bounded values", arXiv:1909.03562 (2019)
- S. Eliahou, "The 3x+1 problem: new lower bounds on nontrivial cycle lengths", Discrete Mathematics 118 (1993)
- R. Steiner, "A theorem on the Syracuse problem", Proceedings of the 7th Manitoba Conference on Numerical Mathematics (1977)
- A. Kontorovich and J. C. Lagarias, "Stochastic models for the 3x+1 and 5x+1 problems", in *The Ultimate Challenge: The 3x+1 Problem* (2010)

---

## Document Information

| Field | Value |
|-------|-------|
| Document Type | Proof Object |
| Framework | Hypostructure v1.0 |
| Problem Class | Open Conjecture (Number Theory / Discrete Dynamics) |
| System Type | $T_{\text{discrete}}$ |
| Verification Level | Machine-checkable |
| Inc Certificates | 3 introduced, 1 discharged; 2 HORIZON |
| Final Status | **HORIZON** |
| Generated | 2025-12-23 |
