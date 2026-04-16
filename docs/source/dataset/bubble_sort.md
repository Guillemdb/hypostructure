# Structural Sieve Proof: Bubble Sort Termination and Complexity

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Termination and quadratic worst-case complexity of bubble sort on finite arrays |
| **System Type** | $T_{\text{algorithmic}}$ (finite discrete algorithmic dynamics) |
| **Target Claim** | Bubble sort terminates on every input permutation and uses at most $\binom{n}{2}$ swaps |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2026-04-15 |

### Label Naming Conventions

This instance uses the slug `bubble-sort`.

| Type | Pattern | Example |
|------|---------|---------|
| Definitions | `def-bubble-sort-*` | `def-bubble-sort-arena` |
| Theorems | `thm-bubble-sort-*` | `thm-bubble-sort-main` |
| Lemmas | `lem-bubble-sort-*` | `lem-bubble-sort-inversion-drop` |
| Remarks | `rem-bubble-sort-*` | `rem-bubble-sort-fixed-point` |
| Proofs | `proof-bubble-sort-*` | `proof-thm-bubble-sort-main` |
| Proof Sketches | `sketch-bubble-sort-*` | `sketch-thm-bubble-sort-main` |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{algorithmic}}$ is a good type (finite state stratification with constructive transition rules).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** (Definition {prf:ref}`def-automation-guarantee`), hence profile extraction, admissibility, and surgery factories are available.
- **Scope note:** The automation witness discharges the factory layer only. The Lock certificate, finite-descent package, and final complexity backend certificate are certified explicitly below.

**Certificate:**
$$
K_{\mathrm{Auto}}^+
=
\bigl(
T_{\text{algorithmic}}\ \text{good},
\ \text{AutomationGuarantee holds},
\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, RESOLVE-AutoSurgery}
\bigr).
$$

---

## Local-to-Global Certificate Discipline

The hypostructure proof is certified from local inputs only:

$$
\Gamma_{\mathrm{local}}
=
\{\text{thin-object data},\ \text{interface permits},\ \text{node certificates},\ \text{local backend permits},\ \text{Lock/local completeness packages if used}\}.
$$

This closes locally to the goal certificate and then extracts the global theorem output:

$$
\Gamma_{\mathrm{local}}
\xRightarrow{\text{named promotion / extraction}}
K_{\mathrm{Goal}}^+
\xRightarrow{\text{Part III-B / Part IV}}
\text{global theorem output}.
$$

**Never reverse this arrow.** Global regularity, global exclusion, convergence, or classification statements must never be used as local premises, upgrade inputs, or Lock hypotheses.

## Abstract

This document presents a **machine-checkable proof object** for **termination and worst-case complexity of bubble sort** using the Hypostructure framework.

**Approach:** We instantiate the algorithmic hypostructure on the finite permutation state space
$$
S_n
=
\{\sigma:\{1,\dots,n\}\to\{1,\dots,n\}\ \text{bijective}\}
$$
with the standard adjacent-swap bubble-sort transition rule. The primary height is the inversion count
$$
\Phi(\sigma)
=
\#\{(i,j):1\le i<j\le n,\ \sigma(i)>\sigma(j)\},
$$
and the route-critical control is the exact discrete descent law
$$
\Phi(\sigma')=\Phi(\sigma)-1
$$
for every executed swap $\sigma\to\sigma'$.

**Result:** The active route uses positive core certificates, a closed-system boundary branch, and a blocked Lock obtained by Tactic E2 (invariant mismatch). Two diagnostic `inc` certificates are retained at the mixing and gradient nodes, but they are explicitly outside the dependency cone of the designated goal. The declared finite-descent backend package upgrades structural exclusion to the final termination-and-complexity certificate
$$
K_{\mathrm{BubbleSortTerm}}^+.
$$

---

## Theorem Statement

::::{prf:theorem} Bubble Sort Termination and Complexity
:label: thm-bubble-sort-main

**Given:**
- State space:
  $$
  \mathcal X
  =
  \left\{
  (\sigma,n):
  \sigma\in S_n,\ n\in\mathbb N_{>0}
  \right\}.
  $$
- Dynamics:
  one bubble-sort pass step swaps adjacent out-of-order entries, and does nothing at indices already in nondecreasing order.
- Initial data:
  any input permutation $\sigma_0\in S_n$.

**Claim:** Bubble sort terminates at the sorted permutation after finitely many swaps, and the total number of swaps is at most
$$
\binom{n}{2},
$$
hence the swap complexity is $O(n^2)$.

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\mathcal{X}$ | permutation state space |
| $\Phi$ | inversion count |
| $\mathfrak{D}$ | unit swap cost on executed swaps |
| $S_t$ | discrete bubble-sort evolution |
| $\Sigma$ | bad-state set for route-relative singular analysis |

::::

---

:::{dropdown} **LLM Execution Protocol** (Click to expand)

## Note for LLM Agents: Complete Algorithmic Protocol

This instance is executed as a deterministic proof-object construction.

### **A.1 Mindset Shift**

1. Fill each permit with explicit finite-state algorithmic data.
2. Emit exactly one certificate at every node.
3. Use only declared packages: inversion count, finite permutation space, adjacent-swap dynamics, and the finite-descent backend package.
4. Treat the Lock and the termination/complexity extraction as separate certified steps.
5. Keep non-goal diagnostics explicit; do not force them into the designated goal route.

### **A.2 Certificate Outcome Types**

| Outcome | Symbol | Used Here | Meaning |
|---------|--------|-----------|---------|
| YES | $K_X^+$ | Yes | gate verified |
| INC | $K_X^{\mathrm{inc}}$ | Yes | recorded diagnostic outside the goal cone |
| BLOCKED | $K_X^{\mathrm{blk}}$ | Yes | Lock verdict |
| BREACHED | $K_X^{\mathrm{br}}$ | No | no surgery route selected |

### **A.3 Inc Permit Protocol**

Two residual diagnostics are recorded:

- $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ because the route is a deterministic descent algorithm, not a genuine mixing system.
- $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ because the route is not certified through a formal gradient-flow representation, only through discrete descent.

Both lie outside $\Downarrow(K_{\mathrm{BubbleSortTerm}}^+)$.

### **A.4 Upgrade Rule Execution**

No goal-relevant `inc` certificate is upgraded on the designated route. The only final promotion is
$$
K_{\mathrm{StructBubbleSort}}^+
\wedge
K_{\mathrm{FiniteDescentBackend}}^+
\Longrightarrow
K_{\mathrm{BubbleSortTerm}}^+.
$$

### **A.5 Breach Detection and Surgery**

No barrier breach occurs. No surgery is selected.

### **A.6 Obligation Tracking**

The goal-cone ledger is empty. Residual diagnostics are retained only for the non-goal mixing and gradient nodes.

### **A.7 Completion Criteria**

The proof object closes iff:

- all core nodes are executed;
- the closed-system branch is recorded at Node 13;
- Node 17 yields a certified Lock verdict;
- the explicit finite-descent backend upgrade is present;
- no obligation remains in $\Downarrow(K_{\mathrm{BubbleSortTerm}}^+)$.

### **A.8 Step-by-Step Implementation Guide for New Problems**

For this instance:

1. instantiate the permutation arena and inversion-count height;
2. execute Nodes 1-13 directly;
3. record the non-mixing and non-gradient diagnostics as non-goal `inc` certificates;
4. close the Lock using the bounded strict-descent invariant mismatch;
5. apply the finite-descent backend upgrade.

:::

---

## **Part 0: Interface Permit Implementation Checklist**
*Complete this section before running the Sieve. Each permit requires specific mathematical structures to be defined.*

### **0.1 Core Interface Permits (Nodes 1-12)**

| #  | Permit ID                  | Node           | Question                 | Required Implementation                                                   | Certificate                          |
|----|----------------------------|----------------|--------------------------|---------------------------------------------------------------------------|--------------------------------------|
| 1  | $D_E$                      | EnergyCheck    | Is Energy Finite?        | inversion-count height $\Phi$, unit swap cost $\mathfrak D$, explicit bound $\binom{n}{2}$ | $K_{D_E}^+$                      |
| 2  | $\mathrm{Rec}_N$           | ZenoCheck      | Are Events Finite?       | bad set = malformed non-permutations, no recovery events on admissible route, bounded event counter | $K_{\mathrm{Rec}_N}^+$           |
| 3  | $C_\mu$                    | CompactCheck   | Does Energy Concentrate? | finite symmetric group, trivial quotient, finite-state compactness | $K_{C_\mu}^+$                    |
| 4  | $\mathrm{SC}_\lambda$      | ScaleCheck     | Is Profile Subcritical?  | discrete no-scaling branch, static scale data | $K_{\mathrm{SC}_\lambda}^+$      |
| 5  | $\mathrm{SC}_{\partial c}$ | ParamCheck     | Are Constants Stable?    | parameter object $n$, fixed reference size, constant-size route | $K_{\mathrm{SC}_{\partial c}}^+$ |
| 6  | $\mathrm{Cap}_H$           | GeomCheck      | Is Codim $\geq 2$?       | empty bad singular set $\Sigma$, route-relative capacity zero | $K_{\mathrm{Cap}_H}^+$           |
| 7  | $\mathrm{LS}_\sigma$       | StiffnessCheck | Is Gap Certified?        | discrete inversion gap of size $1$, unique critical set at sorted permutation | $K_{\mathrm{LS}_\sigma}^+$       |
| 8  | $\mathrm{TB}_\pi$          | TopoCheck      | Is Sector Preserved?     | whole permutation sector as safe sector, no escape from $S_n$ | $K_{\mathrm{TB}_\pi}^+$          |
| 9  | $\mathrm{TB}_O$            | TameCheck      | Is Topology Tame?        | finite discrete state space, finite stratification | $K_{\mathrm{TB}_O}^+$            |
| 10 | $\mathrm{TB}_\rho$         | ErgoCheck      | Does Flow Mix?           | deterministic descent route; no genuine mixing certificate | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$         |
| 11 | $\mathrm{RepDesc}_K$       | ComplexCheck   | Is Description Finite?   | finite permutation language, bounded description complexity, faithful encoding | $K_{\mathrm{RepDesc}_K}^+$       |
| 12 | $\mathrm{GC}_\nabla$       | OscillateCheck | Is Flow Gradient?        | discrete descent route, no certified gradient-flow package | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$       |

#### **Template: $D_E$ (Energy Interface)**
- [x] **Height Functional $\Phi$:**
  $$
  \Phi(\sigma)
  =
  \#\{(i,j):1\le i<j\le n,\ \sigma(i)>\sigma(j)\}.
  $$
- [x] **Dissipation Rate $\mathfrak{D}$:** $\mathfrak D(\sigma\to\sigma')=1$ for each executed swap.
- [x] **Energy Inequality:** every executed bubble-sort swap satisfies
      $\Phi(\sigma')=\Phi(\sigma)-1\le \Phi(\sigma)+\int_0^1 \mathfrak D\,ds$ in the route-relative discrete form.
- [x] **Bound Witness:** $0\le \Phi(\sigma)\le \binom{n}{2}$.

#### **Template: Derived Witness Certificates (Optional)**
- [x] **$K_{D_{\max}}^+$ (diameter witness):** not instantiated on the designated route.
- [x] **$K_{\rho_{\max}}^+$ (density witness):** not instantiated on the designated route.

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [x] **Bad Set $\mathcal{B}$:** malformed states outside $S_n$.
- [x] **Recovery Map $\mathcal{R}$:** not used on the admissible route because the input already lies in $S_n$.
- [x] **Event Counter:** number of executed swaps.
- [x] **Finiteness:** $N(T)\le \Phi(\sigma_0)\le \binom{n}{2}$.

#### **Template: $C_\mu$ (Compactness Interface)**
- [x] **Symmetry Group $G$:** trivial group.
- [x] **Group Action $\rho$:** identity action on $S_n$.
- [x] **Quotient Space:** $\mathcal X//G=S_n$.
- [x] **Concentration Measure:** finite state spaces are compact and admit no concentration-escape phenomenon.

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [x] **Scaling Action:** no nontrivial scaling action is used on the designated route.
- [x] **Height Exponent $\alpha$:** route-relative no-scaling branch, recorded as $\alpha=0$.
- [x] **Dissipation Exponent $\beta$:** route-relative no-scaling branch, recorded as $\beta=0$.
- [x] **Criticality:** $\beta-\alpha=0$ on the discrete no-scaling branch.

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [x] **Parameter Space $\Theta$:** $\mathbb N_{>0}$.
- [x] **Parameter Map $\theta$:** $\theta(\sigma,n)=n$.
- [x] **Reference Point $\theta_0$:** the fixed input size $n$.
- [x] **Stability Bound:** $d(\theta(S_t x),\theta_0)=0$ on the route.

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [x] **Capacity Functional:** route-relative discrete cardinality capacity.
- [x] **Singular Set $\Sigma$:** empty on the admissible route.
- [x] **Codimension:** $\mathrm{codim}(\Sigma)=\infty$.
- [x] **Capacity Bound:** $\mathrm{Cap}(\Sigma)=0$.

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [x] **Gradient Operator $\nabla$:** discrete inversion-count variation under one adjacent swap.
- [x] **Critical Set $M$:** singleton consisting of the sorted permutation.
- [x] **Łojasiewicz Exponent $\theta$:** $\theta=1$ from the exact unit gap.
- [x] **Łojasiewicz-Simon Inequality:** the route uses the strict discrete gap witness $\Phi(\sigma)-\Phi(\sigma')=1$ on executed swaps.

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [x] **Topological Invariant $\tau$:** membership in the admissible permutation sector $S_n$.
- [x] **Sector Classification:** the designated route uses the single safe sector $S_n$.
- [x] **Sector Preservation:** $\tau(S_t x)=\tau(x)$ because every bubble-sort step stays inside $S_n$.
- [x] **Tunneling Events:** none on the route.

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [x] **O-minimal Structure $\mathcal{O}$:** finite discrete structure.
- [x] **Definability $\mathrm{Def}$:** all states and transitions are finitely definable.
- [x] **Singular Set Tameness:** $\Sigma$ is empty and therefore definable.
- [x] **Cell Decomposition:** finite discrete stratification.

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [x] **Measure $\mathcal{M}$:** counting measure on $S_n$.
- [x] **Invariant Measure $\mu$:** no mixing invariant is used on the designated route.
- [x] **Mixing Time $\tau_{\mathrm{mix}}$:** not certified on the designated route.
- [x] **Mixing Property:** recorded only as the non-goal diagnostic $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$.

#### **Template: $\mathrm{RepDesc}_K$ (Dictionary / Description Interface)**
- [x] **Language $\mathcal{L}$:** permutation arrays of length $n$.
- [x] **Dictionary $D$:** permutation array representation.
- [x] **Complexity Measure $K$:** description length of the array together with inversion-count complexity bound.
- [x] **Faithfulness:** the array representation faithfully determines the state and inversion count.

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [x] **Metric Tensor $g$:** route-relative discrete Hamming-type metric on permutations.
- [x] **Vector Field $v$:** adjacent swap of an out-of-order pair selected by bubble sort.
- [x] **Gradient Compatibility:** not certified on the designated route.
- [x] **Monotonicity:** discrete strict descent replaces a gradient-square identity.

### **0.2 Boundary Interface Permits (Nodes 13-16)**
*For open systems with inputs/outputs. Skip if system is closed.*

The finite algorithmic instance yields the closed-system branch.

| # | Permit ID | Node | Question | Required Implementation | Certificate |
|---|-----------|------|----------|------------------------|-------------|
| 13 | $\mathrm{Bound}_\partial$ | BoundaryCheck | Is System Open? | no external input/output spaces; no boundary maps $\iota,\pi$ on the designated route | $K_{\mathrm{Bound}_\partial}^-$ |
| 14 | $\mathrm{Bound}_B$ | OverloadCheck | Is Input Bounded? | not applicable after the closed-system branch | N/A |
| 15 | $\mathrm{Bound}_{\Sigma}$ | StarveCheck | Is Input Sufficient? | not applicable after the closed-system branch | N/A |
| 16 | $\mathrm{GC}_T$ | AlignCheck | Is Control Matched? | not applicable after the closed-system branch | N/A |

### **0.2b Derived Witness Certificates (Optional)**
*These are **not** gate permits. They are payload certificates derived from the gate results or supplied explicitly.*

| Certificate | Derived From | Payload | Notes |
|---|---|---|---|
| $K_{D_{\max}}^+$ | not instantiated | none | no diameter witness is used on the designated route |
| $K_{\rho_{\max}}^+$ | not instantiated | none | no density witness is used on the designated route |

**Record any witness certificates** in the Execution Trace payloads; if they are used to
justify analytic bridge admissibility, cite them explicitly in the Lock Mechanism section.

### **0.3 The Lock (Node 17)**

| Permit ID | Node | Question | Required Implementation | Certificate |
|-----------|------|----------|------------------------|-------------|
| $\mathrm{Cat}_{\mathrm{Hom}}$ | Lock | Is $\mathrm{Hom}(\mathcal H_{\mathrm{bad}},\mathcal H)=\emptyset$? | category $\mathbf{Hypo}_{T_{\text{algorithmic}}}$, universal bad pattern "infinite bubble-sort execution", certified completeness package $(K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+)$, tactic E2 strict-descent invariant mismatch | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |

| Item | Value |
|---|---|
| Category | $\mathbf{Hypo}_{T_{\text{algorithmic}}}$ |
| Universal bad object | infinite bubble-sort execution |
| Certified completeness package | present |
| Primary tactics | E2 (invariant mismatch by strict bounded descent) |
| Lock output | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |

### **0.3b Goal and Backend Certificates**
*These are goal-level or backend-level certificates that the run may require even after the thin interfaces have been instantiated.*

| Certificate | Role | Required When |
|---|---|---|
| $K_{\mathrm{Germ}}^+$ | classifiable bad-pattern germ package | this Lock-based structural exclusion route |
| $K_{\mathrm{init}}^+$ | universal bad object package | this Lock-based structural exclusion route |
| $K_{\mathrm{CatLib}}^+$ | completeness of the finite bad-pattern library | this Lock-based structural exclusion route |
| $K_{\mathrm{WellFoundedNat}}^+$ | bounded strict-descent package on $\mathbb N$ | route-relative finite-descent support |
| $K_{\mathrm{FiniteDescentBackend}}^+$ | converts structural exclusion into termination and explicit swap bound | final backend upgrade |
| $K_{\mathrm{StructBubbleSort}}^+$ | structural exclusion certificate mined from the blocked Lock | after Node 17, before final promotion |
| $K_{\mathrm{BubbleSortTerm}}^+$ | designated termination-and-complexity goal certificate | final closure of the proof object |

### **0.4 Implementation Templates**

#### **Template: $D_E$ (Energy Interface)**
- [x] **Height Functional $\Phi$:** inversion count $\Phi(\sigma)$.
- [x] **Dissipation Rate $\mathfrak{D}$:** $\mathfrak D=1$ on every executed swap.
- [x] **Energy Inequality:** each executed swap satisfies $\Phi(\sigma')=\Phi(\sigma)-1$.
- [x] **Bound Witness:** $B=\binom{n}{2}$.

#### **Template: Derived Witness Certificates (Optional)**
- [x] **$K_{D_{\max}}^+$ (diameter witness):** not instantiated on the designated route.
- [x] **$K_{\rho_{\max}}^+$ (density witness):** not instantiated on the designated route.

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [x] **Bad Set $\mathcal{B}$:** malformed states outside the permutation sector.
- [x] **Recovery Map $\mathcal{R}$:** not used on the admissible route because the input excludes $\mathcal B$.
- [x] **Event Counter:** swap count.
- [x] **Finiteness:** $N(T)\le \Phi(\sigma_0)\le \binom{n}{2}$.

#### **Template: $C_\mu$ (Compactness Interface)**
- [x] **Symmetry Group $G$:** trivial group.
- [x] **Group Action $\rho$:** identity.
- [x] **Quotient Space:** $\mathcal X // G=S_n$.
- [x] **Concentration Measure:** finite state space implies no concentration escape.

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [x] **Scaling Action:** no nontrivial scaling is used on the designated route.
- [x] **Height Exponent $\alpha$:** route-relative no-scaling branch, recorded as $\alpha=0$.
- [x] **Dissipation Exponent $\beta$:** route-relative no-scaling branch, recorded as $\beta=0$.
- [x] **Criticality:** $\beta-\alpha=0$ on the discrete no-scaling branch.

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [x] **Parameter Space $\Theta$:** $\mathbb N_{>0}$.
- [x] **Parameter Map $\theta$:** $\theta(\sigma,n)=n$.
- [x] **Reference Point $\theta_0$:** fixed input size $n$.
- [x] **Stability Bound:** $d(\theta(S_t x),\theta_0)=0$ on the route.

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [x] **Capacity Functional:** discrete cardinality capacity.
- [x] **Singular Set $\Sigma$:** empty on the route.
- [x] **Codimension:** $\mathrm{codim}(\Sigma)=\infty$.
- [x] **Capacity Bound:** $\mathrm{Cap}(\Sigma)=0$.

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [x] **Gradient Operator $\nabla$:** discrete inversion-count variation under one swap.
- [x] **Critical Set $M$:** singleton sorted permutation.
- [x] **Łojasiewicz Exponent $\theta$:** $\theta=1$.
- [x] **Łojasiewicz-Simon Inequality:** the route uses the exact unit inversion gap on every executed swap.

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [x] **Topological Invariant $\tau$:** membership in the safe permutation sector $S_n$.
- [x] **Sector Classification:** single safe sector $S_n$ on the designated route.
- [x] **Sector Preservation:** $\tau(S_t x)=\tau(x)$.
- [x] **Tunneling Events:** none on the route.

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [x] **O-minimal Structure $\mathcal{O}$:** finite discrete structure.
- [x] **Definability $\mathrm{Def}$:** all states and transitions are finitely definable.
- [x] **Singular Set Tameness:** $\Sigma$ is empty and definable.
- [x] **Cell Decomposition:** finite discrete stratification.

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [x] **Measure $\mathcal{M}$:** counting measure on $S_n$.
- [x] **Invariant Measure $\mu$:** no mixing invariant is used on the designated route.
- [x] **Mixing Time $\tau_{\mathrm{mix}}$:** not certified on the designated route.
- [x] **Mixing Property:** recorded only as the non-goal diagnostic $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$.

#### **Template: $\mathrm{RepDesc}_K$ (Dictionary / Description Interface)**
- [x] **Language $\mathcal{L}$:** permutation arrays of length $n$.
- [x] **Dictionary $D$:** array representation of the current permutation.
- [x] **Complexity Measure $K$:** route-relative description length together with inversion-count complexity bound.
- [x] **Faithfulness:** the description faithfully determines the state and its inversion count.

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [x] **Metric Tensor $g$:** route-relative discrete Hamming-type metric.
- [x] **Vector Field $v$:** adjacent inversion-removing swap.
- [x] **Gradient Compatibility:** not certified on the designated route.
- [x] **Monotonicity:** exact discrete descent replaces a gradient-square identity.

#### **Template: $\mathrm{Cat}_{\mathrm{Hom}}$ (Lock Interface)**
- [x] **Category $\mathbf{Hypo}_T$:** $\mathbf{Hypo}_{T_{\text{algorithmic}}}$ with admissible algorithmic morphisms on finite-state transition systems.
- [x] **Universal Bad Pattern $\mathcal{H}_{\text{bad}}$:** infinite bubble-sort execution.
- [x] **Certified Completeness Package:** $(K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+)$ is supplied.
- [x] **Primary Tactic Selected:** E2 (invariant mismatch).
- [x] **Tactic Logic:**
      * $I(\mathcal H)=\Phi(\sigma)\in\{0,\dots,\binom{n}{2}\}$ with strict drop by $1$ on every executed swap.
      * $I(\mathcal H_{\mathrm{bad}})$ would require infinitely many strict decreases.
      * Conclusion: mismatch implies $\mathrm{Hom}(\mathcal H_{\mathrm{bad}},\mathcal H)=\emptyset$ on the designated route.
- [x] **Preservation Lemmas (if needed):**
      - [x] $K_{\mathrm{MorphPresDim}}^+$ present if E1 is used: not needed.
      - [x] $K_{\mathrm{MorphPresMix}}^+$ present if E9 is used: not needed.
      - [x] $K_{\mathrm{MorphPresTame}}^+$ present if E10 is used: not needed.
- [x] **Exclusion Tactics Available:**
      - [x] E1 (Dimension): not used.
      - [x] E2 (Invariant): used.
      - [x] E3 (Positivity): not used.
      - [x] E4 (Integrality): not used.
      - [x] E5 (Functional): not used.
      - [x] E6 (Causal): not used.
      - [x] E7 (Thermodynamic): not used.
      - [x] E8 (Holographic): not used.
      - [x] E9 (Ergodic): not used.
      - [x] E10 (Definability): not used.
      - [x] E11 (Galois-Monodromy): not used.
      - [x] E12 (Algebraic Compressibility): not used.
      - [x] E13 (Algorithmic Completeness): not used.

:::{dropdown} **Part 0.5: Certificate Schemas and Upgrade Protocol** (Reference - Click to expand)

*Reference: For formal definitions, see the current interface and upgrade chapters, especially {prf:ref}`def-interface-goalpermits`, {prf:ref}`def-lock-contract`, {prf:ref}`mt-up-lock`, and the upgrade/promotion definitions cited throughout this template.*

### **0.5.1 Certificate Schemas**

#### **Positive Certificate ($K_X^+$)**

Used throughout the route, for example
$$
K_{D_E}^+
=
\bigl(
\Phi,
\ \mathfrak D=1,
\ 0\le \Phi\le \binom{n}{2}
\bigr).
$$

#### **NO-with-Witness Certificate ($K_X^{\mathrm{wit}}$)**

Not used on the designated route.

#### **NO-Inconclusive Certificate ($K_X^{\mathrm{inc}}$)**

The route records two non-goal diagnostics:

$$
K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
=
\left\{
\text{obligation: literal mixing certificate},
\text{missing: }[K_{\mathrm{Mix}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: deterministic descent algorithm, not a mixing flow}
\right\},
$$

$$
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
=
\left\{
\text{obligation: gradient-flow representation},
\text{missing: }[K_{\mathrm{GradAlgo}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: discrete descent replaces certified gradient structure}
\right\}.
$$

#### **Blocked Certificate ($K_X^{\mathrm{blk}}$)**

The Lock emits
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
=
\bigl(
\text{E2 invariant mismatch by bounded strict descent},
\{K_{D_E}^+,K_{\mathrm{Rec}_N}^+,K_{\mathrm{LS}_\sigma}^+,K_{\mathrm{RepDesc}_K}^+,K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+,K_{\mathrm{WellFoundedNat}}^+\}
\bigr).
$$

#### **Breached Certificate ($K_X^{\mathrm{br}}$)**

Not used on the designated route.

### **0.5.2 Upgrade Rule Schema**

No goal-relevant `inc` certificate is upgraded on the designated route.

#### **Rule Template**

The only final upgrade used here is
$$
K_{\mathrm{StructBubbleSort}}^+
\wedge
K_{\mathrm{FiniteDescentBackend}}^+
\Longrightarrow
K_{\mathrm{BubbleSortTerm}}^+.
$$

#### **Non-Circularity Guard**

$K_{\mathrm{FiniteDescentBackend}}^+$ is an explicit backend package and is not derived from $K_{\mathrm{BubbleSortTerm}}^+$, so the upgrade is non-circular.

#### **Upgrade Types**

| Type | Used Here | Source |
|------|-----------|--------|
| Instantaneous | No | none |
| A-posteriori | Yes | backend termination/complexity promotion after the Lock |

### **0.5.2b Promotion Permits (Blocked → YES$^\sim$)**

No blocked-to-YES$^\sim$ promotion is used. The Lock remains explicitly blocked and is mined in Part III-B for the structural exclusion certificate.

### **0.5.3 Surgery Certificate Schema**

No surgery certificate is used on the designated route.

### **0.5.4 Re-entry Certificate Schema**

No re-entry certificate is used on the designated route.

### **0.5.5 Context Accumulation**

$$
\Gamma_{\mathrm{route}}
=
\{
K_{D_E}^+,
K_{\mathrm{Rec}_N}^+,
K_{C_\mu}^+,
K_{\mathrm{SC}_\lambda}^+,
K_{\mathrm{SC}_{\partial c}}^+,
K_{\mathrm{Cap}_H}^+,
K_{\mathrm{LS}_\sigma}^+,
K_{\mathrm{TB}_\pi}^+,
K_{\mathrm{TB}_O}^+,
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},
K_{\mathrm{RepDesc}_K}^+,
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},
K_{\mathrm{Bound}_\partial}^-,
K_{\mathrm{Germ}}^+,
K_{\mathrm{init}}^+,
K_{\mathrm{CatLib}}^+,
K_{\mathrm{WellFoundedNat}}^+,
K_{\mathrm{FiniteDescentBackend}}^+,
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
\}.
$$

:::

---

## **Part I: The Instantiation (Thin Object Definitions)**

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**

- **State Space ($\mathcal X$):** permutations of $\{1,\dots,n\}$ together with fixed input size $n$.
- **Metric ($d$):** route-relative discrete Hamming-type metric on permutations.
- **Measure ($\mu$):** counting measure on $S_n$.
- **Auxiliary Object:** inversion set of the current permutation.

### **2. The Potential ($\Phi^{\text{thin}}$)**

- **Height Functional ($\Phi$):**
  $$
  \Phi(\sigma)
  =
  \#\{(i,j):1\le i<j\le n,\ \sigma(i)>\sigma(j)\}.
  $$
- **Secondary Height:** remaining swap budget inside $[0,\binom{n}{2}]$.
- **Equilibrium Set:** singleton sorted permutation.
- **Scaling Exponent ($\alpha$):** $\alpha=0$ on the discrete no-scaling branch.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**

- **Static Cost Branch:**
  $$
  \mathfrak D(\sigma\to\sigma')=1
  $$
  on executed swaps.
- **Dynamics:** adjacent out-of-order swap selected by bubble sort.
- **Backend Evaluation:** the exact termination and swap bound are supplied by the declared finite-descent backend package.

### **4. The Invariance ($G^{\text{thin}}$)**

- **Symmetry Group ($\mathrm{Grp}$):** trivial group.
- **Scaling ($\mathcal S$):** no nontrivial scaling used on the route.
- **Conserved Quantity:** membership in the permutation sector $S_n$.
- **Auxiliary Reconstruction:** inversion-count descent and finite-state well-foundedness.

---

## **Part II: Sieve Execution (Verification Run)**

### **EXECUTION PROTOCOL**

The designated route executes Nodes 1-13 directly, skips Nodes 14-16 on the closed-system branch, and then executes the Lock at Node 17. Two diagnostic `inc` certificates are recorded at Nodes 10 and 12, but they are excluded from the designated goal dependency cone.

### **Level 1: Conservation**

#### **Node 1: EnergyCheck ($D_E$)**

**Question:** Is the inversion-count height well-defined and bounded on the route?

**Step-by-step execution:**
1. [x] Every state on the route is a permutation $\sigma\in S_n$.
2. [x] The inversion count satisfies
   $$
   0\le \Phi(\sigma)\le \binom{n}{2}.
   $$
3. [x] Every executed bubble-sort swap removes exactly one inversion.

**Certificate:**
$$
K_{D_E}^+=(\Phi,\mathfrak D=1,0\le \Phi\le \binom{n}{2}).
$$

#### **Node 2: ZenoCheck ($\mathrm{Rec}_N$)**

**Question:** Are recovery events finite?

**Step-by-step execution:**
1. [x] The admissible route excludes malformed states outside $S_n$.
2. [x] No repair or restart event is introduced.
3. [x] The event counter equals the number of executed swaps and is bounded by the initial inversion count.

**Certificate:**
$$
K_{\mathrm{Rec}_N}^+=(N(T)\le \Phi(\sigma_0)\le \binom{n}{2},\text{empty-recovery route}).
$$

#### **Node 3: CompactCheck ($C_\mu$)**

**Question:** Does the route exhibit compactness modulo the tracked symmetry?

**Step-by-step execution:**
1. [x] The state space $S_n$ is finite.
2. [x] The symmetry group is trivial.
3. [x] The route therefore stays in a finite compact quotient shell.

**Certificate:**
$$
K_{C_\mu}^+=(G=\{e\},\text{finite compact state shell}).
$$

### **Level 2: Duality & Symmetry**

#### **Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)**

**Question:** Is the designated route scale-stable?

**Step-by-step execution:**
1. [x] No nontrivial scaling action is used by the discrete algorithmic route.
2. [x] The route is recorded on the no-scaling branch.
3. [x] The state sector is unchanged.

**Certificate:**
$$
K_{\mathrm{SC}_\lambda}^+=(\alpha=0,\beta=0,\text{discrete no-scaling branch}).
$$

#### **Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)**

**Question:** Are system parameters stable?

**Step-by-step execution:**
1. [x] The only parameter is the input size $n$.
2. [x] No parameter drift occurs during execution.
3. [x] The route remains in the same size sector.

**Certificate:**
$$
K_{\mathrm{SC}_{\partial c}}^+=(n,\text{fixed size sector}).
$$

### **Level 3: Geometry & Stiffness**

#### **Node 6: GeomCheck ($\mathrm{Cap}_H$)**

**Question:** Is the route-relative singular set negligible?

**Step-by-step execution:**
1. [x] The admissible route has no singular states.
2. [x] The bad singular set is empty.
3. [x] The route therefore carries zero capacity obstruction.

**Certificate:**
$$
K_{\mathrm{Cap}_H}^+=(\Sigma=\varnothing,\mathrm{Cap}(\Sigma)=0).
$$

#### **Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)**

**Question:** Is there a discrete gap certificate?

**Step-by-step execution:**
1. [x] Every executed swap decreases inversion count by exactly $1$.
2. [x] The sorted permutation is the unique zero-inversion critical state.
3. [x] The route therefore carries an exact unit gap.

**Certificate:**
$$
K_{\mathrm{LS}_\sigma}^+=(\theta=1,\text{exact unit inversion gap}).
$$

### **Level 4: Topology**

#### **Node 8: TopoCheck ($\mathrm{TB}_\pi$)**

**Question:** Is the route sector preserved?

**Step-by-step execution:**
1. [x] The route sector is the permutation space $S_n$.
2. [x] Every bubble-sort step maps permutations to permutations.
3. [x] No tunneling event leaves the safe sector.

**Certificate:**
$$
K_{\mathrm{TB}_\pi}^+=(\tau=S_n,\text{sector preserved}).
$$

#### **Node 9: TameCheck ($\mathrm{TB}_O$)**

**Question:** Is the topology tame?

**Step-by-step execution:**
1. [x] The state space is finite and discrete.
2. [x] Every subset is definable in the route-relative formalization.
3. [x] The corresponding stratification is finite.

**Certificate:**
$$
K_{\mathrm{TB}_O}^+=(\mathcal O,\text{finite discrete stratification}).
$$

### **Level 5: Mixing**

#### **Node 10: ErgoCheck ($\mathrm{TB}_\rho$)**

**Question:** Does the flow mix?

**Step-by-step execution:**
1. [x] The route is a deterministic descent algorithm, not a stochastic mixing system.
2. [x] No literal mixing-time certificate is produced on this route.
3. [x] This diagnostic is not used in the designated goal chain.

**Certificate:**
$$
K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
=
\left\{
\text{obligation: literal mixing certificate},
\text{missing: }[K_{\mathrm{Mix}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: deterministic descent algorithm, not a mixing flow}
\right\}.
$$

### **Level 6: Complexity**

#### **Node 11: ComplexCheck ($\mathrm{RepDesc}_K$)**

**Question:** Is the description finite and faithful?

**Step-by-step execution:**
1. [x] Every state is represented by an array of length $n$.
2. [x] The inversion count and enabled swap positions are finitely computable from that description.
3. [x] The representation faithfully determines the route-relative state.

**Certificate:**
$$
K_{\mathrm{RepDesc}_K}^+=(\mathcal L,D,K,\text{faithful}).
$$

#### **Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)**

**Question:** Is the route gradient-compatible?

**Step-by-step execution:**
1. [x] The route is discrete strict descent rather than a certified gradient flow.
2. [x] No gradient representation is needed for the designated goal.
3. [x] This diagnostic is outside the designated goal chain.

**Certificate:**
$$
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
=
\left\{
\text{obligation: gradient-flow representation},
\text{missing: }[K_{\mathrm{GradAlgo}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: discrete descent replaces certified gradient structure}
\right\}.
$$

### **Level 7: Boundary (Open Systems)**

#### **Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)**

**Question:** Is the system open?

**Step-by-step execution:**
1. [x] The finite algorithmic instance has no external control input.
2. [x] There are no route-level boundary maps $\iota,\pi$.
3. [x] The run enters the closed-system branch.

**Certificate:**
$$
K_{\mathrm{Bound}_\partial}^-.
$$

#### **Node 14: OverloadCheck ($\mathrm{Bound}_B$)**

**Question:** Is input bounded?

**Outcome:** not applicable on the closed-system branch.

#### **Node 15: StarveCheck ($\mathrm{Bound}_{\Sigma}$)**

**Question:** Is input sufficient?

**Outcome:** not applicable on the closed-system branch.

#### **Node 16: AlignCheck ($\mathrm{GC}_T$)**

**Question:** Is control matched?

**Outcome:** not applicable on the closed-system branch.

### **Level 8: The Lock**

#### **Node 17: BarrierExclusion ($\mathrm{Cat}_{\mathrm{Hom}}$)**

**Question:** Is $\mathrm{Hom}(\mathcal H_{\mathrm{bad}},\mathcal H)=\emptyset$?

**Step-by-step execution:**
1. [x] The bad-pattern library consists of infinite bubble-sort executions.
2. [x] The certified completeness package $(K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+)$ is present.
3. [x] The actual route carries the strict bounded invariant
   $$
   \Phi(\sigma)\in\{0,\dots,\binom{n}{2}\}
   $$
   with unit decrease on every executed swap.
4. [x] The bad pattern would require infinitely many strict decreases.
5. [x] Apply **E2 (Invariant mismatch)**: infinite strict descent is incompatible with a bounded natural-valued invariant.

**Lock Verdict:**
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
=
\bigl(
\text{E2 invariant mismatch by bounded strict descent},
\{K_{D_E}^+,K_{\mathrm{Rec}_N}^+,K_{\mathrm{LS}_\sigma}^+,K_{\mathrm{RepDesc}_K}^+,K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+,K_{\mathrm{WellFoundedNat}}^+\}
\bigr).
$$

---

## **Part II-B: Upgrade Pass**

### **Upgrade Pass Protocol**

No goal-relevant `inc` certificate is introduced.

| ID | Node | Obligation | Missing | In Goal Cone? |
|----|------|------------|---------|---------------|
| OBL-1 | 10 | literal mixing certificate | $K_{\mathrm{Mix}}^+$ | No |
| OBL-2 | 12 | gradient-flow representation | $K_{\mathrm{GradAlgo}}^+$ | No |

No upgrade is required before the Lock. The final termination-and-complexity promotion is handled in Part III-B as a backend theorem application.

---

## **Part II-C: Breach/Surgery/Re-entry Protocol**

### **Breach Detection**

No $K_X^{\mathrm{br}}$ certificate was emitted.

### **Surgery Selection**

No surgery selected.

### **Surgery Execution**

Not applicable.

### **Re-entry Protocol**

Not applicable.

---

## **Part III-A: Lyapunov Reconstruction**

### **Lyapunov Existence Check**

The designated route does not invoke KRNL-Lyapunov reconstruction. The goal closes through explicit inversion-count descent and the declared finite-descent backend package rather than through a reconstructed Lyapunov chain.

### **Step 1: Value Function Construction (KRNL-Lyapunov)**

Not invoked on the designated route.

### **Step 2: Jacobi Metric Reconstruction (KRNL-Jacobi)**

Not invoked on the designated route.

### **Step 3: Hamilton-Jacobi PDE (KRNL-HamiltonJacobi)**

Not invoked on the designated route.

### **Step 4: Verify Lyapunov Properties**

Not invoked on the designated route.

---

## **Part III-B: Result Extraction (Mining the Run)**

### **3.1 Global Theorems**

- **Structural Exclusion Theorem:** from the blocked Lock together with the certified completeness package and the declared bounded-descent support certificates,
  $$
  K_{\mathrm{StructBubbleSort}}^+.
  $$
  Statement: an infinite bubble-sort execution cannot occur on the designated route.

- **Algorithmic Backend Theorem:** from structural exclusion plus the explicit finite-descent backend package,
  $$
  K_{\mathrm{StructBubbleSort}}^+
  \wedge K_{\mathrm{FiniteDescentBackend}}^+
  \Longrightarrow
  K_{\mathrm{BubbleSortTerm}}^+.
  $$
  Statement: bubble sort terminates and performs at most $\binom{n}{2}$ swaps.

- **Scattering / Backend Analytic Upgrade:** not used beyond the declared finite-descent backend package.
- **Observer-Relative Censorship Theorem:** not used.
- **Singularity Classification:** the only route-relevant bad-profile family consists of nonterminating executions.

### **3.2 Quantitative Bounds**

- **Inversion bound:**
  $$
  0\le \Phi(\sigma)\le \binom{n}{2}.
  $$
- **Descent bound:**
  $$
  \Phi(\sigma')=\Phi(\sigma)-1
  $$
  on every executed swap.
- **Swap-count bound:**
  $$
  N_{\mathrm{swap}}\le \Phi(\sigma_0)\le \binom{n}{2}.
  $$

### **3.3 Functional Objects**

- **Well-founded descent package:** $K_{\mathrm{WellFoundedNat}}^+$.
- **Finite-descent backend package:** $K_{\mathrm{FiniteDescentBackend}}^+$.
- **Inversion-count description package:** carried through $K_{\mathrm{RepDesc}_K}^+$.

### **3.4 Retroactive Upgrades**

- No goal-relevant `inc` certificate required discharge.
- The two residual diagnostics remain outside the goal cone.
- Final termination/complexity extraction is upgraded from structural exclusion by the declared backend package.

### **3.5 ZFC Proof Export (Appendix Bridge)**

Not requested. The proof object stops at the certified termination-and-complexity certificate.

---

## **Part III-C: Obligation Ledger**

### **Introduced Obligations**

| ID | Node | Certificate | Obligation | Missing | In Goal Cone? | Status |
|----|------|-------------|------------|---------|---------------|--------|
| OBL-1 | 10 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | literal mixing certificate | $K_{\mathrm{Mix}}^+$ | No | Residual diagnostic |
| OBL-2 | 12 | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | gradient-flow representation | $K_{\mathrm{GradAlgo}}^+$ | No | Residual diagnostic |

### **Discharge Events**

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| — | — | — | — |

### **Remaining Obligations**

**Count:** 2

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| OBL-1 | literal mixing certificate | deterministic descent route does not require mixing |
| OBL-2 | gradient-flow representation | discrete descent route does not require gradient structure |

### **Ledger Validation**

- [x] All goal-relevant `inc` certificates upgraded or absent.
- [x] All goal-relevant breach obligations discharged or absent.
- [x] The remaining obligations are explicitly outside the designated goal dependency cone.

**Ledger Status:** GOAL-CONE EMPTY for $K_{\mathrm{BubbleSortTerm}}^+$ with two residual non-goal diagnostics.

---

## **Part IV: Final Certificate Chain**

### **4.1 Validity Checklist**

- [x] **All 12 core nodes executed**
- [x] **Boundary nodes handled correctly** (closed-system branch)
- [x] **Lock executed**
- [x] **Lock verdict obtained:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- [x] **Designated goal certificate reached:** $K_{\mathrm{BubbleSortTerm}}^+$
- [x] **If claiming structural exclusion:** certified completeness package is present
- [x] **If claiming backend termination/complexity extraction:** finite-descent backend package is present
- [x] **Upgrade pass completed**
- [x] **Surgery/Re-entry completed if needed:** not needed
- [x] **No unresolved obligations remain in the designated goal dependency cone**

**Validity Status:** UNCONDITIONAL PROOF for the designated goal $K_{\mathrm{BubbleSortTerm}}^+$.

### **4.2 Certificate Accumulation Trace**

```text
Node 1:  K_{D_E}^+
Node 2:  K_{\mathrm{Rec}_N}^+
Node 3:  K_{C_\mu}^+
Node 4:  K_{\mathrm{SC}_\lambda}^+
Node 5:  K_{\mathrm{SC}_{\partial c}}^+
Node 6:  K_{\mathrm{Cap}_H}^+
Node 7:  K_{\mathrm{LS}_\sigma}^+
Node 8:  K_{\mathrm{TB}_\pi}^+
Node 9:  K_{\mathrm{TB}_O}^+
Node 10: K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
Node 11: K_{\mathrm{RepDesc}_K}^+
Node 12: K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
Node 13: K_{\mathrm{Bound}_\partial}^-
Node 14: N/A
Node 15: N/A
Node 16: N/A
Support: K_{\mathrm{Germ}}^+, K_{\mathrm{init}}^+, K_{\mathrm{CatLib}}^+, K_{\mathrm{WellFoundedNat}}^+, K_{\mathrm{FiniteDescentBackend}}^+
Node 17: K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
Part III-A: not invoked on designated route
Part III-B: K_{\mathrm{StructBubbleSort}}^+ \wedge K_{\mathrm{FiniteDescentBackend}}^+ -> K_{\mathrm{BubbleSortTerm}}^+
```

### **4.3 Final Certificate Set**

$$
\Gamma_{\mathrm{final}}
=
\{
K_{D_E}^+,
K_{\mathrm{Rec}_N}^+,
K_{C_\mu}^+,
K_{\mathrm{SC}_\lambda}^+,
K_{\mathrm{SC}_{\partial c}}^+,
K_{\mathrm{Cap}_H}^+,
K_{\mathrm{LS}_\sigma}^+,
K_{\mathrm{TB}_\pi}^+,
K_{\mathrm{TB}_O}^+,
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},
K_{\mathrm{RepDesc}_K}^+,
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},
K_{\mathrm{Bound}_\partial}^-,
K_{\mathrm{Germ}}^+,
K_{\mathrm{init}}^+,
K_{\mathrm{CatLib}}^+,
K_{\mathrm{WellFoundedNat}}^+,
K_{\mathrm{FiniteDescentBackend}}^+,
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}},
K_{\mathrm{StructBubbleSort}}^+,
K_{\mathrm{BubbleSortTerm}}^+
\}.
$$

### **4.4 Conclusion**

**Conclusion:** The designated target claim is **ESTABLISHED**. Bubble sort admits a complete template-level proof object whose final goal certificate is $K_{\mathrm{BubbleSortTerm}}^+$.

---

## **Formal Proof**

::::{prf:proof} Proof of Theorem {prf:ref}`thm-bubble-sort-main`
:label: proof-thm-bubble-sort-main

The proof proceeds by structural sieve analysis in seven phases.

**Phase 1 (Instantiation):** Part I defines the bubble-sort thin objects $(\mathcal X,\Phi,\mathfrak D,G)$ on the finite permutation state space.

**Phase 2 (Conservation):** Nodes 1-3 produce $K_{D_E}^+$, $K_{\mathrm{Rec}_N}^+$, and $K_{C_\mu}^+$, certifying bounded inversion count, bounded swap-event count, and finite-state compactness.

**Phase 3 (Scaling):** Nodes 4-5 produce $K_{\mathrm{SC}_\lambda}^+$ and $K_{\mathrm{SC}_{\partial c}}^+$, recording the discrete no-scaling branch and the fixed input-size sector.

**Phase 4 (Geometry):** Nodes 6-9 produce the empty-singular-set, unit-gap, safe-sector, and tame finite-state certificates required on the designated route.

**Phase 5 (Diagnostics):** Nodes 10 and 12 emit $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ and $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$, but Part III-C records that both obligations are outside the dependency cone of the designated goal. Node 11 supplies the faithful permutation-description certificate.

**Phase 6 (Boundary):** Node 13 records the closed-system branch, so Nodes 14-16 are not applicable.

**Phase 7 (Lock / Backend Upgrade):** Node 17 blocks the bad pattern via $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ using E2 with the certified completeness package and the bounded strict-descent mismatch. Part III-B first extracts the structural certificate from that blocked route, then combines it with $K_{\mathrm{FiniteDescentBackend}}^+$ to derive the final termination-and-complexity certificate $K_{\mathrm{BubbleSortTerm}}^+$.

Therefore the designated goal certificate is established and the residual diagnostics do not obstruct it because they lie outside $\Downarrow(K_{\mathrm{BubbleSortTerm}}^+)$.
$$
\therefore K_{\mathrm{BubbleSortTerm}}^+ \quad \square
$$

::::

---

## **Verification Summary**

| Component | Status | Certificate |
|-----------|--------|-------------|
| Nodes 1-12 (Core) | PASS / DIAGNOSTIC | positive route with two non-goal `inc` diagnostics |
| Nodes 13-16 (Boundary) | N/A / PASS | closed-system branch via $K_{\mathrm{Bound}_\partial}^-$ |
| Node 17 (Lock) | BLOCKED | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |
| Goal Certificate | REACHED | $K_{\mathrm{BubbleSortTerm}}^+$ |
| Obligation Ledger | GOAL-CONE EMPTY | residual diagnostics only |
| Upgrade Pass | COMPLETE | backend termination/complexity promotion only |

**Final Verdict:** [x] UNCONDITIONAL PROOF / [ ] CONDITIONAL PROOF / [ ] SINGULARITY CONFIRMED / [ ] GOAL NOT REACHED

---

## **References**

1. Hypostructure Framework v1.0 (current Jupyter Book formalism)
2. Standard inversion-count analysis of bubble sort
3. Finite-state well-founded descent principles for deterministic algorithms

---

## Appendix: Replay Bundle (Machine-Checkability)

This proof object is replayed by providing:
1. `trace.json`: ordered node outcomes
2. `certs/`: serialized certificates with payload hashes
3. `inputs.json`: thin objects and initial-state hash
4. `closure.cfg`: promotion/closure settings

```json
{
  "problem": "bubble-sort",
  "goal": "K_BubbleSortTerm^+",
  "route": [
    "K_DE^+",
    "K_RecN^+",
    "K_Cmu^+",
    "K_SClambda^+",
    "K_SCpartialc^+",
    "K_CapH^+",
    "K_LSsigma^+",
    "K_TBpi^+",
    "K_TBO^+",
    "K_TBrho^inc",
    "K_RepDescK^+",
    "K_GCnabla^inc",
    "K_Boundpartial^-",
    "K_Germ^+",
    "K_init^+",
    "K_CatLib^+",
    "K_WellFoundedNat^+",
    "K_FiniteDescentBackend^+",
    "K_CatHom^blk",
    "K_StructBubbleSort^+",
    "K_BubbleSortTerm^+"
  ],
  "obligations": {
    "OBL-1": {
      "certificate": "K_TBrho^inc",
      "in_goal_cone": false,
      "status": "residual_diagnostic"
    },
    "OBL-2": {
      "certificate": "K_GCnabla^inc",
      "in_goal_cone": false,
      "status": "residual_diagnostic"
    }
  },
  "goal_cone_empty": true
}
```

**Replay acceptance criterion:** The checker recomputes the same $\Gamma_{\mathrm{final}}$ and emits `FINAL`.

---

## Executive Summary: The Proof Dashboard
*Fill this section after completing the sieve run (Phase 0: Dashboard Generation).*

### 1. System Instantiation (The Physics)
*Mapping the physical problem to the Hypostructure categories.*

| Object | Definition | Role |
| :--- | :--- | :--- |
| **Arena ($\mathcal{X}$)** | finite permutation state space $S_n$ | State Space |
| **Potential ($\Phi$)** | inversion count | Lyapunov Functional |
| **Cost ($\mathfrak{D}$)** | unit swap cost on executed swaps | Dissipation |
| **Invariance ($G$)** | trivial symmetry and sector preservation inside $S_n$ | Symmetry Sector |

### 2. Execution Trace (The Logic)

| Node | Check | Outcome | Certificate Payload | Ledger State |
|---|---|---:|---|---|
| 1 | Energy Bound | YES | inversion count bounded by $\binom{n}{2}$ | `[]` |
| 2 | Zeno Check | YES | swap count bounded by initial inversions | `[]` |
| 3 | Compact Check | YES | finite compact state shell | `[]` |
| 4 | Scale Check | YES | discrete no-scaling branch | `[]` |
| 5 | Param Check | YES | fixed input size | `[]` |
| 6 | Geom Check | YES | empty singular set | `[]` |
| 7 | Stiffness Check | YES | exact unit inversion gap | `[]` |
| 8 | Topo Check | YES | safe permutation sector preserved | `[]` |
| 9 | Tame Check | YES | finite discrete stratification | `[]` |
| 10 | Ergo Check | INC | deterministic descent, not mixing | `[OBL-1]` |
| 11 | Complex Check | YES | finite faithful permutation description | `[OBL-1]` |
| 12 | Oscillate Check | INC | no certified gradient flow | `[OBL-1, OBL-2]` |
| 13 | Boundary Check | CLOSED | no open-system branch | `[OBL-1, OBL-2]` |
| 17 | LOCK | BLOCK | E2 bounded strict-descent mismatch | `[OBL-1, OBL-2]` |

### 3. Lock Mechanism (The Exclusion)

| Tactic | Description | Status | Reason / Mechanism |
|---|---|---:|---|
| E1 | Dimension | N/A | not used |
| E2 | Invariant | PASS | infinite execution cannot match bounded strict descent of $\Phi$ |
| E3 | Positivity | N/A | not used |
| E4 | Integrality | N/A | not used |
| E5 | Functional | N/A | not used |
| E6 | Causal | N/A | not used |
| E7 | Thermodynamic | N/A | not used |
| E8 | Holographic | N/A | not used |
| E9 | Ergodic | N/A | not used |
| E10 | Definability | N/A | not used |
| E11 | Galois-Monodromy | N/A | not used |
| E12 | Algebraic Compressibility | N/A | not used |
| E13 | Algorithmic Completeness | N/A | not used |

### 4. Final Verdict

- **Designated Goal Certificate:** $K_{\mathrm{BubbleSortTerm}}^+$
- **Status:** UNCONDITIONAL
- **Goal-Cone Ledger:** EMPTY
- **Residual Non-Goal Obligations:** `OBL-1`, `OBL-2`
- **Singularity Set:** $\Sigma=\varnothing$
- **Primary Final Route:** direct sieve execution + E2-blocked Lock + finite-descent backend upgrade

---

## Document Information

| Field | Value |
|-------|-------|
| **Document Type** | Proof Object |
| **Framework** | Hypostructure v1.0 |
| **Problem Class** | Classical algorithms |
| **Problem Type** | Termination and complexity theorem |
| **System Type** | $T_{\text{algorithmic}}$ |
| **Singularity Type** | `FINITE_DESCENT` |
| **Verification Level** | Machine-checkable |
| **Inc Certificates** | 2 introduced, 0 discharged |
| **Final Status** | [x] UNCONDITIONAL |
| **Generated** | 2026-04-15 |

*This document constitutes a machine-checkable proof object under the Hypostructure framework.*
*Each certificate can be independently verified against the definitions in the current formalism chapters of this Jupyter Book.*

**QED**
