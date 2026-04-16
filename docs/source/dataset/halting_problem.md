# Halting Problem

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Classify the halting set $\mathcal K=\{e : \varphi_e(e)\downarrow\}$ in the algorithmic phase semantics of the Structural Sieve |
| **System Type** | $T_{\text{algorithmic}}$ (Computability / AIT) |
| **Target Claim** | Liquid-phase horizon classification of the halting set |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2026-04-14 |

---

## Automation Witness

We certify that this instance is eligible for the algorithmic factory layer of the Hypostructure framework.

- **Type witness:** $T_{\text{algorithmic}}$ is treated as a good type for finite encodings, prefix-trace extraction, and algorithmic phase classification.
- **Automation witness:** the Automation Guarantee supplies the thin parsing and certificate bookkeeping needed for the halting-set instance.
- **Scope note:** this automation witness discharges the factory layer only. The computably enumerable witness, the Axiom R failure witness, the liquid-phase certificate, and the horizon goal certificate are certified explicitly below.

**Certificate:**
$$
K_{\mathrm{Auto}}^+
=
\bigl(
T_{\text{algorithmic}}\ \text{good},
\ \mathrm{AutomationGuarantee},
\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, AIT-Phase}
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

This document presents a machine-checkable Hypostructure proof object for the halting set
$$
\mathcal K=\{e : \varphi_e(e)\downarrow\}.
$$

The thin object is the family of finite prefix traces
$$
\tau_n=\chi_{\mathcal K}\!\upharpoonright [0,n-1]\in \{0,1\}^n.
$$
The route-relevant thin certificates record finite description, finite prefix event count, parameter stability, and sector bookkeeping. The decisive backend data are:
$$
K_{\mathrm{CE}}^+,
\qquad
K_{\mathrm{AxiomR}}^{\mathrm{wit}}.
$$
Here $K_{\mathrm{CE}}^+$ is the computably enumerable witness for $\mathcal K$, and $K_{\mathrm{AxiomR}}^{\mathrm{wit}}$ is the diagonal witness that no total recovery operator decides $\mathcal K$.

By the framework theorem {prf:ref}`thm-sieve-thermo-correspondence` together with {prf:ref}`def-algorithmic-phases`, these certificates force the phase certificate
$$
K_{\mathrm{Liquid}}^+
$$
and the designated goal certificate
$$
K^{\mathrm{hor}}.
$$

---

## Theorem Statement

::::{prf:theorem} Halting Problem
:label: thm-halting-problem

Let $U$ be a fixed universal prefix-free Turing machine, let
$$
\mathcal K=\{e : \varphi_e(e)\downarrow\},
$$
and for each $n\ge 1$ let
$$
\tau_n:=\chi_{\mathcal K}\!\upharpoonright [0,n-1]\in\{0,1\}^n
$$
be the length-$n$ prefix of the characteristic sequence of $\mathcal K$.

Assume the certified package recorded in Parts 0 and I is present:

1. route-relevant thin certificates
   $$
   K_{D_E}^+,\ 
   K_{\mathrm{Rec}_N}^+,\ 
   K_{\mathrm{SC}_\lambda}^+,\ 
   K_{\mathrm{SC}_{\partial c}}^+,\ 
   K_{\mathrm{TB}_\pi}^+,\ 
   K_{\mathrm{RepDesc}_K}^+,\ 
   K_{\mathrm{Bound}_\partial}^-;
   $$
2. the computably enumerable witness
   $$
   K_{\mathrm{CE}}^+;
   $$
3. the diagonal witness
   $$
   K_{\mathrm{AxiomR}}^{\mathrm{wit}},
   $$
   certifying that Axiom R fails for $\mathcal K$.

Then the framework derives
$$
K_{\mathrm{Liquid}}^+
\qquad\text{and}\qquad
K^{\mathrm{hor}}.
$$

Equivalently, the halting set is classified as a Liquid-phase algorithmic problem, and the Structural Sieve returns the horizon verdict.

**Notation:**

| Symbol | Definition |
|--------|------------|
| $U$ | Fixed universal prefix-free Turing machine |
| $\mathcal K$ | Halting set $\{e : \varphi_e(e)\downarrow\}$ |
| $\tau_n$ | Length-$n$ prefix of the characteristic sequence of $\mathcal K$ |
| $\Phi(\tau_n)$ | Finite-description energy proxy for the prefix trace |
| $\mathfrak D(\tau_n)$ | Prefix computation-depth proxy |
| $K_{\mathrm{CE}}^+$ | Computably enumerable witness for $\mathcal K$ |
| $K_{\mathrm{AxiomR}}^{\mathrm{wit}}$ | Witness that Axiom R fails for $\mathcal K$ |
| $K_{\mathrm{Liquid}}^+$ | Liquid-phase classification certificate |
| $K^{\mathrm{hor}}$ | Designated horizon goal certificate |

::::

---

## Part 0: Interface Permit Implementation

### 0.1 Core Interface Permits (Nodes 1-12)

#### Permit $D_E$ (Energy Interface)

- [x] **Height Functional:** $\Phi(\tau_n):=K_\epsilon(\tau_n)$, the finite-description proxy for the thin trace.
- [x] **Energy Witness:** every finite prefix $\tau_n$ has a finite program description.
- [x] **Gate Output:** the run emits
  $$
  K_{D_E}^+=(\Phi,\text{finite-description witness}).
  $$

#### Permit $\mathrm{Rec}_N$ (Recovery / Event Interface)

- [x] **Bad Set:** unresolved diagonal queries inside a fixed prefix trace.
- [x] **Recovery Map:** stage-by-stage dovetailing on diagonal inputs.
- [x] **Event Counter:** for the prefix trace $\tau_n$, the event count is
  $$
  N(n)=n.
  $$
- [x] **Gate Output:** the run emits
  $$
  K_{\mathrm{Rec}_N}^+=(\mathcal B,\mathcal R,N(n)<\infty).
  $$

#### Permit $C_\mu$ (Compactness Interface)

- [x] **Route Policy:** no compactness/profile backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{C_\mu}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

#### Permit $\mathrm{SC}_\lambda$ (Scaling Interface)

- [x] **Scaling Variable:** prefix length $n$.
- [x] **Trace Bookkeeping:** raw prefix size grows linearly with $n$.
- [x] **Gate Output:** the run emits
  $$
  K_{\mathrm{SC}_\lambda}^+=(\alpha=1,\beta=0,\lambda_c=0,\beta-\alpha<0).
  $$

#### Permit $\mathrm{SC}_{\partial c}$ (Parameter Interface)

- [x] **Parameter Tuple:** $(U,n)$.
- [x] **Reference Point:** the chosen universal machine $U$ is fixed throughout the run.
- [x] **Gate Output:** the run emits
  $$
  K_{\mathrm{SC}_{\partial c}}^+=(U,n,\text{parameter stability}).
  $$

#### Permit $\mathrm{Cap}_H$ (Capacity Interface)

- [x] **Route Policy:** no capacity backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{\mathrm{Cap}_H}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

#### Permit $\mathrm{LS}_\sigma$ (Stiffness Interface)

- [x] **Route Policy:** no stiffness backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

#### Permit $\mathrm{TB}_\pi$ (Topology / Sector Interface)

- [x] **Sector Map:** decidable / computably enumerable but undecidable / random.
- [x] **Instance Placement:** the halting set is assigned to the computably enumerable sector before the final phase certificate is emitted.
- [x] **Gate Output:** the run emits
  $$
  K_{\mathrm{TB}_\pi}^+=(\tau,\text{algorithmic sector bookkeeping}).
  $$

#### Permit $\mathrm{TB}_O$ (Tameness Interface)

- [x] **Route Policy:** no tame/o-minimal backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{\mathrm{TB}_O}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

#### Permit $\mathrm{TB}_\rho$ (Mixing Interface)

- [x] **Route Policy:** no mixing backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

#### Permit $\mathrm{RepDesc}_K$ (Finite-Description Interface)

- [x] **Language:** universal-machine code, diagonal input, and dovetailing trace representation.
- [x] **Dictionary:** finite encoding of $U$ and the prefix extractor $\tau_n$.
- [x] **Gate Output:** the run emits
  $$
  K_{\mathrm{RepDesc}_K}^+=(\mathcal L,D,K_\epsilon,\text{finite algorithmic description}).
  $$

#### Permit $\mathrm{GC}_\nabla$ (Gradient / Oscillation Interface)

- [x] **Route Policy:** no gradient backend is used in the designated goal route.
- [x] **Gate Output:** the run records
  $$
  K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
  $$
  outside the dependency cone of the designated goal.

### 0.2 Boundary Interface Permit

The halting-set instance is treated as a closed symbolic system with boundary tuple
$$
\partial^{\mathrm{thin}}_{\mathcal K}=(1,\mathrm{Tr},0,0),
$$
so the run emits
$$
K_{\mathrm{Bound}_\partial}^-.
$$

### 0.3 Declared Backend Certificate Bundle

| Certificate | Role in This File |
|-------------|-------------------|
| $K_{\mathrm{CE}}^+$ | computably enumerable witness for $\mathcal K$ |
| $K_{\mathrm{AxiomR}}^{\mathrm{wit}}$ | diagonal witness that Axiom R fails for $\mathcal K$ |
| $K_{\mathrm{Liquid}}^+$ | liquid-phase certificate from {prf:ref}`thm-sieve-thermo-correspondence` and {prf:ref}`def-algorithmic-phases` |
| $K^{\mathrm{hor}}$ | designated horizon goal certificate |

### 0.4 Route Selection

The designated goal route does not use the Lock. The halting-set proof closes at the algorithmic phase layer:
$$
K_{\mathrm{CE}}^+
\wedge
K_{\mathrm{AxiomR}}^{\mathrm{wit}}
\Longrightarrow
K_{\mathrm{Liquid}}^+
\Longrightarrow
K^{\mathrm{hor}}.
$$

---

## Part I: The Instantiation

### 1. Thin Algorithmic Object

For each $n\ge 1$, define the prefix trace
$$
\tau_n=\chi_{\mathcal K}\!\upharpoonright [0,n-1]\in\{0,1\}^n.
$$

Set
$$
\mathcal X_{\mathcal K}^{\mathrm{thin}}:=\{\tau_n : n\ge 1\},
$$
with the normalized Hamming metric on equal-length traces and the counting measure on each finite level.

Define the thin potentials by
$$
\Phi_{\mathcal K}^{\mathrm{thin}}(\tau_n):=K_\epsilon(\tau_n),
\qquad
\mathfrak D_{\mathcal K}^{\mathrm{thin}}(\tau_n):=d_s(\tau_n),
$$
where $d_s$ is the computational-depth proxy used by the algorithmic phase backend.

Take the invariance data to be the standard computable reindexing and padding symmetries of the chosen universal-machine presentation. The resulting thin kernel object is
$$
\mathcal T_{\mathcal K}^{\mathrm{thin}}
=
\bigl(
\mathcal X_{\mathcal K}^{\mathrm{thin}},
\Phi_{\mathcal K}^{\mathrm{thin}},
\mathfrak D_{\mathcal K}^{\mathrm{thin}},
G_{\mathcal K}^{\mathrm{thin}},
\partial_{\mathcal K}^{\mathrm{thin}}
\bigr),
$$
and the associated hypostructure is
$$
\mathbb H_{\mathcal K}:=\mathcal F(\mathcal T_{\mathcal K}^{\mathrm{thin}}).
$$

### 2. Computably Enumerable Witness

Run the standard dovetailing enumerator over all diagonal computations $\varphi_e(e)$. Whenever the computation halts, enumerate $e$ into $\mathcal K$.

This backend construction emits
$$
K_{\mathrm{CE}}^+
=
\bigl(
\text{dovetailing enumerator for }\mathcal K,
\ \mathcal K\in\mathrm{c.e.}
\bigr).
$$

### 3. Axiom R Failure Witness

The halting set is undecidable by the standard diagonal theorem of computability theory. In the algorithmic phase semantics used by {prf:ref}`thm-sieve-thermo-correspondence`, decidability is exactly the positive side of Axiom R.

Therefore the diagonal argument emits the witness certificate
$$
K_{\mathrm{AxiomR}}^{\mathrm{wit}}
=
\bigl(
\text{Turing diagonal witness},
\ \text{Axiom R fails for }\mathcal K
\bigr).
$$

---

## Part II: Sieve Execution

### 2.1 Core Certificates

From the instantiated thin object, the run emits the route-relevant thin certificates
$$
K_{D_E}^+,\ 
K_{\mathrm{Rec}_N}^+,\ 
K_{\mathrm{SC}_\lambda}^+,\ 
K_{\mathrm{SC}_{\partial c}}^+,\ 
K_{\mathrm{TB}_\pi}^+,\ 
K_{\mathrm{RepDesc}_K}^+,\ 
K_{\mathrm{Bound}_\partial}^-.
$$

The run also records the auxiliary off-route diagnostics
$$
K_{C_\mu}^{\mathrm{inc}},\ 
K_{\mathrm{Cap}_H}^{\mathrm{inc}},\ 
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},\ 
K_{\mathrm{TB}_O}^{\mathrm{inc}},\ 
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},\ 
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},
$$
which are not used in the designated goal route.

### 2.2 Backend-Derived Certificates

The algorithmic backend emits
$$
K_{\mathrm{CE}}^+
\qquad\text{and}\qquad
K_{\mathrm{AxiomR}}^{\mathrm{wit}}.
$$

No Lock certificate is required for the designated goal. The phase route closes before any categorical Hom-exclusion stage is invoked.

---

## Part II-B: Derivation of the Goal Certificate

### 1. Liquid-Phase Certificate

By {prf:ref}`thm-sieve-thermo-correspondence`, the Structural Sieve verdict for an algorithmic instance is determined by Axiom R status rather than by complexity alone. By {prf:ref}`def-algorithmic-phases`, a computably enumerable but undecidable set lies in the Liquid phase.

The certificates
$$
K_{\mathrm{CE}}^+
\wedge
K_{\mathrm{AxiomR}}^{\mathrm{wit}}
$$
therefore emit
$$
K_{\mathrm{Liquid}}^+
=
\bigl(
\mathcal K\in\mathrm{c.e.},
\ \text{Axiom R fails},
\ \text{phase}=\text{Liquid}
\bigr).
$$

### 2. Horizon Goal Certificate

The same phase theorem assigns the Sieve verdict `HORIZON` to every Liquid-phase problem. Hence
$$
K_{\mathrm{Liquid}}^+
\Longrightarrow
K^{\mathrm{hor}}.
$$

This is the designated goal certificate for the present proof object.

---

## Part III: Proof Completion

Take the designated goal certificate to be
$$
K_{\mathrm{Goal}}:=K^{\mathrm{hor}}.
$$

The route-relevant context is
$$
\Gamma_{\mathrm{req}}
=
\{
K_{D_E}^+,
K_{\mathrm{Rec}_N}^+,
K_{\mathrm{SC}_\lambda}^+,
K_{\mathrm{SC}_{\partial c}}^+,
K_{\mathrm{TB}_\pi}^+,
K_{\mathrm{RepDesc}_K}^+,
K_{\mathrm{Bound}_\partial}^-,
K_{\mathrm{CE}}^+,
K_{\mathrm{AxiomR}}^{\mathrm{wit}},
K_{\mathrm{Liquid}}^+,
K^{\mathrm{hor}}
\}.
$$

The only inconclusive certificates produced by the run are
$$
K_{C_\mu}^{\mathrm{inc}},\ 
K_{\mathrm{Cap}_H}^{\mathrm{inc}},\ 
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},\ 
K_{\mathrm{TB}_O}^{\mathrm{inc}},\ 
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},\ 
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}.
$$
None of them appears in the route
$$
K_{\mathrm{CE}}^+
\wedge
K_{\mathrm{AxiomR}}^{\mathrm{wit}}
\Longrightarrow
K_{\mathrm{Liquid}}^+
\Longrightarrow
K^{\mathrm{hor}}.
$$

Therefore
$$
\mathsf{Obl}\!\bigl(\mathrm{Cl}(\Gamma_{\mathrm{req}})\bigr)\cap \Downarrow(K_{\mathrm{Goal}})=\varnothing.
$$

The proof object is complete for the designated goal.

---

## Formal Proof

::::{prf:proof} Proof of Theorem {prf:ref}`thm-halting-problem`

Instantiate the thin algorithmic object $\mathcal T_{\mathcal K}^{\mathrm{thin}}$ from the prefix traces $\tau_n=\chi_{\mathcal K}\!\upharpoonright [0,n-1]$. The direct interface checks in Part 0 emit the route-relevant thin certificates
$$
K_{D_E}^+,\ 
K_{\mathrm{Rec}_N}^+,\ 
K_{\mathrm{SC}_\lambda}^+,\ 
K_{\mathrm{SC}_{\partial c}}^+,\ 
K_{\mathrm{TB}_\pi}^+,\ 
K_{\mathrm{RepDesc}_K}^+,\ 
K_{\mathrm{Bound}_\partial}^-,
$$
together with auxiliary off-route diagnostics.

The dovetailing enumerator of Part I emits
$$
K_{\mathrm{CE}}^+.
$$
The standard diagonal undecidability argument emits
$$
K_{\mathrm{AxiomR}}^{\mathrm{wit}},
$$
certifying that Axiom R fails for $\mathcal K$.

Apply {prf:ref}`thm-sieve-thermo-correspondence` and {prf:ref}`def-algorithmic-phases`. Since $\mathcal K$ is computably enumerable and Axiom R fails, the halting set lies in the Liquid phase. Hence the framework emits
$$
K_{\mathrm{Liquid}}^+.
$$
The same theorem assigns the horizon verdict to every Liquid-phase instance, so the designated goal certificate
$$
K^{\mathrm{hor}}
$$
is emitted.

Finally, Part III verifies that every inconclusive certificate produced by the run lies outside the dependency cone of $K^{\mathrm{hor}}$. Therefore the goal-cone obligation ledger is empty, and the theorem follows. ∎

::::

---

## Part IV: Final Certificate Chain

### 4.1 Validity Checklist

| Item | Status | Witness |
|---|---|---|
| All route-relevant nodes executed with explicit certificates | Yes | Parts II and IV.2 |
| Computably enumerable witness present | Yes | $K_{\mathrm{CE}}^+$ |
| Axiom R failure witness present | Yes | $K_{\mathrm{AxiomR}}^{\mathrm{wit}}$ |
| Liquid-phase certificate present | Yes | $K_{\mathrm{Liquid}}^+$ |
| Designated goal certificate reached | Yes | $K^{\mathrm{hor}}$ |
| Goal-relevant obligations discharged | Yes | Part III and IV.4 |
| Validity status | Unconditional proof for the designated goal | GOAL-CONE EMPTY |

### 4.2 Core Node Trace

| Node | Interface | Certificate | Status | Role in the designated goal route |
|---|---|---|---|---|
| 1 | $D_E$ | $K_{D_E}^+$ | Yes | Required |
| 2 | $\mathrm{Rec}_N$ | $K_{\mathrm{Rec}_N}^+$ | Yes | Required |
| 3 | $C_\mu$ | $K_{C_\mu}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 4 | $\mathrm{SC}_\lambda$ | $K_{\mathrm{SC}_\lambda}^+$ | Yes | Required |
| 5 | $\mathrm{SC}_{\partial c}$ | $K_{\mathrm{SC}_{\partial c}}^+$ | Yes | Required |
| 6 | $\mathrm{Cap}_H$ | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 7 | $\mathrm{LS}_\sigma$ | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 8 | $\mathrm{TB}_\pi$ | $K_{\mathrm{TB}_\pi}^+$ | Yes | Required |
| 9 | $\mathrm{TB}_O$ | $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 10 | $\mathrm{TB}_\rho$ | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 11 | $\mathrm{RepDesc}_K$ | $K_{\mathrm{RepDesc}_K}^+$ | Yes | Required |
| 12 | $\mathrm{GC}_\nabla$ | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | Inconclusive | Auxiliary only; outside main derivation |
| 13 | $\mathrm{Bound}_\partial$ | $K_{\mathrm{Bound}_\partial}^-$ | Closed | Routes directly to the algorithmic phase backend |

### 4.3 Backend and Goal Trace

| Stage | Certificate | Status | Source |
|---|---|---|---|
| Enumeration backend | $K_{\mathrm{CE}}^+$ | Yes | Part I.2 |
| Diagonal backend | $K_{\mathrm{AxiomR}}^{\mathrm{wit}}$ | Yes | Part I.3 |
| Phase classification | $K_{\mathrm{Liquid}}^+$ | Yes | Part II-B.1 |
| Goal | $K^{\mathrm{hor}}$ | Yes | Part II-B.2 |

### 4.4 Obligation Ledger Summary

| ID | Certificate | Obligation | In Goal Cone? | Status | Discharge / Reason |
|---|---|---|---|---|---|
| O1 | $K_{C_\mu}^{\mathrm{inc}}$ | supply a compactness/profile backend | No | Residual diagnostic | not used by the phase route |
| O2 | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | supply a capacity backend | No | Residual diagnostic | not used by the phase route |
| O3 | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | supply a stiffness backend | No | Residual diagnostic | not used by the phase route |
| O4 | $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ | supply a tameness backend | No | Residual diagnostic | not used by the phase route |
| O5 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | supply a mixing backend | No | Residual diagnostic | not used by the phase route |
| O6 | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | supply a gradient backend | No | Residual diagnostic | not used by the phase route |
