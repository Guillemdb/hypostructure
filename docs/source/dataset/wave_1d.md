# Global Regularity of the 1D Linear Wave Equation

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Global well-posedness, regularity propagation, and energy conservation for the 1D linear wave equation |
| **System Type** | $T_{\text{hyperbolic}}$ (linear hyperbolic PDE) |
| **Target Claim** | Global regularity for the 1D linear wave equation on $\mathbb R$ |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2026-04-15 |
| **Proof Mode** | Direct sieve execution + explicit D'Alembert backend package |
| **Completion Criterion** | $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$ |

### Label Naming Conventions

This instance uses the slug `wave-1d`.

| Type | Pattern | Example |
|------|---------|---------|
| Definitions | `def-wave-1d-*` | `def-wave-1d-arena` |
| Theorems | `thm-wave-1d-*` | `thm-wave-1d-main` |
| Proofs | `proof-wave-1d-*` | `proof-wave-1d-main` |
| Remarks | `rem-wave-1d-*` | `rem-wave-1d-chiral` |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{hyperbolic}}$ is a good type (finite stratification plus constructible caps).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** (Definition {prf:ref}`def-automation-guarantee`), hence profile extraction, admissibility, and surgery factories are available.
- **Scope note:** The automation witness discharges the factory layer only. The Lock certificate, D'Alembert bridge, and final regularity certificate are certified explicitly below.

**Certificate:**
$$
K_{\mathrm{Auto}}^+
=
\bigl(
T_{\text{hyperbolic}}\ \text{good},
\ \text{AutomationGuarantee holds},
\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, RESOLVE-AutoSurgery}
\bigr).
$$

---

## Abstract

This document presents a **machine-checkable proof object** for **global regularity of the 1D linear wave equation on $\mathbb R$** using the Hypostructure framework.

**Approach:** We instantiate the hyperbolic hypostructure on
$$
u_{tt}=c^2u_{xx},
\qquad c>0,
$$
with phase-space state
$$
(u,\dot u)\in H^1(\mathbb R)\times L^2(\mathbb R).
$$
The primary height is the conserved energy
$$
E(u,\dot u)=\tfrac12\int_{\mathbb R}\bigl(\dot u^2+c^2u_x^2\bigr)\,dx.
$$
The designated route uses finite energy, finite propagation speed, Fourier-faithful description, and the explicit D'Alembert characteristic decomposition into left- and right-traveling waves. The Lock is closed through the D'Alembert functional bridge rather than through any global Lyapunov or mixing package.

**Result:** The active route uses positive core certificates, a closed-system boundary branch, and a blocked Lock obtained by Tactic E5 (explicit functional bridge). Two diagnostic `inc` certificates are retained at the mixing and gradient nodes, but they are explicitly outside the dependency cone of the designated goal. The D'Alembert backend package upgrades structural exclusion to the final analytic regularity certificate
$$
K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+.
$$

---

## Theorem Statement

::::{prf:theorem} Global Regularity of the 1D Linear Wave Equation
:label: thm-wave-1d-main

**Given:**
- State space:
  $$
  \mathcal X
  =
  H^1(\mathbb R)\times L^2(\mathbb R).
  $$
- Dynamics:
  $$
  u_{tt}=c^2u_{xx},
  \qquad
  c>0.
  $$
- Initial data:
  $$
  u(0,\cdot)=u_0\in H^s(\mathbb R),
  \qquad
  u_t(0,\cdot)=u_1\in H^{s-1}(\mathbb R),
  \qquad
  s\ge1.
  $$

**Claim:** For every $s\ge1$, every $c>0$, and every $(u_0,u_1)\in H^s(\mathbb R)\times H^{s-1}(\mathbb R)$, there exists a unique global solution
$$
u\in C([0,\infty);H^s(\mathbb R))
\cap
C^1([0,\infty);H^{s-1}(\mathbb R)),
$$
the energy is conserved for all $t\ge0$, and the designated final certificate
$$
K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+
$$
is derivable from the hypostructure run.

**Designated Goal:**
$$
K_{\mathrm{Goal}}^+:=K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+.
$$

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\dot u$ | time derivative $u_t$ |
| $E(u,\dot u)$ | conserved energy $\frac12\int_{\mathbb R}(\dot u^2+c^2u_x^2)\,dx$ |
| $w_\pm$ | characteristic variables $w_\pm=\dot u\pm cu_x$ |
| $T_t^\pm$ | translation semigroups $(T_t^\pm f)(x)=f(x\mp ct)$ |
| $\Sigma$ | singular set candidate |
| $\mathcal H_{\mathrm{bad}}$ | finite-time Sobolev blow-up bad pattern |

::::

---

:::{dropdown} **LLM Execution Protocol** (Click to expand)

## Note for LLM Agents: Complete Algorithmic Protocol

This instance is executed as a deterministic proof-object construction.

### **A.1 Mindset Shift**

1. Fill each permit with explicit 1D linear wave data.
2. Emit exactly one certificate at every node.
3. Use only declared packages: energy conservation, characteristic transport, D'Alembert decomposition, and the wave backend regularity package.
4. Treat the Lock and the analytic upgrade as separate certified steps.
5. Keep non-goal diagnostics explicit; do not force them into the goal route.

### **A.2 Certificate Outcome Types**

| Outcome | Symbol | Used Here | Meaning |
|---------|--------|-----------|---------|
| YES | $K_X^+$ | Yes | gate verified |
| INC | $K_X^{\mathrm{inc}}$ | Yes | recorded diagnostic outside the goal cone |
| BLOCKED | $K_X^{\mathrm{blk}}$ | Yes | Lock verdict |
| BREACHED | $K_X^{\mathrm{br}}$ | No | no surgery route selected |

### **A.3 Inc Permit Protocol**

Two residual diagnostics are recorded:

- $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ because the free wave flow is transport-dispersive rather than mixing.
- $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ because the free wave flow is Hamiltonian rather than gradient.

Both lie outside $\Downarrow(K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+)$.

### **A.4 Upgrade Rule Execution**

No `inc` certificate is upgraded on the designated route. The only final promotion is
$$
K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+
\wedge
K_{\mathrm{WP}_{s_c}}^+
\Longrightarrow
K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+.
$$

This is the canonical continuation bridge notation used across PDE datasets: $K_{\mathrm{WP}_{s_c}}^+$ is the unified continuation certificate from the gate-evaluator schema (local well-posedness + uniqueness + continuation criterion + critical blow-up condition).


### **A.5 Breach Detection and Surgery**

No barrier breach occurs. No surgery is selected.

### **A.6 Obligation Tracking**

The goal-cone ledger is empty. Residual diagnostics are retained only for the non-goal mixing and gradient nodes.

### **A.7 Completion Criteria**

The proof object closes iff:

- all core nodes are executed;
- the closed-system branch is recorded at Node 13;
- Node 17 yields a certified Lock verdict;
- the explicit wave backend upgrade is present;
- no obligation remains in $\Downarrow(K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+)$.

### **A.8 Step-by-Step Implementation Guide for New Problems**

For this instance:

1. instantiate the 1D linear wave phase space and conserved energy;
2. execute Nodes 1-13 directly;
3. record the transport/mixing and Hamiltonian/gradient diagnostics as non-goal `inc` certificates;
4. close the Lock using the D'Alembert functional bridge;
5. apply the explicit wave backend upgrade.

:::

---

## **Part 0: Interface Permit Implementation Checklist**

### **0.1 Core Interface Permits (Nodes 1-12)**

#### **Template: $D_E$ (Energy Interface)**
- [x] **Height Functional $\Phi$:**
  $$
  \Phi(u,\dot u):=E(u,\dot u)=\tfrac12\int_{\mathbb R}(\dot u^2+c^2u_x^2)\,dx.
  $$
- [x] **Dissipation Rate $\mathfrak D$:**
  $$
  \mathfrak D(u,\dot u):=0.
  $$
- [x] **Energy Inequality:**
  $$
  E(u(t),u_t(t))=E(u_0,u_1).
  $$
- [x] **Bound Witness:** exact energy conservation.

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [x] **Bad Set $\mathcal B$:** finite-time Sobolev blow-up configurations.
- [x] **Recovery Map $\mathcal R$:** not needed on the designated route.
- [x] **Event Counter:** $N(T)=0$.
- [x] **Finiteness:** immediate from the empty-event route.

#### **Template: $C_\mu$ (Compactness Interface)**
- [x] **Symmetry Group $G$:** spatial translations of $\mathbb R$.
- [x] **Group Action $\rho$:** $(\rho_a u)(x)=u(x+a)$.
- [x] **Quotient Space:** characteristic profiles modulo translation.
- [x] **Concentration Measure:** D'Alembert decomposition splits the solution into two translation profiles with no concentration-creating nonlinear interaction.

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [x] **Scaling Action:** 
  $$
  u_\lambda(x,t)=u(\lambda x,\lambda t).
  $$
- [x] **Height Exponent $\alpha$:**
  $$
  E(u_\lambda,\partial_tu_\lambda)=\lambda E(u,\partial_tu),
  \qquad
  \alpha=1.
  $$
- [x] **Dissipation Exponent $\beta$:**
  $$
  \mathfrak D(u_\lambda,\partial_tu_\lambda)=0,
  $$
  recorded formally as the zero-cost branch and not used on the goal route.
- [x] **Criticality:** the designated $H^1\times L^2$ route lies above the scaling-critical index $s_c=\frac12$ for the displacement variable.

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [x] **Parameter Space $\Theta$:**
  $$
  \Theta=\{c\in(0,\infty)\}.
  $$
- [x] **Parameter Map:** $\theta(u,\dot u)=c$.
- [x] **Reference Point:** $c_0>0$ fixed by the equation.
- [x] **Stability Bound:** $c$ is constant in time.

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [x] **Capacity Functional:** Hausdorff capacity / codimension witness.
- [x] **Singular Set $\Sigma$:** finite-time singular set candidate.
- [x] **Codimension:** $\Sigma=\varnothing$ on the designated route.
- [x] **Capacity Bound:** $\mathrm{Cap}(\Sigma)=0$.

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [x] **Gradient Operator $\nabla$:** quadratic energy Hessian on the phase space.
- [x] **Critical Set $M$:** singleton equilibrium $M=\{(0,0)\}$.
- [x] **Łojasiewicz Exponent $\theta$:** $\theta=\frac12$ from the quadratic Hamiltonian.
- [x] **Łojasiewicz-Simon Inequality:** route-relative quadratic coercivity on the energy shell.

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [x] **Topological Invariant $\tau$:** connected phase-space component.
- [x] **Sector Classification:** single connected sector.
- [x] **Sector Preservation:** trivial under the linear flow.
- [x] **Tunneling Events:** none.

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [x] **O-minimal Structure $\mathcal O$:** semialgebraic / real-analytic structure on the linear phase space.
- [x] **Definability $\mathrm{Def}$:** the wave operator and zero singular set are definable in the route-relative formalization.
- [x] **Singular Set Tameness:** $\Sigma=\varnothing$.
- [x] **Cell Decomposition:** trivial linear stratification.

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [x] **Measure $\mathcal M$:** energy measure on phase-space data.
- [x] **Invariant Measure $\mu$:** conservative transport preserves energy-shell measures.
- [x] **Mixing Time $\tau_{\mathrm{mix}}$:** not certified on the free transport route.
- [x] **Mixing Property:** recorded as a non-goal diagnostic `inc`.

#### **Template: $\mathrm{RepDesc}_K$ (Dictionary / Description Interface)**
- [x] **Language $\mathcal L$:** Fourier and characteristic descriptions.
- [x] **Dictionary $D$:**
  $$
  (u,\dot u)\longleftrightarrow (w_+,w_-)
  \longleftrightarrow (\widehat u,\widehat{\dot u}).
  $$
- [x] **Complexity Measure $K$:** Sobolev order / Fourier-weighted norm.
- [x] **Faithfulness:** Fourier and characteristic data determine the solution uniquely.

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [x] **Metric Tensor $g$:** canonical phase-space metric on $H^1\times L^2$.
- [x] **Vector Field $v_{\mathrm{Wave}}$:** first-order wave flow
  $$
  \partial_t
  \binom{u}{\dot u}
  =
  \binom{\dot u}{c^2u_{xx}}.
  $$
- [x] **Gradient Compatibility:** not certified on the designated route.
- [x] **Monotonicity:** exact Hamiltonian conservation rather than gradient decay.

### **0.2 Boundary Interface Permits (Nodes 13-16)**

The spatial domain $\mathbb R$ yields the closed-system branch.

| Permit | Status | Note |
|---|---|---|
| $K_{\mathrm{Bound}_\partial}^-$ | Yes | no external boundary control |
| $K_{\mathrm{Bound}_B}$ | N/A | skipped after Node 13 |
| $K_{\mathrm{Bound}_{\Sigma}}$ | N/A | skipped after Node 13 |
| $K_{\mathrm{GC}_T}$ | N/A | skipped after Node 13 |

### **0.2b Derived Witness Certificates (Optional)**

No optional derived witness certificate is used on the designated route.

### **0.3 The Lock (Node 17)**

| Item | Value |
|---|---|
| Category | $\mathbf{Hypo}_{T_{\text{hyperbolic}}}$ |
| Universal bad object | finite-time Sobolev blow-up |
| Certified completeness package | present |
| Primary tactics | E5 (D'Alembert functional bridge) |
| Lock output | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |

### **0.3b Goal and Backend Certificates**

| Certificate | Status | Role |
|---|---|---|
| $K_{\mathrm{Germ}}^+$ | Yes | classifiable wave blow-up germ package |
| $K_{\mathrm{init}}^+$ | Yes | universal bad object package |
| $K_{\mathrm{CatLib}}^+$ | Yes | completeness of the finite bad-pattern library |
| $K_{\mathrm{DAlembert}}^+$ | Yes | explicit decomposition into left/right translation semigroups |
| $K_{\mathrm{WP}_{s_c}}^+$ | Yes | global well-posedness, uniqueness, Sobolev propagation, and finite propagation package |
| $K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+$ | derived | structural exclusion certificate |
| $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$ | derived | designated final analytic regularity certificate |

### **0.4 Implementation Templates**

#### **Template: $D_E$ (Energy Interface)**
- [x] conserved energy height chosen
- [x] zero-cost conservative branch recorded
- [x] exact conservation law recorded

#### **Template: Derived Witness Certificates (Optional)**
- [x] none used

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [x] bad set specified
- [x] empty-event route recorded

#### **Template: $C_\mu$ (Compactness Interface)**
- [x] translation symmetry fixed
- [x] characteristic profile decomposition recorded

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [x] wave scaling fixed
- [x] energy scaling recorded
- [x] designated Sobolev route recorded

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [x] wave speed parameter fixed
- [x] stability of $c$ recorded

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [x] singular set candidate declared
- [x] empty singular set route recorded

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [x] quadratic Hamiltonian coercivity recorded
- [x] route-relative stiffness package recorded

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [x] single sector identified
- [x] preservation of sector recorded

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [x] tame linear phase space recorded
- [x] trivial stratification recorded

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [x] conservative transport route recorded
- [x] non-goal diagnostic status recorded

#### **Template: $\mathrm{RepDesc}_K$ (Dictionary / Description Interface)**
- [x] Fourier/characteristic dictionary fixed
- [x] faithful finite-description route recorded

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [x] Hamiltonian first-order form recorded
- [x] non-goal diagnostic status recorded

#### **Template: $\mathrm{Cat}_{\mathrm{Hom}}$ (Lock Interface)**
- [x] category and bad object fixed
- [x] certified completeness package recorded
- [x] tactic E5 recorded
- [x] D'Alembert functional bridge recorded

:::{dropdown} **Part 0.5: Certificate Schemas and Upgrade Protocol** (Reference - Click to expand)

### **0.5.1 Certificate Schemas**

#### **Positive Certificate ($K_X^+$)**

Used throughout the route, for example
$$
K_{D_E}^+=(E,\mathfrak D=0,E(t)=E(0)).
$$

#### **NO-with-Witness Certificate ($K_X^{\mathrm{wit}}$)**

Not used on the designated route.

#### **NO-Inconclusive Certificate ($K_X^{\mathrm{inc}}$)**

The route records two non-goal diagnostics:

$$
K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
=
\left\{
\text{obligation: finite mixing certificate},
\text{missing: }[K_{\mathrm{Mix}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: free wave flow is transport, not mixing}
\right\},
$$

$$
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
=
\left\{
\text{obligation: gradient-flow representation},
\text{missing: }[K_{\mathrm{GradWave}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: free wave flow is Hamiltonian, not gradient}
\right\}.
$$

#### **Blocked Certificate ($K_X^{\mathrm{blk}}$)**

The Lock emits
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
=
\bigl(
\text{E5 D'Alembert functional bridge},
\{K_{D_E}^+,K_{\mathrm{RepDesc}_K}^+,K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+,K_{\mathrm{DAlembert}}^+\}
\bigr).
$$

#### **Breached Certificate ($K_X^{\mathrm{br}}$)**

Not used on the designated route.

### **0.5.2 Upgrade Rule Schema**

No goal-relevant `inc` certificate is upgraded on the designated route.

#### **Rule Template**

The only final upgrade used here is
$$
K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+
\wedge
K_{\mathrm{WP}_{s_c}}^+
\Longrightarrow
K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+.
$$

#### **Non-Circularity Guard**

$K_{\mathrm{WP}_{s_c}}^+$ is an external backend package and is not derived from $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$, so the upgrade is non-circular.

#### **Upgrade Types**

| Type | Used Here | Source |
|------|-----------|--------|
| Instantaneous | No | none |
| A-posteriori | Yes | backend analytic promotion after the Lock |

### **0.5.2b Promotion Permits (Blocked → YES$^\sim$)**

No blocked-to-YES$^\sim$ promotion is used. The Lock remains explicitly blocked and is mined in Part III-B for the structural exclusion certificate.

### **0.5.3 Surgery Certificate Schema**

No surgery certificate is used on the designated route.

### **0.5.4 Re-entry Certificate Schema**

No re-entry certificate is used on the designated route.

### **0.5.5 Context Accumulation**

The route context accumulates:
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
K_{\mathrm{DAlembert}}^+,
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
\}.
$$

:::

---

## **Part I: The Instantiation (Thin Object Definitions)**

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**

- **State Space ($\mathcal X$):** $H^1(\mathbb R)\times L^2(\mathbb R)$ for $(u,\dot u)$.
- **Metric ($d$):** energy metric induced by $\|u_x\|_{L^2}^2+\|\dot u\|_{L^2}^2$.
- **Measure ($\mu$):** Lebesgue measure on physical space together with the induced phase-space energy shell structure.
- **Characteristic Split:** left/right traveling components encoded by $(w_+,w_-)$.

### **2. The Potential ($\Phi^{\text{thin}}$)**

- **Height Functional ($\Phi$):**
  $$
  \Phi(u,\dot u)=E(u,\dot u)=\tfrac12\int_{\mathbb R}(\dot u^2+c^2u_x^2)\,dx.
  $$
- **Characteristic Energy:**
  $$
  E=\frac14\int_{\mathbb R}(w_+^2+w_-^2)\,dx.
  $$
- **Equilibrium Set:** the singleton equilibrium $\{(0,0)\}$.
- **Scaling Exponent ($\alpha$):** $\alpha=1$.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**

- **Conservative Cost Branch:**
  $$
  \mathfrak D(u,\dot u)=0.
  $$
- **Dynamics:** first-order Hamiltonian form
  $$
  \partial_t
  \binom{u}{\dot u}
  =
  \binom{\dot u}{c^2u_{xx}}.
  $$
- **Characteristic Transport:** $w_\pm$ satisfy
  $$
  \partial_t w_\pm \mp c\,\partial_x w_\pm = 0.
  $$

### **4. The Invariance ($G^{\text{thin}}$)**

- **Symmetry Group ($\mathrm{Grp}$):** spatial translations, time translations, and reflections.
- **Scaling ($\mathcal S$):** wave scaling on the homogeneous model.
- **Conserved Quantity:** total energy $E$.
- **Auxiliary Reconstruction:** D'Alembert formula from characteristic data.

---

## **Part II: Sieve Execution (Verification Run)**

### **EXECUTION PROTOCOL**

The designated route executes Nodes 1-13 directly, skips Nodes 14-16 on the closed-system branch, and then executes the Lock at Node 17. Two diagnostic `inc` certificates are recorded at Nodes 10 and 12, but they are excluded from the designated goal dependency cone.

### **Level 1: Conservation**

#### **Node 1: EnergyCheck ($D_E$)**

**Question:** Is the energy well-defined and bounded along trajectories?

**Step-by-step execution:**
1. [x] For $(u_0,u_1)\in H^1\times L^2$, the energy is finite at $t=0$.
2. [x] Differentiate the energy and integrate by parts.
3. [x] Use the wave equation $u_{tt}=c^2u_{xx}$ to obtain
   $$
   \frac{d}{dt}E(u(t),u_t(t))=0.
   $$
4. [x] Therefore $E(t)=E(0)$ for all $t\ge0$.

**Certificate:**
$$
K_{D_E}^+=(E,\mathfrak D=0,E(t)=E(0)).
$$

#### **Node 2: ZenoCheck ($\mathrm{Rec}_N$)**

**Question:** Are recovery events finite?

**Step-by-step execution:**
1. [x] The linear wave flow introduces no repair or restart event.
2. [x] The event counter is identically zero.
3. [x] D'Alembert evolution is continuous in time on the phase space.

**Certificate:**
$$
K_{\mathrm{Rec}_N}^+=(N(T)=0,\text{empty-event route}).
$$

#### **Node 3: CompactCheck ($C_\mu$)**

**Question:** Does the route exhibit compactness modulo the tracked symmetry?

**Step-by-step execution:**
1. [x] Rewrite the flow in characteristic variables $w_\pm=\dot u\pm cu_x$.
2. [x] Each characteristic component evolves by pure translation.
3. [x] No nonlinear interaction creates concentrating profile cascades.
4. [x] The route is represented by two transported profiles modulo translation.

**Certificate:**
$$
K_{C_\mu}^+=(G=\mathbb R,\text{characteristic transport decomposition}).
$$

### **Level 2: Duality & Symmetry**

#### **Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)**

**Question:** Is the designated Sobolev route scaling-subcritical?

**Step-by-step execution:**
1. [x] Use $u_\lambda(x,t)=u(\lambda x,\lambda t)$.
2. [x] The energy scales by one power of $\lambda$.
3. [x] The displacement-critical index is $s_c=\frac12$ in one space dimension.
4. [x] The designated route uses $s\ge1>s_c$.

**Certificate:**
$$
K_{\mathrm{SC}_\lambda}^+=(\alpha=1,s\ge1>s_c=\tfrac12).
$$

#### **Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)**

**Question:** Are system parameters stable?

**Step-by-step execution:**
1. [x] The only parameter is the wave speed $c>0$.
2. [x] The equation is autonomous.
3. [x] The parameter is constant in time.

**Certificate:**
$$
K_{\mathrm{SC}_{\partial c}}^+=(c=\text{const},\text{autonomous parameter sector}).
$$

### **Level 3: Geometry & Stiffness**

#### **Node 6: GeomCheck ($\mathrm{Cap}_H$)**

**Question:** Is the route-relative singular set empty?

**Step-by-step execution:**
1. [x] The designated route targets finite-time Sobolev blow-up.
2. [x] The explicit characteristic transport formula preserves the Sobolev class.
3. [x] Hence the route-relative singular set candidate is empty.

**Certificate:**
$$
K_{\mathrm{Cap}_H}^+=(\Sigma=\varnothing,\mathrm{Cap}(\Sigma)=0).
$$

#### **Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)**

**Question:** Is there a route-relative stiffness certificate?

**Step-by-step execution:**
1. [x] The wave Hamiltonian is quadratic on $H^1\times L^2$.
2. [x] The zero-energy equilibrium is isolated on the energy shell.
3. [x] The Hessian of the energy is coercive on the phase-space norm.
4. [x] This supplies the route-relative quadratic stiffness witness.

**Certificate:**
$$
K_{\mathrm{LS}_\sigma}^+=(\theta=\tfrac12,\text{quadratic Hamiltonian coercivity}).
$$

### **Level 4: Topology**

#### **Node 8: TopoCheck ($\mathrm{TB}_\pi$)**

**Question:** Is the sector preserved?

**Step-by-step execution:**
1. [x] The phase space $H^1(\mathbb R)\times L^2(\mathbb R)$ is connected.
2. [x] The linear flow preserves this connected sector.
3. [x] No topological tunneling event occurs.

**Certificate:**
$$
K_{\mathrm{TB}_\pi}^+=(\text{single connected sector}).
$$

#### **Node 9: TameCheck ($\mathrm{TB}_O$)**

**Question:** Is the topology tame?

**Step-by-step execution:**
1. [x] The phase space is linear.
2. [x] The singular set candidate is empty on the designated route.
3. [x] The route admits trivial linear stratification.

**Certificate:**
$$
K_{\mathrm{TB}_O}^+=(\mathcal O,\Sigma=\varnothing,\text{linear tame route}).
$$

### **Level 5: Mixing**

#### **Node 10: ErgoCheck ($\mathrm{TB}_\rho$)**

**Question:** Does the flow mix?

**Step-by-step execution:**
1. [x] The free wave flow transports characteristic data without dissipation.
2. [x] No finite mixing-time certificate is produced on this route.
3. [x] This diagnostic is not used in the designated goal chain.

**Certificate:**
$$
K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
=
\left\{
\text{obligation: finite mixing certificate},
\text{missing: }[K_{\mathrm{Mix}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: conservative transport, not mixing}
\right\}.
$$

### **Level 6: Complexity**

#### **Node 11: ComplexCheck ($\mathrm{RepDesc}_K$)**

**Question:** Is the description finite and faithful?

**Step-by-step execution:**
1. [x] Fourier data determine the linear wave solution uniquely.
2. [x] Characteristic variables $(w_+,w_-)$ also determine the solution uniquely.
3. [x] The two representations are equivalent on the route.

**Certificate:**
$$
K_{\mathrm{RepDesc}_K}^+=(\mathcal L,D,K,\text{faithful}).
$$

#### **Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)**

**Question:** Is the flow gradient-compatible?

**Step-by-step execution:**
1. [x] The wave flow is Hamiltonian in first-order form.
2. [x] The route provides exact energy conservation, not gradient decay.
3. [x] No gradient-flow representation is needed for the designated goal.

**Certificate:**
$$
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
=
\left\{
\text{obligation: gradient-flow representation},
\text{missing: }[K_{\mathrm{GradWave}}^+],
\text{failure\_code: ROUTE\_DIAGNOSTIC},
\text{trace: Hamiltonian transport replaces gradient descent}
\right\}.
$$

### **Level 7: Boundary (Open Systems)**

#### **Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)**

**Question:** Is the system open?

**Step-by-step execution:**
1. [x] The spatial domain is $\mathbb R$ with no external control input.
2. [x] There are no boundary maps $\iota,\pi$ in the route.
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
1. [x] The bad-pattern library consists of the finite-time Sobolev blow-up template.
2. [x] The certified completeness package $(K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+)$ is present.
3. [x] Apply the D'Alembert map to characteristic variables:
   $$
   w_\pm(x,t)=w_\pm(x\mp ct,0).
   $$
4. [x] Characteristic transport is pure translation and preserves all Sobolev norms.
5. [x] Therefore a bad morphism from finite-time blow-up into the actual route would induce impossible norm blow-up under a translation semigroup.
6. [x] This is excluded by the explicit functional bridge.

**Lock Verdict:**
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
=
\bigl(
\text{E5 D'Alembert functional bridge},
\{K_{D_E}^+,K_{\mathrm{RepDesc}_K}^+,K_{\mathrm{Germ}}^+,K_{\mathrm{init}}^+,K_{\mathrm{CatLib}}^+,K_{\mathrm{DAlembert}}^+\}
\bigr).
$$

---

## **Part II-B: Upgrade Pass**

### **Upgrade Pass Protocol**

No goal-relevant `inc` certificate is introduced.

| ID | Node | Obligation | Missing | In Goal Cone? |
|----|------|------------|---------|---------------|
| OBL-1 | 10 | finite mixing certificate | $K_{\mathrm{Mix}}^+$ | No |
| OBL-2 | 12 | gradient-flow representation | $K_{\mathrm{GradWave}}^+$ | No |

No upgrade is required before the Lock. The final analytic promotion is handled in Part III-B as a backend theorem application.

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

The designated route does not invoke KRNL-Lyapunov reconstruction. The goal closes through the explicit D'Alembert bridge and the backend regularity package rather than through a dissipative Lyapunov chain.

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

- **Structural Exclusion Theorem:** from the blocked Lock and the certified completeness package,
  $$
  K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+.
  $$
  Statement: the finite-time Sobolev blow-up bad pattern does not embed into the 1D linear wave flow.

- **Analytic Global Regularity Theorem:** from structural exclusion plus the explicit wave backend package,
  $$
  K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+
  \wedge K_{\mathrm{WP}_{s_c}}^+
  \Longrightarrow
  K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+.
  $$
  Statement: global existence, uniqueness, Sobolev regularity propagation, and finite propagation speed hold for all time.

- **Scattering / Backend Analytic Upgrade:** implicit in the D'Alembert transport package.
- **Observer-Relative Censorship Theorem:** not used.
- **Singularity Classification:** the only route-relevant profile family consists of left- and right-traveling free waves.

### **3.2 Quantitative Bounds**

- **Energy conservation:**
  $$
  E(t)=E(0).
  $$
- **Sobolev propagation:**
  $$
  \|(u(t),u_t(t))\|_{H^s\times H^{s-1}}
  \lesssim_c
  \|(u_0,u_1)\|_{H^s\times H^{s-1}}.
  $$
- **Finite propagation speed:** support propagates inside the light cone of speed $c$.

### **3.3 Functional Objects**

- **D'Alembert bridge:** $K_{\mathrm{DAlembert}}^+$.
- **Characteristic transport package:** $w_\pm(x,t)=w_\pm(x\mp ct,0)$.
- **Backend regularity package:** $K_{\mathrm{WP}_{s_c}}^+$.

### **3.4 Retroactive Upgrades**

- No goal-relevant `inc` certificate required discharge.
- The two residual diagnostics remain outside the goal cone.
- Final analytic regularity is upgraded from structural exclusion by the declared backend package.

### **3.5 ZFC Proof Export (Appendix Bridge)**

Not requested. The proof object stops at the certified analytic regularity certificate.

---

## **Part III-C: Obligation Ledger**

### **Introduced Obligations**

| ID | Node | Certificate | Obligation | Missing | In Goal Cone? | Status |
|----|------|-------------|------------|---------|---------------|--------|
| OBL-1 | 10 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | finite mixing certificate | $K_{\mathrm{Mix}}^+$ | No | Residual diagnostic |
| OBL-2 | 12 | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | gradient-flow representation | $K_{\mathrm{GradWave}}^+$ | No | Residual diagnostic |

### **Discharge Events**

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| — | — | — | — |

### **Remaining Obligations**

**Count:** 2

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| OBL-1 | finite mixing certificate | conservative transport route does not require mixing |
| OBL-2 | gradient-flow representation | Hamiltonian route does not require gradient structure |

### **Ledger Validation**

- [x] All goal-relevant `inc` certificates upgraded or absent.
- [x] All goal-relevant breach obligations discharged or absent.
- [x] The remaining obligations are explicitly outside the designated goal dependency cone.

**Ledger Status:** GOAL-CONE EMPTY for $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$ with two residual non-goal diagnostics.

---

## **Part IV: Final Certificate Chain**

### **4.1 Validity Checklist**

- [x] **All 12 core nodes executed**
- [x] **Boundary nodes handled correctly** (closed-system branch)
- [x] **Lock executed**
- [x] **Lock verdict obtained:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- [x] **Designated goal certificate reached:** $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$
- [x] **If claiming structural exclusion:** certified completeness package is present
- [x] **If claiming analytic regularity through structural exclusion:** backend analytic package is present
- [x] **Upgrade pass completed**
- [x] **Surgery/Re-entry completed if needed:** not needed
- [x] **No unresolved obligations remain in the designated goal dependency cone**

**Validity Status:** UNCONDITIONAL PROOF for the designated goal $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$.

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
Node 17: K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
Part III-A: not invoked on designated route
Part III-B: K_{\mathrm{Germ}}^+, K_{\mathrm{init}}^+, K_{\mathrm{CatLib}}^+, K_{\mathrm{DAlembert}}^+, K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+, K_{\mathrm{WP}_{s_c}}^+ -> K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+
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
K_{\mathrm{DAlembert}}^+,
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}},
K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+,
K_{\mathrm{WP}_{s_c}}^+,
K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+
\}.
$$

### **4.4 Conclusion**

**Conclusion:** The designated target claim is **ESTABLISHED**. The 1D linear wave equation admits a complete template-level proof object whose final analytic certificate is $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$.

---

## **Formal Proof**

::::{prf:proof} Proof of Theorem {prf:ref}`thm-wave-1d-main`
:label: proof-wave-1d-main

The proof proceeds by structural sieve analysis in seven phases.

**Phase 1 (Instantiation):** Part I defines the 1D wave thin objects $(\mathcal X,\Phi,\mathfrak D,G)$ on the phase space $H^1(\mathbb R)\times L^2(\mathbb R)$.

**Phase 2 (Conservation):** Nodes 1-3 produce $K_{D_E}^+$, $K_{\mathrm{Rec}_N}^+$, and $K_{C_\mu}^+$, certifying finite conserved energy, zero repair-event count, and the characteristic transport decomposition.

**Phase 3 (Scaling):** Nodes 4-5 produce $K_{\mathrm{SC}_\lambda}^+$ and $K_{\mathrm{SC}_{\partial c}}^+$, recording the wave scaling and the constant parameter sector $c>0$.

**Phase 4 (Geometry):** Nodes 6-9 produce the geometric, stiffness, topological, and tame certificates required on the designated route.

**Phase 5 (Diagnostics):** Nodes 10 and 12 emit $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ and $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$, but Part III-C records that both obligations are outside the dependency cone of the designated goal. Node 11 supplies the faithful Fourier/characteristic description certificate.

**Phase 6 (Boundary):** Node 13 records the closed-system branch, so Nodes 14-16 are not applicable.

**Phase 7 (Lock / Backend Upgrade):** Node 17 blocks the bad pattern via $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ using E5 with the certified completeness package and the explicit D'Alembert bridge. Part III-B combines the blocked structural certificate with $K_{\mathrm{WP}_{s_c}}^+$ to derive the final analytic regularity certificate $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$.

Therefore the designated goal certificate is established and the residual diagnostics do not obstruct it because they lie outside $\Downarrow(K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+)$.
$$
\therefore K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+ \quad \square
$$

::::

---

## **Verification Summary**

| Component | Status | Certificate |
|-----------|--------|-------------|
| Nodes 1-12 (Core) | PASS / DIAGNOSTIC | positive route with two non-goal `inc` diagnostics |
| Nodes 13-16 (Boundary) | N/A / PASS | closed-system branch via $K_{\mathrm{Bound}_\partial}^-$ |
| Node 17 (Lock) | BLOCKED | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ |
| Goal Certificate | REACHED | $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$ |
| Obligation Ledger | GOAL-CONE EMPTY | residual diagnostics only |
| Upgrade Pass | COMPLETE | backend analytic promotion only |

**Final Verdict:** UNCONDITIONAL proof object for the designated goal.

---

## **References**

1. Hypostructure Framework v1.0 formalism.
2. D'Alembert formula and characteristic decomposition for the 1D wave equation.
3. Standard Sobolev well-posedness and regularity propagation for the linear wave equation on $\mathbb R$.
4. Finite propagation speed and energy conservation for hyperbolic equations.

---

## Appendix: Replay Bundle (Machine-Checkability)

```json
{
  "problem": "wave-1d",
  "goal": "K_Reg_Wave1D^+",
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
    "K_DAlembert^+",
    "K_CatHom^blk",
    "K_StructReg_Wave1D^+",
    "K_{WP_{s_c}}^+",
    "K_Reg_Wave1D^+"
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

---

## Executive Summary: The Proof Dashboard

### 1. System Instantiation (The Physics)

| Object | Definition | Role |
|---|---|---|
| **Arena ($\mathcal X$)** | $H^1(\mathbb R)\times L^2(\mathbb R)$ | phase space |
| **Potential ($\Phi$)** | conserved wave energy $E(u,\dot u)$ | primary height |
| **Cost ($\mathfrak D$)** | zero-cost conservative branch | conservative structure |
| **Invariance ($G$)** | translations, reflections, and wave scaling | symmetry sector |

### 2. Execution Trace (The Logic)

| Node | Check | Outcome | Certificate Payload | Ledger State |
|---|---|---:|---|---|
| 1 | Energy Bound | YES | exact energy conservation | `[]` |
| 2 | Zeno Check | YES | no repair events | `[]` |
| 3 | Compact Check | YES | characteristic transport decomposition | `[]` |
| 4 | Scale Check | YES | $s\ge1>s_c$ | `[]` |
| 5 | Param Check | YES | $c$ constant | `[]` |
| 6 | Geom Check | YES | $\Sigma=\varnothing$ | `[]` |
| 7 | Stiffness Check | YES | quadratic Hamiltonian coercivity | `[]` |
| 8 | Topo Check | YES | single connected sector | `[]` |
| 9 | Tame Check | YES | linear tame route | `[]` |
| 10 | Ergo Check | INC | transport, not mixing | `[OBL-1]` |
| 11 | Complex Check | YES | Fourier/characteristic faithful | `[OBL-1]` |
| 12 | Oscillate Check | INC | Hamiltonian, not gradient | `[OBL-1, OBL-2]` |
| 13 | Boundary Check | CLOSED | no external boundary | `[OBL-1, OBL-2]` |
| 17 | LOCK | BLOCK | E5 D'Alembert bridge | `[OBL-1, OBL-2]` |

### 3. Lock Mechanism (The Exclusion)

| Tactic | Description | Status | Reason / Mechanism |
|---|---|---:|---|
| E1 | Dimension | N/A | not used |
| E2 | Invariant | N/A | not used |
| E3 | Positivity | N/A | not used |
| E4 | Integrality | N/A | not used |
| E5 | Functional | PASS | D'Alembert map reduces the flow to norm-preserving translations |
| E6 | Causal | N/A | not used |
| E7 | Thermodynamic | N/A | not used |
| E8 | Holographic | N/A | not used |
| E9 | Ergodic | N/A | not used |
| E10 | Definability | N/A | not used |
| E11 | Galois-Monodromy | N/A | not used |
| E12 | Algebraic Compressibility | N/A | not used |
| E13 | Algorithmic Completeness | N/A | split semantics are explicit in Lock: complete finite-library closure gives $K_{\mathrm{E13}}^{\mathrm{blk}}$; incomplete closure gives $K_{\mathrm{E13}}^{\mathrm{br-inc}}$ and routes to reconstruction. |

### 4. Final Verdict

### Assumption Provenance

- **Imported from literature?** yes
- **Theorem name(s):** 1D linear wave explicit solution representation and finite speed of propagation with energy conservation
- **Hypotheses required:** smooth initial data in the declared energy class and compatible boundary/domain structure
- **Non-circularity note:** $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$ is deduced from Lock and energy certificates; it is not used as an assumed premise.
- **Goal-certificate location in local-to-global chain:** Part III-B proves $K_{\mathrm{StructReg}_{\mathrm{Wave1D}}}^+ \wedge K_{\mathrm{WP}_{s_c}}^+ \Rightarrow K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$.

- **Designated Goal Certificate:** $K_{\mathrm{Reg}_{\mathrm{Wave1D}}}^+$
- **Status:** UNCONDITIONAL
- **Goal-Cone Ledger:** EMPTY
- **Residual Non-Goal Obligations:** `OBL-1`, `OBL-2`
- **Singularity Set:** $\Sigma=\varnothing$
- **Primary Final Route:** direct sieve execution + E5-blocked Lock + explicit wave backend upgrade

---

## Document Information

| Field | Value |
|-------|-------|
| **Document Type** | Proof Object |
| **Framework** | Hypostructure v1.0 |
| **Problem Class** | Classical hyperbolic PDE |
| **Problem Type** | Solved regularity instance |
| **System Type** | $T_{\text{hyperbolic}}$ |
| **Singularity Type** | `REGULAR` |
| **Verification Level** | Machine-checkable proof object |
| **Inc Certificates** | 2 introduced, both outside the goal cone |
| **Final Status** | UNCONDITIONAL |
| **Generated** | 2026-04-15 |
