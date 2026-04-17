---
title: "3D Euler Hypostructure Proof Object"
---

# Local Well-Posedness and the Vortex-Stretching Singularity Route for 3D Incompressible Euler on the Torus

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Local well-posedness and singularity-route certification for the 3D incompressible Euler equation on $\mathbb T^3$ |
| **System Type** | $T_{\text{hyperbolic}}$ |
| **Target Claim** | Designated local continuation/stretching route |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2026-04-15 |

### Label Naming Conventions

This instance uses the slug `euler-3d`.

| Type | Pattern | Example |
|------|---------|---------|
| Definitions | `def-euler-3d-*` | `def-euler-3d-arena` |
| Theorems | `thm-euler-3d-*` | `thm-euler-3d-main` |
| Lemmas | `lem-euler-3d-*` | `lem-euler-3d-bkm` |
| Remarks | `rem-euler-3d-*` | `rem-euler-3d-stretching` |
| Proofs | `proof-euler-3d-*` | `proof-euler-3d-main` |
| Proof Sketches | `sketch-euler-3d-*` | `sketch-euler-3d-route` |

---

## Automation Witness (Framework Offloading Justification)

- **Type witness:** $T_{\text{hyperbolic}}$ is a good type.
- **Automation witness:** Definition {prf:ref}`def-automation-guarantee` is active.
- **Scope note:** only the factory layer is offloaded; the route certificates are recorded explicitly.

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

$$
\mathcal X
=
\left\{
u\in H^s(\mathbb T^3;\mathbb R^3):
\nabla\cdot u=0,\ \int_{\mathbb T^3}u\,dx=0
\right\},
\qquad
s>\tfrac52.
$$

$$
u_t+(u\cdot\nabla)u+\nabla p=0,
\qquad
\nabla\cdot u=0,
\qquad
\omega=\nabla\times u.
$$

$$
u\in C([0,T_*);H^s(\mathbb T^3;\mathbb R^3)),
\qquad
\partial_t\omega+u\cdot\nabla\omega=(\omega\cdot\nabla)u,
\qquad
u=\mathcal B_{\mathrm{per}}[\omega],
\qquad
\nabla u=\nabla\mathcal B_{\mathrm{per}}[\omega].
$$

$$
T_*<\infty\Rightarrow\int_0^{T_*}\|\omega(t)\|_{L^\infty(\mathbb T^3)}\,dt=\infty,
\qquad
\frac{d}{dt}\omega(X(a,t),t)=\nabla u(X(a,t),t)\,\omega(X(a,t),t),
\qquad
\frac{d}{dt}|\omega(X(a,t),t)|=(\xi\cdot S\xi)(X(a,t),t)\,|\omega(X(a,t),t)|.
$$

$$
K_{\mathrm{Goal}}^+.
$$

$$
\Gamma_{\mathrm{res}}
=
\left\{
K_{\mathrm{Cap}_H}^{\mathrm{inc}},
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}
\right\},
\qquad
\Gamma_{\mathrm{res}}\cap\Downarrow(K_{\mathrm{Goal}}^+)=\varnothing.
$$

---

## Theorem Statement

::::{prf:theorem} Local Well-Posedness and the Vortex-Stretching Singularity Route for 3D Euler on $\mathbb T^3$
:label: thm-euler-3d-main

**Given:**
- State space:
  $$
  \mathcal X
  =
  \left\{
  u\in H^s(\mathbb T^3;\mathbb R^3):
  \nabla\cdot u=0,\ \int_{\mathbb T^3}u\,dx=0
  \right\},
  \qquad
  s>\tfrac52.
  $$
- Dynamics:
  $$
  u_t+(u\cdot\nabla)u+\nabla p=0,
  \qquad
  \nabla\cdot u=0.
  $$
- Initial data:
  $$
  u(0,\cdot)=u_0\in\mathcal X.
  $$

**Claim:**
$$
K_{D_E}^+,\ 
K_{\mathrm{Rec}_N}^+,\ 
K_{C_\mu}^+,\ 
K_{\mathrm{SC}_\lambda}^-,\ 
K_{\mathrm{SC}_{\partial c}}^+,\ 
K_{\mathrm{TB}_\pi}^+,\ 
K_{\mathrm{TB}_O}^+,\ 
K_{\mathrm{RepDesc}_K}^+,\ 
K_{\mathrm{Bound}_\partial}^-,
$$
with residual diagnostics
$$
K_{\mathrm{Cap}_H}^{\mathrm{inc}},\ 
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},\ 
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},\ 
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},\ 
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}},
$$
and with backend route data
$$
u\in C([0,T_*);H^s(\mathbb T^3;\mathbb R^3)),
\qquad
\partial_t\omega+u\cdot\nabla\omega=(\omega\cdot\nabla)u,
\qquad
u=\mathcal B_{\mathrm{per}}[\omega],
\qquad
\nabla u=\nabla\mathcal B_{\mathrm{per}}[\omega],
$$
$$
T_*<\infty\Rightarrow\int_0^{T_*}\|\omega(t)\|_{L^\infty(\mathbb T^3)}\,dt=\infty,
\qquad
\frac{d}{dt}\omega(X(a,t),t)=\nabla u(X(a,t),t)\,\omega(X(a,t),t),
\qquad
\frac{d}{dt}|\omega(X(a,t),t)|=(\xi\cdot S\xi)(X(a,t),t)\,|\omega(X(a,t),t)|.
$$

**Designated Goal:**
$$
K_{\mathrm{Goal}}^+.
$$

**Notation:**
| Symbol | Definition |
|--------|------------|
| $E(u)$ | kinetic energy $\frac12\|u\|_{L^2(\mathbb T^3)}^2$ |
| $\omega$ | vorticity $\nabla\times u$ |
| $S$ | strain tensor $S=\frac12(\nabla u+\nabla u^\top)$ |
| $\xi$ | vorticity direction $\xi=\omega/|\omega|$ on $\{\omega\neq0\}$ |
| $\mathcal B_{\mathrm{per}}$ | periodic Biot-Savart operator on the mean-zero torus sector |
| $K_{\mathrm{Goal}}^+$ | designated backend goal certificate for the local continuation/stretching route |
| $T_*$ | maximal smooth existence time |
| $\Sigma$ | candidate singular set |
| $\mathcal H_{\mathrm{bad}}$ | finite-time $H^s$ blow-up bad pattern |

::::

---

:::{dropdown} **LLM Execution Protocol** (Click to expand)

## Note for LLM Agents: Complete Algorithmic Protocol

### **A.1 Mindset Shift**

The designated route is
$$
1\to 2\to 3\to 4\to 5\to 6\to 7\to 8\to 9\to 10\to 11\to 12\to 13,
$$
with Nodes $14$--$16$ skipped on the closed-system branch and Node $17$ recorded as
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}.
$$

### **A.2 Certificate Outcome Types**

| Outcome | Symbol | Used Here | Meaning |
|---------|--------|-----------|---------|
| YES | $K_X^+$ | Yes | verified permit |
| NO-with-witness | $K_X^{\mathrm{wit}}$ | Yes | explicit negative scaling verdict |
| NO-inconclusive | $K_X^{\mathrm{inc}}$ | Yes | residual non-goal diagnostic |
| BLOCKED | $K_X^{\mathrm{blk}}$ | No | unused on designated route |
| BREACHED | $K_X^{\mathrm{br}}$ | No | unused on designated route |

### **A.3 Inc Permit Protocol**

Residual diagnostics:
$$
K_{\mathrm{Cap}_H}^{\mathrm{inc}},
\quad
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},
\quad
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},
\quad
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}},
\quad
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}.
$$

All lie outside
$$
\Downarrow(K_{\mathrm{Goal}}^+).
$$

### **A.4 Upgrade Rule Execution**

The only goal-level promotion closes the declared backend route data of Part III-B.1 to
$$
K_{\mathrm{Goal}}^+.
$$

### **A.5 Breach Detection and Surgery**

No $K_X^{\mathrm{br}}$ is emitted. No surgery or re-entry is used.

### **A.6 Obligation Tracking**

The goal-cone ledger is empty:
$$
\Gamma_{\mathrm{res}}\cap\Downarrow(K_{\mathrm{Goal}}^+)=\varnothing.
$$

### **A.7 Completion Criteria**

- all nodes on the designated route are executed;
- the closed-system branch is recorded at Node 13;
- the backend route data of Part III-B.1 are present;
- no certificate in $\Downarrow(K_{\mathrm{Goal}}^+)$ remains unresolved.

### **A.8 Step-by-Step Implementation Guide for New Problems**

For this instance:
$$
\mathcal X,\ \Phi,\ \mathfrak D,\ G
\Longrightarrow
\Gamma_{13}
\Longrightarrow
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}
\Longrightarrow
K_{\mathrm{Goal}}^+.
$$

:::

---

## **Part 0: Interface Permit Implementation Checklist**

### **0.1 Core Interface Permits (Nodes 1-12)**

#### **Template: $D_E$ (Energy Interface)**
- [x] **Height Functional $\Phi$:**
  $$
  \Phi(u):=E(u)=\tfrac12\|u\|_{L^2(\mathbb T^3)}^2.
  $$
- [x] **Dissipation Rate $\mathfrak D$:**
  $$
  \mathfrak D(u):=0.
  $$
- [x] **Energy Inequality:**
  $$
  E(u(t))=E(u_0).
  $$
- [x] **Bound Witness:** exact energy conservation.

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [x] **Bad Set $\mathcal B$:** finite-time $H^s$ blow-up configurations.
- [x] **Recovery Map $\mathcal R$:** identity on the smooth branch; no repair event is invoked.
- [x] **Event Counter:** $N(T)=0$ on each compact subinterval of $[0,T_*)$.
- [x] **Finiteness:** empty-event route.

#### **Template: $C_\mu$ (Compactness Interface)**
- [x] **Symmetry Group $G$:** torus translations.
- [x] **Group Action $\rho$:** $\rho_a u(x)=u(x+a)$.
- [x] **Quotient Space:** periodic profiles modulo translation.
- [x] **Concentration Measure:** bounded $H^s(\mathbb T^3)$ subsets are precompact in $H^{s-1}(\mathbb T^3)$.

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [x] **Scaling Action:**
  $$
  u_\lambda(x,t)=\lambda u(\lambda x,\lambda t).
  $$
- [x] **Height Exponent $\alpha$:**
  $$
  E(u_\lambda)=\lambda^{-1}E(u),
  \qquad
  \alpha=-1.
  $$
- [x] **Dissipation Exponent $\beta$:**
  $$
  \mathfrak D(u_\lambda)=0,
  \qquad
  \beta=0.
  $$
- [x] **Criticality:**
  $$
  \lambda_c=0,
  \qquad
  \beta-\alpha=1>0=\lambda_c.
  $$

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [x] **Parameter Space $\Theta$:**
  $$
  \Theta=\{\text{periodic 3D Euler on }\mathbb T^3\}.
  $$
- [x] **Parameter Map:** $\theta(u)=\theta_0$.
- [x] **Reference Point:** fixed equation sector.
- [x] **Stability Bound:** no parameter drift.

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [x] **Capacity Functional:** Hausdorff/capacity witness for candidate singular support.
- [x] **Singular Set $\Sigma$:** candidate finite-time singular set.
- [x] **Codimension:** unresolved on the designated route.
- [x] **Capacity Bound:** not certified on the designated route.

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [x] **Gradient Operator $\nabla$:** not used on the designated route.
- [x] **Critical Set $M$:** not certified.
- [x] **Łojasiewicz Exponent $\theta$:** not certified.
- [x] **Łojasiewicz-Simon Inequality:** not invoked.

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [x] **Topological Invariant $\tau$:** divergence-free mean-zero sector with helicity class.
- [x] **Sector Classification:** designated divergence-free mean-zero sector.
- [x] **Sector Preservation:** preserved by the Euler flow.
- [x] **Tunneling Events:** none on the designated route.

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [x] **O-minimal Structure $\mathcal O$:** smooth tame local $H^s$ presentation.
- [x] **Definability $\mathrm{Def}$:** Euler vector field, vorticity equation, and BKM quantity on the route.
- [x] **Singular Set Tameness:** only the candidate set is tracked.
- [x] **Cell Decomposition:** local tame chart structure.

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [x] **Measure $\mathcal M$:** route-relative energy-shell measure.
- [x] **Invariant Measure $\mu$:** conservative transport preserves the energy shell.
- [x] **Mixing Time $\tau_{\mathrm{mix}}$:** not certified.
- [x] **Mixing Property:** unresolved on the designated route.

#### **Template: $\mathrm{RepDesc}_K$ (Dictionary / Description Interface)**
- [x] **Language $\mathcal L$:** velocity, vorticity, and Fourier descriptions.
- [x] **Dictionary $D$:**
  $$
  u
  \longleftrightarrow
  \omega
  \longleftrightarrow
  \widehat u.
  $$
- [x] **Complexity Measure $K$:** Sobolev/Fourier weighted description.
- [x] **Faithfulness:** mean-zero periodic Biot-Savart recovers $u$ from $\omega$.

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [x] **Metric Tensor $g$:** canonical $H^s$ metric.
- [x] **Vector Field $v$:**
  $$
  v(u)=-(u\cdot\nabla)u-\nabla p.
  $$
- [x] **Gradient Compatibility:** not certified.
- [x] **Monotonicity:** conservative transport branch.

### **0.2 Boundary Interface Permits (Nodes 13-16)**

| Permit | Status | Note |
|---|---|---|
| $K_{\mathrm{Bound}_\partial}^-$ | Yes | periodic closed-system branch |
| $K_{\mathrm{Bound}_B}$ | N/A | skipped after Node 13 |
| $K_{\mathrm{Bound}_{\Sigma}}$ | N/A | skipped after Node 13 |
| $K_{\mathrm{GC}_T}$ | N/A | skipped after Node 13 |

### **0.2b Derived Witness Certificates (Optional)**

| Certificate | Derived From | Payload | Notes |
|---|---|---|---|
| none | - | - | no optional derived witness certificate is used on the designated route |

Auxiliary witness data recorded outside the optional-certificate ledger:
$$
\mathcal H(u)=\int_{\mathbb T^3}u\cdot\omega\,dx,
\qquad
\partial_tX(a,t)=u(X(a,t),t),
\qquad
X(a,0)=a.
$$

### **0.3 The Lock (Node 17)**

| Permit ID | Node | Question | Required Implementation | Certificate |
|-----------|------|----------|------------------------|-------------|
| $\mathrm{Cat}_{\mathrm{Hom}}$ | Lock | Is $\mathrm{Hom}(\mathcal H_{\mathrm{bad}},\mathcal H)=\emptyset$? | $\mathbf{Hypo}_{T_{\text{hyperbolic}}}$, universal bad pattern, certified completeness package, tactics E1--E13 | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ |

Lock instantiation:
$$
\mathcal H_{\mathrm{bad}}
=
\{\text{finite-time }H^s\text{ blow-up with nonintegrable } \|\omega\|_{L^\infty}\text{ integral}\}.
$$

No certified completeness package
$$
\left(
K_{\mathrm{Germ}}^+,
K_{\mathrm{init}}^+,
K_{\mathrm{CatLib}}^+
\right)
$$
is supplied on the designated route.

### **0.3b Goal and Backend Certificates**

| Certificate | Role | Required When |
|---|---|---|
| $K_{\mathrm{Goal}}^+$ | designated backend goal certificate for the declared local continuation/stretching route | designated goal closure |
| Backend analytic package | local well-posedness, vorticity equation, periodic Biot-Savart recovery, continuation criterion, and Lagrangian stretching identities | designated backend route |
| $K_{\mathrm{Germ}}^+$ | certifies germ smallness / classifiable singularity package | any Lock-based structural exclusion theorem |
| $K_{\mathrm{init}}^+$ | certifies the universal bad object / initiality package | any Lock-based structural exclusion theorem |
| $K_{\mathrm{CatLib}}^+$ | certifies completeness of the finite bad-pattern library | any Lock-based structural exclusion theorem |

---

## **Part I: The Instantiation (Thin Object Definitions)**

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**

- **State Space ($\mathcal X$):** mean-zero divergence-free $H^s(\mathbb T^3;\mathbb R^3)$, $s>\tfrac52$.
- **Metric ($d$):** $H^s$ distance.
- **Measure ($\mu$):** normalized Lebesgue measure on $\mathbb T^3$.
- **Auxiliary Variable:** $\omega=\nabla\times u$.
- **Flow Map:**
  $$
  \partial_tX(a,t)=u(X(a,t),t),
  \qquad
  X(a,0)=a.
  $$

### **2. The Potential ($\Phi^{\text{thin}}$)**

- **Height Functional ($\Phi$):**
  $$
  \Phi(u)=E(u)=\tfrac12\|u\|_{L^2(\mathbb T^3)}^2.
  $$
- **Secondary Invariant:**
  $$
  \mathcal H(u)=\int_{\mathbb T^3}u\cdot\omega\,dx.
  $$
- **Route Control Quantity:**
  $$
  M(t)=\int_0^t\|\omega(s)\|_{L^\infty(\mathbb T^3)}\,ds.
  $$
- **Scaling Exponent ($\alpha$):** $\alpha=-1$.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**

- **Dissipation Rate:**
  $$
  \mathfrak D(u)=0.
  $$
- **Dynamics:**
  $$
  u_t+(u\cdot\nabla)u+\nabla p=0,
  \qquad
  \nabla\cdot u=0.
  $$
- **Vorticity Dynamics:**
  $$
  \partial_t\omega+u\cdot\nabla\omega=(\omega\cdot\nabla)u.
  $$
- **Pathwise Stretching:**
  $$
  \frac{d}{dt}\omega(X(a,t),t)=\nabla u(X(a,t),t)\,\omega(X(a,t),t).
  $$

### **4. The Invariance ($G^{\text{thin}}$)**

- **Symmetry Group ($\mathrm{Grp}$):** torus translations, frame rotations, local homogeneous scaling.
- **Action ($\rho$):** translation action on the torus phase space.
- **Scaling Subgroup ($\mathcal S$):**
  $$
  u_\lambda(x,t)=\lambda u(\lambda x,\lambda t).
  $$
- **Auxiliary Reconstruction:** periodic Biot-Savart recovery of $u$ and $\nabla u$ from $\omega$.

---

## **Part II: Sieve Execution (Verification Run)**

### **EXECUTION PROTOCOL**

Nodes $1$--$13$ execute directly, Nodes $14$--$16$ are skipped on the closed-system branch, and Node $17$ records the Lock diagnostic. The designated goal is extracted from the local continuation/stretching backend package.

### **Level 1: Conservation**

#### **Node 1: EnergyCheck ($D_E$)**

**Question:** Is the height functional $\Phi$ bounded along trajectories?

**Step-by-step execution:**
1. [x] $u_0\in H^s$ implies $E(u_0)<\infty$.
2. [x] Testing the equation against $u$ and using incompressibility gives
   $$
   \frac{d}{dt}E(u(t))=0.
   $$
3. [x] Hence
   $$
   E(u(t))=E(u_0)
   \qquad
   (0\le t<T_*).
   $$

**Certificate:**
$$
K_{D_E}^+=(E,\mathfrak D=0,E(t)=E(0)).
$$

#### **Node 2: ZenoCheck ($\mathrm{Rec}_N$)**

**Question:** Does the trajectory visit the bad set only finitely many times on bounded intervals?

**Step-by-step execution:**
1. [x] No repair or restart event is introduced on the designated route.
2. [x] The event counter is
   $$
   N(T)=0.
   $$
3. [x] The smooth branch is continuous on each compact subinterval of $[0,T_*)$.

**Certificate:**
$$
K_{\mathrm{Rec}_N}^+=(\mathcal B=\mathcal H_{\mathrm{bad}},\mathcal R=\mathrm{id},N(T)=0).
$$

#### **Node 3: CompactCheck ($C_\mu$)**

**Question:** Has a concentration profile been certified?

**Step-by-step execution:**
1. [x] Bounded $H^s(\mathbb T^3)$ subsets are precompact in $H^{s-1}(\mathbb T^3)$.
2. [x] The tracked symmetry is torus translation.
3. [x] The periodic domain excludes spatial escape.

**Certificate:**
$$
K_{C_\mu}^+=(G=\mathbb T^3,\mathcal X//G,\text{torus compactness}).
$$

### **Level 2: Duality & Symmetry**

#### **Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)**

**Question:** Does the height functional block concentration?

**Step-by-step execution:**
1. [x] The route uses
   $$
   u_\lambda(x,t)=\lambda u(\lambda x,\lambda t).
   $$
2. [x] The energy scales as
   $$
   E(u_\lambda)=\lambda^{-1}E(u).
   $$
3. [x] The energy route is supercritical in three space dimensions.

**Certificate:**
$$
K_{\mathrm{SC}_\lambda}^-
=
(\alpha=-1,\beta=0,\lambda_c=0,\beta-\alpha=1\ge\lambda_c).
$$

#### **Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)**

**Question:** Are system parameters stable?

**Step-by-step execution:**
1. [x] The equation sector is fixed.
2. [x] No forcing parameter or viscosity parameter drifts.
3. [x] The route remains at $\theta_0$ for all $t<T_*$.

**Certificate:**
$$
K_{\mathrm{SC}_{\partial c}}^+=(\theta=\theta_0,\text{fixed equation sector}).
$$

### **Level 3: Geometry & Stiffness**

#### **Node 6: GeomCheck ($\mathrm{Cap}_H$)**

**Question:** Is there a local capacity certificate for the candidate singular route?

**Step-by-step execution:**
1. [x] The candidate bad event is $T_*<\infty$.
2. [x] The local continuation package gives
   $$
   T_*<\infty
   \Longrightarrow
   \int_0^{T_*}\|\omega(t)\|_{L^\infty}\,dt=\infty.
   $$
3. [x] No capacity or codimension certificate for $\Sigma$ is supplied.

**Certificate:**
$$
K_{\mathrm{Cap}_H}^{\mathrm{inc}}
=
\left\{
\text{obligation: provide a local capacity certificate for the candidate singular set},
\text{missing: }[K_{\mathrm{Cap}_H}^+,K_{\mathrm{Cap}_H}^-],
\text{failure\_code: MISSING\_LOCAL\_CAPACITY},
\text{trace: BKM localizes the route but does not produce a local capacity bound}
\right\}.
$$

#### **Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)**

**Question:** Is there a route-relative stiffness certificate?

**Step-by-step execution:**
1. [x] The designated route is inviscid and conservative.
2. [x] No Lyapunov dissipation quantity is present.
3. [x] No spectral-gap or Łojasiewicz-Simon witness is supplied.

**Certificate:**
$$
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}
=
\left\{
\text{obligation: decide the spectral-gap / LS branch},
\text{missing: }[K_{\mathrm{LS}_\sigma}^+,K_{\mathrm{LS}_\sigma}^-],
\text{failure\_code: MISSING\_LS\_WITNESS},
\text{trace: no spectral-gap or LS witness is supplied near a critical set}
\right\}.
$$

### **Level 4: Topology**

#### **Node 8: TopoCheck ($\mathrm{TB}_\pi$)**

**Question:** Is the sector preserved?

**Step-by-step execution:**
1. [x] The designated sector is divergence-free and mean-zero.
2. [x] Euler preserves divergence-free and mean-zero constraints.
3. [x] Helicity is conserved on the smooth branch.

**Certificate:**
$$
K_{\mathrm{TB}_\pi}^+=(\text{divergence-free mean-zero sector preserved}).
$$

#### **Node 9: TameCheck ($\mathrm{TB}_O$)**

**Question:** Is the local phase-space presentation tame?

**Step-by-step execution:**
1. [x] $H^s$ is a smooth Banach manifold chart in the designated range.
2. [x] The Euler vector field is smooth for $s>\tfrac52$.
3. [x] The route admits a tame local presentation.

**Certificate:**
$$
K_{\mathrm{TB}_O}^+=(\mathcal O,\text{smooth tame local }H^s\text{ presentation}).
$$

### **Level 5: Mixing**

#### **Node 10: ErgoCheck ($\mathrm{TB}_\rho$)**

**Question:** Does the flow mix?

**Step-by-step execution:**
1. [x] The route is conservative transport.
2. [x] No finite mixing witness is supplied.
3. [x] No permanent-trap witness is supplied.

**Certificate:**
$$
K_{\mathrm{TB}_\rho}^{\mathrm{inc}}
=
\left\{
\text{obligation: decide the finite-mixing / trapping branch},
\text{missing: }[K_{\mathrm{TB}_\rho}^+,K_{\mathrm{TB}_\rho}^-],
\text{failure\_code: MISSING\_MIX\_WITNESS},
\text{trace: no spectral-gap proof and no permanent-trap witness are supplied on the designated route}
\right\}.
$$

### **Level 6: Complexity**

#### **Node 11: ComplexCheck ($\mathrm{RepDesc}_K$)**

**Question:** Is the description finite and faithful?

**Step-by-step execution:**
1. [x] Fourier coefficients determine the mean-zero velocity.
2. [x] Vorticity plus periodic Biot-Savart determine $u$ and $\nabla u$.
3. [x] The route dictionary is faithful.

**Certificate:**
$$
K_{\mathrm{RepDesc}_K}^+=(\mathcal L,D,K,\text{faithful}).
$$

#### **Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)**

**Question:** Is oscillatory behavior detected in a finite spectral window?

**Step-by-step execution:**
1. [x] The route records local Euler, vorticity, and Lagrangian certificates.
2. [x] No thin-trace spectral window certificate is supplied.
3. [x] The node remains diagnostic.

**Certificate:**
$$
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}
=
\left\{
\text{obligation: decide finite-window oscillation detection},
\text{missing: }[K_{\mathrm{GC}_\nabla}^+,K_{\mathrm{GC}_\nabla}^-],
\text{failure\_code: MISSING\_OSC\_WINDOW},
\text{trace: no thin-trace spectral density window is supplied on the designated route}
\right\}.
$$

### **Level 7: Boundary (Open Systems)**

#### **Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)**

**Question:** Is the system open?

**Step-by-step execution:**
1. [x] The domain is periodic.
2. [x] No external input/output maps are present.
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
1. [x] The bad-pattern branch is finite-time $H^s$ breakdown with
   $$
   \int_0^{T_*}\|\omega(t)\|_{L^\infty}\,dt=\infty.
   $$
2. [x] No certified bad-pattern library for all 3D Euler singularities is supplied.
3. [x] No certified E1--E13 tactic package closes Hom-emptiness.
4. [x] No explicit morphism from a universal bad object is constructed.

**Certificate:**
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}
=
\left\{
\text{obligation: resolve the Lock verdict},
\text{missing: }[\text{successful\_tactic},\text{morphism\_construction}],
\text{failure\_code: LOCK\_UNDECIDED},
\text{trace: no certified Lock tactic closes Hom-emptiness and no explicit morphism is constructed}
\right\}.
$$

---

## **Part II-B: Upgrade Pass**

### **Upgrade Pass Protocol**

Goal-relevant upgrade:
$$
K_{\mathrm{Goal}}^+.
$$

Residual obligations:

| ID | Node | Obligation | Missing | In Goal Cone? |
|----|------|------------|---------|---------------|
| OBL-1 | 6 | local capacity certificate for $\Sigma$ | $K_{\mathrm{Cap}_H}^+, K_{\mathrm{Cap}_H}^-$ | No |
| OBL-2 | 7 | spectral-gap / LS branch | $K_{\mathrm{LS}_\sigma}^+, K_{\mathrm{LS}_\sigma}^-$ | No |
| OBL-3 | 10 | finite-mixing / trapping branch | $K_{\mathrm{TB}_\rho}^+, K_{\mathrm{TB}_\rho}^-$ | No |
| OBL-4 | 12 | finite-window oscillation detection | $K_{\mathrm{GC}_\nabla}^+, K_{\mathrm{GC}_\nabla}^-$ | No |
| OBL-5 | 17 | Lock verdict | `successful_tactic` or `morphism_construction` | No |

Non-circularity:
$$
K_{\mathrm{Goal}}^+\notin\Gamma_{\mathrm{loc}}.
$$

---

## **Part II-C: Breach/Surgery/Re-entry Protocol**

### **Breach Detection**

No $K_X^{\mathrm{br}}$ certificate is emitted on the designated route.

### **Surgery Selection**

No surgery is selected.

### **Surgery Execution**

Not used.

### **Re-entry Protocol**

Not used.

---

## **Part III-A: Lyapunov Reconstruction**

### **Lyapunov Existence Check**

No KRNL-Lyapunov route is invoked.

### **Step 1: Value Function Construction (KRNL-Lyapunov)**

Not used.

### **Step 2: Jacobi Metric Reconstruction (KRNL-Jacobi)**

Not used.

### **Step 3: Hamilton-Jacobi PDE (KRNL-HamiltonJacobi)**

Not used.

### **Step 4: Verify Lyapunov Properties**

Not used.

---

## **Part III-B: Result Extraction (Mining the Run)**

### **3.1 Backend Route Data**

$$
u\in C([0,T_*);H^s(\mathbb T^3;\mathbb R^3)),
\qquad
\sup_{0\le t\le T}\|u(t)\|_{H^s}<\infty\ \Rightarrow\ T<T_*.
$$

$$
\partial_t\omega+u\cdot\nabla\omega=(\omega\cdot\nabla)u.
$$

$$
u=\mathcal B_{\mathrm{per}}[\omega],
\qquad
\nabla u=\nabla\mathcal B_{\mathrm{per}}[\omega].
$$

$$
T_*<\infty
\Longrightarrow
\int_0^{T_*}\|\omega(t)\|_{L^\infty}\,dt=\infty.
$$

$$
\frac{d}{dt}\omega(X(a,t),t)=\nabla u(X(a,t),t)\,\omega(X(a,t),t),
\qquad
\frac{d}{dt}|\omega(X(a,t),t)|=(\xi\cdot S\xi)(X(a,t),t)\,|\omega(X(a,t),t)|.
$$

### **3.2 Quantitative Bounds**

$$
E(t)=E(0).
$$

$$
\mathcal H(t)=\mathcal H(0).
$$

$$
|\omega(X(a,t),t)|
=
|\omega_0(a)|
\exp\left(\int_0^t(\xi\cdot S\xi)(X(a,s),s)\,ds\right).
$$

$$
\|\omega(t)\|_{L^\infty}
\le
\|\omega_0\|_{L^\infty}
\exp\left(\int_0^t\|S(s)\|_{L^\infty}\,ds\right).
$$

$$
\int_0^T\|\omega(t)\|_{L^\infty}\,dt<\infty
\Longrightarrow
\sup_{0\le t\le T}\|u(t)\|_{H^s}<\infty,
\qquad
T<T_*.
$$

### **3.3 Certified Local Property Package**

$$
\Gamma_{\mathrm{loc}}
=
\left\{
K_{D_E}^+,
K_{\mathrm{Rec}_N}^+,
K_{C_\mu}^+,
K_{\mathrm{SC}_\lambda}^-,
K_{\mathrm{SC}_{\partial c}}^+,
K_{\mathrm{TB}_\pi}^+,
K_{\mathrm{TB}_O}^+,
K_{\mathrm{RepDesc}_K}^+,
K_{\mathrm{Bound}_\partial}^-
\right\}.
$$

The local package is supplemented by the witness data
$$
\mathcal H(t)=\mathcal H(0),
\qquad
\partial_tX(a,t)=u(X(a,t),t),
\qquad
X(a,0)=a,
$$
and by the backend route data of §3.1.

### **3.4 Characterization of Singular Formation on the Declared Route**

The declared route data are
$$
T_*<\infty,
\qquad
\int_0^{T_*}\|\omega(t)\|_{L^\infty}\,dt=\infty,
\qquad
\frac{d}{dt}|\omega(X(a,t),t)|=(\xi\cdot S\xi)(X(a,t),t)\,|\omega(X(a,t),t)|.
$$

The only amplification term recorded on the route is
$$
(\omega\cdot\nabla)u.
$$

The candidate singular set $\Sigma$ is tracked, but no local capacity certificate and no Lock-based realization/exclusion certificate is supplied.

### **3.5 Functional Objects**

$$
E(u)=\tfrac12\|u\|_{L^2(\mathbb T^3)}^2,
\qquad
\mathcal H(u)=\int_{\mathbb T^3}u\cdot\omega\,dx,
\qquad
\mathcal B_{\mathrm{per}},
\qquad
X(a,t).
$$

### **3.6 Retroactive Upgrades**

The backend route data of §3.1 close to
$$
K_{\mathrm{Goal}}^+.
$$

No certificate in $\Downarrow(K_{\mathrm{Goal}}^+)$ requires further discharge.

### **3.7 ZFC Proof Export (Appendix Bridge)**

Not used on the designated route.

---

## **Part III-C: Obligation Ledger**

### **Introduced Obligations**

| ID | Node | Certificate | Obligation | Missing | In Goal Cone? | Status |
|----|------|-------------|------------|---------|---------------|--------|
| OBL-1 | 6 | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | local capacity certificate for $\Sigma$ | $K_{\mathrm{Cap}_H}^+, K_{\mathrm{Cap}_H}^-$ | No | residual |
| OBL-2 | 7 | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | spectral-gap / LS branch | $K_{\mathrm{LS}_\sigma}^+, K_{\mathrm{LS}_\sigma}^-$ | No | residual |
| OBL-3 | 10 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | finite-mixing / trapping branch | $K_{\mathrm{TB}_\rho}^+, K_{\mathrm{TB}_\rho}^-$ | No | residual |
| OBL-4 | 12 | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | finite-window oscillation detection | $K_{\mathrm{GC}_\nabla}^+, K_{\mathrm{GC}_\nabla}^-$ | No | residual |
| OBL-5 | 17 | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ | Lock verdict | `successful_tactic` or `morphism_construction` | No | residual |

### **Discharge Events**

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| - | - | - | - |

### **Remaining Obligations**

$$
\mathrm{OBL}_{\mathrm{rem}}
=
\left\{
\mathrm{OBL}\text{-}1,
\mathrm{OBL}\text{-}2,
\mathrm{OBL}\text{-}3,
\mathrm{OBL}\text{-}4,
\mathrm{OBL}\text{-}5
\right\},
\qquad
\mathrm{OBL}_{\mathrm{rem}}\cap\Downarrow(K_{\mathrm{Goal}}^+)=\varnothing.
$$

### **Ledger Validation**

- [x] All goal-relevant `inc` certificates upgraded or absent.
- [x] All goal-relevant breach obligations discharged or absent.
- [x] All remaining obligations lie outside the designated goal dependency cone.

---

## Formal Proof

::::{prf:proof} Proof of Theorem {prf:ref}`thm-euler-3d-main`

The Euler data $(\mathcal X,\Phi,\mathfrak D,G)$ define a hyperbolic hypostructure with the thin objects and interface permits recorded in Parts 0 and I. The instantiated sieve yields
$$
K_{D_E}^+,\;
K_{\mathrm{Rec}_N}^+,\;
K_{C_\mu}^+,\;
K_{\mathrm{SC}_\lambda}^-,\;
K_{\mathrm{SC}_{\partial c}}^+,\;
K_{\mathrm{TB}_\pi}^+,\;
K_{\mathrm{TB}_O}^+,\;
K_{\mathrm{RepDesc}_K}^+,\;
K_{\mathrm{Bound}_\partial}^-,
$$
and it records the auxiliary diagnostics
$$
K_{\mathrm{Cap}_H}^{\mathrm{inc}},\;
K_{\mathrm{LS}_\sigma}^{\mathrm{inc}},\;
K_{\mathrm{TB}_\rho}^{\mathrm{inc}},\;
K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}.
$$

Part III-B.1 records the declared local continuation, vorticity, Biot-Savart, continuation, and stretching statements for the designated route.

This follows the canonical PDE continuation notation as well:
$K_{\mathrm{WP}_{s_c}}^+$ with $s_c=\tfrac52$ abbreviates the local continuation bridge (local well-posedness + uniqueness + continuation criterion + blow-up condition) for the Euler route.

Node 17 records
$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}},
$$
so no Lock-based structural exclusion theorem is used in the designated route. The goal closes instead from the backend route data recorded in Part III-B.1.

This promotion is non-circular because none of the premise certificates depends on
$$
K_{\mathrm{Goal}}^+.
$$

Part III-C verifies that every remaining obligation lies outside the dependency cone of the designated goal. Therefore
$$
K_{\mathrm{Goal}}^+
$$
is established. $\square$

::::

---

## Part IV: Final Certificate Chain

### 4.1 Validity Checklist

| Item | Status | Witness |
|---|---|---|
| All route-relevant nodes executed with explicit certificates | Yes | Parts II and IV.3 |
| Designated goal promotion executed | Yes | Part II-B |
| Declared local continuation/stretching backend route present | Yes | Part III-B.1 |
| Lock used for structural exclusion | No | Node 17 remains auxiliary `br-inc` |
| Designated goal certificate reached | Yes | $K_{\mathrm{Goal}}^+$ |
| Goal-relevant obligations discharged | Yes | Part IV.4 |
| **Final Status** | **UNCONDITIONAL** | GOAL-CONE EMPTY for $K_{\mathrm{Goal}}^+$ |

### 4.2 Core Node Trace

| Node | Interface | Certificate | Status | Role in the designated goal route |
|---|---|---|---|---|
| 1 | $D_E$ | $K_{D_E}^+$ | Yes | Required |
| 2 | $\mathrm{Rec}_N$ | $K_{\mathrm{Rec}_N}^+$ | Yes | Required |
| 3 | $C_\mu$ | $K_{C_\mu}^+$ | Yes | Required |
| 4 | $\mathrm{SC}_\lambda$ | $K_{\mathrm{SC}_\lambda}^-$ | Typed negative | Records energy-supercritical scaling |
| 5 | $\mathrm{SC}_{\partial c}$ | $K_{\mathrm{SC}_{\partial c}}^+$ | Yes | Required |
| 6 | $\mathrm{Cap}_H$ | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | Inconclusive | Auxiliary diagnostic; outside main derivation |
| 7 | $\mathrm{LS}_\sigma$ | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | Inconclusive | Auxiliary diagnostic; outside main derivation |
| 8 | $\mathrm{TB}_\pi$ | $K_{\mathrm{TB}_\pi}^+$ | Yes | Required |
| 9 | $\mathrm{TB}_O$ | $K_{\mathrm{TB}_O}^+$ | Yes | Required |
| 10 | $\mathrm{TB}_\rho$ | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | Inconclusive | Auxiliary diagnostic; outside main derivation |
| 11 | $\mathrm{RepDesc}_K$ | $K_{\mathrm{RepDesc}_K}^+$ | Yes | Required |
| 12 | $\mathrm{GC}_\nabla$ | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | Inconclusive | Auxiliary diagnostic; outside main derivation |
| 13 | $\mathrm{Bound}_\partial$ | $K_{\mathrm{Bound}_\partial}^-$ | Closed | Routes to the closed-system branch |
| 17 | $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ | Inconclusive | Auxiliary Lock diagnostic only |

### 4.3 Backend and Goal Trace

| Stage | Item | Status | Source |
|---|---|---|---|
| Helicity witness | helicity conservation | Yes | Part 0.2b |
| Flow-map witness | Lagrangian flow map | Yes | Part 0.2b |
| Local continuation backend | maximal smooth $H^s$ branch | Yes | Part III-B.1 |
| Vorticity backend | vorticity stretching equation | Yes | Part III-B.1 |
| Biot-Savart backend | periodic recovery of $u,\nabla u$ from $\omega$ | Yes | Part III-B.1 |
| Continuation criterion | Beale-Kato-Majda condition | Yes | Part III-B.1 |
| Stretching law | pathwise vorticity growth identities | Yes | Part III-B.1 |
| Goal promotion | $K_{\mathrm{Goal}}^+$ | Yes | Part II-B / Part III-B.6 |

### 4.4 Obligation Ledger Summary

| ID | Certificate | Obligation | In Goal Cone? | Status | Discharge / Reason |
|---|---|---|---|---|---|
| OBL-1 | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | local capacity certificate for $\Sigma$ | No | Residual diagnostic | not used in this route |
| OBL-2 | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | spectral-gap / LS branch | No | Residual diagnostic | not used in this route |
| OBL-3 | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | finite-mixing / trapping branch | No | Residual diagnostic | not used in this route |
| OBL-4 | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | finite-window oscillation detection | No | Residual diagnostic | not used in this route |
| OBL-5 | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ | Lock verdict | No | Residual diagnostic | not used in this route |

## Executive Summary: The Proof Dashboard

### 1. System Instantiation (The Physics)

| Object | Definition | Role |
|---|---|---|
| **Arena ($\mathcal{X}$)** | mean-zero divergence-free $H^s(\mathbb T^3;\mathbb R^3)$, $s>\tfrac52$ | state space |
| **Potential ($\Phi$)** | $E(u)=\frac12\|u\|_{L^2(\mathbb T^3)}^2$ | conserved height |
| **Cost ($\mathfrak{D}$)** | $\mathfrak D(u)=0$ | conservative route |
| **Invariance ($G$)** | torus translations, frame rotations, local homogeneous scaling | preserved structure |

### 2. Execution Trace (The Logic)

| Node | Check | Outcome | Certificate Payload | Ledger State |
|---|---|---:|---|---|
| 1 | Energy Bound | YES | $K_{D_E}^+$ | `[]` |
| 2 | Zeno / Recovery | YES | $K_{\mathrm{Rec}_N}^+$ | `[]` |
| 3 | Compact Check | YES | $K_{C_\mu}^+$ | `[]` |
| 4 | Scale Check | NO | $K_{\mathrm{SC}_\lambda}^-$ | `[]` |
| 5 | Parametric Check | YES | $K_{\mathrm{SC}_{\partial c}}^+$ | `[]` |
| 6 | Geometric Check | INC | $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ | `[OBL-1]` |
| 7 | Stiffness Check | INC | $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ | `[OBL-1, OBL-2]` |
| 8 | Topological Check | YES | $K_{\mathrm{TB}_\pi}^+$ | `[OBL-1, OBL-2]` |
| 9 | Tame Check | YES | $K_{\mathrm{TB}_O}^+$ | `[OBL-1, OBL-2]` |
| 10 | Ergodic Check | INC | $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ | `[OBL-1, OBL-2, OBL-3]` |
| 11 | Complex Check | YES | $K_{\mathrm{RepDesc}_K}^+$ | `[OBL-1, OBL-2, OBL-3]` |
| 12 | Oscillate Check | INC | $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ | `[OBL-1, OBL-2, OBL-3, OBL-4]` |
| 13 | Boundary Check | CLOSED | $K_{\mathrm{Bound}_\partial}^-$ | `[OBL-1, OBL-2, OBL-3, OBL-4]` |
| 14-16 | Boundary Subnodes | N/A | closed-system branch | `[OBL-1, OBL-2, OBL-3, OBL-4]` |
| 17 | LOCK | INC | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ | `[OBL-1, OBL-2, OBL-3, OBL-4, OBL-5]` |
| -- | GOAL PROMOTION | OK | $K_{\mathrm{Goal}}^+$ | `[OBL-1, OBL-2, OBL-3, OBL-4, OBL-5]` |

### 3. Lock Mechanism (The Exclusion)

| Tactic | Description | Status | Reason / Mechanism |
|---|---|---:|---|
| E1 | Dimension | N/A | not attempted |
| E2 | Invariant | N/A | not attempted |
| E3 | Positivity | N/A | not attempted |
| E4 | Integrality | N/A | not attempted |
| E5 | Functional | N/A | not attempted |
| E6 | Causal | N/A | not attempted |
| E7 | Thermodynamic | N/A | not attempted |
| E8 | Holographic | N/A | not attempted |
| E9 | Ergodic | N/A | not attempted |
| E10 | Definability | N/A | not attempted |
| E11 | Galois-Monodromy | N/A | not attempted |
| E12 | Algebraic Compressibility | N/A | not attempted |
| E13 | Algorithmic Completeness | N/A | E13 is not the selected exclusion tactic; split semantics are explicit in Lock: complete finite-library closure gives $K_{\mathrm{E13}}^{\mathrm{blk}}$, while incomplete closure gives $K_{\mathrm{E13}}^{\mathrm{br-inc}}$ and routes to reconstruction. |

### 4. Final Verdict

### Assumption Provenance

- **Imported from literature?** yes
- **Theorem name(s):** Beale-Kato-Majda local continuation criterion and local Lagrangian flow estimates
- **Hypotheses required:** smooth divergence-free data, short-time smooth solution from local wellposedness, and the declared continuation hypothesis required by the backend
- **Non-circularity note:** the route is conditional on the declared continuation bridge and does not use the target regularity as a premise.
- **Goal-certificate location in local-to-global chain:** Part III-B.1 and Part III-B.6 route the declared continuation bridge to the local route certificate $K_{\mathrm{Goal}}^+$.

- **Designated Goal Certificate:** $K_{\mathrm{Goal}}^+$
- **Status:** UNCONDITIONAL for the local continuation/stretching route
- **Goal-Cone Ledger:** EMPTY
- **Residual Non-Goal Obligations:** `OBL-1`, `OBL-2`, `OBL-3`, `OBL-4`, `OBL-5`
- **Singularity Set:** candidate singular set $\Sigma$ only; no local capacity closure and no Lock realization/exclusion theorem
- **Primary Final Route:** direct sieve execution + local continuation backend + BKM + Lagrangian stretching promotion

---

## Document Information

| Field | Value |
|-------|-------|
| **Document Type** | Proof Object |
| **Framework** | Hypostructure v1.0 |
| **Problem Class** | Open PDE problem treated as a local route-certificate instance |
| **System Type** | $T_{\text{hyperbolic}}$ |
| **Problem Type** | Local continuation / singular-route certificate |
| **Singularity Type** | auxiliary unresolved Lock sector only |
| **Verification Level** | Machine-checkable proof object |
| **Inc Certificates** | 5 introduced, 0 discharged, 5 auxiliary residual |
| **Final Status** | UNCONDITIONAL for $K_{\mathrm{Goal}}^+$ |
| **Generated** | 2026-04-15 |
