---
title: "ACT-Align - Complexity Theory Translation"
---

# ACT-Align: Surgery Alignment and Boundary Matching

## Overview

This document provides a complete complexity-theoretic translation of the ACT-Align metatheorem (Adjoint Surgery) from the hypostructure framework. The theorem establishes that adjoint representations align surgery boundaries to preserve quantum symmetries. In complexity theory terms, this corresponds to **Surgery Alignment**: boundary conditions preserve solution structure through compositional verification.

**Original Theorem Reference:** {prf:ref}`mt-act-align`

**Central Translation:** Adjoint variables $\lambda$ enforce boundary alignment with gradient matching $\nabla_x f \parallel \nabla_x g$ $\longleftrightarrow$ **Interface Conditions**: Boundary matching ensures compositional correctness across module boundaries.

---

## Complexity Theory Statement

**Theorem (Compositional Boundary Alignment, Computational Form).**
Let $\Pi = \Pi_1 \circ \Pi_2$ be a composite verification problem where $\Pi_1$ and $\Pi_2 share an interface $I$. There exist **interface witnesses** $\lambda$ such that:

**Input**: Component proofs $\pi_1: \Gamma_1 \vdash I$ and $\pi_2: I \vdash \Delta_2$

**Output**:
- Composite proof $\pi: \Gamma_1 \vdash \Delta_2$
- Alignment certificate $(\lambda^*, \pi^*, \text{interface match})$
- Verification that boundary conditions are satisfied

**Guarantees**:
1. **Primal correctness**: Each component $\pi_i$ satisfies its local specification
2. **Interface enforcement**: Boundary conditions $g(\pi_1, \pi_2) = 0$ are met
3. **Gradient alignment**: Local optima are globally consistent

**Formal Statement.** For compositional verification problem $\Pi = \Pi_1 \circ_I \Pi_2$:

1. **Saddle-Point Structure:** The composite problem has the form:
   $$\min_{\pi} \max_{\lambda} \mathcal{L}(\pi, \lambda) = C(\pi) + \lambda^T \cdot \text{mismatch}(I)$$

2. **KKT Conditions:** At the solution $(\pi^*, \lambda^*)$:
   $$\nabla_\pi C(\pi^*) + \lambda^{*T} \nabla_\pi g(\pi^*) = 0$$
   $$g(\pi^*) = 0$$

3. **Alignment Guarantee:** The gradients of cost and constraint are parallel:
   $$\nabla_\pi C \parallel \nabla_\pi g$$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Primal variables $x$ | Component proofs/programs | Verified modules $\pi_1, \pi_2$ |
| Dual variables $\lambda$ | Interface witnesses | Boundary certificates |
| Objective $f(x)$ | Total verification cost | $C(\pi) = C_1(\pi_1) + C_2(\pi_2)$ |
| Constraint $g(x) = 0$ | Interface conditions | Type/contract matching at boundary |
| Lagrangian $\mathcal{L}(x, \lambda)$ | Augmented verification | Cost + interface penalty |
| Saddle-point $\min_x \max_\lambda$ | Primal-dual verification | Compositional proof search |
| Gradient alignment $\nabla f \parallel \nabla g$ | Consistent local/global optima | Modular reasoning soundness |
| KKT conditions | Compositional correctness | Interface satisfaction + optimality |
| Costate $\lambda(t)$ | Temporal interface witness | Incremental proof certificate |
| Hamiltonian $H$ | Verification Hamiltonian | $H = C + \lambda^T \cdot \text{dynamics}$ |
| Actor (primal) | Program/proof component | Module implementation |
| Critic (dual) | Interface verifier | Contract checker |
| Boundary misalignment (Mode B.C) | Interface mismatch | Type error at boundary |
| Alignment enforcement | Contract enforcement | Runtime/static checking |
| Certificate $K_{\text{SurgBC}}$ | Compositional certificate | Boundary-correct proof |
| Pontryagin principle | Optimal control criterion | Bellman optimality |

---

## Compositional Verification and Interface Alignment

### The Compositional Framework

**Definition (Compositional System).** A compositional system consists of:
- **Modules**: $M_1, M_2, \ldots, M_k$ with specifications $\text{Spec}_i$
- **Interfaces**: $I_{ij}$ connecting modules $M_i$ and $M_j$
- **Composition**: $M = M_1 \circ_{I_{12}} M_2 \circ \cdots \circ_{I_{(k-1)k}} M_k$

**Problem (Interface Alignment).** Given proofs $\pi_i: \text{Pre}_i \vdash \text{Post}_i$ for each module:
- Verify that interfaces match: $\text{Post}_i|_I = \text{Pre}_{i+1}|_I$
- Construct composite proof: $\pi: \text{Pre}_1 \vdash \text{Post}_k$

### Connection to Lagrangian Mechanics

**Observation (Constrained Optimization = Compositional Verification).**

The Lagrangian formulation:
$$\mathcal{L}(\pi, \lambda) = C(\pi) + \lambda^T g(\pi)$$

corresponds to:
- **Objective $C(\pi)$**: Total verification cost (proof size, complexity)
- **Constraint $g(\pi) = 0$**: Interface matching conditions
- **Multiplier $\lambda$**: Importance weight for each interface condition

**Saddle-Point Interpretation:**
$$\min_\pi \max_\lambda \mathcal{L}(\pi, \lambda)$$

- **Minimize over $\pi$**: Find simplest proof satisfying constraints
- **Maximize over $\lambda$**: Enforce all interface conditions

### KKT Conditions as Compositional Correctness

**Theorem (KKT = Compositionality).** A compositional proof $\pi^*$ is correct if and only if it satisfies:

1. **Stationarity:** $\nabla_\pi C + \lambda^T \nabla_\pi g = 0$
   - Local optimality respects interface constraints

2. **Primal Feasibility:** $g(\pi^*) = 0$
   - All interfaces match exactly

3. **Dual Feasibility:** $\lambda^* \geq 0$ (for inequalities)
   - Constraint multipliers are non-negative

4. **Complementary Slackness:** $\lambda_i^* g_i(\pi^*) = 0$
   - Active constraints have positive multipliers

---

## Boundary Matching in Complexity Theory

### 1. Type-Theoretic Boundaries

**Definition (Interface Type).** An interface $I$ between modules $M_1: A \to B$ and $M_2: B \to C$ is a type $B$ such that:
- Output type of $M_1$ equals input type of $M_2$
- Type isomorphism: $\text{Post}(M_1) \cong \text{Pre}(M_2)$

**Alignment Condition:**
$$\text{typeof}(M_1.\text{output}) = \text{typeof}(M_2.\text{input})$$

**Failure Mode (Mode B.C):** Type mismatch at interface:
$$\text{typeof}(M_1.\text{output}) \neq \text{typeof}(M_2.\text{input})$$

**Surgery Repair:** Introduce adapter $\lambda: B_1 \to B_2$ to mediate:
$$M_1 \xrightarrow{B_1} \lambda \xrightarrow{B_2} M_2$$

### 2. Contract-Based Verification

**Definition (Design by Contract).** Each module $M_i$ has:
- **Precondition:** $\text{Pre}_i$
- **Postcondition:** $\text{Post}_i$
- **Invariant:** $\text{Inv}_i$

**Interface Contract:** At interface $I_{ij}$:
$$\text{Post}_i \Rightarrow \text{Pre}_j$$

**Lagrange Multiplier Interpretation:**
- $\lambda_{ij}$ = "cost" of interface violation
- $g_{ij} = \text{Post}_i \land \neg\text{Pre}_j$ = interface mismatch
- $\lambda_{ij} \cdot g_{ij}$ = penalty for violation

### 3. Assume-Guarantee Reasoning

**Definition (Assume-Guarantee).** Module $M$ satisfies:
$$\langle A \rangle M \langle G \rangle$$
meaning: if environment provides $A$, then $M$ guarantees $G$.

**Compositional Rule:**
$$\frac{\langle A_1 \rangle M_1 \langle G_1 \rangle \quad \langle A_2 \rangle M_2 \langle G_2 \rangle \quad G_1 \Rightarrow A_2}{\langle A_1 \rangle M_1 \| M_2 \langle G_2 \rangle}$$

**Connection to Adjoint Surgery:**
- $A_i$ = assumptions (primal constraints)
- $G_i$ = guarantees (primal objectives)
- $G_1 \Rightarrow A_2$ = interface alignment (dual constraint)
- Proof search = saddle-point optimization

---

## Proof Sketch: Boundary Alignment = Compositional Correctness

### Setup: Compositional Verification Framework

**Given:**
- Composite system $\Pi = \Pi_1 \circ_I \Pi_2$
- Component specifications $\text{Spec}_i = (\text{Pre}_i, \text{Post}_i)$
- Interface $I$ with alignment condition $g: \text{Post}_1|_I = \text{Pre}_2|_I$

**Goal:** Establish correspondence between adjoint surgery and compositional verification.

---

### Step 1: KKT Conditions = Interface Satisfaction

**Claim.** The KKT stationarity condition corresponds to consistent local/global optimization.

**Proof.**

At the saddle point $(\pi^*, \lambda^*)$:
$$\nabla_\pi C(\pi^*) + \lambda^{*T} \nabla_\pi g(\pi^*) = 0$$

**Interpretation:**
- $\nabla_\pi C$ = direction of decreasing verification cost
- $\nabla_\pi g$ = direction of improving interface alignment
- Stationarity means these are balanced

**Computational Meaning:**
- Cannot reduce cost without violating interface
- Cannot improve interface without increasing cost
- Optimal trade-off achieved

**Correspondence:**
$$\nabla_\pi C \parallel \nabla_\pi g \quad \longleftrightarrow \quad \text{Local optima are globally consistent}$$

The gradient parallelism ensures that local module optimization does not conflict with global compositional correctness. $\square$

---

### Step 2: Gradient Alignment = Modular Soundness

**Claim.** Gradient alignment $\nabla f \parallel \nabla g$ ensures that modular verification is sound.

**Proof.**

**Step 2.1 (Modular Optimization).** Each module optimizes locally:
$$\pi_i^* = \arg\min_{\pi_i} C_i(\pi_i) \quad \text{s.t.} \quad \text{Spec}_i$$

**Step 2.2 (Global Consistency).** For the composition to be correct:
$$\pi^* = (\pi_1^*, \pi_2^*) \quad \text{s.t.} \quad g(\pi_1^*, \pi_2^*) = 0$$

**Step 2.3 (Alignment Condition).** The gradient alignment:
$$\nabla_{\pi} C = -\lambda^T \nabla_{\pi} g$$

ensures that:
- Moving in the cost-reduction direction respects constraints
- Interface constraints are "tangent" to cost level sets

**Soundness Implication:**
If each module is locally optimal and interfaces align, then:
$$C(\pi_1^* \circ \pi_2^*) = C_1(\pi_1^*) + C_2(\pi_2^*)$$

No hidden cost from composition. $\square$

---

### Step 3: Pontryagin Interpretation = Dynamic Alignment

**Claim.** The costate dynamics correspond to incremental interface verification.

**Proof.**

In optimal control, the costate $\lambda(t)$ satisfies:
$$\dot{\lambda} = -\nabla_x H(x, u, \lambda)$$

where $H = f + \lambda^T \dot{x}$ is the Hamiltonian.

**Computational Interpretation:**
- $x(t)$ = program state at time $t$
- $\lambda(t)$ = "interface certificate" at time $t$
- $H$ = instantaneous verification cost + dynamics

**Incremental Verification:**
- At each step, $\lambda(t)$ tracks accumulated interface obligations
- Costate evolution ensures obligations are propagated correctly
- Terminal condition $\lambda(T) = 0$ ensures all obligations discharged

**Correspondence:**
$$\dot{\lambda} = -\nabla_x H \quad \longleftrightarrow \quad \text{Interface certificates evolve consistently}$$

$\square$

---

### Step 4: Actor-Critic = Primal-Dual Verification

**Claim.** The actor-critic mechanism corresponds to primal-dual proof search.

**Proof.**

**Actor (Primal):**
- Updates proof/program to minimize verification cost
- Corresponds to proof search for each module

**Critic (Dual):**
- Estimates "value" of interface conditions
- Corresponds to interface verification oracle

**Convergence Condition:**
Actor-critic convergence requires alignment:
$$\text{Actor gradient} \parallel \text{Critic gradient}$$

preventing interface misalignment.

**Correspondence to Compositional Verification:**

| Actor-Critic | Compositional Verification |
|--------------|---------------------------|
| Actor update | Module proof step |
| Critic update | Interface check |
| Policy gradient | Proof search direction |
| Value function | Interface satisfaction measure |
| Convergence | Compositional correctness |

$\square$

---

## Connections to Compositional Proof Systems

### 1. Separation Logic (Reynolds 2002)

**Classical Framework.** Separation logic extends Hoare logic with:
- Separating conjunction $P * Q$: heap split into $P$ and $Q$ parts
- Frame rule: local reasoning about heap modifications

**Frame Rule:**
$$\frac{\{P\} C \{Q\}}{\{P * R\} C \{Q * R\}}$$

**Connection to Adjoint Surgery:**
- **Frame $R$**: Interface boundary condition
- **Separation $*$**: Compositional structure
- **Frame soundness**: Gradient alignment at boundary
- **Local reasoning**: Primal optimization with dual constraints

**Alignment Interpretation:**
The frame rule ensures that local modifications ($P \to Q$) preserve interface conditions ($R$). This is precisely the gradient alignment condition: modifying the proof in direction $\nabla C$ preserves interface constraint $g = R$.

### 2. Refinement Types (Xi & Pfenning 1999)

**Classical Framework.** Types refined with predicates:
$$\{x : \text{Int} \mid x > 0\}$$

**Compositional Rule:**
$$\frac{f: \{x: A \mid P\} \to \{y: B \mid Q\} \quad g: \{y: B \mid Q\} \to \{z: C \mid R\}}{g \circ f: \{x: A \mid P\} \to \{z: C \mid R\}}$$

**Connection to Adjoint Surgery:**
- **Refinement predicate**: Interface constraint $g$
- **Type composition**: Saddle-point optimization
- **Subtyping check**: Dual feasibility $\lambda \geq 0$
- **Refinement inference**: Lagrange multiplier computation

### 3. Dependent Type Theory (Martin-Lof 1984)

**Classical Framework.** Types depend on values:
$$\Pi(x: A). B(x)$$

**Compositional Alignment:**
For composition $f: A \to B$ and $g: B \to C$:
$$g \circ f: A \to C$$

requires $\text{codomain}(f) = \text{domain}(g)$.

**Connection to Adjoint Surgery:**
- **Dependent types**: Varying interface conditions
- **Type checking**: Constraint satisfaction $g = 0$
- **Unification**: Computing $\lambda^*$ (interface witness)
- **Coherence**: Gradient alignment

### 4. Process Algebra (Milner 1989)

**Classical Framework.** Concurrent processes with synchronization:
$$P \| Q \xrightarrow{a} P' \| Q'$$

**Synchronization Constraint:**
$$P \xrightarrow{\bar{a}} P' \quad Q \xrightarrow{a} Q' \quad \Rightarrow \quad P \| Q \xrightarrow{\tau} P' \| Q'$$

**Connection to Adjoint Surgery:**
- **Channel type**: Interface $I$
- **Synchronization**: Boundary alignment
- **Deadlock freedom**: Feasibility $g = 0$
- **Bisimulation**: Gradient alignment (equivalent behaviors)

### 5. Session Types (Honda 1993)

**Classical Framework.** Types for communication protocols:
$$S = !\text{Int}.?\text{Bool}.S'$$

**Duality Requirement:**
$$\text{dual}(!T.S) = ?T.\text{dual}(S)$$

**Connection to Adjoint Surgery:**
- **Session type**: Interface protocol
- **Duality**: Primal-dual structure
- **Progress**: $\min_x \max_\lambda$ optimization
- **Type preservation**: Conservation under communication

---

## Certificate Construction

**Compositional Alignment Certificate:**

```
K_Align = {
    mode: "Boundary_Correction",
    mechanism: "Adjoint_Variables",

    components: {
        primal_modules: [pi_1, pi_2, ..., pi_k],
        interfaces: [I_12, I_23, ..., I_{(k-1)k}],
        specifications: [(Pre_i, Post_i) for each module]
    },

    lagrangian: {
        objective: C(pi) = sum_i C_i(pi_i),
        constraints: g(pi) = [Post_i|_I - Pre_{i+1}|_I = 0],
        multipliers: lambda = [lambda_12, lambda_23, ...]
    },

    saddle_point: {
        primal_optimal: pi*,
        dual_optimal: lambda*,
        kkt_satisfied: true
    },

    alignment: {
        gradient_cost: nabla_pi C(pi*),
        gradient_constraint: nabla_pi g(pi*),
        parallel: nabla_C || nabla_g,
        multiplier_relation: nabla_C = -lambda* . nabla_g
    },

    certificates: {
        component_proofs: [K_1, K_2, ..., K_k],
        interface_witnesses: [lambda_12*, lambda_23*, ...],
        composite_proof: K_composite
    },

    verification: {
        primal_feasibility: g(pi*) = 0,
        dual_feasibility: lambda* >= 0,
        complementary_slackness: lambda_i* . g_i(pi*) = 0,
        stationarity: nabla_C + lambda* . nabla_g = 0
    }
}
```

---

## Quantitative Summary

| Property | Bound/Guarantee |
|----------|-----------------|
| Interface conditions | $g(\pi^*) = 0$ (exact satisfaction) |
| Gradient alignment | $\nabla C \parallel \nabla g$ (parallel) |
| Composite cost | $C(\pi^*) = \sum_i C_i(\pi_i^*)$ (additive) |
| Verification overhead | $O(\sum_i |I_i|)$ (interface size) |
| Convergence rate | $O(1/\sqrt{t})$ for saddle-point methods |

### Problem-Specific Applications

| Domain | Interface Type | Alignment Condition |
|--------|---------------|---------------------|
| Type checking | Type signature | $\text{typeof}(e) = T$ |
| Hoare logic | Pre/post conditions | $\text{Post}_i \Rightarrow \text{Pre}_{i+1}$ |
| Session types | Protocol duality | $S = \text{dual}(S')$ |
| Separation logic | Frame preservation | $P * R \Rightarrow Q * R$ |
| Model checking | Alphabet matching | $\Sigma_1 \cap \Sigma_2$ compatible |

---

## Extended Connections

### 1. Compositional Model Checking

**Classical Problem.** Verify $M \models \phi$ by decomposing:
$$M = M_1 \| M_2 \| \cdots \| M_k$$

**Assume-Guarantee Rules:**
$$\frac{\langle A_i \rangle M_i \langle G_i \rangle \quad \bigwedge_i (G_i \Rightarrow A_{i+1})}{\langle A_1 \rangle M \langle G_k \rangle}$$

**Alignment = Interface Compatibility:**
- Each $G_i \Rightarrow A_{i+1}$ is an interface constraint
- Lagrange multipliers track importance of each interface
- Saddle-point finds minimal assumptions

### 2. Modular Arithmetic Circuits

**Problem.** Verify correctness of composed circuits:
$$C = C_1 \circ C_2 \circ \cdots \circ C_k$$

**Interface = Wire Type:**
- Input/output wire counts must match
- Signal types must be compatible
- Timing constraints preserved

**Alignment via Constraint Propagation:**
- Forward pass: compute outputs from inputs
- Backward pass: propagate constraints (dual variables)
- Convergence: circuit is correctly composed

### 3. Proof-Carrying Code (Necula 1997)

**Framework.** Program + proof shipped together:
$$(P, \pi)$$ where $\pi: \text{Spec}(P)$

**Compositional PCC:**
- Module $M_i$ carries proof $\pi_i$
- Interface proof $\lambda_{ij}$: $\text{Post}_i \Rightarrow \text{Pre}_j$
- Composite proof: $\pi = \pi_1 \circ \lambda_{12} \circ \pi_2 \circ \cdots$

**Alignment = Proof Linking:**
- Interface proofs serve as Lagrange multipliers
- Linking ensures all interface obligations met
- Verified composition has complete proof

### 4. Gradual Typing (Siek & Taha 2006)

**Framework.** Mix static and dynamic typing:
$$\text{Int} \sim ? \sim \text{String}$$

**Interface = Cast:**
- Casts mediate type mismatches
- Runtime checks enforce alignment
- Gradual guarantee: static types predict runtime

**Connection to Adjoint Surgery:**
- Cast = interface adapter $\lambda$
- Type mismatch = constraint violation $g \neq 0$
- Cast insertion = saddle-point optimization
- Blame = dual variable identifying failure

### 5. Incremental Computation (Acar 2005)

**Framework.** Recompute only changed parts:
$$\text{result}' = f(\text{input}')$$

**Self-Adjusting Computation:**
- Track dependencies between modules
- Propagate changes through interfaces
- Recompute only affected components

**Connection to Adjoint Surgery:**
- Dependency graph = interface structure
- Change propagation = costate dynamics $\dot{\lambda}$
- Incremental update = local primal optimization
- Correctness = gradient alignment preserved

---

## Summary

The ACT-Align theorem, translated to complexity theory, establishes:

**1. Saddle-Point Structure:**
$$\boxed{\min_\pi \max_\lambda \mathcal{L}(\pi, \lambda) = C(\pi) + \lambda^T g(\pi)}$$

Compositional verification is a primal-dual optimization:
- Primal: minimize verification cost
- Dual: enforce interface constraints

**2. KKT = Compositional Correctness:**
$$\boxed{\nabla C + \lambda^T \nabla g = 0 \quad \land \quad g(\pi^*) = 0}$$

The optimality conditions ensure:
- Interface constraints satisfied ($g = 0$)
- Local optima are globally consistent (stationarity)

**3. Gradient Alignment = Modular Soundness:**
$$\boxed{\nabla_\pi C \parallel \nabla_\pi g}$$

Cost reduction and interface preservation are aligned:
- No conflict between local and global optimization
- Modular reasoning is sound

**Physical Interpretation (Computational Analogue):**

- **Primal $x$** = Module implementations/proofs
- **Dual $\lambda$** = Interface witnesses/certificates
- **Objective $f$** = Verification cost
- **Constraint $g = 0$** = Interface matching
- **Gradient alignment** = Compositional soundness

**The Alignment Certificate:**

$$K_{\text{Align}}^+ = \begin{cases}
\pi^* & \text{composite proof} \\
\lambda^* & \text{interface witnesses} \\
\nabla C \parallel \nabla g & \text{gradient alignment} \\
g(\pi^*) = 0 & \text{interface satisfaction}
\end{cases}$$

This translation reveals that ACT-Align is a generalization of fundamental principles in compositional verification: **interface conditions** (boundary matching) are enforced via **dual variables** (interface witnesses) with **gradient alignment** ensuring that **modular reasoning is sound** (local optima are globally consistent).

---

## Literature

1. **Pontryagin, L. S. (1962).** *The Mathematical Theory of Optimal Processes.* Wiley. *Optimal control and costate dynamics.*

2. **Lions, J. L. (1971).** *Optimal Control of Systems Governed by Partial Differential Equations.* Springer. *PDE control and adjoint methods.*

3. **Konda, V. R. & Tsitsiklis, J. N. (2003).** "On Actor-Critic Algorithms." *SIAM Journal on Control and Optimization.* *Actor-critic convergence.*

4. **Bertsekas, D. P. (2019).** *Reinforcement Learning and Optimal Control.* Athena Scientific. *Unified optimal control view.*

5. **Reynolds, J. C. (2002).** "Separation Logic: A Logic for Shared Mutable Data Structures." *LICS.* *Compositional heap reasoning.*

6. **O'Hearn, P., Yang, H., & Reynolds, J. C. (2001).** "Local Reasoning about Programs that Alter Data Structures." *CSL.* *Frame rule and locality.*

7. **Xi, H. & Pfenning, F. (1999).** "Dependent Types in Practical Programming." *POPL.* *Refinement types.*

8. **Martin-Lof, P. (1984).** *Intuitionistic Type Theory.* Bibliopolis. *Dependent type theory.*

9. **Milner, R. (1989).** *Communication and Concurrency.* Prentice Hall. *Process algebra.*

10. **Honda, K. (1993).** "Types for Dyadic Interaction." *CONCUR.* *Session types.*

11. **Necula, G. C. (1997).** "Proof-Carrying Code." *POPL.* *Compositional code verification.*

12. **Siek, J. G. & Taha, W. (2006).** "Gradual Typing for Functional Languages." *Scheme Workshop.* *Gradual types and casts.*

13. **Acar, U. A. (2005).** "Self-Adjusting Computation." *PhD Thesis, CMU.* *Incremental computation.*

14. **Pnueli, A. (1985).** "In Transition from Global to Modular Temporal Reasoning about Programs." *Logics and Models of Concurrent Systems.* *Compositional model checking.*

15. **Clarke, E. M., Long, D. E., & McMillan, K. L. (1989).** "Compositional Model Checking." *LICS.* *Assume-guarantee reasoning.*
