---
title: "FACT-Barrier - Complexity Theory Translation"
---

# FACT-Barrier: Barrier Synthesis as Certificate Generation

## Overview

This document provides a complete complexity-theoretic translation of the FACT-Barrier (Barrier Implementation Factory) metatheorem from the hypostructure framework. The theorem establishes that for any system type, there exist default barrier implementations that generate correct certificates (blocked/breached) with non-circular preconditions. In complexity theory terms, this corresponds to **Barrier Synthesis**: generating certificates that prove constraints are satisfied (blocked) or violated (breached).

**Original Theorem Reference:** {prf:ref}`mt-fact-barrier`

---

## Complexity Theory Statement

**Theorem (FACT-Barrier, Computational Form).**
Let $\mathcal{C} = (V, E, \phi)$ be a constraint satisfaction system where:
- $V$ is a set of constraint variables
- $E$ is a set of constraint edges (dependencies)
- $\phi: V \to \mathbb{R}$ is a potential/barrier function

A **barrier synthesis procedure** $\mathcal{B}: \text{State} \times \text{Constraint} \to \text{Certificate}$ is **complete** if:

1. **Blocked Certificate:** If constraint $C$ is satisfiable at state $s$, produce witness $K^{\mathrm{blk}} = (s, w)$ where $w \vdash C(s)$
2. **Breached Certificate:** If constraint $C$ is violated at state $s$, produce counterexample $K^{\mathrm{br}} = (s, v, \sigma)$ where $v$ is the violation witness and $\sigma$ is surgery routing data
3. **Non-Circularity:** The trigger predicate for barrier $B$ is not in the precondition set of the gate that invokes $B$
4. **Completeness:** Every gate failure path has at least one applicable barrier

**Formal Statement.** Given:
1. A type specification $T$ with constraint set $\{C_1, \ldots, C_n\}$
2. Literature lemmas $\mathcal{L} = \{L_1, \ldots, L_m\}$ providing barrier methods
3. Gate failure certificates $\{K_i^-\}_{i \in I}$ as triggers

The barrier factory produces for each barrier $B_j$:
$$\mathcal{B}_j^T: K_i^- \times \Gamma \to K_j^{\mathrm{blk}} \sqcup K_j^{\mathrm{br}}$$

where:
- $K_j^{\mathrm{blk}}$ witnesses that the obstruction is transient (constraint will be satisfied)
- $K_j^{\mathrm{br}}$ witnesses that the obstruction persists (constraint violated, surgery needed)

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Barrier $\mathcal{B}$ | Constraint verifier / Certificate generator | Decides feasibility and produces witness |
| Blocked certificate $K^{\mathrm{blk}}$ | Feasibility witness | Proof that constraint is satisfiable |
| Breached certificate $K^{\mathrm{br}}$ | Infeasibility certificate / Farkas witness | Proof that constraint is violated |
| Gate NO outcome $K_i^-$ | Constraint violation trigger | Input to barrier synthesis |
| Foster-Lyapunov barrier | Drift condition / Supermartingale certificate | $\mathcal{L}V \leq -\gamma V + C\mathbf{1}_K$ |
| Scattering barrier | Dispersion estimate / Strichartz bound | $\|u\|_{L^p L^q} \leq C$ decay certificate |
| Type II barrier | Monotonicity violation detector | Scale-critical obstruction witness |
| Capacity barrier | Epsilon-regularity certificate | $\mathcal{H}^{n-2}(\Sigma) = 0$ measure witness |
| Trigger predicate $\mathrm{Trig}(\mathcal{B})$ | Activation condition | When barrier synthesis is invoked |
| Precondition $\mathrm{Pre}(V)$ | Required prior certificates | Dependencies for barrier evaluation |
| Non-circularity | Acyclic dependency graph | DAG structure of certificate flow |
| Surgery routing | Repair/relaxation procedure | How to handle breached constraints |
| Barrier soundness | Certificate correctness | Witnesses actually prove the claim |
| Barrier completeness | Total coverage | Every failure has a barrier |
| Certificate payload | Witness structure | Complete data for downstream consumers |
| Epsilon-regularity | Small-set exclusion | Singular set has zero capacity |
| Drift bound | Expected decrease | Value function decreases in expectation |
| Geometric ergodicity | Rapid mixing | Exponential convergence to stationary |

---

## Barrier Synthesis Framework

### Definition (Barrier Certificate System)

A **barrier certificate system** is a tuple $\mathcal{B} = (\mathcal{C}, \mathcal{V}, \mathcal{W}, \mathcal{S})$ where:

1. **Constraints $\mathcal{C}$:** A set of predicates $\{C_1, \ldots, C_n\}$ over state space $\mathcal{X}$
2. **Verifiers $\mathcal{V}$:** For each $C_i$, a verifier $V_i: \mathcal{X} \to \{\text{SAT}, \text{UNSAT}, \text{UNKNOWN}\}$
3. **Witnesses $\mathcal{W}$:** Certificate types $W_i^+$ (satisfaction) and $W_i^-$ (violation)
4. **Synthesis $\mathcal{S}$:** Algorithms producing witnesses from verification outcomes

### Definition (Blocked/Breached Dichotomy)

For constraint $C$ at state $s$, the barrier produces:

**Blocked:** $K^{\mathrm{blk}} = (\text{barrier\_type}, \text{obstruction}, \text{bound}, \text{literature\_ref})$
- Meaning: The obstruction is *transient*; the system will eventually satisfy $C$
- Example: Foster-Lyapunov drift ensures return to compact set

**Breached:** $K^{\mathrm{br}} = (\text{mode}, \text{profile}, \text{surgery\_data}, \text{capacity})$
- Meaning: The obstruction is *persistent*; surgical intervention needed
- Example: Singular set has positive capacity, epsilon-regularity fails

### Definition (Non-Circularity Condition)

Barrier $\mathcal{B}_j$ is **non-circular** with respect to gate $V_i$ if:
$$\mathrm{Trig}(\mathcal{B}_j) \cap \mathrm{Pre}(V_i) = \emptyset$$

This ensures:
- The barrier cannot depend on the gate that triggers it
- Certificate dependencies form a DAG
- No infinite regress in verification

---

## Proof Sketch

### Setup: Constraint Verification as Certificate Generation

**Translation.** We model barrier synthesis as a certificate generation procedure for constraint satisfaction:

- **Input:** State $s \in \mathcal{X}$, constraint $C$, context $\Gamma$ (prior certificates)
- **Output:** Certificate $K$ witnessing satisfaction or violation
- **Invariant:** Certificates are independently verifiable

**Correspondence to Hypostructure:**

| Sieve Component | Barrier Synthesis Component |
|-----------------|----------------------------|
| Gate NO outcome | Constraint violation detected |
| Barrier activation | Certificate generation invoked |
| Blocked certificate | Feasibility witness produced |
| Breached certificate | Infeasibility witness produced |
| Surgery routing | Constraint relaxation / repair |
| Non-circularity | Acyclic witness dependencies |

### Step 1: Barrier Catalog Construction

**Claim.** For each constraint type, there exists a corresponding barrier method from the literature.

**Proof.** The barrier catalog maps gate failures to known verification techniques:

| Gate Failure | Barrier Type | Certificate Method | Literature |
|--------------|-------------|-------------------|------------|
| EnergyCheck NO | Foster-Lyapunov | Drift condition $\mathcal{L}V \leq -\gamma V + C$ | {cite}`MeynTweedie93` |
| CompactCheck NO | Scattering | Dispersion estimate $\|u\|_{L^p L^q} \leq C$ | {cite}`Tao06` |
| ScaleCheck NO | Type II | Monotonicity formula violation | {cite}`Hamilton82` |
| GeomCheck NO | Capacity | Epsilon-regularity $\mathcal{H}^{n-2}(\Sigma) = 0$ | {cite}`CaffarelliKohnNirenberg82` |

**Instantiation Protocol:**
1. Extract constraint structure from type $T$
2. Match to literature barrier template
3. Substitute type-specific functionals $(\Phi, \mathfrak{D})$
4. Produce parameterized barrier $\mathcal{B}_j^T$

This is analogous to **constraint instantiation** in SAT/SMT solvers: generic constraint schemas are instantiated with problem-specific terms.

### Step 2: Non-Circularity Verification

**Claim.** The dependency structure of barriers is acyclic.

**Proof.** For each barrier $\mathcal{B}_j$ triggered by gate $V_i$:

1. **Trigger Source:** $\mathrm{Trig}(\mathcal{B}_j)$ uses data from $K_i^-$ (the gate's NO certificate)
2. **Precondition Source:** $\mathrm{Pre}(V_i)$ uses data from prior context $\Gamma$
3. **Temporal Ordering:** $K_i^-$ is produced *after* $V_i$ evaluates, so $K_i^- \not\in \Gamma$

Therefore:
$$\mathrm{Trig}(\mathcal{B}_j) \subseteq \{K_i^-\} \cup \text{derived data}$$
$$\mathrm{Pre}(V_i) \subseteq \Gamma$$
$$\{K_i^-\} \cap \Gamma = \emptyset$$

This gives $\mathrm{Trig}(\mathcal{B}_j) \cap \mathrm{Pre}(V_i) = \emptyset$.

**Complexity Interpretation:** The certificate dependency graph is a DAG. This is equivalent to topological sortability of the verification procedure, ensuring termination.

### Step 3: Barrier Soundness (Two Directions)

**Claim.** Barrier certificates correctly witness their claims.

**Proof (Blocked Soundness):**

If $\mathcal{B}_j$ returns $K_j^{\mathrm{blk}}$, then the obstruction cannot persist.

**Example (Foster-Lyapunov):** Given drift condition:
$$\mathcal{L}V(x) \leq -\gamma V(x) + C \cdot \mathbf{1}_K(x)$$

By {cite}`MeynTweedie93` Theorem 15.0.1:
- The process is geometrically ergodic
- Unbounded excursions are transient (finite expected return time)
- Energy constraint will eventually be satisfied

The certificate $K^{\mathrm{blk}} = (V, \gamma, C, K)$ witnesses these parameters.

**Proof (Breached Soundness):**

If $\mathcal{B}_j$ returns $K_j^{\mathrm{br}}$, the barrier method is insufficient.

**Example (Capacity Barrier):** If $\mathrm{Cap}(\Sigma) > \varepsilon_{\text{reg}}$:
- Epsilon-regularity ({cite}`CaffarelliKohnNirenberg82`) cannot be applied
- Singular set may carry non-zero measure
- Singularity is *not excluded* (but also *not proven to exist*)

The certificate $K^{\mathrm{br}} = (\Sigma, \mathrm{Cap}(\Sigma), \varepsilon_{\text{reg}})$ routes to surgery.

**Note:** Breached is a *routing signal*, not a semantic claim. It means "this barrier cannot exclude the obstruction," not "the obstruction definitely exists."

### Step 4: Certificate Payload Construction

**Claim.** Barrier certificates contain complete information for downstream consumption.

**Certificate Structures:**

**Blocked Certificate:**
```
K^blk = {
    barrier_type: BarrierID,
    obstruction: ObstructionDescription,
    bound: NumericalBound,
    literature_ref: Citation,
    verification_data: WitnessData
}
```

**Breached Certificate:**
```
K^br = {
    mode: FailureModeID,
    profile: LocalStructure,
    surgery_data: RepairParameters,
    capacity: MeasureEstimate,
    routing: NextNodeID
}
```

**Payload Completeness:** Downstream consumers (surgery, Lock) receive all data needed without re-querying the barrier. This corresponds to **witness self-containment** in proof systems.

### Step 5: Completeness (Full Coverage)

**Claim.** Every gate NO path has at least one applicable barrier.

**Proof.** By construction of the barrier catalog:

1. The catalog covers all 17 gates with NO outcomes
2. Each barrier terminates (finite computation on bounded data)
3. Union coverage: $\bigcup_j \mathrm{Trig}(\mathcal{B}_j) \supseteq \bigcup_i \{K_i^-\}$

No gate failure is orphaned; every failure triggers at least one barrier.

**Complexity Interpretation:** This is analogous to **complete case analysis** in proof by cases. Every branch of the decision tree leads to a defined outcome.

---

## Connections to Barrier Methods in Optimization

### 1. Interior Point Methods (Karmarkar 1984, Nesterov-Nemirovski 1994)

**Classical Result.** Interior point methods solve linear and convex programs by following a central path defined by barrier functions:
$$\min c^T x \quad \text{s.t.} \quad Ax = b, \, x \geq 0$$

The logarithmic barrier transforms this to:
$$\min c^T x - \mu \sum_i \log x_i$$

**Connection to FACT-Barrier:**

| Interior Point Method | Hypostructure Analog |
|----------------------|---------------------|
| Feasible region | State space satisfying constraints |
| Barrier function $-\sum \log x_i$ | Barrier functional $\mathcal{B}(x)$ |
| Barrier parameter $\mu$ | Barrier threshold / tolerance |
| Central path | Trajectory through constraint-satisfying states |
| Barrier goes to $\infty$ at boundary | Breached certificate at constraint violation |
| Finite barrier value | Blocked certificate (constraint satisfied) |

**Certificate Interpretation:**
- $\mathcal{B}(x) < \infty$: Point is interior (blocked - constraint satisfied)
- $\mathcal{B}(x) \to \infty$: Point approaches boundary (near breach)
- $\mathcal{B}(x) = \infty$: Point is infeasible (breached - constraint violated)

**Algorithmic Correspondence:**
- Interior point iteration = Sieve DAG traversal
- Reducing $\mu$ = Tightening constraint verification
- Convergence = Reaching VICTORY or classified failure

### 2. LP Duality and Farkas Certificates

**Classical Result (Farkas' Lemma).** For $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$:

Either $\exists x \geq 0: Ax = b$ (feasible) or $\exists y: A^T y \geq 0, b^T y < 0$ (infeasible certificate)

**Connection to FACT-Barrier:**

| Farkas Certificate | Hypostructure Analog |
|-------------------|---------------------|
| Feasibility witness $x \geq 0$ | Blocked certificate $K^{\mathrm{blk}}$ |
| Infeasibility certificate $y$ | Breached certificate $K^{\mathrm{br}}$ |
| Separating hyperplane | Barrier surface |
| $b^T y < 0$ | Obstruction measure $> 0$ |
| Dual solution | Surgery routing data |

**Structural Correspondence:**

The Blocked/Breached dichotomy is a generalization of Farkas duality:
- **Blocked:** Primal-feasible witness exists
- **Breached:** Dual certificate of infeasibility exists

Both outcomes produce *constructive certificates* that can be independently verified.

### 3. SDP Relaxations and Sum-of-Squares Certificates

**Classical Result.** Semidefinite programming (SDP) provides certificates for polynomial optimization:

A polynomial $p(x) \geq 0$ for all $x$ if $p(x) = \sum_i q_i(x)^2$ (sum of squares).

**Connection to FACT-Barrier:**

| SDP Relaxation | Hypostructure Analog |
|----------------|---------------------|
| SDP feasibility | Barrier blocked |
| SOS certificate | Blocked witness (constructive proof) |
| SDP infeasibility | Barrier breached |
| Dual SDP solution | Breached witness with capacity bound |
| Hierarchy level | Barrier strength / precision |
| Convergence | Certificate refinement |

**Certificate Construction:**
- **SOS Certificate:** $p = \sum q_i^2$ provides a machine-checkable proof of non-negativity
- **Barrier Certificate:** $K^{\mathrm{blk}} = (V, \gamma, \ldots)$ provides a machine-checkable proof of constraint satisfaction

**Lasserre/Parrilo Hierarchy:**

The SDP hierarchy provides progressively stronger relaxations:
$$\text{SDP}_1 \supseteq \text{SDP}_2 \supseteq \cdots \supseteq \text{True Feasible Set}$$

This corresponds to barrier refinement in the hypostructure:
- Weak barriers may return BREACHED for marginally feasible states
- Stronger barriers (higher hierarchy level) can certify more states as BLOCKED

### 4. Barrier Certificates in Control Theory (Prajna 2004)

**Classical Result.** A barrier certificate $B(x)$ proves safety for dynamical systems:
$$B(x_0) \leq 0, \quad B(x_u) > 0 \quad \forall x_u \in \mathcal{X}_u, \quad \dot{B}(x) \leq 0$$

If such $B$ exists, trajectories from $x_0$ never reach unsafe set $\mathcal{X}_u$.

**Connection to FACT-Barrier:**

| Control Barrier Certificate | Hypostructure Analog |
|----------------------------|---------------------|
| Initial region $B(x_0) \leq 0$ | Initial state satisfies constraints |
| Unsafe region $B(x_u) > 0$ | Singular/failure states |
| Flow condition $\dot{B} \leq 0$ | Dissipation $\mathfrak{D} \geq 0$ |
| Certificate existence | BLOCKED outcome |
| No certificate found | BREACHED outcome |
| Safety proof | Certificate chain to VICTORY |

**Lyapunov-Barrier Duality:**
- Lyapunov functions certify *convergence* (reaching good states)
- Barrier functions certify *avoidance* (not reaching bad states)
- The hypostructure uses both: Lyapunov for equilibration, barriers for singularity exclusion

### 5. SAT/SMT Conflict Analysis and UNSAT Certificates

**Classical Result.** Modern SAT solvers produce resolution proofs (UNSAT certificates) when a formula is unsatisfiable.

**Connection to FACT-Barrier:**

| SAT/SMT | Hypostructure Analog |
|---------|---------------------|
| SAT (satisfiable) | Blocked (constraint satisfiable) |
| UNSAT (unsatisfiable) | Breached (constraint violated) |
| Satisfying assignment | Blocked witness |
| Resolution proof | Breached witness chain |
| Conflict clause | Obstruction certificate |
| Unit propagation | Certificate implication |
| Backtracking | Surgery/retry |
| CDCL learning | Literature lemma application |

**Certificate Verification:**
- SAT certificate: Check assignment satisfies all clauses - $O(n)$
- UNSAT certificate: Check resolution proof is valid - $O(|proof|)$
- Barrier certificates: Verify witness data matches claim - $O(|K|)$

---

## Certificate Construction

### Barrier Certificate Schema

```
BarrierCertificate = {
    barrier_id: BarrierIdentifier,
    trigger: GateNOCertificate,
    outcome: Blocked | Breached,

    // Blocked payload
    blocked_data: Option<{
        obstruction_type: String,
        bound: Numerical,
        transience_proof: WitnessData,
        literature: Citation
    }>,

    // Breached payload
    breached_data: Option<{
        failure_mode: ModeIdentifier,
        local_profile: StructuralData,
        capacity_bound: Numerical,
        surgery_routing: SurgeryID,
        admissibility_data: AdmissibilityInput
    }>,

    // Verification metadata
    verification: {
        algorithm: String,
        complexity: ComplexityBound,
        checkable: Boolean
    }
}
```

### Verification Algorithm

```
function VerifyBarrierCertificate(K: BarrierCertificate):
    // Check trigger is valid
    if not ValidNOCertificate(K.trigger):
        return INVALID("Invalid trigger certificate")

    // Check non-circularity
    if Intersects(Trigger(K), Preconditions(K.trigger.gate)):
        return INVALID("Circular dependency detected")

    match K.outcome:
        case Blocked:
            // Verify transience claim
            if not VerifyTransience(K.blocked_data):
                return INVALID("Transience proof invalid")

            // Check literature reference
            if not ValidCitation(K.blocked_data.literature):
                return INVALID("Invalid literature reference")

            return VALID_BLOCKED

        case Breached:
            // Verify capacity bound
            if not ValidCapacity(K.breached_data.capacity_bound):
                return INVALID("Invalid capacity bound")

            // Check surgery routing exists
            if not ValidSurgery(K.breached_data.surgery_routing):
                return INVALID("Invalid surgery routing")

            return VALID_BREACHED
```

### Complexity of Barrier Synthesis

| Operation | Time Complexity | Certificate Size |
|-----------|----------------|------------------|
| Foster-Lyapunov verification | $O(n^3)$ (SDP) | $O(n^2)$ (matrix) |
| Capacity computation | $O(|\Sigma| \cdot n)$ | $O(|\Sigma|)$ |
| Epsilon-regularity check | $O(n^d)$ for $d$-dim | $O(1)$ (bound) |
| Scattering estimate | $O(T \cdot n)$ | $O(1)$ (norm bound) |
| Non-circularity check | $O(|E|)$ (DAG edges) | $O(1)$ |
| Certificate verification | $O(|K|)$ | N/A |

---

## Quantitative Summary

| Property | Bound |
|----------|-------|
| Barrier catalog size | $\leq$ 17 (one per gate) |
| Certificate production time | Polynomial in state dimension |
| Certificate size | $O(n^2)$ for matrix witnesses |
| Verification time | $O(|K|)$ linear in certificate |
| Dependency graph depth | $\leq$ DAG depth of Sieve |
| Coverage guarantee | 100% (every NO has a barrier) |

---

## Algorithmic Implications

### Barrier Synthesis Pattern

The FACT-Barrier theorem suggests an implementation pattern for constraint verification:

```
function SynthesizeBarrier(gate_failure: K_minus, context: Gamma):
    // Step 1: Identify barrier type from gate
    barrier_type := BarrierCatalog[gate_failure.gate_id]

    // Step 2: Instantiate barrier for type T
    barrier := InstantiateBarrier(barrier_type, T.functionals)

    // Step 3: Evaluate barrier
    (outcome, data) := EvaluateBarrier(barrier, gate_failure, context)

    // Step 4: Produce certificate
    match outcome:
        case TRANSIENT:
            return K_blocked(barrier_type, data.bound, data.literature)

        case PERSISTENT:
            return K_breached(data.mode, data.profile, data.surgery_routing)

        case UNKNOWN:
            // Conservative: treat as breached, route to surgery
            return K_breached(UNKNOWN_MODE, data, DEFAULT_SURGERY)
```

### Integration with Optimization Solvers

For computational implementations, barrier synthesis can leverage existing solvers:

```
function ComputeBarrierCertificate(constraint, state):
    // Formulate as optimization problem
    problem := FormulateAsSDP(constraint, state)

    // Solve with interior point method
    (status, solution) := SolveSDP(problem)

    match status:
        case FEASIBLE:
            // Extract blocked certificate from primal solution
            return ExtractBlockedCertificate(solution)

        case INFEASIBLE:
            // Extract breached certificate from dual solution
            return ExtractBreachedCertificate(solution.dual)

        case UNKNOWN:
            return InconclsuiveCertificate(problem)
```

---

## Summary

The FACT-Barrier theorem, translated to complexity theory, establishes that:

1. **Barrier synthesis is well-defined:** For every constraint type, there exists a certificate generation procedure based on established mathematical techniques.

2. **Certificates are sound:** Blocked certificates genuinely witness transience of obstructions; breached certificates genuinely witness persistence (within the barrier's scope).

3. **Dependencies are acyclic:** The non-circularity condition ensures barrier synthesis terminates and certificates form a valid proof DAG.

4. **Coverage is complete:** Every gate failure has at least one applicable barrier, ensuring no verification dead-ends.

5. **Certificates are constructive:** Both blocked and breached outcomes produce explicit witness data that can be independently verified.

This translation reveals that FACT-Barrier generalizes several classical constructions:

| Classical Construction | FACT-Barrier Instance |
|-----------------------|----------------------|
| Interior point barriers | State constraint satisfaction |
| Farkas lemma duality | Blocked/Breached dichotomy |
| SOS certificates | Polynomial constraint witnesses |
| Control barrier certificates | Safety/avoidance proofs |
| SAT/UNSAT certificates | Discrete constraint witnesses |

The key insight is that **barrier methods provide a uniform framework for constraint verification with constructive certificates**, bridging continuous optimization (interior point, SDP) and discrete verification (SAT, SMT).

---

## Literature

**Barrier Methods in Optimization:**
- {cite}`Nesterov94`: Interior-Point Polynomial Algorithms in Convex Programming. The logarithmic barrier corresponds to constraint satisfaction certificates.
- {cite}`Karmarkar84`: A New Polynomial-Time Algorithm for Linear Programming. Interior point methods follow central path defined by barriers.

**Duality and Certificates:**
- {cite}`Farkas02`: Original Farkas lemma. Primal feasibility vs. dual infeasibility certificates correspond to Blocked/Breached.
- {cite}`Schrijver86`: Theory of Linear and Integer Programming. Comprehensive treatment of LP duality and certificates.

**SDP and SOS Certificates:**
- {cite}`Lasserre01`: Global Optimization with Polynomials and the Problem of Moments. SOS hierarchy provides progressively stronger certificates.
- {cite}`Parrilo00`: Structured Semidefinite Programs and Semialgebraic Geometry. SDP-based certificate construction.

**Control Barrier Certificates:**
- {cite}`Prajna04`: Safety Verification of Hybrid Systems Using Barrier Certificates. Control-theoretic barrier certificates for safety verification.
- {cite}`Ames19`: Control Barrier Functions. Modern treatment of barrier functions in control.

**Probabilistic Barriers:**
- {cite}`MeynTweedie93`: Markov Chains and Stochastic Stability. Foster-Lyapunov conditions for geometric ergodicity.
- {cite}`Glynn08`: Bounding Stationary Expectations of Markov Chains. Drift conditions for probabilistic barriers.

**SAT/SMT Certificates:**
- {cite}`Nieuwenhuis06`: Solving SAT and SAT Modulo Theories. CDCL and certificate generation.
- {cite}`Barrett18`: Satisfiability Modulo Theories. SMT solving with certificate production.

**PDE Regularity Barriers:**
- {cite}`CaffarelliKohnNirenberg82`: Partial Regularity of Suitable Weak Solutions. Epsilon-regularity and capacity barriers.
- {cite}`Hamilton82`: Three-Manifolds with Positive Ricci Curvature. Singularity barriers in geometric flows.
