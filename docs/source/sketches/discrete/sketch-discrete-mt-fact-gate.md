---
title: "FACT-Gate - Complexity Theory Translation"
---

# FACT-Gate: Verifier Synthesis and Decidability Classification

## Overview

This document provides a complete complexity-theoretic translation of the FACT-Gate (Gate Evaluator Factory) metatheorem from the hypostructure framework. The theorem establishes that the factory generates **Correct-by-Construction** code for all 17 gate nodes. In complexity theory terms, this corresponds to **Verifier Synthesis**: automatic generation of verifiers from specifications with decidability classification across the arithmetical hierarchy.

**Original Theorem Reference:** {prf:ref}`mt-fact-gate`

---

## Complexity Theory Statement

**Theorem (FACT-Gate, Computational Form).**
Let $\mathcal{S} = (\mathcal{T}, \mathcal{P}, \mathcal{V})$ be a verification system where:
- $\mathcal{T}$ is a type specification language
- $\mathcal{P} = \{P_1, \ldots, P_n\}$ is a set of predicates with decidability classifications
- $\mathcal{V}$ is a verifier synthesis function

The **Verifier Synthesis Problem** is: given a predicate specification $P_i \in \mathcal{P}$, automatically generate a verifier $V_i$ such that:

1. **Soundness:** $V_i(x) = \text{YES} \Rightarrow P_i(x)$ holds
2. **Termination:** $V_i$ halts on all inputs (possibly with INCONCLUSIVE)
3. **Completeness (where possible):** If $P_i$ is decidable, then $V_i(x) = \text{YES} \Leftrightarrow P_i(x)$

**Formal Statement.** Given:
1. A specification functor $\mathcal{T}: \mathbf{Type} \to \mathbf{Pred}$ mapping types to predicate systems
2. A verifier functor $\mathcal{V}: \mathbf{Pred} \to \mathbf{Verifier}$ mapping predicates to certified verifiers
3. The factory composition $\mathcal{F} = \mathcal{V} \circ \mathcal{T}$

For each gate $i \in \{1, \ldots, 17\}$, the factory produces:
- Predicate instantiation $P_i^T$
- Verifier $V_i^T: X \times \Gamma \to \{\text{YES}, \text{NO}, \text{INC}\} \times \mathcal{K}_i$

satisfying the **Soundness Property**:
$$V_i^T(x, \Gamma) = (\text{YES}, K_i^+) \Rightarrow P_i^T(x)$$

**Corollary (Rice's Theorem Boundary).** For predicates in $\Pi_2^0$ or higher, no total computable verifier can be both sound and complete. The factory respects this boundary by producing verifiers that:
- Return YES only when a finite witness exists
- Return NO only when a finite refutation exists
- Return INCONCLUSIVE (INC) when the predicate is undecidable on the given input

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Gate predicate $P_i^T$ | Property to be verified | Subset of inputs $\{x : P(x)\}$ |
| Verifier $V_i^T$ | Decision procedure / semidecision procedure | Turing machine with output |
| Certificate $K_i^+$ | Proof witness / accepting computation | Evidence of membership |
| Certificate $K_i^-$ | Refutation witness | Evidence of non-membership |
| Certificate $K_i^{\text{inc}}$ | Timeout / resource exhaustion | Inconclusive status |
| Type $T$ | Problem instance class | Parameterized family |
| Factory $\mathcal{F}$ | Verifier generator / synthesizer | Higher-order compiler |
| Natural transformation | Program transformation | Semantics-preserving compilation |
| Decidability class | Arithmetical hierarchy level | $\Sigma_n^0$, $\Pi_n^0$, $\Delta_n^0$ |
| $\Sigma_1^0$ (semi-decidable) | RE (recursively enumerable) | $\exists$-quantified property |
| $\Pi_1^0$ (co-semi-decidable) | co-RE | $\forall$-quantified property |
| $\Pi_2^0$ | Beyond RE $\cap$ co-RE | $\forall\exists$-quantified |
| Undecidable in general | Not in $\Delta_1^0$ | No total decision procedure |
| Timeout with $K^{\text{inc}}$ | Bounded model checking | Resource-limited verification |
| Tactic library E1-E12 | Heuristic decision procedures | Incomplete but sound methods |
| Curry-Howard correspondence | Proofs-as-programs | $K^+ \cong$ proof term |
| Context $\Gamma$ | Assumption set / oracle | Previously verified certificates |
| Witness type $W_i^T$ | Certificate format | Proof object specification |

---

## Complete Decidability Classification for All 17 Gates

The following table provides the decidability classification for each gate in the Structural Sieve:

| Gate | Name | Predicate | Decidability | Arithmetical Class | Witness Type | Undecidability Source |
|------|------|-----------|--------------|-------------------|--------------|----------------------|
| 1 | EnergyCheck | $\sup_t \Phi(u(t)) < \infty$ | Semi-decidable | $\Sigma_1^0$ | $(t^*, \Phi(u(t^*)), M)$ | Supremum over infinite time |
| 2 | ZenoCheck | Event count $N(T) < \infty$ | Semi-decidable | $\Sigma_1^0$ | $(T, N(T), \text{bound})$ | Infinite horizon counting |
| 3 | CompactCheck | $\exists V: \mu(B_\varepsilon(V)) > 0$ | Semi-decidable | $\Sigma_1^0$ | $(V, \varepsilon, \mu_{\text{witness}})$ | Profile library enumeration |
| 4 | ScaleCheck | $\alpha < \beta + \lambda_c$ | **Decidable** | $\Delta_1^0$ | $(\alpha, \beta, \lambda_c)$ | None (computable arithmetic) |
| 5 | ParamCheck | $\partial_c \mathcal{I} = 0$ | Semi-decidable | $\Sigma_1^0$ | $(c, \mathcal{I}(c), \text{stability proof})$ | Parameter space search |
| 6 | GeomCheck | $\text{Cap}(\Sigma) = 0$ | Co-semi-decidable | $\Pi_1^0$ | $(\text{capacity bound}, \varepsilon_{\text{reg}})$ | Capacity is limit of finite approx |
| 7 | StiffnessCheck | $\|\nabla\Phi\| \geq C|\Delta\Phi|^\theta$ | Undecidable | $\Pi_2^0$ | $(C, \theta, \text{gradient\_bound})$ | Infimum over manifold |
| 7a | BifurcateCheck | Dynamically unstable | Semi-decidable | $\Sigma_1^0$ | Eigenvalue witness | Spectral computation |
| 7b | SymCheck | $G$ acts non-trivially | **Decidable** | $\Delta_1^0$ | Group action witness | Finite group enumeration |
| 7c | CheckSC | Parameters stable | Semi-decidable | $\Sigma_1^0$ | Stability certificate | Modulation analysis |
| 7d | CheckTB | Tunneling action finite | Semi-decidable | $\Sigma_1^0$ | $(S, \text{action value})$ | Action computation |
| 8 | TopoCheck | Topological sector accessible | **Decidable** (finite) | $\Delta_1^0$ | $(\tau, \text{sector label})$ | Finite sector classification |
| 9 | TameCheck | O-minimal definable | **Decidable** | $\Delta_1^0$ | Definability certificate | O-minimal theory decidability |
| 10 | ErgoCheck | Spectral gap $\rho > 0$ | Semi-decidable | $\Sigma_1^0$ | $(\rho, \text{gap bound})$ | Spectral approximation |
| 11 | ComplexCheck | $K(u) < B$ | Undecidable | $\Pi_2^0$ | $(D, K(D), B)$ | Kolmogorov complexity |
| 12 | OscillateCheck | Oscillatory behavior | **Decidable** | $\Delta_1^0$ | Monotonicity witness | Finite time window |
| 13 | BoundaryCheck | System is open | **Decidable** | $\Delta_1^0$ | Boundary specification | Syntactic check |
| 14 | OverloadCheck | Input bounded | Semi-decidable | $\Sigma_1^0$ | $(M, \text{bound})$ | Supremum over inputs |
| 15 | StarveCheck | Input sufficient | Semi-decidable | $\Sigma_1^0$ | $(r_{\min}, \text{sufficiency})$ | Infimum over supply |
| 16 | AlignCheck | Control aligned | Semi-decidable | $\Sigma_1^0$ | Gradient alignment | Optimization landscape |
| 17 | Lock | $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathcal{H}) = \emptyset$ | **Undecidable** | $\Pi_2^0+$ | Obstruction cocycle | Rice's Theorem |

### Decidability Summary

| Decidability Class | Gates | Count |
|-------------------|-------|-------|
| **Decidable** ($\Delta_1^0$) | 4, 7b, 8, 9, 12, 13 | 6 |
| **Semi-decidable** ($\Sigma_1^0$) | 1, 2, 3, 5, 7a, 7c, 7d, 10, 14, 15, 16 | 11 |
| **Co-semi-decidable** ($\Pi_1^0$) | 6 | 1 |
| **Undecidable** ($\Pi_2^0$ or higher) | 7, 11, 17 | 3 |

---

## Proof Sketch for Correctness

### Setup: The Verification Synthesis Problem

**Definition (Verification Synthesis).** Given a specification $\phi$ in a formal language $\mathcal{L}$, synthesize a program $V_\phi$ such that:
$$\forall x.\ V_\phi(x) = \text{YES} \Leftrightarrow \llbracket \phi \rrbracket(x) = \text{true}$$

This is generally undecidable by Rice's Theorem. The FACT-Gate theorem circumvents this by:
1. **Weakening completeness:** Allow INCONCLUSIVE outputs
2. **Structuring predicates:** Decompose into decidability classes
3. **Providing tactics:** Use heuristic decision procedures for hard cases

### Step 1: Decidable Gates (Direct Synthesis)

**Claim.** For decidable predicates, the factory produces complete verifiers.

**Proof.** For gates in $\Delta_1^0$:

**Gate 4 (ScaleCheck):** The predicate $\alpha < \beta + \lambda_c$ involves computable real arithmetic. Given representations of $\alpha, \beta, \lambda_c$ as computable reals with effective moduli of continuity:
- Compute approximations $\tilde{\alpha}, \tilde{\beta}, \tilde{\lambda}_c$ to precision $\epsilon$
- If $\tilde{\alpha} + 2\epsilon < \tilde{\beta} + \tilde{\lambda}_c - \epsilon$: return YES
- If $\tilde{\alpha} - \epsilon > \tilde{\beta} + \tilde{\lambda}_c + 2\epsilon$: return NO
- Refine precision and repeat (terminates by density of rationals)

**Gate 8 (TopoCheck):** Topological sectors form a finite set $\mathcal{T} = \{\tau_1, \ldots, \tau_k\}$. The predicate "sector $\tau$ is accessible" reduces to:
- Enumerate all sectors (finite)
- Check homotopy equivalence (decidable for finite CW complexes)
- Return YES if accessible, NO otherwise

**Gate 9 (TameCheck):** O-minimality is decidable for fixed o-minimal structures. The predicate "definable in $\mathcal{R}_{\text{exp}}$" (real exponential field) is decidable by Wilkie's theorem for bounded formulas.

**Complexity:** Decidable gates have polynomial or exponential verifiers depending on the specific predicate.

### Step 2: Semi-decidable Gates (One-sided Synthesis)

**Claim.** For $\Sigma_1^0$ predicates, the factory produces sound verifiers that may return INCONCLUSIVE on negative instances.

**Proof.** For gates in $\Sigma_1^0$:

**Gate 1 (EnergyCheck):** The predicate $\sup_t \Phi(u(t)) < M$ is $\Sigma_1^0$ because:
$$\sup_t \Phi(u(t)) < M \Leftrightarrow \forall t.\ \Phi(u(t)) < M$$

The negation requires finding a time $t^*$ with $\Phi(u(t^*)) \geq M$, which is a search over infinite time.

**Verifier Strategy:**
```
EnergyVerifier(u, M, timeout):
    for t in 0, dt, 2dt, ..., timeout:
        if Phi(u(t)) >= M:
            return (NO, K^- := (t, Phi(u(t))))
    return (INC, K^inc := timeout)
```

**Soundness:** If the verifier returns NO with witness $(t^*, \Phi(u(t^*)))$, then $\Phi(u(t^*)) \geq M$, so the predicate fails.

**Incompleteness:** The verifier may return INC even when the predicate is false (the violation occurs after the timeout).

**Gate 3 (CompactCheck):** The predicate "exists profile $V$ with positive mass" is $\Sigma_1^0$:
$$\exists V \in \mathcal{L}.\ \mu(B_\varepsilon(V)) > 0$$

**Verifier Strategy:**
```
CompactVerifier(u, library L, epsilon):
    for each profile V in L:
        mass := compute mu(B_epsilon(V))
        if mass > 0:
            return (YES, K^+ := (V, epsilon, mass))
    return (INC, K^inc := "library exhausted")
```

**Soundness:** YES is returned only when a positive-mass profile is found.

### Step 3: Co-semi-decidable Gates (Dual Synthesis)

**Claim.** For $\Pi_1^0$ predicates, the factory produces verifiers that may return INCONCLUSIVE on positive instances.

**Proof.** For Gate 6 (GeomCheck):

The predicate $\text{Cap}(\Sigma) = 0$ is $\Pi_1^0$ because capacity is defined as:
$$\text{Cap}(\Sigma) = 0 \Leftrightarrow \forall \epsilon > 0.\ \text{Cap}_\epsilon(\Sigma) < \epsilon$$

where $\text{Cap}_\epsilon$ is the $\epsilon$-approximate capacity.

**Verifier Strategy:**
```
GeomVerifier(Sigma, tolerance):
    cap := approximate_capacity(Sigma, tolerance)
    if cap > epsilon_regularity:
        return (NO, K^- := (cap, "above threshold"))
    if cap < tolerance:
        return (YES, K^+ := (tolerance, "capacity bounded"))
    return (INC, K^inc := "inconclusive at tolerance")
```

**Soundness:** NO is returned only when capacity exceeds the regularity threshold.

### Step 4: Undecidable Gates (Tactic-based Synthesis)

**Claim.** For predicates in $\Pi_2^0$ or higher, the factory produces tactic-based verifiers that always terminate with one of {YES, NO, INC}.

**Proof.** For Gate 7 (StiffnessCheck):

The Lojasiewicz-Simon inequality $\|\nabla\Phi(x)\| \geq C|\Phi(x) - \Phi_{\min}|^\theta$ is $\Pi_2^0$:
$$\exists C > 0, \theta \in (0,1).\ \forall x \in U.\ \|\nabla\Phi(x)\| \geq C|\Phi(x) - \Phi_{\min}|^\theta$$

This is undecidable in general (requires finding uniform constants over infinite domains).

**Tactic Library:**
```
StiffnessVerifier(Phi, x):
    // Tactic 1: Algebraic check (polynomial Phi)
    if is_polynomial(Phi):
        (C, theta) := compute_lojasiewicz_exponent(Phi)
        return (YES, K^+ := (C, theta, "algebraic"))

    // Tactic 2: Spectral gap check
    if has_spectral_gap(Hessian(Phi)):
        return (YES, K^+ := (lambda_1, 1/2, "spectral"))

    // Tactic 3: Known function class
    if Phi in KNOWN_LS_CLASS:
        return (YES, K^+ := lookup_constants(Phi))

    // Tactic exhaustion
    return (INC, K^inc := "tactics exhausted")
```

**For Gate 17 (Lock):**

The predicate $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathcal{H}) = \emptyset$ is undecidable by Rice's Theorem:
- Checking whether a particular morphism class is empty is equivalent to checking a semantic property of the target structure
- Rice's Theorem: all non-trivial semantic properties of programs are undecidable

**Tactic Library E1-E12:**
1. **E1 (Geometric):** Check Hausdorff dimension bounds
2. **E2 (Topological):** Check homotopy obstructions
3. **E3 (Variational):** Mountain pass / critical point arguments
4. **E4 (Cohomological):** Cech cohomology obstructions
5. **E5 (Representation):** Schur orthogonality / invariant theory
6. **E6-E12:** Additional specialized tactics

```
LockVerifier(H_bad, H, tactics):
    for tactic in E1, E2, ..., E12:
        result := apply_tactic(tactic, H_bad, H)
        if result == BLOCKED:
            return (YES, K^+ := (tactic, obstruction_class))
        if result == MORPHISM_FOUND:
            return (NO, K^- := (morphism_witness))
    return (INC, K^inc := "tactics exhausted, horizon")
```

### Step 5: Soundness Composition

**Claim.** The factory produces sound verifiers by construction.

**Proof.** By the Curry-Howard correspondence, each certificate $K_i^+$ is a proof term:
- $K_i^+$ encodes a witness for predicate $P_i$
- The witness structure is defined by the type $W_i^T$
- Verification of $K_i^+$ is decidable (certificate checking)

**Certificate Validity Invariant:**
$$\text{Valid}(K_i^+) \Leftrightarrow \exists w \in W_i^T.\ \text{Verify}(w) = \text{true} \wedge \text{Extract}(K_i^+) = w$$

**Soundness Proof:** If $V_i^T(x, \Gamma) = (\text{YES}, K_i^+)$, then:
1. $K_i^+$ was produced by the verifier
2. By construction, $K_i^+$ contains a valid witness $w \in W_i^T$
3. The witness $w$ certifies $P_i^T(x)$ by the witness semantics
4. Therefore $P_i^T(x)$ holds

This is constructive: the certificate is the proof.

---

## Connections to Rice's Theorem and the Halting Problem

### Rice's Theorem

**Statement (Rice, 1953).** Let $\mathcal{P}$ be a non-trivial semantic property of partial computable functions. Then the decision problem "Does function $f$ have property $\mathcal{P}$?" is undecidable.

**Implications for FACT-Gate:**

1. **Gate 17 (Lock):** The predicate $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathcal{H}) = \emptyset$ is a semantic property (it depends on the behavior of the hypostructure, not its syntactic representation). By Rice's Theorem, this is undecidable in general.

2. **Gate 11 (ComplexCheck):** Kolmogorov complexity $K(x)$ is uncomputable. The predicate $K(x) < B$ is undecidable (otherwise we could compute $K$).

3. **Gate 7 (StiffnessCheck):** The Lojasiewicz exponent involves quantification over all points in a neighborhood, which cannot be finitely verified for arbitrary smooth functions.

**How the Factory Respects Rice's Theorem:**

The factory does **not** claim to solve undecidable problems. Instead, it:
1. Identifies the decidability class of each predicate
2. Produces **sound but incomplete** verifiers for undecidable predicates
3. Uses the INCONCLUSIVE outcome to honestly report when verification fails
4. Provides tactic libraries that succeed on **practically occurring** instances

### The Halting Problem

**Statement.** There is no algorithm that correctly determines, for all programs $P$ and inputs $I$, whether $P$ halts on $I$.

**Implications for FACT-Gate:**

1. **Gate 1 (EnergyCheck):** Checking whether energy remains bounded is equivalent to checking whether the dynamical system "halts" at a bounded value. For Turing-complete dynamics, this reduces to the halting problem.

2. **Gate 2 (ZenoCheck):** Counting events to check for Zeno behavior is equivalent to checking termination of an event-driven process.

3. **Timeout Mechanism:** The factory's timeout with $K^{\text{inc}}$ fallback is the standard solution:
   - Run the verifier for bounded time/resources
   - If no answer, return INCONCLUSIVE
   - This is sound (never produces false positives)
   - This is incomplete (may miss true positives)

**Relativization:** With oracle access to prior certificates $\Gamma$, some predicates become easier:
- If $\Gamma$ contains energy bounds from earlier gates, Gate 1 becomes trivial
- This is why the Sieve passes context between gates

### Post's Theorem and the Arithmetical Hierarchy

**Statement (Post).** The arithmetical hierarchy classifies predicates by quantifier complexity:
- $\Sigma_1^0$: Existentially quantified ($\exists n.\ R(n)$ where $R$ is decidable)
- $\Pi_1^0$: Universally quantified ($\forall n.\ R(n)$)
- $\Sigma_2^0$: $\exists n.\ \forall m.\ R(n, m)$
- $\Pi_2^0$: $\forall n.\ \exists m.\ R(n, m)$

**Implications for FACT-Gate:**

| Hierarchy Level | Decidability | Factory Strategy |
|-----------------|--------------|------------------|
| $\Delta_1^0$ | Decidable | Complete verifier |
| $\Sigma_1^0$ | Semi-decidable | Search-based verifier |
| $\Pi_1^0$ | Co-semi-decidable | Refutation-based verifier |
| $\Sigma_2^0$, $\Pi_2^0$ | Neither RE nor co-RE | Tactic library + INC |

The factory's tactic library is designed to handle the common cases within undecidable classes:
- **Algebraic structures:** Decidable by Tarski's theorem (real closed fields)
- **Finite approximations:** Replace $\forall x \in X$ with $\forall x \in X_N$
- **Known function classes:** Lookup tables for well-studied cases

---

## Program Synthesis Perspective

### The Verifier Synthesis Pipeline

```
                    +------------------+
                    |   Specification  |
                    |   (Type T, P_i)  |
                    +--------+---------+
                             |
                             v
                    +--------+---------+
                    | Decidability     |
                    | Classification   |
                    +--------+---------+
                             |
              +--------------+---------------+
              |              |               |
              v              v               v
        +-----+----+   +-----+----+    +-----+-----+
        | Decidable|   | Semi-    |    | Undecid.  |
        | Synthesis|   | Decidable|    | Tactics   |
        +-----+----+   +-----+----+    +-----+-----+
              |              |               |
              v              v               v
        +-----+----+   +-----+----+    +-----+-----+
        | Complete |   | One-sided|    | Tactic    |
        | Verifier |   | Verifier |    | Verifier  |
        +-----+----+   +-----+----+    +-----+-----+
              |              |               |
              +--------------+---------------+
                             |
                             v
                    +--------+---------+
                    | Unified Verifier |
                    | V_i^T: X -> {Y,N,I}|
                    +------------------+
```

### Synthesis from Specifications

**Input:** Logical specification $\phi_i$ for gate $i$

**Output:** Verifier $V_i$ satisfying:
$$\forall x.\ V_i(x) = \text{YES} \Rightarrow \phi_i(x)$$

**Synthesis Algorithm:**
1. **Parse specification:** Extract quantifier structure of $\phi_i$
2. **Classify decidability:** Determine arithmetical hierarchy level
3. **Select synthesis strategy:**
   - $\Delta_1^0$: Direct decision procedure
   - $\Sigma_1^0$: Bounded search with witness extraction
   - $\Pi_1^0$: Bounded refutation with counterexample
   - Higher: Tactic composition with INC fallback
4. **Generate verifier code:** Type-directed synthesis
5. **Attach certificate schema:** Define witness type $W_i^T$

### Correctness by Construction

The factory achieves **Correct-by-Construction** verifiers through:

1. **Type-directed synthesis:** Verifier structure follows specification structure
2. **Witness types:** Certificates are typed proof objects
3. **Compositionality:** Complex verifiers compose from simpler ones
4. **Soundness preservation:** Each transformation preserves soundness

**Formal Guarantee:**
$$\mathcal{F}(\phi_i) = V_i \quad \text{where} \quad \forall x.\ V_i(x) = \text{YES} \Rightarrow \phi_i(x)$$

---

## Certificate Structure

### Witness Types for Each Gate

```
-- Gate 1: EnergyCheck
Witness[EnergyCheck] := {
    state: X,
    time: R+,
    energy_value: R+,
    bound: R+,
    proof: energy_value < bound
}

-- Gate 3: CompactCheck
Witness[CompactCheck] := {
    profile: Profile,
    scale: R+,
    mass: R+,
    proof: mass > 0
}

-- Gate 4: ScaleCheck
Witness[ScaleCheck] := {
    alpha: R,
    beta: R,
    lambda_c: R,
    proof: alpha < beta + lambda_c
}

-- Gate 7: StiffnessCheck
Witness[StiffnessCheck] := {
    equilibrium: X,
    constant_C: R+,
    exponent_theta: (0,1),
    gradient_bound: R+,
    proof: gradient_bound >= C * |Phi(x) - Phi_min|^theta
}

-- Gate 17: Lock
Witness[LockCheck] := {
    obstruction_class: Cohomology,
    tactic_trace: List[TacticResult],
    hom_emptiness: Proof(Hom = empty),
    proof: tactic_trace |- hom_emptiness
}
```

### Certificate Verification Algorithm

```
function VerifyCertificate(gate_id, certificate, input):
    witness_type := lookup_witness_type(gate_id)

    // Check certificate has correct type
    if not type_check(certificate, witness_type):
        return INVALID("Type mismatch")

    // Extract and verify witness
    witness := extract_witness(certificate)

    // Check witness validity
    if not verify_witness(witness, input):
        return INVALID("Witness invalid")

    // Check proof object
    if not check_proof(certificate.proof):
        return INVALID("Proof invalid")

    return VALID
```

---

## Complexity Bounds

### Per-Gate Verification Complexity

| Gate | Decidability | Time Complexity | Space Complexity |
|------|--------------|-----------------|------------------|
| 1 (EnergyCheck) | $\Sigma_1^0$ | $O(T \cdot C_\Phi)$ | $O(1)$ |
| 2 (ZenoCheck) | $\Sigma_1^0$ | $O(N \cdot C_{\text{event}})$ | $O(N)$ |
| 3 (CompactCheck) | $\Sigma_1^0$ | $O(|L| \cdot C_\mu)$ | $O(|L|)$ |
| 4 (ScaleCheck) | $\Delta_1^0$ | $O(\log(1/\epsilon))$ | $O(1)$ |
| 5 (ParamCheck) | $\Sigma_1^0$ | $O(|C| \cdot C_\mathcal{I})$ | $O(|C|)$ |
| 6 (GeomCheck) | $\Pi_1^0$ | $O(C_{\text{cap}} \cdot \log(1/\epsilon))$ | $O(1)$ |
| 7 (StiffnessCheck) | $\Pi_2^0$ | $O(|\text{tactics}| \cdot C_{\text{tactic}})$ | $O(1)$ |
| 8 (TopoCheck) | $\Delta_1^0$ | $O(|\mathcal{T}|)$ | $O(|\mathcal{T}|)$ |
| 9 (TameCheck) | $\Delta_1^0$ | $O(C_{\text{definability}})$ | $O(1)$ |
| 10 (ErgoCheck) | $\Sigma_1^0$ | $O(C_{\text{spectral}})$ | $O(n^2)$ |
| 11 (ComplexCheck) | $\Pi_2^0$ | $O(|D| \cdot C_K)$ | $O(|D|)$ |
| 12 (OscillateCheck) | $\Delta_1^0$ | $O(T \cdot C_\nabla)$ | $O(1)$ |
| 13 (BoundaryCheck) | $\Delta_1^0$ | $O(1)$ | $O(1)$ |
| 14 (OverloadCheck) | $\Sigma_1^0$ | $O(T \cdot C_B)$ | $O(1)$ |
| 15 (StarveCheck) | $\Sigma_1^0$ | $O(T \cdot C_\Sigma)$ | $O(1)$ |
| 16 (AlignCheck) | $\Sigma_1^0$ | $O(C_{\text{align}})$ | $O(d)$ |
| 17 (Lock) | $\Pi_2^0+$ | $O(|\text{tactics}| \cdot C_{\text{tactic}})$ | $O(|\mathbb{H}|)$ |

Where:
- $T$ = time horizon / timeout
- $C_\Phi$ = cost to evaluate energy functional
- $|L|$ = size of profile library
- $|\mathcal{T}|$ = number of topological sectors
- $|D|$ = size of dictionary
- $|\text{tactics}|$ = number of tactics in library

### Total Factory Complexity

$$C_{\text{factory}} = O\left(\sum_{i=1}^{17} C_{\text{gate}_i}\right) = O(T \cdot C_{\max} + |\text{tactics}| \cdot C_{\text{tactic}})$$

---

## Algorithmic Implications

### Verifier Implementation Pattern

```
function GateVerifier(gate_id, input, context, timeout):
    spec := lookup_specification(gate_id)

    // Dispatch based on decidability class
    match decidability_class(spec):
        case DECIDABLE:
            return decide(spec, input)

        case SEMI_DECIDABLE:
            return bounded_search(spec, input, timeout)

        case CO_SEMI_DECIDABLE:
            return bounded_refutation(spec, input, timeout)

        case UNDECIDABLE:
            return tactic_based_verify(spec, input, context)
```

### Tactic-Based Verification

```
function TacticVerify(spec, input, context):
    for tactic in TACTIC_LIBRARY:
        // Check tactic applicability
        if not applicable(tactic, spec, input):
            continue

        // Apply tactic
        result := apply_tactic(tactic, spec, input, context)

        match result:
            case SUCCESS(witness):
                return (YES, K^+ := (tactic, witness))
            case FAILURE(counterexample):
                return (NO, K^- := (tactic, counterexample))
            case INAPPLICABLE:
                continue

    // All tactics exhausted
    return (INC, K^inc := "tactics exhausted")
```

---

## Summary

The FACT-Gate theorem, translated to complexity theory, establishes:

1. **Verifier Synthesis is Achievable:** For each predicate specification, the factory produces a sound verifier that correctly identifies positive instances.

2. **Decidability Classification is Essential:** The 17 gates partition into decidable (6), semi-decidable (11), co-semi-decidable (1), and undecidable (3) classes.

3. **Rice's Theorem Sets the Boundary:** Undecidable gates (7, 11, 17) cannot have complete verifiers. The factory produces tactic-based verifiers with honest INCONCLUSIVE outcomes.

4. **Correctness-by-Construction:** The factory achieves soundness through type-directed synthesis, ensuring certificates are valid proof terms.

5. **Practical Completeness via Tactics:** For practically occurring instances, the tactic library provides effective verification even for theoretically undecidable predicates.

| Classical Result | FACT-Gate Analog |
|------------------|-----------------|
| Rice's Theorem | Gates 7, 11, 17 are undecidable |
| Halting Problem | Gates 1, 2 reduce to termination checking |
| Post's Theorem | Arithmetical hierarchy classification |
| Curry-Howard | Certificates are proof terms |
| Program Synthesis | Factory generates verifiers from specs |

The key insight is that **undecidability does not prevent useful verification**. By accepting INCONCLUSIVE outcomes and providing rich tactic libraries, the factory produces verifiers that succeed on the cases that matter in practice while honestly reporting when verification is beyond current theory.

---

## Literature

**Decidability and Computability:**
1. **Rice, H.G. (1953).** "Classes of Recursively Enumerable Sets and Their Decision Problems." Transactions of the AMS. *The undecidability of semantic properties.*

2. **Rogers, H. (1967).** *Theory of Recursive Functions and Effective Computability.* MIT Press. *Comprehensive treatment of computability theory.*

3. **Soare, R.I. (1987).** *Recursively Enumerable Sets and Degrees.* Springer. *Modern treatment of the arithmetical hierarchy.*

**Program Synthesis:**
4. **Manna, Z. & Waldinger, R. (1980).** "A Deductive Approach to Program Synthesis." TOPLAS. *Synthesis from specifications.*

5. **Alur, R. et al. (2013).** "Syntax-Guided Synthesis." FMCAD. *Modern program synthesis techniques.*

**Type Theory and Verification:**
6. **The Univalent Foundations Program (2013).** *Homotopy Type Theory.* Institute for Advanced Study. *Certificates as proof terms.*

7. **Leroy, X. (2009).** "Formal Verification of a Realistic Compiler." CACM. *Certified compilation.*

**O-Minimality and Decidability:**
8. **Wilkie, A.J. (1996).** "Model Completeness Results for Expansions of the Ordered Field of Real Numbers." JAMS. *Decidability of real exponential field.*

9. **van den Dries, L. (1998).** *Tame Topology and O-minimal Structures.* Cambridge University Press. *O-minimal decidability.*

**Kolmogorov Complexity:**
10. **Li, M. & Vitanyi, P. (2008).** *An Introduction to Kolmogorov Complexity and Its Applications.* Springer. *Uncomputability of K(x).*
