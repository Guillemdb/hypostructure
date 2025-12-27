---
title: "RESOLVE-WeakestPre - Complexity Theory Translation"
---

# RESOLVE-WeakestPre: Backward Computation of Minimal Requirements

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-WeakestPre theorem (Weakest Precondition Principle) from the hypostructure framework. The translation establishes a formal correspondence between the Sieve's computation of barrier outcomes and **Dijkstra's weakest precondition calculus**, where backward analysis determines the minimal requirements for achieving a specified postcondition.

**Original Theorem Reference:** {prf:ref}`mt-resolve-weakest-pre`

**Core Insight:** The Structural Sieve computes regularity as an output, not an input. Users provide interface implementations (preconditions), and the Sieve automatically derives the regularity verdict (postcondition). This mirrors weakest precondition semantics: given a desired postcondition $Q$, compute the weakest precondition $wp(S, Q)$ such that executing statement $S$ from any state satisfying $wp(S, Q)$ guarantees termination in a state satisfying $Q$.

---

## Hypostructure Context

The RESOLVE-WeakestPre theorem states that to instantiate the Structural Sieve for a dynamical system, users need only:

1. **Map Types**: Define state space $X$, height functional $\Phi$, dissipation $\mathfrak{D}$, symmetry group $G$
2. **Implement Interfaces**: Provide computable formulas for each interface predicate $\mathcal{P}_n$:
   - Scaling exponents $\alpha, \beta$ (for $\mathrm{SC}_\lambda$)
   - Dimension estimates $\dim(\Sigma)$ (for $\mathrm{Cap}_H$)
   - Lojasiewicz exponent $\theta$ (for $\mathrm{LS}_\sigma$)
   - Topological invariant $\tau$ (for $\mathrm{TB}_\pi$)
3. **Run the Sieve**: Execute the algorithm to obtain verdict $\mathcal{V} \in \{\text{YES}, \text{NO}, \text{Blocked}\}$

**The Sieve automatically determines regularity.** Users do not need to:
- Prove global existence a priori
- Assume solutions are smooth
- Know where singularities occur
- Classify all possible blow-up profiles in advance

---

## Complexity Theory Statement

**Theorem (Weakest Precondition Computation).**
Let $\mathcal{P} = (S, \text{Pre}, \text{Post}, \text{Inv})$ be a program verification problem where:
- $S$ is a program (statement sequence)
- $\text{Pre}$ is the user-provided precondition
- $\text{Post}$ is the desired postcondition
- $\text{Inv}$ are loop invariants (if any)

The **weakest precondition transformer** $wp: \text{Stmt} \times \text{Pred} \to \text{Pred}$ computes:

$$wp(S, Q) = \text{weakest predicate } P \text{ such that } \{P\} S \{Q\}$$

Given user-provided interface implementations (partial preconditions), the WP calculus:

1. **Backward propagates** postcondition requirements through the program
2. **Computes minimal** sufficient conditions for correctness
3. **Produces certificates** (verification conditions) checkable in polynomial time

**Guarantee:** If the user-provided precondition $\text{Pre}$ implies the computed weakest precondition $wp(S, \text{Post})$, then the Hoare triple $\{\text{Pre}\} S \{\text{Post}\}$ is valid.

**Complexity:**
- **Straight-line code:** $O(n)$ where $n$ is program length
- **With loops (invariant given):** $O(n \cdot |Q|)$ where $|Q|$ is predicate size
- **Invariant inference:** Undecidable in general; decidable for restricted domains

---

## Terminology Translation Table

| Hypostructure Concept | WP Calculus Equivalent | Formal Correspondence |
|-----------------------|------------------------|----------------------|
| Interface implementation $\mathcal{P}_n$ | Precondition annotation | User-provided specification |
| Sieve verdict $\mathcal{V}$ | Verification outcome | Valid / Invalid / Unknown |
| Regularity | Postcondition satisfaction | Program terminates correctly |
| Singularity | Postcondition violation | Runtime error or non-termination |
| Type mapping $(X, \Phi, \mathfrak{D}, G)$ | Program state space | Variables, heap, control flow |
| Barrier outcome | Loop termination condition | Variant/invariant pair |
| Certificate $K^+$ | Verification condition | Proof obligation |
| Certificate $K^{\mathrm{wit}}$ | Counterexample trace | Execution violating spec |
| Certificate $K^{\mathrm{inc}}$ | Inconclusive result | Timeout or undecidable case |
| Scaling exponent $\alpha$ | Loop bound exponent | Complexity of termination proof |
| Lojasiewicz exponent $\theta$ | Convergence rate | Speed of approach to fixed point |
| Profile library | Proof templates | Pre-verified proof patterns |
| SectorMap | Control flow graph | Program structure |
| Dictionary | Variable environment | $\Gamma: \text{Var} \to \text{Type}$ |
| Energy dissipation | Variant decrease | $wp$ of loop body decreases measure |
| Concentration-compactness | Case splitting | Disjunctive weakest precondition |
| Automatic regularity | Automated verification | User provides hints, system proves |

---

## Predicate Transformer Semantics

### Dijkstra's Weakest Precondition

**Definition (Weakest Precondition).** For statement $S$ and postcondition $Q$, the weakest precondition $wp(S, Q)$ is the weakest predicate such that:

$$\forall \sigma.\ \sigma \models wp(S, Q) \Rightarrow \llbracket S \rrbracket(\sigma) \models Q$$

where $\llbracket S \rrbracket$ denotes the denotational semantics of $S$.

**Fundamental Rules:**

| Statement | Weakest Precondition |
|-----------|---------------------|
| $\texttt{skip}$ | $wp(\texttt{skip}, Q) = Q$ |
| $x := e$ | $wp(x := e, Q) = Q[e/x]$ |
| $S_1; S_2$ | $wp(S_1; S_2, Q) = wp(S_1, wp(S_2, Q))$ |
| $\texttt{if } b \texttt{ then } S_1 \texttt{ else } S_2$ | $wp(\cdot, Q) = (b \Rightarrow wp(S_1, Q)) \land (\neg b \Rightarrow wp(S_2, Q))$ |
| $\texttt{while } b \texttt{ do } S$ | Requires invariant $I$: $I \land \neg b \Rightarrow Q$ and $\{I \land b\} S \{I\}$ |

### Strongest Postcondition (Dual)

**Definition (Strongest Postcondition).** For statement $S$ and precondition $P$:

$$sp(S, P) = \text{strongest predicate } Q \text{ such that } \{P\} S \{Q\}$$

**Duality:** $P \Rightarrow wp(S, Q) \iff sp(S, P) \Rightarrow Q$

**Forward vs. Backward Analysis:**

| Direction | Transformer | Hypostructure Analog |
|-----------|-------------|---------------------|
| Backward | $wp(S, Q)$ | Sieve computing barrier preconditions |
| Forward | $sp(S, P)$ | Flow evolution $S_t$ propagating state |

---

## Proof Sketch

### Setup: The Verification Correspondence

We establish the correspondence between hypostructure components and program verification:

| Hypostructure | Program Verification |
|---------------|---------------------|
| State space $\mathcal{X}$ | Program state $\Sigma = \text{Var} \to \text{Val}$ |
| Flow $S_t: \mathcal{X} \to \mathcal{X}$ | Program semantics $\llbracket S \rrbracket: \Sigma \to \Sigma$ |
| Energy $\Phi: \mathcal{X} \to \mathbb{R}$ | Ranking function $r: \Sigma \to \mathbb{N}$ |
| Dissipation $\mathfrak{D}$ | Variant decrease condition |
| Interface $\mathcal{P}_n$ | Precondition/invariant annotations |
| Verdict $\mathcal{V}$ | Verification result |

### Step 1: Interface to Precondition (Annotation Translation)

**Claim.** User-provided interface implementations translate to precondition annotations in Hoare logic.

**Construction.** For each hypostructure interface:

| Interface | Hoare Annotation | Role |
|-----------|------------------|------|
| $D_E$ (Energy bound) | $\{E \leq B\}$ | Resource bound precondition |
| $\mathrm{SC}_\lambda$ (Scaling) | $\{\|x\| \leq \lambda^\alpha\}$ | Size bound |
| $\mathrm{LS}_\sigma$ (Lojasiewicz) | $\{\|\nabla f\| \geq c \|f - f^*\|^\theta\}$ | Convergence rate |
| $\mathrm{Cap}_H$ (Capacity) | $\{\dim(\Sigma) \leq d\}$ | Complexity bound |
| $\mathrm{TB}_\pi$ (Topological) | Loop invariant $I$ | Structural property |

**User Burden Reduction:**

The hypostructure principle states users provide only:
- 10 primitive components (thin specification)

The Sieve derives:
- Full ~30-component kernel object specification

**WP Analog:** Users provide annotations at key points (function boundaries, loop headers). The WP calculus derives verification conditions for all intermediate points.

### Step 2: Backward Propagation (WP Computation)

**Claim.** The Sieve's barrier analysis corresponds to backward WP propagation.

**Algorithm (WP Backward Pass):**

```
function WP_BACKWARD(S, postcondition Q):
    match S with
    | skip         -> return Q
    | x := e       -> return Q[e/x]
    | S1; S2       -> return WP_BACKWARD(S1, WP_BACKWARD(S2, Q))
    | if b then S1 else S2 ->
        Q1 <- WP_BACKWARD(S1, Q)
        Q2 <- WP_BACKWARD(S2, Q)
        return (b => Q1) /\ (~b => Q2)
    | while b do S with invariant I ->
        # Invariant provided by user (interface implementation)
        check: I /\ b => WP_BACKWARD(S, I)   # Preservation
        check: I /\ ~b => Q                   # Establishment
        return I
```

**Correspondence to Sieve:**

| Sieve Operation | WP Operation |
|-----------------|--------------|
| Evaluate barrier $\mathcal{B}_n$ | Compute $wp(S_n, Q_n)$ |
| Chain barriers | Compose $wp$ transformers |
| Check interface $\mathcal{P}_n$ | Verify $\text{Pre} \Rightarrow wp$ |
| Emit certificate | Generate verification condition |

**Key Insight:** The Sieve computes backward from desired regularity (postcondition) to required interface properties (precondition). This is exactly WP semantics.

### Step 3: Loop Analysis (Barrier Iteration)

**Claim.** Barrier-based singularity analysis corresponds to loop invariant verification with ranking functions.

**Loop Verification Structure:**

For loop $\texttt{while } b \texttt{ do } S$:

1. **Invariant $I$:** Property preserved by each iteration
2. **Variant $v$:** Well-founded measure strictly decreasing each iteration
3. **Termination:** $I \land b \Rightarrow wp(S, I \land v' < v)$

**Barrier Correspondence:**

| Barrier Concept | Loop Verification | Role |
|-----------------|-------------------|------|
| Height functional $\Phi$ | Variant function $v$ | Termination measure |
| Barrier predicate | Invariant $I$ | Preserved property |
| Barrier crossing | Loop iteration | One step of computation |
| Barrier exhaustion | Variant reaches 0 | Loop terminates |
| Profile concentration | Invariant strengthening | Case analysis |

**Ranking Function Construction:**

The energy functional $\Phi$ becomes a ranking function:

$$v(\sigma) = \lceil \Phi(\sigma) / \epsilon \rceil$$

for discretization parameter $\epsilon > 0$. The energy-dissipation inequality:

$$\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x)\, ds \leq \Phi(x)$$

translates to the variant decrease condition:

$$\{I \land b \land v = v_0\}\ S\ \{v < v_0\}$$

### Step 4: Certificate Generation (Verification Conditions)

**Claim.** Sieve certificates correspond to verification conditions (VCs) in program verification.

**Verification Condition Generation:**

For program $S$ with precondition $P$ and postcondition $Q$, the VCs are:

$$\text{VC}(S, P, Q) = \{P \Rightarrow wp(S, Q)\}$$

For programs with loops annotated with invariant $I$:

$$\text{VC} = \{P \Rightarrow I\} \cup \{I \land b \Rightarrow wp(S_{\text{body}}, I)\} \cup \{I \land \neg b \Rightarrow Q\}$$

**Certificate Correspondence:**

| Sieve Certificate | Verification Condition |
|-------------------|----------------------|
| $K^+$ (positive) | VC holds (proved valid) |
| $K^{\mathrm{wit}}$ (witness) | VC fails with counterexample |
| $K^{\mathrm{inc}}$ (inconclusive) | VC undecidable/timeout |

**Explicit VC Certificate:**

```
VC_Certificate := (
    program          : S,
    precondition     : P,
    postcondition    : Q,
    invariants       : [I_1, ..., I_k],
    variants         : [v_1, ..., v_k],
    vc_obligations   : [(P => wp(S, Q)), ...],
    proof_status     : Valid | Invalid(cex) | Unknown
)
```

### Step 5: Completeness (Automation Guarantee)

**Claim.** For "good" program classes, WP computation is decidable and complete.

**Decidable Fragments:**

| Program Class | WP Decidability | Complexity |
|---------------|-----------------|------------|
| Loop-free | Decidable | $O(n \cdot |Q|)$ |
| Bounded loops | Decidable | $O(k \cdot n \cdot |Q|)$ for bound $k$ |
| Affine loops | Decidable | $O(n^3)$ via linear algebra |
| Polynomial invariants | Decidable | Doubly exponential (Tarski) |
| General loops | Undecidable | Halting problem |

**Good Types Correspondence:**

| Hypostructure Good Type | Program Class | Verification Approach |
|-------------------------|---------------|----------------------|
| Parabolic | Monotone programs | Energy-based invariants |
| Dispersive | Distributed algorithms | Bounded communication |
| Hyperbolic | Finite-state protocols | Model checking |

**Automation Guarantee Translation:**

The hypostructure Automation Guarantee states that for good types, the Sieve automatically derives regularity. The WP analog:

**Theorem (WP Automation).** For programs in decidable fragments:
1. $wp(S, Q)$ is computable
2. Verification is decidable: $P \Rightarrow wp(S, Q)$ is checkable
3. Counterexamples are constructive: if $\neg(P \Rightarrow wp(S, Q))$, a witness exists

---

## Connections to Hoare Logic

### Hoare Triples and Partial Correctness

**Definition (Hoare Triple).** The triple $\{P\} S \{Q\}$ is valid if:

$$\forall \sigma.\ \sigma \models P \land \llbracket S \rrbracket(\sigma) \text{ terminates} \Rightarrow \llbracket S \rrbracket(\sigma) \models Q$$

**Partial vs. Total Correctness:**

| Correctness | Notation | Requirement |
|-------------|----------|-------------|
| Partial | $\{P\} S \{Q\}$ | If $S$ terminates, $Q$ holds |
| Total | $[P]\ S\ [Q]$ | $S$ terminates and $Q$ holds |

**WP Variants:**

| Transformer | Correctness | Definition |
|-------------|-------------|------------|
| $wp(S, Q)$ | Total | Terminates in state satisfying $Q$ |
| $wlp(S, Q)$ | Partial | If terminates, satisfies $Q$ |

**Hypostructure Correspondence:**

- Partial correctness: Conditional regularity (if well-posed, then smooth)
- Total correctness: Unconditional regularity (well-posed and smooth)

The Sieve aims for **total correctness**: proving both well-posedness (termination) and regularity (postcondition).

### Proof Rules as Predicate Transformers

**Rule (Consequence):**
$$\frac{P' \Rightarrow P \quad \{P\} S \{Q\} \quad Q \Rightarrow Q'}{\{P'\} S \{Q'\}}$$

**WP Formulation:** $P' \Rightarrow P \Rightarrow wp(S, Q) \Rightarrow wp(S, Q')$

**Sieve Analog:** Interface strengthening. If $\mathcal{P}_n$ implies $\mathcal{P}_n'$, barrier analysis at $\mathcal{P}_n$ implies analysis at $\mathcal{P}_n'$.

**Rule (Composition):**
$$\frac{\{P\} S_1 \{R\} \quad \{R\} S_2 \{Q\}}{\{P\} S_1; S_2 \{Q\}}$$

**WP Formulation:** $wp(S_1; S_2, Q) = wp(S_1, wp(S_2, Q))$

**Sieve Analog:** Barrier chaining. Sequential barriers compose via intermediate certificates.

**Rule (Loop):**
$$\frac{\{I \land b\} S \{I\}}{\{I\} \texttt{while } b \texttt{ do } S \{I \land \neg b\}}$$

**WP Formulation:** $wp(\texttt{while}, Q)$ requires invariant $I$ with $I \land \neg b \Rightarrow Q$

**Sieve Analog:** Barrier iteration with invariant = interface predicate.

---

## Program Verification Framework

### The Verification Condition Generator

**Algorithm (VCGen):**

```
function VCGEN(P, S, Q):
    match S with
    | skip -> return {P => Q}
    | x := e -> return {P => Q[e/x]}
    | S1; S2 ->
        # Find intermediate assertion R (can be inferred or annotated)
        R <- INFER_MID(S1, S2, Q)
        return VCGEN(P, S1, R) + VCGEN(R, S2, Q)
    | if b then S1 else S2 ->
        return VCGEN(P /\ b, S1, Q) + VCGEN(P /\ ~b, S2, Q)
    | while b do S with invariant I, variant v ->
        return {
            P => I,                           # Initiation
            I /\ b => wp(S, I),               # Preservation
            I /\ b /\ v = v0 => wp(S, v < v0), # Termination (variant)
            I /\ ~b => Q                       # Establishment
        }
```

**Correspondence to Sieve:**

| VCGen Phase | Sieve Phase |
|-------------|-------------|
| Parse program | Build SectorMap |
| Extract annotations | Collect interface implementations |
| Compute VCs | Evaluate barrier predicates |
| Check VCs | Run SMT/theorem prover |
| Return verdict | Emit certificate |

### Symbolic Execution with WP

**Definition (Symbolic State).** A symbolic state $\sigma_s$ maps variables to symbolic expressions:

$$\sigma_s: \text{Var} \to \text{Expr}$$

**WP via Symbolic Execution:**

1. Start with postcondition $Q$
2. Execute program backward symbolically
3. Substitute symbolic expressions into $Q$
4. Simplify to obtain $wp(S, Q)$

**Example:**

```
Program: x := x + 1; y := x * 2
Postcondition: y = 10

Backward pass:
  After y := x * 2:  wp = (x * 2 = 10) = (x = 5)
  After x := x + 1:  wp = ((x + 1) = 5) = (x = 4)

Result: wp(S, y = 10) = (x = 4)
```

**Sieve Analog:** Barrier evaluation symbolically tracks interface predicates backward through the flow structure.

---

## Quantitative Bounds

### Verification Complexity

**Theorem (VCGen Complexity).** For program $S$ of size $n$ with predicates of size $|Q|$:

| Structure | VC Generation | VC Checking |
|-----------|---------------|-------------|
| Straight-line | $O(n)$ | $O(n \cdot |Q|)$ |
| Branching (depth $d$) | $O(n)$ | $O(2^d \cdot n \cdot |Q|)$ |
| Loops (invariant given) | $O(n)$ | Depends on invariant complexity |

### Invariant Inference Complexity

**Theorem (Invariant Inference Bounds).**

| Domain | Complexity | Technique |
|--------|------------|-----------|
| Interval (numerical) | $O(n^2)$ | Abstract interpretation |
| Octagon | $O(n^3)$ | Difference-bound matrices |
| Polyhedra | Exponential | Convex hull algorithms |
| Polynomial | Doubly exponential | Quantifier elimination |

**Connection to Hypostructure:**

The interface complexity bounds correspond:

| Hypostructure | Invariant Domain | Complexity |
|---------------|------------------|------------|
| $D_E$ (energy) | Interval domain | Linear |
| $\mathrm{SC}_\lambda$ (scaling) | Octagon domain | Polynomial |
| $\mathrm{LS}_\sigma$ (Lojasiewicz) | Polynomial domain | High |

### Certificate Size

**Lemma (VC Certificate Compactness).** For straight-line programs:

$$|\text{VC}| = O(n \cdot |P| \cdot |Q|)$$

For programs with $k$ loops:

$$|\text{VC}| = O(k \cdot |I| + n \cdot |Q|)$$

where $|I|$ is the total size of loop invariants.

---

## Worked Example: Energy Dissipation as Loop Verification

**Dynamical System (Hypostructure):**

```
State: x in R^n
Flow: dx/dt = -nabla E(x)    # Gradient descent
Energy: Phi(x) = E(x)
Dissipation: D(x) = |nabla E(x)|^2
Interface: |nabla E| >= c * |E - E*|^theta  (Lojasiewicz)
```

**Program (Verification):**

```
// Precondition: E(x) <= B
while |nabla E(x)| > epsilon do
    x := x - alpha * nabla E(x)
// Postcondition: E(x) <= E* + delta
```

**Verification Conditions:**

1. **Initiation:** $E(x) \leq B \Rightarrow I(x)$ where $I(x) \equiv E(x) \leq B$

2. **Preservation:** $I(x) \land |\nabla E| > \epsilon \Rightarrow I(x - \alpha \nabla E)$

   Proof: By energy dissipation:
   $$E(x - \alpha \nabla E) \approx E(x) - \alpha |\nabla E|^2 < E(x) \leq B$$

3. **Termination (Variant):** $v(x) = \lceil E(x) / \epsilon^2 \rceil$

   By Lojasiewicz: $|\nabla E| \geq c |E - E^*|^\theta$

   Each iteration decreases energy by $\geq \alpha c^2 |E - E^*|^{2\theta}$

   So $v$ decreases by at least 1 when $|E - E^*| > \epsilon/c$

4. **Establishment:** $I(x) \land |\nabla E| \leq \epsilon \Rightarrow E(x) \leq E^* + \delta$

   By Lojasiewicz: $|\nabla E| \leq \epsilon \Rightarrow |E - E^*| \leq (\epsilon/c)^{1/\theta}$

**WP Computation:**

$$wp(\texttt{gradient\_descent}, E \leq E^* + \delta) = E \leq B \land \text{Lojasiewicz}(\theta, c)$$

**Certificate:**

```
VC_Certificate := (
    program       : gradient_descent,
    precondition  : E(x) <= B /\ Lojasiewicz(theta, c),
    postcondition : E(x) <= E* + delta,
    invariant     : E(x) <= B,
    variant       : ceil(E(x) / eps^2),
    vc_status     : Valid
)
```

---

## Connections to Classical Results

### 1. Dijkstra's Guarded Command Language (1976)

**Reference:** {cite}`Dijkstra76`

**Theorem (WP Healthiness Conditions).** The predicate transformer $wp$ satisfies:

1. **Excluded Miracle:** $wp(S, \texttt{false}) = \texttt{false}$
2. **Monotonicity:** $Q \Rightarrow Q' \Rightarrow wp(S, Q) \Rightarrow wp(S, Q')$
3. **Conjunctivity:** $wp(S, Q \land Q') = wp(S, Q) \land wp(S, Q')$
4. **Disjunctivity:** $wp(S, Q \lor Q') \Leftarrow wp(S, Q) \lor wp(S, Q')$

**Connection to RESOLVE-WeakestPre:**

| WP Property | Sieve Property |
|-------------|----------------|
| Excluded miracle | No regularity from inconsistent input |
| Monotonicity | Stronger interfaces give stronger conclusions |
| Conjunctivity | Independent properties verify independently |

### 2. Back's Refinement Calculus (1980)

**Reference:** {cite}`Back80`

**Theorem (Refinement).** Program $S'$ refines $S$ (written $S \sqsubseteq S'$) if:

$$\forall Q.\ wp(S, Q) \Rightarrow wp(S', Q)$$

**Connection:** The Sieve's template matching corresponds to refinement: a template $T$ refines the abstract barrier specification.

### 3. Abstract Interpretation (Cousot & Cousot 1977)

**Theorem (Soundness).** Abstract interpretation computes sound over-approximations:

$$\gamma(\alpha(wp(S, Q))) \supseteq wp(S, \gamma(\alpha(Q)))$$

where $\alpha$ is abstraction and $\gamma$ is concretization.

**Connection to Sieve:** Interface predicates abstract the full state space. The Sieve works with these abstractions, computing sound (but possibly incomplete) verdicts.

### 4. Separation Logic (Reynolds 2002)

**Extension:** For heap-manipulating programs, separation logic extends Hoare logic with:

$$\{P * R\}\ S\ \{Q * R\}$$

The frame rule allows local reasoning about heap portions.

**Connection:** The Sieve's SectorMap partitions the state space similarly, allowing local barrier analysis that composes globally.

### 5. Dependent Types as Refinement Types

**Connection:** Refinement types $\{x : \tau \mid \phi(x)\}$ carry logical predicates. Type checking becomes verification:

$$\text{typecheck}(e : \{x : \tau \mid \phi\}) \iff wp(e, \phi)$$

The Sieve's interface predicates are analogous to refinement type annotations.

---

## Separation of Concerns

The RESOLVE-WeakestPre theorem embodies a clean separation:

| Layer | Responsibility | User/Framework |
|-------|----------------|----------------|
| Domain | Interface implementations ($\mathcal{P}_n$) | User (domain expert) |
| Framework | Sieve algorithm, barrier evaluation | Framework |
| Verification | Certificate checking | Framework (automated) |

**Verification Analog:**

| Layer | Responsibility | Role |
|-------|----------------|------|
| Annotation | Preconditions, invariants | User (developer) |
| VCGen | Generate verification conditions | Compiler/verifier |
| SMT | Prove VCs valid | Automated solver |

**Benefit:** A researcher can implement new interface predicates without understanding Sieve internals, just as a developer can annotate programs without understanding SMT solving.

---

## Summary

The RESOLVE-WeakestPre theorem, translated to complexity theory, states:

**The Structural Sieve computes regularity backward from desired outcomes, exactly as weakest precondition calculus computes verification conditions backward from postconditions.**

This principle:

1. **Separates concerns:** Users provide local specifications (interfaces/annotations)
2. **Automates derivation:** The system computes global properties (regularity/correctness)
3. **Produces certificates:** Verification conditions are checkable proof obligations
4. **Enables modularity:** Local properties compose to global guarantees

The translation illuminates deep connections:

| Hypostructure | Program Verification |
|---------------|---------------------|
| Interface predicates | Precondition annotations |
| Barrier analysis | WP computation |
| Energy dissipation | Ranking function decrease |
| Profile library | Proof templates |
| Sieve verdict | Verification outcome |
| Regularity as output | Correctness as derived property |

**Key Insight:** Just as Dijkstra's WP calculus allows programmers to specify what they want (postcondition) and compute what they need (precondition), the hypostructure framework allows analysts to specify interface properties and derive regularity automatically. Both achieve the same goal: **making the hard part (proof/regularity) an output rather than an input.**

---

## Literature

1. **Dijkstra, E. W. (1976).** *A Discipline of Programming.* Prentice-Hall. *Foundational WP calculus.*

2. **Back, R.-J. (1980).** "Correctness Preserving Program Refinements: Proof Theory and Applications." *Mathematical Centre Tracts.* *Refinement calculus.*

3. **Hoare, C. A. R. (1969).** "An Axiomatic Basis for Computer Programming." *CACM.* *Hoare logic foundations.*

4. **Cousot, P. & Cousot, R. (1977).** "Abstract Interpretation: A Unified Lattice Model for Static Analysis." *POPL.* *Abstract interpretation.*

5. **Floyd, R. W. (1967).** "Assigning Meanings to Programs." *Mathematical Aspects of Computer Science.* *Floyd-Hoare logic precursor.*

6. **Manna, Z. & Pnueli, A. (1974).** "Axiomatic Approach to Total Correctness of Programs." *Acta Informatica.* *Total correctness.*

7. **Nelson, G. (1989).** "A Generalization of Dijkstra's Calculus." *TOPLAS.* *Extended predicate transformers.*

8. **Leino, K. R. M. (2005).** "Efficient Weakest Preconditions." *Information Processing Letters.* *Efficient WP algorithms.*

9. **Barnett, M. et al. (2006).** "Boogie: A Modular Reusable Verifier for Object-Oriented Programs." *FMCO.* *Practical WP-based verification.*

10. **Reynolds, J. C. (2002).** "Separation Logic: A Logic for Shared Mutable Data Structures." *LICS.* *Separation logic.*

11. **Cousot, P. (2005).** "Proving Program Invariance and Termination by Parametric Abstraction, Lagrangian Relaxation and Semidefinite Programming." *VMCAI.* *Advanced invariant inference.*

12. **Podelski, A. & Rybalchenko, A. (2004).** "A Complete Method for the Synthesis of Linear Ranking Functions." *VMCAI.* *Ranking function synthesis.*
