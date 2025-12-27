---
title: "FACT-Instantiation - Complexity Theory Translation"
---

# FACT-Instantiation: Full Compiler Pipeline

## Overview

This document provides a complete complexity-theoretic translation of the FACT-Instantiation metatheorem from the hypostructure framework. The theorem establishes that factory-generated code synthesizes all required components from minimal thin kernel objects, producing a sound sieve implementation. In complexity theory terms, this corresponds to a **Full Compiler Pipeline**: complete compilation from high-level specification to verified implementation through a sequence of certified transformation stages.

**Original Theorem Reference:** {prf:ref}`mt-fact-instantiation`

---

## Complexity Theory Statement

**Theorem (Full Compiler Pipeline).**
Given a high-level specification $\mathcal{S}$ consisting of:
- Type selection $T$ (target architecture/semantics)
- User-supplied primitives $(\mathcal{X}, \Phi, \mathfrak{D}, G, \ldots)$ (source program + annotations)

There exists a certified compilation pipeline $\mathcal{P} = (F_1, F_2, \ldots, F_k)$ such that:

1. **Factory Composition:** Each factory $F_i$ transforms intermediate representation $\text{IR}_{i-1}$ to $\text{IR}_i$ with certificate $K_i$
2. **Non-Circularity:** The dependency graph of factories is acyclic; $F_i$ depends only on $(F_1, \ldots, F_{i-1})$
3. **Soundness Inheritance:** Each $K_i$ witnesses semantic preservation: $\llbracket \text{IR}_i \rrbracket \subseteq \llbracket \text{IR}_{i-1} \rrbracket$
4. **Contract Satisfaction:** Interface specifications are preserved across all transformations
5. **Termination:** The pipeline completes in bounded time with definite output

**Output Trichotomy.** The pipeline terminates with exactly one of:
- **Success:** Verified implementation with end-to-end correctness certificate
- **Iterative Refinement:** Intermediate repair (surgery) followed by re-compilation
- **Classified Failure:** Explicit obstruction certificate identifying the unsatisfiable requirement

**Formal Statement.** Let $\mathcal{C} = (\mathcal{T}, \mathcal{I}, \mathcal{F})$ be a certified compilation system where:
- $\mathcal{T}$ is the type/architecture specification
- $\mathcal{I}$ is the set of interface contracts
- $\mathcal{F} = \{F_1, \ldots, F_k\}$ is the factory sequence

The end-to-end compilation $\text{Compile}: \text{Source} \times \mathcal{T} \to \text{Target} \times \text{Certificate}$ satisfies:

$$\text{Compile}(S, T) = (P, K_{\text{total}}) \quad \text{where} \quad K_{\text{total}} = K_1 \circ K_2 \circ \cdots \circ K_k$$

and $K_{\text{total}}$ witnesses: $\llbracket P \rrbracket_T = \llbracket S \rrbracket_{\text{spec}}$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Type $T$ (parabolic, dispersive, etc.) | Target architecture / semantics | x86, ARM, LLVM, JVM |
| User-supplied functionals $(\Phi, \mathfrak{D}, G)$ | Source program + annotations | Annotated source code with contracts |
| Factory TM-1 (Gate Evaluators) | Frontend / Parser + Type Checker | Lexing, parsing, semantic analysis |
| Factory TM-2 (Barrier Implementations) | Middle-end / Optimizer | IR transformations, optimizations |
| Factory TM-3 (Surgery Schemas) | Error Recovery / Repair Engine | Automatic program repair, refactoring |
| Factory TM-4 (Equivalence + Transport) | Backend / Code Generator | Instruction selection, register allocation |
| Factory TM-5 (Lock Backend) | Verifier / Proof Checker | Certificate validation, proof checking |
| Sieve run | Compilation execution | End-to-end compilation trace |
| VICTORY | Successful compilation | Well-typed, verified output |
| Surgery path | Iterative refinement | Fix-and-recompile cycle |
| $K^{\text{inc}}$ | Classified error | Specific failure mode with diagnostic |
| Gate contract | Interface specification | Pre/post conditions, type signatures |
| Certificate composition | Proof composition | Transitive closure of refinement proofs |
| Acyclic factory graph | Phase ordering | Topologically sorted compilation passes |
| Soundness inheritance | Semantic preservation | Refinement relation preserved |
| Contract satisfaction | Interface adherence | All specifications met |
| Termination bound | Compilation complexity | $O(n \cdot k \cdot B)$ where $B$ = max iterations |

---

## Pipeline Stages

### Stage 1: Frontend (Factory TM-1 - Gate Evaluators)

**Compiler Analog:** Lexical analysis, parsing, semantic analysis, type checking.

**Input:** Raw source specification + type selection $T$

**Output:** Typed intermediate representation + gate certificates $\{K_i^+\}_{i=1}^{17}$

**Operations:**
- Lexical analysis: Tokenize source into lexemes
- Parsing: Build abstract syntax tree (AST)
- Name resolution: Resolve identifiers to declarations
- Type checking: Verify type correctness, produce typing derivations
- Semantic analysis: Check contract satisfaction, produce gate certificates

**Certificate Production:**
```
K_Frontend = {
  parse_certificate: AST is valid parse of source,
  type_certificate: Gamma |- e : tau,
  gate_certificates: [K_1^+, K_2^+, ..., K_17^+],
  well_formedness: All interfaces satisfied
}
```

**Correspondence to FACT-Instantiation:**
- TM-1 instantiates gate evaluators from user-supplied $(\Phi, \mathfrak{D})$
- Each gate $V_i^T$ corresponds to a specific semantic check
- 17 gates cover: energy bounds, dissipation, compactness, scale, barriers, topology, etc.

---

### Stage 2: Middle-end (Factory TM-2 - Barrier Implementations)

**Compiler Analog:** Intermediate representation transformations, optimizations.

**Input:** Typed IR from Stage 1 + gate certificates

**Output:** Optimized IR + barrier certificates $\{\mathcal{B}_j^T\}$

**Operations:**
- Control flow analysis: Build CFG, compute dominators
- Data flow analysis: Reaching definitions, liveness, available expressions
- Optimization passes: Dead code elimination, constant propagation, loop optimization
- Barrier checking: Verify optimization preconditions, detect potential issues

**Certificate Production:**
```
K_Middleend = {
  optimization_proofs: [proof that each transformation preserves semantics],
  barrier_certificates: {
    B_TypeII: No type-II singularity (controlled complexity),
    B_Cap: Capacity bounds satisfied (memory safety),
    B_Gap: Spectral gap maintained (termination guarantee)
  },
  invariant_preservation: All loop invariants maintained
}
```

**Correspondence to FACT-Instantiation:**
- TM-2 instantiates barrier implementations from TM-1 outputs
- Barriers detect genuine obstructions (not false alarms)
- Barrier breach triggers surgery (Stage 3) rather than compilation failure

---

### Stage 3: Repair Engine (Factory TM-3 - Surgery Schemas)

**Compiler Analog:** Automatic program repair, refactoring, error recovery.

**Input:** Barrier breach notification + obstruction certificate

**Output:** Repaired IR + surgery certificate (if successful)

**Operations:**
- Fault localization: Identify the precise location of the issue
- Repair synthesis: Generate candidate repairs from profile library
- Repair validation: Verify that repairs preserve intended semantics
- Energy-decreasing guarantee: Ensure repair makes measurable progress

**Certificate Production:**
```
K_Surgery = {
  fault_location: Precise identification of obstruction,
  repair_action: Description of applied surgery,
  progress_proof: Energy decreased by delta >= epsilon_T,
  admissibility: Repair satisfies canonicity + capacity bounds,
  iteration_bound: At most N_max surgeries before completion
}
```

**Correspondence to FACT-Instantiation:**
- TM-3 instantiates surgery schemas from profile library (if type admits surgery)
- Surgery is only applied when barrier detects genuine obstruction
- Each surgery decreases a well-founded measure (energy + surgery count)
- Surgery count bounded by $N_{\max} = \lfloor \Phi(x_0)/\delta_{\text{surgery}} \rfloor$

---

### Stage 4: Backend (Factory TM-4 - Equivalence + Transport)

**Compiler Analog:** Instruction selection, register allocation, code generation.

**Input:** Optimized IR + barrier/surgery certificates

**Output:** Target code + semantic equivalence certificate

**Operations:**
- Instruction selection: Map IR operations to target instructions
- Register allocation: Assign variables to registers (graph coloring)
- Instruction scheduling: Order instructions for pipeline efficiency
- Code emission: Generate target machine code or bytecode

**Certificate Production:**
```
K_Backend = {
  instruction_selection: IR_op |--> Target_instruction (bijective),
  register_allocation: No spill errors, all live ranges covered,
  scheduling: No hazards, correct ordering,
  equivalence: [[Target]] = [[IR]] (simulation relation),
  transport: Certificates valid across representation change
}
```

**Correspondence to FACT-Instantiation:**
- TM-4 instantiates equivalence/transport from symmetry group $G$ and scaling data
- Transport lemmas ensure certificate validity preserved across equivalence moves
- Simulation relation witnesses semantic preservation

---

### Stage 5: Verification (Factory TM-5 - Lock Backend)

**Compiler Analog:** Final verification, proof checking, certificate validation.

**Input:** Target code + complete certificate chain

**Output:** Final verified output + end-to-end certificate

**Operations:**
- Certificate checking: Verify all individual certificates are valid
- Composition verification: Check certificate chain is consistent
- Property verification: Confirm all specifications are satisfied
- Lock mechanism: Seal the verified artifact against tampering

**Certificate Production:**
```
K_Lock = {
  individual_checks: All K_i verified,
  composition_valid: K_1 o K_2 o ... o K_k is well-formed,
  specification_met: Output satisfies all input contracts,
  seal: Cryptographic/logical seal on verified artifact
}
```

**Correspondence to FACT-Instantiation:**
- TM-5 instantiates Lock backend from representation data (if available)
- Lock blocks when all gates pass: emit $K_{\text{Lock}}^{\text{blk}}$ (Global Regularity)
- Lock certificate seals the entire compilation

---

## Connections to Verified Systems

### 1. CompCert: Verified C Compiler

**System Description.** CompCert is a formally verified compiler for C, proven correct in Coq. Every compilation pass has a machine-checked proof of semantic preservation.

**Connection to FACT-Instantiation:**

| CompCert Component | FACT-Instantiation Analog |
|-------------------|---------------------------|
| C source program | User-supplied functionals $(\Phi, \mathfrak{D}, G)$ |
| Target architecture | Type selection $T$ |
| 8 front-end passes | Factory TM-1 (Gate Evaluators) |
| 6 back-end passes | Factory TM-4 (Equivalence + Transport) |
| Coq proofs per pass | Certificates $K_i$ per factory |
| Semantic preservation | Soundness inheritance |
| CompCert main theorem | FACT-Instantiation output guarantee |

**Structural Parallel:**
- CompCert: Source $\xrightarrow{P_1}$ Clight $\xrightarrow{P_2}$ C#minor $\xrightarrow{P_3}$ ... $\xrightarrow{P_{14}}$ Assembly
- FACT-Instantiation: User data $\xrightarrow{TM-1}$ Gates $\xrightarrow{TM-2}$ Barriers $\xrightarrow{TM-3}$ Surgery $\xrightarrow{TM-4}$ Transport $\xrightarrow{TM-5}$ Lock

**Key Insight:** Both systems achieve end-to-end correctness by:
1. Decomposing into modular, independently verified passes
2. Proving each pass preserves a refinement relation
3. Composing refinement proofs transitively

**CompCert Main Theorem:**
$$\forall P.\ \text{Safe}_C(P) \land \text{CompCert}(P) = \text{Some}(A) \Rightarrow \llbracket A \rrbracket = \llbracket P \rrbracket$$

**FACT-Instantiation Analog:**
$$\forall (T, \Phi, \mathfrak{D}, G).\ \text{Factories}(T, \Phi, \mathfrak{D}, G) = (\text{Sieve}, K_{\text{total}}) \Rightarrow \text{Sound}(\text{Sieve})$$

---

### 2. seL4: Verified Microkernel

**System Description.** seL4 is a formally verified microkernel with machine-checked proofs of functional correctness, security properties, and binary-level verification.

**Connection to FACT-Instantiation:**

| seL4 Component | FACT-Instantiation Analog |
|----------------|---------------------------|
| Abstract specification | Type $T$ + interface contracts |
| Executable specification | Factory TM-1 outputs (gate evaluators) |
| C implementation | Factory TM-2, TM-4 outputs |
| Binary code | Final sieve implementation |
| Functional correctness proof | Certificate chain $K_1 \circ \cdots \circ K_5$ |
| Security properties (integrity, confidentiality) | Barrier certificates (no unauthorized access) |
| Binary verification | Lock certificate (sealed artifact) |

**Refinement Stack:**

seL4 uses a multi-level refinement approach:
```
Abstract Spec (Isabelle/HOL)
       |
       | Functional correctness proof
       v
Executable Spec (Haskell)
       |
       | Refinement proof
       v
C Implementation
       |
       | Translation validation
       v
Binary (ARM/x86)
```

This directly parallels the factory sequence:
```
User Specification (T, Phi, D, G)
       |
       | TM-1 (Gate Evaluators)
       v
Intermediate Representation
       |
       | TM-2, TM-3 (Barriers, Surgery)
       v
Optimized Representation
       |
       | TM-4 (Equivalence + Transport)
       v
Target Implementation
       |
       | TM-5 (Lock Backend)
       v
Verified Artifact
```

**Full Abstraction Property:** seL4 provides full abstraction: the abstract specification fully captures the behavior observable at the binary level. Similarly, FACT-Instantiation guarantees that the sieve output (VICTORY, surgery path, or $K^{\text{inc}}$) is fully determined by the user specification.

---

### 3. CakeML: Verified ML Compiler

**System Description.** CakeML is a verified compiler for a subset of ML, with a verified runtime and garbage collector.

**Connection to FACT-Instantiation:**

| CakeML Component | FACT-Instantiation Analog |
|------------------|---------------------------|
| ML source | User-supplied functionals |
| Type inference | Gate evaluators (type-related) |
| CPS transformation | Barrier implementations (control flow) |
| Closure conversion | Surgery schemas (representation change) |
| Data representation | Transport lemmas (equivalence moves) |
| Verified GC | Lock backend (resource management) |
| Bootstrap verification | Self-applicability (meta-level soundness) |

**End-to-End Theorem:**
$$\text{CakeML}(\text{prog}) = \text{Some}(\text{binary}) \Rightarrow \text{binary} \text{ behaves as } \llbracket \text{prog} \rrbracket_{\text{ML}}$$

**Self-Applicability:** CakeML is verified to compile itself. This corresponds to the meta-level soundness of FACT-Instantiation: the factory system can instantiate itself, producing verified factory implementations.

---

### 4. Certified Toolchains

**Definition.** A certified toolchain is a sequence of verified tools that together provide end-to-end guarantees from source specification to deployed artifact.

**Examples:**

| Toolchain Component | FACT-Instantiation Stage | Certificate |
|--------------------|-------------------------|-------------|
| Verified parser (e.g., Menhir) | TM-1 | Parsing correctness |
| Verified optimizer (e.g., Alive) | TM-2 | Optimization soundness |
| Verified allocator (e.g., verified GC) | TM-3/TM-4 | Memory safety |
| Verified linker | TM-4 | Linking correctness |
| Proof-carrying code | TM-5 | End-to-end certificate |

**Toolchain Composition Theorem:**

If each tool $T_i$ is verified with certificate $K_i$, and the interfaces match:
$$\text{Post}(T_i) \subseteq \text{Pre}(T_{i+1})$$

Then the composed toolchain $T_1 \circ T_2 \circ \cdots \circ T_k$ is verified with certificate:
$$K_{\text{toolchain}} = K_1 \circ K_2 \circ \cdots \circ K_k$$

This is exactly the contract satisfaction property of FACT-Instantiation.

---

### 5. Full Abstraction

**Definition.** A compiler provides full abstraction if:
$$\forall P_1, P_2.\ \llbracket P_1 \rrbracket_{\text{source}} \approx \llbracket P_2 \rrbracket_{\text{source}} \Leftrightarrow \llbracket C(P_1) \rrbracket_{\text{target}} \approx \llbracket C(P_2) \rrbracket_{\text{target}}$$

where $\approx$ denotes observational equivalence.

**Connection to FACT-Instantiation:**

Full abstraction ensures that:
1. **Soundness:** Equivalent source programs compile to equivalent target programs
2. **Completeness:** Inequivalent source programs compile to inequivalent target programs
3. **No information leakage:** The target reveals nothing beyond what the source specifies

The factory composition in FACT-Instantiation preserves observational equivalence through:
- Gate certificates witnessing property preservation
- Transport lemmas ensuring equivalence moves are valid
- Lock certificates sealing the full abstraction property

**Secure Compilation:**

Full abstraction implies secure compilation: adversarial target contexts cannot distinguish compiled programs beyond what source contexts can distinguish. This corresponds to the barrier certificates preventing unauthorized information flow.

---

## Proof Structure

### Step 1: Factory Composition (Acyclic Dependencies)

**Claim:** The factory sequence has acyclic dependencies.

**Proof:** By construction, factories are numbered in topological order:
- TM-1 depends only on user-supplied data
- TM-2 depends on TM-1 outputs
- TM-3 depends on TM-1, TM-2 outputs
- TM-4 depends on TM-1, TM-2, TM-3 outputs
- TM-5 depends on TM-1, TM-2, TM-3, TM-4 outputs

No factory TM-$k$ depends on TM-$(k+1)$ or later. The dependency graph is a DAG.

**Compiler Interpretation:** This corresponds to phase ordering in compilers:
- Parsing before type checking
- Type checking before optimization
- Optimization before code generation
- Code generation before verification

---

### Step 2: Non-Circularity (No Self-Reference)

**Claim:** No factory output depends on itself.

**Proof:** Each factory $F_i$ transforms input $\text{IR}_{i-1}$ to output $\text{IR}_i$ where:
- $\text{IR}_{i-1}$ is fully determined before $F_i$ executes
- $F_i$ produces $\text{IR}_i$ as a function of $\text{IR}_{i-1}$ and fixed templates
- No feedback loop exists within a single factory

**Compiler Interpretation:** Each compiler pass reads its input, performs a transformation, and produces output without modifying the input. This is the standard functional compilation model.

---

### Step 3: Soundness Inheritance (Semantic Preservation)

**Claim:** Each factory produces sound output; composition preserves soundness.

**Proof:** By the soundness theorems for individual factories:
- TM-1: $V_i^T(x, \Gamma) = (\text{YES}, K_i^+) \Rightarrow P_i^T(x)$ holds
- TM-2: Barrier blocks $\Rightarrow$ obstruction is genuine
- TM-3: Surgery decreases energy and satisfies admissibility
- TM-4: Transport preserves certificate validity
- TM-5: Lock verifies all certificates

Composition: If $K_i$ witnesses $\llbracket \text{IR}_i \rrbracket \subseteq \llbracket \text{IR}_{i-1} \rrbracket$, then:
$$\llbracket \text{IR}_k \rrbracket \subseteq \llbracket \text{IR}_{k-1} \rrbracket \subseteq \cdots \subseteq \llbracket \text{IR}_0 \rrbracket$$

By transitivity, the composed certificate witnesses end-to-end refinement.

---

### Step 4: Contract Satisfaction (Interface Preservation)

**Claim:** All interface specifications are preserved across factories.

**Proof:** Each gate contract specifies:
- Pre-certificates required
- Post-certificates produced
- Routing rules

Factory composition ensures $\text{Post}(F_i) \subseteq \text{Pre}(F_{i+1})$ by:
- Post-certificates of $F_i$ are inputs to $F_{i+1}$
- Type-checking the factory interfaces at meta-level
- Transport lemmas handling representation changes

**Compiler Interpretation:** This is interface compatibility in modular compilation. Each pass produces output that the next pass can consume.

---

### Step 5: Termination (Bounded Execution)

**Claim:** The pipeline terminates in bounded time.

**Proof:**
1. **Gate termination:** Each $V_i^T$ terminates (finite computation on bounded data)
2. **Surgery termination:** Each surgery decreases well-founded measure $(E, n)$ where:
   - $E$ = current energy
   - $n$ = surgery count
   - Lexicographic ordering: $(E_1, n_1) < (E_2, n_2)$ iff $E_1 < E_2$ or ($E_1 = E_2$ and $n_1 < n_2$)
3. **Surgery bound:** $N_{\max} = \lfloor \Phi(x_0)/\delta_{\text{surgery}} \rfloor$
4. **Total bound:** Steps $\leq 5 \cdot N_{\max} \cdot B$ where $B$ = max factory iterations

**Complexity:** $O(n \cdot k \cdot N_{\max} \cdot B)$ where $n$ = input size, $k$ = number of factories.

---

### Step 6: Output Trichotomy (Complete Classification)

**Claim:** The pipeline terminates with exactly one of three outcomes.

**Proof:** At each point, the execution state is one of:
1. **Processing:** Currently executing some factory
2. **VICTORY:** All factories completed successfully, Lock blocked
3. **Surgery path:** Barrier breach detected, admissible repair available
4. **$K^{\text{inc}}$:** Barrier breach detected, no admissible repair

The trichotomy is exhaustive because:
- If processing completes without barrier breach: VICTORY
- If barrier breach with admissible repair: Surgery path (return to factory sequence)
- If barrier breach without admissible repair: $K^{\text{inc}}$ (explicit obstruction)

Surgery iteration is bounded (Step 5), so eventually one of VICTORY or $K^{\text{inc}}$ is reached.

---

## Certificate Construction

The proof is constructive. Given valid user specification, we extract explicit certificates:

**End-to-End Certificate:**
```
K_total = {
  source: (T, Phi, D, G),
  target: Sieve_implementation,

  factory_chain: [
    { factory: TM-1, input: user_data, output: gates, cert: K_1 },
    { factory: TM-2, input: gates, output: barriers, cert: K_2 },
    { factory: TM-3, input: barriers, output: surgery_ready, cert: K_3 },
    { factory: TM-4, input: surgery_ready, output: transport, cert: K_4 },
    { factory: TM-5, input: transport, output: locked, cert: K_5 }
  ],

  composition_proof: K_1 o K_2 o K_3 o K_4 o K_5,

  properties: {
    soundness: "Sieve output matches specification",
    termination: "Bounded by N_max surgeries",
    completeness: "All specifications addressed",
    security: "No unauthorized information flow"
  },

  outcome: VICTORY | SurgeryPath(repairs) | K^inc(obstruction)
}
```

---

## Quantitative Summary

| Property | Bound |
|----------|-------|
| Number of factories | 5 (TM-1 through TM-5) |
| Gates per factory TM-1 | 17 |
| Maximum surgery iterations | $N_{\max} = \lfloor \Phi(x_0)/\delta_{\text{surgery}} \rfloor$ |
| Certificate chain length | 5 (one per factory) |
| Total compilation steps | $O(n \cdot k \cdot N_{\max} \cdot B)$ |
| Verification complexity | $O(\text{poly}(n))$ for certificate checking |
| Output modes | 3 (VICTORY, Surgery path, $K^{\text{inc}}$) |

---

## Algorithmic Implementation Pattern

```
function CertifiedPipeline(spec: Specification, T: Type):
    // Stage 1: Frontend (TM-1)
    (gates, K_1) := GateEvaluators(spec, T)
    if gates.has_fatal_error():
        return FatalError("Invalid specification")

    // Stage 2: Middle-end (TM-2)
    (barriers, K_2) := BarrierImplementations(gates, K_1)

    // Stage 3: Surgery loop
    surgery_count := 0
    while barriers.has_breach() and surgery_count < N_max:
        if not barriers.breach.is_admissible():
            return K_inc(barriers.breach.obstruction)

        (repaired, K_3) := SurgerySchemas(barriers, barriers.breach)
        barriers := BarrierImplementations(repaired, K_2 o K_3)
        surgery_count++

    // Stage 4: Backend (TM-4)
    (transport, K_4) := EquivalenceTransport(barriers, K_2)

    // Stage 5: Verification (TM-5)
    (locked, K_5) := LockBackend(transport, K_4)

    // Compose certificates
    K_total := K_1 o K_2 o K_3 o K_4 o K_5

    return (locked.sieve, K_total, VICTORY)
```

---

## Summary

The FACT-Instantiation theorem, translated to complexity theory, establishes that:

1. **Full compilation pipelines are compositional:** A sequence of verified factories, each with its own correctness certificate, composes to give an end-to-end verified compiler.

2. **Non-circularity ensures well-foundedness:** The acyclic factory dependency graph prevents self-reference and ensures termination.

3. **Soundness is transitive:** Semantic preservation at each stage implies semantic preservation end-to-end.

4. **Contracts enable modularity:** Interface specifications allow independent verification of each factory while ensuring they compose correctly.

5. **Surgery provides error recovery:** When compilation encounters obstructions, bounded repair iterations restore progress toward completion.

6. **Output is completely classified:** Every compilation attempt terminates with one of three outcomes: success with proof, iterative repair, or explicit obstruction.

**Connections to Verified Systems:**

| Verified System | FACT-Instantiation Analog |
|----------------|---------------------------|
| CompCert (verified C compiler) | Factory sequence with refinement proofs |
| seL4 (verified microkernel) | Multi-level refinement stack |
| CakeML (verified ML compiler) | Bootstrap verification, self-applicability |
| Certified toolchains | Composable verified components |
| Full abstraction | Observational equivalence preservation |

The key insight is that **verified compilation is factory-based instantiation**: given a specification, a sequence of certified factories produces a verified implementation, with the certificate chain witnessing end-to-end correctness.

---

## Literature

**Verified Compilation:**
- {cite}`Leroy09`: CompCert verified C compiler. The multi-pass structure with per-pass proofs corresponds to the factory sequence in FACT-Instantiation.
- {cite}`Kumar14`: CakeML verified ML compiler. Bootstrap verification corresponds to self-instantiation of factories.

**Verified Operating Systems:**
- {cite}`Klein09`: seL4 verified microkernel. Multi-level refinement stack parallels the factory hierarchy.
- {cite}`Klein14`: Comprehensive verification from specification to binary.

**Type Theory:**
- {cite}`HoTTBook`: Homotopy Type Theory. Type-theoretic interpretation of certificates and proofs.
- {cite}`Aczel77`: Well-founded induction. Termination of surgery iterations.

**Secure Compilation:**
- {cite}`Abadi99`: Full abstraction for compiler correctness. Observational equivalence preservation.
- {cite}`Patrignani19`: Secure compilation survey. Connection between full abstraction and security.

**Proof-Carrying Code:**
- {cite}`Necula97`: Original PCC formulation. Certificates attached to compiled code.
- {cite}`Appel01`: Foundational proof-carrying code. Minimal trusted computing base.

**Singularity Resolution:**
- {cite}`Perelman03`: Ricci flow with surgery. Repair iterations for geometric obstructions, corresponding to compilation surgery.

