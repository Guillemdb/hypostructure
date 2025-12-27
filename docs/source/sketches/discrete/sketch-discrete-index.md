---
title: "Complexity Theory Translation Sketches - Master Index"
---

# Complexity Theory Translation Sketches

## Overview

This index provides a comprehensive guide to the complexity-theoretic translations of hypostructure theorems. Each sketch establishes formal correspondences between continuous dynamical systems concepts (energy functionals, semiflows, singularities, surgery) and discrete computational models (complexity classes, reductions, proof systems, circuit families).

**Purpose:** These translations serve complexity theorists seeking to understand hypostructure through familiar computational concepts, and dynamical systems researchers interested in the algorithmic content of their theorems.

**Core Translation Framework:** The fundamental correspondence maps:

| Hypostructure | Complexity Theory |
|---------------|-------------------|
| State space $\mathcal{X}$ | Configuration space / Input domain |
| Semiflow $S_t$ | Transition function / Reduction |
| Energy functional $\Phi$ | Complexity measure (time/space/depth) |
| Dissipation $\mathfrak{D}$ | Resource consumption rate |
| Singularity $\Sigma$ | Hard instance / Barrier / Cut |
| Surgery | Proof transformation / Cut-elimination |
| Global attractor | Fixed point / Halting configuration |
| Certificate $K$ | Proof witness / Verification oracle |

---

## Quick Reference: The Translation Dictionary

### Energy and Resources
- **Energy $\Phi(x)$** = Computational resource (time, space, circuit depth)
- **Dissipation $\mathfrak{D}(x) > 0$** = Strict resource consumption per step
- **Compactness** = Finite reachability / Bounded configuration space
- **Lyapunov function** = Ranking function / Termination proof

### Dynamics and Computation
- **Trajectory** = Computation trace / Reduction chain
- **Fixed point** = Halting state / Accepting configuration
- **Periodic orbit** = Infinite loop / Non-termination
- **Attractor basin** = Decidable sublanguage

### Singularities and Hardness
- **Singularity** = Hard instance / Complexity barrier
- **Surgery** = Cut-elimination / Proof simplification
- **Profile extraction** = Kernelization / Hard core identification
- **Codimension** = Measure of "how rare" hard instances are

### Certificates and Proofs
- **Interface permit** = Complexity class membership
- **Lock certificate** = Lower bound / Separation result
- **Barrier blocked** = Reduction impossible / Oracle separation

---

## Kernel Theorems (KRNL-*)

The kernel theorems establish foundational properties of computational systems: fixed-point characterizations, symmetry preservation, tractability inheritance, and structural dichotomies.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-krnl-consistency.md](sketch-discrete-mt-krnl-consistency.md) | KRNL-Consistency | Fixed-point characterization via Immerman-Vardi theorem and LFP logic |
| [sketch-discrete-mt-krnl-equivariance.md](sketch-discrete-mt-krnl-equivariance.md) | KRNL-Equivariance | Symmetric circuit optimality: symmetry in problems forces symmetry in solutions |
| [sketch-discrete-mt-krnl-exclusion.md](sketch-discrete-mt-krnl-exclusion.md) | KRNL-Exclusion | Structural exclusion principle: no reduction from complete problem implies tractability |
| [sketch-discrete-mt-krnl-hamilton-jacobi.md](sketch-discrete-mt-krnl-hamilton-jacobi.md) | KRNL-HamiltonJacobi | Bellman equation characterization of optimal cost functions |
| [sketch-discrete-mt-krnl-jacobi.md](sketch-discrete-mt-krnl-jacobi.md) | KRNL-Jacobi | Resource-weighted shortest path as computational distance |
| [sketch-discrete-mt-krnl-lyapunov.md](sketch-discrete-mt-krnl-lyapunov.md) | KRNL-Lyapunov | Canonical ranking functions for termination proofs |
| [sketch-discrete-mt-krnl-metric-action.md](sketch-discrete-mt-krnl-metric-action.md) | KRNL-MetricAction | Extended resource complexity for general computational models |
| [sketch-discrete-mt-krnl-openness.md](sketch-discrete-mt-krnl-openness.md) | KRNL-Openness | Robust complexity separation: tractability is stable under perturbation |
| [sketch-discrete-mt-krnl-shadowing.md](sketch-discrete-mt-krnl-shadowing.md) | KRNL-Shadowing | Approximate-to-exact solution lifting and self-correction principles |
| [sketch-discrete-mt-krnl-stiff-pairing.md](sketch-discrete-mt-krnl-stiff-pairing.md) | KRNL-StiffPairing | Matrix rigidity bounds and tensor rank lower bounds |
| [sketch-discrete-mt-krnl-stiff-pairing-v2.md](sketch-discrete-mt-krnl-stiff-pairing-v2.md) | KRNL-StiffPairing v2 | Non-degeneracy in constraint satisfaction systems |
| [sketch-discrete-mt-krnl-subsystem.md](sketch-discrete-mt-krnl-subsystem.md) | KRNL-Subsystem | Tractability inheritance: restrictions of P problems remain in P |
| [sketch-discrete-mt-krnl-trichotomy.md](sketch-discrete-mt-krnl-trichotomy.md) | KRNL-Trichotomy | Computational trichotomy: P vs NP-intermediate vs NP-complete |
| [sketch-discrete-mt-krnl-weak-strong.md](sketch-discrete-mt-krnl-weak-strong.md) | KRNL-WeakStrong | Weak-to-strong solution lifting via regularity bootstrapping |

---

## Sieve and DAG Structure Theorems (THM-*)

These theorems establish the computational structure of the verification framework: soundness, termination, and categorical completeness.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-thm-dag.md](sketch-discrete-thm-dag.md) | THM-DAG | Branching program structure and polynomial-depth decision |
| [sketch-discrete-thm-soundness.md](sketch-discrete-thm-soundness.md) | THM-SOUNDNESS | Proof system soundness: every derivation step is valid |
| [sketch-discrete-thm-finite-runs.md](sketch-discrete-thm-finite-runs.md) | THM-FiniteRuns | Bounded-depth computation trees and termination guarantees |
| [sketch-discrete-thm-epoch-termination.md](sketch-discrete-thm-epoch-termination.md) | THM-EpochTermination | Phase-bounded computation and iterative refinement termination |
| [sketch-discrete-thm-closure-termination.md](sketch-discrete-thm-closure-termination.md) | THM-ClosureTermination | Transitive closure termination in polynomial time |
| [sketch-discrete-thm-compactness-resolution.md](sketch-discrete-thm-compactness-resolution.md) | THM-CompactnessResolution | Finite model property and SAT-based resolution |
| [sketch-discrete-thm-categorical-completeness.md](sketch-discrete-thm-categorical-completeness.md) | THM-CategoricalCompleteness | Completeness of the verification framework |
| [sketch-discrete-thm-meta-identifiability.md](sketch-discrete-thm-meta-identifiability.md) | THM-MetaIdentifiability | Problem fingerprinting: identical signatures imply equivalence |
| [sketch-discrete-thm-non-circularity.md](sketch-discrete-thm-non-circularity.md) | THM-NonCircularity | Well-foundedness of the reduction hierarchy |
| [sketch-discrete-thm-soft-backend-completeness.md](sketch-discrete-thm-soft-backend-completeness.md) | THM-SoftBackendCompleteness | Backend solver completeness for soft constraints |
| [sketch-discrete-thm-168-slots.md](sketch-discrete-thm-168-slots.md) | THM-168Slots | Classification of 168 proof strategy patterns |
| [sketch-discrete-thm-bode-sensitivity.md](sketch-discrete-thm-bode-sensitivity.md) | THM-BodeSensitivity | Sensitivity analysis and robustness bounds |
| [sketch-discrete-thm-expansion-adjunction.md](sketch-discrete-thm-expansion-adjunction.md) | THM-ExpansionAdjunction | Adjoint functor characterization of problem expansions |

---

## Factory and Instantiation Theorems (FACT-*)

Factory theorems govern the construction and validation of computational instances: witness compactness, compiler soundness, and fixed-point convergence.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-fact-barrier.md](sketch-discrete-mt-fact-barrier.md) | FACT-Barrier | Barrier methods for complexity lower bounds |
| [sketch-discrete-mt-fact-gate.md](sketch-discrete-mt-fact-gate.md) | FACT-Gate | Gate complexity and circuit depth bounds |
| [sketch-discrete-mt-fact-germ-density.md](sketch-discrete-mt-fact-germ-density.md) | FACT-GermDensity | Witness compactness: finite epsilon-nets cover witness classes |
| [sketch-discrete-mt-fact-instantiation.md](sketch-discrete-mt-fact-instantiation.md) | FACT-Instantiation | Instance generation and parameterized problem families |
| [sketch-discrete-mt-fact-lock.md](sketch-discrete-mt-fact-lock.md) | FACT-Lock | Lock construction for lower bound proofs |
| [sketch-discrete-mt-fact-min-inst.md](sketch-discrete-mt-fact-min-inst.md) | FACT-MinInstantiation | Minimal witness construction and kernelization |
| [sketch-discrete-mt-fact-soft-attr.md](sketch-discrete-mt-fact-soft-attr.md) | FACT-SoftAttr | Fixed-point convergence via PPAD and Brouwer/Banach theorems |
| [sketch-discrete-mt-fact-soft-km.md](sketch-discrete-mt-fact-soft-km.md) | FACT-SoftKM | Krasnoselskii-Mann iteration and averaged operator convergence |
| [sketch-discrete-mt-fact-soft-morse.md](sketch-discrete-mt-fact-soft-morse.md) | FACT-SoftMorse | SCC decomposition and DAG structure of computation graphs |
| [sketch-discrete-mt-fact-soft-profdec.md](sketch-discrete-mt-fact-soft-profdec.md) | FACT-SoftProfDec | Profile decomposition and multi-scale analysis |
| [sketch-discrete-mt-fact-soft-rigidity.md](sketch-discrete-mt-fact-soft-rigidity.md) | FACT-SoftRigidity | Rigidity phenomena in constraint propagation |
| [sketch-discrete-mt-fact-soft-wp.md](sketch-discrete-mt-fact-soft-wp.md) | FACT-SoftWP | Weakest precondition semantics and program verification |
| [sketch-discrete-mt-fact-surgery.md](sketch-discrete-mt-fact-surgery.md) | FACT-Surgery | Surgery construction for proof transformation |
| [sketch-discrete-mt-fact-transport.md](sketch-discrete-mt-fact-transport.md) | FACT-Transport | Optimal transport and Wasserstein distance in algorithms |
| [sketch-discrete-mt-fact-valid-inst.md](sketch-discrete-mt-fact-valid-inst.md) | FACT-ValidInstantiation | Compiler soundness: well-typed programs produce valid code |

---

## Resolution Theorems (RESOLVE-*)

Resolution theorems handle surgery procedures, obstruction removal, and the mechanics of simplifying complex structures.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-resolve-admissibility.md](sketch-discrete-mt-resolve-admissibility.md) | RESOLVE-Admissibility | Admissibility conditions for valid reductions |
| [sketch-discrete-mt-resolve-auto-admit.md](sketch-discrete-mt-resolve-auto-admit.md) | RESOLVE-AutoAdmit | Automatic type checking and decidable verification |
| [sketch-discrete-mt-resolve-auto-surgery.md](sketch-discrete-mt-resolve-auto-surgery.md) | RESOLVE-AutoSurgery | Automatic proof repair and gap filling |
| [sketch-discrete-mt-resolve-autoprofile.md](sketch-discrete-mt-resolve-autoprofile.md) | RESOLVE-AutoProfile | Automatic complexity profiling and bottleneck detection |
| [sketch-discrete-mt-resolve-conservation.md](sketch-discrete-mt-resolve-conservation.md) | RESOLVE-Conservation | Resource conservation: parsimonious reductions preserve measures |
| [sketch-discrete-mt-resolve-expansion.md](sketch-discrete-mt-resolve-expansion.md) | RESOLVE-Expansion | Instance expansion and parameterized reduction |
| [sketch-discrete-mt-resolve-obstruction.md](sketch-discrete-mt-resolve-obstruction.md) | RESOLVE-Obstruction | Obstruction theory for reduction impossibility |
| [sketch-discrete-mt-resolve-profile.md](sketch-discrete-mt-resolve-profile.md) | RESOLVE-Profile | Complexity profile extraction and analysis |
| [sketch-discrete-mt-resolve-tower.md](sketch-discrete-mt-resolve-tower.md) | RESOLVE-Tower | Tower-of-exponentials bounds in proof complexity |
| [sketch-discrete-mt-resolve-weakest-pre.md](sketch-discrete-mt-resolve-weakest-pre.md) | RESOLVE-WeakestPre | Weakest precondition calculus and Hoare logic |

---

## Promotion Theorems (UP-*)

Promotion theorems describe how local certificates combine to establish global properties: from approximate solutions to exact ones, from weak complexity bounds to strong separations.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-up-absorbing.md](sketch-discrete-mt-up-absorbing.md) | UP-Absorbing | Absorbing states and ergodic convergence |
| [sketch-discrete-mt-up-algorithm-depth.md](sketch-discrete-mt-up-algorithm-depth.md) | UP-AlgorithmDepth | Circuit depth bounds and parallel complexity |
| [sketch-discrete-mt-up-capacity.md](sketch-discrete-mt-up-capacity.md) | UP-Capacity | Capacity bounds and information-theoretic limits |
| [sketch-discrete-mt-up-catastrophe.md](sketch-discrete-mt-up-catastrophe.md) | UP-Catastrophe | Phase transitions and sharp thresholds |
| [sketch-discrete-mt-up-causal-barrier.md](sketch-discrete-mt-up-causal-barrier.md) | UP-CausalBarrier | Causal structure and communication complexity |
| [sketch-discrete-mt-up-censorship.md](sketch-discrete-mt-up-censorship.md) | UP-Censorship | IP = PSPACE via causal censorship and round bounds |
| [sketch-discrete-mt-up-ergodic.md](sketch-discrete-mt-up-ergodic.md) | UP-Ergodic | Ergodic theorems and average-case complexity |
| [sketch-discrete-mt-up-holographic.md](sketch-discrete-mt-up-holographic.md) | UP-Holographic | Holographic proofs and bulk-boundary correspondence |
| [sketch-discrete-mt-up-inc-aposteriori.md](sketch-discrete-mt-up-inc-aposteriori.md) | UP-IncAPosteriori | A posteriori error bounds and adaptive algorithms |
| [sketch-discrete-mt-up-inc-complete.md](sketch-discrete-mt-up-inc-complete.md) | UP-IncComplete | Incremental completeness and online algorithms |
| [sketch-discrete-mt-up-lock.md](sketch-discrete-mt-up-lock.md) | UP-Lock | Lock promotion and lower bound propagation |
| [sketch-discrete-mt-up-lock-back.md](sketch-discrete-mt-up-lock-back.md) | UP-LockBack | Backward lock propagation in proof search |
| [sketch-discrete-mt-up-o-minimal.md](sketch-discrete-mt-up-o-minimal.md) | UP-OMinimal | Advice hierarchy collapse via o-minimal definability |
| [sketch-discrete-mt-up-saturation.md](sketch-discrete-mt-up-saturation.md) | UP-Saturation | Saturation-based theorem proving |
| [sketch-discrete-mt-up-scattering.md](sketch-discrete-mt-up-scattering.md) | UP-Scattering | Scattering theory and asymptotic behavior |
| [sketch-discrete-mt-up-shadow.md](sketch-discrete-mt-up-shadow.md) | UP-Shadow | Shadow complexity and projection bounds |
| [sketch-discrete-mt-up-shadow-retro.md](sketch-discrete-mt-up-shadow-retro.md) | UP-ShadowRetro | Retrospective shadow analysis |
| [sketch-discrete-mt-up-spectral.md](sketch-discrete-mt-up-spectral.md) | UP-Spectral | Spectral methods and eigenvalue bounds |
| [sketch-discrete-mt-up-surgery.md](sketch-discrete-mt-up-surgery.md) | UP-Surgery | Proof system simulation and extended Frege |
| [sketch-discrete-mt-up-symmetry-bridge.md](sketch-discrete-mt-up-symmetry-bridge.md) | UP-SymmetryBridge | Symmetry-based algorithm design |
| [sketch-discrete-mt-up-tame-smoothing.md](sketch-discrete-mt-up-tame-smoothing.md) | UP-TameSmoothing | Smoothing techniques for discrete optimization |
| [sketch-discrete-mt-up-type-ii.md](sketch-discrete-mt-up-type-ii.md) | UP-TypeII | Type II error bounds in property testing |
| [sketch-discrete-mt-up-variety-control.md](sketch-discrete-mt-up-variety-control.md) | UP-VarietyControl | Algebraic variety bounds in constraint solving |

---

## Lock Theorems (LOCK-*)

Lock theorems establish lower bounds and impossibility results: when computation genuinely requires certain resources or when reductions cannot exist.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-lock-antichain.md](sketch-discrete-mt-lock-antichain.md) | LOCK-Antichain | Antichain bounds via Dilworth's theorem and parallel complexity |
| [sketch-discrete-mt-lock-entropy.md](sketch-discrete-mt-lock-entropy.md) | LOCK-Entropy | Entropy-based lower bounds and information complexity |
| [sketch-discrete-mt-lock-ergodic-mixing.md](sketch-discrete-mt-lock-ergodic-mixing.md) | LOCK-ErgodicMixing | Mixing time bounds and Markov chain complexity |
| [sketch-discrete-mt-lock-hodge.md](sketch-discrete-mt-lock-hodge.md) | LOCK-Hodge | Hodge-theoretic bounds and cohomological obstructions |
| [sketch-discrete-mt-lock-kodaira.md](sketch-discrete-mt-lock-kodaira.md) | LOCK-Kodaira | Vanishing theorem analogs and structural impossibilities |
| [sketch-discrete-mt-lock-motivic.md](sketch-discrete-mt-lock-motivic.md) | LOCK-Motivic | Motivic complexity and algebraic invariants |
| [sketch-discrete-mt-lock-periodic.md](sketch-discrete-mt-lock-periodic.md) | LOCK-Periodic | Strategy tables: problem classification determines algorithm class |
| [sketch-discrete-mt-lock-product.md](sketch-discrete-mt-lock-product.md) | LOCK-Product | Product structure bounds and tensor methods |
| [sketch-discrete-mt-lock-reconstruction.md](sketch-discrete-mt-lock-reconstruction.md) | LOCK-Reconstruction | Reconstruction complexity and learning lower bounds |
| [sketch-discrete-mt-lock-schematic.md](sketch-discrete-mt-lock-schematic.md) | LOCK-Schematic | Schematic complexity and representation bounds |
| [sketch-discrete-mt-lock-spectral-dist.md](sketch-discrete-mt-lock-spectral-dist.md) | LOCK-SpectralDist | Spectral distribution bounds in random matrix theory |
| [sketch-discrete-mt-lock-spectral-gen.md](sketch-discrete-mt-lock-spectral-gen.md) | LOCK-SpectralGen | Generator bounds via spectral gap on Cayley graphs |
| [sketch-discrete-mt-lock-spectral-quant.md](sketch-discrete-mt-lock-spectral-quant.md) | LOCK-SpectralQuant | Quantum spectral bounds and Hamiltonian complexity |
| [sketch-discrete-mt-lock-tactic-capacity.md](sketch-discrete-mt-lock-tactic-capacity.md) | LOCK-TacticCapacity | Capacity-based tactic selection |
| [sketch-discrete-mt-lock-tactic-scale.md](sketch-discrete-mt-lock-tactic-scale.md) | LOCK-TacticScale | Scale-dependent algorithm selection |
| [sketch-discrete-mt-lock-tannakian.md](sketch-discrete-mt-lock-tannakian.md) | LOCK-Tannakian | Tannakian reconstruction and categorical bounds |
| [sketch-discrete-mt-lock-unique-attractor.md](sketch-discrete-mt-lock-unique-attractor.md) | LOCK-UniqueAttractor | Unique fixed-point theorems and contraction mapping |
| [sketch-discrete-mt-lock-virtual.md](sketch-discrete-mt-lock-virtual.md) | LOCK-Virtual | Virtual structure bounds and motivic invariants |

---

## Action Theorems (ACT-*)

Action theorems govern surgery operations: how to transform proofs, align boundaries, and execute computational modifications.

| File | Theorem | Complexity Translation |
|------|---------|------------------------|
| [sketch-discrete-mt-act-align.md](sketch-discrete-mt-act-align.md) | ACT-Align | Compositional boundary alignment and interface verification |
| [sketch-discrete-mt-act-compactify.md](sketch-discrete-mt-act-compactify.md) | ACT-Compactify | Compactification methods and finite approximation |
| [sketch-discrete-mt-act-ghost.md](sketch-discrete-mt-act-ghost.md) | ACT-Ghost | Ghost variable elimination and auxiliary structure removal |
| [sketch-discrete-mt-act-horizon.md](sketch-discrete-mt-act-horizon.md) | ACT-Horizon | Horizon bounds and lookahead complexity |
| [sketch-discrete-mt-act-lift.md](sketch-discrete-mt-act-lift.md) | ACT-Lift | Lifting theorems and communication complexity |
| [sketch-discrete-mt-act-projective.md](sketch-discrete-mt-act-projective.md) | ACT-Projective | Projective methods and dimension reduction |
| [sketch-discrete-mt-act-surgery.md](sketch-discrete-mt-act-surgery.md) | ACT-Surgery | Cut-elimination and proof simplification via Gentzen's Hauptsatz |

---

## Cross-Reference: Translation Guide

For a systematic introduction to the translation methodology, see:

- **Hypostructure Primer**: Core concepts of the framework
- **Complexity Theory Background**: Standard complexity classes and their relationships
- **The Sieve Algorithm**: How the verification procedure maps to proof search

### Key Correspondences by Complexity Class

| Complexity Class | Hypostructure Analog | Key Theorems |
|-----------------|---------------------|--------------|
| P | Dispersion regime (energy spreads) | KRNL-Trichotomy, KRNL-Subsystem |
| NP | Certificate-verifiable outcomes | FACT-GermDensity, FACT-ValidInstantiation |
| PSPACE | Bounded surgery depth | UP-Censorship, RESOLVE-Tower |
| IP | Interactive certificate exchange | UP-Censorship (IP = PSPACE) |
| PH | Alternating quantifier depth | KRNL-Trichotomy, UP-TypeII |
| PPAD | Fixed-point existence | FACT-SoftAttr |
| Circuit Classes | Energy barriers at depth bounds | KRNL-StiffPairing, UP-AlgorithmDepth |

### Key Correspondences by Proof Technique

| Proof Technique | Hypostructure Method | Key Theorems |
|-----------------|---------------------|--------------|
| Diagonalization | Singularity construction | KRNL-Exclusion |
| Reductions | Flow continuation | RESOLVE-Conservation |
| Padding | Energy scaling | UP-Surgery |
| Relativization | Barrier analysis | KRNL-Exclusion, LOCK-* |
| Algebraic methods | Spectral analysis | LOCK-SpectralGen, UP-Spectral |
| Probabilistic methods | Ergodic averaging | UP-Ergodic, LOCK-ErgodicMixing |

---

## Statistics

- **Total sketches**: 101 files
- **Kernel theorems**: 14
- **Sieve/DAG theorems**: 13
- **Factory theorems**: 15
- **Resolution theorems**: 10
- **Promotion theorems**: 22
- **Lock theorems**: 18
- **Action theorems**: 7
- **Miscellaneous**: 2

---

## Contributing

Each sketch follows a standard format:
1. **Title and Overview**: The theorem name and its complexity-theoretic essence
2. **Complexity Theory Statement**: A formal theorem in computational terms
3. **Terminology Translation Table**: Systematic mapping of concepts
4. **Proof Sketch**: Key ideas translated to algorithmic reasoning
5. **Examples and Applications**: Concrete illustrations

When adding new translations, ensure:
- The core correspondence is stated explicitly
- All terminology mappings are bidirectional
- Computational implications are spelled out
- Connections to known complexity results are noted

---

*This index is part of the hypostructure documentation. For the original mathematical theorems, see the main proof files.*
