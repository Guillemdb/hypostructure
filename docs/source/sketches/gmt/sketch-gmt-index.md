---
title: "GMT Translation Sketches - Master Index"
---

# Geometric Measure Theory Translation Sketches

## Overview

This index provides a comprehensive guide to the Geometric Measure Theory (GMT) translations of hypostructure theorems. Each sketch establishes formal correspondences between abstract dynamical systems concepts and concrete geometric measure-theoretic constructions.

**Purpose:** These translations serve GMT researchers seeking to understand hypostructure through familiar measure-theoretic and geometric concepts, and dynamical systems researchers interested in the geometric aspects of their theorems.

## Translation Framework

| Hypostructure | GMT |
|---------------|-----|
| State space $\mathcal{X}$ | Metric measure space $(X, d, \mu)$ |
| Semiflow $S_t$ | Geometric flow, Heat flow |
| Energy functional $\Phi$ | Dirichlet energy, Area functional |
| Dissipation $\mathfrak{D}$ | Entropy production, Energy decay |
| Singularity $\Sigma$ | Singular set, Blow-up locus |
| Surgery | Excision, Regularization |
| Global attractor | Minimal surface, Harmonic map |
| Certificate $K$ | Regularity estimate, Monotonicity formula |

---

## Kernel Theorems (KRNL-*)

Kernel theorems establish foundational properties: fixed-point characterizations, equivariance, regularity inheritance, and structural dichotomies.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [krnl-consistency.md](krnl-consistency.md) | KRNL-Consistency | Regularity via monotonicity formulas |
| [krnl-equivariance.md](krnl-equivariance.md) | KRNL-Equivariance | Symmetric minimal surfaces |
| [krnl-exclusion.md](krnl-exclusion.md) | KRNL-Exclusion | Measure-theoretic obstructions |
| [krnl-hamilton-jacobi.md](krnl-hamilton-jacobi.md) | KRNL-HamiltonJacobi | Viscosity solutions in geometric flows |
| [krnl-jacobi.md](krnl-jacobi.md) | KRNL-Jacobi | Jacobi field analysis |
| [krnl-lyapunov.md](krnl-lyapunov.md) | KRNL-Lyapunov | Entropy functionals as Lyapunov functions |
| [krnl-metric-action.md](krnl-metric-action.md) | KRNL-MetricAction | Optimal transport and Wasserstein geometry |
| [krnl-openness.md](krnl-openness.md) | KRNL-Openness | Stability of regularity |
| [krnl-shadowing.md](krnl-shadowing.md) | KRNL-Shadowing | Epsilon-regularity theorems |
| [krnl-stiff-pairing.md](krnl-stiff-pairing.md) | KRNL-StiffPairing | Rigidity in calibrated geometry |
| [krnl-subsystem.md](krnl-subsystem.md) | KRNL-Subsystem | Dimensional reduction and slicing |
| [krnl-trichotomy.md](krnl-trichotomy.md) | KRNL-Trichotomy | Regular/singular/intermediate trichotomy |
| [krnl-weak-strong.md](krnl-weak-strong.md) | KRNL-WeakStrong | Weak-to-strong regularity lifting |

---

## Structure Theorems (THM-*)

Structure theorems establish the geometric architecture of the framework: decomposition, termination, and completeness.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [thm-168-slots.md](thm-168-slots.md) | THM-168Slots | Classification of singularity types |
| [thm-categorical-completeness.md](thm-categorical-completeness.md) | THM-CategoricalCompleteness | Completeness of geometric constructions |
| [thm-closure-termination.md](thm-closure-termination.md) | THM-ClosureTermination | Flow termination and limiting behavior |
| [thm-compactness-resolution.md](thm-compactness-resolution.md) | THM-CompactnessResolution | Compactness and bubble tree analysis |
| [thm-dag.md](thm-dag.md) | THM-DAG | Stratified structure of singular sets |
| [thm-epoch-termination.md](thm-epoch-termination.md) | THM-EpochTermination | Epoch-based flow analysis |
| [thm-expansion-adjunction.md](thm-expansion-adjunction.md) | THM-ExpansionAdjunction | Blow-up and blow-down adjunction |
| [thm-finite-runs.md](thm-finite-runs.md) | THM-FiniteRuns | Finite-time singularity analysis |
| [thm-meta-identifiability.md](thm-meta-identifiability.md) | THM-MetaIdentifiability | Tangent cone uniqueness |
| [thm-non-circularity.md](thm-non-circularity.md) | THM-NonCircularity | Well-foundedness of constructions |
| [thm-soundness.md](thm-soundness.md) | THM-Soundness | Soundness of regularity criteria |

---

## Factory Theorems (FACT-*)

Factory theorems govern geometric constructions: barrier methods, concentration, instantiation, and convergence.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [fact-barrier.md](fact-barrier.md) | FACT-Barrier | Barrier construction for regularity |
| [fact-gate.md](fact-gate.md) | FACT-Gate | Neck regions and surgery gates |
| [fact-germ-density.md](fact-germ-density.md) | FACT-GermDensity | Density estimates for currents |
| [fact-instantiation.md](fact-instantiation.md) | FACT-Instantiation | Geometric measure instantiation |
| [fact-lock.md](fact-lock.md) | FACT-Lock | Rigidity and locking phenomena |
| [fact-min-inst.md](fact-min-inst.md) | FACT-MinInstantiation | Minimal surface construction |
| [fact-soft-attr.md](fact-soft-attr.md) | FACT-SoftAttr | Soft attraction to regular sets |
| [fact-soft-km.md](fact-soft-km.md) | FACT-SoftKM | Kähler-Morse theory |
| [fact-soft-morse.md](fact-soft-morse.md) | FACT-SoftMorse | Morse theory for energy functionals |
| [fact-soft-profdec.md](fact-soft-profdec.md) | FACT-SoftProfDec | Profile decomposition |
| [fact-soft-rigidity.md](fact-soft-rigidity.md) | FACT-SoftRigidity | Soft rigidity phenomena |
| [fact-soft-wp.md](fact-soft-wp.md) | FACT-SoftWP | Soft well-posedness |
| [fact-surgery.md](fact-surgery.md) | FACT-Surgery | Geometric surgery constructions |
| [fact-transport.md](fact-transport.md) | FACT-Transport | Optimal transport theory |
| [fact-valid-inst.md](fact-valid-inst.md) | FACT-ValidInstantiation | Valid geometric realizations |

---

## Resolution Theorems (RESOLVE-*)

Resolution theorems handle surgery procedures, obstruction removal, and regularization mechanics.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [resolve-admissibility.md](resolve-admissibility.md) | RESOLVE-Admissibility | Admissibility of surgery modifications |
| [resolve-auto-admit.md](resolve-auto-admit.md) | RESOLVE-AutoAdmit | Automatic regularity detection |
| [resolve-auto-profile.md](resolve-auto-profile.md) | RESOLVE-AutoProfile | Automatic stratification |
| [resolve-auto-surgery.md](resolve-auto-surgery.md) | RESOLVE-AutoSurgery | Automated surgery procedures |
| [resolve-conservation.md](resolve-conservation.md) | RESOLVE-Conservation | Mass conservation under surgery |
| [resolve-expansion.md](resolve-expansion.md) | RESOLVE-Expansion | Blow-up expansion analysis |
| [resolve-obstruction.md](resolve-obstruction.md) | RESOLVE-Obstruction | Obstruction theory for regularity |
| [resolve-profile.md](resolve-profile.md) | RESOLVE-Profile | Profile extraction and analysis |
| [resolve-tower.md](resolve-tower.md) | RESOLVE-Tower | Tower construction for blow-up |
| [resolve-weakest-pre.md](resolve-weakest-pre.md) | RESOLVE-WeakestPre | Weakest regularity preconditions |

---

## Promotion Theorems (UP-*)

Promotion theorems describe how local regularity extends to global regularity: epsilon-regularity, unique continuation, and propagation.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [up-absorbing.md](up-absorbing.md) | UP-Absorbing | Absorbing sets in flow dynamics |
| [up-algorithm-depth.md](up-algorithm-depth.md) | UP-AlgorithmDepth | Stratification depth bounds |
| [up-capacity.md](up-capacity.md) | UP-Capacity | Capacity estimates for singular sets |
| [up-catastrophe.md](up-catastrophe.md) | UP-Catastrophe | Catastrophe theory for singularities |
| [up-causal-barrier.md](up-causal-barrier.md) | UP-CausalBarrier | Causal structure in flows |
| [up-censorship.md](up-censorship.md) | UP-Censorship | Cosmic censorship analogs |
| [up-ergodic.md](up-ergodic.md) | UP-Ergodic | Ergodic properties of flows |
| [up-holographic.md](up-holographic.md) | UP-Holographic | Holographic principle for boundaries |
| [up-inc-aposteriori.md](up-inc-aposteriori.md) | UP-IncAPosteriori | A posteriori regularity estimates |
| [up-inc-complete.md](up-inc-complete.md) | UP-IncComplete | Completeness of regularity theory |
| [up-lock.md](up-lock.md) | UP-Lock | Rigidity and locking |
| [up-lock-back.md](up-lock-back.md) | UP-LockBack | Backward propagation of regularity |
| [up-o-minimal.md](up-o-minimal.md) | UP-OMinimal | O-minimal structure of singular sets |
| [up-saturation.md](up-saturation.md) | UP-Saturation | Saturation phenomena |
| [up-scattering.md](up-scattering.md) | UP-Scattering | Scattering theory for flows |
| [up-shadow.md](up-shadow.md) | UP-Shadow | Shadow estimates |
| [up-shadow-retro.md](up-shadow-retro.md) | UP-ShadowRetro | Retrospective shadow analysis |
| [up-spectral.md](up-spectral.md) | UP-Spectral | Spectral analysis of operators |
| [up-surgery.md](up-surgery.md) | UP-Surgery | Surgery efficiency bounds |
| [up-symmetry-bridge.md](up-symmetry-bridge.md) | UP-SymmetryBridge | Symmetry breaking and restoration |
| [up-tame-smoothing.md](up-tame-smoothing.md) | UP-TameSmoothing | Tame smoothing of singularities |
| [up-type-ii.md](up-type-ii.md) | UP-TypeII | Type-II singularity analysis |
| [up-variety-control.md](up-variety-control.md) | UP-VarietyControl | Variety control for singular sets |

---

## Lock Theorems (LOCK-*)

Lock theorems establish lower bounds and rigidity results: when regularity genuinely fails or when uniqueness holds.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [lock-antichain.md](lock-antichain.md) | LOCK-Antichain | Antichain bounds for singularities |
| [lock-entropy.md](lock-entropy.md) | LOCK-Entropy | Entropy lower bounds |
| [lock-ergodic-mixing.md](lock-ergodic-mixing.md) | LOCK-ErgodicMixing | Mixing time bounds |
| [lock-hodge.md](lock-hodge.md) | LOCK-Hodge | Hodge-theoretic obstructions |
| [lock-kodaira.md](lock-kodaira.md) | LOCK-Kodaira | Kodaira-type vanishing |
| [lock-motivic.md](lock-motivic.md) | LOCK-Motivic | Motivic obstructions |
| [lock-periodic.md](lock-periodic.md) | LOCK-Periodic | Periodic orbit obstructions |
| [lock-product.md](lock-product.md) | LOCK-Product | Product structure bounds |
| [lock-reconstruction.md](lock-reconstruction.md) | LOCK-Reconstruction | Reconstruction obstructions |
| [lock-schematic.md](lock-schematic.md) | LOCK-Schematic | Schematic structure constraints |
| [lock-spectral-dist.md](lock-spectral-dist.md) | LOCK-SpectralDist | Spectral distribution bounds |
| [lock-spectral-gen.md](lock-spectral-gen.md) | LOCK-SpectralGen | Spectral gap lower bounds |
| [lock-spectral-quant.md](lock-spectral-quant.md) | LOCK-SpectralQuant | Spectral quantization |
| [lock-tactic-capacity.md](lock-tactic-capacity.md) | LOCK-TacticCapacity | Capacity lower bounds |
| [lock-tactic-scale.md](lock-tactic-scale.md) | LOCK-TacticScale | Scale separation barriers |
| [lock-tannakian.md](lock-tannakian.md) | LOCK-Tannakian | Tannakian obstructions |
| [lock-unique-attractor.md](lock-unique-attractor.md) | LOCK-UniqueAttractor | Unique tangent cone theorems |
| [lock-virtual.md](lock-virtual.md) | LOCK-Virtual | Virtual fundamental class |

---

## Action Theorems (ACT-*)

Action theorems govern surgery operations: how to modify geometric objects, align boundaries, and execute geometric modifications.

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [act-align.md](act-align.md) | ACT-Align | Boundary alignment and gluing |
| [act-compactify.md](act-compactify.md) | ACT-Compactify | Compactification procedures |
| [act-ghost.md](act-ghost.md) | ACT-Ghost | Ghost mass and auxiliary structures |
| [act-horizon.md](act-horizon.md) | ACT-Horizon | Horizon extension theorems |
| [act-lift.md](act-lift.md) | ACT-Lift | Lifting to higher regularity |
| [act-projective.md](act-projective.md) | ACT-Projective | Projective methods |
| [act-surgery.md](act-surgery.md) | ACT-Surgery | Geometric surgery procedures |

---

## Lemmas (LEM-*)

| File | Theorem | GMT Translation |
|------|---------|-----------------|
| [lem-bridge.md](lem-bridge.md) | LEM-Bridge | Bridging lemma for regularity |

---

## Statistics

- **Total sketches**: 98 files
- **Kernel theorems**: 13
- **Structure theorems**: 11
- **Factory theorems**: 15
- **Resolution theorems**: 10
- **Promotion theorems**: 23
- **Lock theorems**: 18
- **Action theorems**: 7
- **Lemmas**: 1

---

*This index is part of the hypostructure documentation. For the original mathematical theorems, see the main proof files.*
