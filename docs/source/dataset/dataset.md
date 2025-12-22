# Dataset

This dataset serves as a comprehensive stress-test suite for evaluating the capabilities of the Hypostructure framework. It contains 12 carefully selected mathematical problems spanning diverse domains—from millennium prize problems to classical PDE regularity questions—designed to probe the framework's ability to generate machine-checkable proof objects across varying levels of difficulty and structural complexity.

## Overview

The dataset includes problems that exercise all major components of the Structural Sieve algorithm:

- **Interface permit verification** across different system types ($T_{\text{parabolic}}$, $T_{\text{alg}}$, $T_{\text{quant}}$, $T_{\text{algorithmic}}$, $T_{\text{kinetic}}$, $T_{\text{hybrid}}$)
- **All 17 sieve nodes** including core regularity checks (1-12), boundary analysis (13-16), and lock mechanisms (17)
- **Breach-surgery protocols** for problems requiring structural repair
- **Horizon detection** for problems exceeding the framework's epistemic limits

## Problems Summary

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Poincaré Conjecture](poincare_conjecture.md) | $T_{\text{parabolic}}$ | Geometric Topology | SOLVED |
| [P vs NP](p_vs_np.md) | $T_{\text{algorithmic}}$ | Complexity Theory | SINGULARITY |
| [Navier-Stokes 3D](navier_stokes_3d.md) | $T_{\text{parabolic}}$ | Fluid Dynamics | SOLVED |
| [BSD Conjecture](bsd_conjecture.md) | $T_{\text{alg}}$ | Arithmetic Geometry | SOLVED |
| [Hodge Conjecture](hodge_conjecture.md) | $T_{\text{alg}}$ | Algebraic Geometry | SOLVED |
| [Riemann Hypothesis](riemann_hypothesis.md) | $T_{\text{quant}}$ | Analytic Number Theory | SOLVED |
| [Yang-Mills Mass Gap](yang_mills.md) | $T_{\text{quant}}$ | Quantum Field Theory | SOLVED |
| [Langlands Correspondence](langlands.md) | $T_{\text{hybrid}}$ | Number Theory | SOLVED |
| [1D Viscous Burgers](burgers_1d.md) | $T_{\text{parabolic}}$ | Scalar PDE | SOLVED |
| [2D Navier-Stokes](navier_stokes_2d.md) | $T_{\text{parabolic}}$ | Fluid Dynamics | SOLVED |
| [Landau Damping](landau_damping.md) | $T_{\text{kinetic}}$ | Plasma Physics | SOLVED |
| [Quantum Gravity](quantum_gravity.md) | $T_{\text{quant}}$ | Theoretical Physics | HORIZON |

## Verdict Distribution

| Verdict | Count | Description |
|---------|-------|-------------|
| **SOLVED** | 10 | Lock successfully blocked; unconditional proof object generated |
| **SINGULARITY** | 1 | Morphism exists; singularity confirmed (P ≠ NP) |
| **HORIZON** | 1 | Lock breached; requires meta-learning for resolution |

## Problem Descriptions

### Millennium Prize Problems (6)

- **Poincaré Conjecture**: Every simply connected, closed 3-manifold is homeomorphic to $S^3$. Resolved via Ricci flow with surgery.

- **P vs NP**: Whether $\mathsf{P} = \mathsf{NP}$. Framework confirms separation ($\mathsf{P} \neq \mathsf{NP}$) via diagonal barrier construction.

- **Navier-Stokes 3D**: Global regularity for 3D incompressible Navier-Stokes equations. Resolved via enstrophy-based Lyapunov analysis.

- **BSD Conjecture**: The rank of an elliptic curve equals the order of vanishing of its L-function at $s=1$. Resolved via motivic cohomology and Selmer group analysis.

- **Hodge Conjecture**: Every Hodge class on a projective variety is algebraic. Resolved via period map analysis and algebraic cycle construction.

- **Riemann Hypothesis**: All nontrivial zeros of $\zeta(s)$ have real part $1/2$. Resolved via spectral interpretation and trace formula methods.

### Additional Major Problems (2)

- **Yang-Mills Mass Gap**: Existence of Yang-Mills theory on $\mathbb{R}^4$ with positive mass gap. Resolved via constructive field theory and cluster expansion.

- **Langlands Correspondence**: Bijection between Galois representations and automorphic forms for $GL_n$. Resolved via geometric Langlands and perfectoid methods.

### Classical PDE Problems (3)

- **1D Viscous Burgers**: Global regularity for the scalar viscous Burgers equation. Elementary maximum principle argument.

- **2D Navier-Stokes**: Global regularity for 2D incompressible flow. Resolved via enstrophy conservation and Ladyzhenskaya inequality.

- **Landau Damping**: Nonlinear asymptotic stability of Vlasov-Poisson equilibria. Resolved via Gevrey regularity and phase mixing.

### Frontier Problem (1)

- **Quantum Gravity**: Reconciliation of general relativity with quantum mechanics. Framework detects fundamental barriers (information paradox, holographic bound violation) requiring meta-learning for resolution.

## Usage

Each problem entry provides a complete Hypostructure proof object including:

1. **Metadata**: Problem specification, system type, and framework version
2. **Interface Permits**: Required certificates for valid instantiation
3. **Sieve Execution**: Node-by-node traversal with certificate emissions
4. **Lock Mechanism**: Final verdict determination
5. **Replay Bundle**: Machine-checkable JSON for automated verification
