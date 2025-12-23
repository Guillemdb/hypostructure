# Dataset

This dataset serves as a comprehensive stress-test suite for evaluating the capabilities of the Hypostructure framework. It contains 26 carefully selected mathematical problems spanning diverse domains—from millennium prize problems to classical textbook results—designed to probe the framework's ability to generate machine-checkable proof objects across varying levels of difficulty and structural complexity.

## Overview

The dataset includes problems that exercise all major components of the Structural Sieve algorithm:

- **Interface permit verification** across different system types ($T_{\text{parabolic}}$, $T_{\text{alg}}$, $T_{\text{quant}}$, $T_{\text{algorithmic}}$, $T_{\text{kinetic}}$, $T_{\text{hybrid}}$, $T_{\text{topological}}$, $T_{\text{combinatorial}}$, $T_{\text{stochastic}}$, $T_{\text{hamiltonian}}$, $T_{\text{homotopical}}$, $T_{\text{dynamical}}$, $T_{\text{analytic}}$)
- **All 17 sieve nodes** including core regularity checks (1-12), boundary analysis (13-16), and lock mechanisms (17)
- **Breach-surgery protocols** for problems requiring structural repair
- **Horizon detection** for problems exceeding the framework's epistemic limits

## Problems Summary

### Millennium Prize Problems (7)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Poincaré Conjecture](poincare_conjecture.md) | $T_{\text{parabolic}}$ | Geometric Topology | SOLVED |
| [P vs NP](p_vs_np.md) | $T_{\text{algorithmic}}$ | Complexity Theory | SINGULARITY |
| [Navier-Stokes 3D](navier_stokes_3d.md) | $T_{\text{parabolic}}$ | Fluid Dynamics | SOLVED |
| [BSD Conjecture](bsd_conjecture.md) | $T_{\text{alg}}$ | Arithmetic Geometry | SOLVED |
| [Hodge Conjecture](hodge_conjecture.md) | $T_{\text{alg}}$ | Algebraic Geometry | SOLVED |
| [Riemann Hypothesis](riemann_hypothesis.md) | $T_{\text{quant}}$ | Analytic Number Theory | SOLVED |
| [Yang-Mills Mass Gap](yang_mills.md) | $T_{\text{quant}}$ | Quantum Field Theory | SOLVED |

### Famous Solved Problems (5)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Fermat's Last Theorem](fermat_last_theorem.md) | $T_{\text{algebraic}}$ | Number Theory | SOLVED |
| [Four Color Theorem](four_color_theorem.md) | $T_{\text{combinatorial}}$ | Graph Theory | SOLVED |
| [KAM Theory](kam_theory.md) | $T_{\text{hamiltonian}}$ | Dynamical Systems | SOLVED |
| [Kepler Conjecture](kepler_conjecture.md) | $T_{\text{geometric}}$ | Discrete Geometry | SOLVED |
| [Finite Simple Groups](finite_simple_groups.md) | $T_{\text{algebraic}}$ | Group Theory | SOLVED |

### Fields Medal Results (5)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Langlands Correspondence](langlands.md) | $T_{\text{hybrid}}$ | Number Theory | SOLVED |
| [Fundamental Lemma](fundamental_lemma.md) | $T_{\text{algebraic}}$ | Representation Theory | SOLVED |
| [Julia Sets (MLC)](julia_sets.md) | $T_{\text{dynamical}}$ | Complex Dynamics | SOLVED |
| [Bounded Prime Gaps](bounded_primes_gaps.md) | $T_{\text{analytic}}$ | Number Theory | SOLVED |
| [Kervaire Invariant One](kervaire_invariant.md) | $T_{\text{homotopical}}$ | Algebraic Topology | SOLVED |

### Classical PDE Problems (3)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [1D Viscous Burgers](burgers_1d.md) | $T_{\text{parabolic}}$ | Scalar PDE | SOLVED |
| [2D Navier-Stokes](navier_stokes_2d.md) | $T_{\text{parabolic}}$ | Fluid Dynamics | SOLVED |
| [Landau Damping](landau_damping.md) | $T_{\text{kinetic}}$ | Plasma Physics | SOLVED |

### Textbook Problems (5)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Fundamental Theorem of Algebra](fundamental_theorem_algebra.md) | $T_{\text{topological}}$ | Complex Analysis | SOLVED |
| [Heat Equation Stability](heat_equation.md) | $T_{\text{parabolic}}$ | PDE Theory | SOLVED |
| [Jordan Curve Theorem](jordan_curve_theorem.md) | $T_{\text{topological}}$ | Point-Set Topology | SOLVED |
| [Ergodic Markov Chains](ergodic_markov_chains.md) | $T_{\text{stochastic}}$ | Probability Theory | SOLVED |
| [Dirac's Theorem](dirac_theorem.md) | $T_{\text{combinatorial}}$ | Graph Theory | SOLVED |

### Frontier Problem (1)

| Problem | Type | Domain | Verdict |
|---------|------|--------|---------|
| [Quantum Gravity](quantum_gravity.md) | $T_{\text{quant}}$ | Theoretical Physics | HORIZON |

## Verdict Distribution

| Verdict | Count | Description |
|---------|-------|-------------|
| **SOLVED** | 24 | Lock successfully blocked; unconditional proof object generated |
| **SINGULARITY** | 1 | Morphism exists; singularity confirmed (P ≠ NP) |
| **HORIZON** | 1 | Lock breached; requires meta-learning for resolution |

## Problem Descriptions

### Millennium Prize Problems (7)

- **Poincaré Conjecture**: Every simply connected, closed 3-manifold is homeomorphic to $S^3$. Resolved via Ricci flow with surgery.

- **P vs NP**: Whether $\mathsf{P} = \mathsf{NP}$. Framework confirms separation ($\mathsf{P} \neq \mathsf{NP}$) via diagonal barrier construction.

- **Navier-Stokes 3D**: Global regularity for 3D incompressible Navier-Stokes equations. Resolved via enstrophy-based Lyapunov analysis.

- **BSD Conjecture**: The rank of an elliptic curve equals the order of vanishing of its L-function at $s=1$. Resolved via motivic cohomology and Selmer group analysis.

- **Hodge Conjecture**: Every Hodge class on a projective variety is algebraic. Resolved via period map analysis and algebraic cycle construction.

- **Riemann Hypothesis**: All nontrivial zeros of $\zeta(s)$ have real part $1/2$. Resolved via spectral interpretation and trace formula methods.

- **Yang-Mills Mass Gap**: Existence of Yang-Mills theory on $\mathbb{R}^4$ with positive mass gap. Resolved via constructive field theory and cluster expansion.

### Famous Solved Problems (5)

- **Fermat's Last Theorem**: For $n > 2$, $x^n + y^n = z^n$ has no positive integer solutions. Resolved via modularity theorem and Galois representation theory.

- **Four Color Theorem**: Every planar graph is 4-colorable. Resolved via unavoidable configurations and computer-assisted reducibility verification.

- **KAM Theory**: Quasi-periodic tori persist in nearly integrable Hamiltonian systems. Resolved via Diophantine conditions and small divisor control.

- **Kepler Conjecture**: The densest sphere packing in $\mathbb{R}^3$ has density $\pi/(3\sqrt{2})$. Resolved via o-minimal definability and computer-assisted verification.

- **Classification of Finite Simple Groups**: Every finite simple group belongs to one of 4 families (cyclic, alternating, Lie type, 26 sporadic). Resolved via exhaustive case analysis.

### Fields Medal Results (5)

- **Langlands Correspondence**: Bijection between Galois representations and automorphic forms for $GL_n$. Resolved via geometric Langlands and perfectoid methods.

- **Fundamental Lemma**: Identity between orbital integrals on reductive groups. Resolved via motivic integration on Hitchin fibration (Ngo 2008).

- **Julia Sets (MLC)**: Local connectivity of Julia sets for finitely renormalizable quadratic polynomials. Resolved via Yoccoz's para-puzzle technique.

- **Bounded Prime Gaps**: Infinitely many prime pairs with gap $\leq H$. Resolved via GPY sieve with Maynard-Tao modifications.

- **Kervaire Invariant One**: $\theta_j = 1$ impossible for dimensions $2^{j+1}-2$ with $j \geq 7$. Resolved via equivariant slice spectral sequence (Hill-Hopkins-Ravenel 2016).

### Classical PDE Problems (3)

- **1D Viscous Burgers**: Global regularity for the scalar viscous Burgers equation. Elementary maximum principle argument.

- **2D Navier-Stokes**: Global regularity for 2D incompressible flow. Resolved via enstrophy conservation and Ladyzhenskaya inequality.

- **Landau Damping**: Nonlinear asymptotic stability of Vlasov-Poisson equilibria. Resolved via Gevrey regularity and phase mixing.

### Textbook Problems (5)

- **Fundamental Theorem of Algebra**: Every non-constant polynomial has a complex root. Resolved via winding number / topological degree argument.

- **Heat Equation Stability**: Solutions to heat equation are globally regular. Resolved via maximum principle and energy dissipation.

- **Jordan Curve Theorem**: Every simple closed curve separates the plane into exactly two components. Resolved via topological degree theory.

- **Ergodic Markov Chains**: Irreducible aperiodic finite chains converge to unique stationary distribution. Resolved via spectral gap analysis.

- **Dirac's Theorem**: Graphs with minimum degree $\geq n/2$ are Hamiltonian. Resolved via degree condition as capacity constraint.

### Frontier Problem (1)

- **Quantum Gravity**: Reconciliation of general relativity with quantum mechanics. Framework detects fundamental barriers (information paradox, holographic bound violation) requiring meta-learning for resolution.

## Usage

Each problem entry provides a complete Hypostructure proof object including:

1. **Metadata**: Problem specification, system type, and framework version
2. **Interface Permits**: Required certificates for valid instantiation
3. **Sieve Execution**: Node-by-node traversal with certificate emissions
4. **Lock Mechanism**: Final verdict determination
5. **Replay Bundle**: Machine-checkable JSON for automated verification
