# GMT Surgery Translations: Geometric Modifications and Regularization

## Overview

This document provides comprehensive translations of surgery operations, geometric modifications, and regularization techniques from hypostructure theory into the language of **Geometric Measure Theory (GMT)**. Surgeries represent active interventions that modify geometric structures to resolve singularities, improve regularity, or achieve desired topological properties.

---

## Part I: Singularity Resolution Surgeries

### 1. Excision and Cutoff

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Ball Excision** | Remove B_δ(x₀), glue smooth cap | Cut out singularity neighborhood |
| **Neck Excision** | Cut along cylindrical neck S^{n-1} × I | Separate components at neck |
| **Singular Set Removal** | Excise ε-neighborhood of Sing(u) | Remove singular region |
| **Cutoff Function Surgery** | Multiply by φ ∈ C_c^∞ with compact support | Localize to bounded region |
| **Truncation** | u_M = min(u, M) | Cap function values |
| **Mollification Near Singularity** | u_ε = u * ρ_ε away from singular set | Smooth except at singularities |

### 2. Regularization and Smoothing

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Mollification** | u_ε = u * ρ_ε where ρ_ε = ε^{-n}ρ(·/ε) | Convolution with smooth kernel |
| **Friedrichs Mollification** | ũ_ε smooth, ũ_ε → u in H¹ | Sobolev space approximation |
| **Heat Flow Regularization** | ∂_t u = Δu for t ∈ (0,ε) | Short-time heat flow |
| **Steklov Averaging** | u_h(x) = (1/h) ∫₀^h u(x+ty) dt | Directional averaging |
| **Harmonic Extension** | Solve Δw = 0 in B_r(x₀), w = u on ∂B_r | Replace by harmonic |
| **Mean Value Replacement** | u(x) ← ∫_{B_r(x)} u dμ | Average over ball |

### 3. Topological Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Connect Sum** | M₁ # M₂ = (M₁ \ B¹) ∪ (M₂ \ B²) | Glue along boundaries |
| **Handle Attachment** | Attach D^k × D^{n-k} along S^{k-1} × D^{n-k} | Add k-handle |
| **Blow-Up** | Replace point with ℂℙⁿ⁻¹ | Resolve singularity algebraically |
| **Morse Surgery** | Modify along unstable manifold of critical point | Cancel critical points |
| **Dehn Surgery** | Remove solid torus, reglue with different framing | Topological modification |
| **Cerf Theory Surgery** | Generic family of functions | Handle slides and cancellations |

---

## Part II: Mean Curvature Flow Surgeries

### 4. Ricci Flow Surgery (Perelman)

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Neck Detection** | Identify regions with cylindrical geometry | Find S² × ℝ regions |
| **Cutoff at Neck** | Cut along S² where curvature is high | Separate components |
| **Cap Gluing** | Glue standard caps (Bryant solitons) | Smooth closure |
| **Flow Continuation** | Resume Ricci flow from modified manifold | Continue evolution |
| **δ-Neck** | Region ε-close to standard S² × I | Almost cylindrical |
| **Surgery Scale** | δ(t) controls neck width threshold | Adaptive parameter |

### 5. Mean Curvature Flow Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Neckpinch Surgery** | Cut before pinching, replace with caps | Prevent singularity |
| **Rescaling** | Zoom in: u_λ(x) = λu(λx) | Analyze blowup |
| **Matched Asymptotic Expansion** | Inner/outer solutions near singularity | Multi-scale analysis |
| **Gluing Cylinders** | Connect components via S^{n-1} × [0,1] | Bridge building |
| **Tip Smoothing** | Round off sharp tips | Regularize boundary |
| **Level Set Surgery** | Modify level sets {u = c} | Topological control |

### 6. Minimal Surface Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Bridge Position** | Replace with minimal area bridge | Reduce genus |
| **Catenoid Insertion** | Glue in catenoid neck | Classical minimal surface |
| **Schwarz Reflection** | Extend by reflection symmetry | Double covering |
| **Conformal Welding** | Glue via conformal map | Boundary identification |
| **Plateau Boundary Modification** | Change spanning curve Γ | Alter boundary condition |
| **Genus Reduction** | Surgery to decrease g(Σ) | Simplify topology |

---

## Part III: Analytical Surgeries

### 7. Energy Modification

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Energy Cutoff** | E_M(u) = min(E(u), M) | Cap energy |
| **Concentration Removal** | Remove regions with ∫ \|∇u\|² > ε₀ | Eliminate concentration |
| **Profile Decomposition** | u = ∑ᵢ u⁽ⁱ⁾ + o(1) | Split into profiles |
| **Scale Separation** | u = u_< + u_> via frequency cutoff | Low/high frequency split |
| **Renormalization** | ũ = u/‖u‖ | Normalize energy |
| **Energy Redistribution** | Move energy from concentrations to dispersed | Smooth out spikes |

### 8. Variational Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Relaxation** | Replace E by lower semicontinuous envelope E̅ | Convexify |
| **Young Measure Replacement** | Replace u by measure-valued map | Generalized solution |
| **Convex Hull** | Replace non-convex set by convex hull | Convexification |
| **Quasiconvex Envelope** | Replace by qc envelope | Vectorial calculus of variations |
| **Polyconvex Envelope** | Take polyconvex hull | Weaker relaxation |
| **Γ-Limit Approximation** | Replace E by Γ-lim E_n | Limiting functional |

### 9. Obstacle and Constraint Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Projection onto Constraint** | u ↦ Proj_K(u) | Enforce constraint |
| **Penalty Method** | E + (1/ε) dist(u,K)² | Approximate constraint |
| **Augmented Lagrangian** | E + λ·g(u) + (c/2)\|g(u)\|² | Constraint enforcement |
| **Free Boundary Regularization** | Smooth free boundary ∂{u > 0} | Regularize contact set |
| **Obstacle Removal** | Lift obstacle locally | Change constraint |
| **Capacity Potential** | Replace by capacitary potential | Harmonic extension |

---

## Part IV: Blowup Analysis and Rescaling

### 10. Rescaling Surgeries

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Parabolic Rescaling** | v(x,s) = λu(λx, λ²s) | Scale for parabolic equation |
| **Elliptic Rescaling** | v(x) = u(x₀ + rx) | Zoom in at point |
| **Self-Similar Variables** | ξ = x/√t, τ = -log t | Change to similarity coordinates |
| **Blowup Limit** | u_∞ = lim_{λ→∞} λ^α u(λx) | Extract limiting profile |
| **Blowdown** | Zoom out: v(x) = r^α u(rx) | View at large scale |
| **Multi-Scale Decomposition** | u = u_macro + u_meso + u_micro | Separate scales |

### 11. Concentration Compactness Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Extract Bubble** | u = u_bubble + u_remainder | Isolate concentration |
| **Bubble Tree Formation** | Hierarchical bubbles at different scales | Multi-bubble decomposition |
| **Neck Stretching** | Rescale neck region S^{n-1} × ℝ | Cylindrical limit |
| **Energy Quantization** | E = nE₀ + E_disp where E_disp → 0 | Discrete energy levels |
| **Splitting Lemma** | u_n = v_n + w_n with orthogonality | Decompose sequence |
| **Localization** | φ_R(x) = φ(x/R), u_R = φ_R · u | Extract local piece |

### 12. Singular Limit Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **ε → 0 Limit** | Allen-Cahn → minimal surface as ε → 0 | Phase transition limit |
| **Ginzburg-Landau Limit** | Vortices as ε → 0 | Topological defects |
| **Thin Film Limit** | 3D → 2D as thickness → 0 | Dimension reduction |
| **Homogenization** | Oscillatory → effective medium | Average microstructure |
| **Concentration-Diffusion** | Sharp interface limit | Free boundary emergence |
| **Weak-* Limit** | u_n ⇀ u in measure | Measure-valued limit |

---

## Part V: Boundary and Domain Surgery

### 13. Boundary Modification

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Boundary Smoothing** | Replace ∂Ω by smooth approximation | Regularize boundary |
| **Reflection Extension** | u(-x) = u(x) across boundary | Symmetry extension |
| **Harmonic Extension** | Solve Δu = 0 in Ω_ext, u = g on ∂Ω | Extend harmonically |
| **Sobolev Extension** | Extend H¹(Ω) to H¹(ℝⁿ) | Functional extension |
| **Change of Boundary Condition** | Dirichlet → Neumann or vice versa | BC modification |
| **Partial Boundary Excision** | Remove part of ∂Ω | Domain modification |

### 14. Domain Deformation

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Domain Perturbation** | Ω_ε = {x + εφ(x) : x ∈ Ω} | Perturb domain |
| **Hadamard Variation** | δE/δΩ = ... | Shape derivative |
| **Conformal Mapping** | Map to simpler domain via conformal map | Change coordinates |
| **Flattening Boundary** | Local coordinate change to half-space | Straighten boundary |
| **Domain Approximation** | Ω_n → Ω in Hausdorff metric | Approximate domain |
| **Hole Filling** | Add material to cavity | Topological modification |

### 15. Geometric Transformations

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Symmetrization (Steiner)** | Replace E by Steiner symmetric E* | Make spherically symmetric |
| **Polarization** | Replace by polarization in direction e | Rearrangement |
| **Decreasing Rearrangement** | u* = radial decreasing rearrangement | Radialize function |
| **Schwarz Symmetrization** | E* has same measure, spherical | Classical symmetrization |
| **Cap Symmetrization** | Partial symmetrization | Intermediate step |
| **Continuous Steiner Symmetrization** | Ė = -∇_Steiner E | Gradient flow in domain space |

---

## Part VI: Measure and Varifold Surgery

### 16. Measure Modification

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Radon-Nikodym Surgery** | μ = f·ν + μ_s (decomposition) | Split absolute continuous + singular |
| **Mass Redistribution** | Move measure from concentrated to diffuse | Spread measure |
| **Measure Approximation** | μ_n → μ weakly-* | Approximate by simpler measures |
| **Atomic Measure Removal** | μ = μ_ac + ∑ a_i δ_{x_i}, remove atoms | Eliminate point masses |
| **Hausdorff Measure Regularization** | Replace by smooth k-dimensional measure | Regularize k-rectifiable |
| **Projection to Rectifiable** | μ ↦ μ_rect | Remove non-rectifiable part |

### 17. Varifold Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Varifold Smoothing** | Approximate integral varifold by smooth | Regularization |
| **Rectification** | Replace by rectifiable varifold | Remove fractal part |
| **Integer Multiplicity** | Round multiplicities to integers | Quantize |
| **Tangent Cone Replacement** | Replace by tangent cone at singularity | Local model |
| **Stationary Varifold Approximation** | Approximate by δ-stationary | Near-minimal |
| **Support Regularization** | Smooth support of varifold | Geometric regularization |

### 18. Current Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Current Rectification** | Replace by rectifiable current | Integer multiplicities |
| **Boundary Correction** | Modify ∂T to make closed | Add boundary term |
| **Slicing** | ⟨T, φ, ·⟩ gives family of slices | Coarea surgery |
| **Normal Current** | Restrict to normal currents | Finite mass + boundary |
| **Flat Chain Approximation** | Approximate by polyhedral chains | Discrete approximation |
| **Calibration Modification** | Adjust calibration ω | Change minimal criterion |

---

## Part VII: Flow-Based Surgeries

### 19. Geometric Flow Modification

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Flow Restart** | Stop at singularity time, modify, restart | Intervention during evolution |
| **Regularized Flow** | Add higher-order term: ∂_t u = Δu - ε Δ²u | Stabilize flow |
| **Implicit Time Discretization** | u^{n+1} = u^n - τΔu^{n+1} | Numerical surgery |
| **Adaptive Time Stepping** | Reduce Δt near singularity formation | Control evolution |
| **Flow Direction Modification** | ∂_t u = -∇E₁ → ∂_t u = -∇E₂ | Change energy |
| **Damping Addition** | ∂_t u + γu = -∇E | Add friction |

### 20. Gradient Flow Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Metric Change** | Change inner product ⟨·,·⟩ | Different gradient |
| **Proximal Point** | u^{n+1} = argmin{E(v) + (1/2τ)‖v-u^n‖²} | Implicit step |
| **Retraction** | Project back to manifold after gradient step | Constraint maintenance |
| **Line Search Modification** | Optimize step size α_n | Adaptive descent |
| **Preconditioned Gradient** | ∂_t u = -M^{-1}∇E | Preconditioning |
| **Natural Gradient** | Use manifold metric for gradient | Riemannian gradient |

### 21. Discrete-to-Continuous Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Finite Element Projection** | u_h = ∑ u_i φ_i | Piecewise approximation |
| **Mesh Refinement** | h → h/2, increase resolution | Adaptive surgery |
| **Finite Difference to PDE** | Δ_h u → Δu as h → 0 | Continuum limit |
| **Spectral Truncation** | u_N = ∑_{k≤N} û_k e_k | Frequency cutoff |
| **Lattice to Continuum** | ε-lattice → continuum as ε → 0 | Homogenization |
| **Upsampling** | Interpolate to finer grid | Resolution increase |

---

## Part VIII: Functional Analytic Surgery

### 22. Function Space Modification

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Completion** | L^p → L^p closure | Add limit points |
| **Sobolev Embedding** | H^k ↪ C^j for k > j + n/2 | Higher regularity space |
| **Trace Map** | u ↦ u\|_{∂Ω} | Boundary restriction |
| **Extension Operator** | H¹(Ω) → H¹(ℝⁿ) | Extend beyond domain |
| **Restriction** | u ↦ u\|_Ω' for Ω' ⊂ Ω | Localize |
| **Density Argument** | C_c^∞ dense in H¹ | Approximate by smooth |

### 23. Duality and Adjoint Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Dual Space** | V → V* | Linear functionals |
| **Adjoint Operator** | L → L* | Transpose |
| **Bidual Embedding** | V → V** | Canonical embedding |
| **Weak-* Compactification** | Add weak-* limits | Compactify |
| **Riesz Representation** | Identify H with H* | Self-duality |
| **Polar Set** | K° = {f : f(k) ≤ 1 ∀k ∈ K} | Duality |

### 24. Spectral Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Spectral Truncation** | u ↦ ∑_{λ_k ≤ Λ} ⟨u,φ_k⟩φ_k | Low-pass filter |
| **High-Frequency Removal** | Remove modes with λ_k > Λ | Smoothing |
| **Eigenvalue Shifting** | (L - σI)^{-1} | Shift spectrum |
| **Spectral Decimation** | Keep every k-th eigenvalue | Reduce dimension |
| **Mode Projection** | Project onto span{φ_1,...,φ_N} | Finite-dimensional |
| **Spectral Clustering** | Group eigenfunctions by proximity | Mode grouping |

---

## Part IX: Topological and Homological Surgery

### 25. Homology Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Cycle Elimination** | Fill in cycle to kill homology | ∂(chain) = cycle |
| **Handle Addition** | Attach handle to change H_k | Modify Betti numbers |
| **Surgery on Circle** | S¹ surgery changes π₁ | Fundamental group modification |
| **Surgery Formula** | H_k(M') = H_k(M) ⊕ ... | Homology change |
| **Cobordism** | M₁ ~ M₂ via cobordism W | Equivalence surgery |
| **Thom Space** | Collapse complement to point | Pointed space |

### 26. Homotopy Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Sphere Attachment** | Attach D^{k+1} along S^k | Kill π_k |
| **CW Complex Building** | Attach cells iteratively | Build up space |
| **Whitehead Tower** | Kill lower homotopy groups | Simplify π_i |
| **Postnikov Tower** | Kill higher homotopy groups | Simplify from above |
| **Fibration Splitting** | p: E → B with fiber F | Decompose |
| **Loop Space** | ΩX = {γ: S¹ → X} | Iterated loops |

### 27. Covering Space Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Universal Cover** | π: X̃ → X | Simply connected lift |
| **Branched Cover** | Ramified covering | Algebraic extension |
| **Deck Transformation** | Quotient by group action X̃/G | Symmetry surgery |
| **Covering Degree** | n-fold cover | Multiplicity |
| **Pull-Back** | f*E = pullback bundle | Lift structure |
| **Quotient Map** | X → X/~ | Identify equivalences |

---

## Part X: Geometric Approximation Surgery

### 28. Polyhedral Approximation

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Triangulation** | Replace M by simplicial complex | Discrete approximation |
| **Piecewise Linear Approximation** | Replace by PL manifold | Combinatorial structure |
| **Cubical Decomposition** | Partition into cubes | Grid structure |
| **Voronoi Tessellation** | Partition based on point set | Delaunay dual |
| **Mesh Generation** | Create finite element mesh | Computational geometry |
| **Refinement** | Subdivide simplices | Increase resolution |

### 29. Smooth Approximation

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Whitney Approximation** | Lipschitz → C^k approximation | Smooth approximation |
| **Tubular Neighborhood** | ν(M) ≅ M × D^{n-k} | Normal bundle |
| **Isotopy** | Smooth deformation between embeddings | Ambient surgery |
| **Regularization Operator** | R_ε: BV → C^∞ | Smoothing map |
| **Partition of Unity** | {φ_i} subordinate to cover | Localization |
| **Extension by Smooth Function** | Extend via cutoff | Smooth gluing |

### 30. Convergence-Based Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Γ-Convergent Approximation** | E_n →^Γ E ⟹ min E_n → min E | Variational convergence |
| **Mosco Convergence** | Lim sup + lim inf conditions | Weak convergence |
| **Kuratowski Limits** | Lim sup K_n, lim inf K_n | Set convergence |
| **Hausdorff Convergence** | d_H(A_n, A) → 0 | Metric convergence |
| **Varifold Convergence** | V_n → V in varifold topology | Measure convergence |
| **Current Convergence** | T_n → T in flat norm | Homological convergence |

---

## Part XI: Constraint and Optimization Surgery

### 31. Constraint Relaxation

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Lagrange Multiplier** | Add λ·g(u) to objective | Constraint to penalty |
| **Barrier Method** | Add -log(g(u)) | Interior point |
| **Penalty Method** | Add (1/ε)g(u)² | Soft constraint |
| **Augmented Lagrangian** | L(u,λ) + (ρ/2)‖g(u)‖² | Combined method |
| **Trust Region** | Add constraint ‖δu‖ ≤ Δ | Globalization |
| **Active Set Modification** | Change active constraints | Constraint selection |

### 32. Convexification Surgery

| Surgery Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Convex Relaxation** | Replace f by convex hull co(f) | Lower convex envelope |
| **Legendre Transform** | f* (p) = sup_x px - f(x) | Dual problem |
| **Fenchel Conjugate** | Involutive transform | Self-dual |
| **Linearization** | Replace by tangent plane | Local approximation |
| **SDP Relaxation** | Lift to semidefinite program | Convex relaxation |
| **McCormick Envelope** | Bilinear → convex/concave envelope | Bound tightening |

---

## Conclusion

This comprehensive catalog of GMT surgeries establishes the complete toolkit for geometric modifications and regularizations:

**Singularity Resolution** (excision, mollification, topological surgery) removes or regularizes singular points through geometric interventions.

**Mean Curvature Flow Surgery** (Ricci flow, neckpinch, minimal surfaces) modifies evolving geometric structures to continue past singularities.

**Analytical Surgeries** (energy modification, variational surgery, obstacles) transform functionals and constraints to improve analytical properties.

**Blowup Analysis** (rescaling, concentration compactness) zooms in on singularities to extract limiting profiles and decompose solutions.

**Boundary and Domain Surgery** (boundary smoothing, domain deformation, symmetrization) modifies geometric domains and boundaries.

**Measure and Varifold Surgery** (measure modification, varifold smoothing, current rectification) operates on generalized surfaces and currents.

**Flow-Based Surgeries** (geometric flows, gradient flows, discrete-to-continuous) intervene in dynamic evolution processes.

**Functional Analytic** (function space modification, duality, spectral surgery) changes the analytic framework and function spaces.

**Topological and Homological** (homology surgery, homotopy surgery, covering spaces) modifies topological invariants through geometric operations.

**Geometric Approximation** (polyhedral, smooth approximation, convergence) replaces complex geometry with tractable approximations.

**Constraint and Optimization** (constraint relaxation, convexification) transforms optimization problems into solvable forms.

These surgeries are not ad-hoc modifications but systematic geometric transformations that preserve essential features while improving regularity, resolving singularities, or achieving desired topological properties. They form the active toolkit of GMT, complementing the passive barriers and providing constructive methods for geometric analysis.

---

**Cross-References:**
- [GMT Index](sketch-gmt-index.md) - Complete catalog of GMT sketches
- [GMT Interfaces](sketch-gmt-interfaces.md) - Core concept translations
- [GMT Barriers](sketch-gmt-barriers.md) - Fundamental constraints
- [GMT Failure Modes](sketch-gmt-failure-modes.md) - Outcome classifications
- [AI Surgeries](../ai/sketch-ai-surgeries.md) - Machine learning modifications
- [Complexity Surgeries](../discrete/sketch-discrete-surgeries.md) - Computational transformations
- [Arithmetic Surgeries](../arithmetic/sketch-arithmetic-surgeries.md) - Number-theoretic operations
