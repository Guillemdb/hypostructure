# GMT Barrier Translations: Fundamental Constraints and Comparison Principles

## Overview

This document provides comprehensive translations of barrier theorems, comparison principles, and fundamental constraints from hypostructure theory into the language of **Geometric Measure Theory (GMT)**. Barriers represent structural constraints, impossibility results, and comparison principles that govern the behavior of geometric variational problems.

---

## Part I: Classical Barriers

### 1. Comparison Principles

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Maximum Principle** | max u = max_{∂Ω} u | Harmonic functions attain max on boundary |
| **Minimum Principle** | min u = min_{∂Ω} u | Dual to maximum principle |
| **Strong Maximum Principle** | Δu ≥ 0, u(x₀) = max u ⟹ u constant | Interior maximum forces constancy |
| **Hopf Lemma** | u(x₀) = max u at ∂Ω ⟹ ∂u/∂n < 0 | Boundary maximum has inward normal derivative |
| **Supersolution Barrier** | Δv ≥ f, v|_{∂Ω} ≥ g ⟹ v ≥ u | Supersolution provides upper bound |
| **Subsolution Barrier** | Δw ≤ f, w|_{∂Ω} ≤ g ⟹ w ≤ u | Subsolution provides lower bound |

### 2. Monotonicity Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Energy Monotonicity** | d/dt E(u(t)) ≤ 0 | Energy non-increasing along flow |
| **Entropy Monotonicity** | d/dt ∫ u² log u² dμ ≤ 0 | Entropy production non-negative |
| **Perimeter Monotonicity** | d/dt P(E_t) ≤ 0 | Perimeter decreases under mean curvature flow |
| **Volume Preservation** | d/dt |E_t| = 0 | Volume constraint |
| **Monotonicity Formula (Almgren)** | Θ(r) = r^{-n} ∫_{B_r} e non-decreasing | Scale-invariant energy is monotone |
| **Huisken's Monotonicity** | d/dt ∫ (4πt)^{-n/2} e^{-|x|²/4t} dH^n ≤ 0 | Gaussian-weighted area decreasing |

### 3. Energy Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Coercivity Barrier** | E(u) ≥ c‖u‖² - C | Energy controls norm from below |
| **Lower Semicontinuity** | E(u) ≤ lim inf E(u_n) | Energy lower semicontinuous |
| **Energy Gap** | E(u) ∈ {0} ∪ [ε₀, ∞) | No solutions with energy in (0, ε₀) |
| **Energy Quantization** | E(u) ∈ {nE₀ : n ∈ ℕ} | Energy takes discrete values |
| **Sobolev Barrier** | ‖u‖_{L^{p*}} ≤ C‖∇u‖_{L^p} | Sobolev embedding constant |
| **Poincaré Barrier** | ‖u‖_{L²} ≤ λ₁^{-1/2}‖∇u‖_{L²} | First eigenvalue provides bound |

---

## Part II: Topological and Geometric Barriers

### 4. Topological Obstructions

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Genus Barrier** | g(Σ) ≥ g₀ | Minimal surface has minimum genus |
| **Homotopy Barrier** | [u] ∈ π_n(M) fixed | Homotopy class preserved |
| **Homology Barrier** | [E] ∈ H_n(X) non-trivial | Homology class prevents deformation |
| **Linking Number** | lk(γ₁, γ₂) = n | Topological linking invariant |
| **Degree Barrier** | deg(u: S^n → S^n) = d | Topological degree preserved |
| **Index Barrier** | ind(u) = ∑ sgn(det ∇²u(x_i)) | Morse index |

### 5. Isoperimetric and Capacity Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Isoperimetric Inequality** | P(E)^n ≥ nω_n|E|^{n-1} | Perimeter-volume relation |
| **Sobolev Inequality** | ‖u‖_{p*} ≤ S_n‖∇u‖_p | Best Sobolev constant |
| **Capacity Lower Bound** | cap(K) ≥ c(diam K)^{n-2} | Geometric capacity bound |
| **Cheeger Inequality** | λ₁ ≥ h²/4 | First eigenvalue ≥ Cheeger constant |
| **Faber-Krahn Inequality** | λ₁(Ω) ≥ λ₁(B)|B|/|Ω| | Ball minimizes first eigenvalue |
| **Pólya-Szegő Inequality** | cap(K*) ≤ cap(K) | Symmetrization decreases capacity |

### 6. Regularity Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Dimension Barrier** | dim(Sing u) ≤ n - 2 | Singular set has codimension ≥ 2 |
| **Hausdorff Measure Barrier** | H^{n-2}(Sing u) < ∞ | Finite (n-2)-dimensional measure |
| **ε-Regularity Barrier** | ∫_{B_1} |∇u|² < ε ⟹ u ∈ C^{k,α} | Small energy implies smoothness |
| **Morrey Decay** | ∫_{B_r} |∇u|² ≤ Cr^n | Energy decays like volume |
| **De Giorgi-Nash-Moser** | osc_{B_r} u ≤ Cr^α | Hölder continuity |
| **Schauder Barrier** | ‖u‖_{C^{k,α}} ≤ C‖f‖_{C^{k-2,α}} | Higher regularity bootstrap |

---

## Part III: Analytical Barriers

### 7. Comparison and Sub/Supersolutions

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Barrier Function Method** | Construct w: Δw ≥ f, w|_{∂Ω} ≥ g | Explicit barrier for existence |
| **Perron Method** | u = sup{v : Δv ≥ f, v ≤ ū} | Solution via supremum of subsolutions |
| **Viscosity Barrier** | Φ ∈ C²: Δ(u-Φ) ≥ 0 at max | Viscosity supersolution |
| **Kato Inequality** | Δ|u| ≥ Re(sgn(u)Δu) | Barrier for absolute value |
| **Brezis-Merle Barrier** | Δu = e^u ⟹ u bounded or blows up | Dichotomy for exponential nonlinearity |
| **Distance Function Barrier** | Δd(x, ∂Ω) ≥ -(n-1)κ | Signed distance function |

### 8. Blow-Up Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Type I Barrier** | ‖u(t)‖_{L^∞} ≤ C/(T-t)^{1/2} | Type I blow-up rate bound |
| **Type II Barrier** | ‖u(t)‖ ≫ 1/(T-t)^{1/2} | Faster than Type I |
| **Concentration Barrier** | ∫_{B_r(x₀)} |∇u|² ≥ ε₀ | Energy concentration lower bound |
| **Non-Concentration** | sup_x ∫_{B_r(x)} |∇u|² → 0 | Energy disperses |
| **Palais-Smale Barrier** | E(u_n) bounded, ‖∇E(u_n)‖ → 0 ⟹ subsequence converges | Compactness condition |
| **Mountain Pass Barrier** | c = inf_{γ} max_{t} E(γ(t)) | Saddle point energy level |

### 9. Geometric Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Mean Curvature Barrier** | H ≥ H₀ > 0 ⟹ finite extinction time | Positive mean curvature shrinks |
| **Alexandrov Barrier** | Embedded, H = const ⟹ sphere | Rigidity for constant mean curvature |
| **Simons Cone** | x²₁ + ... + x²_n = x²_{n+1} + ... + x²_{2n} | Minimal cone, singularity barrier |
| **Area Minimizing Barrier** | Area(Σ) ≤ Area(Σ') for all Σ' ∼ Σ | Minimality condition |
| **Calibration Barrier** | ω: ω|_Σ = vol_Σ, dω = 0 ⟹ Σ minimal | Calibration proves minimality |
| **Plateau Barrier** | ∂Σ = Γ, Area(Σ) = min | Least-area surface with given boundary |

---

## Part IV: Spectral and Harmonic Barriers

### 10. Eigenvalue Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Weyl Asymptotic** | λ_k ∼ (k/|Ω|)^{2/n} | Asymptotic eigenvalue growth |
| **First Eigenvalue Gap** | λ₁ > 0 (unless u = const) | Spectral gap |
| **Courant Nodal Barrier** | # nodal domains of φ_k ≤ k | Nodal domain bound |
| **Hot Spots Conjecture** | max φ₂ at boundary | Second eigenfunction |
| **Payne-Pólya-Weinberger** | λ₂/λ₁ ≤ 3 (for convex Ω ⊂ ℝ²) | Gap ratio bound |
| **Ashbaugh-Benguria** | λ₂/λ₁ ≥ (j₁'/j₁)² for ball | Sharp lower bound |

### 11. Harmonic Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Harnack Inequality** | sup_{B_r} u ≤ C inf_{B_r} u | Positive harmonic function control |
| **Liouville Barrier** | u harmonic, bounded ⟹ constant | No nontrivial bounded harmonics on ℝⁿ |
| **Gradient Estimate** | |∇u| ≤ C/r on B_{r/2} | Interior gradient bound |
| **Bôcher Theorem** | Positive harmonic in punctured ball ⟹ removable singularity | Singularity barrier |
| **Riesz Representation** | u(x) = ∫ G(x,y)Δu(y) dy | Green's function representation |
| **Harmonic Measure Barrier** | ω^x(E) = P^x(B_t hits E) | Probabilistic barrier |

---

## Part V: Curvature and Flow Barriers

### 12. Mean Curvature Flow Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Avoidance Principle** | M₁(0) ∩ M₂(0) = ∅ ⟹ M₁(t) ∩ M₂(t) = ∅ | Non-intersection preserved |
| **Convexity Preservation** | M₀ convex ⟹ M_t convex | Convexity preserved under MCF |
| **Huisken's Theorem** | M₀ convex ⟹ shrinks to round point | Asymptotic roundness |
| **Grayson's Theorem** | Embedded curve ⟹ becomes convex before extinction | Convexity achieved |
| **Neckpinch Formation** | Rotationally symmetric ⟹ neck pinch possible | Type of singularity |
| **Degenerate Neckpinch Barrier** | Non-rotationally symmetric ⟹ no neckpinch | Symmetry requirement |

### 13. Curvature Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Positive Sectional Curvature** | K_σ > 0 ⟹ no minimal submanifolds | Curvature prevents minimizers |
| **Ricci Curvature Barrier** | Ric ≥ (n-1)κ ⟹ diameter bound | Myers theorem |
| **Scalar Curvature Barrier** | R > 0 ⟹ no stable minimal hypersurfaces | Positive scalar curvature obstruction |
| **Gauss Curvature Integral** | $\int_{\Sigma} K\, dA = 2\pi\chi(\Sigma)$ | Gauss-Bonnet constraint |
| **Alexandrov-Bakelman-Pucci** | max u ≤ max_{∂Ω} u + C(diam Ω)²‖f‖_{L^n} | Maximum principle with measure of f |
| **Krylov-Safonov** | Harnack inequality for non-divergence form | Harnack for general elliptic |

---

## Part VI: Variational Barriers

### 14. Direct Method Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Coercivity Requirement** | E(u) → ∞ as ‖u‖ → ∞ | Minimizer exists |
| **Lower Semicontinuity Requirement** | u_n ⇀ u ⟹ E(u) ≤ lim inf E(u_n) | Weak limit minimizes |
| **Γ-Convergence Barrier** | Γ-lim E_n = E ⟹ min E_n → min E | Limit of minimizers |
| **Relaxation Barrier** | E̅(u) = inf{lim inf E(u_n) : u_n → u} | Relaxed functional |
| **Convexity Barrier** | E convex ⟹ unique minimizer | Convexity implies uniqueness |
| **Quasiconvexity Barrier** | E quasiconvex ⟹ lower semicontinuous | Necessary for vectorial calculus of variations |

### 15. Concentration-Compactness Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Dichotomy Barrier** | Either compact or loses mass at ∞ | P.L. Lions dichotomy |
| **Vanishing Barrier** | lim sup_n sup_y ∫_{B_R(y)} |u_n|² = 0 | Sequence vanishes |
| **Compactness at Scale** | ∃x_n: ∫_{B_R(x_n)} |u_n|² ≥ δ | Concentration at some point |
| **Splitting Barrier** | u_n = u^{(1)}_n + u^{(2)}_n + o(1) | Profile decomposition |
| **Mass Concentration** | ∫_{B_r(x_n)} ρ_n ≥ m₀ | Positive mass concentrates |
| **Profile Orthogonality** | ⟨u^{(i)}_n, u^{(j)}_n⟩ → 0 for i ≠ j | Profiles asymptotically orthogonal |

---

## Part VII: Singularity Formation Barriers

### 16. Regularity-Singularity Dichotomy

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **ε-Regularity Dichotomy** | Either E < ε (smooth) or E ≥ ε (singular) | Energy threshold for regularity |
| **Small Data Global Regularity** | ‖u₀‖ < δ ⟹ global smooth solution | Smallness implies regularity |
| **Critical Norm Barrier** | ‖u₀‖_{Ḣ^{sc}} controls regularity | Scale-critical norm |
| **Uniqueness Barrier** | Two solutions ⟹ singular initial data | Uniqueness fails at singularity |
| **Finite-Time Blowup Barrier** | E(u₀) < 0 ⟹ T_max < ∞ | Negative energy forces blowup |
| **ODE Comparison Barrier** | y' = y² ⟹ blowup in finite time | ODE provides blowup rate |

### 17. Surgery Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Topological Surgery Obstruction** | $\chi(M)$ prevents certain surgeries | Euler characteristic invariant |
| **Minimum Width** | Neck width ≥ r₀ > 0 | Cannot pinch arbitrarily thin |
| **Surgery Time Barrier** | t_surgery > t_min | Minimum time before surgery possible |
| **Standard Solution Barrier** | Post-surgery flow matches standard solution | Canonical model after surgery |
| **Excision Ball Radius** | r_excision ≤ r_max | Maximum excision radius |
| **Gluing Map Smoothness** | φ ∈ C^{k,α} required | Regularity of gluing map |

---

## Part VIII: Lower Bound Barriers

### 18. Energy Lower Bounds

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Minimal Energy** | E(u) ≥ E_min > 0 for u ≠ 0 | Energy gap from zero |
| **Action Bound** | ∫₀^T L dt ≥ A₀ | Classical action lower bound |
| **Harmonic Map Energy** | E(u) ≥ 4π|deg(u)| | Degree provides lower bound |
| **Ginzburg-Landau Energy** | E_ε(u) ≥ π|deg(u)| log(1/ε) + O(1) | Vortex energy |
| **Pohozaev Identity** | (n-2)E = boundary term | Virial identity obstruction |
| **Derrick's Theorem** | No stable solitons in n ≥ 3 | Dimensional barrier |

### 19. Distance and Comparison Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Triangle Inequality** | d(x,z) ≤ d(x,y) + d(y,z) | Metric barrier |
| **Geodesic Barrier** | d(x,y) = inf ∫₀¹ |γ'(t)| dt | Distance via length |
| **Cut Locus Barrier** | Beyond cut locus, no unique minimizing geodesic | Minimizer non-uniqueness |
| **Injectivity Radius** | inj(M) = min cut locus distance | Global geometry constraint |
| **Filling Radius Barrier** | FillRad(M) ≤ Cr | Contractibility scale |
| **Gromov-Hausdorff Distance** | d_GH(M,N) < ε | Closeness of metric spaces |

---

## Part IX: Causality and Information Barriers

### 20. Propagation Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Finite Speed of Propagation** | supp u(t) ⊆ {x : d(x, supp u₀) ≤ ct} | Information travels at finite speed |
| **Domain of Dependence** | u(x,t) depends only on {y : |x-y| ≤ ct} | Causality cone |
| **Domain of Influence** | u₀(x) influences {(y,t) : |x-y| ≤ ct} | Future light cone |
| **Non-Locality Barrier** | Fractional Laplacian has infinite speed | Non-local operator barrier |
| **Instantaneous Smoothing** | Heat equation: u(·,t) ∈ C^∞ for t > 0 | Infinite speed smoothing |
| **Backward Uniqueness** | u(T) = 0 ⟹ u ≡ 0 | Unique continuation backward |

### 21. Information Content Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Entropy Production** | S'(t) ≥ 0 | Second law of thermodynamics |
| **Fisher Information** | I(t) = ∫ |∇log u|² u dx | Information monotonicity |
| **Logarithmic Sobolev** | ∫ u² log u² ≤ 2‖∇u‖² + C | Entropy-energy relation |
| **Hypercontractivity** | ‖T_t‖_{L^p → L^q} ≤ 1 for appropriate p,q | Smoothing bound |
| **Nash Inequality** | ‖u‖²_{2+4/n} ≤ C‖∇u‖²‖u‖^{4/n}_1 | Entropy production bound |
| **Beckner Inequality** | ‖u‖_p ≤ C_p‖∇u‖_2 with sharp C_p | Optimal Sobolev constant |

---

## Part X: Existence and Non-Existence Barriers

### 22. Existence Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Fredholm Alternative** | Lu = f solvable ⟺ f ⊥ ker L* | Compatibility condition |
| **Lax-Milgram Barrier** | Coercivity + continuity ⟹ existence | Abstract existence theorem |
| **Schauder Fixed Point** | Compact convex map ⟹ fixed point | Topological existence |
| **Leray-Schauder Degree** | deg(I - K, U, 0) ≠ 0 ⟹ solution exists | Degree theory |
| **Mountain Pass Theorem** | Geometry of E ⟹ critical point exists | Variational existence |
| **Linking Theorem** | Topological linking ⟹ critical value | Minimax principle |

### 23. Non-Existence Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Pohozaev Obstruction** | Star-shaped domain ⟹ no positive solution | Virial identity |
| **Kazdan-Warner Obstruction** | R = f(u) has topological obstruction | Prescribed curvature barrier |
| **Nirenberg Problem** | K = f(x) on S² requires ∫ f dA > 0 or f < 0 | Gauss curvature prescription |
| **Yamabe Problem Obstruction** | Conformal class determines sign of scalar curvature | Conformal invariant |
| **Positive Mass Theorem** | ADM mass ≥ 0 | General relativity barrier |
| **Penrose Inequality** | M ≥ √(A/16π) | Black hole thermodynamics |

---

## Part XI: Stability and Instability Barriers

### 24. Stability Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Linear Stability** | δ²E(u) ≥ 0 | Second variation non-negative |
| **Spectral Stability** | All eigenvalues of linearization ≥ 0 | Spectrum barrier |
| **Orbital Stability** | d(u(t), orbit) ≤ Cε for d(u₀, orbit) < ε | Nearby solutions stay nearby |
| **Lyapunov Stability** | ‖u(t) - u_∞‖ → 0 | Asymptotic stability |
| **Index Barrier** | ind(u) = # negative eigenvalues | Morse index counts unstable directions |
| **Nullity Barrier** | null(u) = dim ker(δ²E) | Marginal directions |

### 25. Instability Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Payne-Sattinger Barrier** | E(u) < 0 ⟹ unstable | Negative energy implies instability |
| **Ekeland Variational Principle** | Approximate minimizer near true minimizer | Almost optimal is nearly optimal |
| **Mountain Pass Unstable** | Saddle point is unstable | Type-1 critical point |
| **Degenerate Stability** | Eigenvalue = 0 ⟹ bifurcation possible | Critical parameter |
| **Blow-Up Instability** | Perturbation leads to finite-time blowup | Loss of global existence |
| **Mode Instability** | Growing mode λ > 0 | Exponential growth |

---

## Part XII: Asymptotic Barriers

### 26. Long-Time Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Relaxation to Equilibrium** | ‖u(t) - u_∞‖ ≤ Ce^{-λt} | Exponential decay to steady state |
| **Polynomial Decay** | ‖u(t)‖ ≤ Ct^{-α} | Algebraic decay rate |
| **Slow Manifold** | u(t) ∈ M_ε for t > t₀ | Dynamics confined to manifold |
| **Metastability Barrier** | u(t) ≈ u_meta for t ∈ [t₁, t₂] | Long-lived transient |
| **Aging** | Correlation time τ(t) ~ t^α | Time-translation symmetry broken |
| **Coarsening Rate** | Characteristic length ℓ(t) ~ t^β | Domain growth law |

### 27. Scaling Barriers

| Barrier Type | GMT Translation | Description |
|--------------|-----------------|-------------|
| **Scaling Invariance** | u_λ(x,t) = λ^α u(λx, λ^β t) | Self-similar solutions |
| **Scaling Exponent** | Critical exponent p_c = 1 + 2/n | Dimension-dependent criticality |
| **Fujita Exponent** | p_c = 1 + 2/n for u_t = Δu + u^p | Blowup vs global existence |
| **Joseph-Lundgren Exponent** | p_JL(n) for supercritical problems | Higher-order critical exponent |
| **Self-Similar Blowup** | u(x,t) ∼ (T-t)^{-α} f(x/√(T-t)) | Universal blowup profile |
| **Anomalous Scaling** | ζ_p ≠ p/3 (Kolmogorov deviation) | Intermittency in turbulence |

---

## Conclusion

This comprehensive catalog of GMT barriers establishes the fundamental constraints governing geometric variational problems:

**Comparison Principles** provide upper and lower bounds via sub/supersolutions, enabling existence and uniqueness proofs.

**Monotonicity Barriers** (energy, entropy, perimeter) ensure dissipative behavior and prevent pathological dynamics.

**Topological Obstructions** (genus, homotopy, degree) impose rigid constraints that cannot be overcome by continuous deformation.

**Regularity Barriers** (ε-regularity, dimension bounds) separate smooth from singular behavior via energy thresholds.

**Curvature Constraints** (positive curvature obstructions, Ricci bounds) control geometry through analytical inequalities.

**Isoperimetric Inequalities** provide sharp relationships between geometric quantities (perimeter-volume, Sobolev constants).

**Spectral Gaps** bound eigenvalues and control long-time dynamics via the principal eigenvalue.

**Blowup Barriers** classify singularity formation as Type I (controlled) vs Type II (catastrophic).

**Causality Barriers** enforce finite vs infinite propagation speed, determining information flow.

**Existence/Non-Existence** barriers (Pohozaev, Kazdan-Warner) use integral identities to prove impossibility results.

These barriers are not obstacles to overcome but structural features that define the landscape of geometric analysis, providing rigorous bounds, comparison principles, and impossibility theorems that shape all GMT results.

---

**Cross-References:**
- [GMT Index](sketch-gmt-index.md) - Complete catalog of GMT sketches
- [GMT Interfaces](sketch-gmt-interfaces.md) - Core concept translations
- [GMT Failure Modes](sketch-gmt-failure-modes.md) - Outcome classifications
- [AI Barriers](../ai/sketch-ai-barriers.md) - Machine learning barriers
- [Complexity Barriers](../discrete/sketch-discrete-barriers.md) - Computational barriers
- [Arithmetic Barriers](../arithmetic/sketch-arithmetic-barriers.md) - Number-theoretic barriers
