# Arithmetic Geometry Barrier Translations: Fundamental Diophantine Constraints

## Overview

This document provides comprehensive translations of barrier theorems, impossibility results, and arithmetic constraints from hypostructure theory into the language of **Arithmetic Geometry and Number Theory**. Barriers represent local-global obstructions, height bounds, finiteness results, and fundamental limitations that govern Diophantine equations and arithmetic dynamics.

---

## Part I: Local-Global Principles and Obstructions

### 1. Hasse Principle and Failures

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Hasse Principle (Success)** | X(K) ≠ ∅ ⟺ X(K_v) ≠ ∅ ∀v | Local solutions ⟹ global solution |
| **Hasse Principle (Quadratic Forms)** | Holds for quadrics over ℚ | Minkowski-Hasse theorem |
| **Selmer's Cubic** | 3x³ + 4y³ + 5z³ = 0 no rational solutions | Local everywhere, not global |
| **Counterexample Density** | Hasse principle fails for genus g ≥ 1 | Failure becomes generic |
| **Torsion Obstruction** | Finite group prevents local→global | Algebraic obstruction |
| **Height Barrier** | Height prevents rational points | Analytic obstruction |

### 2. Brauer-Manin Obstruction

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Brauer Group** | Br(X) measures obstruction | Cohomological invariant |
| **Brauer-Manin Set** | X(𝔸_K)^{Br} ⊇ X(K) | Refined approximation |
| **Azumaya Algebras** | Elements of Br(X) | Non-commutative structure |
| **Evaluation Map** | ev: X(𝔸_K) → Hom(Br(X), ℚ/ℤ) | Topological constraint |
| **Colliot-Thélène Conjecture** | Br explains all failures for rationally connected | Optimistic barrier |
| **Poonen's Counterexample** | Brauer-Manin insufficient in general | Stronger obstruction needed |

### 3. Descent Obstructions

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Selmer Group** | Sel(E/K) sits between E(K) and Ш | Cohomological object |
| **Tate-Shafarevich Group** | Ш(E/K) = ker(H¹(K,E) → ∏_v H¹(K_v,E)) | Local-global failure |
| **Ш Finiteness (Conjectural)** | \|Ш(E/K)\| < ∞ | Fundamental conjecture |
| **Cassels-Tate Pairing** | ⟨·,·⟩: Ш × Ш → ℚ/ℤ | Perfect pairing |
| **Descent via Isogeny** | φ-descent for φ: E → E' | Systematic obstruction computation |
| **n-Descent** | [n]: E → E descent | Group structure exploitation |

---

## Part II: Height Theory Barriers

### 4. Northcott-Type Finiteness

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Northcott's Theorem** | #{P ∈ ℙⁿ(K̄) : h(P) ≤ B, [K(P):K] ≤ D} < ∞ | Finite points of bounded height and degree |
| **Schanuel's Theorem** | #{P ∈ E(K̄)_tors : [K(P):K] ≤ D} < ∞ | Finite torsion of bounded degree |
| **Height Lower Bound** | h(P) ≥ c/[K(P):K]² | Positive height barrier |
| **Height Gap** | Torsion has h = 0, others h > ε | Dichotomy |
| **Lehmer's Conjecture** | h(α) ≥ c/deg(α) for non-torsion | Sharp lower bound conjecture |
| **Absolute Mahler Measure** | M(α) ≥ 1 + ε for non-cyclotomic | Mahler measure gap |

### 5. Canonical Heights on Abelian Varieties

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Néron-Tate Height** | ĥ: E(K) → ℝ≥₀ quadratic form | Canonical pairing |
| **Positive-Definite on E/E_tors** | ĥ(P) = 0 ⟺ P ∈ E_tors | Torsion characterization |
| **Height Pairing** | ⟨P,Q⟩ = ĥ(P+Q) - ĥ(P) - ĥ(Q) | Bilinear form |
| **Faltings Height** | h_Fal(A/K) measures complexity | Abelian variety height |
| **Vojta's Height Inequality** | Height comparison with divisors | Diophantine approximation |
| **Height Machine** | Functorial height theory | Systematic framework |

### 6. Equidistribution and Heights

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Bogomolov Conjecture** | Small points are torsion | Height lower bound |
| **Equidistribution** | Points of bounded height equidistribute | Probabilistic limit |
| **Szpiro's Conjecture** | Discriminant-conductor relation | ABC precursor |
| **abc Conjecture** | rad(abc)^{1+ε} > c for coprime a+b=c | Fundamental barrier |
| **Hall's Conjecture** | |x³ - y²| ≥ x^{1/2-ε} | Mordell curve approximation |
| **Pillai's Conjecture** | Gaps in perfect power sequences | Exponential Diophantine |

---

## Part III: Torsion and Rank Barriers

### 7. Torsion Subgroup Bounds

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Mazur's Theorem** | E(ℚ)_tors ∈ {ℤ/nℤ (n ≤ 10, n ≠ 11), ℤ/2ℤ × ℤ/2nℤ (n ≤ 4)} | Complete classification over ℚ |
| **Merel's Theorem** | \|E(K)_tors\| ≤ B(d) for [K:ℚ] = d | Uniform boundedness |
| **Ogg's Conjecture** | Modular curve X₀(N) rational ⟺ N ∈ {1,2,...,10,12,13,16,18,25} | Torsion-level connection |
| **Parent's Theorem** | Explicit B(d) for small d | Uniform bound computation |
| **Rational Torsion** | 16 possibilities for E(ℚ)_tors | Finite list |
| **Torsion Growth in Towers** | Bounded torsion in extensions | Growth limitation |

### 8. Mordell-Weil Rank Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Mordell-Weil Theorem** | E(K) finitely generated | Fundamental finiteness |
| **Rank Unknown** | No algorithm to compute rank(E/K) | Computational barrier |
| **BSD Conjecture** | rank(E/K) = ord_{s=1} L(E,s) | Analytic rank = algebraic rank |
| **Parity Conjecture** | rank ≡ ord_{s=1} (mod 2) | Known for many cases |
| **Rank Records** | Current record rank > 28 | Explicit high-rank curves |
| **Average Rank Conjecture** | Avg rank = 1/2 | Statistical prediction |

### 9. Selmer and Shafarevich-Tate Bounds

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Selmer Rank** | dim Sel^{(φ)}(E/K) ≥ rank(E/K) | Upper bound on rank |
| **Ш Finiteness** | \|Ш\| < ∞ (conjectural for all E/K) | Fundamental conjecture |
| **Ш is Square** | \|Ш\| = □ (Cassels-Tate) | Perfect pairing consequence |
| **n-Selmer Parity** | (-1)^{dim Sel^{(n)}} = ... | Parity formula |
| **Visibility** | Ш → J[n] for auxiliary Jacobian J | Geometric manifestation |
| **Kolyvagin's Result** | L(E,1) ≠ 0 ⟹ Ш finite | Analytic condition |

---

## Part IV: Finiteness Theorems as Barriers

### 10. Faltings' Theorem (Mordell Conjecture)

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Mordell's Conjecture** | g(C) ≥ 2 ⟹ #C(K) < ∞ | Finiteness for high genus |
| **Faltings' Proof** | Via heights on moduli space | Proof technique |
| **Effectivity Barrier** | No effective bound on #C(K) | Computational limitation |
| **Bombieri's Effective Result** | Effective for hyperelliptic | Special case |
| **Chabauty's Method** | rank < g ⟹ effective bound | Rank condition |
| **Coleman's Bound** | p-adic integration | Refined Chabauty |

### 11. Siegel's Theorem and Integral Points

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Siegel's Theorem** | #C(O_K) < ∞ for g ≥ 1 | Finite integral points |
| **S-Integral Points** | #{P ∈ C(K) : v(P) ≥ 0 ∀v ∉ S} < ∞ | Generalization |
| **Baker's Effective Bound** | Explicit bound via linear forms in logs | Effectivity |
| **Roth's Theorem** | \|α - p/q\| < 1/q^{2+ε} finitely often | Diophantine approximation |
| **Thue Equation** | F(x,y) = m has finitely many solutions | Homogeneous form |
| **Hyperelliptic Integral Points** | Effective bounds available | Explicit computation possible |

### 12. Isogeny and Endomorphism Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Isogeny Theorem (Faltings)** | #{E' : E ∼_{K̄} E'} < ∞ | Finite isogeny classes |
| **Tate's Isogeny Theorem** | E₁[ℓ^∞] ≅ E₂[ℓ^∞] over K_v ⟹ E₁ ∼ E₂ | Local determines global |
| **Endomorphism Ring** | End(E/K̄) ∈ {ℤ, order in imaginary quadratic field} | Structure theorem |
| **CM vs Non-CM** | CM curves have extra structure | Dichotomy |
| **Frobenius Endomorphism** | π: x ↦ x^p for E/𝔽_p | Finite field structure |
| **Isogeny Graph** | ℓ-isogeny graph structure | Combinatorial object |

---

## Part V: Class Field Theory Barriers

### 13. Abelian Extensions Only

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **CFT Limitation** | Describes only abelian extensions | Non-abelian barrier |
| **Kronecker-Weber** | Every abelian ext of ℚ ⊆ ℚ(ζ_n) | Cyclotomic fields suffice |
| **Artin Reciprocity** | Gal(K^{ab}/K) ≅ C_K/N_{L/K}C_L | Explicit isomorphism |
| **Local CFT** | For local fields K_v | Local description |
| **Global CFT** | For number fields K | Global description |
| **Non-Abelian Langlands** | Conjectural generalization | Beyond CFT |

### 14. Ramification Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Ramification Degree** | e_p = [D_p : I_p] measures ramification | Inertia subgroup index |
| **Conductor-Discriminant** | f_p · e_p divides discriminant exponent | Fundamental relation |
| **Wild Ramification** | p \| e_p ⟹ wild | Harder to control |
| **Tame vs Wild** | Tame easier to understand | Dichotomy |
| **Abhyankar's Lemma** | Ramification in covers | Geometric constraint |
| **Hurwitz Genus Formula** | 2g_X - 2 = deg(π)(2g_Y - 2) + deg(R) | Ramification affects genus |

### 15. Adelic Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Diagonal Embedding** | K ↪ 𝔸_K not closed | Topological barrier |
| **Weak Approximation** | K dense in ∏_S K_v | Local approximation |
| **Strong Approximation Failure** | Not always possible | Obstruction exists |
| **Brauer-Manin Set** | Closure of X(K) in X(𝔸_K) | Topological closure |
| **Adelic Points Empty** | X(𝔸_K) = ∅ ⟹ X(K) = ∅ | Trivial obstruction |
| **Adelic Points Non-Empty** | X(𝔸_K) ≠ ∅ ≠> X(K) ≠ ∅ | Non-trivial obstruction |

---

## Part VI: Good and Bad Reduction

### 16. Reduction Types (Elliptic Curves)

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Good Reduction** | Ẽ_p smooth elliptic curve | Best case |
| **Multiplicative Reduction** | Ẽ_p ≅ 𝔾_m / q^ℤ (Tate curve) | Node singularity |
| **Additive Reduction** | Ẽ_p has cusp | Worst case |
| **Potential Good Reduction** | Good reduction after extension | Obstruction measure |
| **Conductor Exponent** | f_p = 0 (good), ≥ 1 (bad) | Quantitative measure |
| **Kodaira-Néron Classification** | Types I_n, I_n*, II, III, IV, II*, III*, IV* | Complete classification |

### 17. Néron Models

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Néron Model Existence** | Smooth group scheme with Néron property | Universal smooth model |
| **Component Group** | Φ_p = Ẽ_p^sm / Ẽ_p⁰ | Discrete reduction info |
| **Tamagawa Number** | c_p = #Φ_p(𝔽_p) | Local contribution |
| **Global Tamagawa Product** | ∏_p c_p in BSD formula | Global invariant |
| **Néron-Ogg-Shafarevich** | Good reduction ⟺ unramified Galois rep | Representation-theoretic criterion |
| **Semistable Reduction** | Achieved after finite extension | Minimal model |

### 18. Minimal Discriminant and Conductor

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Minimal Weierstrass Equation** | Smallest v(Δ) for all Weierstrass forms | Canonical form |
| **Minimal Discriminant** | Δ_min = ∏_p p^{f_p} | Global invariant |
| **Conductor** | N_E = ∏_p p^{f_p} | Product of local conductors |
| **Szpiro's Ratio** | σ(E) = log \|Δ_min\| / log N_E | Conjectural bound σ < 6+ε |
| **Ogg's Formula** | f_p = ord_p(Δ_min) - ord_p(j) (good red) | Local-global relation |
| **abc ⟹ Szpiro** | Masser-Oesterlé observation | Conjecture equivalence |

---

## Part VII: Galois Representations and Modular Forms

### 19. ℓ-adic Representations

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Tate Module** | T_ℓ(E) = lim ← E[ℓ^n] | ℓ-adic object |
| **Galois Representation** | ρ_{E,ℓ}: Gal(K̄/K) → GL₂(ℤ_ℓ) | 2-dimensional representation |
| **Image of Galois** | Im(ρ_{E,ℓ}) ⊆ GL₂(ℤ_ℓ) | Galois action structure |
| **Serre's Conjecture (Proven)** | Non-CM curves have open image | Maximality |
| **Torsion Constraint** | Torsion structure constrains representation | Finite subgroup |
| **Ramification** | Representation unramified at good reduction primes | Local behavior |

### 20. Modularity and L-Functions

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Modularity Theorem** | Every elliptic curve over ℚ is modular | Taniyama-Shimura-Weil |
| **Fermat's Last Theorem** | Follows from modularity | Famous consequence |
| **L-Function** | L(E,s) = ∏_p L_p(E,s) | Global analytic object |
| **Functional Equation** | Λ(s) = w·Λ(2-s) | Symmetry |
| **BSD Conjecture** | ord_{s=1} L(E,s) = rank(E/K) | Central conjecture |
| **Birch-Swinnerton-Dyer Formula** | lim_{s→1} L(E,s)/(s-1)^r = Ω·Reg·#Ш/∏ c_p·#E_tors² | Full conjecture |

### 21. Weil Conjectures (Resolved)

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Rationality** | Z(X/𝔽_q, t) ∈ ℚ(t) | Zeta function rational |
| **Functional Equation** | $Z(1/(qt)) = \pm q^{\chi/2} t^{\chi} Z(t)$ | Self-duality |
| **Riemann Hypothesis** | \|α_i\| = q^{i/2} | Zero locations |
| **Betti Numbers** | deg P_i = b_i (ℓ-adic cohomology) | Topological interpretation |
| **Deligne's Theorem** | Proof of Riemann hypothesis part | Major result |
| **Étale Cohomology** | Tool for Weil conjectures | Technical machinery |

---

## Part VIII: Effectivity and Decidability Barriers

### 22. Effective Computability Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Faltings Non-Effective** | Mordell finiteness without bound | Computational barrier |
| **Rank Computation** | No known algorithm for arbitrary E/K | Undecidability barrier |
| **BSD Computational** | Cannot verify BSD for general curve | Verification barrier |
| **Baker's Method** | Linear forms in logs | Effective tool |
| **Chabauty-Coleman** | Effective when rank < genus | Conditional effectivity |
| **Height Bounds** | Silverman's bounds sometimes effective | Partial effectivity |

### 23. Undecidability Results (MRDP)

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Hilbert's 10th Problem** | Diophantine equations undecidable over ℤ | Fundamental barrier |
| **MRDP Theorem** | Recursively enumerable ⟺ Diophantine | Characterization |
| **Pell Equation Encoding** | Universal Diophantine equation | Reduction technique |
| **Matiyasevich's Contribution** | Fibonacci encoding | Key innovation |
| **Extensions to ℚ** | Open problem | Major question |
| **Definability** | ℤ definable in ℚ ⟹ H10 over ℚ undecidable | Conditional result |

### 24. Analytic Number Theory Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Generalized Riemann Hypothesis** | $L(s,\chi)$ zeros on Re(s) = 1/2 | Prime distribution |
| **Twin Prime Conjecture** | Infinitely many p with p+2 prime | Prime gaps |
| **Goldbach Conjecture** | Every even n > 2 is sum of two primes | Additive number theory |
| **Collatz Conjecture** | 3n+1 problem | Dynamics of integers |
| **Perfect Number** | Odd perfect number existence | Multiplicative structure |
| **Catalan-Mihăilescu** | Only x^p - y^q = 1 with p,q > 1 is 3² - 2³ = 1 | Exponential Diophantine |

---

## Part IX: Approximation and Irrationality

### 25. Diophantine Approximation Barriers

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Liouville's Theorem** | \|α - p/q\| ≥ c(α)/q^d for algebraic α deg d | Basic barrier |
| **Roth's Theorem** | \|α - p/q\| < 1/q^{2+ε} finitely often | Optimal exponent |
| **Schmidt's Subspace Theorem** | Generalization to higher dimensions | Multi-dimensional |
| **Thue-Siegel-Roth** | Historical development | Progressive improvement |
| **Effective Roth** | No effective constant c(α,ε) | Effectivity barrier |
| **Baker's Theorem** | Linear forms in logarithms | Effective tool |

### 26. Irrationality and Transcendence

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **π Irrational** | π ∉ ℚ | Classical result |
| **e Irrational** | e ∉ ℚ | Euler's result |
| **π Transcendental** | π transcendental over ℚ | Lindemann 1882 |
| **e Transcendental** | e transcendental over ℚ | Hermite 1873 |
| **Lindemann-Weierstrass** | e^{α₁},...,e^{αₙ} algebraically independent | General result |
| **Schanuel's Conjecture** | tr.deg. ℚ(α₁,...,e^{α₁},...) ≥ n | Conjectural barrier |

### 27. Special Values and Periods

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **ζ(3) Irrational** | Apéry's result | Zeta value |
| **ζ(2n+1) Irrationality** | Open for n ≥ 2 | Major problem |
| **Multiple Zeta Values** | ζ(s₁,...,s_k) | Generalization |
| **Euler's Formula** | ζ(2n) ∈ ℚ·π^{2n} | Rational multiple of π^{2n} |
| **Periods** | Integrals of algebraic forms | Kontsevich-Zagier |
| **Period Conjecture** | Algebraic relations among periods | Structural conjecture |

---

## Part X: Geometric and Topological Barriers

### 28. Rational Points and Geometry

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Genus Dichotomy** | g = 0: potential density, g = 1: group structure, g ≥ 2: finite | Fundamental division |
| **Rational Curves** | g = 0 with K-point ⟹ ℙ¹_K | Parametrization |
| **Cubic Curves** | g = 1 with K-point ⟹ elliptic curve | Group law |
| **Higher Genus** | g ≥ 2 ⟹ finitely many K-points | Faltings |
| **Rational Surfaces** | Potentially dense rational points | Dimension matters |
| **Unirationality ≠ Rationality** | Cubic threefolds counterexample | Birational geometry |

### 29. Abelian Varieties and Jacobians

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Poincaré Complete Reducibility** | Every abelian subvariety has complement | Structure theorem |
| **Mordell-Weil for Abelian Varieties** | A(K) finitely generated | Generalization |
| **Jacobian Universal Property** | Jac(C) represents Pic⁰(C) | Functorial characterization |
| **Torelli Theorem** | Jac(C₁) ≅ Jac(C₂) ⟹ C₁ ≅ C₂ (g ≥ 2) | Curve reconstruction |
| **Schottky Problem** | Characterize Jacobians among PPAVs | Open for general g |
| **Height Pairing on Abelian Varieties** | Néron-Tate generalization | Multi-dimensional heights |

### 30. Algebraic Cycles and Motives

| Barrier Type | Arithmetic Translation | Description |
|--------------|------------------------|-------------|
| **Hodge Conjecture** | Algebraic cycles = Hodge classes | Major open problem |
| **Tate Conjecture** | ℓ-adic cycles = Galois-invariant | ℓ-adic analogue |
| **Beilinson Conjectures** | Special values of L-functions | Motivic framework |
| **Bloch-Kato Conjecture** | Tamagawa number conjecture | Arithmetic geometry synthesis |
| **Standard Conjectures** | Weil's conjectures on algebraic cycles | Foundational |
| **Motives** | Universal cohomology theory | Grothendieck's vision |

---

## Conclusion

This comprehensive catalog of arithmetic-geometric barriers establishes the fundamental constraints governing Diophantine equations and arithmetic varieties:

**Local-Global Principles** (Hasse, Brauer-Manin) reveal that local solvability does not always imply global solvability, with cohomological obstructions.

**Height Theory** (Northcott, Néron-Tate, Faltings) provides finiteness results via metric bounds on algebraic points.

**Torsion Bounds** (Mazur, Merel) completely classify or bound torsion subgroups, with explicit finite lists.

**Mordell-Weil Theorem** establishes finite generation but leaves rank computation algorithmically undecidable.

**Finiteness Theorems** (Faltings, Siegel) prove that high-genus curves and integral points are finite, though often non-effectively.

**Class Field Theory** describes all abelian extensions but cannot handle non-abelian phenomena (Langlands program).

**Reduction Theory** (Néron, Kodaira) classifies bad reduction types and connects local behavior to global arithmetic.

**Galois Representations** (ℓ-adic, modularity) encode arithmetic in representation theory, leading to profound connections (Fermat's Last Theorem).

**BSD Conjecture** conjecturally relates analytic (L-function) and algebraic (rank, Ш) invariants.

**Undecidability** (MRDP, Hilbert's 10th) shows that Diophantine equations are algorithmically undecidable over ℤ.

**Diophantine Approximation** (Roth, Baker) provides impossibility results for approximating algebraic numbers.

**Geometry-Arithmetic Interplay** (genus dichotomy, rational points) shows how topology constrains arithmetic.

These barriers are not obstacles but structural features that define arithmetic geometry, providing finiteness results, impossibility theorems, and deep conjectural relationships that guide all research in the field.

---

**Cross-References:**
- [Arithmetic Index](sketch-arithmetic-index.md) - Complete catalog of arithmetic sketches
- [Arithmetic Interfaces](sketch-arithmetic-interfaces.md) - Core concept translations
- [Arithmetic Failure Modes](sketch-arithmetic-failure-modes.md) - Outcome classifications
- [GMT Barriers](../gmt/sketch-gmt-barriers.md) - Geometric analysis barriers
- [AI Barriers](../ai/sketch-ai-barriers.md) - Machine learning barriers
- [Complexity Barriers](../discrete/sketch-discrete-barriers.md) - Computational barriers
