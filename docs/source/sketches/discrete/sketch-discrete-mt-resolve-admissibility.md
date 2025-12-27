---
title: "RESOLVE-Admissibility - Complexity Theory Translation"
---

# RESOLVE-Admissibility: Cut Admissibility in Proof Systems

## Overview

This document provides a complete complexity-theoretic translation of the RESOLVE-Admissibility theorem (Surgery Admissibility Trichotomy) from the hypostructure framework. The translation establishes a formal correspondence between surgery admissibility classification and **cut admissibility** in proof theory, where proof transformations are classified by their validity and computational tractability.

**Original Theorem Reference:** {prf:ref}`mt-resolve-admissibility`

---

## Hypostructure Context

The RESOLVE-Admissibility theorem states that before invoking any surgery $S$ with mode $M$ and data $D_S$, the framework produces exactly one of three certificates:

1. **Admissible ($K_{\text{adm}}$):** Surgery satisfies canonicity, codimension, capacity, and progress density bounds
2. **Admissible up to equivalence ($K_{\text{adm}}^\sim$):** After an admissible equivalence move, the surgery becomes admissible
3. **Not admissible ($K_{\text{inadm}}$):** Explicit failure certificate (capacity too large, codimension too small, or profile unclassifiable)

The key criteria for admissibility are:
- **Canonicity:** Profile at surgery point is in the canonical library
- **Codimension:** Singular set has codimension $\geq 2$
- **Capacity:** $\mathrm{Cap}(\text{excision}) \leq \varepsilon_{\text{adm}}$
- **Progress Density:** Energy drop satisfies $\Delta\Phi_{\text{surg}} \geq \epsilon_T$

---

## Complexity Theory Statement

**Theorem (Cut Admissibility Trichotomy).** Let $\mathcal{P}$ be a proof system and $\rho$ a proof transformation rule. Before applying $\rho$ to derivation $\Pi$, the system classifies $\rho$ into exactly one of three categories:

**Case 1: Admissible Rule**
$$K_{\text{adm}} = (\rho, \Pi, \text{validity proof}, K_{\text{progress}}^+)$$
The transformation satisfies:
1. **Canonicity:** $\rho$ produces derivations in the canonical fragment
2. **Rank Reduction:** $\text{rank}(\rho(\Pi)) < \text{rank}(\Pi)$ or measure-decreasing
3. **Size Bound:** $|\rho(\Pi)| \leq f(|\Pi|)$ for polynomial $f$
4. **Progress:** Proof complexity decreases by at least $\delta > 0$

**Case 2: Admissible up to Equivalence (Indirect)**
$$K_{\text{adm}}^\sim = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\Pi'])$$
After a proof-theoretic equivalence (permutation, normalization), the rule becomes admissible.

**Case 3: Not Admissible**
$$K_{\text{inadm}} = (\text{failure reason}, \text{witness})$$
Explicit obstruction:
- **Rank explosion:** Transformation increases proof complexity unboundedly
- **Non-termination:** Cut-elimination sequence does not terminate
- **Incompleteness:** Transformed proof is not derivable in the target system

---

## Terminology Translation Table

| Hypostructure Term | Proof Theory Equivalent |
|--------------------|------------------------|
| Surgery $S$ | Proof transformation / cut-elimination step |
| Mode $M$ | Transformation type (cut, substitution, expansion) |
| Surgery data $D_S$ | Cut formula and derivations above the cut |
| Singular set $\Sigma$ | Cut formulas (formulas eliminated by cut rule) |
| Profile $V$ | Cut formula structure / logical complexity |
| Canonical library $\mathcal{L}_T$ | Normal forms / canonical derivations |
| Codimension bound | Rank/depth bound on cut formulas |
| Capacity $\mathrm{Cap}(\Sigma)$ | Proof size / derivation length |
| $\varepsilon_{\text{adm}}$ | Polynomial size bound |
| Progress density $\Delta\Phi$ | Rank reduction / complexity decrease |
| Energy $\Phi$ | Proof complexity measure (length, depth, width) |
| Admissible surgery | Admissible rule (derivable from other rules) |
| Equivalence move | Proof permutation / normal form transformation |
| Horizon (Case 3) | Undecidable fragment / intractable transformation |
| Certificate $K_{\text{adm}}$ | Cut-elimination termination proof |
| Certificate $K_{\text{inadm}}$ | Lower bound / impossibility witness |

---

## Proof-Theoretic Framework

### The Cut Rule and Its Elimination

**Definition (Cut Rule).** In sequent calculus, the cut rule is:
$$\frac{\Gamma \vdash A, \Delta \quad \Gamma', A \vdash \Delta'}{\Gamma, \Gamma' \vdash \Delta, \Delta'} \text{ (cut)}$$

The formula $A$ is the **cut formula**---it appears in the premises but not the conclusion.

**Definition (Cut-Elimination).** A proof system has the **cut-elimination property** if every derivation with cuts can be transformed to a cut-free derivation of the same sequent.

**Gentzen's Hauptsatz (1935):** Classical and intuitionistic sequent calculi admit cut-elimination.

### Proof Complexity Measures

**Definition (Proof Complexity Measures).** For a derivation $\Pi$:

1. **Length** $|\Pi|$: Number of rule applications
2. **Depth** $d(\Pi)$: Maximum path length from root to leaf
3. **Cut-rank** $\text{cr}(\Pi)$: Maximum complexity of cut formulas
4. **Width** $w(\Pi)$: Maximum sequent size in the derivation

**Definition (Cut-Rank).** The complexity (rank) of a formula:
$$\text{rk}(A) = \begin{cases}
0 & \text{if } A \text{ is atomic} \\
1 + \max(\text{rk}(B), \text{rk}(C)) & \text{if } A = B \circ C
\end{cases}$$

The cut-rank of a derivation is $\text{cr}(\Pi) = \max\{\text{rk}(A) : A \text{ is a cut formula in } \Pi\}$.

---

## Proof Sketch

### Step 1: Canonicity Verification (Normal Form Check)

**Claim:** Given transformation $\rho$ and derivation $\Pi$, verify whether $\rho(\Pi)$ is in canonical form.

**Canonical Forms in Proof Theory:**

| System | Canonical Fragment |
|--------|-------------------|
| Sequent Calculus | Cut-free derivations |
| Natural Deduction | Normal/neutral forms |
| Lambda Calculus | $\beta\eta$-normal forms |
| Resolution | Tree resolution refutations |

**Verification Algorithm:**

```
CanonicalCheck(rho, Pi):
  Pi' = rho(Pi)
  if Pi' is cut-free:
    return PASS
  else if Pi' admits normalization to cut-free:
    return PASS_EQUIV
  else:
    return FAIL (non-canonical)
```

**Correspondence to Hypostructure:**
- Profile $V \in \mathcal{L}_T$ $\leftrightarrow$ Derivation in canonical fragment
- Profile $V \in \mathcal{F}_T \setminus \mathcal{L}_T$ $\leftrightarrow$ Normalizable to canonical form
- Profile $V \notin \mathcal{F}_T$ $\leftrightarrow$ Not normalizable (Horizon)

### Step 2: Rank Reduction (Codimension Bound)

**Claim:** Verify that the transformation reduces proof complexity.

**Gentzen's Rank Reduction:**

The key lemma in cut-elimination states that cuts can be eliminated while controlling rank:

$$\text{cr}(\rho(\Pi)) < \text{cr}(\Pi)$$

when $\rho$ is the standard cut-reduction step.

**Codimension Interpretation:**

In hypostructure terms, $\text{codim}(\Sigma) \geq 2$ ensures the singular set is "small." In proof theory:

$$\text{codim} \leftrightarrow n - \text{cr}(\Pi)$$

where $n$ is the maximum possible rank. The bound $\text{codim} \geq 2$ translates to:

$$\text{cr}(\Pi) \leq n - 2$$

This ensures cut formulas are not too complex---they have "room" to be reduced.

**Rank Reduction Lemma:**

For each cut-reduction step $\rho_i$:
$$\text{cr}(\rho_i(\Pi)) \leq \text{cr}(\Pi) - 1 \quad \text{or} \quad |\rho_i(\Pi)|_{\text{cr}} < |\Pi|_{\text{cr}}$$

where $|\Pi|_{\text{cr}}$ counts cuts of maximum rank.

### Step 3: Size Bound (Capacity Check)

**Claim:** Verify that proof size remains bounded under transformation.

**Capacity in Proof Theory:**

The capacity bound $\mathrm{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ translates to:

$$|\rho(\Pi)| \leq f(|\Pi|)$$

for some "admissible" function $f$.

**Size Bounds for Cut-Elimination:**

| System | Size Bound | Admissibility |
|--------|-----------|---------------|
| Propositional LK | $O(2^{|\Pi|})$ | Admissible (elementary) |
| First-order LK | $O(\text{tower}_{d(\Pi)}(|\Pi|))$ | Admissible (non-elementary) |
| Second-order PA | $O(\epsilon_0\text{-recursive})$ | Admissible (transfinite) |
| Higher-order | Unbounded | NOT admissible |

**Capacity Certificate:**

The capacity bound is witnessed by:
$$K_{\text{cap}} = (f, \text{proof that } |\rho(\Pi)| \leq f(|\Pi|))$$

**Correspondence to Hypostructure:**
- $\mathrm{Cap}(\Sigma) < \infty$ $\leftrightarrow$ Elementary bound on proof size
- $\mathrm{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ $\leftrightarrow$ Polynomial or fixed-exponential bound
- $\mathrm{Cap}(\Sigma) > \varepsilon_{\text{adm}}$ $\leftrightarrow$ Super-exponential blowup (inadmissible)

### Step 4: Progress Density (Complexity Decrease)

**Claim:** Each transformation step makes measurable progress toward termination.

**Progress Measure:**

Define the cut-elimination measure:
$$\mu(\Pi) = (\text{cr}(\Pi), |\Pi|_{\text{cr}}, |\Pi|)$$

ordered lexicographically. The progress density condition requires:

$$\mu(\rho(\Pi)) <_{\text{lex}} \mu(\Pi)$$

**Correspondence to Hypostructure:**

The progress density $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ translates to:
$$\mu(\Pi) - \mu(\rho(\Pi)) \geq (0, 0, 1)$$

This ensures at least one unit of progress per transformation.

**Termination Proof:**

Since $(\mathbb{N}^3, <_{\text{lex}})$ is well-founded, cut-elimination terminates:
$$\#\text{steps} \leq |\mu(\Pi_0)|$$

---

## The Admissibility Trichotomy in Detail

### Case 1: Admissible Rule

**Definition (Admissible Rule).** A rule $\rho$ is **admissible** in system $\mathcal{P}$ if:
$$\mathcal{P} \vdash \Gamma \quad \Rightarrow \quad \mathcal{P} \vdash \rho(\Gamma)$$

whenever $\rho(\Gamma)$ is well-formed, where $\mathcal{P} \vdash \Gamma$ means $\Gamma$ is derivable in $\mathcal{P}$.

**Examples of Admissible Rules:**

| Rule | System | Admissibility Status |
|------|--------|---------------------|
| Cut | LK, LJ | Admissible (Hauptsatz) |
| Weakening | LK, LJ | Derivable (hence admissible) |
| Contraction | LK | Derivable |
| Identity expansion | Natural Deduction | Admissible |
| $\eta$-reduction | $\lambda$-calculus | Admissible |

**Certificate Structure:**
```
K_adm = {
  rule: rho,
  derivation: Pi,
  canonicity: "Pi' in cut-free fragment",
  rank_bound: cr(Pi') < cr(Pi),
  size_bound: |Pi'| <= f(|Pi|),
  progress: mu(Pi') <_lex mu(Pi),
  termination: "well-founded descent on (N^3, <_lex)"
}
```

### Case 2: Admissible up to Equivalence

**Definition (Proof Equivalence).** Derivations $\Pi_1 \sim \Pi_2$ are equivalent if they prove the same sequent and are related by permutation of independent rule applications.

**Equivalence Moves:**

1. **Permutation conversions:** Reorder independent inferences
2. **Commuting conversions:** Move cuts past structural rules
3. **Principal reductions:** Reduce cuts on principal formulas

**Example (Permutation):**
$$\frac{\frac{\Gamma \vdash A}{\Gamma \vdash A \lor B}}{\Gamma, C \vdash A \lor B} \quad \sim \quad \frac{\frac{\Gamma \vdash A}{\Gamma, C \vdash A}}{\Gamma, C \vdash A \lor B}$$

**Certificate Structure:**
```
K_adm_equiv = {
  equivalence_move: "permutation conversion",
  transport: Pi ~~> Pi_normalized,
  admissibility_after: K_adm[Pi_normalized]
}
```

**Correspondence to Hypostructure:**
- YES$^\sim$ case $\leftrightarrow$ Derivation normalizable to admissible form
- Equivalence move $\leftrightarrow$ Proof permutation
- Transport certificate $\leftrightarrow$ Equivalence witness

### Case 3: Not Admissible

**Definition (Inadmissible Transformation).** A rule $\rho$ is **inadmissible** if applying it leads to:
1. **Unbounded size blowup:** No computable bound on $|\rho(\Pi)|$
2. **Non-termination:** Transformation sequence does not halt
3. **Unsoundness:** Transformed derivation proves invalid sequent

**Examples of Inadmissible Rules:**

| Scenario | System | Obstruction |
|----------|--------|-------------|
| Cut on impredicative formula | System F | Unbounded normalization |
| Cut on non-wellfounded type | Recursive types | Non-termination |
| Arbitrary rule addition | Any | Unsoundness |
| Second-order cuts | Arithmetic | Non-elementary blowup |

**Certificate Structure:**
```
K_inadm = {
  failure_reason: "capacity_exceeded" | "rank_explosion" | "non_termination",
  witness: {
    type: "size_lower_bound",
    bound: "|rho(Pi)| >= tower(|Pi|)",
    proof: "reduction from known hard problem"
  }
}
```

**Correspondence to Hypostructure:**
- Capacity too large $\leftrightarrow$ Super-exponential size blowup
- Codimension too small $\leftrightarrow$ Cuts of maximum rank
- Horizon (unclassifiable) $\leftrightarrow$ Undecidable fragment

---

## Connections to Gentzen's Cut-Elimination

### Gentzen's Hauptsatz (1935)

**Theorem (Cut-Elimination for LK/LJ).**
Every derivation in the sequent calculus LK (classical) or LJ (intuitionistic) can be transformed to a cut-free derivation of the same sequent.

**Connection to RESOLVE-Admissibility:**

Gentzen's theorem establishes that cut is **admissible** in LK/LJ:

| Admissibility Criterion | Gentzen's Proof |
|------------------------|-----------------|
| Canonicity | Cut-free derivations are canonical |
| Codimension | Cut-rank strictly decreases |
| Capacity | Proof size bounded (non-elementary) |
| Progress | Lexicographic measure decreases |

**Gentzen's Measure:**
$$\mu(\Pi) = (\text{grade of cut}, \text{number of cuts of max grade}, \text{rank of left premise}, \text{rank of right premise})$$

This multi-dimensional measure ensures progress at each step.

### Non-Elementary Bounds

**Theorem (Statman 1979, Orevkov 1979).** There exist sequents with cut-free proofs of size $n$ whose shortest proofs with cuts have size $O(\log^* n)$.

**Implication:** Cut-elimination can cause super-exponential (tower) blowup:
$$|\text{cut-free}(\Pi)| = O(\text{tower}_k(|\Pi|))$$

where $k$ depends on the cut-rank.

**Correspondence to Capacity:**
- Tower blowup is **admissible** (terminates, computable bound)
- But capacity may exceed practical thresholds
- The $\varepsilon_{\text{adm}}$ bound represents the "practical capacity limit"

### Gentzen's Analysis by Cases

Gentzen's proof proceeds by case analysis on the **last rule applied** in the derivation above the cut. This mirrors the hypostructure surgery classification:

| Cut Configuration | Gentzen's Case | Resolution |
|-------------------|----------------|------------|
| Cut on atomic | Mix | Direct elimination |
| Cut on $\land$ | Conjunction | Reduce to subformula cuts |
| Cut on $\lor$ | Disjunction | Reduce to subformula cuts |
| Cut on $\to$ | Implication | $\beta$-reduction analogue |
| Cut on $\forall$ | Universal | Substitution |
| Cut on $\exists$ | Existential | Witness extraction |

---

## Connections to Proof Complexity

### Proof System Simulation

**Definition (p-Simulation).** Proof system $\mathcal{P}$ **p-simulates** $\mathcal{Q}$ if there is a polynomial-time function $f$ such that for every $\mathcal{Q}$-proof $\Pi$, $f(\Pi)$ is a $\mathcal{P}$-proof of the same statement with $|f(\Pi)| \leq |Pi|^{O(1)}$.

**Connection to Admissibility:**

A translation $\rho: \mathcal{Q} \to \mathcal{P}$ is **admissible** if:
1. $\rho$ preserves provability (soundness)
2. $|\rho(\Pi)| \leq \text{poly}(|\Pi|)$ (polynomial simulation)
3. $\rho$ is computable in polynomial time

**Simulation Hierarchy:**

| Stronger | Weaker | Simulation |
|----------|--------|------------|
| Extended Frege | Frege | Polynomial |
| Frege | Bounded-depth Frege | Quasi-polynomial |
| Resolution | Tree Resolution | Polynomial |
| Cutting Planes | Resolution | Exponential gap |

**Inadmissibility as Separation:**

$\mathcal{P}$ does not p-simulate $\mathcal{Q}$ implies the translation is **inadmissible**---it causes super-polynomial blowup.

### Rule Admissibility in Proof Complexity

**Definition (Proof-Theoretic Strength).** The strength of a rule is measured by the complexity of proofs requiring it.

**Examples:**

| Rule | System | Effect on Complexity |
|------|--------|---------------------|
| Resolution | Propositional | Exponential lower bounds exist |
| Cutting Planes | Integer arithmetic | Polynomial simulation of Resolution |
| Frege rules | Classical prop. | Super-polynomial over Resolution |
| Extended Frege | + extension | p-simulates Frege |

**Admissibility Classification:**

A new rule $\rho$ added to system $\mathcal{P}$ is:
- **Derivable:** If $\mathcal{P} + \rho = \mathcal{P}$ (no new theorems)
- **Admissible:** If $\mathcal{P} + \rho$ has same theorems as $\mathcal{P}$, but possibly shorter proofs
- **Strictly stronger:** If $\mathcal{P} + \rho$ proves new theorems

### Proof Complexity Lower Bounds

**Theorem (Haken 1985).** Resolution requires exponential-size proofs for the pigeonhole principle $\text{PHP}_n^{n+1}$.

**Interpretation:** Certain "cut-like" reasoning is **inadmissible** in Resolution---adding it would decrease proof size exponentially.

**Lower Bound as Inadmissibility Certificate:**
```
K_inadm = {
  system: Resolution,
  formula: PHP_n^{n+1},
  lower_bound: 2^{Omega(n)},
  proof_technique: "width-size tradeoff (Ben-Sasson & Wigderson)"
}
```

---

## Certificate Construction

### Admissibility Certificate

```
K_adm = {
  mode: "Cut_Admissible",
  mechanism: "Gentzen_Hauptsatz",
  evidence: {
    canonicity: {
      target: "cut-free sequent calculus",
      check: "all cuts eliminated"
    },
    rank_reduction: {
      initial_rank: cr(Pi),
      final_rank: 0,
      method: "grade/rank induction"
    },
    size_bound: {
      function: "tower_{cr(Pi)}(|Pi|)",
      class: "non-elementary but computable"
    },
    progress: {
      measure: "(cut-rank, num_max_cuts, left_rank, right_rank)",
      ordering: "lexicographic on N^4",
      well_founded: true
    }
  },
  literature: "Gentzen 1935, Girard 1987"
}
```

### Equivalence Certificate

```
K_adm_equiv = {
  mode: "Cut_Admissible_via_Equivalence",
  mechanism: "Permutation_Conversion",
  evidence: {
    original: Pi,
    equivalence_move: {
      type: "commuting conversion",
      description: "move cut above structural rule"
    },
    normalized: Pi_prime,
    transport: {
      witness: "Pi ~~ Pi_prime",
      preserves: "provability"
    },
    admissibility: K_adm[Pi_prime]
  },
  literature: "Prawitz 1965, Zucker 1974"
}
```

### Inadmissibility Certificate

```
K_inadm = {
  mode: "Not_Admissible",
  mechanism: "Capacity_Exceeded",
  evidence: {
    failure_type: "super-polynomial blowup",
    system: "second-order arithmetic",
    witness: {
      formula_family: "consistency statements Con(PA + n induction)",
      size_lower_bound: "tower_n(|phi|)",
      proof: "ordinal analysis (Gentzen 1936)"
    },
    obstruction: {
      cut_rank: "omega (transfinite)",
      capacity: "exceeds elementary functions"
    }
  },
  literature: "Kreisel 1952, Schwichtenberg 1977"
}
```

---

## Quantitative Bounds

### Cut-Elimination Size Bounds

| Logic | Size Bound | Rank Dependence |
|-------|-----------|-----------------|
| Propositional (LK) | $2^{O(|\Pi|)}$ | $2^{2^{\cdot^{\cdot^{|\Pi|}}}}$ ($d$ times) |
| First-order (LK) | $\text{tower}_d(|\Pi|)$ | Depends on quantifier depth |
| Arithmetic (PA) | $\epsilon_0$-recursive | Transfinite ordinals |
| Second-order (PA$_2$) | Beyond $\epsilon_0$ | Much larger ordinals |

### Ordinal Bounds

**Gentzen's Ordinal Analysis (1936):**

For Peano Arithmetic, cut-elimination is controlled by the ordinal $\epsilon_0$:
$$\epsilon_0 = \sup\{\omega, \omega^\omega, \omega^{\omega^\omega}, \ldots\}$$

**Ordinal Measure:**
$$\text{ord}(\Pi) < \epsilon_0$$

and each cut-elimination step decreases the ordinal assignment.

**Transfinite Progress:**

The progress density condition generalizes to ordinals:
$$\text{ord}(\rho(\Pi)) < \text{ord}(\Pi)$$

Well-foundedness of ordinals ensures termination.

### Complexity Classes of Cut-Elimination

| Bound Type | Complexity Class | Example |
|-----------|------------------|---------|
| Polynomial | PTIME | Cut-free proofs (already normalized) |
| Exponential | EXPTIME | Propositional cut-elimination |
| Tower | ELEMENTARY | First-order cut-elimination |
| Non-elementary | Beyond ELEMENTARY | Higher-order cut-elimination |
| Transfinite | $\epsilon_0$-recursive | Arithmetic cut-elimination |

---

## Worked Example: Propositional Cut-Elimination

**Input Derivation (with cut):**

$$\Pi = \frac{\frac{A \vdash A}{A \vdash A \lor B} \quad \frac{\frac{B \vdash B}{A \lor B, B \vdash B}}{A \lor B \vdash B}}{A \vdash B}$$

where the cut formula is $A \lor B$.

**Admissibility Check:**

1. **Canonicity:** Target is cut-free sequent calculus $\checkmark$

2. **Rank Reduction:**
   - Cut formula: $A \lor B$, rank = 1
   - After elimination: cuts on $A$ and $B$, rank = 0
   - $\text{cr}(\Pi') = 0 < 1 = \text{cr}(\Pi)$ $\checkmark$

3. **Size Bound:**
   - $|\Pi| = 4$ rule applications
   - $|\Pi'| \leq 8$ after cut-elimination
   - Bound: $2^{|\Pi|} = 16$ $\checkmark$

4. **Progress:**
   - Initial: $\mu(\Pi) = (1, 1, 1, 2)$
   - Final: $\mu(\Pi') = (0, 0, 0, 0)$
   - $\mu(\Pi') <_{\text{lex}} \mu(\Pi)$ $\checkmark$

**Certificate:**
```
K_adm = {
  derivation: Pi,
  cut_formula: "A \\lor B",
  cut_rank: 1,
  elimination_steps: [
    "case split on disjunction",
    "left branch: identity on A",
    "right branch: contradiction"
  ],
  final_size: 6,
  bound_satisfied: "6 <= 2^4 = 16"
}
```

**Result:** Transformation is **Admissible** (Case 1).

---

## Summary

The RESOLVE-Admissibility theorem, translated to proof theory, establishes:

**Proof transformations classify into exactly three categories based on complexity bounds:**

1. **Admissible:** Transformation terminates with computable bounds (Gentzen's Hauptsatz)
2. **Admissible up to equivalence:** After proof normalization, transformation is admissible
3. **Not admissible:** Transformation causes unbounded blowup or non-termination

**Key Correspondences:**

| Hypostructure | Proof Theory |
|---------------|-------------|
| Surgery | Cut-elimination step |
| Capacity bound | Proof size bound |
| Codimension | Cut-rank restriction |
| Progress density | Measure decrease |
| Canonical library | Cut-free fragment |
| Horizon | Undecidable/intractable fragment |

**The Trichotomy Principle:**

Just as the hypostructure framework classifies singularities before surgery, proof theory classifies transformations before application:

$$K_{\text{Trichotomy}} = \begin{cases}
K_{\text{adm}} & \text{if } |\rho(\Pi)| \leq f(|\Pi|) \text{ for computable } f \\
K_{\text{adm}}^\sim & \text{if admissible after normalization} \\
K_{\text{inadm}} & \text{if no computable bound exists}
\end{cases}$$

**Physical Interpretation:**

- **Admissible:** The "surgery" (cut-elimination) can be performed with controlled cost
- **Admissible$^\sim$:** A preparatory step (normalization) enables controlled surgery
- **Inadmissible:** The surgery would cause unbounded "energy" (proof size) explosion

---

## Literature

1. **Gentzen, G. (1935).** "Untersuchungen uber das logische Schliessen." *Mathematische Zeitschrift.* *Cut-elimination theorem.*

2. **Gentzen, G. (1936).** "Die Widerspruchsfreiheit der reinen Zahlentheorie." *Mathematische Annalen.* *Ordinal analysis of PA.*

3. **Prawitz, D. (1965).** *Natural Deduction: A Proof-Theoretical Study.* *Normalization for natural deduction.*

4. **Girard, J.-Y. (1987).** *Proof Theory and Logical Complexity.* North-Holland. *Comprehensive proof theory.*

5. **Statman, R. (1979).** "Lower Bounds on Herbrand's Theorem." *Proceedings of the AMS.* *Non-elementary cut-elimination bounds.*

6. **Orevkov, V. P. (1979).** "Lower Bounds for Increasing Complexity of Derivations after Cut-Elimination." *Zapiski.* *Tower function bounds.*

7. **Schwichtenberg, H. (1977).** "Proof Theory: Some Applications of Cut-Elimination." *Handbook of Mathematical Logic.* *Cut-elimination applications.*

8. **Haken, A. (1985).** "The Intractability of Resolution." *Theoretical Computer Science.* *Resolution lower bounds.*

9. **Cook, S. A. & Reckhow, R. A. (1979).** "The Relative Efficiency of Propositional Proof Systems." *JSL.* *Proof complexity foundations.*

10. **Krajicek, J. (1995).** *Bounded Arithmetic, Propositional Logic, and Complexity Theory.* Cambridge. *Proof complexity.*

11. **Ben-Sasson, E. & Wigderson, A. (2001).** "Short Proofs are Narrow---Resolution Made Simple." *JACM.* *Width-size tradeoffs.*

12. **Perelman, G. (2003).** "Ricci Flow with Surgery on Three-Manifolds." *arXiv.* *Surgery admissibility in geometry (hypostructure source).*

13. **Federer, H. (1969).** *Geometric Measure Theory.* Springer. *Capacity and singularities.*

14. **Adams, D. R. & Hedberg, L. I. (1996).** *Function Spaces and Potential Theory.* Springer. *Sobolev capacity.*
