---
title: "Non-Circularity - Complexity Theory Translation"
---

# THM-NON-CIRCULARITY: Proof Acyclicity

## Overview

This document provides a complete complexity-theoretic translation of the Non-Circularity theorem from the hypostructure framework. The theorem establishes that promotion rules cannot create circular dependencies in certificate derivations, corresponding to proof acyclicity in formal derivation systems. In complexity theory terms, this is the foundation for well-founded proofs, stratified inference, and termination guarantees.

**Original Theorem Reference:** {prf:ref}`thm-non-circularity`

---

## Original Theorem Statement

**Theorem (Non-Circularity Rule).** A barrier invoked because predicate $P_i$ failed **cannot** assume $P_i$ as a prerequisite. Formally:
$$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow P_i \notin \mathrm{Pre}(B)$$

**Scope of Non-Circularity:** This syntactic check ($K_i^- \notin \Gamma$) prevents direct circular dependencies. Semantic circularity (proof implicitly using an equivalent of the target conclusion) is addressed by the derivation-dependency constraint: certificate proofs must cite only lemmas of lower rank in the proof DAG. The ranking is induced by the topological sort of the Sieve, ensuring well-foundedness.

**Literature:** Well-founded semantics (Van Gelder 1991); stratification in logic programming (Apt, Bol, Pedreschi 1994).

---

## Complexity Theory Statement

**Theorem (THM-NON-CIRCULARITY, Computational Form).**
Let $\mathcal{D} = (\Sigma, \mathcal{R}, \prec)$ be a derivation system where:
- $\Sigma$ is the set of propositions (certificate types)
- $\mathcal{R} = \{R_1, \ldots, R_m\}$ is the set of inference rules
- $\prec$ is a strict partial order (dependency ordering) on $\Sigma$

The derivation system satisfies **proof acyclicity** if for every rule $R \in \mathcal{R}$ with conclusion $C$ and premises $\{P_1, \ldots, P_k\}$:
$$\forall i.\ P_i \prec C$$

That is, every premise is strictly lower in the dependency order than the conclusion.

**Formal Acyclicity Condition.** Let $\mathrm{Dep}: \Sigma \to \mathcal{P}(\Sigma)$ be the dependency function where $\mathrm{Dep}(C)$ is the set of all propositions on which $C$ depends (transitively). Then:
$$\forall C \in \Sigma.\ C \notin \mathrm{Dep}(C)$$

No proposition depends on itself.

**Corollary (Well-Founded Derivation).** In an acyclic derivation system:
1. **No circular proofs:** A proposition cannot be used to prove itself
2. **Terminating inference:** Every derivation chain has finite length
3. **Stratified evaluation:** Propositions can be computed in dependency order

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Barrier $B$ | Inference rule | Derivation step |
| Trigger $\text{Trigger}(B)$ | Rule trigger condition | Antecedent that activates rule |
| Gate $\text{Gate}_i$ NO | Failed check / negated premise | Negative condition |
| Predicate $P_i$ | Proposition to derive | Target of proof |
| Prerequisite $\mathrm{Pre}(B)$ | Rule premises | Required certificates |
| Certificate $K_i^-$ | Negation witness | Evidence of failure |
| Context $\Gamma$ | Proof context | Accumulated lemmas |
| Proof DAG | Dependency graph | Proposition ordering |
| Topological sort | Stratification | Dependency-respecting order |
| Rank in proof DAG | Stratum number | Derivation level |
| Syntactic check | Static analysis | Compile-time verification |
| Semantic circularity | Implicit dependency | Hidden circular reasoning |

---

## The Acyclicity Correspondence

### Derivation Systems as Dependency Graphs

The non-circularity constraint transforms derivation systems into well-founded structures:

**Structure Mapping:**

| Sieve Component | Derivation System Element |
|-----------------|---------------------------|
| Barrier triggering | Rule activation |
| Gate failure (NO) | Negative premise |
| Certificate derivation | Proof step |
| Context extension | Lemma accumulation |
| Re-entry target | Derived conclusion |
| Proof DAG | Dependency graph |

**Key Insight:** The non-circularity rule ensures that the dependency graph induced by inference rules is acyclic. This is the complexity-theoretic foundation for:
- **Termination:** No infinite derivation chains
- **Stratification:** Layered evaluation possible
- **Decidability:** Finite proof search

---

## Proof Sketch

### Step 1: Dependency Graph Construction

**Definition (Dependency Graph).**
The dependency graph $G_{\mathcal{D}} = (V, E)$ for derivation system $\mathcal{D}$ is:

- **Vertices:** $V = \Sigma$ (all propositions)
- **Edges:** $(P, C) \in E$ if there exists a rule $R \in \mathcal{R}$ with:
  - Conclusion $C$
  - Premises including $P$

The edge $(P, C)$ represents "$C$ depends on $P$".

**Definition (Transitive Dependency).**
The transitive dependency relation $\mathrm{Dep}^*$ is the transitive closure of the dependency graph:
$$P \in \mathrm{Dep}^*(C) \iff \text{there exists a path } P \to \cdots \to C \text{ in } G_{\mathcal{D}}$$

**Lemma (Acyclicity Characterization).** The following are equivalent:
1. The derivation system satisfies non-circularity
2. The dependency graph $G_{\mathcal{D}}$ is acyclic (a DAG)
3. There exists a strict total order $\prec$ on $\Sigma$ respecting all dependencies

**Proof.**
$(1 \Rightarrow 2)$: If non-circularity holds, no rule has $C$ in the premises when $C$ is the conclusion. Suppose for contradiction $G_{\mathcal{D}}$ has a cycle $C_1 \to C_2 \to \cdots \to C_k \to C_1$. Then there exist rules deriving $C_2$ from $C_1$, $C_3$ from $C_2$, etc. By transitivity, $C_1$ depends on itself, violating non-circularity.

$(2 \Rightarrow 3)$: Every DAG admits a topological sort, which induces a strict total order.

$(3 \Rightarrow 1)$: If $\prec$ is a strict total order and every rule respects it, then no proposition can be its own premise. $\square$

### Step 2: Stratification from Topological Order

**Definition (Stratum Assignment).**
A stratification of $\mathcal{D}$ is a function $\sigma: \Sigma \to \mathbb{N}$ such that for every rule $R$ with conclusion $C$ and premises $\{P_1, \ldots, P_k\}$:
$$\forall i.\ \sigma(P_i) < \sigma(C)$$

**Lemma (Stratification Existence).** A derivation system admits a stratification if and only if it satisfies non-circularity.

**Proof.**
$(\Rightarrow)$: If $\sigma$ is a stratification and $C \in \mathrm{Dep}(C)$, then $\sigma(C) < \sigma(C)$, contradiction.

$(\Leftarrow)$: If non-circularity holds, $G_{\mathcal{D}}$ is a DAG. Define:
$$\sigma(C) = \max\{\sigma(P) + 1 : P \in \mathrm{Dep}(C)\} \cup \{0\}$$

This is well-defined because the DAG has finite depth. $\square$

**Connection to Datalog.** This is precisely **stratified Datalog**: rules are evaluated stratum by stratum, with lower strata fully computed before higher strata. Non-circularity ensures the stratification is well-defined.

### Step 3: The Syntactic Non-Circularity Check

**Theorem (Syntactic Verification).** Non-circularity is decidable in polynomial time by the syntactic check:
$$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow K_i^- \notin \mathrm{Pre}(B)$$

**Proof.**
For each barrier $B$ in the sieve:
1. Identify the trigger condition: which gate's NO outcome activates $B$
2. Extract the trigger certificate $K_i^-$ (the witness of gate failure)
3. Check that $K_i^-$ does not appear in $\mathrm{Pre}(B)$

This is a syntactic inspection requiring $O(|B| \cdot |\mathrm{Pre}(B)|)$ time per barrier, and $O(\sum_B |B| \cdot |\mathrm{Pre}(B)|)$ total.

**Verification Algorithm:**
```
function CheckNonCircularity(Sieve):
    for each barrier B in Sieve:
        trigger_gate := GetTriggerGate(B)
        trigger_cert := K_trigger_gate^-
        for each prerequisite P in Pre(B):
            if P == trigger_cert:
                return CIRCULAR(B, trigger_cert)
    return ACYCLIC
```

**Complexity:** $O(|V| \cdot d)$ where $|V|$ is the number of barriers and $d$ is the maximum prerequisite count.

### Step 4: Semantic Non-Circularity via Ranking

**Definition (Proof DAG Ranking).**
The proof DAG is the graph of all certificate derivations during a sieve run. Each certificate $K$ is assigned a rank:
$$\mathrm{rank}(K) = \max\{\mathrm{rank}(K') + 1 : K' \in \mathrm{premises}(K)\} \cup \{0\}$$

**Theorem (Well-Founded Derivation).** In an acyclic derivation system, every certificate has finite rank, and the derivation terminates.

**Proof.**
1. **Finite rank:** If $K$ has rank $r$, then all premises have rank $< r$. By induction on rank, all certificates have finite rank.

2. **Termination:** The rank function is bounded by $|\Sigma|$ (at most one certificate per proposition at each rank). Each derivation step increases rank. Hence derivation terminates in at most $|\Sigma|^2$ steps.

3. **No circular reasoning:** If $K$ were used to derive itself, then $\mathrm{rank}(K) > \mathrm{rank}(K)$, contradiction. $\square$

### Step 5: Connecting to Barrier-Trigger Constraint

**Theorem (Barrier Non-Circularity).** The barrier constraint "$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow P_i \notin \mathrm{Pre}(B)$" ensures that barriers cannot create circular dependencies.

**Proof.**
Suppose barrier $B$ is triggered by Gate$_i$ returning NO, producing certificate $K_i^-$.

1. **Trigger semantics:** $K_i^-$ witnesses that predicate $P_i$ failed.

2. **Barrier purpose:** $B$ attempts to prove some property $Q$ that allows progress despite $P_i$ failing.

3. **Non-circularity constraint:** $B$ cannot assume $P_i$ holds (via any form of $K_i^+$) because:
   - $K_i^- \notin \Gamma$ at barrier invocation (the context contains the failure, not success)
   - Assuming $P_i$ would create circular reasoning: "If $P_i$ held, we could prove $Q$; from $Q$, we proceed as if $P_i$'s failure is handled"

4. **Dependency direction:** $Q$ depends on the barrier's premises. If those premises included $P_i$, we would need to derive $P_i$ to invoke the barrier, but the barrier was triggered precisely because $P_i$ failed.

5. **Conclusion:** The constraint ensures $\mathrm{Dep}(Q)$ does not include $P_i$, preventing circularity. $\square$

---

## Certificate Construction

The non-circularity verification produces explicit certificates:

**Acyclicity Certificate $K_{\mathrm{Acyc}}^+$:**
$$K_{\mathrm{Acyc}}^+ = \left(\sigma: \Sigma \to \mathbb{N}, \text{proof that } \sigma \text{ is a valid stratification}\right)$$

This certificate consists of:
- The stratification function $\sigma$
- For each rule $R$: verification that premises have lower strata than conclusion

**Non-Circularity Certificate $K_{\mathrm{NC}}^+$:**
$$K_{\mathrm{NC}}^+ = \left(\text{barrier\_checks}, \text{proof that each barrier passes syntactic check}\right)$$

where $\text{barrier\_checks}$ is the list of $(B, \text{Trigger}(B), \mathrm{Pre}(B), \text{status})$ for each barrier.

**Well-Foundedness Certificate $K_{\mathrm{WF}}^+$:**
$$K_{\mathrm{WF}}^+ = \left(\mathrm{rank}: \mathcal{K} \to \mathbb{N}, \text{proof that ranks are well-defined and finite}\right)$$

---

## Connections to Classical Results

### 1. Stratified Datalog (Chandra & Harel 1985, Apt, Blair & Walker 1988)

**Definition (Stratified Datalog).** A Datalog program is stratified if rules can be partitioned into strata such that:
- If rule $R$ in stratum $s$ has body literal $p$, then $p$ is defined in stratum $\leq s$
- If rule $R$ in stratum $s$ has negated body literal $\neg p$, then $p$ is defined in stratum $< s$

**Connection to THM-NON-CIRCULARITY.**
The sieve's non-circularity rule is the certificate-theoretic analog of stratification:

| Stratified Datalog | Non-Circularity |
|-------------------|-----------------|
| Predicate strata | Certificate ranks |
| Positive body literals | Positive prerequisites |
| Negated body literals | Trigger conditions (NO outcomes) |
| No recursion through negation | Trigger $\neq$ Prerequisite |
| Unique minimal model | Unique derivation result |

**Stratified Evaluation:**
```
for stratum s = 0 to max_stratum:
    evaluate all rules in stratum s
    (all premises already computed)
```

This corresponds to the sieve's layered certificate derivation.

**Theorem (Apt, Blair, Walker 1988).** Stratified Datalog programs have a unique minimal model computable in polynomial time.

**Implication for Sieve:** The non-circularity constraint ensures the certificate derivation has a unique, computable result.

### 2. Well-Founded Semantics (Van Gelder, Ross, Schlipf 1991)

**Definition (Well-Founded Semantics).** For logic programs with negation, the well-founded semantics assigns truth values via a three-valued fixed point:
- **True:** Definitely derivable
- **False:** Definitely not derivable
- **Undefined:** Depends on circular reasoning

**Connection to THM-NON-CIRCULARITY.**
The non-circularity constraint ensures the sieve operates in the **two-valued** fragment of well-founded semantics:

| Well-Founded Semantics | Non-Circularity |
|----------------------|-----------------|
| Three-valued logic | Two-valued (no undefined) |
| Well-founded model | Unique derivation result |
| Unfounded sets | Circular dependencies (prevented) |
| Alternating fixed point | Stratified iteration |

**Key Result:** By preventing circular dependencies, the sieve avoids the complexity of computing unfounded sets and the ambiguity of undefined propositions.

### 3. Termination Proofs via Well-Founded Orders (Floyd 1967, Dershowitz & Manna 1979)

**Definition (Termination via Ranking).** A program terminates if there exists a well-founded order $(W, \prec)$ and a ranking function $\rho: \text{States} \to W$ such that each transition strictly decreases $\rho$.

**Connection to THM-NON-CIRCULARITY.**
The stratification $\sigma$ is a ranking function for derivations:

| Termination Proof | Non-Circularity |
|-------------------|-----------------|
| Program states | Derivation states |
| Transitions | Inference steps |
| Ranking function | Stratification $\sigma$ |
| Strict decrease | Premises lower than conclusion |
| Well-founded order | $(\mathbb{N}, <)$ |

**Theorem (Termination Guarantee).** If the non-circularity constraint holds, every derivation terminates in finitely many steps.

**Proof.** Each inference step increases the stratum of the derived certificate. Since strata are bounded by $|\Sigma|$, termination follows. $\square$

### 4. Acyclic Dependencies in Type Systems (Pierce 2002)

**Definition (Acyclic Type Dependencies).** A type system has acyclic dependencies if no type's definition depends on itself.

**Connection to THM-NON-CIRCULARITY.**
Certificate types in the sieve satisfy the same constraint:

| Type Systems | Non-Circularity |
|--------------|-----------------|
| Type definitions | Certificate rules |
| Type dependencies | Premise dependencies |
| Acyclic type graph | Acyclic derivation graph |
| Well-founded recursion | Well-founded derivation |

**Examples:**
- **ML type inference:** No type variable can unify with a term containing itself (occurs check)
- **Coq universes:** Type$_i$ : Type$_{i+1}$ (strict hierarchy)
- **Sieve certificates:** $K_i^-$ cannot be in $\mathrm{Pre}(B)$ when $B$ is triggered by Gate$_i$ NO

### 5. Proof-Theoretic Ordinals and Cut Elimination (Gentzen 1936, Takeuti 1987)

**Definition (Proof-Theoretic Ordinal).** The proof-theoretic ordinal of a formal system measures the complexity of its termination proofs.

**Connection to THM-NON-CIRCULARITY.**
The non-circularity constraint ensures the sieve has proof-theoretic ordinal $\omega$:

| Proof Theory | Non-Circularity |
|--------------|-----------------|
| Cut elimination | Dependency elimination |
| Hauptsatz | Stratified derivation |
| Ordinal assignment | Stratum assignment |
| $\omega$ (finite ordinal) | Finite strata |

**Significance:** Systems with ordinal $\omega$ have simple termination proofs (just counting). The non-circularity constraint places the sieve in this simple class.

### 6. Dependency Parsing and Acyclic Grammars (Eisner 1996)

**Definition (Projective Dependency Parse).** A dependency parse is projective if all arcs are non-crossing. Projectivity implies acyclicity.

**Connection to THM-NON-CIRCULARITY.**
Certificate derivations form a projective structure:

| Dependency Parsing | Non-Circularity |
|-------------------|-----------------|
| Words | Certificates |
| Dependency arcs | Derivation edges |
| Non-crossing constraint | Stratification |
| Parse tree | Derivation tree |

**Algorithmic Implication:** Projective parsing is $O(n^3)$; non-projective is $O(n^5)$ or NP-hard. The non-circularity constraint keeps sieve derivation in the efficient class.

---

## Quantitative Refinements

### Stratification Depth

**Definition (Stratification Depth).** The depth of a stratification $\sigma$ is:
$$d(\sigma) = \max_{C \in \Sigma} \sigma(C)$$

**Bound:** For the sieve with $n$ certificate types:
$$d(\sigma) \leq n - 1$$

Each stratum adds at least one certificate type.

### Derivation Complexity

| Metric | Bound | Notes |
|--------|-------|-------|
| Stratification depth | $O(|\Sigma|)$ | Maximum strata |
| Verification time | $O(|V| \cdot d)$ | Syntactic check |
| Derivation length | $O(|\Sigma|^2)$ | Worst-case steps |
| Space for ranking | $O(|\Sigma|)$ | Store strata |

### Comparison with Non-Stratified Systems

| Property | Stratified (Non-Circular) | Non-Stratified |
|----------|---------------------------|----------------|
| Model existence | Unique | Multiple / None |
| Evaluation complexity | PTIME | PSPACE-complete |
| Termination | Guaranteed | May not terminate |
| Semantic ambiguity | None | Undefined values |

---

## Algorithmic Implications

### Non-Circularity Verification Algorithm

**Input:** Derivation system $\mathcal{D} = (\Sigma, \mathcal{R})$

**Output:** Boolean indicating non-circularity, plus stratification if acyclic

```
function VerifyNonCircularity(D):
    // Build dependency graph
    G := empty graph on vertices Sigma
    for each rule R in R:
        C := conclusion(R)
        for each premise P in premises(R):
            add edge (P, C) to G

    // Check for cycles via topological sort
    order, is_dag := TopologicalSort(G)
    if not is_dag:
        cycle := FindCycle(G)
        return (CIRCULAR, cycle)

    // Build stratification
    sigma := {}
    for vertex C in order:
        sigma[C] := max({sigma[P] + 1 : (P,C) in G}) or 0

    return (ACYCLIC, sigma)
```

**Complexity:** $O(|\Sigma| + |\mathcal{R}|)$ using Kahn's algorithm for topological sort.

### Stratified Derivation Algorithm

**Input:** Initial context $\Gamma_0$, rules $\mathcal{R}$, stratification $\sigma$

**Output:** Final derivation context $\Gamma_{\text{final}}$

```
function StratifiedDerivation(Gamma_0, R, sigma):
    Gamma := Gamma_0
    max_stratum := max({sigma(C) : C in conclusions(R)})

    for s := 0 to max_stratum:
        rules_s := {R in R : sigma(conclusion(R)) == s}
        repeat:
            Gamma_new := Gamma
            for each rule R in rules_s:
                if all premises of R are in Gamma:
                    Gamma_new := Gamma_new + {conclusion(R)}
        until Gamma_new == Gamma

    return Gamma
```

**Complexity:** $O(d \cdot |\mathcal{R}| \cdot |\Sigma|)$ where $d$ is stratification depth.

### Incremental Non-Circularity Maintenance

When adding a new rule $R_{\text{new}}$:

```
function AddRuleIncrementally(D, R_new, sigma):
    C := conclusion(R_new)
    max_premise_stratum := max({sigma[P] : P in premises(R_new)})

    if C in premises(R_new):
        return CIRCULAR  // Direct self-reference

    if sigma[C] <= max_premise_stratum:
        // Need to recompute strata from C upward
        RecomputeStrata(D, C, max_premise_stratum + 1)

    return ACYCLIC
```

---

## Extension: Semantic Non-Circularity

The syntactic check prevents direct circularity. Semantic circularity requires deeper analysis:

**Definition (Semantic Circularity).** A derivation exhibits semantic circularity if a proposition $P$ is used to derive a proposition $Q$ that is logically equivalent to $P$.

**Detection:**
1. **Subsumption check:** Does $Q \Rightarrow P$ hold?
2. **Equivalence check:** Are $P$ and $Q$ logically equivalent?
3. **Implicit dependency:** Does the proof of $Q$ rely on $P$'s truth?

**Resolution via Ranking.** The derivation-dependency constraint addresses semantic circularity:

> Certificate proofs must cite only lemmas of lower rank in the proof DAG.

**Theorem (Semantic Acyclicity).** If all derivations respect the ranking constraint, no semantic circularity can occur.

**Proof.** Suppose $P$ is used to derive equivalent $Q$.
1. By the ranking constraint, $\mathrm{rank}(P) < \mathrm{rank}(Q)$.
2. If $Q \Rightarrow P$, any use of $Q$ to prove something about $P$ would require $\mathrm{rank}(Q) < \mathrm{rank}(P)$.
3. Contradiction: $\mathrm{rank}(P) < \mathrm{rank}(Q) < \mathrm{rank}(P)$ is impossible.
4. Hence no semantic circularity. $\square$

---

## Summary

The THM-NON-CIRCULARITY theorem, translated to complexity theory, establishes:

1. **Proof Acyclicity:** The constraint "$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow P_i \notin \mathrm{Pre}(B)$" ensures no circular reasoning in certificate derivations.

2. **Well-Founded Derivation:** The dependency graph of the derivation system is acyclic, admitting a stratification that orders propositions by derivation level.

3. **Termination Guarantee:** Every derivation terminates because the stratification provides a ranking function that strictly increases with each inference step.

4. **Stratified Evaluation:** Certificates can be derived stratum by stratum, ensuring a unique, computable result. This corresponds to stratified Datalog evaluation.

5. **Classical Connections:** The theorem connects to:
   - Stratified Datalog (unique minimal models)
   - Well-founded semantics (two-valued fragment)
   - Floyd's termination proofs (ranking functions)
   - Type system acyclicity (occurs check)
   - Proof-theoretic ordinals ($\omega$-termination)

**The Non-Circularity Certificate:**

$$K_{\mathrm{NC}} = \begin{cases}
K_{\mathrm{Acyc}}^+ = (\sigma, \text{stratification proof}) & \text{acyclic derivation system} \\
K_{\mathrm{NC}}^+ = (\text{barrier checks}, \text{verification}) & \text{syntactic non-circularity} \\
K_{\mathrm{WF}}^+ = (\mathrm{rank}, \text{well-foundedness}) & \text{semantic non-circularity}
\end{cases}$$

This translation reveals that the hypostructure's non-circularity constraint is the dynamical-systems generalization of stratified logic programming, ensuring that the sieve's certificate derivation system has unique, computable fixed points and guaranteed termination.

---

## Literature

1. **Van Gelder, A., Ross, K. A., & Schlipf, J. S. (1991).** "The Well-Founded Semantics for General Logic Programs." *Journal of the ACM.* *Well-founded model theory.*

2. **Apt, K. R., Blair, H. A., & Walker, A. (1988).** "Towards a Theory of Declarative Knowledge." *Foundations of Deductive Databases and Logic Programming.* *Stratified semantics.*

3. **Apt, K. R., Bol, R. N., & Pedreschi, D. (1994).** "On Termination of Logic Programs." *Journal of Logic Programming.* *Termination via stratification.*

4. **Floyd, R. W. (1967).** "Assigning Meanings to Programs." *Proceedings of Symposia in Applied Mathematics.* *Ranking function termination.*

5. **Dershowitz, N. & Manna, Z. (1979).** "Proving Termination with Multiset Orderings." *Communications of the ACM.* *Well-founded orderings.*

6. **Chandra, A. K. & Harel, D. (1985).** "Horn Clause Queries and Generalizations." *Journal of Logic Programming.* *Datalog foundations.*

7. **Immerman, N. (1986).** "Relational Queries Computable in Polynomial Time." *Information and Control.* *LFP = PTIME.*

8. **Pierce, B. C. (2002).** *Types and Programming Languages.* MIT Press. *Type system acyclicity.*

9. **Gentzen, G. (1936).** "Die Widerspruchsfreiheit der reinen Zahlentheorie." *Mathematische Annalen.* *Cut elimination and ordinal analysis.*

10. **Eisner, J. (1996).** "Three New Probabilistic Models for Dependency Parsing." *COLING.* *Projective parsing complexity.*

11. **Kowalski, R. & Van Emden, M. (1976).** "The Semantics of Predicate Logic as a Programming Language." *Journal of the ACM.* *Logic programming foundations.*

12. **Przymusinski, T. C. (1988).** "On the Declarative Semantics of Deductive Databases and Logic Programs." *Foundations of Deductive Databases and Logic Programming.* *Stratified negation.*
