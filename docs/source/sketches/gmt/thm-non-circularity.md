# THM-NonCircularity: Non-circularity — GMT Translation

## Original Statement (Hypostructure)

The permit derivation system contains no circular dependencies: no permit is used to derive itself, directly or indirectly.

## GMT Setting

**Derivation:** $\Pi \vdash K$ — permit $K$ derived from $\Pi$

**Dependency:** $K_1 \to K_2$ — $K_1$ is used in deriving $K_2$

**Non-circularity:** No $K \to^+ K$ (no permit depends on itself)

## GMT Statement

**Theorem (Non-circularity).** The permit derivation system satisfies:

1. **Direct:** No rule has form $K, \ldots \vdash K$

2. **Indirect:** No derivation path $K \to K_1 \to \cdots \to K_n \to K$

3. **Well-Founded:** There exists a well-founded ordering on permits

## Proof Sketch

### Step 1: Rule Inspection

**Claim:** No derivation rule is directly circular.

*Verification:* Examine each rule:
- $K_{D_E}^+, K_{C_\mu}^+ \vdash K_{\text{WP}}^+$ — conclusion different from premises
- $K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+ \vdash K_{\text{Conv}}^+$ — conclusion different from premises
- etc.

**Systematic Check:** For each rule $R: K_1, \ldots, K_n \vdash K_{n+1}$:
$$K_{n+1} \notin \{K_1, \ldots, K_n\}$$

### Step 2: Energy Level Ordering

**Ordering by Energy:** Define partial order on permits:
$$K_1 \prec K_2 \iff \text{energy level}(K_1) < \text{energy level}(K_2)$$

**Energy Levels:**
- Level 0: Basic certificates ($K_{D_E}$, $K_{C_\mu}$)
- Level 1: Derived certificates ($K_{\text{SC}}$, $K_{\text{LS}}$)
- Level 2: Compiled theorems ($K_{\text{WP}}$, $K_{\text{Prof}}$)
- Level 3: Meta-properties ($K_{\text{GlobExist}}$)

**Rule Preservation:** Each rule $K_1, \ldots, K_n \vdash K_{n+1}$ satisfies:
$$\max_i \text{level}(K_i) < \text{level}(K_{n+1})$$

### Step 3: Well-Founded Induction

**Definition:** A relation $\prec$ is **well-founded** if there is no infinite descending chain:
$$x_1 \succ x_2 \succ x_3 \succ \cdots$$

**Theorem:** The permit ordering $\prec$ is well-founded.

*Proof:* Energy levels are non-negative integers. Any descending chain has length $\leq$ max level.

**Reference:** For well-founded orderings: Kunen, K. (1980). *Set Theory: An Introduction to Independence Proofs*. North-Holland.

### Step 4: Transitive Closure Analysis

**Dependency Graph:** $\mathcal{G} = (V, E)$ where:
- $V = \{\text{permits}\}$
- $E = \{(K_i, K_j) : K_i \text{ used in deriving } K_j\}$

**Transitive Closure:** $E^+ = $ transitive closure of $E$

**Non-Circularity:** $(K, K) \notin E^+$ for all permits $K$.

*Proof:* The graph $\mathcal{G}$ is a DAG (THM-DAG). DAGs have no cycles in transitive closure.

### Step 5: Stratification Proof

**Stratified Permits:** Partition permits into levels:
$$\mathcal{P} = \mathcal{P}_0 \sqcup \mathcal{P}_1 \sqcup \cdots \sqcup \mathcal{P}_d$$

**Rule Stratification:** Each rule $R$ with conclusion in $\mathcal{P}_i$ has premises only in $\bigcup_{j<i} \mathcal{P}_j$.

**Consequence:** No circular dependencies possible across strata.

### Step 6: Inductive Argument

**Theorem:** Every derivation is non-circular.

*Proof by strong induction on derivation height:*

**Base Case (height 0):** Axioms have no dependencies.

**Inductive Case:** Let $D$ be derivation of height $h$ with conclusion $K$.
- $D$ uses rule $R$ with premises $K_1, \ldots, K_n$
- Each $K_i$ has derivation of height $< h$
- By induction, each $K_i$'s derivation is non-circular
- By rule stratification, $K \notin \{K_1, \ldots, K_n\}$ and $K$ doesn't appear in sub-derivations

Hence $K$'s derivation is non-circular.

### Step 7: Consistency Consequence

**Theorem:** The permit system is consistent (no $K$ and $\neg K$ derivable).

*Proof:* If $K$ and $\neg K$ were both derivable:
- Their derivations would form a cycle via contradiction rules
- But system has no circular derivations
- Contradiction

### Step 8: GMT Interpretation

**Geometric Non-Circularity:** No geometric property is used to prove itself.

**Examples:**
- Mass bounds don't depend on mass bounds
- Regularity doesn't presuppose regularity
- Convergence is derived from more basic properties

**Physical Meaning:** The permit system respects causal ordering of geometric properties.

### Step 9: Verification Algorithm

**Cycle Detection:**
```
def check_non_circularity(rules):
    G = build_dependency_graph(rules)

    # Tarjan's algorithm for SCCs
    sccs = tarjan_scc(G)

    for scc in sccs:
        if len(scc) > 1:
            return "CIRCULAR: " + str(scc)
        if self_loop_exists(scc[0], G):
            return "SELF-LOOP: " + str(scc[0])

    return "NON-CIRCULAR"
```

**Reference:** Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM J. Comput.*, 1.

### Step 10: Compilation Theorem

**Theorem (Non-circularity):**

1. **Direct:** No rule has conclusion among premises

2. **Transitive:** Dependency graph is acyclic

3. **Well-Founded:** Permit ordering is well-founded

4. **Algorithmic:** Non-circularity is verifiable in $O(|V| + |E|)$

**Constructive Content:**
- Algorithm to verify non-circularity
- Proof that stratification implies non-circularity
- Consistency as consequence

## Key GMT Inequalities Used

1. **Level Ordering:**
   $$K_i \to K_j \implies \text{level}(K_i) < \text{level}(K_j)$$

2. **Well-Foundedness:**
   $$\text{No infinite descending chain}$$

3. **DAG Property:**
   $$(K, K) \notin E^+$$

4. **Stratification:**
   $$\mathcal{P} = \bigsqcup_i \mathcal{P}_i$$

## Literature References

- Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM J. Comput.*, 1.
- Kunen, K. (1980). *Set Theory*. North-Holland.
- Apt, K. R., Blair, H. A., Walker, A. (1988). Towards a theory of declarative knowledge. *Foundations of Deductive Databases and Logic Programming*.
- Van Gelder, A. (1989). Negation as failure using tight derivations for general logic programs. *J. Logic Programming*, 6.
