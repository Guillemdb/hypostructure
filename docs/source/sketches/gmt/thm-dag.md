# THM-DAG: DAG Structure — GMT Translation

## Original Statement (Hypostructure)

The dependency structure of permits and tactics forms a directed acyclic graph (DAG), ensuring no circular dependencies and enabling topological ordering of resolution steps.

## GMT Setting

**Nodes:** Permits $K^+$, tactics $E$, and resolution steps

**Edges:** Dependencies $K_1 \to K_2$ (prerequisite relation)

**DAG:** No directed cycles in the dependency graph

## GMT Statement

**Theorem (DAG Structure).** The permit dependency graph $\mathcal{G} = (V, E)$ is a DAG:

1. **Acyclicity:** There is no sequence $K_0 \to K_1 \to \cdots \to K_n \to K_0$

2. **Topological Order:** Permits can be linearly ordered respecting dependencies

3. **Finite Depth:** The maximum dependency chain has bounded length

## Proof Sketch

### Step 1: Dependency Graph Construction

**Nodes:** The vertex set consists of:
- Soft certificates: $K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+, \ldots$
- Tactics: $E_1, \ldots, E_{10}$
- Compiled theorems: WP, ProfDec, KM-Rigidity, etc.

**Edges:** $K_1 \to K_2$ if $K_2$ requires $K_1$ as prerequisite.

**Examples:**
- $K_{C_\mu}^+ \to K_{\text{ProfDec}}$ (compactness needed for profile decomposition)
- $K_{D_E}^+ \to K_{\text{WP}}$ (dissipation needed for well-posedness)
- $K_{\text{LS}_\sigma}^+ \to K_{\text{Convergence}}$ (Łojasiewicz needed for convergence)

### Step 2: Energy-Based Ordering

**Observation:** Dependencies follow energy hierarchy:
1. $K_{D_E}^+$ (dissipation) — level 0
2. $K_{C_\mu}^+$ (compactness) — level 1
3. $K_{\text{SC}_\lambda}^+$ (scale coherence) — level 2
4. $K_{\text{LS}_\sigma}^+$ (stiffness) — level 3

**Order Preservation:** Higher levels depend on lower levels, not vice versa.

### Step 3: Acyclicity Proof

**Theorem:** $\mathcal{G}$ contains no directed cycles.

*Proof by Contradiction:* Suppose $K_0 \to K_1 \to \cdots \to K_n \to K_0$ is a cycle.

**Energy Argument:** Each edge $K_i \to K_{i+1}$ represents:
- $K_{i+1}$ controls behavior at scale $\lambda_{i+1}$
- $K_i$ controls behavior at scale $\lambda_i \leq \lambda_{i+1}$

Scales are monotone along dependencies. A cycle would require $\lambda_0 < \lambda_1 < \cdots < \lambda_n < \lambda_0$, contradiction.

**Logical Argument:** Circular dependency would mean:
$$K_0 \text{ needs } K_0 \text{ to be verified}$$

This is logically impossible for well-founded verification.

### Step 4: Topological Ordering

**Kahn's Algorithm (1962):**
```
L = empty list (topological order)
S = set of nodes with no incoming edges

while S is non-empty:
    n = remove a node from S
    add n to L
    for each edge (n, m):
        remove edge (n, m)
        if m has no other incoming edges:
            add m to S

if graph has edges remaining:
    error: cycle exists
else:
    return L
```

**Reference:** Kahn, A. B. (1962). Topological sorting of large networks. *Comm. ACM*, 5, 558-562.

**Application:** Order permits for sequential verification.

### Step 5: Depth Bound

**Definition:** The **depth** of node $K$ is:
$$\text{depth}(K) := \max\{\text{depth}(K') + 1 : K' \to K\}$$

with $\text{depth}(K) = 0$ if $K$ has no prerequisites.

**Theorem:** $\text{depth}(K) \leq D$ for universal constant $D$.

*Proof:* The permit types are finite (soft certificates + tactics + compiled theorems). Each dependency chain has length $\leq D = |\text{permit types}|$.

### Step 6: GMT Dependency Examples

**Example 1: Well-Posedness Chain**
```
K_{D_E}^+ → K_{C_μ}^+ → K_{SC_λ}^+ → K_{LS_σ}^+ → WP
```

**Example 2: Profile Decomposition Chain**
```
K_{C_μ}^+ → Compactness → Profile Extraction → ProfDec
```

**Example 3: Surgery Chain**
```
K_{SC_λ}^+ → Tangent Cone → Profile Class → Surgery Admissibility → Surgery
```

### Step 7: Parallel Resolution

**Independent Permits:** Permits at the same level can be verified in parallel.

**Level Sets:**
$$L_i := \{K : \text{depth}(K) = i\}$$

**Parallel Verification:** Verify all permits in $L_i$ before proceeding to $L_{i+1}$.

**Speedup:** Parallel time = $D$ (depth), vs. sequential time = $|V|$ (node count).

### Step 8: Incremental Updates

**Dynamic DAG:** When new permits are added:
1. Insert new node $K_{\text{new}}$
2. Add dependency edges
3. Re-verify topological order
4. Update affected downstream nodes

**Correctness:** If original graph is DAG and new edges don't create cycles, updated graph is DAG.

### Step 9: Verification Algorithm

**DAG Verification:**
```
def verify_DAG(G):
    # Compute topological order
    order = topological_sort(G)
    if order is None:
        return "CYCLE_DETECTED"

    # Verify in topological order
    verified = set()
    for K in order:
        # Check prerequisites
        for K' in prerequisites(K):
            assert K' in verified
        # Verify K
        result = verify_permit(K)
        if result == PASS:
            verified.add(K)
        else:
            return f"FAILED at {K}"

    return "ALL_VERIFIED"
```

### Step 10: Compilation Theorem

**Theorem (DAG Structure):**

1. **Acyclicity:** The permit dependency graph is acyclic

2. **Topological Order:** Permits admit linear ordering for verification

3. **Depth Bound:** Maximum chain length $\leq D$

4. **Parallel Structure:** Independent permits can be verified in parallel

**Constructive Content:**
- Algorithm to detect cycles
- Algorithm to compute topological order
- Algorithm to identify parallel verification opportunities

## Key GMT Inequalities Used

1. **Scale Ordering:**
   $$K_i \to K_{i+1} \implies \lambda_i \leq \lambda_{i+1}$$

2. **Energy Ordering:**
   $$K_i \to K_{i+1} \implies \Phi_i \geq \Phi_{i+1}$$

3. **Depth Bound:**
   $$\text{depth}(K) \leq |\text{permit types}|$$

4. **Topological Existence:**
   $$\mathcal{G} \text{ is DAG} \iff \exists \text{ topological order}$$

## Literature References

- Kahn, A. B. (1962). Topological sorting of large networks. *Comm. ACM*, 5.
- Cormen, T., Leiserson, C., Rivest, R., Stein, C. (2009). *Introduction to Algorithms*. MIT Press.
- Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM J. Comput.*, 1.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 1. Addison-Wesley.
