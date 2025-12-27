---
title: "LOCK-Tactic-Capacity - Complexity Theory Translation"
---

# LOCK-Tactic-Capacity: Capacity Barriers as Space and Memory Bounds

## Original Hypostructure Statement

**Theorem (LOCK-Tactic-Capacity, Capacity Barrier):** Let $\mathcal{S}$ be a hypostructure with geometric background (BG) satisfying interface permit $\mathrm{Cap}_H$. Let $(B_k)$ be a sequence of subsets with increasing "thinness" (e.g., tubular neighborhoods of codimension-$\kappa$ sets with radius $r_k \to 0$) such that:

$$\sum_k \text{Cap}(B_k) < \infty$$

Then **occupation time bounds** hold: the trajectory cannot spend infinite time in thin sets.

**Sieve Target:** BarrierCap --- zero-capacity sets cannot sustain energy

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $\mathrm{TB}_\pi$ (Background Geometry)

**Prevented Failure Modes:** C.D (Geometric Collapse)

**Certificate Produced:** $K_{\text{Cap}}^{\text{blk}}$ with payload $(\text{Cap}(B), d_c, \mu_T)$

**Original Reference:** {prf:ref}`mt-lock-tactic-capacity`

---

## Complexity Theory Statement

**Theorem (Region Bounds via Capacity Limits):** Let $\mathcal{M}$ be a computational model with space bound $S(n)$ and let $\mathcal{R} \subseteq \{0,1\}^{S(n)}$ be a "region" (subset of memory configurations). Define the **computational capacity** of $\mathcal{R}$ as:

$$\text{Cap}_{\text{comp}}(\mathcal{R}) := \frac{|\mathcal{R}|}{2^{S(n)}}$$

If $\sum_k \text{Cap}_{\text{comp}}(\mathcal{R}_k) < \infty$ for a sequence of regions $(\mathcal{R}_k)$ with decreasing capacity, then:

1. **Occupation bound:** Any computation cannot spend more than $O\left(\frac{T(n)}{\sum_k 1/\text{Cap}_{\text{comp}}(\mathcal{R}_k)}\right)$ steps in all regions $\mathcal{R}_k$ combined
2. **Region exclusion:** Computations with bounded resources cannot concentrate in zero-capacity regions
3. **Memory collapse prevention:** Blow-up requiring concentration on thin memory sets is blocked

**Formal Statement:** Let $\mathcal{A}$ be an algorithm running in time $T(n)$ and space $S(n)$. For any sequence of memory regions $(\mathcal{R}_k)$ with:

$$\sum_{k=1}^{\infty} \text{Cap}_{\text{comp}}(\mathcal{R}_k) < C < \infty$$

the total occupation time satisfies:

$$\sum_{k=1}^{\infty} \mu_T(\mathcal{R}_k) \leq \frac{C \cdot (E + T)}{c_{\text{cap}}}$$

where $\mu_T(\mathcal{R}) = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}_{\mathcal{R}}(\text{config}(t))$ is the occupation measure and $E$ is the "computational energy" (total state changes).

**Key Insight:** The hypostructure capacity barrier---preventing trajectories from concentrating on thin sets---translates to complexity theory as **space bounds preventing computational concentration**. Just as zero-capacity sets cannot sustain energy in analysis, zero-capacity memory regions cannot sustain computation.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Interpretation |
|-----------------------|------------------------------|----------------|
| Singular set $\Sigma$ | Restricted memory region $\mathcal{R}$ | Subset of valid configurations |
| Sobolev capacity $\text{Cap}_{1,2}(\Sigma)$ | Computational capacity $\frac{|\mathcal{R}|}{2^S}$ | Fraction of memory space |
| Zero capacity $\text{Cap} = 0$ | Sparse region $|\mathcal{R}| = o(2^S)$ | Vanishing fraction of configurations |
| Occupation measure $\mu_T(B)$ | Time in region $\frac{1}{T}\sum_t \mathbf{1}_{\mathcal{R}}$ | Fraction of steps in region |
| Energy functional $\Phi(x)$ | Computational energy (state changes) | Resource consumption measure |
| Codimension $\kappa$ | Memory constraint dimension | Bits fixed by constraints |
| Tubular neighborhood radius $r$ | Hamming ball radius | Configurations within distance $r$ |
| Background geometry (BG) | Memory model structure | How configurations are organized |
| Capacity permit $\mathrm{Cap}_H$ | Space bound certificate | Verified memory limit |
| Geometric collapse (C.D) | Memory overflow/collapse | Computation exceeds space bounds |
| Occupation time bound | Step count bound in region | Time spent in restricted configurations |
| $K_{\text{Cap}}^{\text{blk}}$ certificate | Space-bounded exclusion proof | Proves region inaccessible |
| Critical codimension $d_c$ | Critical memory dimension | Threshold for exclusion |

---

## Proof Sketch (Complexity Theory Version)

### Setup: Space-Bounded Computation and Memory Regions

**Definitions:**

1. **Memory Configuration Space:** For space bound $S(n)$, the configuration space is $\mathcal{C} = \{0,1\}^{S(n)}$ with $|\mathcal{C}| = 2^{S(n)}$.

2. **Memory Region:** A subset $\mathcal{R} \subseteq \mathcal{C}$ representing constrained configurations.

3. **Computational Capacity:** The capacity of region $\mathcal{R}$ is:
   $$\text{Cap}_{\text{comp}}(\mathcal{R}) := \frac{|\mathcal{R}|}{2^{S(n)}}$$

4. **Occupation Measure:** For a computation with configuration sequence $(c_1, c_2, \ldots, c_T)$:
   $$\mu_T(\mathcal{R}) := \frac{|\{t : c_t \in \mathcal{R}\}|}{T}$$

5. **Computational Energy:** The total state-change energy:
   $$E := \sum_{t=1}^{T-1} d_H(c_t, c_{t+1})$$
   where $d_H$ is Hamming distance.

---

### Step 1: Capacity-Codimension Correspondence

**Claim (Memory Capacity Bound):** For memory regions defined by fixing $\kappa$ bits:
$$\text{Cap}_{\text{comp}}(\mathcal{R}) \leq 2^{-\kappa}$$

**Proof:**

**Step 1.1 (Constraint Analysis):** A region $\mathcal{R}$ defined by $\kappa$ linear constraints (fixed bit positions) has:
$$|\mathcal{R}| \leq 2^{S(n) - \kappa}$$

**Step 1.2 (Capacity Computation):**
$$\text{Cap}_{\text{comp}}(\mathcal{R}) = \frac{|\mathcal{R}|}{2^{S(n)}} \leq \frac{2^{S(n)-\kappa}}{2^{S(n)}} = 2^{-\kappa}$$

**Step 1.3 (Codimension Interpretation):** The parameter $\kappa$ is the "codimension" of the region---the number of degrees of freedom removed.

**Correspondence to Hypostructure:** This mirrors the capacity-codimension bound (BG4):
$$\text{Cap}(B) \leq C \cdot r^{d-\kappa}$$

The discrete version has $r = 1$ (single-step neighborhoods) and the bound becomes $2^{-\kappa}$.

---

### Step 2: Occupation Measure Bound

**Claim (Occupation-Capacity Inequality):** For a computation with energy $E$ and time $T$:
$$\mu_T(\mathcal{R}) \leq \frac{C_{\text{cap}} \cdot (E + T)}{\text{Cap}_{\text{comp}}(\mathcal{R})^{-1}}$$

**Proof:**

**Step 2.1 (State Counting):** Let $N_{\mathcal{R}} = |\{t : c_t \in \mathcal{R}\}|$ be the number of steps in region $\mathcal{R}$.

**Step 2.2 (Entry/Exit Analysis):** Each entry into $\mathcal{R}$ requires crossing the boundary $\partial \mathcal{R}$. The number of boundary crossings is bounded by the energy:
$$\text{crossings} \leq E$$

**Step 2.3 (Isoperimetric Bound):** By the Boolean cube isoperimetric inequality (Harper's theorem), the boundary of $\mathcal{R}$ satisfies:
$$|\partial \mathcal{R}| \geq |\mathcal{R}| \cdot h\left(\frac{|\mathcal{R}|}{2^{S(n)}}\right) \cdot S(n)$$

where $h(p) = -p \log p - (1-p) \log(1-p)$ is binary entropy.

**Step 2.4 (Occupation Bound):** For small capacity regions:
$$\mu_T(\mathcal{R}) = \frac{N_{\mathcal{R}}}{T} \leq \frac{\text{crossings} \cdot \text{avg. stay length}}{T}$$

The average stay length is bounded by the mixing time within $\mathcal{R}$, which for random walks is $O(|\mathcal{R}|^2) = O(2^{2(S(n)-\kappa)})$.

**Step 2.5 (Final Bound):** Combining:
$$\mu_T(\mathcal{R}) \leq C \cdot \frac{E + T}{\text{Cap}_{\text{comp}}(\mathcal{R})^{-1}} = C \cdot (E + T) \cdot \text{Cap}_{\text{comp}}(\mathcal{R})$$

---

### Step 3: Summability and Thin Region Exclusion

**Claim (Summable Capacity Bound):** If $\sum_k \text{Cap}_{\text{comp}}(\mathcal{R}_k) < \infty$, then:
$$\sum_k \mu_T(\mathcal{R}_k) < \infty$$

The computation can spend at most finite total time in all thin regions combined.

**Proof:**

**Step 3.1 (Term-by-Term Bound):** For each region $\mathcal{R}_k$:
$$\mu_T(\mathcal{R}_k) \leq C \cdot (E + T) \cdot \text{Cap}_{\text{comp}}(\mathcal{R}_k)$$

**Step 3.2 (Summation):**
$$\sum_k \mu_T(\mathcal{R}_k) \leq C \cdot (E + T) \cdot \sum_k \text{Cap}_{\text{comp}}(\mathcal{R}_k) < \infty$$

**Step 3.3 (Interpretation):** The computation cannot "hide" in thin regions. The total fraction of time in all capacity-bounded regions is finite.

**Certificate Production:**
$$K_{\text{Cap}}^{\text{blk}} = \left(\sum_k \text{Cap}(\mathcal{R}_k), d_c, \sum_k \mu_T(\mathcal{R}_k) < \infty\right)$$

---

### Step 4: Blocking Mechanism for Computational Collapse

**Claim (Zero-Capacity Region Exclusion):** If a computational "blow-up" requires concentration on regions with:
$$\dim(\mathcal{R}) < d_c \quad \text{(critical dimension)}$$
then the capacity is too small to support the computation:
$$\sum_{t \in \text{blow-up}} \mathbf{1}_{\mathcal{R}}(c_t) = 0$$

**Proof:**

**Step 4.1 (Critical Codimension):** Define the critical codimension $\kappa_c$ such that regions with $\kappa > \kappa_c$ have zero capacity in the limit:
$$\lim_{n \to \infty} 2^{S(n)} \cdot \text{Cap}_{\text{comp}}(\mathcal{R}_{\kappa}) = 0 \quad \text{for } \kappa > \kappa_c$$

**Step 4.2 (Energy Requirement):** A computation concentrating on $\mathcal{R}_{\kappa}$ would need:
$$E_{\text{required}} = \Omega\left(\frac{T}{\text{Cap}_{\text{comp}}(\mathcal{R}_{\kappa})}\right) = \Omega(T \cdot 2^{\kappa})$$

**Step 4.3 (Resource Violation):** For $\kappa > \kappa_c$, this exceeds any polynomial energy bound:
$$E_{\text{required}} > \text{poly}(n) \quad \text{for } \kappa = \omega(\log n)$$

**Step 4.4 (Exclusion):** A zero-capacity region cannot support polynomial-time computation. The capacity barrier blocks collapse.

---

### Step 5: Connection to Space Hierarchy

**Theorem (Space Hierarchy via Capacity):** For space-constructible $f$ and $g$ with $f(n) = o(g(n))$:
$$\text{DSPACE}(f(n)) \subsetneq \text{DSPACE}(g(n))$$

**Capacity Interpretation:**

**Step 5.1 (Region Definition):** Define the "f-bounded region":
$$\mathcal{R}_f := \{c \in \{0,1\}^{g(n)} : \text{only first } f(n) \text{ bits used}\}$$

**Step 5.2 (Capacity Computation):**
$$\text{Cap}_{\text{comp}}(\mathcal{R}_f) = \frac{2^{f(n)}}{2^{g(n)}} = 2^{f(n) - g(n)} \to 0$$

as $n \to \infty$ when $f(n) = o(g(n))$.

**Step 5.3 (Separation):** Computations in $\text{DSPACE}(f(n))$ are confined to region $\mathcal{R}_f$. By the capacity barrier, these cannot compute functions requiring exploration of the full $g(n)$-space.

**Step 5.4 (Diagonalization):** Construct a language $L$ that:
- Uses $g(n)$ space
- Requires visiting configurations outside $\mathcal{R}_f$
- Cannot be computed by machines confined to $\mathcal{R}_f$

The capacity barrier provides the structural reason why $f$-space machines cannot simulate $g$-space machines.

---

## Certificate Construction

**Mode: Capacity Barrier (Region Exclusion)**

```
K_CapacityBarrier = {
  mode: "Region_Exclusion",
  mechanism: "Capacity_Bound",
  evidence: {
    region_sequence: (R_1, R_2, ..., R_k),
    capacity_sum: "sum_k Cap(R_k) < C",
    occupation_bound: "sum_k mu_T(R_k) < C * (E + T)",
    critical_codimension: kappa_c
  },
  certificate_logic: "Cap^+ AND BG^+ => Cap^blk",
  payload: (Cap(B), d_c, mu_T)
}
```

**Mode: Space-Bounded Exclusion**

```
K_SpaceExclusion = {
  mode: "Space_Bounded",
  mechanism: "Capacity_Barrier",
  evidence: {
    space_bound: S(n),
    restricted_region: R_S,
    capacity: "Cap(R_S) = 2^{f(n) - S(n)}",
    exclusion: "f(n) = o(S(n)) => computation excluded from R_S"
  },
  certificate_logic: "Summable capacity => finite occupation",
  literature: "Savitch 1970, Stearns-Hartmanis-Lewis 1965"
}
```

---

## Connections to Space Complexity

### 1. Space Hierarchy Theorem (Stearns-Hartmanis-Lewis 1965)

**Statement:** For space-constructible $f, g$ with $f(n) = o(g(n))$:
$$\text{DSPACE}(f(n)) \subsetneq \text{DSPACE}(g(n))$$

**Capacity Translation:**

| Hypostructure | Space Hierarchy |
|---------------|-----------------|
| Zero-capacity region | $f$-bounded configurations |
| Cannot sustain energy | Cannot compute $g$-space functions |
| Occupation bound | Simulation overhead bound |
| Capacity barrier | Space separation |

**Key Insight:** The space hierarchy theorem states that computations confined to smaller memory regions cannot compute everything computable with larger regions. This is exactly the capacity barrier: thin regions (low capacity) cannot support complex computation.

### 2. Savitch's Theorem (1970)

**Statement:** $\text{NSPACE}(f(n)) \subseteq \text{DSPACE}(f(n)^2)$

**Capacity Interpretation:**

Nondeterministic space $f(n)$ explores a configuration graph of size $2^{f(n)}$. The deterministic simulation uses:
- Configuration space: $2^{f(n)^2}$
- Region for nondeterministic paths: $\text{Cap} = 2^{f(n)} / 2^{f(n)^2} = 2^{-f(n)(f(n)-1)}$

The quadratic overhead is the minimum needed to deterministically explore the exponential configuration space.

### 3. Immerman-Szelepcsényi Theorem (1988)

**Statement:** $\text{NSPACE}(f(n)) = \text{coNSPACE}(f(n))$ for $f(n) \geq \log n$

**Capacity Interpretation:**

Complementation requires counting reachable configurations. The capacity of the reachable set can be computed inductively:
- Level $k$: configurations reachable in $k$ steps
- Capacity grows as graph exploration proceeds
- The counting technique leverages capacity structure

### 4. PSPACE-Completeness

**Statement:** TQBF is PSPACE-complete.

**Capacity Interpretation:**

TQBF requires exploring a game tree of polynomial depth. The capacity of winning configurations at each quantifier level:
- $\exists$-level: $\text{Cap} \geq$ some winning branch exists
- $\forall$-level: $\text{Cap} \leq$ must handle all branches

The alternation of quantifiers creates a capacity hierarchy that requires polynomial space to resolve.

---

## Connections to Graph Capacity

### 1. Max-Flow Min-Cut (Ford-Fulkerson)

**Theorem:** In a flow network, the maximum flow equals the minimum cut capacity.

**Capacity Translation:**

| Hypostructure | Flow Networks |
|---------------|---------------|
| Sobolev capacity | Edge capacities |
| Occupation measure | Flow through vertex |
| Thin region | Bottleneck edges |
| Capacity barrier | Min-cut bound |

**Key Insight:** The min-cut prevents more flow than its capacity, just as zero-capacity regions prevent computational concentration.

### 2. Graph Capacity (Shannon Capacity)

**Definition:** The Shannon capacity of a graph $G$ is:
$$\Theta(G) := \sup_k \sqrt[k]{\alpha(G^k)}$$

where $\alpha(G^k)$ is the independence number of the $k$-th strong product.

**Capacity Interpretation:**

Shannon capacity bounds the rate of zero-error communication over a noisy channel. The capacity barrier in this context:
- High capacity: many distinguishable codewords
- Low capacity: few distinguishable codewords (thin region)
- Zero capacity: no reliable communication

**Certificate:** The famous result $\Theta(C_5) = \sqrt{5}$ (Lovász 1979) provides an explicit capacity bound via semidefinite programming.

### 3. Expander Graphs and Boundary Expansion

**Definition:** A graph is a $(k, \epsilon)$-expander if for all sets $S$ with $|S| \leq k$:
$$|\partial S| \geq \epsilon |S|$$

**Capacity Interpretation:**

In expander graphs:
- All small sets have large boundary (high isoperimetric constant)
- No region can have low capacity relative to its boundary
- Random walks mix rapidly

The capacity barrier for computation on expanders:
- Cannot concentrate in small regions
- Must explore the full graph
- Space-time tradeoffs are tight

### 4. Isoperimetric Inequalities

**Boolean Cube Isoperimetry (Harper 1966):**
$$|\partial S| \geq |S| \cdot \log\left(\frac{2^n}{|S|}\right)$$

**Capacity Interpretation:**

For a region $\mathcal{R}$ with capacity $p = |\mathcal{R}|/2^n$:
$$|\partial \mathcal{R}| \geq |\mathcal{R}| \cdot \log(1/p) = |\mathcal{R}| \cdot \log(1/\text{Cap}(\mathcal{R}))$$

Low capacity regions have:
- Large boundary-to-volume ratio
- High "surface tension"
- Difficulty maintaining occupancy

This is exactly the mechanism behind the occupation time bound.

---

## Quantitative Bounds

### Capacity-Codimension Table

| Codimension $\kappa$ | Capacity $2^{-\kappa}$ | Occupation Bound | Complexity Class |
|---------------------|----------------------|------------------|------------------|
| $0$ | $1$ | Unbounded | Full space |
| $\log n$ | $1/n$ | $O(T/n)$ | Polynomial-sparse |
| $\sqrt{n}$ | $2^{-\sqrt{n}}$ | $O(T \cdot 2^{-\sqrt{n}})$ | Subexponential-sparse |
| $n/2$ | $2^{-n/2}$ | $O(T \cdot 2^{-n/2})$ | Exponential-sparse |
| $n - O(\log n)$ | $\text{poly}(n)/2^n$ | Negligible | Polynomial-size |

### Space Hierarchy Gaps

| Lower Bound $f(n)$ | Upper Bound $g(n)$ | Capacity Ratio | Separation |
|-------------------|-------------------|----------------|------------|
| $\log n$ | $\log^2 n$ | $n^{1-\log n}$ | Strict |
| $\sqrt{n}$ | $n$ | $2^{-n/2}$ | Exponential |
| $n$ | $n^2$ | $2^{-n(n-1)}$ | Double-exponential |

### Occupation Time Bounds

For computation with time $T$ and space $S$:

| Region Type | Capacity | Max Occupation Time |
|-------------|----------|---------------------|
| Full space | $1$ | $T$ |
| Half space | $1/2$ | $T$ |
| Polynomial region | $\text{poly}(n)/2^S$ | $O(T/2^S \cdot \text{poly}(n))$ |
| Single configuration | $2^{-S}$ | $O(T \cdot 2^{-S})$ |
| Zero capacity (limit) | $0$ | $0$ |

---

## Algorithmic Implementation

### Capacity Barrier Verification

```
function VerifyCapacityBarrier(regions: List[Region], time: T, energy: E):
    total_capacity := 0
    occupation_bound := 0

    for R_k in regions:
        cap_k := ComputeCapacity(R_k)
        total_capacity += cap_k
        occupation_bound += C * (E + T) * cap_k

    if total_capacity < CAPACITY_THRESHOLD:
        return Certificate(
            mode="Region_Exclusion",
            total_capacity=total_capacity,
            occupation_bound=occupation_bound,
            status="BLOCKED"
        )
    else:
        return Certificate(
            mode="No_Exclusion",
            status="PASS"
        )
```

### Space Bound Verification

```
function VerifySpaceBound(computation: Computation, space_limit: S):
    config_sequence := computation.configurations()
    restricted_region := ConfigurationsUsingSpace(space_limit)

    occupation_time := 0
    for config in config_sequence:
        if config in restricted_region:
            occupation_time += 1

    occupation_fraction := occupation_time / len(config_sequence)
    capacity := |restricted_region| / 2^(computation.max_space())

    return Certificate(
        occupation_fraction=occupation_fraction,
        capacity=capacity,
        bound_satisfied=(occupation_fraction <= C * capacity)
    )
```

---

## Worked Example: Space-Bounded Palindrome Recognition

**Problem:** Recognize palindromes $\{w w^R : w \in \{0,1\}^*\}$ with various space bounds.

**Analysis:**

1. **Full space ($S = n$):** All configurations available. Capacity = 1.
   - Simple algorithm: store $w$, compare with $w^R$
   - Occupation: uniform over all configurations

2. **Sublinear space ($S = o(n)$):** Restricted configurations.
   - Capacity of reachable configs: $2^{o(n)}/2^n = 2^{-\Omega(n)}$
   - Cannot store full input
   - Must use crossing sequence arguments

3. **Logarithmic space ($S = O(\log n)$):** Severely restricted.
   - Capacity: $\text{poly}(n)/2^n = 2^{-\Omega(n)}$
   - **Result:** Palindromes $\notin$ DSPACE$(\log n)$ (Hennie 1965)

**Capacity Barrier Proof:**

The language of palindromes requires distinguishing $2^{n/2}$ different "positions" in the input. With $O(\log n)$ space:
- Available configurations: $\text{poly}(n)$
- Required distinctions: $2^{n/2}$
- Capacity ratio: $\text{poly}(n) / 2^{n/2} \to 0$

By the capacity barrier, the computation cannot maintain enough distinct states to solve the problem.

**Certificate:**
$$K_{\text{Palindrome}}^{\text{blk}} = (\text{Cap} = \text{poly}(n)/2^{n/2}, d_c = n/2, \text{insufficient})$$

---

## Summary

The LOCK-Tactic-Capacity theorem, translated to complexity theory, establishes:

1. **Capacity Limits Bound Regions:** Computational capacity (fraction of memory space) bounds how long a computation can occupy restricted regions.

2. **Summable Capacity Implies Finite Occupation:** If a sequence of regions has summable capacity, the total time in all regions is finite.

3. **Zero-Capacity Regions Cannot Sustain Computation:** Just as zero-capacity sets cannot sustain energy in analysis, zero-capacity memory regions cannot support unbounded computation.

4. **Space Hierarchy via Capacity:** The space hierarchy theorem is a manifestation of the capacity barrier---smaller spaces have smaller capacity and cannot simulate larger spaces.

5. **Graph Capacity Connections:** Flow networks, Shannon capacity, and isoperimetric inequalities all embody the capacity barrier principle.

**The Complexity-Theoretic Insight:**

The capacity barrier prevents computational collapse into thin regions. This explains:
- Why space hierarchies exist (small space = small capacity)
- Why sparse sets cannot be NP-complete (zero capacity)
- Why space-time tradeoffs are necessary (concentration requires capacity)

**Physical Interpretation:**

| Hypostructure | Complexity Theory |
|---------------|-------------------|
| Zero-capacity singular set | Sparse memory region |
| Cannot sustain energy | Cannot sustain computation |
| Occupation time bound | Step count bound |
| Geometric collapse | Memory overflow |
| Capacity barrier | Space hierarchy |

---

## Literature

1. **Federer, H. (1969).** *Geometric Measure Theory.* Springer. *Capacity and removable singularities.*

2. **Evans, L. C. & Gariepy, R. F. (1992).** *Measure Theory and Fine Properties of Functions.* CRC Press. *Sobolev capacity.*

3. **Adams, D. R. & Hedberg, L. I. (1996).** *Function Spaces and Potential Theory.* Springer. *Capacity theory foundations.*

4. **Maz'ya, V. G. (1985).** *Sobolev Spaces.* Springer. *Capacity and function spaces.*

5. **Stearns, R. E., Hartmanis, J., & Lewis, P. M. (1965).** "Hierarchies of Memory Limited Computations." *FOCS.* *Space hierarchy theorem.*

6. **Savitch, W. J. (1970).** "Relationships Between Nondeterministic and Deterministic Tape Complexities." *JCSS.* *Space simulation theorem.*

7. **Immerman, N. (1988).** "Nondeterministic Space is Closed Under Complementation." *SIAM J. Comput.* *NSPACE closure.*

8. **Harper, L. H. (1966).** "Optimal Numberings and Isoperimetric Problems on Graphs." *J. Combin. Theory.* *Boolean cube isoperimetry.*

9. **Ford, L. R. & Fulkerson, D. R. (1956).** "Maximal Flow Through a Network." *Canadian J. Math.* *Max-flow min-cut.*

10. **Lovász, L. (1979).** "On the Shannon Capacity of a Graph." *IEEE Trans. Inform. Theory.* *Shannon capacity via theta function.*

11. **Hennie, F. C. (1965).** "One-Tape, Off-Line Turing Machine Computations." *Information and Control.* *Palindrome lower bound.*

12. **Arora, S. & Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge. *Space complexity reference.*
