---
title: "UP-Shadow - Complexity Theory Translation"
---

# UP-Shadow: Topological Sector Suppression via SCC Decomposition

## Overview

This document provides a complete complexity-theoretic translation of the UP-Shadow theorem (Topological Sector Suppression) from the hypostructure framework. The translation establishes a formal correspondence between exponential suppression of nontrivial topological sectors and structural simplification in computational graphs via strongly connected component (SCC) decomposition, reachability analysis, and topological sorting.

**Original Theorem Reference:** {prf:ref}`mt-up-shadow`

**Core Translation:**
- **Topological sector organization suppresses shadow sectors** $\to$ **Sector Organization**: Topological structure simplifies complexity
- **Action gap $\Delta > 0$** $\to$ **Edge weight gap in DAG structure**
- **Log-Sobolev inequality** $\to$ **Spectral expansion / mixing properties**
- **Exponential measure suppression** $\to$ **Exponential reduction in reachable states**

---

## Complexity Theory Statement

**Theorem (Topological Sector Suppression, Computational Form).**
Let $G = (V, E)$ be a directed graph representing a computational state space with:
1. **Sector decomposition:** $V = V_0 \sqcup V_1 \sqcup \cdots \sqcup V_k$ (partition into strongly connected components)
2. **Trivial sector:** $V_0$ is the primary/trivial component (containing initial states)
3. **Nontrivial sectors:** $V_i$ for $i \geq 1$ represent "shadow" or exceptional computational states
4. **Transition cost:** Each edge $e \in E$ crossing sectors has weight $w(e) \geq \Delta > 0$
5. **Stationary distribution:** The computation admits an invariant measure $\pi$ satisfying a log-Sobolev inequality with constant $\lambda_{\text{LS}} > 0$

**Statement (Exponential Sector Suppression):**
The probability of the computation residing in any nontrivial sector is exponentially suppressed:
$$\pi(\{v : \tau(v) \neq 0\}) \leq \exp\left(-c \lambda_{\text{LS}} \frac{\Delta^2}{L^2}\right)$$

where $\tau: V \to \{0, 1, \ldots, k\}$ is the sector labeling, $L$ is the Lipschitz constant of the cost functional, and $c = 1/8$ is a universal constant.

**Corollary (Effective Triviality).** For computations with:
- Large action gap $\Delta$, or
- Strong mixing (large $\lambda_{\text{LS}}$), or
- Small Lipschitz constant $L$

the nontrivial sectors become negligible, and the computation is effectively confined to the trivial sector $V_0$.

**Certificate Logic:**
$$K_{\mathrm{TB}_\pi}^+ \wedge K_{\lambda_{\text{LS}}}^+ \wedge K_{\Delta}^+ \Rightarrow K_{\text{Action}}^{\mathrm{blk}}$$

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Topological background (TB) | Directed graph structure $G = (V, E)$ | State transition system |
| Sector decomposition $\mathcal{X} = \bigsqcup S_i$ | SCC decomposition of $G$ | Tarjan/Kosaraju partition |
| Trivial topological sector $\tau = 0$ | Trivial/primary SCC (containing source) | Reachable main component |
| Nontrivial sector $\tau \neq 0$ | Shadow SCC (hard to reach/escape) | Exceptional computational states |
| Action gap $\Delta > 0$ | Minimum cross-SCC edge weight | Barrier to sector transitions |
| Action functional $\mathcal{A}$ | Path cost / potential function | Cumulative transition cost |
| Lipschitz constant $L$ | Maximum cost gradient | Sensitivity of path cost |
| Invariant measure $\mu$ | Stationary distribution $\pi$ | Long-run state occupancy |
| Log-Sobolev inequality (LSI) | Spectral gap / mixing time bound | Concentration of measure |
| LSI constant $\lambda_{\text{LS}}$ | Spectral gap of transition matrix | Rate of convergence to $\pi$ |
| Herbst argument | Concentration inequality proof | Measure tail bounds |
| Exponential suppression $e^{-c\Delta^2}$ | Negligible probability | Effective unreachability |
| Certificate $K_{\text{Action}}^{\mathrm{blk}}$ | Sector suppression proof | Barrier certificate |
| Topological obstruction | Unreachable component | Dead code / unreachable states |
| Shadow sector measure | Exception probability | Rare event frequency |

---

## Proof Sketch

### Setup: State Space as Directed Graph

**Definitions:**

1. **State Space Graph:** $G = (V, E)$ where:
   - $V$ = set of computational states
   - $E \subseteq V \times V$ = allowed transitions
   - $w: E \to \mathbb{R}^+$ = transition costs

2. **Strongly Connected Component (SCC):** A maximal subset $S \subseteq V$ such that every vertex in $S$ is reachable from every other vertex in $S$.

3. **Condensation DAG:** The quotient graph $G/\text{SCC}$ where each SCC is contracted to a single node, forming a directed acyclic graph.

4. **Sector Labeling:** $\tau: V \to \{0, 1, \ldots, k\}$ assigns each vertex to its SCC, with $\tau = 0$ for the primary component.

5. **Cross-Sector Edges:** $E_{\text{cross}} := \{(u, v) \in E : \tau(u) \neq \tau(v)\}$

**Resource Functional (Action):**

The action functional assigns a cost to each path:
$$\mathcal{A}(p) = \sum_{e \in p} w(e)$$

The minimum action to reach sector $i$ from sector $0$ is:
$$\Delta_i := \min\{\mathcal{A}(p) : p \text{ is a path from } V_0 \text{ to } V_i\}$$

The action gap is $\Delta := \min_{i \geq 1} \Delta_i > 0$.

---

### Step 1: SCC Decomposition (Topological Background)

**Claim (Sector Structure).** Every directed graph admits a unique decomposition into strongly connected components, forming a DAG structure.

**Proof:**

**Step 1.1 (Tarjan's Algorithm):** Compute SCCs in $O(|V| + |E|)$ time:
```
TARJAN-SCC(G):
    index = 0
    S = empty stack
    for each v in V:
        if v.index is undefined:
            STRONGCONNECT(v)

STRONGCONNECT(v):
    v.index = v.lowlink = index++
    S.push(v)
    v.onStack = true

    for each (v, w) in E:
        if w.index is undefined:
            STRONGCONNECT(w)
            v.lowlink = min(v.lowlink, w.lowlink)
        else if w.onStack:
            v.lowlink = min(v.lowlink, w.index)

    if v.lowlink == v.index:
        start new SCC
        repeat:
            w = S.pop()
            w.onStack = false
            add w to current SCC
        until w == v
```

**Step 1.2 (DAG Structure):** The condensation graph $G_{\text{SCC}}$ is acyclic:
- If there were a cycle $C_1 \to C_2 \to \cdots \to C_k \to C_1$ in $G_{\text{SCC}}$
- Then all vertices in $C_1 \cup \cdots \cup C_k$ would form a single SCC
- Contradiction with maximality of SCCs

**Step 1.3 (Certificate):**
$$K_{\mathrm{TB}_\pi}^+ = (G, \{V_0, \ldots, V_k\}, G_{\text{SCC}}, \text{DAG\_proof})$$

---

### Step 2: Action Gap Analysis (Barrier Construction)

**Claim (Positive Action Gap).** If nontrivial sectors are separated from the trivial sector by costly transitions, then $\Delta > 0$.

**Proof:**

**Step 2.1 (Cross-Sector Cost):** For each cross-sector edge $(u, v)$ with $\tau(u) = 0$ and $\tau(v) \neq 0$:
$$w(u, v) \geq \delta_{\text{min}} > 0$$

This represents the minimum "cost" to exit the trivial sector.

**Step 2.2 (Path Cost Lower Bound):** Any path from $V_0$ to $V_i$ (for $i \geq 1$) must cross at least one cross-sector edge:
$$\mathcal{A}(p) \geq \delta_{\text{min}} = \Delta$$

**Step 2.3 (Lipschitz Bound):** The action functional $\mathcal{A}$ is Lipschitz with constant $L$:
$$|\mathcal{A}(p_1) - \mathcal{A}(p_2)| \leq L \cdot d_{\text{graph}}(p_1, p_2)$$

where $L = \max_{e \in E} w(e)$.

**Step 2.4 (Certificate):**
$$K_{\Delta}^+ = (\Delta, L, E_{\text{cross}}, \text{gap\_proof})$$

---

### Step 3: Log-Sobolev Inequality (Mixing Condition)

**Claim (Spectral Expansion).** If the transition matrix has a spectral gap, the system satisfies a log-Sobolev inequality.

**Proof:**

**Step 3.1 (Transition Matrix):** Define the transition matrix $P: V \times V \to [0, 1]$:
$$P(u, v) = \Pr[\text{transition from } u \text{ to } v]$$

**Step 3.2 (Reversibility):** If $P$ is reversible with respect to $\pi$:
$$\pi(u) P(u, v) = \pi(v) P(v, u)$$

**Step 3.3 (Spectral Gap):** Let $\lambda_2$ be the second-largest eigenvalue of $P$. The spectral gap is:
$$\gamma := 1 - \lambda_2 > 0$$

**Step 3.4 (LSI from Spectral Gap):** For reversible Markov chains, the log-Sobolev constant satisfies:
$$\lambda_{\text{LS}} \geq \frac{\gamma}{2 \log(1/\pi_{\min})}$$

where $\pi_{\min} = \min_v \pi(v)$.

**Step 3.5 (Certificate):**
$$K_{\lambda_{\text{LS}}}^+ = (\lambda_{\text{LS}}, \gamma, P, \text{spectral\_proof})$$

---

### Step 4: Herbst Argument (Concentration Bound)

**Claim (Exponential Suppression).** The log-Sobolev inequality implies exponential concentration of measure.

**Proof:**

**Step 4.1 (Herbst Inequality):** For any 1-Lipschitz function $f: V \to \mathbb{R}$:
$$\pi(\{f \geq \mathbb{E}_\pi[f] + t\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} t^2}{2}\right)$$

**Step 4.2 (Action as Test Function):** Consider $f = \mathcal{A}/L$, which is 1-Lipschitz. Then:
$$\pi(\{v : \mathcal{A}(v) \geq \mathbb{E}_\pi[\mathcal{A}] + tL\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} t^2}{2}\right)$$

**Step 4.3 (Sector Inclusion):** Vertices in nontrivial sectors satisfy $\mathcal{A}(v) \geq \Delta$. If $\mathbb{E}_\pi[\mathcal{A}] < \Delta$ (typical case), then:
$$\{v : \tau(v) \neq 0\} \subseteq \{\mathcal{A} \geq \Delta\} \subseteq \{\mathcal{A} \geq \mathbb{E}_\pi[\mathcal{A}] + \Delta/L \cdot L\}$$

**Step 4.4 (Final Bound):** Setting $t = \Delta/L$:
$$\pi(\{v : \tau(v) \neq 0\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} \Delta^2}{2L^2}\right)$$

With careful constant tracking, this becomes:
$$\pi(\{v : \tau(v) \neq 0\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} \Delta^2}{8L^2}\right)$$

**Step 4.5 (Certificate):**
$$K_{\text{Action}}^{\mathrm{blk}} = (\Delta, \lambda_{\text{LS}}, L, \exp(-c\lambda_{\text{LS}}\Delta^2/L^2))$$

---

### Step 5: Effective Triviality (Shadow Elimination)

**Claim (Sector Suppression).** For large $\Delta$ or $\lambda_{\text{LS}}$, nontrivial sectors are effectively unreachable.

**Proof:**

**Step 5.1 (Negligible Probability):** For $\lambda_{\text{LS}} \Delta^2 / L^2 \gg 1$:
$$\pi(\{v : \tau(v) \neq 0\}) \ll 1$$

The computation spends negligible time in nontrivial sectors.

**Step 5.2 (Effective Reduction):** The computation is effectively equivalent to one restricted to $V_0$:
- Error probability: $O(\exp(-c\lambda_{\text{LS}}\Delta^2/L^2))$
- Effective state space: $|V_0| \ll |V|$

**Step 5.3 (Topological Simplification):** The condensation DAG reduces to effectively a single node (the trivial sector), eliminating topological obstructions.

**Step 5.4 (Computational Benefit):**
- State space reduction: $|V| \to |V_0|$
- Path enumeration: Exponentially fewer relevant paths
- Reachability analysis: Trivial for the reduced system

---

## Connections to SCC Decomposition

### 1. Tarjan's Algorithm and Sector Identification

**Theorem (Linear-Time SCC Decomposition).**
Given a directed graph $G = (V, E)$, all strongly connected components can be computed in $O(|V| + |E|)$ time.

**Connection to UP-Shadow:**

| Tarjan's Algorithm | UP-Shadow |
|-------------------|-----------|
| DFS tree construction | Trajectory exploration |
| Low-link values | Action functional values |
| Back edges | Cycles within sectors |
| Cross edges | Cross-sector transitions |
| SCC identification | Sector labeling $\tau$ |
| Condensation DAG | Topological sector structure |

**Algorithmic Implementation:**
```
SECTOR-SUPPRESSION-ANALYSIS(G, Delta, lambda_LS, L):
    // Phase 1: SCC Decomposition
    SCCs = TARJAN-SCC(G)
    condensation = BUILD-CONDENSATION-DAG(SCCs)

    // Phase 2: Identify Trivial Sector
    V_0 = FIND-SOURCE-SCC(condensation, initial_states)

    // Phase 3: Compute Action Gaps
    for each SCC C != V_0:
        Delta_C = MIN-CROSS-SECTOR-COST(V_0, C)

    // Phase 4: Compute Suppression Bound
    Delta = min(Delta_C for C != V_0)
    suppression = exp(-lambda_LS * Delta^2 / (8 * L^2))

    // Phase 5: Certificate
    if suppression < threshold:
        return K_Action^blk (shadow sectors suppressed)
    else:
        return K_Action^- (suppression insufficient)
```

### 2. Kosaraju's Algorithm and Two-Pass Analysis

**Alternative SCC Algorithm:**
1. First pass: DFS on $G$, record finish times
2. Second pass: DFS on $G^T$ (transpose) in reverse finish order

**Connection to UP-Shadow:**
- First pass: Forward reachability (potential to reach shadow sectors)
- Second pass: Backward reachability (potential to return from shadow sectors)
- SCCs are exactly sets with both forward and backward reachability

### 3. DAG Properties and Topological Sorting

**Theorem (Topological Ordering).**
The condensation DAG admits a topological ordering $C_1, C_2, \ldots, C_k$ such that all edges go from earlier to later components.

**Connection to UP-Shadow:**

| Topological Sort | UP-Shadow |
|-----------------|-----------|
| Source SCCs | Trivial sector (reachable from start) |
| Sink SCCs | Terminal shadow sectors |
| Linear ordering | Action level sets |
| Path in DAG | Sequence of sector transitions |
| DAG depth | Maximum number of sector transitions |

**Implication:** The action gap $\Delta$ corresponds to the minimum edge weight in the condensation DAG. Large $\Delta$ means crossing to the next "level" is expensive.

---

## Connections to Reachability Analysis

### 1. Single-Source Reachability

**Problem:** Given source $s$, determine all vertices reachable from $s$.

**Algorithm:** BFS/DFS from $s$ in $O(|V| + |E|)$.

**Connection to UP-Shadow:**
- Reachable set from trivial sector = $\bigcup_{i} V_i$ (all sectors)
- But probability-weighted reachability is concentrated on $V_0$
- The exponential suppression means "effective reachability" is much smaller

### 2. Cost-Bounded Reachability

**Problem:** Given source $s$ and budget $B$, find vertices reachable with cost $\leq B$.

**Algorithm:** Dijkstra's algorithm or BFS on weighted graph.

**Connection to UP-Shadow:**
- With budget $B < \Delta$: only $V_0$ is reachable
- With budget $B \geq \Delta$: shadow sectors become reachable
- The LSI + action gap implies that typical trajectories have low cost

### 3. Probabilistic Reachability

**Problem:** Given transition probabilities, compute $\pi(\text{Reach}(S))$ for target set $S$.

**Connection to UP-Shadow:**
- Target set $S = \{v : \tau(v) \neq 0\}$ (nontrivial sectors)
- UP-Shadow bounds $\pi(S) \leq \exp(-c\lambda_{\text{LS}}\Delta^2/L^2)$
- This is a concentration inequality for probabilistic reachability

---

## Connections to Topological Sorting

### 1. Layered Structure

**Definition (Layer Decomposition).**
Assign each vertex to layer $\ell(v) = $ length of longest path from a source to $v$.

**Connection to UP-Shadow:**
- Layer 0: Trivial sector $V_0$ (source SCC)
- Layer $i$: Sectors reachable in exactly $i$ cross-sector transitions
- Action gap: Minimum cost per layer transition

**Visualization:**
```
Layer 0:    [V_0: Trivial Sector]
               |
               | (cost >= Delta)
               v
Layer 1:    [V_1] [V_2] [V_3]  (Shadow sectors)
               |     |
               | (cost >= Delta)
               v     v
Layer 2:    [V_4] [V_5]  (Deeper shadow sectors)
               ...
```

### 2. Critical Path Analysis

**Definition (Critical Path).**
The longest weighted path in a DAG.

**Connection to UP-Shadow:**
- Critical path through condensation DAG = maximum action path
- Suppression strength depends on minimum cross-sector cost
- Dense DAGs with many paths: suppression depends on typical, not worst, cost

### 3. Level Ancestors and Sector Hierarchy

**Definition (Level Ancestor).**
For vertex $v$ at layer $\ell$, its level-$k$ ancestor is the sector reached by going back $k$ layers.

**Connection to UP-Shadow:**
- The trivial sector $V_0$ is the "root ancestor"
- Action to reach sector $V_i$ = sum of costs along path from $V_0$ to $V_i$
- Exponential suppression: probability decays exponentially with layer depth

---

## Certificate Payload Structure

```
K_Shadow := {
  topological_structure: {
    graph: G = (V, E),
    sccs: [V_0, V_1, ..., V_k],
    condensation_dag: G_SCC,
    topological_order: [C_1, ..., C_k],
    trivial_sector: V_0
  },

  action_gap: {
    Delta: minimum cross-sector cost,
    L: Lipschitz constant of action,
    cross_edges: E_cross,
    gap_certificate: proof that Delta > 0
  },

  mixing_properties: {
    transition_matrix: P,
    spectral_gap: gamma,
    log_sobolev_constant: lambda_LS,
    stationary_distribution: pi
  },

  suppression_bound: {
    exponent: c * lambda_LS * Delta^2 / L^2,
    probability_bound: exp(-exponent),
    effective_state_space: V_0
  },

  certificate_type: K_Action^blk,
  mechanism: "Herbst concentration via LSI"
}
```

---

## Quantitative Bounds

### Suppression Exponent

$$\text{Suppression} = \exp\left(-\frac{\lambda_{\text{LS}} \Delta^2}{8 L^2}\right)$$

**Regime Analysis:**

| Parameter Regime | Suppression | Interpretation |
|-----------------|-------------|----------------|
| $\Delta \gg L$ | $\exp(-\Omega(\Delta^2))$ | Strong barrier, negligible shadow |
| $\lambda_{\text{LS}} \gg 1$ | $\exp(-\Omega(\lambda_{\text{LS}}))$ | Rapid mixing concentrates measure |
| $\Delta \sim L$ | $\exp(-O(1))$ | Moderate suppression |
| $\Delta \ll L$ | $\sim 1$ | Weak suppression, sectors accessible |

### Mixing Time vs. Suppression

**Relationship:**
$$t_{\text{mix}} \approx \frac{1}{\gamma} \approx \frac{\log(1/\pi_{\min})}{\lambda_{\text{LS}}}$$

For strong suppression:
- Fast mixing ($t_{\text{mix}}$ small) $\Rightarrow$ large $\lambda_{\text{LS}}$ $\Rightarrow$ strong suppression

### SCC Statistics

For random graphs $G(n, p)$:
- Expected number of SCCs: $O(1)$ for $p > 1/n$
- Giant SCC size: $\Theta(n)$ for $p > 1/n$
- Shadow sector fraction: typically $o(1)$

---

## Worked Example: Web Graph Crawling

**Problem:** Analyze reachability in a web graph with link costs.

**Setup:**
- Vertices: Web pages
- Edges: Hyperlinks
- Costs: PageRank-based transition costs (inverse of link value)
- Sectors: SCCs correspond to web communities

**Analysis:**

1. **SCC Decomposition:**
   - Giant SCC: Main web (well-connected pages)
   - Shadow SCCs: Isolated communities, dead links, spam islands

2. **Action Gap:**
   - $\Delta =$ cost to reach spam/isolated pages from main web
   - High-quality sites: low exit cost to spam
   - Spam sites: high exit cost to main web

3. **Suppression:**
   - Random walks concentrate on giant SCC
   - Spam islands have negligible PageRank
   - Suppression: $\pi(\text{spam}) \propto \exp(-\text{link\_gap}^2)$

**Certificate:**
```
K_Web = {
  trivial_sector: "Main web (giant SCC)",
  shadow_sectors: ["Spam islands", "Dead links", "Orphan pages"],
  action_gap: Delta = link_quality_threshold,
  suppression: exp(-PageRank_gap^2)
}
```

---

## Worked Example: Control Flow Analysis

**Problem:** Identify effectively unreachable code regions in a program.

**Setup:**
- Vertices: Basic blocks
- Edges: Control flow transitions
- Costs: Branch probability (inverse of likelihood)
- Sectors: SCCs in control flow graph

**Analysis:**

1. **SCC Decomposition:**
   - Main loop: Primary SCC (frequently executed)
   - Error handlers: Shadow SCCs (rarely executed)
   - Dead code: Unreachable SCCs

2. **Action Gap:**
   - $\Delta =$ negative log-probability of exceptional path
   - Normal execution: low action
   - Exception path: high action

3. **Suppression:**
   - Normal paths dominate execution time
   - Exception handlers: $\pi(\text{exception}) \propto p_{\text{exception}}$
   - For rare exceptions: exponentially suppressed

**Application:** Profile-guided optimization uses empirical $\pi$ to identify hot (trivial sector) vs. cold (shadow sector) code.

---

## Summary

The UP-Shadow theorem translates to complexity theory as **Topological Sector Suppression via SCC Decomposition**:

1. **Sector Structure:** Strongly connected components partition the state space into sectors, with a DAG structure on the quotient graph.

2. **Action Gap:** The minimum cost to transition from the trivial (primary) sector to shadow (nontrivial) sectors provides a barrier $\Delta > 0$.

3. **Mixing Properties:** The log-Sobolev inequality (or spectral gap) ensures concentration of the invariant measure.

4. **Exponential Suppression:** Combining the action gap with mixing properties yields:
   $$\pi(\text{shadow sectors}) \leq \exp\left(-c\lambda_{\text{LS}}\frac{\Delta^2}{L^2}\right)$$

5. **Effective Triviality:** For large gap or strong mixing, the computation is effectively confined to the trivial sector, eliminating topological obstructions.

**Key Algorithmic Connections:**

| Algorithm | Role in UP-Shadow |
|-----------|-------------------|
| Tarjan's SCC | Sector identification |
| Topological sort | Layer structure of sectors |
| Dijkstra/BFS | Action gap computation |
| Power iteration | Spectral gap estimation |
| PageRank | Stationary distribution |

**The Core Insight:**

Just as topological sectors in dynamical systems can be exponentially suppressed by action gaps and mixing, computational state spaces can be effectively simplified by identifying strongly connected components and quantifying the "cost" of accessing rare components. The log-Sobolev inequality provides the concentration mechanism that converts structural separation (action gap) into probabilistic suppression (negligible measure).

$$K_{\mathrm{TB}_\pi}^+ \wedge K_{\lambda_{\text{LS}}}^+ \wedge K_{\Delta}^+ \Rightarrow K_{\text{Action}}^{\mathrm{blk}}$$

translates to:

$$\text{SCC Structure} \wedge \text{Spectral Gap} \wedge \text{Transition Cost Gap} \Rightarrow \text{Shadow Sector Suppression}$$

---

## Literature

1. **Tarjan, R. E. (1972).** "Depth-First Search and Linear Graph Algorithms." SIAM J. Comput. 1(2), 146-160. *SCC algorithm.*

2. **Kosaraju, S. R. (1978).** Unpublished manuscript. *Alternative SCC algorithm.*

3. **Sharir, M. (1981).** "A Strong-Connectivity Algorithm and its Applications." Computers & Math. with Applications. *SCC applications.*

4. **Herbst, I. W. (1975).** "Spectral Theory of the Operator $(p^2 + m^2)^{1/2} - Ze^2/r$." Commun. Math. Phys. *Herbst argument for concentration.*

5. **Ledoux, M. (2001).** *The Concentration of Measure Phenomenon.* AMS. *Log-Sobolev inequalities and concentration.*

6. **Bobkov, S. G. & Gotze, F. (1999).** "Exponential Integrability and Transportation Cost Related to Log-Sobolev Inequalities." J. Funct. Anal. *LSI and transport.*

7. **Diaconis, P. & Saloff-Coste, L. (1996).** "Logarithmic Sobolev Inequalities for Finite Markov Chains." Ann. Appl. Probab. *Discrete LSI.*

8. **Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).** "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab. *Stationary distribution on web graphs.*

9. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).** *Introduction to Algorithms.* MIT Press. *Graph algorithms.*

10. **Motwani, R. & Raghavan, P. (1995).** *Randomized Algorithms.* Cambridge. *Probabilistic analysis.*

11. **Lojasiewicz, S. (1963).** "Sur le probleme de la division." Studia Math. *Gradient inequalities.*

12. **Spielman, D. A. & Teng, S.-H. (2011).** "Spectral Sparsification of Graphs." SIAM J. Comput. *Spectral graph theory.*
