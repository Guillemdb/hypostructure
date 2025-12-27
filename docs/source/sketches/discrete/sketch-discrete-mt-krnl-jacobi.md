---
title: "KRNL-Jacobi - Complexity Theory Translation"
---

# KRNL-Jacobi: Action Reconstruction

## Original Theorem

**[KRNL-Jacobi] Action Reconstruction** (Theorem {prf:ref}`mt-krnl-jacobi`)

Given a hypostructure satisfying interface permits $D_E$ (dissipation-energy inequality), $\mathrm{LS}_\sigma$ (linear stability), and $\mathrm{GC}_\nabla$ (gradient consistency), the canonical Lyapunov functional equals the geodesic distance in the Jacobi metric:

$$\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$$

where $g_{\mathfrak{D}} := \mathfrak{D} \cdot g$ is the conformal scaling of the base metric by the dissipation rate.

---

## Complexity Theory Statement

**Theorem (Computational Distance as Resource-Weighted Shortest Path):**

Let $G = (V, E, w)$ be a weighted directed graph representing a computational state space where:
- $V$ is the set of computational states
- $E$ is the set of valid transitions between states
- $w: E \to \mathbb{R}_{>0}$ assigns resource costs to each transition

Let $A \subseteq V$ be the set of accepting (safe/terminal) states. Then the **optimal remaining complexity** from any state $v \in V$ is:

$$L(v) = \mathrm{dist}_w(v, A) := \min_{\pi: v \leadsto A} \sum_{e \in \pi} w(e)$$

where the minimum is over all paths $\pi$ from $v$ to any accepting state in $A$.

**Key Properties:**
1. $L(v) = 0$ if and only if $v \in A$
2. $L$ is monotonically non-increasing along any computation path
3. The gradient of $L$ (discrete: the edge minimizing $L(v) - L(u) + w(v,u)$) points toward optimal computation

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent | Interpretation |
|----------------------|------------------------------|----------------|
| State space $\mathcal{X}$ | Vertex set $V$ | Set of all computational configurations |
| Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$ | Weighted graph metric | Edge weight = resource cost of transition |
| Dissipation rate $\mathfrak{D}(x)$ | Local transition cost $w(x, \cdot)$ | Computational resources consumed at state $x$ |
| Conformal scaling | Edge weight assignment | Cost varies by state: expensive states have heavy outgoing edges |
| Geodesic in Jacobi metric | Shortest weighted path | Minimum-cost computation sequence |
| Safe manifold $M$ | Accepting states $A$ | Target configurations (halting, correct output, etc.) |
| Lyapunov functional $\mathcal{L}(x)$ | Distance to acceptance $\mathrm{dist}_w(v, A)$ | Remaining computational work |
| Gradient flow $\dot{u} = -\nabla \Phi$ | Greedy descent algorithm | Follow locally optimal transitions |
| Boundary condition $\mathcal{L}|_M = \Phi_{\min}$ | $L(v) = 0$ for $v \in A$ | Zero remaining work at acceptance |
| Gradient consistency $\mathrm{GC}_\nabla$ | Edge costs match local work | $w(u,v) = $ actual resources for $u \to v$ |

---

## Proof Sketch

### Setup: Weighted Computation Graphs

**Definition (Weighted Computation Graph):**
A weighted computation graph is a triple $G = (V, E, w)$ where:
- $V$ is a finite or countable set of **computational states**
- $E \subseteq V \times V$ is the set of **valid transitions** (directed edges)
- $w: E \to \mathbb{R}_{>0}$ is the **cost function** assigning positive weights to edges

**Examples of Edge Weights:**
- **Time complexity:** $w(u, v) = $ number of elementary operations for transition
- **Space complexity:** $w(u, v) = $ memory cells accessed or modified
- **Communication complexity:** $w(u, v) = $ bits transmitted
- **Energy complexity:** $w(u, v) = $ energy consumed by transition
- **Query complexity:** $w(u, v) = $ number of oracle calls

**Definition (Path Cost):**
For a path $\pi = (v_0, v_1, \ldots, v_k)$ in $G$, the total cost is:
$$\mathrm{cost}(\pi) := \sum_{i=0}^{k-1} w(v_i, v_{i+1})$$

**Definition (Accepting States):**
A subset $A \subseteq V$ of **accepting states** represents successful termination configurations. In different contexts:
- **Decision problems:** States where the correct answer is computed
- **Optimization:** States achieving the optimum (or within tolerance)
- **Verification:** States where the certificate is validated
- **Dynamical systems:** Stable equilibria or absorbing states

---

### Step 1: Weighted Graph Model of Computation

**Construction:** Given a computational problem, construct the weighted graph $G = (V, E, w)$ as follows:

**Vertices (Computational States):**
Each vertex $v \in V$ encodes:
- Current memory configuration
- Program counter / control state
- Input/output tape contents
- Any auxiliary data structures

For a Turing machine with $n$-bit tape and $q$ states: $|V| \leq q \cdot 2^n$.

**Edges (Transitions):**
An edge $(u, v) \in E$ exists if the computation can transition from configuration $u$ to configuration $v$ in one step according to the computational model.

**Edge Weights (Resource Costs):**
The weight $w(u, v)$ captures the **local computational cost** of the transition $u \to v$. This is the discrete analog of the dissipation rate $\mathfrak{D}(x)$:

$$w(u, v) \sim \mathfrak{D}(x) \cdot \|x' - x\|$$

In the continuous setting, $\mathfrak{D}(x)$ scales the metric conformally; in the discrete setting, edge weights directly encode this scaling.

**Key Insight:** The Jacobi metric's conformal scaling $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$ translates to: **high-dissipation regions have expensive transitions**. In complexity terms:
- Computationally intensive states have high outgoing edge costs
- Easy transitions (simple operations) have low edge weights

---

### Step 2: Shortest Path Distance as Lyapunov Function

**Theorem (Distance Function Properties):**
Define $L: V \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ by:
$$L(v) := \mathrm{dist}_w(v, A) = \inf_{\pi: v \leadsto A} \mathrm{cost}(\pi)$$

Then $L$ satisfies:

1. **Boundary Condition:** $L(v) = 0$ for all $v \in A$

2. **Non-negativity:** $L(v) \geq 0$ for all $v \in V$

3. **Monotonicity (Lyapunov Property):** For any edge $(u, v) \in E$:
   $$L(v) \leq L(u) - w(u, v) + w(u, v) = L(u)$$

   More precisely, by the **Bellman equation**:
   $$L(u) = \min_{(u,v) \in E} \{w(u, v) + L(v)\}$$

   This is the discrete analog of $\frac{d}{dt}\mathcal{L}(u(t)) = -\mathfrak{D}(u(t)) \leq 0$.

4. **Strict Decrease Along Optimal Paths:** If $(u, v)$ lies on an optimal path to $A$:
   $$L(v) = L(u) - w(u, v) < L(u)$$

   The Lyapunov function strictly decreases by exactly the edge cost.

**Proof:**
The Bellman equation follows from the principle of optimality: the optimal path from $u$ consists of one step to some neighbor $v$, then the optimal path from $v$ to $A$. The minimum over all choices of $v$ gives the optimal cost from $u$.

This is the **discrete Hamilton-Jacobi equation**:
- Continuous: $\|\nabla_g \mathcal{L}\|_g^2 = \mathfrak{D}$
- Discrete: $L(u) = \min_v \{w(u,v) + L(v)\}$

---

### Step 3: Gradient Flow as Greedy Descent

**Definition (Discrete Gradient):**
At state $u \notin A$, the **discrete gradient direction** is the neighbor minimizing future cost:
$$v^* = \arg\min_{(u,v) \in E} \{w(u, v) + L(v)\}$$

**Greedy Descent Algorithm:**
```
Algorithm GreedyDescent(G, w, A, start):
    v := start
    path := [v]
    while v not in A:
        v := argmin_{(v,u) in E} {w(v,u) + L(u)}
        path.append(v)
    return path
```

**Theorem (Greedy Descent = Optimal Path):**
If $L$ is computed exactly (e.g., via Dijkstra's algorithm), then `GreedyDescent` produces a minimum-cost path from `start` to $A$.

**Proof:**
By the Bellman equation, each greedy choice $(u, v^*)$ satisfies:
$$L(u) = w(u, v^*) + L(v^*)$$

Summing along the path $\pi = (v_0, v_1, \ldots, v_k)$ where $v_k \in A$:
$$L(v_0) = \sum_{i=0}^{k-1} w(v_i, v_{i+1}) + L(v_k) = \mathrm{cost}(\pi) + 0 = \mathrm{cost}(\pi)$$

Since $L(v_0) = \mathrm{dist}_w(v_0, A)$ by definition, $\pi$ achieves the minimum cost.

**Interpretation:** Following the gradient flow (greedy descent) in the weighted graph produces the **computationally optimal trajectory** --- the sequence of transitions that minimizes total resource expenditure to reach acceptance.

---

### Step 4: Metric Space Structure and Triangle Inequality

**Theorem (Weighted Graph Metric):**
Define $d_w: V \times V \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ by:
$$d_w(u, v) := \inf_{\pi: u \leadsto v} \mathrm{cost}(\pi)$$

Then $d_w$ satisfies (for symmetric edge weights or undirected graphs):

1. **Identity:** $d_w(v, v) = 0$
2. **Positivity:** $d_w(u, v) \geq 0$, with equality iff $u = v$ (if graph is connected)
3. **Triangle Inequality:** $d_w(u, w) \leq d_w(u, v) + d_w(v, w)$

**Proof of Triangle Inequality:**
Any path from $u$ to $w$ passing through $v$ has cost at least $d_w(u, v) + d_w(v, w)$ (by definition of infimum). The infimum over all paths (including those not through $v$) can only be smaller.

**Complexity Interpretation:**
The triangle inequality states: **path concatenation cannot reduce cost**.
- You cannot compute "faster" by taking a detour
- Any intermediate state $v$ adds at least $d_w(u, v) + d_w(v, w)$ cost
- This is the **subadditivity of computational resources**

**Connection to Original Theorem:**
In the continuous Jacobi metric setting:
$$\mathrm{dist}_{g_{\mathfrak{D}}}(x, M) = \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \|\dot{\gamma}(s)\|_g \, ds$$

The discrete analog replaces:
- Path integral $\to$ sum over edges
- $\sqrt{\mathfrak{D}} \cdot \|\dot{\gamma}\| \to$ edge weight $w$
- Infimum over continuous curves $\to$ infimum over graph paths

---

## Certificate Construction

**Explicit Certificate:** $(G, T, \ell)$ where:
- $G = (V, E, w)$: The weighted computation graph
- $T$: Shortest path tree rooted at $A$ (or shortest path DAG if multiple optimal paths exist)
- $\ell: V \to \mathbb{R}_{\geq 0}$: Distance labels with $\ell(v) = \mathrm{dist}_w(v, A)$

**Certificate Verification:**
Given $(G, T, \ell)$, verify in polynomial time:
1. **Boundary Check:** $\ell(v) = 0$ for all $v \in A$
2. **Bellman Consistency:** For all $u \notin A$:
   $$\ell(u) = \min_{(u,v) \in E} \{w(u,v) + \ell(v)\}$$
3. **Tree Validity:** $T$ forms a valid shortest path tree:
   - Every non-accepting vertex has exactly one outgoing edge in $T$
   - Following $T$ from any vertex reaches $A$
   - For edge $(u, v) \in T$: $\ell(u) = w(u,v) + \ell(v)$

**Completeness:** If the distance labels satisfy Bellman's equation with correct boundary conditions, they must equal the true shortest path distances (by uniqueness of solutions to the Bellman equation).

**Certificate Size:**
- $|V|$ distance labels (one real number per vertex)
- $|V| - |A|$ tree edges (one per non-accepting vertex)
- Total: $O(|V| + |E|)$ space

---

## Connections to Classical Results

### Dijkstra's Algorithm

**Dijkstra (1959):** Computes single-source shortest paths in $O(|E| + |V| \log |V|)$ time using a priority queue.

**Connection:** Dijkstra's algorithm constructs exactly the certificate $(T, \ell)$:
- Processes vertices in order of increasing distance from source
- Maintains the invariant that processed vertices have final distance labels
- Produces the shortest path tree as a byproduct

**Complexity Translation:** Computing the Lyapunov function $L(v) = \mathrm{dist}_w(v, A)$ for all vertices is equivalent to running Dijkstra from the accepting set $A$ (reversed edges for directed graphs).

### Floyd-Warshall All-Pairs Shortest Paths

**Floyd-Warshall (1962):** Computes all-pairs shortest paths in $O(|V|^3)$ time via dynamic programming.

**Recurrence:**
$$d^{(k)}_{ij} = \min\{d^{(k-1)}_{ij}, d^{(k-1)}_{ik} + d^{(k-1)}_{kj}\}$$

where $d^{(k)}_{ij}$ is the shortest path from $i$ to $j$ using only intermediate vertices $\{1, \ldots, k\}$.

**Connection:** Floyd-Warshall computes the full weighted graph metric $d_w(u, v)$ for all pairs, not just distances to the accepting set. This is useful when:
- The accepting set $A$ may change
- Multiple Lyapunov functions needed for different targets
- Analyzing the global metric structure of the state space

### Bellman-Ford Algorithm

**Bellman-Ford (1958):** Handles negative edge weights, runs in $O(|V| \cdot |E|)$ time.

**Connection:** While the complexity translation assumes positive costs ($w > 0$, corresponding to $\mathfrak{D} > 0$ dissipation), Bellman-Ford handles cases where some transitions have "negative cost" --- computational operations that generate resources (e.g., memoization savings, garbage collection freeing memory).

**Negative Cycles:** A negative-weight cycle would correspond to a computation that generates unbounded resources --- physically unrealistic but detectable by Bellman-Ford.

### Graph Metric Embeddings

**Bourgain's Theorem (1985):** Any $n$-point metric space embeds into $\ell_p$ with distortion $O(\log n)$.

**Connection:** The weighted graph metric $d_w$ can be approximately embedded into low-dimensional Euclidean space. This enables:
- Approximate nearest-neighbor queries for finding "nearby" accepting states
- Dimensionality reduction for visualizing computation state spaces
- Fast approximation algorithms trading accuracy for speed

**Johnson-Lindenstrauss Lemma:** Random projections preserve distances up to $(1 \pm \epsilon)$ factor in $O(\log n / \epsilon^2)$ dimensions.

### Potential Functions in Algorithm Analysis

**Amortized Analysis:** The distance function $L(v)$ is a **potential function** in the sense of amortized complexity analysis:
- Each transition $(u, v)$ has **amortized cost** = actual cost + change in potential
- $\text{amortized}(u \to v) = w(u,v) + L(v) - L(u) \geq 0$
- Equality holds on optimal paths: amortized cost = 0 for optimal transitions

**Connection to Hypostructure:** The gradient consistency condition $\mathrm{GC}_\nabla$ ensures that the "accounting" is exact:
- Dissipation $\mathfrak{D}$ matches the rate of Lyapunov decrease
- No "hidden costs" or unaccounted resource consumption

---

## Extended Complexity Interpretations

### Time-Space Tradeoffs

**Multi-Dimensional Edge Weights:** Generalize to $w: E \to \mathbb{R}^k_{>0}$ where:
- $w_1(e)$ = time cost
- $w_2(e)$ = space cost
- ...
- $w_k(e)$ = cost in resource $k$

**Pareto-Optimal Paths:** Instead of a single shortest path, compute the Pareto frontier of paths trading off different resources.

**Connection:** Multi-objective shortest path corresponds to multi-dimensional Jacobi metrics, relevant for systems with multiple conserved quantities or multiple constraints.

### Communication Complexity

**Two-Party Computation:** States $V$ encode joint configurations of Alice and Bob. Edge weights $w(u, v)$ represent communication bits required for the transition.

**Communication-Optimal Protocol:** The shortest weighted path from initial state to accepting state gives the minimum-communication protocol.

**Jacobi Metric Interpretation:** Regions of state space requiring heavy communication have large "dissipation" --- information must be exchanged to make progress.

### Circuit Complexity

**States as Partial Circuits:** Vertices represent partially constructed circuits; edges represent adding gates.

**Gate Costs:** $w(u, v) = $ cost of the gate added (e.g., 1 for standard gates, higher for expensive operations).

**Minimum Circuit:** Shortest path to acceptance = minimum-cost circuit computing the target function.

---

## Summary

The KRNL-Jacobi theorem translates naturally to complexity theory:

| Continuous Setting | Discrete/Complexity Setting |
|-------------------|----------------------------|
| Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$ | Edge-weighted graph $(V, E, w)$ |
| Geodesic distance to safe manifold | Shortest weighted path to accepting set |
| Lyapunov function $\mathcal{L}(x)$ | Distance function $L(v) = \mathrm{dist}_w(v, A)$ |
| Gradient flow | Greedy descent / optimal policy |
| Hamilton-Jacobi equation | Bellman optimality equation |
| Conformal scaling | State-dependent transition costs |

**Core Message:** The "remaining computational complexity" of a configuration is precisely its weighted shortest-path distance to acceptance. The Jacobi metric framework provides a principled way to assign edge weights (transition costs) based on local resource consumption, and the resulting distance function serves as both:
1. A **Lyapunov certificate** (monotonically decreasing, zero at acceptance)
2. An **optimal cost-to-go function** (guiding greedy computation to minimum-resource solutions)

This unifies dynamical systems theory (Lyapunov stability) with computational complexity (resource-optimal algorithms) through the language of weighted graphs and shortest paths.
