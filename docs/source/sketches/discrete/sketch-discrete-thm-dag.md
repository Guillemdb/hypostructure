---
title: "DAG Structure - Complexity Theory Translation"
---

# THM-DAG: Directed Acyclic Graph Structure

## Overview

This document provides a complete complexity-theoretic translation of the DAG Structure theorem from the hypostructure framework. The translation establishes a formal correspondence between the sieve diagram's directed acyclic graph structure and branching programs in computational complexity theory, revealing deep connections to circuit complexity and decision procedures.

**Original Theorem Reference:** {prf:ref}`thm-dag`

---

## Original Theorem Statement

**Theorem (DAG Structure).** The sieve diagram is a directed acyclic graph (DAG). All edges, including dotted surgery re-entry edges, point forward in the topological ordering. Consequently:
1. No backward edges exist
2. Each epoch visits at most $|V|$ nodes where $|V|$ is the number of nodes
3. The sieve terminates

**Literature:** Topological sorting of DAGs (Kahn 1962); termination via well-founded orders (Floyd 1967).

---

## Complexity Theory Statement

**Theorem (THM-DAG, Computational Form).**
Let $\mathcal{B} = (V, E, s, T, \lambda)$ be a branching program where:
- $V$ is a finite set of nodes (decision points)
- $E \subseteq V \times V$ is the edge relation (computational transitions)
- $s \in V$ is the start node
- $T \subseteq V$ is the set of terminal (accepting/rejecting) nodes
- $\lambda: V \setminus T \to \{0, 1\}^*$ assigns query variables to internal nodes

Suppose $\mathcal{B}$ satisfies the **progress property**: there exists a ranking function $r: V \to \mathbb{N}$ such that for all edges $(u, v) \in E$:
$$r(v) > r(u)$$

Then:
1. **Acyclicity:** The computation graph $(V, E)$ contains no directed cycles
2. **Bounded Evaluation:** Every computation path has length at most $|V|$
3. **Termination Guarantee:** Every input reaches a terminal node in finite steps

**Corollary (Polynomial-Depth Decision).** If $|V| = \mathrm{poly}(n)$ for input size $n$, then the branching program computes in polynomial time with respect to path length.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Analog | Formal Correspondence |
|-----------------------|--------------------------|------------------------|
| Sieve diagram | Branching program | Decision graph with queries |
| Directed acyclic graph | Cycle-free computation | No state revisits |
| Edges pointing forward | Progress guarantee | Ranking function increases |
| Topological ordering | Evaluation order | Valid computation sequence |
| Node $N_i$ | Decision node / query | Predicate evaluation point |
| Solid edges | Deterministic transitions | Query-dependent successor |
| Dotted surgery edges | Non-local jumps / gotos | Still forward in ranking |
| Terminal nodes (VICTORY, Mode) | Accepting/rejecting states | Halting configurations |
| Epoch | Single program execution | Start-to-terminal path |
| Node count $|V|$ | Program size | Branching program width |
| Path length | Computation depth | Time complexity |

---

## The DAG Correspondence

### Sieve Diagram as Branching Program

The sieve diagram implements a branching program where each node corresponds to a predicate query:

**Structure Mapping:**

| Sieve Component | Branching Program Element |
|-----------------|---------------------------|
| Regularity gate (Reg) | First query node |
| Energy gate ($D_E$) | Resource check query |
| Concentration gate ($C_\mu$) | Branching decision |
| Admissibility nodes (A1-A17) | Conditional queries |
| Lock tactics (E1-E12) | Terminal decision nodes |
| Surgery re-entry | Forward jump (goto) |
| VICTORY | Accept state |
| Mode $D.D$ | Reject state |

**Key Insight:** The sieve is a read-once branching program (ROBP) where each predicate is queried at most once per path, and all transitions respect the topological ordering.

---

## Proof Sketch

### Step 1: Definition of the Computation DAG

**Definition (Sieve Computation Graph).**
The sieve computation graph $\mathcal{G} = (V, E)$ is defined as:

- **Nodes:** $V = \{N_0, N_1, \ldots, N_k\} \cup \{A_1, \ldots, A_m\} \cup T$ where:
  - $N_i$ are decision gates (Reg, $D_E$, $C_\mu$, etc.)
  - $A_j$ are admissibility check nodes
  - $T = \{\text{VICTORY}, \text{Mode}_1, \ldots, \text{Mode}_{15}, \text{FATAL}\}$ are terminal nodes

- **Edges:** $(u, v) \in E$ if there exists a transition from $u$ to $v$ in the sieve diagram, including:
  - Solid edges: standard predicate-outcome transitions
  - Dotted edges: surgery re-entry transitions

**Definition (Topological Ranking).**
Assign each node $v \in V$ a rank $r(v) \in \mathbb{N}$ according to the fixed node numbering in the sieve diagram. The numbering is designed such that:
- START has rank 0
- Each subsequent gate has increasing rank
- Terminal nodes have maximal rank
- Surgery re-entry targets have rank strictly greater than their source

### Step 2: Verification of the Forward Property

**Lemma (Solid Edge Forward Property).** All solid edges $(u, v)$ satisfy $r(v) > r(u)$.

**Proof.** By inspection of the sieve diagram:
1. **Gate-to-gate edges:** Each decision gate $N_i$ transitions to gates $N_j$ with $j > i$ or to terminal nodes
2. **Gate-to-admissibility edges:** Barrier breach triggers admissibility checks at strictly higher rank
3. **Admissibility-to-terminal edges:** All admissibility outcomes flow to terminals (maximal rank)
4. **Restoration subtree (7a-7d):** Internal edges flow forward; exits go to TopoCheck or TameCheck at higher rank

The solid edges form the "downward" flow in the diagram, monotonically increasing in the node numbering. $\square$

**Lemma (Dotted Edge Forward Property).** All dotted surgery re-entry edges $(u, v)$ satisfy $r(v) > r(u)$.

**Proof.** Surgery edges originate from mode outcomes and target reconstruction entry points. By design:
1. Mode nodes occur at admissibility check depth
2. Re-entry targets are positioned at restoration subtree nodes
3. The diagram explicitly ensures $r(\text{target}) > r(\text{source})$ to prevent cycles

Specifically, if surgery from Mode $M$ re-enters at node $R$, the diagram satisfies:
$$r(R) > r(M) > r(\text{barrier that triggered } M)$$

This forward constraint is a design invariant of the sieve architecture. $\square$

**Theorem (DAG Property).** The sieve computation graph $\mathcal{G} = (V, E)$ is a directed acyclic graph.

**Proof.** Suppose for contradiction that $\mathcal{G}$ contains a directed cycle:
$$v_0 \to v_1 \to \cdots \to v_k \to v_0$$

By the Forward Property (Lemmas above), each edge strictly increases rank:
$$r(v_0) < r(v_1) < \cdots < r(v_k) < r(v_0)$$

This yields $r(v_0) < r(v_0)$, a contradiction. Therefore no cycle exists. $\square$

### Step 3: Topological Sort and Evaluation Order

**Definition (Topological Sort).** A topological sort of DAG $\mathcal{G} = (V, E)$ is a linear ordering $\prec$ of $V$ such that for all edges $(u, v) \in E$: $u \prec v$.

**Lemma (Existence and Constructibility).** The sieve DAG admits a unique topological sort computable in $O(|V| + |E|)$ time via Kahn's algorithm.

**Proof.** By the DAG property, the graph is acyclic. Every DAG admits at least one topological sort (Kahn 1962). The sieve's explicit node numbering provides the canonical topological ordering. Kahn's algorithm:
1. Initialize queue $Q$ with nodes having in-degree 0 (START node)
2. While $Q$ non-empty:
   - Remove node $u$ from $Q$, append to sorted order
   - For each outgoing edge $(u, v)$: decrement in-degree of $v$; if zero, add $v$ to $Q$
3. If all nodes processed, output is valid topological sort

Time complexity: $O(|V| + |E|)$. $\square$

**Corollary (Evaluation Order).** The topological sort defines a valid evaluation order for the sieve: each node is evaluated only after all its predecessors have been processed.

### Step 4: Bounded Path Length and Termination

**Theorem (Bounded Computation Depth).** Every computation path in the sieve has length at most $|V|$.

**Proof.** Let $\pi = v_0 \to v_1 \to \cdots \to v_m$ be a computation path from START to a terminal.
1. By acyclicity, all $v_i$ are distinct
2. Therefore $m + 1 \leq |V|$
3. Hence the path length $m \leq |V| - 1$

The bound is tight: the longest path visits every node exactly once. $\square$

**Theorem (Termination).** Every execution of the sieve terminates in a terminal node within $|V|$ steps.

**Proof.** From the START node:
1. Each step follows an edge to a strictly higher-ranked node (Forward Property)
2. The rank function is bounded: $r(v) \leq |V| - 1$ for all $v$
3. Therefore, after at most $|V|$ steps, either:
   - A terminal node is reached (halting)
   - Or the path would exceed the maximum rank (impossible by construction)

Since all paths are bounded and the graph is finite, every execution terminates. $\square$

---

## Certificate Construction

The DAG structure proof is constructive and yields explicit certificates:

**Acyclicity Certificate $K_{\mathrm{DAG}}^+$:**
$$K_{\mathrm{DAG}}^+ = \left(r: V \to \mathbb{N}, \text{proof that } \forall (u,v) \in E.\ r(v) > r(u)\right)$$

This certificate consists of:
- The ranking function $r$ (the node numbering)
- Verification that each edge respects the ranking

**Termination Certificate $K_{\mathrm{Term}}^+$:**
$$K_{\mathrm{Term}}^+ = \left(\pi = (v_0, v_1, \ldots, v_m), t \in T, \text{proof that } v_m = t\right)$$

where $\pi$ is the computation path and $t$ is the reached terminal.

**Topological Order Certificate $K_{\mathrm{Topo}}^+$:**
$$K_{\mathrm{Topo}}^+ = \left(\prec\ \text{total order on } V, \text{proof that } \forall (u,v) \in E.\ u \prec v\right)$$

---

## Quantitative Refinements

### Path Length Distribution

For the sieve with $|V|$ nodes:
- **Minimum path length:** $O(1)$ (immediate VICTORY on regularity)
- **Maximum path length:** $|V| - 1$ (all gates traversed)
- **Expected path length:** Depends on predicate distribution

### Branching Factor Analysis

**Definition (Branching Factor).** The branching factor at node $v$ is $b(v) = |\{u : (v, u) \in E\}|$.

For the sieve:
- **Binary decisions:** Most gates have $b(v) = 2$ (YES/NO outcomes)
- **Ternary decisions:** Some gates include BLOCKED outcome: $b(v) = 3$
- **Multi-way branches:** Admissibility nodes may have $b(v) > 3$

**Total Edges:** $|E| = \sum_{v \in V} b(v) = O(|V|)$ for the sieve (sparse graph).

### Depth vs. Width Tradeoff

| Metric | Sieve Value | Branching Program Analog |
|--------|-------------|--------------------------|
| Depth $d$ | $\leq |V|$ | Computation time |
| Width $w$ | $O(1)$ | Memory / parallelism |
| Size $|V|$ | Fixed by design | Program size |

The sieve is a **polynomial-size, polynomial-depth** branching program with **constant width** (single active node at each step).

---

## Connections to Classical Results

### 1. Branching Programs (Barrington 1986)

**Theorem (Barrington).** Width-5 branching programs of polynomial length can compute exactly the functions in $\mathrm{NC}^1$.

**Connection to THM-DAG:** The sieve diagram is a width-1 (oblivious) branching program, hence captures at most $\mathrm{NC}^1$ computations per epoch. The theorem ensures:
- Each epoch is efficiently computable
- Multiple epochs compose polynomially (by epoch-termination bounds)

**Interpretation:** The sieve's DAG structure guarantees that the decision procedure lies in $\mathrm{NC}^1$ (polylogarithmic depth, polynomial size circuits), ensuring efficient parallelizability.

### 2. Decision Trees and Query Complexity

**Definition (Decision Tree).** A decision tree is a rooted tree where internal nodes are labeled with queries and leaves are labeled with outcomes.

**Connection to THM-DAG:** The sieve is a generalized decision tree (decision DAG) where:
- Queries correspond to predicate evaluations
- The DAG structure allows shared subcomputations
- Polynomial depth bounds query complexity

**Query Complexity Bound:** For a sieve with $n$ predicates:
$$Q(\text{sieve}) \leq |V| = O(n)$$

Each predicate is queried at most once per path (read-once property).

### 3. Circuit DAGs and Straight-Line Programs

**Definition (Circuit DAG).** A Boolean circuit is a DAG where:
- Source nodes are inputs
- Internal nodes are gates (AND, OR, NOT)
- Sink nodes are outputs

**Connection to THM-DAG:** The sieve diagram is a decision circuit:
- Input: System configuration and certificate states
- Gates: Predicate evaluations
- Output: Terminal classification (VICTORY/Mode/FATAL)

**Size-Depth Tradeoff:** By the DAG theorem:
- Circuit size: $|V| + |E| = O(|V|)$
- Circuit depth: $\leq |V|$
- Can be parallelized to depth $O(\log |V|)$ with polynomial blowup

### 4. Well-Founded Orders and Termination Proofs (Floyd 1967)

**Theorem (Floyd).** A program terminates if there exists a well-founded order $\prec$ on states such that each transition decreases the state in this order.

**Connection to THM-DAG:** The ranking function $r: V \to \mathbb{N}$ induces a well-founded order:
$$u \prec v \iff r(u) < r(v)$$

Since $(\mathbb{N}, <)$ is well-founded and each edge increases rank:
- The edge relation is well-founded
- No infinite descending chains exist
- Termination follows immediately

**Constructive Content:** Floyd's method provides not just termination but a bound:
$$\text{steps to termination} \leq r(\text{START}) - r(\text{terminal}) + 1$$

### 5. Topological Sorting (Kahn 1962)

**Theorem (Kahn).** A directed graph admits a topological sort if and only if it is acyclic.

**Connection to THM-DAG:** The theorem provides:
- **Existence:** DAG property implies topological sort exists
- **Uniqueness:** The sieve's canonical numbering is the unique topological sort
- **Algorithm:** Kahn's algorithm computes the sort in linear time

**Verification:** Given a purported topological sort $\prec$:
- Check each edge $(u, v)$: verify $u \prec v$
- Time: $O(|E|)$
- Certificate: the sorted sequence plus edge-wise verification

### 6. Read-Once Branching Programs

**Definition (Read-Once Branching Program, ROBP).** A branching program is read-once if every source-to-sink path queries each variable at most once.

**Theorem (ROBP Characterization).** ROBPs compute exactly the functions with polynomial-size formulas (NC$^1$).

**Connection to THM-DAG:** The sieve is an ROBP:
- Each predicate (variable) is evaluated at most once per path
- The DAG structure ensures no variable is re-queried
- This places sieve computation in NC$^1$

**Implication:** The sieve's decision procedure is:
- Efficiently parallelizable (NC$^1$)
- Computable by polynomial-size formulas
- Verifiable in logarithmic space

---

## Extension: Multi-Epoch Composition

The DAG theorem applies to single epochs. For complete sieve runs:

**Definition (Multi-Epoch Computation).** A complete sieve run is a sequence of epochs $\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_k$ where:
- Each epoch is a START-to-terminal path
- Surgery triggers initiate subsequent epochs
- The run terminates when no surgery is triggered

**Theorem (Finite Epoch Bound).** A complete sieve run consists of finitely many epochs.

**Proof Sketch.** Each epoch produces a certificate that:
1. Resolves an obstruction (surgery success), or
2. Terminates at a final mode (no surgery possible)

The obstruction count is bounded by system topology. Each successful surgery reduces obstructions. Hence the epoch count is finite. $\square$

**Complexity:** For $k$ epochs, each of depth $\leq |V|$:
$$\text{Total computation} = O(k \cdot |V|) = O(\text{surgery bound} \cdot |V|)$$

---

## Algorithmic Implications

### DAG Verification Algorithm

Given a sieve diagram specification:

```
Input: Graph G = (V, E) with designated START and terminal nodes
Output: True if G is a valid sieve DAG, False otherwise

1. Compute topological sort via Kahn's algorithm
2. If sort fails (cycle detected): return False
3. Verify START has in-degree 0
4. Verify all terminals have out-degree 0
5. Verify all non-terminals have out-degree >= 1
6. Return True
```

**Complexity:** $O(|V| + |E|)$

### Path Enumeration Algorithm

```
Input: Sieve DAG G, START node s
Output: All possible computation paths

1. Initialize: paths = {[s]}
2. While exists incomplete path in paths:
   a. Select path p ending at non-terminal v
   b. For each successor u of v:
      - Extend p with u, add to paths
3. Return paths (all ending at terminals)
```

**Complexity:** $O(|V|! / \text{constraints})$ worst case, but typically polynomial due to DAG structure.

### Certificate Extraction

The DAG structure enables efficient certificate extraction:

1. **Trace Certificate:** The actual path taken through the DAG
2. **Ranking Certificate:** The ranking function values along the path
3. **Termination Certificate:** The terminal node reached and its classification

---

## Summary

The THM-DAG theorem, translated to complexity theory, establishes that:

1. **Computation is Acyclic:** The sieve diagram forms a directed acyclic graph, ensuring no circular reasoning or infinite loops in the decision procedure.

2. **Progress is Guaranteed:** The forward edge property, formalized as a ranking function, ensures every step makes monotonic progress toward termination.

3. **Evaluation Order Exists:** Topological sorting provides a valid evaluation order, enabling systematic traversal of the decision procedure.

4. **Termination is Bounded:** The path length bound $|V|$ provides an explicit complexity measure: each epoch completes in polynomial time.

5. **Classical Foundations:** The theorem connects to fundamental results in complexity theory:
   - Branching programs (Barrington): NC$^1$ computation
   - Decision trees: Query complexity bounds
   - Circuit DAGs: Parallel evaluation
   - Well-founded orders (Floyd): Termination verification
   - Topological sorting (Kahn): Linear-time algorithms

This translation reveals that the sieve's DAG structure is the complexity-theoretic foundation for its correctness: acyclicity prevents unsound circular arguments, the topological order enables systematic evaluation, and the depth bound ensures polynomial-time decision procedures.
