---
title: "KRNL-Trichotomy - Complexity Theory Translation"
---

# KRNL-Trichotomy: Structural Resolution

## Complexity Theory Statement

**Theorem (Computational Trichotomy):** Every computational problem in NP classifies into exactly one of three structural categories:

1. **Tractable (P):** The problem admits a polynomial-time algorithm. Computation traces terminate within $O(n^k)$ resources for some fixed $k$.

2. **Intermediate (NP-intermediate):** If P $\neq$ NP, there exist problems in NP that are neither in P nor NP-complete. These problems concentrate computational resources in identifiable substructures but admit barrier-based resolution.

3. **Hard (NP-complete):** The problem is NP-complete. Computational resources genuinely escape polynomial bounds, and the problem contains a self-similar hard kernel resistant to kernelization.

**Formal Statement:** Let $L \in \text{NP}$ be a decision problem. Define:
- $T_L(n) := \min\{t : \text{some algorithm decides } L \text{ in time } t \text{ on inputs of size } n\}$
- $K_L(n) := \min\{k : L \text{ admits a kernelization to size } k \text{ on inputs of size } n\}$

Then exactly one of the following holds:

| Outcome | Resource Behavior | Kernel Structure |
|---------|-------------------|------------------|
| **P** (Dispersion) | $T_L(n) = O(n^k)$ for some $k$ | $K_L(n) = O(1)$ or trivial |
| **NP-intermediate** (Concentration with barriers) | $T_L(n) = \omega(n^k)$ for all $k$, but $L \not\leq_p \text{SAT}$ | $K_L(n) = O(f(k) \cdot n^c)$ for FPT |
| **NP-complete** (Genuine singularity) | $L \leq_p \text{SAT}$ | $K_L(n) \geq n^{1-\epsilon}$ unless coNP $\subseteq$ NP/poly |

---

## Terminology Translation Table

| Hypostructure Term | Complexity Theory Equivalent |
|--------------------|------------------------------|
| Trajectory $u(t)$ | Computation trace $\mathcal{C}(x)$ on input $x$ |
| Breakdown time $T_*$ | Resource exhaustion point (time/space bound exceeded) |
| Energy functional $\Phi$ | Computational resource measure (time, space, circuit depth) |
| Energy dispersion $\Phi_* = 0$ | Problem in P: resources disperse polynomially |
| Energy concentration $\Phi_* > 0$ | Resource concentration in subproblem structure |
| Genuine singularity | NP-complete: problem genuinely hard |
| Profile extraction | Kernelization: extracting the hard instance core |
| Limiting profile $v^*$ | Irreducible hard kernel $\kappa(x)$ |
| Symmetry group $G$ | Problem automorphisms (permutations preserving structure) |
| Interface permits | Structural tractability conditions |
| Mode D.D (Dispersion-Decay) | Problem in P with polynomial decay |
| Mode C.E (Concentration-Escape) | NP-complete with escaping complexity |
| Global Regularity | FPT or XP tractability via parameterized complexity |

---

## Proof Sketch

### Setup: Computational Complexity Framework

**Definitions:**

1. **P (Polynomial Time):** The class of decision problems solvable by a deterministic Turing machine in time $O(n^k)$ for some constant $k$.

2. **NP (Nondeterministic Polynomial Time):** Problems with polynomial-time verifiable certificates.

3. **NP-intermediate:** Problems in NP $\setminus$ P that are not NP-complete (if P $\neq$ NP).

4. **Kernelization:** A polynomial-time algorithm that transforms instance $(x, k)$ to $(x', k')$ where:
   - $|x'| + k' \leq f(k)$ for some computable $f$
   - $(x, k) \in L \Leftrightarrow (x', k') \in L$

5. **Computation Trace:** For input $x$ and algorithm $\mathcal{A}$, the trace $\mathcal{C}_\mathcal{A}(x) = (c_0, c_1, \ldots, c_T)$ is the sequence of configurations from initial state $c_0$ to halting state $c_T$.

**Resource Functional (Energy Analogue):**

Define the computational energy of a trace at step $t$:
$$\Phi(c_t) := \text{Space}(c_t) + \log(\text{RemainingTime}(c_t))$$

This measures the "computational potential" remaining in the trace. For efficient algorithms, $\Phi$ decreases steadily (dissipation). For hard problems, $\Phi$ concentrates in irreducible subcomputations.

---

### Step 1: Resource Dichotomy

**Claim:** For any problem $L \in \text{NP}$ and optimal algorithm $\mathcal{A}$, either:
- **Dispersion:** $T_\mathcal{A}(n) = O(n^k)$ for some $k$ (polynomial resources)
- **Concentration:** $T_\mathcal{A}(n) = \omega(n^k)$ for all $k$ (super-polynomial resources)

**Proof:**

By definition of asymptotic complexity, for any algorithm $\mathcal{A}$ and problem $L$:

$$\lim_{n \to \infty} \frac{T_\mathcal{A}(n)}{n^k} \in \{0, c, \infty\} \text{ for each } k$$

**Case 1 (Dispersion):** If there exists $k$ such that $\limsup_{n \to \infty} T_\mathcal{A}(n)/n^k < \infty$, then $T_\mathcal{A}(n) = O(n^k)$.

This corresponds to Mode D.D in the hypostructure framework: computational resources "disperse" polynomially, the problem is tractable, and the algorithm terminates efficiently. The energy functional $\Phi$ decays to zero as the computation progresses.

**Certificate Produced:** $(L \in \text{P}, k, \mathcal{A})$ where $\mathcal{A}$ is the polynomial-time algorithm.

**Case 2 (Concentration):** If for all $k$, $\limsup_{n \to \infty} T_\mathcal{A}(n)/n^k = \infty$, then $T_\mathcal{A}(n) = \omega(n^k)$ for all $k$.

Resources concentrate in the problem structure. We proceed to Step 2 to extract the hard kernel.

**Connection to Lions' Dichotomy:** This mirrors the concentration-compactness dichotomy of Lions (1984). In the computational setting:
- **Vanishing (Dispersion):** Computation spreads across polynomial resources, no bottleneck.
- **Concentration:** Computation bottlenecks at identifiable hard substructures.

---

### Step 2: Profile Decomposition = Subproblem Identification

**Assumption:** We are in Case 2 (concentration), so $T_\mathcal{A}(n) = \omega(n^k)$ for all $k$.

**Claim (Kernelization Extraction):** For parameterized problems $(L, k) \in \text{NP}$, there exists a decomposition:
$$x = \kappa(x) \oplus r(x)$$

where:
- $\kappa(x)$ is the **hard kernel** (irreducible core)
- $r(x)$ is the **polynomial residual** (easily reducible part)
- $|\kappa(x)| = K_L(|x|, k)$ is the kernel size

**Proof (Profile Extraction via Reduction Rules):**

**Step 2.1 (Reduction Sequence):**

Define a sequence of reduction rules $\rho_1, \rho_2, \ldots, \rho_m$ where each $\rho_i$ is a polynomial-time transformation preserving the solution:
$$\rho_i: (x, k) \to (x', k') \text{ with } |x'| < |x| \text{ or } k' < k$$

Apply reductions exhaustively until no rule applies:
$$x \xrightarrow{\rho_{i_1}} x_1 \xrightarrow{\rho_{i_2}} x_2 \xrightarrow{\rho_{i_3}} \cdots \xrightarrow{\rho_{i_t}} \kappa(x)$$

The fixed point $\kappa(x)$ is the **irreducible kernel**.

**Step 2.2 (Kernel Compactness Modulo Symmetries):**

Let $\text{Aut}(L)$ be the automorphism group of problem $L$ (permutations of the input that preserve membership). The kernel is unique up to $\text{Aut}(L)$-equivalence:
$$g \cdot \kappa(x) \cong \kappa(g \cdot x) \text{ for } g \in \text{Aut}(L)$$

This is the computational analogue of profile extraction modulo symmetry group $G$ in the Bahouri-Gerard decomposition.

**Step 2.3 (Non-Triviality):**

If $\kappa(x) = \emptyset$ (trivial kernel), then $x$ was polynomial-time solvable, contradicting Case 2. Hence $|\kappa(x)| \geq 1$.

**Step 2.4 (Profile Decomposition):**

For NP problems, the Cook-Levin theorem implies any instance $x$ decomposes as:
$$\phi_x = \bigwedge_{j=1}^{J} \phi^{(j)}(x) \wedge \psi_{\text{easy}}(x)$$

where:
- $\phi^{(j)}$ are **hard clauses** (the "bubbles" in Bahouri-Gerard)
- $\psi_{\text{easy}}$ is the polynomial-time solvable residual
- Energy decouples: $|\phi_x| = \sum_j |\phi^{(j)}| + |\psi_{\text{easy}}| + o(1)$

**Certificate Produced:** $(\kappa(x), |\kappa(x)|, \text{Aut}(L), \{\rho_i\})$

---

### Step 3: Permit Classification

We now classify the extracted kernel $\kappa(x)$ according to whether it satisfies tractability permits.

**Interface Permits for Computational Problems:**

| Permit | Complexity Condition | Interpretation |
|--------|---------------------|----------------|
| $\mathrm{SC}_\lambda$ (Subcriticality) | $K_L(n,k) = O(f(k))$ | Kernel size depends only on parameter |
| $\mathrm{SC}_{\partial c}$ (Parameter Stability) | $k' \leq k$ under reductions | Parameter non-increasing |
| $\mathrm{Cap}_H$ (Capacity) | $\text{tw}(\kappa) \leq f(k)$ | Bounded treewidth/pathwidth |
| $\mathrm{LS}_\sigma$ (Local Stability) | Local search converges | No exponential plateaus |
| $\mathrm{TB}_\pi$ (Topological Boundary) | $\kappa$ has bounded genus | Topological tractability |

**Trichotomy Split:**

---

#### Case 3.1: All Permits Satisfied - Fixed-Parameter Tractability

**Assumption:** The kernel $\kappa(x)$ satisfies:
- $K_L(n,k) = O(f(k))$ for computable $f$ (polynomial kernelization)
- Bounded treewidth: $\text{tw}(\kappa) \leq g(k)$
- Parameter stability under reductions

**Theorem (FPT via Kernel Bounds):**

If $(L, k)$ admits a kernelization to size $f(k)$ and bounded treewidth $g(k)$, then:
$$T_L(n, k) = O(n^c + h(k))$$

for some computable $h$, placing $L$ in FPT (fixed-parameter tractable).

**Proof (Kenig-Merle Rigidity Analogue):**

1. **Kernelization Phase:** Reduce $(x, k)$ to $(\kappa(x), k')$ in time $O(n^c)$.

2. **Bounded Search Phase:** Since $|\kappa| \leq f(k)$ and $\text{tw}(\kappa) \leq g(k)$, apply Courcelle's theorem or dynamic programming:
   $$T_{\text{solve}}(\kappa) = O(f(k)^{g(k)}) = h(k)$$

3. **Total Time:** $T_L(n,k) = O(n^c) + h(k) = O(n^c + h(k))$

**Rigidity Interpretation:** The permits form "barriers" preventing the problem from being NP-hard. Just as Kenig-Merle rigidity forces dispersive PDEs with bounded energy to scatter, computational permits force parameterized problems to be FPT.

**Modes Covered:** S.E (Subcritical-Equilibration), C.D (Concentration-Dispersion), T.E (Topological-Extension)

**Certificate Produced:** $(L \in \text{FPT}, f(k), g(k), \mathcal{A}_{\text{FPT}})$

---

#### Case 3.2: At Least One Permit Violated - NP-Hardness

**Assumption:** The kernel $\kappa(x)$ violates at least one permit:
- $K_L(n,k) \geq n^{1-\epsilon}$ for all $\epsilon > 0$ (no polynomial kernelization), or
- Unbounded treewidth: $\text{tw}(\kappa) = \Omega(n)$, or
- Self-similar structure: $\kappa$ contains a reduction from SAT

**Theorem (NP-Completeness via Permit Violation):**

If $\kappa(x)$ has:
1. **Supercritical scaling:** $K_L(n,k) \geq n^{1-\epsilon}$ implies no polynomial kernelization unless coNP $\subseteq$ NP/poly
2. **Unbounded structural width:** $\text{tw}(\kappa) = \omega(1)$ implies exponential blowup in dynamic programming
3. **Self-similarity:** SAT $\leq_p L$ implies NP-completeness

Then $L$ is NP-complete (assuming P $\neq$ NP).

**Proof (Struwe Singularity Analysis Analogue):**

**Self-Similar Blowup Construction:**

For NP-complete problems, the kernel exhibits **self-similar structure** under scaling. Define the scaling operation:
$$\lambda \cdot \kappa := \text{amplification of } \kappa \text{ by factor } \lambda$$

For SAT and NP-complete problems, the reduction $\text{SAT} \leq_p L$ implies:
$$L(\lambda \cdot x) \text{ contains } L(x) \text{ as a subproblem}$$

This is the computational analogue of Type II blowup in Struwe's analysis: the hard kernel is **scale-invariant** and cannot be reduced.

**Ladner's Theorem Connection:**

Ladner's theorem (1975) proves: If P $\neq$ NP, then NP-intermediate problems exist. The construction uses a **padding argument** that creates problems with intermediate resource profiles:

$$L_{\text{Ladner}} := \{x \mid x \in \text{SAT} \text{ and } |x| \geq f(|x|)\}$$

where $f$ is a slowly-growing function. This problem:
- Is not in P (inherits SAT hardness for large inputs)
- Is not NP-complete (padding breaks reductions)

**Permit Violation Catalog:**

| Violated Permit | Consequence | Hardness Type |
|-----------------|-------------|---------------|
| $K_{\mathrm{SC}_\lambda}^-$ | No poly kernel | Kernelization lower bound |
| $K_{\mathrm{Cap}_H}^-$ | Unbounded width | Exponential DP |
| $K_{\mathrm{LS}_\sigma}^-$ | Search instability | PPAD-hardness |
| $K_{\mathrm{TB}_\pi}^-$ | Topological obstruction | Genus lower bound |

**Certificate Produced:** $(L \in \text{NP-complete}, \text{SAT} \leq_p L, \text{violated permits})$

---

### Step 4: Kenig-Merle Rigidity = Structural Classification

**Theorem (No "Almost Hard" Regime):**

For structured computational problems (those with well-defined parameters and decompositions), there is no continuous spectrum between P and NP-complete. The trichotomy is **rigid**:

1. **P:** Admits polynomial algorithm (all permits satisfied, trivial kernel)
2. **FPT/XP:** Admits parameterized algorithm (permits satisfied with parameter dependence)
3. **NP-complete:** No polynomial algorithm unless P = NP (permits violated, self-similar kernel)

**Proof (Dichotomy Theorem Analogue):**

This rigidity is the computational analogue of Kenig-Merle's "no intermediate regime" for energy-critical NLS. The mechanism is:

**Schaefer's Dichotomy (1978):** For Boolean constraint satisfaction problems (CSPs) with constraint language $\Gamma$:
$$\text{CSP}(\Gamma) \in \text{P} \Leftrightarrow \Gamma \text{ is tractable (Schaefer conditions)}$$
$$\text{CSP}(\Gamma) \text{ is NP-complete} \Leftrightarrow \Gamma \text{ is not tractable}$$

There is no intermediate case.

**Feder-Vardi Conjecture (1998) / Bulatov-Zhuk Theorem (2017):**

Extended to all finite-domain CSPs: every CSP is either in P or NP-complete. The proof uses algebraic invariants (polymorphisms) as the structural barrier.

**Rigidity Mechanism:**

The polymorphism algebra $\text{Pol}(\Gamma)$ acts as the "symmetry group" $G$:
- If $\text{Pol}(\Gamma)$ contains certain operations (Mal'cev, near-unanimity), the CSP is tractable.
- Otherwise, the CSP expresses all of Boolean logic and is NP-complete.

This is the computational Kenig-Merle rigidity: the algebraic structure forces a dichotomy, leaving no room for "almost tractable" problems.

---

### Certificate Construction

For each outcome, we produce an explicit certificate:

**Mode P (Dispersion):**
```
K_P = {
  mode: "Dispersion",
  mechanism: "Polynomial Algorithm",
  evidence: {
    algorithm: A,
    time_bound: O(n^k),
    correctness_proof: pi,
    kernel_size: O(1)
  },
  literature: "Cobham-Edmonds Thesis"
}
```

**Mode FPT (Global Regularity via Barriers):**
```
K_FPT = {
  mode: "Global_Regularity",
  mechanism: "Parameterized_Tractability",
  evidence: {
    parameter: k,
    kernel_size: f(k),
    treewidth_bound: g(k),
    algorithm: A_FPT,
    time_bound: O(n^c + h(k)),
    permit_certificates: {
      SC_lambda: "poly_kernel",
      Cap_H: "bounded_treewidth",
      LS_sigma: "convergent_search"
    }
  },
  literature: "Downey-Fellows 1999, Cygan et al. 2015"
}
```

**Mode NP-complete (Genuine Singularity):**
```
K_NPC = {
  mode: "Genuine_Singularity",
  mechanism: "NP-Completeness",
  evidence: {
    reduction: "SAT <=_p L",
    violated_permits: {
      SC_lambda: "no_poly_kernel (unless coNP in NP/poly)",
      Cap_H: "unbounded_treewidth",
      self_similarity: "SAT_embedding"
    },
    kernel_lower_bound: "n^{1-epsilon}",
    hardness_class: "NP-complete"
  },
  literature: "Cook 1971, Karp 1972, Ladner 1975"
}
```

---

## Connections to Classical Results

### 1. Ladner's Theorem (NP-intermediate exists if P $\neq$ NP)

**Statement:** If P $\neq$ NP, there exists a problem $L \in \text{NP} \setminus (\text{P} \cup \text{NP-complete})$.

**Connection:** Ladner's theorem establishes the existence of the "concentration with barriers" regime. The constructed problem:
- Has super-polynomial complexity (concentration)
- Cannot be reduced to SAT (barrier prevents full singularity)
- Is not in P (non-trivial kernel)

This is the computational analogue of Mode S.E/C.D: energy concentrates but structural constraints prevent genuine blowup.

**Certificate Correspondence:**
- $K_{C_\mu}^+ = 1$ (concentration): Problem is super-polynomial
- All structural permits satisfied: No SAT reduction exists
- $K_{\text{Reg}}$: Problem is NP-intermediate

### 2. Kernelization Theory (FPT)

**Statement:** A parameterized problem $(L, k)$ is FPT if and only if it admits a kernelization.

**Connection:** Kernelization is the computational analogue of profile extraction in concentration-compactness. The kernel $\kappa(x)$ is the "limiting profile" $v^*$ of the computation:
- Kernel size $|\kappa| = f(k)$ corresponds to profile energy $\Phi(v^*)$
- Polynomial-time kernelization corresponds to profile convergence modulo symmetries
- FPT solvability corresponds to global regularity via subcritical scaling

**Key Results:**
- **Vertex Cover:** Kernel of size $2k$ (quadratic dependence on parameter)
- **k-Path:** Kernel of size $O(k^2)$ via color-coding
- **Dominating Set:** No polynomial kernel unless coNP $\subseteq$ NP/poly

### 3. Dichotomy Theorems (Schaefer, Feder-Vardi, Bulatov-Zhuk)

**Statement:** For constraint satisfaction problems over finite domains, every CSP is either in P or NP-complete.

**Connection:** Dichotomy theorems are the computational Kenig-Merle rigidity:
- The polymorphism algebra $\text{Pol}(\Gamma)$ acts as the symmetry group $G$
- Tractability conditions (Mal'cev, near-unanimity) are interface permits
- The dichotomy is **rigid**: no intermediate regime exists for structured CSPs

**Schaefer's Six Tractable Cases (1978):**
1. 0-valid (all-zeros is a solution)
2. 1-valid (all-ones is a solution)
3. Horn (implications)
4. Dual-Horn (co-implications)
5. Affine (XOR-SAT, linear algebra over GF(2))
6. Bijunctive (2-SAT)

**Bulatov-Zhuk Theorem (2017):**

For any finite constraint language $\Gamma$:
$$\text{CSP}(\Gamma) \in \text{P} \Leftrightarrow \text{Pol}(\Gamma) \text{ contains a weak near-unanimity operation}$$

This algebraic criterion is the computational analogue of the Kenig-Merle energy threshold.

---

## Quantitative Bounds

### Energy Threshold (Universal)

**Critical Complexity:**
$$T_c := \inf\{T : \exists L \in \text{NP} \text{ with } T_L(n) = \Theta(T) \text{ and } L \text{ NP-complete}\}$$

Under standard assumptions (ETH - Exponential Time Hypothesis):
$$T_c = 2^{\Omega(n)} \text{ for 3-SAT}$$

### Dispersion Regime (P)

For $L \in \text{P}$ with optimal algorithm $\mathcal{A}$:
$$T_\mathcal{A}(n) = O(n^k) \text{ for some } k \leq \text{poly-degree}(L)$$

**Time Hierarchy Theorem:** For constructible $f, g$ with $f(n) \log f(n) = o(g(n))$:
$$\text{DTIME}(f(n)) \subsetneq \text{DTIME}(g(n))$$

### Concentration Regime (FPT)

For $(L, k) \in \text{FPT}$:
$$T_L(n, k) = O(f(k) \cdot n^c) \text{ for computable } f$$

**Classification:**
- **FPT:** $f(k) \cdot n^{O(1)}$
- **XP:** $n^{f(k)}$
- **para-NP:** NP-hard for fixed $k$

### Singularity Regime (NP-complete)

For $L$ NP-complete, under ETH:
$$T_L(n) = 2^{\Omega(n)}$$

**Sparsification Lemma (Impagliazzo-Paturi-Zane 2001):**
$$\text{3-SAT on } n \text{ variables requires } 2^{\delta n} \text{ time for some } \delta > 0$$

---

## Conclusion

The KRNL-Trichotomy theorem translates to complexity theory as a structural classification of computational problems:

1. **Mode D.D (Dispersion) = P:** Problems where computational resources disperse polynomially. The algorithm "scatters" the computation across polynomial time, and no hard kernel remains.

2. **Mode S.E/C.D (Global Regularity) = NP-intermediate/FPT:** Problems where resources concentrate but structural barriers (kernelization, bounded width, algebraic constraints) prevent NP-completeness. The Kenig-Merle rigidity analogue forces these problems to be tractable via parameterized complexity.

3. **Mode C.E (Genuine Singularity) = NP-complete:** Problems where the hard kernel is self-similar and irreducible. The SAT reduction witnesses "energy escape" into the universal NP-complete structure.

**Physical Interpretation (Computational Analogue):**

- **D.D:** Computation has insufficient "energy density" to sustain hard subproblems. Disperses into polynomial time.
- **S.E/C.D:** Computation concentrates but structural constraints (permits) prevent blow-up. Reaches tractable equilibrium via FPT algorithms.
- **C.E:** Computation has concentrated energy and evades all tractability barriers. Genuine hardness - exponential blowup required.

**The Trichotomy Certificate:**

$$K_{\text{Trichotomy}} = \begin{cases}
K_{\text{P}} & \text{if } T_L(n) = O(n^k) \\
K_{\text{FPT}} & \text{if } T_L(n,k) = O(f(k) \cdot n^c) \text{ with permits satisfied} \\
K_{\text{NPC}} & \text{if } \text{SAT} \leq_p L \text{ with permits violated}
\end{cases}$$

---

## Literature

1. **Cook, S. A. (1971).** "The Complexity of Theorem-Proving Procedures." STOC. *Establishes NP-completeness.*

2. **Karp, R. M. (1972).** "Reducibility Among Combinatorial Problems." *21 NP-complete problems.*

3. **Ladner, R. E. (1975).** "On the Structure of Polynomial Time Reducibility." JACM. *NP-intermediate existence.*

4. **Schaefer, T. J. (1978).** "The Complexity of Satisfiability Problems." STOC. *Boolean CSP dichotomy.*

5. **Downey, R. G. & Fellows, M. R. (1999).** *Parameterized Complexity.* Springer. *FPT foundations.*

6. **Impagliazzo, R., Paturi, R., & Zane, F. (2001).** "Which Problems Have Strongly Exponential Complexity?" JCSS. *ETH and sparsification.*

7. **Bulatov, A. A. (2017).** "A Dichotomy Theorem for Nonuniform CSPs." FOCS. *General CSP dichotomy.*

8. **Zhuk, D. (2020).** "A Proof of the CSP Dichotomy Conjecture." JACM. *Independent CSP dichotomy proof.*

9. **Cygan, M. et al. (2015).** *Parameterized Algorithms.* Springer. *Modern FPT techniques.*

10. **Lions, P.-L. (1984).** "The Concentration-Compactness Principle." *Annales IHP.* *Original concentration-compactness.*

11. **Kenig, C. E. & Merle, F. (2006).** "Global Well-Posedness, Scattering and Blow-Up for the Energy-Critical NLS." *Inventiones.* *Rigidity theorem.*
