---
title: "FACT-Transport - Complexity Theory Translation"
---

# FACT-Transport: Equivalence Transport in Complexity Theory

## Original Theorem (Hypostructure Context)

The FACT-Transport theorem establishes that for any type $T$, there exists a factory generating:
1. **Equivalence moves** $\mathrm{Eq}_1^T, \ldots, \mathrm{Eq}_k^T$ (scaling, symmetry, gauge, modulation) with comparability bounds
2. **Transport lemmas** $T_1^T, \ldots, T_6^T$ that carry properties across equivalences
3. **YES$^\sim$ production rules** for generating equivalent-YES verdicts
4. **Promotion rules** (immediate and a-posteriori) lifting YES$^\sim$ to YES$^+$

**Core insight:** When two structures are equivalent, certificates for one can be transported to certificates for the other via natural transformations.

**Original Reference:** {prf:ref}`mt-fact-transport`

---

## Complexity Theory Statement

**Theorem (Equivalence Transport for Computational Problems):** Let $\mathcal{P}$ be a decision problem and let $\sim$ be an equivalence relation on instances such that $x \sim y \Rightarrow \mathcal{P}(x) = \mathcal{P}(y)$ (isomorphism-closed). Then:

1. **Certificate Transport:** If $w$ is a certificate (witness) for $\mathcal{P}(x) = \text{YES}$ and $\sigma: x \to y$ is an isomorphism witnessing $x \sim y$, then $\sigma(w)$ is a certificate for $\mathcal{P}(y) = \text{YES}$.

2. **Reduction Lifting:** Polynomial-time reductions between equivalent formulations preserve certificate structure.

3. **Canonicalization:** There exists a canonical form $\mathrm{canon}(x)$ such that $x \sim y \iff \mathrm{canon}(x) = \mathrm{canon}(y)$.

4. **Promotion:** Certificates valid "up to equivalence" can be promoted to exact certificates when the equivalence is efficiently computable.

**Formal Statement:** Let $(\mathcal{X}, \sim)$ be a set of problem instances with equivalence relation, and let $\mathcal{P}: \mathcal{X} \to \{\text{YES}, \text{NO}\}$ be $\sim$-invariant. Define:

- **Isomorphism:** $\mathrm{Iso}(x, y) = \{\sigma \in \mathrm{Aut}(\mathcal{X}) : \sigma(x) = y\}$
- **Certificate functor:** $\mathcal{W}: \mathcal{X} \to \mathbf{Set}$ where $\mathcal{W}(x)$ is the set of certificates for $x$
- **Transport map:** For $\sigma \in \mathrm{Iso}(x, y)$, define $T_\sigma: \mathcal{W}(x) \to \mathcal{W}(y)$

Then:

| Property | Mathematical Statement |
|----------|----------------------|
| **Functoriality** | $T_{\sigma \circ \tau} = T_\sigma \circ T_\tau$ and $T_{\mathrm{id}} = \mathrm{id}$ |
| **Certificate Preservation** | $w \in \mathcal{W}(x) \Rightarrow T_\sigma(w) \in \mathcal{W}(y)$ |
| **Verification Invariance** | $\mathrm{Verify}(x, w) = \mathrm{Verify}(y, T_\sigma(w))$ |
| **Canonicity** | $\exists!$ canonical representative per equivalence class |

---

## Terminology Translation Table

| Hypostructure Concept | Complexity Theory Equivalent |
|----------------------|------------------------------|
| Type $T$ structural assumptions | Problem structure (graph, formula, etc.) |
| Equivalence $\mathrm{Eq}_i^T$ | Isomorphism relation on instances |
| Eq1 (Scaling) $u \sim_\lambda \lambda^\alpha u$ | Instance scaling (size normalization) |
| Eq2 (Symmetry) $u \sim_g g \cdot u$ | Automorphism action on instance |
| Eq3 (Gauge) $u \sim_\phi e^{i\phi} u$ | Representation change (basis, encoding) |
| Eq4 (Modulation) | Parameterized transformation family |
| Comparability bounds $C \cdot d_{\mathrm{Eq}}$ | Reduction overhead bounds |
| Transport lemma $T_i^T$ | Certificate lifting/transformation |
| Certificate $K_P^+(u)$ | Witness for $\mathcal{P}(x) = \text{YES}$ |
| Equivalence certificate $K_{\mathrm{Eq}}(u, u')$ | Isomorphism proof (bijection, permutation) |
| Transported certificate $K_P^\sim(u')$ | Lifted witness via isomorphism |
| YES$^+$ (definite YES) | Direct certificate verification |
| YES$^\sim$ (equivalent YES) | Certificate valid up to isomorphism |
| Immediate promotion | Efficiently computable equivalence |
| A-posteriori promotion | Later-discovered equivalence bounds |
| Univalence principle | Equivalent types have equivalent properties |
| Symmetry group $G$ | Automorphism group $\mathrm{Aut}(x)$ |
| Energy functional $\Phi$ | Complexity measure (size, depth) |

---

## Proof Sketch

### Setup: Isomorphism-Respecting Computation

**Definition (Isomorphism-Closed Problem):** A decision problem $\mathcal{P}$ on structured objects is isomorphism-closed if:
$$x \cong y \Rightarrow \mathcal{P}(x) = \mathcal{P}(y)$$

Most natural computational problems are isomorphism-closed: SAT (variable renaming), Graph Coloring (vertex relabeling), Hamiltonian Path (vertex permutation), etc.

**Definition (Certificate Transport):** For an NP problem $\mathcal{P}$ with verification relation $R(x, w)$, the transport property states:
$$R(x, w) \land \sigma: x \cong y \Rightarrow R(y, \sigma(w))$$

where $\sigma(w)$ applies the isomorphism $\sigma$ to the certificate $w$.

**Definition (Reduction Equivalence):** Two problem formulations $\mathcal{P}_1$ and $\mathcal{P}_2$ are reduction-equivalent if:
$$\mathcal{P}_1 \leq_p \mathcal{P}_2 \quad \text{and} \quad \mathcal{P}_2 \leq_p \mathcal{P}_1$$

with polynomial-time computable reductions that preserve certificates.

---

### Step 1: Equivalence Instantiation (Isomorphism Types)

**Claim:** For each problem domain, there exists a canonical set of isomorphism types.

**Proof:**

**Eq1 (Scaling Equivalence):**
In complexity theory, scaling corresponds to problem instance size normalization:
- **Instance padding:** Adding dummy elements that don't affect the answer
- **Encoding changes:** Different representations of the same mathematical object
- **Parameter scaling:** Multiplying numerical parameters by constants

For example, in SAT:
$$\varphi(x_1, \ldots, x_n) \sim \varphi(x_1, \ldots, x_n) \land (y \lor \neg y)$$

The comparability bound ensures polynomial-time conversion between scaled versions.

**Eq2 (Symmetry Equivalence):**
The automorphism group action on instances:
- **Graph problems:** $\sigma \in S_n$ acting on vertices
- **Boolean formulas:** Variable permutations
- **Algebraic structures:** Group/ring automorphisms

For graphs $G = (V, E)$:
$$G \sim_\sigma G' \iff \exists \sigma \in S_{|V|}: (u,v) \in E \Leftrightarrow (\sigma(u), \sigma(v)) \in E'$$

**Eq3 (Gauge/Representation Equivalence):**
Different encodings of the same underlying problem:
- **Adjacency matrix vs. edge list** for graphs
- **CNF vs. DNF** for Boolean formulas
- **Different bases** for linear algebra problems

These correspond to "gauge freedom" in the hypostructure.

**Eq4 (Parameterized Equivalence):**
Families of transformations indexed by continuous parameters:
- **Weighted versions:** $w_e \mapsto \alpha \cdot w_e$ for edge weights
- **Threshold shifts:** $k$-colorability vs. $(k+c)$-colorability
- **Approximation ratios:** $(1+\epsilon)$-approximation families

**Comparability Bounds:** The transport error is bounded:
$$|\mathrm{Cost}(\mathcal{P}, x) - \mathrm{Cost}(\mathcal{P}, y)| \leq C \cdot d_{\mathrm{Eq}}(x, y)$$

where $\mathrm{Cost}$ measures computational complexity (circuit size, time, etc.) and $d_{\mathrm{Eq}}$ measures the "distance" of the equivalence transformation. $\square$

---

### Step 2: Transport Soundness (Certificate Lifting)

**Claim:** If $w$ certifies $\mathcal{P}(x) = \text{YES}$ and $\sigma: x \cong y$, then $\sigma(w)$ certifies $\mathcal{P}(y) = \text{YES}$.

**Proof of Soundness:**

Let $R(x, w)$ be the polynomial-time verification relation for NP problem $\mathcal{P}$:
$$\mathcal{P}(x) = \text{YES} \iff \exists w: R(x, w)$$

**Step 2a (Structure Preservation):** The isomorphism $\sigma$ defines a bijection on the components of $x$ and $w$. Since $R$ is defined in terms of structural relationships (e.g., "edge $(i,j)$ exists" or "clause $c$ is satisfied"), and $\sigma$ preserves all such relationships:
$$R(x, w) \Rightarrow R(\sigma(x), \sigma(w)) = R(y, \sigma(w))$$

**Step 2b (Transport Map Definition):** Define the transport map $T_\sigma: \mathcal{W}(x) \to \mathcal{W}(y)$ by:
$$T_\sigma(w) := \sigma(w)$$

This is well-defined because:
1. $\sigma$ is a bijection (isomorphism)
2. $R$ is isomorphism-invariant (natural problem)
3. Certificate structure is preserved (same encoding)

**Step 2c (Error Bound):** The transported certificate carries:
- The original witness transformed by $\sigma$
- The isomorphism certificate $\sigma$ itself
- Bound on transport overhead: $|T_\sigma(w)| = |w| + O(|\sigma|)$

This follows the univalence principle: equivalent structures have equivalent properties. $\square$

---

### Step 3: YES$^\sim$ Production (Equivalence-Relative Certificates)

**Claim:** Certificates valid under equivalence can be systematically produced.

**Production Rules:**

The YES$^\sim$ (equivalent-YES) production rule is:
$$\frac{R(x, w) \quad x \sim y \quad \sigma \in \mathrm{Iso}(x,y)}{R^\sim(y, T_\sigma(w))}$$

**Interpretation:**
- If we have a certificate $w$ for $x$
- And an isomorphism $\sigma: x \to y$
- Then we can construct an "equivalent certificate" for $y$

**Computational Realization:**

```
TransportCertificate(x, w, sigma, y):
    Input: Instance x, certificate w, isomorphism sigma, target y
    Precondition: R(x, w) = true, sigma(x) = y

    1. Transform witness: w' := sigma(w)
    2. Attach equivalence proof: K_eq := (x, y, sigma)
    3. Construct transported certificate:
       K_transport := (w', K_eq, R(x,w))

    Output: (YES^~, K_transport)
```

**Verification:**
$$\mathrm{Verify}_{\sim}(y, K_{\mathrm{transport}}) := \mathrm{Verify}(y, w') \land \mathrm{ValidIso}(\sigma, x, y)$$

The verification checks both the transformed witness and the isomorphism validity. $\square$

---

### Step 4: Promotion Rules (Lifting Equivalent to Direct Certificates)

**Claim:** YES$^\sim$ promotes to YES$^+$ under bounded equivalence parameters.

**Immediate Promotion:**

If the equivalence is "small" (close to identity), the equivalent certificate becomes a direct certificate:

$$\text{dist}(\sigma, \mathrm{id}) < \epsilon_{\mathrm{prom}} \Rightarrow K^\sim \leadsto K^+$$

**Examples:**
- **Graph relabeling:** If $\sigma$ is a "local" permutation (affects only $O(\log n)$ vertices), verification is direct
- **Formula simplification:** If the equivalence only removes redundant clauses
- **Encoding change:** If both encodings are standard and efficiently interconvertible

**A-Posteriori Promotion:**

Later analysis may reveal that an equivalence was actually small:

1. Initially: Certificate $K^\sim$ with unknown equivalence distance
2. Later gate provides bound: $\text{dist}(\sigma, \mathrm{id}) < \epsilon$
3. Promotion: $K^\sim \leadsto K^+$

**Complexity-Theoretic Interpretation:**
- **Immediate:** Equivalence computed in time $O(|x|)$
- **A-posteriori:** Equivalence discovered during later computation phases

**Promotion Thresholds:**

| Equivalence Type | Promotion Threshold $\epsilon_{\mathrm{prom}}$ | Condition |
|-----------------|---------------------------------------------|-----------|
| Vertex permutation | $|\mathrm{supp}(\sigma)| < k$ | $k$-local isomorphism |
| Variable renaming | Consistent renaming | Syntactic equivalence |
| Encoding change | Polynomial conversion | Standard encoding pair |
| Scaling | $|\lambda - 1| < \delta$ | Near-identity scaling |

$\square$

---

### Step 5: Completeness of Equivalence Library

**Claim:** The equivalence library is complete for well-structured problem classes.

**Proof:**

For problem $\mathcal{P}$ on domain $\mathcal{X}$ with intrinsic equivalence $\sim_{\mathcal{P}}$:
$$\forall x, y \in \mathcal{X}: x \sim_{\mathcal{P}} y \Rightarrow \exists i: x \sim_{\mathrm{Eq}_i} y$$

**Completeness for Standard Classes:**

1. **Graph Problems:** Completeness follows from the classification of graph automorphisms (vertex permutations exhaust equivalences for simple graphs)

2. **Boolean Formulas:** Equivalences are captured by variable permutations plus logical equivalence (SAT-equivalence)

3. **Linear Algebra:** Equivalences are basis changes (captured by $GL_n$ action)

4. **Algebraic Structures:** Group/ring automorphisms provide complete equivalence

**Connection to Symmetry Classification:** The completeness of equivalence libraries follows from the classification of symmetries (cf. Olver's symmetry classification for differential equations, here applied to discrete structures). $\square$

---

## Connections to Graph Isomorphism

### The Graph Isomorphism Problem

**Problem (GI):** Given graphs $G_1, G_2$, decide if $G_1 \cong G_2$.

**Connection to FACT-Transport:**

| Transport Concept | GI Instantiation |
|-------------------|------------------|
| Equivalence $x \sim y$ | Graph isomorphism $G_1 \cong G_2$ |
| Isomorphism certificate $K_{\mathrm{Eq}}$ | Bijection $\sigma: V_1 \to V_2$ |
| Transport map $T_\sigma$ | Apply $\sigma$ to vertex-based witnesses |
| Comparability bound | $O(n^2)$ edge comparisons |
| Canonical form | Canonical labeling (Babai et al.) |

### Certificate Transport for Graph Problems

**Example (Hamiltonian Path):**

If $G_1$ has Hamiltonian path $P = (v_1, v_2, \ldots, v_n)$ and $\sigma: G_1 \to G_2$ is an isomorphism:

$$T_\sigma(P) = (\sigma(v_1), \sigma(v_2), \ldots, \sigma(v_n))$$

is a Hamiltonian path in $G_2$.

**Example (Graph Coloring):**

If $G_1$ has $k$-coloring $c: V_1 \to \{1, \ldots, k\}$ and $\sigma: G_1 \to G_2$:

$$T_\sigma(c) = c \circ \sigma^{-1}: V_2 \to \{1, \ldots, k\}$$

is a $k$-coloring of $G_2$.

### Weisfeiler-Leman and Equivalence Detection

The **Weisfeiler-Leman algorithm** computes a canonical coloring refinement:

1. Initialize: Color each vertex by degree
2. Iterate: Refine colors based on multisets of neighbor colors
3. Stabilize: Stop when coloring is stable

**Connection to Transport:**
- WL computes a "partial canonicalization"
- If $\mathrm{WL}(G_1) \neq \mathrm{WL}(G_2)$: definitely $G_1 \not\cong G_2$
- If $\mathrm{WL}(G_1) = \mathrm{WL}(G_2)$: potential isomorphism (need further test)

The WL algorithm provides an efficiently computable approximation to the equivalence relation, enabling fast promotion of YES$^\sim$ to YES$^+$ when WL succeeds.

### Babai's Quasipolynomial Algorithm

**Theorem (Babai 2016):** Graph isomorphism is in quasipolynomial time: $O(2^{(\log n)^{O(1)}})$.

**Connection to Transport:**
- The algorithm computes canonical forms for equivalence classes
- Equivalence certificates (isomorphisms) are extractable from the algorithm
- Transport overhead is quasipolynomial in the worst case

---

## Connections to Equivalence Checking

### Program Equivalence

**Problem:** Given programs $P_1, P_2$, decide if $\forall x: P_1(x) = P_2(x)$.

**Transport Structure:**
- **Equivalence:** Semantic equivalence of programs
- **Certificate:** Bisimulation relation or proof of equivalence
- **Transport:** If $P_1 \equiv P_2$ and $P_1$ has property $\phi$, then $P_2$ has property $\phi$

### Circuit Equivalence

**Problem:** Given circuits $C_1, C_2$, decide if they compute the same function.

**Transport Structure:**
- **Equivalence:** Functional equivalence $\forall x: C_1(x) = C_2(x)$
- **Certificate:** BDD equivalence proof or SAT-based certificate
- **Canonical form:** Reduced ordered BDD (ROBDD)

### Isomorphism-Respecting Reductions

**Definition:** A reduction $f: \mathcal{P}_1 \to \mathcal{P}_2$ is isomorphism-respecting if:
$$x \sim_1 y \Rightarrow f(x) \sim_2 f(y)$$

**Theorem:** All polynomial-time reductions between natural problems are isomorphism-respecting.

**Proof Sketch:**
Natural problems have "uniform" definitions that don't depend on arbitrary labeling. Any reduction that exploits labeling would be artificial and non-robust. The structure-preserving nature of polynomial-time computation ensures isomorphism-respecting reductions. $\square$

---

## Certificate Construction

**Transport Certificate Structure:**

```
K_Transport := {
  mode: "Equivalence_Transport",
  mechanism: "Certificate_Lifting",

  source: {
    instance: x,
    certificate: w,
    verification: R(x, w) = true
  },

  equivalence: {
    target: y,
    isomorphism: sigma,
    type: Eq_i (scaling | symmetry | gauge | modulation),
    proof: K_Eq(x, y, sigma)
  },

  transport: {
    lifted_certificate: T_sigma(w),
    overhead_bound: |T_sigma(w)| <= |w| + O(|sigma|),
    verification_time: poly(|y|, |sigma|)
  },

  promotion: {
    status: YES^~ or YES^+,
    threshold: epsilon_prom,
    promotion_condition: dist(sigma, id) < epsilon_prom
  },

  complexity: {
    transport_time: O(|w| * |sigma|),
    verification_time: poly(|y|),
    space: O(|w| + |sigma|)
  }
}
```

**Verification Algorithm:**

```
VerifyTransportCertificate(K_Transport):
    1. Extract: x, w, sigma, y, w'
    2. Verify source: Check R(x, w) = true
    3. Verify isomorphism: Check sigma(x) = y
    4. Verify transport: Check w' = T_sigma(w)
    5. Verify target: Check R(y, w') = true
    6. Check promotion: If dist(sigma, id) < epsilon, upgrade to YES^+

    Return: (verified, YES^~ or YES^+)
```

---

## Quantitative Bounds

### Transport Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Apply isomorphism $\sigma$ to instance $x$ | $O(|x|)$ | $O(|x|)$ |
| Transform certificate $w$ | $O(|w|)$ | $O(|w|)$ |
| Verify isomorphism | $O(|x|)$ | $O(1)$ |
| Canonical form (graphs) | $O(2^{(\log n)^c})$ | $O(n^2)$ |
| WL stabilization | $O(n^2 \log n)$ | $O(n^2)$ |

### Reduction Overhead

For reduction $f: \mathcal{P}_1 \to \mathcal{P}_2$:

| Reduction Type | Instance Overhead | Certificate Overhead |
|---------------|-------------------|---------------------|
| Linear | $|f(x)| = O(|x|)$ | $|f(w)| = O(|w|)$ |
| Polynomial | $|f(x)| = |x|^{O(1)}$ | $|f(w)| = |w|^{O(1)}$ |
| Parsimonious | Preserves solution count | Bijective on certificates |

### Promotion Thresholds

| Equivalence Type | Threshold | Complexity to Check |
|-----------------|-----------|---------------------|
| Identity | $\sigma = \mathrm{id}$ | $O(1)$ |
| $k$-local permutation | $|\mathrm{supp}(\sigma)| \leq k$ | $O(k)$ |
| Bounded relabeling | Hamming distance $\leq d$ | $O(d)$ |
| Polynomial encoding | Standard encoding pairs | $O(|x|^{O(1)})$ |

---

## Theoretical Implications

### Univalence in Complexity Theory

The transport theorem embodies a form of the **univalence axiom** from Homotopy Type Theory:

$$\text{Isomorphic structures are identical for all computational purposes.}$$

This means:
1. Solving $\mathcal{P}$ on one representative suffices for the entire equivalence class
2. Lower bounds transfer across equivalent formulations
3. Certificate structures are preserved under equivalence

### Certificate Transfer and NP

**Theorem (Certificate Transfer Preserves NP):**
If $\mathcal{P} \in \mathrm{NP}$ and $\mathcal{Q}$ is reduction-equivalent to $\mathcal{P}$, then $\mathcal{Q} \in \mathrm{NP}$ with certificates of comparable size.

**Proof:**
Let $w$ be a certificate for $\mathcal{P}(x)$. The reduction $f: \mathcal{P} \to \mathcal{Q}$ gives:
- Instance transformation: $y = f(x)$
- Certificate transformation: $w' = f_w(w)$
- Both $f$ and $f_w$ are polynomial-time

The transport preserves certificate properties:
$$|w'| \leq p(|w|) \text{ for polynomial } p$$

Hence $\mathcal{Q} \in \mathrm{NP}$. $\square$

### Connections to Type Theory

The FACT-Transport theorem corresponds to several type-theoretic principles:

| Hypostructure | Type Theory | Complexity |
|--------------|-------------|------------|
| Transport lemma | Path induction | Reduction |
| Equivalence move | Type equivalence | Isomorphism |
| YES$^\sim$ | Propositional truncation | Existence claim |
| Promotion | Untruncation | Certificate extraction |
| Canonical form | Normal form | Canonical instance |

---

## Worked Example: SAT Certificate Transport

**Problem:** SAT - Boolean satisfiability

**Instance:** $\varphi = (x_1 \lor \neg x_2) \land (x_2 \lor x_3) \land (\neg x_1 \lor \neg x_3)$

**Certificate:** $w = \{x_1 = T, x_2 = T, x_3 = F\}$

**Equivalence:** Variable renaming $\sigma: x_1 \mapsto y_1, x_2 \mapsto y_3, x_3 \mapsto y_2$

**Target Instance:**
$$\varphi' = (y_1 \lor \neg y_3) \land (y_3 \lor y_2) \land (\neg y_1 \lor \neg y_2)$$

**Transport:**
$$T_\sigma(w) = \{y_1 = T, y_3 = T, y_2 = F\}$$

**Verification:**
- Clause 1: $y_1 \lor \neg y_3 = T \lor F = T$ ✓
- Clause 2: $y_3 \lor y_2 = T \lor F = T$ ✓
- Clause 3: $\neg y_1 \lor \neg y_2 = F \lor T = T$ ✓

**Certificate:**
```
K_Transport = {
  source: (phi, w, verified),
  equivalence: (sigma, variable_renaming),
  transport: ({y1=T, y3=T, y2=F}, overhead=O(n)),
  promotion: YES^+ (syntactic equivalence, immediate)
}
```

---

## Worked Example: Graph Coloring Transport

**Problem:** 3-Colorability

**Instance:** $G_1 = K_4$ (complete graph on 4 vertices $\{1,2,3,4\}$)

**Observation:** $K_4$ is NOT 3-colorable (needs 4 colors)

**Equivalence:** Vertex permutation $\sigma = (1\,2\,3\,4) \to (a, b, c, d)$

**Target Instance:** $G_2 = K_4$ on vertices $\{a,b,c,d\}$

**Transport (for non-3-colorability):**

The certificate for NO is the structure of $K_4$ itself:
- Any 3-coloring would require two vertices of the same color
- Those vertices are adjacent in $K_4$, violating coloring constraint

This structural certificate transports via $\sigma$:
- $G_2$ is also $K_4$
- Same structural obstruction applies
- NO certificate transports to NO certificate

---

## Literature

1. **Babai, L. (2016).** "Graph Isomorphism in Quasipolynomial Time." STOC. *Breakthrough algorithm using equivalence structure.*

2. **Immerman, N. (1999).** *Descriptive Complexity.* Springer. *Logical characterizations and isomorphism.*

3. **Grohe, M. (2017).** *Descriptive Complexity, Canonisation, and Definable Graph Structure Theory.* Cambridge. *Canonical forms and equivalence.*

4. **Univalent Foundations Program. (2013).** *Homotopy Type Theory: Univalent Foundations of Mathematics.* *Transport across equivalences.*

5. **MacLane, S. (1971).** *Categories for the Working Mathematician.* Springer. *Functorial transport of structure.*

6. **Olver, P. J. (1993).** *Applications of Lie Groups to Differential Equations.* Springer. *Symmetry classification.*

7. **Ajtai, M. (1983).** "$\Sigma_1^1$-formulae on Finite Structures." Annals of Pure and Applied Logic. *Isomorphism-closed properties.*

8. **Arora, S. & Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge. *Reductions and certificate structure.*

9. **Cai, J.-Y., Furer, M., & Immerman, N. (1992).** "An Optimal Lower Bound on the Number of Variables for Graph Identification." Combinatorica. *Limits of equivalence detection.*

10. **Weisfeiler, B. & Leman, A. (1968).** "The Reduction of a Graph to Canonical Form and the Algebra which Appears Therein." NTI, Series 2. *Classical canonicalization algorithm.*

---

## Summary

The FACT-Transport theorem, translated to complexity theory, establishes:

1. **Certificate Transport:** Witnesses for isomorphism-closed problems can be systematically transformed across equivalent instances via the equivalence map.

2. **Equivalence Types:** Four fundamental equivalence types (scaling, symmetry, gauge, modulation) capture all natural problem equivalences with bounded transport overhead.

3. **YES$^\sim$ Production:** Equivalent-YES verdicts are systematically producible when an equivalence certificate accompanies the original witness.

4. **Promotion:** Equivalence-relative certificates promote to direct certificates when the equivalence is efficiently computable or sufficiently "small."

5. **Completeness:** The equivalence library is complete for well-structured problem classes, following from symmetry classification.

**Computational Interpretation:**

The transport theorem provides the theoretical foundation for:
- **Reduction-based complexity:** Polynomial reductions preserve certificate structure
- **Canonical forms:** Equivalence classes have unique representatives enabling direct comparison
- **Symmetry exploitation:** Problem symmetries can be leveraged for algorithmic efficiency
- **Certificate lifting:** Solutions to one formulation transfer to equivalent formulations

This is the algorithmic manifestation of the category-theoretic principle that natural transformations preserve structure, applied to the realm of computational complexity and certificate-based verification.
