---
title: "LOCK-Reconstruction - Complexity Theory Translation"
---

# LOCK-Reconstruction: Unique Decoding from Structural Constraints

## Overview

This document provides a complete complexity-theoretic translation of the LOCK-Reconstruction theorem (the Structural Reconstruction Principle) from the hypostructure framework. The translation establishes a formal correspondence between holographic reconstruction in dynamical systems and unique/list decoding in error-correcting codes, revealing deep connections to locally decodable codes and the theory of constraint propagation.

**Original Theorem Reference:** {prf:ref}`mt-lock-reconstruction`

---

## Complexity Theory Statement

**Theorem (LOCK-Reconstruction, Computational Form).**
Let $\mathcal{C} = (E, D, n, k, d)$ be an error-correcting code with:
- $E: \Sigma^k \to \Sigma^n$ encoding function
- $D: \Sigma^n \to \Sigma^k \cup \{\bot\}$ decoding function
- Distance $d = d(\mathcal{C})$ (minimum Hamming distance between codewords)
- Redundancy $r = n - k$

Suppose $\mathcal{C}$ satisfies the following **structural reconstruction conditions**:

1. **Bounded Redundancy (Energy Bound):** The redundancy $r$ is bounded: $r \leq r_{\max}$
2. **Locality (Concentration):** Each message symbol $m_i$ is decodable from at most $\ell$ codeword symbols
3. **Regularity (Scaling):** The code has regular check structure with degree $\Delta$
4. **Gradient Structure (Lojasiewicz):** Error syndromes satisfy: $\|\mathrm{syndrome}(c)\| \geq \gamma \cdot \mathrm{dist}(c, \mathcal{C})^\theta$ for some $\theta \in (0,1]$

Then there exists a **unique decoding algorithm** $D^*: \Sigma^n \to \Sigma^k$ such that:

1. **Unique Decoding:** For any received word $r$ with $\mathrm{dist}(r, \mathcal{C}) < d/2$:
   $$D^*(r) = m \iff E(m) \text{ is the unique closest codeword to } r$$

2. **List Decoding Extension:** For $\mathrm{dist}(r, \mathcal{C}) < d - t$ (Johnson bound regime):
   $$|D^*_{\mathrm{list}}(r)| \leq L(t)$$
   where $L(t)$ is polynomially bounded in the block length $n$.

3. **Local Decodability:** Each message symbol $m_i$ is recoverable by querying $q = O(\ell)$ positions of $r$.

**Corollary (Constraint Propagation Correspondence).**
The LOCK-Reconstruction principle is equivalent to the statement that arc consistency propagation on structured constraint satisfaction problems terminates with a unique solution (or a polynomially-bounded list of candidates) when the constraint graph has sufficient expansion.

---

## Terminology Translation Table

| Hypostructure Concept | Complexity/Coding Theory Analog | Formal Correspondence |
|-----------------------|--------------------------------|------------------------|
| Analytic observables category $\mathcal{A}$ | Received words / noisy codewords | Observations with potential errors |
| Structural objects category $\mathcal{S}$ | Valid codewords / messages | Error-free encoded data |
| Reconstruction functor $F_{\mathrm{Rec}}: \mathcal{A} \to \mathcal{S}$ | Decoding function $D: \Sigma^n \to \Sigma^k$ | Error correction map |
| Holographic reconstruction | Error correction from boundary | Recovery from partial/noisy data |
| Bulk-boundary correspondence | Encoded-decoded relationship | Message $\leftrightarrow$ codeword |
| Reconstruction wedge | Decodable region | Symbols recoverable from subset |
| Entanglement wedge | Information-theoretically accessible region | Maximum decodable information |
| Bad pattern $\mathcal{H}_{\mathrm{bad}}$ | Uncorrectable error pattern | Errors exceeding distance bound |
| Hom isomorphism | Bijection between decodings | Unique decoding guarantee |
| Energy bound $K_{D_E}^+$ | Redundancy bound $r \leq r_{\max}$ | Code rate constraint |
| Concentration $K_{C_\mu}^+$ | Locality parameter $\ell$ | Local decodability |
| Scaling $K_{\mathrm{SC}_\lambda}^+$ | Check regularity $\Delta$ | LDPC structure |
| Lojasiewicz $K_{\mathrm{LS}_\sigma}^+$ | Syndrome-distance inequality | Iterative decoding convergence |
| Bridge certificate $K_{\mathrm{Bridge}}$ | Symmetry-preserving code structure | Automorphism group compatibility |
| Rigidity certificate $K_{\mathrm{Rigid}}$ | Unique decoding radius $t < d/2$ | Error correction capability |
| Lock resolution | Decoding success/failure | Unique solution or list output |

---

## Proof Sketch

### Setup: Error-Correcting Codes as Structural Systems

**Definition (Structured Error-Correcting Code).**
A structured error-correcting code is a tuple $\mathcal{C} = (\Sigma, n, k, H, \mathcal{G})$ where:

- $\Sigma$ is the alphabet (typically $\mathbb{F}_q$ for some prime power $q$)
- $n$ is the block length (codeword size)
- $k$ is the message length (information symbols)
- $H \in \mathbb{F}_q^{(n-k) \times n}$ is the parity-check matrix
- $\mathcal{G} = (V, E)$ is the Tanner graph (bipartite graph encoding check structure)

**Definition (Syndrome and Distance).**
For received word $r \in \Sigma^n$:
- **Syndrome:** $s(r) = H \cdot r \in \Sigma^{n-k}$
- **Distance to code:** $\mathrm{dist}(r, \mathcal{C}) = \min_{c \in \mathcal{C}} d_H(r, c)$
- **Error pattern:** $e = r - c$ for closest codeword $c$

**Definition (Decodable Region).**
The decodable region for unique decoding is:
$$\mathcal{W}_{\mathrm{unique}} = \{r \in \Sigma^n : \mathrm{dist}(r, \mathcal{C}) < d/2\}$$

For list decoding with list size $L$:
$$\mathcal{W}_{\mathrm{list}}^{(L)} = \{r \in \Sigma^n : |\{c \in \mathcal{C} : d_H(r, c) \leq t\}| \leq L\}$$

These regions correspond to the **reconstruction wedge** in holographic terminology.

### Main Argument

#### Step 1: Breached-Inconclusive Analysis (Partial Decoding Information)

**Claim.** The syndrome $s(r) = Hr$ provides partial information about the error pattern, analogous to the "partial progress" certificates in the hypostructure.

**Construction.** Given received word $r$ with unknown error pattern $e$:

1. **Syndrome Computation:** $s = Hr = H(c + e) = He$ (since $Hc = 0$ for codewords)
2. **Constraint Graph:** Each non-zero syndrome component $s_j \neq 0$ indicates that check $j$ is unsatisfied
3. **Error Localization:** The support of the error $\mathrm{supp}(e) \subseteq \{i : \exists j, H_{ji} \neq 0, s_j \neq 0\}$

**Correspondence to Hypostructure.** The syndrome computation maps to the certificate analysis in Step 1 of the LOCK-Reconstruction proof:

| Hypostructure Certificate | Coding Theory Analog |
|---------------------------|---------------------|
| Dimension bound from $K_{C_\mu}^+$ | $|\mathrm{supp}(e)| \leq t$ (bounded errors) |
| Scaling constraints from $K_{\mathrm{SC}_\lambda}^+$ | Regular check degree $\Delta$ |
| Gradient regularity from $K_{\mathrm{LS}_\sigma}^+$ | Syndrome weight $\|s\| \geq \gamma |e|^\theta$ |
| Critical symmetry $G_{\mathrm{crit}}$ | Code automorphism group $\mathrm{Aut}(\mathcal{C})$ |

**Partial Progress.** Even when unique decoding is not immediately possible, the syndrome provides:
- **Upper bound on error weight:** $|e| \leq (n-k)/\log_q(|\Sigma|)$ by counting argument
- **Error localization:** Intersection of check neighborhoods
- **Symmetry structure:** Coset structure $r + \mathcal{C}$ under $\mathrm{Aut}(\mathcal{C})$

#### Step 2: Bridge Certificate (Symmetry-Preserving Structure)

**Claim.** The code automorphism group $\mathrm{Aut}(\mathcal{C})$ acts on both the received words and the codewords, preserving the decoding structure.

**Definition (Code Automorphism).** An automorphism $\pi \in \mathrm{Aut}(\mathcal{C})$ is a permutation $\pi: [n] \to [n]$ such that:
$$c \in \mathcal{C} \implies \pi(c) \in \mathcal{C}$$

where $\pi(c)_i = c_{\pi^{-1}(i)}$.

**Bridge Properties (Analog to $K_{\mathrm{Bridge}}$):**

1. **Distance Preservation:** $d_H(\pi(r), \pi(c)) = d_H(r, c)$ for all $\pi \in \mathrm{Aut}(\mathcal{C})$

   *Correspondence:* Energy preservation $\Phi(\rho(g) \cdot x) = \Phi(x)$

2. **Syndrome Equivariance:** $s(\pi(r)) = \pi_H(s(r))$ where $\pi_H$ is the induced action on syndromes

   *Correspondence:* Stratification preservation $\rho(g)(\Sigma_k) = \Sigma_k$

3. **Decoding Compatibility:** $D(\pi(r)) = \pi_k(D(r))$ where $\pi_k$ is the induced action on messages

   *Correspondence:* Gradient flow commutation $\rho(g) \circ \nabla\Phi = \nabla\Phi \circ \rho(g)$

**Critical Symmetry Extraction.** The group $G_{\mathrm{crit}} \subseteq \mathrm{Aut}(\mathcal{C})$ consists of automorphisms that:
- Preserve the error pattern structure
- Map between equivalent error cosets
- Generate the ambiguity in list decoding

For unique decoding, $|G_{\mathrm{crit}}| = 1$ (trivial symmetry). For list decoding with list size $L$, the orbit structure under $G_{\mathrm{crit}}$ determines the list.

#### Step 3: Rigidity Certificate (Unique Decoding Guarantee)

**Claim.** Sufficient code structure guarantees unique decoding within the error-correction radius.

**Rigidity Types (Analog to $K_{\mathrm{Rigid}}$ Cases):**

**Case A: Reed-Solomon Codes (Algebraic Rigidity)**

For Reed-Solomon codes over $\mathbb{F}_q$ with evaluation points $\alpha_1, \ldots, \alpha_n$:
- **Semisimplicity:** The code is the image of the polynomial evaluation map
- **Galois structure:** $\mathrm{Aut}(\mathcal{C}) \cong \mathrm{Aff}(\mathbb{F}_q)$ (affine group)
- **Unique decoding:** Berlekamp-Massey algorithm decodes up to $t < (n-k)/2$ errors

*Correspondence:* Tannakian rigidity $\mathcal{S} \simeq \mathrm{Rep}_k(G)$

**Case B: LDPC Codes (Expansion Rigidity)**

For LDPC codes with $(d_v, d_c)$-regular Tanner graph:
- **Expansion property:** The graph is a $(\gamma, \delta)$-expander
- **Tame stratification:** Error patterns stratify by weight and syndrome structure
- **Unique decoding:** Belief propagation converges for $t < d/2$ errors

*Correspondence:* O-minimal stratification with finite strata

**Case C: Quantum Codes (Spectral Rigidity)**

For stabilizer codes with stabilizer group $\mathcal{S}$:
- **Spectral gap:** The code Hamiltonian $H = \sum_{s \in \mathcal{S}} (I - s)$ has gap $\Delta > 0$
- **Ground state:** The code space is $\ker(H)$
- **Unique decoding:** Syndrome measurement projects to unique error class

*Correspondence:* Spectral gap $\inf(\sigma(L_G) \setminus \{0\}) \geq \delta > 0$

**Rigidity Implies Unique Decoding.** In each case, the rigidity structure ensures:
$$|\{c \in \mathcal{C} : d_H(r, c) \leq t\}| = 1 \quad \text{for } t < d/2$$

#### Step 4: Dictionary Construction (Decoding Algorithm)

**Claim.** The reconstruction functor $F_{\mathrm{Rec}}$ corresponds to an explicit decoding algorithm.

**Type-Specific Decoders:**

**Algebraic Decoder (Reed-Solomon, BCH):**
$$D^{\mathrm{alg}}(r) = \text{Berlekamp-Massey}(s(r))$$

Algorithm:
1. Compute syndrome $s = Hr$
2. Find error-locator polynomial $\Lambda(x)$ via Berlekamp-Massey
3. Find error locations as roots of $\Lambda(x)$
4. Solve for error values via Forney's formula
5. Output $m = D(r - e)$

**Iterative Decoder (LDPC, Turbo):**
$$D^{\mathrm{iter}}(r) = \lim_{t \to \infty} \text{BP}^{(t)}(r)$$

Algorithm:
1. Initialize likelihoods from channel observations
2. Iterate message passing on Tanner graph
3. Converge to posterior probabilities
4. Threshold to obtain hard decisions
5. Verify syndrome $s(D^{\mathrm{iter}}(r)) = 0$

**Spectral Decoder (Quantum/Stabilizer):**
$$D^{\mathrm{spec}}(r) = \Pi_0 \cdot r$$

Algorithm:
1. Measure syndrome generators
2. Identify error coset from syndrome
3. Apply minimum-weight correction
4. Project to code space

**Correspondence to Hypostructure.** Each decoder implements the reconstruction functor:

| Decoder Type | Functor Action | Certificate Used |
|--------------|----------------|------------------|
| Algebraic | Polynomial interpolation | $K_{\mathrm{Rigid}}^{\mathrm{alg}}$ |
| Iterative | Fixed-point iteration | $K_{\mathrm{Rigid}}^{\mathrm{para}}$ |
| Spectral | Eigenspace projection | $K_{\mathrm{Rigid}}^{\mathrm{quant}}$ |

#### Step 5: Hom Isomorphism (Decoding Bijectivity)

**Claim.** The decoding function establishes a bijection between error cosets and messages within the unique decoding radius.

**Formal Statement.** Define:
- $\mathrm{Hom}_{\mathcal{A}}(e, r) = \{c \in \mathcal{C} : r = c + e'\text{ for some } e' \sim e\}$ (error pattern embeddings)
- $\mathrm{Hom}_{\mathcal{S}}(D(e), D(r)) = \{(m', m) : E(m) + e' = r\}$ (decoded message relationships)

**Isomorphism Verification:**

**Injectivity.** Suppose $D(r) = D(r')$ for $r, r' \in \mathcal{W}_{\mathrm{unique}}$.

Then $r = c + e$ and $r' = c' + e'$ for unique closest codewords $c, c'$.

If $D(r) = D(r')$, then the decoded messages are equal, so $c = c'$ (encoding is injective).

Thus $e = r - c = r' - c' + (r - r') = e' + (r - r')$.

If $r \neq r'$, this contradicts unique decoding unless $r = r'$. $\square$

**Surjectivity.** For any message $m \in \Sigma^k$:
- The codeword $c = E(m)$ exists
- Any $r$ with $d_H(r, c) < d/2$ satisfies $D(r) = m$
- The fiber $D^{-1}(m) = \{r : d_H(r, E(m)) < d/2\}$ is non-empty

Thus every message is in the image of $D$ when restricted to $\mathcal{W}_{\mathrm{unique}}$. $\square$

**Naturality.** The decoding commutes with code automorphisms:

```
Hom_A(e, r)  -----> Hom_S(D(e), D(r))
    |                     |
   π*                   π_k*
    |                     |
    v                     v
Hom_A(π(e), π(r)) --> Hom_S(D(π(e)), D(π(r)))
```

This diagram commutes by the Bridge Certificate properties.

#### Step 6: Lock Resolution (Decoding Outcome)

**Claim.** The decoding problem is decidable: either unique decoding succeeds, or a bounded list is produced, or failure is detected.

**Case Analysis (Analog to Node 17 Lock Resolution):**

**Case: $\mathrm{dist}(r, \mathcal{C}) < d/2$ (Unique Decoding Region)**

The decoder outputs unique $m = D(r)$ with certificate:
$$K_{\mathrm{decode}}^{\mathrm{unique}} = (m, c = E(m), e = r - c, \|s(r)\| = 0 \text{ after correction})$$

*Correspondence:* $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ -- the "bad pattern" (uncorrectable error) cannot embed.

**Case: $d/2 \leq \mathrm{dist}(r, \mathcal{C}) < d - t_J$ (List Decoding Region)**

The decoder outputs list $\{m_1, \ldots, m_L\}$ with $L \leq L(t_J)$ (Johnson bound):
$$K_{\mathrm{decode}}^{\mathrm{list}} = (\{m_i\}_{i=1}^L, \{c_i = E(m_i)\}_{i=1}^L, L \leq q^{O(\sqrt{n})})$$

*Correspondence:* $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ with explicit witnesses (the list).

**Case: $\mathrm{dist}(r, \mathcal{C}) \geq d - t_J$ (Beyond List Decoding)**

Decoding fails with diagnostic:
$$K_{\mathrm{decode}}^{\mathrm{fail}} = (\text{FAIL}, \|s(r)\| > \gamma (d - t_J)^\theta)$$

*Correspondence:* Fatal error mode requiring external intervention.

**Decidability.** The case distinction is computable:
1. Compute syndrome $s(r)$
2. Run decoding algorithm with syndrome-weight monitoring
3. If $\|s\| = 0$ after correction: unique decoding
4. If $\|s\| > 0$ but list found: list decoding
5. If no valid decoding: failure

### Certificate Construction

The proof is constructive. Given a received word $r$ and structured code $\mathcal{C}$:

**Reconstruction Certificate $K_{\mathrm{Rec}}^+$:**
$$K_{\mathrm{Rec}}^+ = (D, \Phi_{\mathrm{decode}}, \mathrm{outcome}, \mathrm{type}, \mathcal{D}_{\mathrm{Rec}})$$

where:
- $D: \Sigma^n \to \Sigma^k$: Decoding function
- $\Phi_{\mathrm{decode}}$: Bijection between error cosets and messages (Hom isomorphism)
- $\mathrm{outcome} \in \{\mathrm{unique}, \mathrm{list}, \mathrm{fail}\}$: Resolution status
- $\mathrm{type} \in \{\mathrm{alg}, \mathrm{iter}, \mathrm{spec}\}$: Code structure type
- $\mathcal{D}_{\mathrm{Rec}}$: Decoding dictionary with complexity bounds

**Component Certificates:**

| Certificate | Content | Verification |
|-------------|---------|--------------|
| $K_{D_E}^+$ (Redundancy) | $(n, k, r = n-k)$ | $r \leq r_{\max}$ |
| $K_{C_\mu}^+$ (Locality) | $(\ell, \text{query positions})$ | Each symbol from $\leq \ell$ positions |
| $K_{\mathrm{SC}_\lambda}^+$ (Regularity) | $(\Delta, \text{check degrees})$ | LDPC graph is $\Delta$-regular |
| $K_{\mathrm{LS}_\sigma}^+$ (Gradient) | $(\gamma, \theta, \text{syndrome bound})$ | $\|s\| \geq \gamma \cdot |e|^\theta$ |
| $K_{\mathrm{Bridge}}$ (Symmetry) | $(\mathrm{Aut}(\mathcal{C}), \rho, \text{equivariance proof})$ | Decoding commutes with automorphisms |
| $K_{\mathrm{Rigid}}$ (Uniqueness) | $(d, t, \text{unique decoding proof})$ | $t < d/2$ implies unique closest codeword |

---

## Connections to Classical Results

### 1. Shannon's Noisy Channel Coding Theorem

**Theorem (Shannon 1948).** For a discrete memoryless channel with capacity $C$, there exist codes with rate $R < C$ achieving arbitrarily small error probability, and no codes with rate $R > C$ can achieve vanishing error.

**Connection to LOCK-Reconstruction.** The theorem establishes:

| Shannon | LOCK-Reconstruction |
|---------|---------------------|
| Channel capacity $C$ | Energy bound $K_{D_E}^+$ |
| Achievable rate $R < C$ | Structural category $\mathcal{S}$ |
| Encoding | Embedding into $\mathcal{A}$ |
| Decoding | Reconstruction functor $F_{\mathrm{Rec}}$ |
| Error probability $\to 0$ | Lock resolution success |

**Interpretation.** Shannon's theorem guarantees that structured codes (achieving capacity) exist. The LOCK-Reconstruction theorem characterizes when decoding from such codes is unique---not just probable, but guaranteed.

### 2. Singleton Bound and MDS Codes

**Theorem (Singleton 1964).** For any $(n, k, d)$ code: $d \leq n - k + 1$.

Codes achieving equality are called **Maximum Distance Separable (MDS)**.

**Connection to LOCK-Reconstruction.** MDS codes have optimal rigidity:

| MDS Property | LOCK Certificate |
|--------------|------------------|
| $d = n - k + 1$ | Maximal $K_{\mathrm{Rigid}}$ |
| Any $k$ symbols determine message | Minimal locality $\ell = k$ |
| Reed-Solomon codes | Algebraic rigidity type |
| Unique decoding up to $(n-k)/2$ errors | Complete lock resolution |

**Interpretation.** MDS codes are the coding-theoretic analog of "maximally constrained" hypostructures where the structural category $\mathcal{S}$ is as small as possible relative to $\mathcal{A}$.

### 3. Johnson Bound and List Decoding

**Theorem (Johnson 1962, Guruswami-Sudan 1999).** For a code with distance $d$ over alphabet $\Sigma$:
- Unique decoding radius: $t < d/2$
- List decoding radius: $t < d(1 - 1/|\Sigma|)$ with polynomial list size

**Connection to LOCK-Reconstruction.** The Johnson bound determines the transition:

| Regime | LOCK Status | Output |
|--------|-------------|--------|
| $t < d/2$ | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Unique codeword |
| $d/2 \leq t < d(1-1/q)$ | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ | Bounded list |
| $t \geq d(1-1/q)$ | FAIL | Exponential list |

**Interpretation.** The list decoding regime corresponds to the "breached-with-witness" case where multiple reconstructions exist but are polynomially enumerable.

### 4. Locally Decodable Codes (LDCs)

**Definition (LDC).** A $(q, \delta, \epsilon)$-LDC allows recovery of any message bit by querying $q$ codeword positions, tolerating $\delta n$ errors, with success probability $\geq 1 - \epsilon$.

**Theorem (Katz-Trevisan 2000).** Any $q$-query LDC with constant $\delta$ and $\epsilon$ has length $n \geq 2^{\Omega(k/q^2)}$.

**Connection to LOCK-Reconstruction.** LDCs correspond to strong locality certificates:

| LDC Property | LOCK Certificate |
|--------------|------------------|
| Query complexity $q$ | Locality $K_{C_\mu}^+$ with $\ell = q$ |
| Error tolerance $\delta$ | Rigidity $K_{\mathrm{Rigid}}$ strength |
| Success probability $1-\epsilon$ | Resolution confidence |
| Exponential length | Trade-off with locality |

**Interpretation.** The LOCK-Reconstruction theorem with strong locality ($K_{C_\mu}^+$ with small $\ell$) produces LDCs. The exponential length lower bound reflects the tension between locality and redundancy.

### 5. Constraint Satisfaction and Arc Consistency

**Definition (Arc Consistency).** A CSP is arc consistent if for every constraint and every variable in its scope, each value in the variable's domain has a consistent extension.

**Theorem (Freuder 1982).** For tree-structured CSPs, arc consistency propagation solves the CSP in linear time.

**Connection to LOCK-Reconstruction.** Decoding is CSP solving:

| CSP Concept | Coding Theory Analog |
|-------------|---------------------|
| Variables | Codeword positions $c_1, \ldots, c_n$ |
| Domains | Alphabet $\Sigma$ |
| Constraints | Parity checks $\sum_i H_{ji} c_i = 0$ |
| Arc consistency | Syndrome-guided elimination |
| Solution | Decoded codeword |

**Interpretation.** The LDPC Tanner graph is a CSP constraint graph. Arc consistency propagation corresponds to belief propagation decoding. Tree-like graphs (high girth, good expansion) guarantee unique solutions---the structural rigidity required for LOCK resolution.

### 6. Tanner Graphs and Expander Codes

**Definition (Expander Graph).** A bipartite graph $G = (L, R, E)$ is a $(\gamma, \delta)$-expander if every subset $S \subseteq L$ with $|S| \leq \gamma |L|$ has $|\Gamma(S)| \geq (1-\delta) \cdot d \cdot |S|$.

**Theorem (Sipser-Spielman 1996).** LDPC codes from expander graphs with expansion $> 3/4$ are uniquely decodable in linear time.

**Connection to LOCK-Reconstruction.** Expansion provides rigidity:

| Expansion Property | LOCK Certificate |
|--------------------|------------------|
| Vertex expansion $\gamma$ | Error tolerance bound |
| Edge expansion $\delta$ | Syndrome growth rate |
| Linear-time decoding | Polynomial reconstruction |
| Unique decoding | Full lock resolution |

**Interpretation.** Expander graphs are the combinatorial foundation for the rigidity certificate $K_{\mathrm{Rigid}}^{\mathrm{para}}$ (tame stratification type). The expansion property ensures that local structure propagates globally, enabling unique reconstruction.

---

## Quantitative Refinements

### Decoding Complexity Bounds

| Code Family | Unique Decoding Complexity | List Decoding Complexity |
|-------------|---------------------------|--------------------------|
| Reed-Solomon | $O(n^2)$ (Berlekamp-Massey) | $O(n^3)$ (Guruswami-Sudan) |
| LDPC | $O(n \log n)$ (Belief Propagation) | $O(n^2)$ (Linear Programming) |
| Polar | $O(n \log n)$ (Successive Cancellation) | $O(n \log^2 n)$ (List SC) |
| Expander | $O(n)$ (Sipser-Spielman) | $O(n \log n)$ |

### Error Correction Capacity

For code rate $R = k/n$ and error fraction $\tau = t/n$:

**Unique Decoding:** $\tau < (1-R)/2$ (half minimum distance)

**List Decoding:** $\tau < 1 - R$ (Singleton bound)

**Capacity-Achieving:** $\tau \to 1 - R - \epsilon$ as block length $n \to \infty$

### Locality-Redundancy Trade-off

For $q$-query LDC with rate $R$:

$$n \geq \begin{cases}
2^{\Omega(k)} & q = 2 \\
k^{1+1/(q-1)+o(1)} & q \geq 3 \\
k^{1+o(1)} & q = O(\log k)
\end{cases}$$

This quantifies the tension between locality ($K_{C_\mu}^+$) and efficiency ($K_{D_E}^+$).

---

## Literature References

### Error-Correcting Codes

- **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
- **Singleton, R. C.** (1964). "Maximum distance q-nary codes." *IEEE Trans. Information Theory*.
- **Berlekamp, E. R.** (1968). *Algebraic Coding Theory*. McGraw-Hill.
- **Guruswami, V., Sudan, M.** (1999). "Improved decoding of Reed-Solomon and algebraic-geometry codes." *IEEE Trans. Information Theory*.

### LDPC and Iterative Decoding

- **Gallager, R. G.** (1962). "Low-density parity-check codes." *IRE Trans. Information Theory*.
- **Sipser, M., Spielman, D. A.** (1996). "Expander codes." *IEEE Trans. Information Theory*.
- **Richardson, T., Urbanke, R.** (2008). *Modern Coding Theory*. Cambridge University Press.

### Locally Decodable Codes

- **Katz, J., Trevisan, L.** (2000). "On the efficiency of local decoding procedures for error-correcting codes." *STOC*.
- **Yekhanin, S.** (2012). "Locally Decodable Codes." *Foundations and Trends in TCS*.
- **Dvir, Z.** (2010). "On matrix rigidity and locally self-correctable codes." *CCC*.

### Constraint Satisfaction

- **Freuder, E. C.** (1982). "A sufficient condition for backtrack-free search." *JACM*.
- **Dechter, R.** (2003). *Constraint Processing*. Morgan Kaufmann.
- **Mezard, M., Montanari, A.** (2009). *Information, Physics, and Computation*. Oxford University Press.

### Holographic Codes and Quantum Error Correction

- **Almheiri, A., Dong, X., Harlow, D.** (2015). "Bulk locality and quantum error correction in AdS/CFT." *JHEP*.
- **Pastawski, F., Yoshida, B., Harlow, D., Preskill, J.** (2015). "Holographic quantum error-correcting codes." *JHEP*.
- **Harlow, D.** (2017). "The Ryu-Takayanagi formula from quantum error correction." *Communications in Mathematical Physics*.

---

## Summary

The LOCK-Reconstruction theorem, translated to complexity and coding theory, establishes that:

1. **Unique decoding is guaranteed by structural rigidity:** When a code satisfies the structural reconstruction conditions (bounded redundancy, locality, regularity, gradient structure), received words within the unique decoding radius can be decoded unambiguously.

2. **The reconstruction functor is a decoding algorithm:** The functor $F_{\mathrm{Rec}}: \mathcal{A} \to \mathcal{S}$ corresponds to concrete decoding algorithms (Berlekamp-Massey, belief propagation, spectral projection) depending on the code's rigidity type.

3. **Lock resolution corresponds to decoding outcome:** The three-way case analysis (unique/list/fail) mirrors the coding-theoretic trichotomy of being within unique decoding radius, list decoding radius, or beyond correction capability.

4. **Holographic reconstruction generalizes error correction:** The bulk-boundary correspondence in AdS/CFT, where bulk operators can be reconstructed from boundary data within the entanglement wedge, is mathematically equivalent to error correction from partial codeword access.

This translation reveals that the hypostructure framework's Structural Reconstruction Principle is the categorical generalization of fundamental results in coding theory and constraint satisfaction, providing a unified language for understanding unique recovery from partial or noisy observations.
