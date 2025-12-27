---
title: "LOCK-Entropy - Complexity Theory Translation"
---

# LOCK-Entropy: Holographic Entropy Bounds and Information Complexity

## Overview

This document provides a complete complexity-theoretic translation of the LOCK-Entropy theorem (Holographic Entropy Lock, mt-lock-entropy) from the hypostructure framework. The translation establishes a formal correspondence between the Bekenstein-Hawking entropy bound from quantum gravity and **information complexity bounds** in theoretical computer science.

**Original Theorem Reference:** {prf:ref}`mt-lock-entropy`

**Core Translation:** The holographic entropy bound---stating that bulk entropy is bounded by boundary area---corresponds to communication complexity lower bounds via the Ryu-Takayanagi formula and mutual information. Specifically: the information required to solve a problem (bulk entropy) cannot exceed the capacity of the interface (boundary area) used to communicate about it.

---

## Hypostructure Context

The LOCK-Entropy theorem states that if a hypostructure has bounded boundary capacity and topologically controlled boundary, then the bulk information content is bounded by an area law. This provides an information-theoretic lock mechanism: certain "bad" configurations requiring excessive information density are excluded because they cannot fit within the holographic bound.

**Key Certificates:**
- $K_{\mathrm{Cap}_H}^+$: Capacity bound on boundary $\text{Cap}_H(\partial\mathcal{X}) \leq C_{\max}$
- $K_{\mathrm{TB}_\pi}^+$: Topological bound on fundamental group $|\pi_1(\partial\mathcal{X})| < \infty$
- $K_{\text{Holo}}^+$: Holographic entropy certificate (bulk entropy bounded by boundary capacity)

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{Holo}}^+$$

**Physical Interpretation:** The Bekenstein-Hawking bound $S \leq A/(4G_N)$ states that the entropy (information content) of a region cannot exceed its boundary area in Planck units. This is the foundation of the holographic principle: the physics inside a volume is entirely encoded on its surface.

---

## Complexity Theory Statement

**Theorem (LOCK-Entropy, Computational Form).**

Let $\Pi$ be a computational problem with:
- Input space $\mathcal{X}$ (the "bulk") with $n$ bits of description
- Interface/boundary $\partial\mathcal{X}$ through which parties communicate
- Communication protocol $\pi$ using $c$ bits across the interface

Suppose the problem has an **information-theoretic structure** satisfying:

1. **Bounded Interface:** The communication channel has capacity $C = O(|\partial\mathcal{X}|)$
2. **Entropy Bound:** The mutual information required satisfies $I(X; Y) \leq S_{\max}$
3. **Area Law:** The interface capacity scales as $c = O(|\partial\mathcal{X}|)$, not $O(|\mathcal{X}|)$

**Statement (Information Complexity Bound):**

1. **Entropy-Communication Correspondence:** For any protocol solving $\Pi$:
   $$\text{IC}(\pi) \geq I(X; Y | \Pi) = \Omega(S_{\text{bulk}})$$
   where $\text{IC}(\pi)$ is the information complexity of protocol $\pi$.

2. **Area Law for Communication:** If $\Pi$ admits a holographic structure:
   $$\text{CC}(\Pi) = \Omega(|\partial\mathcal{X}|)$$
   Communication complexity is bounded below by the "boundary area."

3. **Holographic Exclusion:** If a problem instance requires $I_{\text{bad}} > C_{\max}$:
   $$\nexists \text{ protocol } \pi: \text{CC}(\pi) \leq C_{\max}$$
   The instance is excluded by information-theoretic capacity.

4. **Certificate Implication:**
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{Holo}}^+$$

translates to: Bounded interface capacity + structured topology implies bounded information complexity.

**Corollary (Holographic Communication Bound):**
For any two-party protocol $\pi$ computing $f: \mathcal{X} \times \mathcal{Y} \to \mathcal{Z}$:
$$\text{CC}(f) \geq I(X; \pi | Y) + I(Y; \pi | X) - O(\log n)$$
The communication cost is lower-bounded by the information revealed about inputs.

---

## Terminology Translation Table

| Hypostructure Term | Complexity Theory Equivalent | Formal Correspondence |
|--------------------|------------------------------|------------------------|
| Bulk entropy $S(\mathcal{X})$ | Internal randomness / input entropy $H(X)$ | Bits needed to describe bulk state |
| Boundary area $A(\partial\mathcal{X})$ | Communication channel capacity $C$ | Bits transmittable across interface |
| Bekenstein bound $S \leq A/(4G_N)$ | Information complexity bound $\text{IC}(\pi) \leq c$ | Entropy bounded by interface capacity |
| Holographic principle | Bulk-boundary correspondence | Interior encoded on surface |
| Holographic entropy | Ryu-Takayanagi formula | $S_A = \frac{\text{Area}(\gamma_A)}{4G_N}$ |
| Minimal surface $\gamma_A$ | Min-cut in communication graph | Optimal entanglement cut |
| Entanglement entropy $S(\rho_A)$ | Mutual information $I(A; B)$ | Correlation across bipartition |
| AdS bulk | Full problem instance space | High-dimensional input |
| CFT boundary | Communication transcript | Low-dimensional encoding |
| $K_{\mathrm{Cap}_H}^+$ | Channel capacity certificate | $C \leq C_{\max}$ |
| $K_{\mathrm{TB}_\pi}^+$ | Protocol structure certificate | Finite-round, bounded complexity |
| $K_{\text{Holo}}^+$ | Information complexity certificate | $\text{IC}(\pi) \leq S_{\max}$ |
| Area law scaling | Sublinear entropy scaling | $S_A \sim |\partial A|$, not $|A|$ |
| Volume law scaling | Extensive entropy scaling | $S_A \sim |A|$ |
| Holographic screen | Communication front | Spacelike slice of protocol |

---

## Proof Sketch

### Setup: Information Complexity and Entropy Bounds

**Definitions:**

1. **Information Complexity (Bar-Yossef et al. 2004):**
For a protocol $\pi$ computing $f(x, y)$ with inputs $(X, Y)$:
$$\text{IC}(\pi) = I(X; \pi | Y) + I(Y; \pi | X)$$
The total information revealed about inputs during communication.

2. **Communication Complexity:**
$$\text{CC}(f) = \min_{\pi \text{ computes } f} |\pi|$$
The minimum bits exchanged by any correct protocol.

3. **Fundamental Inequality (Braverman 2012):**
$$\text{CC}(f) \geq \text{IC}(f) = \min_\pi \text{IC}(\pi)$$

4. **Holographic Entropy (Ryu-Takayanagi 2006):**
For a region $A$ in a CFT with AdS dual, the entanglement entropy is:
$$S_A = \frac{\text{Area}(\gamma_A)}{4G_N}$$
where $\gamma_A$ is the minimal surface in AdS homologous to $A$.

### Step 1: Bekenstein Bound as Information Capacity

**Physical Statement (Bekenstein 1981):**
For a system of energy $E$ confined to radius $R$:
$$S \leq \frac{2\pi E R}{\hbar c}$$

**Computational Translation:**
For a computational problem with input of size $n$ bits:
- Energy $E$ corresponds to computational resources
- Radius $R$ corresponds to problem size parameter
- Entropy $S$ corresponds to information content $H(X)$

**Bound:** The information content is bounded by the "surface" of the problem:
$$H(X) \leq O(|\partial\mathcal{X}|)$$

### Step 2: Ryu-Takayanagi as Min-Cut

**Physical Statement (Ryu-Takayanagi 2006):**
In AdS/CFT, the entanglement entropy of a boundary region $A$ equals the area of the minimal bulk surface anchored to $\partial A$.

**Computational Translation:**
Consider a bipartite communication problem $f: \mathcal{X} \times \mathcal{Y} \to \mathcal{Z}$:
- Boundary = partition of inputs $(X, Y)$
- Bulk = internal computation / protocol execution
- Minimal surface = optimal communication cut

**Min-Cut Bound:**
$$I(X; Y) \leq \text{Cut}(X : Y)$$
The mutual information is bounded by the min-cut capacity.

**Graph-Theoretic Formulation:**
For a communication network $G = (V, E)$ with source $s \in X$ and sink $t \in Y$:
$$I(X; Y) \leq \min_{S: s \in S, t \notin S} \sum_{(u,v): u \in S, v \notin S} c(u,v)$$

### Step 3: Area Law vs. Volume Law

**Physical Dichotomy:**
- **Area Law:** Ground states of gapped local Hamiltonians satisfy $S_A \sim |\partial A|$
- **Volume Law:** Highly excited or random states satisfy $S_A \sim |A|$

**Computational Dichotomy:**

**Area Law Problems (Low Communication Complexity):**
- Problems where relevant information concentrates on interfaces
- Examples: Local search, nearest-neighbor queries
- Communication scales with boundary: $\text{CC}(f) = O(|\partial A|)$

**Volume Law Problems (High Communication Complexity):**
- Problems requiring global information access
- Examples: Set Disjointness, Inner Product
- Communication scales with volume: $\text{CC}(f) = \Omega(|A|)$

**Theorem (Area Law Implies Efficient Communication):**
If problem $\Pi$ satisfies an area law for entanglement entropy:
$$S(\rho_A) \leq c \cdot |\partial A| + o(|\partial A|)$$
then:
$$\text{CC}(\Pi) \leq O(|\partial A| \cdot \text{poly}(\log |A|))$$

### Step 4: Holographic Lock Mechanism

**Physical Lock:**
If a configuration $\mathbb{H}_{\text{bad}}$ requires entropy $S_{\text{bad}} > A/(4G_N)$:
- It cannot fit inside a region with boundary area $A$
- The morphism $\mathbb{H}_{\text{bad}} \to \mathcal{X}$ does not exist

**Computational Lock:**
If a problem instance requires information $I_{\text{bad}} > C_{\max}$:
- No protocol with communication $c \leq C_{\max}$ can solve it
- The instance is excluded from the complexity class

**Formal Statement:**
$$I_{\text{required}}(x) > C_{\max} \Rightarrow x \notin \Pi_{C_{\max}}$$

where $\Pi_{C_{\max}}$ is the set of instances solvable with communication at most $C_{\max}$.

### Step 5: Certificate Logic Translation

**Original Certificate Logic (Hypostructure):**
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{Holo}}^+$$

**Complexity Theory Translation:**

| Certificate | Meaning | Formal Statement |
|-------------|---------|------------------|
| $K_{\mathrm{Cap}_H}^+$ | Bounded interface capacity | Channel capacity $C \leq C_{\max}$ |
| $K_{\mathrm{TB}_\pi}^+$ | Structured protocol | Finite rounds, polynomial transcript |
| $K_{\text{Holo}}^+$ | Information complexity bound | $\text{IC}(\pi) \leq S_{\max}$ |

**Theorem (Certificate Implication):**
If:
- The communication interface has bounded capacity $C \leq C_{\max}$
- The protocol has structured topology (finite rounds, bounded state)

Then:
- The information complexity satisfies $\text{IC}(\pi) \leq f(C_{\max})$
- Instances requiring more information are excluded

---

## Certificate Construction

### Holographic Entropy Certificate Structure

$$K_{\text{Holo}}^+ = \left(\text{entropy\_bound}, \text{cut\_certificate}, \text{area\_law}, \text{exclusion}\right)$$

**Components:**

1. **entropy_bound:** Maximum bulk entropy $S_{\max}$
   - Derived from: $S_{\max} = C \cdot \log(|\Sigma|)$ for alphabet $\Sigma$

2. **cut_certificate:** Minimal cut witnessing the bound
   - Cut edges: $\{(u_i, v_i)\}$ with total capacity $C$
   - Optimality proof: No smaller cut exists

3. **area_law:** Verification that entropy scales with boundary
   - Scaling exponent: $\alpha$ where $S \sim |\partial A|^\alpha$
   - Area law satisfied if $\alpha \leq 1$

4. **exclusion:** Set of excluded instances
   - $\mathcal{E} = \{x : I_{\text{req}}(x) > S_{\max}\}$
   - Proof that $\mathcal{E}$ cannot be solved within capacity

### Explicit Certificate Tuple

```
K_Holo^+ := (
    mode:                "Holographic_Entropy_Lock"
    mechanism:           "Information_Capacity_Bound"

    entropy_analysis: {
        bulk_entropy:    H(X)
        boundary_area:   |partial X|
        capacity:        C = c(partial X)
        bekenstein:      S <= A / (4 G_N) -> IC <= C
    }

    ryu_takayanagi: {
        minimal_surface: gamma_A
        area:            Area(gamma_A)
        entanglement:    S_A = Area / (4 G_N)
        correspondence:  min-cut in communication graph
    }

    information_complexity: {
        protocol:        pi
        IC_pi:           I(X; pi | Y) + I(Y; pi | X)
        lower_bound:     IC(f) >= I(X; Y | f)
        direct_sum:      IC(f^n) >= n * IC(f)
    }

    area_law: {
        satisfied:       true/false
        scaling:         S_A ~ |partial A|^alpha
        exponent:        alpha <= 1
        gap:             lambda_1 > 0 (for gapped systems)
    }

    exclusion: {
        threshold:       I_max = S_max
        excluded_set:    {x : I_req(x) > I_max}
        mechanism:       "capacity_violation"
    }

    literature: {
        bekenstein:      "Bekenstein81"
        ryu_takayanagi:  "RyuTakayanagi06"
        braverman:       "Braverman12"
        information_IC:  "BarYossefJayramKumarSivakumar04"
    }
)
```

---

## Connections to Classical Results

### 1. Communication Complexity

**Set Disjointness (Razborov 1990, Kalyanasundaram-Schnitger 1992):**
$$\text{CC}(\text{DISJ}_n) = \Omega(n)$$

The communication complexity of Set Disjointness is linear---this is a "volume law" problem where information cannot be compressed to the boundary.

**Holographic Interpretation:**
- Bulk: $n$-bit input sets $(X, Y)$
- No area law: Information is extensive in $n$
- No holographic encoding possible: every bit matters

**Gap Amplification (Raz 1999):**
For problems with a gap between YES and NO cases, communication complexity often admits holographic structure:
$$\text{CC}(f^{\otimes k}) \geq k \cdot \Omega(\text{IC}(f))$$

### 2. Information Complexity

**Direct Sum Theorem (Braverman-Rao 2011):**
$$\text{IC}(f^{\otimes n}) = n \cdot \text{IC}(f)$$

Information complexity is additive under parallel repetition.

**Holographic Interpretation:**
- Each copy of $f$ contributes independently to total entropy
- No holographic compression across copies
- Area scales linearly: $|\partial(A^n)| = n \cdot |\partial A|$

**Information = Communication (Braverman 2012):**
$$\text{CC}(f) \approx \text{IC}(f)$$

For many problems, communication and information complexity coincide (up to logarithmic factors).

**Holographic Interpretation:**
- Communication is the "area" of the protocol
- Information is the "entropy" revealed
- Ryu-Takayanagi: entropy = area (up to constants)

### 3. Quantum Communication and Entanglement

**Quantum Communication Complexity:**
With prior entanglement, some problems admit exponential speedup:
$$\text{QCC}(f) = O(\sqrt{\text{CC}(f)})$$

for certain problems (e.g., disjointness variants).

**Holographic Interpretation:**
- Entanglement provides "bulk connectivity"
- Quantum correlations reduce effective boundary area
- Holographic dual: wormholes (ER = EPR correspondence)

**Entanglement Entropy Bounds:**
For quantum communication protocols:
$$\text{QCC}(f) \geq S(\rho_{AB}) / 2$$

The quantum communication is at least half the entanglement entropy.

### 4. Streaming and Space Complexity

**Streaming Lower Bounds via Communication:**
$$\text{Space}(f) \geq \text{CC}(f)$$

Space complexity in streaming is bounded below by communication complexity.

**Holographic Interpretation:**
- Streaming memory = "boundary" at each time step
- Input stream = "bulk" data flowing through
- Area law: memory scales with interface, not total input

**Frequency Moments (Alon-Matias-Szegedy 1996):**
$$\text{Space}(F_k) = \tilde{\Theta}(n^{1-2/k})$$

The space complexity interpolates between area law ($F_2$: $O(\log n)$) and volume law ($F_\infty$: $\Omega(n)$).

---

## Quantitative Bounds

### Information-Communication Correspondence

| Problem | Information Complexity $\text{IC}(f)$ | Communication Complexity $\text{CC}(f)$ | Area/Volume |
|---------|--------------------------------------|----------------------------------------|-------------|
| Equality | $\Theta(1)$ | $O(\log n)$ | Area |
| Inner Product | $\Theta(n)$ | $\Theta(n)$ | Volume |
| Set Disjointness | $\Theta(n)$ | $\Theta(n)$ | Volume |
| Gap Hamming | $\Theta(\sqrt{n})$ | $\Theta(\sqrt{n})$ | Intermediate |
| Indexing | $\Theta(n)$ | $\Theta(n)$ | Volume |
| Pointer Chasing | $\Theta(n/k)$ | $\Theta(n)$ | Depends on rounds |

### Entanglement Entropy Scaling

| System Type | Entropy Scaling | Communication Implication |
|-------------|-----------------|---------------------------|
| Gapped ground state | $S_A \sim |\partial A|$ | Efficient protocols |
| Critical point | $S_A \sim |\partial A| \log |A|$ | Log overhead |
| Random state | $S_A \sim |A|$ | No compression |
| Thermal state | $S_A \sim |A|$ | Volume law |
| Topological order | $S_A \sim |\partial A| - \gamma$ | Topological correction |

### Holographic Bounds

| Regime | Bekenstein Bound | Communication Bound |
|--------|------------------|---------------------|
| Classical | $S \leq 2\pi ER/(\hbar c)$ | $\text{CC} \leq O(n)$ |
| Holographic | $S \leq A/(4G_N)$ | $\text{CC} \leq O(|\partial X|)$ |
| Quantum | $S \leq \min(|A|, |\bar{A}|) \log d$ | $\text{QCC} \leq O(\sqrt{n})$ |
| With entanglement | $S = S(\rho_A)$ | $\text{QCC} + E \leq O(\text{IC})$ |

---

## Applications

### 1. Lower Bounds via Holographic Principle

**Strategy:** To prove $\text{CC}(f) \geq c$:
1. Identify the minimal "entanglement cut" in the problem
2. Compute the mutual information across this cut
3. Apply Ryu-Takayanagi: $\text{CC} \geq \text{Area}/4G_N \equiv I(X; Y)$

**Example: Set Disjointness**
- Bulk: Input sets $X, Y \subseteq [n]$
- Minimal cut: Must distinguish $(X \cap Y = \emptyset)$ from $(|X \cap Y| = 1)$
- Information: Every pair $(x_i, y_i)$ contributes $\Omega(1)$ bits
- Area = Volume: No holographic compression

### 2. Efficient Protocols via Area Laws

**Strategy:** If a problem satisfies an area law:
1. Identify boundary structure
2. Construct protocol that only communicates boundary data
3. Bulk information is locally reconstructible

**Example: Nearest Neighbor Search**
- Bulk: Database of $n$ points
- Boundary: Query region interface
- Area law: Only points near boundary are relevant
- Protocol: $O(|\partial Q| \cdot \log n)$ communication

### 3. Tensor Network Protocols

**Holographic Algorithms via Tensor Networks:**

1. **MERA-based Protocols:** Multi-scale entanglement renormalization provides hierarchical communication structure
   - Each layer: $O(\log n)$ communication
   - Total: $O(\log^2 n)$ for tree-structured problems

2. **Quantum Advantage:** Tensor network structure enables quantum speedups
   - Shared entanglement: reduces effective boundary
   - ER = EPR: wormholes = quantum correlations

### 4. Streaming with Holographic Structure

**Area-Law Streaming Algorithms:**
For problems with bounded entanglement across time:
$$\text{Space}(f) = O(|\partial \mathcal{X}| \cdot \text{poly}(\log n))$$

**Examples:**
- Heavy hitters: Space $O(\log n / \epsilon^2)$
- Distinct elements: Space $O(\log n / \epsilon^2)$
- Frequency moments $F_2$: Space $O(\log n / \epsilon^2)$

All satisfy area laws: memory scales with query interface, not input volume.

---

## Summary

The LOCK-Entropy theorem, translated to complexity theory, establishes **Information Complexity Bounds via Holographic Principles**:

1. **Fundamental Correspondence:**
   - Bekenstein bound $\leftrightarrow$ Information capacity bound
   - Ryu-Takayanagi formula $\leftrightarrow$ Min-cut communication bound
   - Entanglement entropy $\leftrightarrow$ Mutual information across partition
   - Area law $\leftrightarrow$ Sublinear communication scaling
   - Holographic screen $\leftrightarrow$ Communication front in protocol

2. **Main Result:** If a computational interface has bounded capacity:
   - Bulk information (problem complexity) is bounded by boundary area
   - Problems requiring more information are excluded
   - Communication complexity admits holographic lower bounds

3. **Certificate Structure:**
   $$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \Rightarrow K_{\text{Holo}}^+$$

   Bounded interface + structured topology implies information complexity bound.

4. **Connections to TCS:**
   - **Communication Complexity:** Holographic bounds via information theory
   - **Information Complexity:** Direct sum, compression, and Ryu-Takayanagi
   - **Streaming:** Area laws determine space complexity
   - **Quantum Communication:** Entanglement as holographic resource

5. **Classical Foundations:**
   - Bekenstein bound (1981): entropy bounded by energy-radius
   - Ryu-Takayanagi (2006): entanglement = minimal surface area
   - Braverman (2012): information = communication
   - Direct sum theorems: additivity of information cost

This translation reveals that the LOCK-Entropy theorem is the complexity-theoretic statement that **communication complexity is fundamentally bounded by interface capacity**. The holographic principle---that bulk physics is encoded on the boundary---becomes the statement that problem complexity is bounded by the communication channel capacity. Problems violating this bound are excluded from the corresponding complexity class, just as configurations violating the Bekenstein bound cannot exist in a bounded region of spacetime.

---

## Literature

1. **Bekenstein, J. D. (1981).** "Universal Upper Bound on the Entropy-to-Energy Ratio for Bounded Systems." *Physical Review D* 23(2), 287-298. *Original Bekenstein bound.*

2. **Bekenstein, J. D. (1973).** "Black Holes and Entropy." *Physical Review D* 7(8), 2333-2346. *Black hole thermodynamics.*

3. **'t Hooft, G. (1993).** "Dimensional Reduction in Quantum Gravity." *arXiv:gr-qc/9310026*. *Original holographic principle.*

4. **Susskind, L. (1995).** "The World as a Hologram." *Journal of Mathematical Physics* 36(11), 6377-6396. *Holographic bound formulation.*

5. **Ryu, S. & Takayanagi, T. (2006).** "Holographic Derivation of Entanglement Entropy from AdS/CFT." *Physical Review Letters* 96, 181602. *Ryu-Takayanagi formula.*

6. **Bousso, R. (2002).** "The Holographic Principle." *Reviews of Modern Physics* 74(3), 825-874. *Comprehensive holographic review.*

7. **Bar-Yossef, Z., Jayram, T. S., Kumar, R., & Sivakumar, D. (2004).** "An Information Statistics Approach to Data Stream and Communication Complexity." *FOCS*, 209-218. *Information complexity foundations.*

8. **Braverman, M. (2012).** "Interactive Information Complexity." *STOC*, 505-524. *Information = communication.*

9. **Braverman, M. & Rao, A. (2011).** "Information Equals Amortized Communication." *FOCS*, 748-757. *Direct sum theorem.*

10. **Razborov, A. A. (1990).** "On the Distributional Complexity of Disjointness." *Theoretical Computer Science* 106(2), 385-390. *Set disjointness lower bound.*

11. **Kalyanasundaram, B. & Schnitger, G. (1992).** "The Probabilistic Communication Complexity of Set Intersection." *SIAM Journal on Discrete Mathematics* 5(4), 545-557. *Randomized disjointness bound.*

12. **Alon, N., Matias, Y., & Szegedy, M. (1996).** "The Space Complexity of Approximating the Frequency Moments." *STOC*, 20-29. *Streaming lower bounds.*

13. **Van Raamsdonk, M. (2010).** "Building Up Spacetime with Quantum Entanglement." *General Relativity and Gravitation* 42(10), 2323-2329. *ER = EPR.*

14. **Almheiri, A., Dong, X., & Harlow, D. (2015).** "Bulk Locality and Quantum Error Correction in AdS/CFT." *JHEP* 2015(4), 163. *Holographic error correction.*

15. **Eisert, J., Cramer, M., & Plenio, M. B. (2010).** "Area Laws for the Entanglement Entropy." *Reviews of Modern Physics* 82(1), 277-306. *Area laws in physics.*

16. **Hastings, M. B. (2007).** "An Area Law for One-Dimensional Quantum Systems." *Journal of Statistical Mechanics* P08024. *Area law for gapped systems.*
