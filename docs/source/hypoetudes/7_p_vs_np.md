# Étude 7: P versus NP and Hypostructure in Computational Complexity

## Abstract

We develop a hypostructure-theoretic framework for the P versus NP problem, the central open question in computational complexity theory. The class separation is reinterpreted through axiom satisfaction: P corresponds to problems where Axiom R (Recovery) holds with polynomial resources, while NP represents problems where only verification (a weaker form) is polynomial. We establish that P $\neq$ NP is equivalent to the assertion that witness recovery fundamentally requires super-polynomial resources for certain problems. The framework extends to characterize the polynomial hierarchy, interactive proofs, and circuit complexity through graded axiom structures. This étude demonstrates that hypostructure theory provides a unified language for computational barriers and resource-bounded computation.

---

## 1. Introduction

### 1.1. Complexity Classes

**Definition 1.1.1** (Decision Problem). *A decision problem is a subset $L \subseteq \{0,1\}^*$ of binary strings.*

**Definition 1.1.2** (Class P). *P is the class of decision problems decidable by a deterministic Turing machine in time $O(n^k)$ for some constant $k$:*
$$\text{P} = \bigcup_{k \geq 1} \text{DTIME}(n^k)$$

**Definition 1.1.3** (Class NP). *NP is the class of decision problems with polynomial-time verifiable witnesses:*
$$L \in \text{NP} \Leftrightarrow \exists \text{ poly-time } V, \exists c : x \in L \Leftrightarrow \exists w (|w| \leq |x|^c \land V(x,w) = 1)$$

### 1.2. The P versus NP Problem

**Problem 1.2.1** (P vs NP). *Does P = NP? Equivalently: for every problem with efficiently verifiable solutions, can solutions be efficiently found?*

**Definition 1.2.2** (NP-Completeness). *A problem $L$ is NP-complete if:*
1. *$L \in \text{NP}$*
2. *For all $L' \in \text{NP}$: $L' \leq_p L$ (polynomial-time many-one reducible)*

**Theorem 1.2.3** (Cook-Levin 1971). *SAT (Boolean satisfiability) is NP-complete.*

### 1.3. Significance

**Observation 1.3.1**. *If P = NP, then:*
- *Cryptography based on computational hardness fails*
- *Creative mathematical discovery becomes routine*
- *Optimization problems become tractable*

*If P $\neq$ NP, certain problems are fundamentally intractable.*

---

## 2. The Space of Computational Problems

### 2.1. Problem Space Structure

**Definition 2.1.1** (Problem Instance Space). *For a problem $L$, the instance space is:*
$$\mathcal{I}_L = \{0,1\}^*$$
*equipped with the length metric $d(x,y) = |n - m|$ where $|x| = n$, $|y| = m$.*

**Definition 2.1.2** (Solution Space). *For $L \in \text{NP}$ with witness relation $R$:*
$$\mathcal{S}_L(x) = \{w : R(x,w) = 1\}$$

**Definition 2.1.3** (Complexity Metric). *Define distance between problems:*
$$d(L_1, L_2) = \inf\{1/\text{poly}(n) : L_1 \leq_p L_2 \text{ with reduction of degree } \leq \log(1/d)\}$$

### 2.2. The NP Landscape

**Definition 2.2.1** (NP-Intermediate). *A problem is NP-intermediate if it is in NP, not in P, and not NP-complete (assuming P $\neq$ NP).*

**Theorem 2.2.2** (Ladner 1975). *If P $\neq$ NP, then NP-intermediate problems exist.*

*Proof.* Construct a problem by diagonalization that avoids both P and NP-completeness through careful padding. $\square$

### 2.3. Promise Problems and Average-Case

**Definition 2.3.1** (Promise Problem). *A promise problem is a pair $(L_{yes}, L_{no})$ where we promise $x \in L_{yes} \cup L_{no}$.*

**Definition 2.3.2** (Average-Case Complexity). *$(L, D) \in \text{AvgP}$ if there exists algorithm $A$ and polynomial $p$ such that:*
$$\mathbb{E}_{x \sim D_n}[T_A(x)] \leq p(n)$$

---

## 3. Hypostructure Data for Complexity

### 3.1. Primary Structures

**Definition 3.1.1** (Complexity Hypostructure). *The P vs NP hypostructure consists of:*
- *State space: $X = 2^{\{0,1\}^*}$ (space of problems)*
- *Scale parameter: $\lambda = n^{-k}$ (polynomial resource bound)*
- *Energy functional: $E(L,n) = $ minimum circuit size for $L$ on inputs of length $n$*
- *Flow: Resource-bounded computation*

### 3.2. Resource Hierarchy

**Definition 3.2.1** (Resource Levels). *At resource level $k$:*
$$X_k = \{L : L \text{ decidable in time } O(n^k)\}$$

**Proposition 3.2.2** (Strict Hierarchy). *By the time hierarchy theorem:*
$$X_1 \subsetneq X_2 \subsetneq X_3 \subsetneq \cdots \subsetneq \text{P} = \bigcup_k X_k$$

### 3.3. The Verification-Search Gap

**Definition 3.3.1** (Verification Complexity). *For $L \in \text{NP}$:*
$$V(L) = \min\{k : \text{witnesses verifiable in } O(n^k)\}$$

**Definition 3.3.2** (Search Complexity). *For $L \in \text{NP}$:*
$$S(L) = \min\{k : \text{witnesses findable in } O(n^k)\} \text{ (possibly } \infty \text{)}$$

**Observation 3.3.3**. *P = NP iff $S(L) < \infty$ for all $L \in$ NP.*

---

## 4. Axiom C: Compactness in Complexity

### 4.1. Finite Approximations

**Definition 4.1.1** (Truncated Problem). *For $L \subseteq \{0,1\}^*$:*
$$L_{\leq n} = L \cap \{0,1\}^{\leq n}$$

**Theorem 4.1.1** (Compactness for P). *If $L \in$ P with time bound $T(n) = n^k$, then finite approximations determine $L$:*

*The problem $L_{\leq n}$ is decidable by a circuit of size $O(n^{k+1})$, and these circuits converge to $L$.*

*Proof.* The algorithm on inputs up to length $n$ can be hardcoded into a circuit. Circuit families represent $L$ uniformly. $\square$

**Invocation 4.1.2** (Metatheorem 7.1). *Problems in P satisfy Axiom C:*
$$\text{Polynomial-size circuits witness compactness}$$

### 4.2. Compactness for NP

**Theorem 4.2.1** (NP Compactness via Witnesses). *If $L \in$ NP, then:*
$$x \in L \Leftrightarrow \text{witness exists of size } |x|^c$$

*Compactness holds for witness verification, not witness finding.*

**Corollary 4.2.2**. *NP satisfies Axiom C for verification, but potentially not for search.*

---

## 5. Axiom D: Dissipation and Computation Time

### 5.1. Time as Dissipation

**Definition 5.1.1** (Computational Energy). *For algorithm $A$ on input $x$:*
$$E_t(A,x) = \mathbf{1}_{A \text{ not halted by step } t}$$

**Theorem 5.1.1** (Dissipation for P). *If $L \in$ P with bound $n^k$, then for inputs of length $n$:*
$$E_t(A,x) = 0 \quad \text{for } t \geq n^k$$

*Energy dissipates in polynomial time.*

*Proof.* By definition of P, computation halts within $n^k$ steps. $\square$

**Invocation 5.1.2** (Metatheorem 7.2). *P satisfies Axiom D with polynomial dissipation rate.*

### 5.2. NP and Non-Deterministic Dissipation

**Theorem 5.2.1** (NP Dissipation Structure). *For $L \in$ NP:*
- *Verification dissipates in polynomial time*
- *Exhaustive search dissipates in exponential time*
- *P = NP iff search also dissipates polynomially*

*Proof.* Verification is polynomial by definition. Exhaustive search over $2^{n^c}$ witnesses takes exponential time. $\square$

---

## 6. Axiom SC: Scale Coherence and the Polynomial Hierarchy

### 6.1. The Polynomial Hierarchy

**Definition 6.1.1** (Polynomial Hierarchy). *Define inductively:*
- *$\Sigma_0^p = \Pi_0^p = $ P*
- *$\Sigma_{k+1}^p = \text{NP}^{\Sigma_k^p}$*
- *$\Pi_{k+1}^p = \text{coNP}^{\Sigma_k^p}$*
- *$\text{PH} = \bigcup_k \Sigma_k^p$*

**Proposition 6.1.2** (Hierarchy Relations).
- *$\Sigma_1^p = $ NP, $\Pi_1^p = $ coNP*
- *$\Sigma_k^p \cup \Pi_k^p \subseteq \Sigma_{k+1}^p \cap \Pi_{k+1}^p$*

### 6.2. Scale Coherence by Level

**Theorem 6.2.1** (Quantifier-Scale Correspondence). *A problem in $\Sigma_k^p$ has $k$ levels of quantifier alternation:*
$$L \in \Sigma_k^p \Leftrightarrow x \in L \Leftrightarrow \exists y_1 \forall y_2 \exists y_3 \cdots Q_k y_k \, R(x, \vec{y})$$
*where $R$ is polynomial-time computable and $|y_i| \leq |x|^c$.*

**Invocation 6.2.2** (Metatheorem 7.3). *The polynomial hierarchy measures scale coherence depth:*
$$\text{PH level } k = \text{Axiom SC with } k \text{ coherence layers}$$

### 6.3. Hierarchy Collapse

**Theorem 6.3.1** (Collapse Theorem). *If $\Sigma_k^p = \Pi_k^p$ for some $k$, then PH $= \Sigma_k^p$.*

**Corollary 6.3.2**. *P = NP implies PH = P (total collapse).*

---

## 7. Axiom LS: Local Stiffness and Hardness Amplification

### 7.1. Worst-Case to Average-Case

**Definition 7.1.1** (Locally Stiff Problem). *$L$ is locally stiff if hardness is uniform:*
$$\Pr_{x \sim U_n}[A(x) \text{ correct}] \leq 1 - 1/\text{poly}(n) \Rightarrow L \notin \text{P}$$

**Theorem 7.1.1** (Hardness Amplification). *For certain NP-complete problems (lattice problems, some coding theory problems):*
*Worst-case hardness implies average-case hardness.*

*Proof.* Via random self-reducibility: a random instance can encode a worst-case instance with high probability. $\square$

**Invocation 7.1.2** (Metatheorem 7.4). *Problems with worst-case to average-case reduction satisfy Axiom LS:*
$$\text{Local hardness} \Rightarrow \text{Global hardness}$$

### 7.2. Cryptographic Hardness

**Definition 7.2.1** (One-Way Function). *$f: \{0,1\}^* \to \{0,1\}^*$ is one-way if:*
1. *$f$ computable in polynomial time*
2. *For all PPT $A$: $\Pr[f(A(f(x))) = f(x)] \leq \text{negl}(n)$*

**Theorem 7.2.2** (OWF Characterization). *One-way functions exist iff P $\neq$ NP $\cap$ coNP in a certain distributional sense.*

---

## 8. Axiom Cap: Capacity and Circuit Complexity

### 8.1. Circuit Complexity

**Definition 8.1.1** (Circuit Size). *For $L \subseteq \{0,1\}^*$:*
$$\text{SIZE}(L,n) = \min\{|C| : C \text{ computes } L_n\}$$
*where $L_n = L \cap \{0,1\}^n$.*

**Theorem 8.1.1** (Shannon 1949). *For most Boolean functions on $n$ variables:*
$$\text{SIZE}(f) \geq \frac{2^n}{n}$$

*Proof.* Counting argument: $2^{2^n}$ functions, at most $n^{O(s)}$ circuits of size $s$. $\square$

### 8.2. Capacity Bounds and P vs NP

**Theorem 8.2.1** (P/poly Characterization). *$L \in$ P/poly iff $\text{SIZE}(L,n) \leq n^{O(1)}$.*

**Theorem 8.2.2** (Karp-Lipton 1980). *If NP $\subseteq$ P/poly, then PH $= \Sigma_2^p$.*

**Invocation 8.2.2** (Metatheorem 7.5). *Axiom Cap in complexity:*
$$\text{Cap}(L) = \limsup_{n \to \infty} \frac{\log \text{SIZE}(L,n)}{\log n}$$
*P = problems with $\text{Cap}(L) < \infty$.*

### 8.3. Lower Bounds

**Theorem 8.3.1** (Razborov-Smolensky 1980s). *PARITY requires superpolynomial-size $AC^0$ circuits:*
$$\text{SIZE}_{AC^0}(\text{PARITY}, n) \geq 2^{n^{\Omega(1)}}$$

**Open Problem 8.3.2**. *Prove $\text{SIZE}(\text{SAT}, n) \geq n^{\omega(1)}$ for general circuits.*

---

## 9. Axiom R: Recovery and the P vs NP Barrier

### 9.1. The Core Dichotomy

**Theorem 9.1.1** (Recovery Characterization). *P = NP if and only if Axiom R holds polynomially for NP:*

*For every $L \in$ NP with witness relation $R$, there exists polynomial-time $S$ such that:*
$$x \in L \Rightarrow R(x, S(x)) = 1$$

*Proof.*
$(\Rightarrow)$ If P = NP, search reduces to decision. Given oracle for $L$, recover witness bit-by-bit by self-reduction.

$(\Leftarrow)$ If witnesses are polynomial-time recoverable, then $x \in L$ iff $S(x)$ is a valid witness, decidable in P. $\square$

**Invocation 9.1.2** (Metatheorem 7.6). *P $\neq$ NP is equivalent to Axiom R failure:*
$$\text{Witness recovery requires super-polynomial resources}$$

### 9.2. Search vs Decision

**Theorem 9.2.1** (Self-Reducibility). *For NP-complete problems, search reduces to decision:*
$$L \in \text{P} \Leftrightarrow \text{witnesses for } L \text{ findable in P}$$

*Proof.* For SAT: given SAT oracle, fix variables one by one, checking satisfiability at each step. This recovers a satisfying assignment in polynomial time. $\square$

### 9.3. Witness Complexity

**Definition 9.3.1** (Witness Complexity). *For $L \in$ NP:*
$$W(L,x) = \min\{|w| : R(x,w) = 1\} \quad \text{for } x \in L$$

**Theorem 9.3.2** (Witness Lower Bounds). *If P $\neq$ NP, then for NP-complete $L$:*
$$\text{No polynomial-time algorithm computes witnesses}$$

---

## 10. Axiom TB: Topological Background for Complexity

### 10.1. The Boolean Cube

**Definition 10.1.1** (Boolean Cube). *The $n$-dimensional Boolean cube is $\{0,1\}^n$ with Hamming metric:*
$$d_H(x,y) = |\{i : x_i \neq y_i\}|$$

**Proposition 10.1.2** (Cube Properties).
- *$2^n$ vertices*
- *Regular degree $n$*
- *Diameter $n$*

**Invocation 10.1.3** (Metatheorem 7.7.1). *Axiom TB satisfied: the Boolean cube provides stable combinatorial background.*

### 10.2. Complexity Classes as Topological Objects

**Definition 10.2.1** (Complexity Class Topology). *Equip complexity classes with the metric:*
$$d(L_1, L_2) = \limsup_{n \to \infty} \frac{|L_1 \triangle L_2 \cap \{0,1\}^n|}{2^n}$$

**Proposition 10.2.2**. *This defines a pseudometric; classes at distance 0 are "essentially equal" (differ on negligible fraction).*

---

## 11. Barriers to Proving P $\neq$ NP

### 11.1. Relativization

**Theorem 11.1.1** (Baker-Gill-Solovay 1975). *There exist oracles $A$ and $B$ such that:*
- *$\text{P}^A = \text{NP}^A$*
- *$\text{P}^B \neq \text{NP}^B$*

**Corollary 11.1.2** (Relativization Barrier). *No proof technique that relativizes can resolve P vs NP.*

### 11.2. Natural Proofs

**Definition 11.2.1** (Natural Proof). *A natural proof against circuit class $\mathcal{C}$ is a property $P$ of Boolean functions such that:*
1. *(Constructivity) $P$ decidable in $2^{O(n)}$ time*
2. *(Largeness) $P$ holds for $\geq 2^{-O(n)}$ fraction of functions*
3. *(Usefulness) $P(f) \Rightarrow f \notin \mathcal{C}$*

**Theorem 11.2.2** (Razborov-Rudich 1997). *If one-way functions exist, no natural proof shows NP $\not\subseteq$ P/poly.*

### 11.3. Algebrization

**Definition 11.3.1** (Algebraic Extension). *An algebraic extension of a language $L$ is $\tilde{L}$ defined over a field extension.*

**Theorem 11.3.2** (Aaronson-Wigderson 2009). *There exist oracles $A$ with algebraic extensions such that $\text{P}^A = \text{NP}^A$. No algebrizing proof can separate P from NP.*

### 11.4. Hypostructure Interpretation

**Theorem 11.4.1** (Barriers as Axiom Constraints). *The barriers correspond to axiom restrictions:*
- *Relativization: proofs must work without Axiom TB modification*
- *Natural proofs: Axiom Cap arguments fail if OWF exist*
- *Algebrization: Axiom SC must respect algebraic structure*

**Invocation 11.4.2** (Metatheorem 9.10). *A proof of P $\neq$ NP must violate at least one barrier, requiring non-relativizing, non-natural, non-algebrizing techniques.*

---

## 12. Related Complexity Classes

### 12.1. Probabilistic Classes

**Definition 12.1.1** (BPP). *Bounded-error probabilistic polynomial time:*
$$L \in \text{BPP} \Leftrightarrow \exists \text{ PTM } M : \Pr[M(x) = L(x)] \geq 2/3$$

**Theorem 12.1.2** (Sipser-Gács-Lautemann). *BPP $\subseteq \Sigma_2^p \cap \Pi_2^p$.*

**Conjecture 12.1.3** (Derandomization). *P = BPP.*

### 12.2. Counting Classes

**Definition 12.2.1** (Sharp-P). *$\#$P is the class of functions counting witnesses:*
$$f \in \#\text{P} \Leftrightarrow f(x) = |\{w : R(x,w) = 1\}| \text{ for some NP relation } R$$

**Theorem 12.2.2** (Toda 1991). *PH $\subseteq$ P$^{\#\text{P}}$.*

### 12.3. Interactive Proofs

**Definition 12.3.1** (IP). *Interactive polynomial time: problems with polynomial-round interactive proofs.*

**Theorem 12.3.2** (Shamir 1990). *IP = PSPACE.*

**Theorem 12.3.3** (LFKN + Shamir). *For $\#$P-complete problems, interactive proofs exist.*

---

## 13. Approaches to P vs NP

### 13.1. Geometric Complexity Theory

**Definition 13.1.1** (GCT Program). *Use algebraic geometry and representation theory to prove circuit lower bounds.*

**Theorem 13.1.1** (Mulmuley-Sohoni 2001). *The permanent vs determinant question can be formulated as:*
$$\text{Orbit closure of permanent} \not\subseteq \text{Orbit closure of determinant}$$

### 13.2. Proof Complexity

**Definition 13.2.1** (Proof System). *A proof system for language $L$ is a polynomial-time function $V$ such that:*
$$x \in L \Leftrightarrow \exists \pi : V(x, \pi) = 1$$

**Theorem 13.2.2** (NP vs coNP). *NP $\neq$ coNP iff no propositional proof system has polynomial-size proofs for all tautologies.*

### 13.3. Descriptive Complexity

**Theorem 13.3.1** (Fagin 1974). *NP = $\exists$SO (existential second-order logic).*

**Theorem 13.3.2** (Immerman-Vardi 1982). *P = FO + LFP on ordered structures.*

---

## 14. Conditional Results

### 14.1. Assuming P $\neq$ NP

**Theorem 14.1.1** (Conditional Separations). *If P $\neq$ NP:*
- *NP $\neq$ coNP*
- *NP-intermediate problems exist*
- *One-way functions likely exist*

### 14.2. Assuming Stronger Hypotheses

**Definition 14.2.1** (Exponential Time Hypothesis, ETH). *SAT requires $2^{\Omega(n)}$ time.*

**Theorem 14.2.2** (ETH Consequences). *Under ETH:*
- *$k$-SAT requires $2^{\Omega(n)}$ time for $k \geq 3$*
- *Many tight lower bounds follow*

**Definition 14.2.3** (Strong ETH). *$k$-SAT requires $2^{(1-o(1))n}$ time as $k \to \infty$.*

---

## 15. The Main Theorem: P vs NP as Axiom R Threshold

### 15.1. Statement

**Theorem 15.1.1** (Main Classification). *The P versus NP problem characterizes polynomial-time Axiom R satisfaction:*

| Axiom | Class P | Class NP (general) |
|-------|---------|-------------------|
| C (Compactness) | $\checkmark$ Poly circuits | $\checkmark$ Poly verification |
| D (Dissipation) | $\checkmark$ Poly time | Search: unknown |
| SC (Scale Coherence) | Level 0 | Level 1 |
| LS (Local Stiffness) | Problem-dependent | Amplification possible |
| Cap (Capacity) | Poly bounded | Poly bounded for verification |
| R (Recovery) | $\checkmark$ **Poly** | **Unknown (= P vs NP)** |
| TB (Background) | $\checkmark$ | $\checkmark$ |

### 15.2. Proof

*Proof.*
**Class P analysis:**
- Axiom C: Problems in P have polynomial-size circuits (compactness).
- Axiom D: Algorithms halt in polynomial time (dissipation).
- Axiom SC: No quantifier alternation needed (level 0).
- Axiom Cap: Circuit size polynomially bounded.
- Axiom R: Solutions computable, hence recoverable.
- Axiom TB: Boolean cube provides stable background.

**Class NP analysis:**
- Axiom C: Verification has polynomial circuits.
- Axiom D: Verification dissipates polynomially; search unknown.
- Axiom SC: One existential quantifier (level 1).
- Axiom Cap: Witness size polynomially bounded.
- **Axiom R: Witness recovery polynomial iff P = NP.**
- Axiom TB: Same background.

The critical distinction is Axiom R for the search problem. $\square$

### 15.3. Corollaries

**Corollary 15.3.1** (P = NP Characterization). *P = NP if and only if every NP problem satisfies Axiom R with polynomial resources.*

**Corollary 15.3.2** (Separation Criterion). *To prove P $\neq$ NP, exhibit an NP problem where Axiom R provably fails for polynomial resources.*

---

## 16. Connections to Other Études

### 16.1. Halting Problem (Étude 5)

**Observation 16.1.1**. *The halting problem shows Axiom R can fail absolutely (undecidability). P vs NP asks whether Axiom R can fail for bounded resources while verification remains efficient.*

**Theorem 16.1.2** (Complexity vs Computability). *The gap between:*
- *Computability: Axiom R fails absolutely for $K$*
- *Complexity: Axiom R fails resource-wise for NP (conjecturally)*

### 16.2. Riemann Hypothesis (Étude 6)

**Observation 16.2.1**. *RH concerns optimal scale coherence. P vs NP concerns whether scale coherence at level 1 (NP) can be reduced to level 0 (P).*

### 16.3. BSD Conjecture (Étude 3)

**Observation 16.3.1**. *Computing Mordell-Weil rank is at least as hard as certain NP problems. Axiom R failure may be inherited from computational complexity.*

---

## 17. Summary and Synthesis

### 17.1. Complete Axiom Assessment

**Table 17.1.1** (Final Classification):

| Axiom | Status for NP | Obstruction |
|-------|--------------|-------------|
| C | Holds (verification) | None |
| D | Partial (verification only) | Search time unknown |
| SC | Level 1 | Single quantifier |
| LS | Problem-dependent | Some amplification |
| Cap | Holds | Polynomial witness size |
| R | **Unknown** | **= P vs NP question** |
| TB | Holds | Boolean cube stable |

### 17.2. Central Insight

**Theorem 17.2.1** (Fundamental Characterization). *The P versus NP problem is precisely the question of whether polynomial-time computation satisfies Axiom R universally over NP:*

$$\text{P} = \text{NP} \Leftrightarrow \text{Axiom R holds polynomially for all NP problems}$$

*Proof.* Axiom R for an NP problem $L$ with relation $R$ requires recovering witness $w$ from $x \in L$. This is exactly the NP search problem, which reduces to decision for NP-complete problems. P = NP iff this search is polynomial-time. $\square$

**Invocation 17.2.2** (Chapter 18 Isomorphism). *P vs NP occupies the same structural position as:*
- *Halting problem: Axiom R failure (absolute)*
- *Navier-Stokes: Recovery of smooth solutions from data*
- *RH: Recovery of primes from zeros*

*All are questions about the strength of recovery mechanisms.*

---

## 18. References

1. [C71] S.A. Cook, "The complexity of theorem proving procedures," Proc. STOC 1971, 151-158.

2. [L73] L.A. Levin, "Universal search problems," Probl. Inf. Transm. 9 (1973), 265-266.

3. [K72] R.M. Karp, "Reducibility among combinatorial problems," Complexity of Computer Computations, 1972.

4. [BGS75] T. Baker, J. Gill, R. Solovay, "Relativizations of the P=?NP question," SIAM J. Comput. 4 (1975), 431-442.

5. [RR97] A.A. Razborov, S. Rudich, "Natural proofs," J. Comput. System Sci. 55 (1997), 24-35.

6. [AW09] S. Aaronson, A. Wigderson, "Algebrization: A new barrier in complexity theory," TOCT 1 (2009), 1-54.

7. [MS01] K.D. Mulmuley, M. Sohoni, "Geometric complexity theory I," SIAM J. Comput. 31 (2001), 496-526.

8. [Sha90] A. Shamir, "IP = PSPACE," J. ACM 39 (1992), 869-877.

9. [T91] S. Toda, "PP is as hard as the polynomial-time hierarchy," SIAM J. Comput. 20 (1991), 865-877.

10. [AB09] S. Arora, B. Barak, "Computational Complexity: A Modern Approach," Cambridge University Press, 2009.
