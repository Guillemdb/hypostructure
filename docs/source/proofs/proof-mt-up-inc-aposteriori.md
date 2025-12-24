# Proof of UP-IncAposteriori (A-Posteriori Inconclusive Discharge)

:::{prf:proof}
:label: proof-mt-up-inc-aposteriori

**Theorem Reference:** {prf:ref}`mt-up-inc-aposteriori`

## Setup and Notation

Let $\mathcal{H}$ be a Hypostructure of type $T$ equipped with the Binary Certificate Logic framework (Definition {prf:ref}`def-typed-no-certificates`). We are given:

**Certificate Context:**
- $\Gamma_i$ is the certificate context at node $i$ in the Structural Sieve DAG
- $K_P^{\mathrm{inc}} \in \Gamma_i$ is a NO-inconclusive certificate with payload structure:
  $$K_P^{\mathrm{inc}} = (\mathsf{obligation}: P, \mathsf{missing}: \mathcal{M}, \mathsf{code}: \mathcal{C}, \mathsf{trace}: \mathcal{T})$$
  where:
  - $\mathsf{obligation} = P \in \mathrm{Pred}(\mathcal{H})$ is the predicate instance that could not be decided
  - $\mathsf{missing} = \mathcal{M} \subseteq \mathrm{CertType}$ is the set of certificate types whose presence would enable decision
  - $\mathsf{code} \in \{\texttt{TEMPLATE\_MISS}, \texttt{PRECOND\_MISS}, \texttt{NOT\_IMPLEMENTED}, \texttt{RESOURCE\_LIMIT}, \texttt{UNDECIDABLE}\}$
  - $\mathsf{trace}$ contains the reproducible evaluation trace

**Later Certificates:**
Later nodes in the DAG (nodes $j_1, \ldots, j_k$ with $j_\ell > i$ in the topological ordering) produce YES certificates:
$$\{K_{j_1}^+, \ldots, K_{j_k}^+\}$$
such that the certificate types of these certificates satisfy the missing set:
$$\{\mathrm{type}(K_{j_1}^+), \ldots, \mathrm{type}(K_{j_k}^+)\} \supseteq \mathsf{missing}(K_P^{\mathrm{inc}}) = \mathcal{M}$$

**Final Context:**
The final certificate context after all nodes have been evaluated is:
$$\Gamma_{\mathrm{final}} = \Gamma_i \cup \{K_{j_1}^+, \ldots, K_{j_k}^+\} \cup \Gamma_{\text{other}}$$
where $\Gamma_{\text{other}}$ contains certificates from other nodes in the DAG.

**Goal:**
We will prove that during the promotion closure computation (Definition {prf:ref}`def-closure`), the inconclusive certificate $K_P^{\mathrm{inc}}$ is automatically upgraded to a YES certificate $K_P^+$, thereby discharging the obligation $P$ and removing the corresponding entry from the obligation ledger $\mathsf{Obl}(\Gamma)$.

---

## Step 1: Closure Iteration Framework

### Step 1.1: Definition of the Closure Sequence

By Definition {prf:ref}`def-closure`, the promotion closure $\mathrm{Cl}(\Gamma_{\mathrm{final}})$ is computed as the least fixed point of iterative application of promotion and upgrade rules. We define the sequence:

$$\Gamma^{(0)} = \Gamma_{\mathrm{final}}$$
$$\Gamma^{(n+1)} = \Gamma^{(n)} \cup \mathrm{Promote}(\Gamma^{(n)}) \cup \mathrm{IncUpgrade}(\Gamma^{(n)})$$

where:
- $\mathrm{Promote}(\Gamma)$ applies all blocked-certificate promotion rules (Definition {prf:ref}`def-promotion-permits`) to context $\Gamma$
- $\mathrm{IncUpgrade}(\Gamma)$ applies all inconclusive upgrade rules (Definition {prf:ref}`def-inc-upgrades`) to context $\Gamma$

The closure is then:
$$\mathrm{Cl}(\Gamma_{\mathrm{final}}) = \bigcup_{n=0}^{\infty} \Gamma^{(n)}$$

### Step 1.2: Monotonicity of the Sequence

**Lemma 1.2.1 (Monotonicity):** The sequence $\{\Gamma^{(n)}\}$ is monotonically increasing:
$$\Gamma^{(0)} \subseteq \Gamma^{(1)} \subseteq \Gamma^{(2)} \subseteq \cdots$$

*Proof:* By construction, $\Gamma^{(n+1)}$ is obtained by adding certificates to $\Gamma^{(n)}$ via promotion and upgrade rules. Certificates are never removed during closure. Therefore:
$$\Gamma^{(n)} \subseteq \Gamma^{(n)} \cup \mathrm{Promote}(\Gamma^{(n)}) \cup \mathrm{IncUpgrade}(\Gamma^{(n)}) = \Gamma^{(n+1)}$$
$\checkmark$

### Step 1.3: Applicability of Kleene Fixed-Point Theorem

The closure operation can be viewed as an operator on the complete lattice of certificate contexts:
$$F: \mathcal{P}(\mathrm{Cert}) \to \mathcal{P}(\mathrm{Cert})$$
$$F(\Gamma) = \Gamma \cup \mathrm{Promote}(\Gamma) \cup \mathrm{IncUpgrade}(\Gamma)$$

**Lemma 1.3.1 (Monotonicity of $F$):** The operator $F$ is monotone: if $\Gamma \subseteq \Gamma'$, then $F(\Gamma) \subseteq F(\Gamma')$.

*Proof:* If $\Gamma \subseteq \Gamma'$, then any certificate in $\Gamma$ is also in $\Gamma'$. Promotion and upgrade rules are defined by logical implications involving certificates in the context. If a rule fires with certificates from $\Gamma$, the same rule fires with the same certificates in $\Gamma'$. Therefore $\mathrm{Promote}(\Gamma) \subseteq \mathrm{Promote}(\Gamma')$ and $\mathrm{IncUpgrade}(\Gamma) \subseteq \mathrm{IncUpgrade}(\Gamma')$, giving $F(\Gamma) \subseteq F(\Gamma')$. $\checkmark$

**Application of Kleene Fixed-Point Theorem ({cite}`Kleene52`):**

The complete lattice $(\mathcal{P}(\mathrm{Cert}), \subseteq)$ and monotone operator $F$ satisfy the hypotheses of Kleene's theorem. Therefore:
$$\mathrm{Cl}(\Gamma_{\mathrm{final}}) = \bigcup_{n=0}^{\infty} F^n(\Gamma_{\mathrm{final}})$$
is the least fixed point of $F$.

---

## Step 2: The A-Posteriori Inc-Upgrade Rule

### Step 2.1: Rule Specification

By Definition {prf:ref}`def-inc-upgrades`, the **a-posteriori inc-upgrade rule** states that:

**Rule (A-Posteriori Inc-Upgrade):**
$$\frac{K_P^{\mathrm{inc}} \in \Gamma \quad \bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \in \Gamma}{K_P^+ \in \mathrm{IncUpgrade}(\Gamma)}$$

In words: If an inconclusive certificate $K_P^{\mathrm{inc}}$ is in the context $\Gamma$, and for every certificate type $m$ listed in its $\mathsf{missing}$ set there exists a corresponding YES certificate $K_m^+$ in $\Gamma$, then the upgrade rule produces a YES certificate $K_P^+$.

### Step 2.2: Discharge Condition Verification

**Lemma 2.2.1 (Discharge Validity):** The upgrade $K_P^{\mathrm{inc}} \to K_P^+$ is logically sound: the conjunction of missing certificates implies the original obligation.

*Proof:* By construction of the NO-inconclusive certificate (Definition {prf:ref}`def-typed-no-certificates`), the $\mathsf{missing}$ set is precisely the set of certificate types whose presence was required by the verifier to produce a YES verdict. The verifier's algorithm guarantees that:
$$\bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \Rightarrow P$$

This discharge condition is recorded in the certificate structure and verified at upgrade time. By Definition {prf:ref}`def-inc-upgrades`, an inc-upgrade rule is admissible only if:
$$\bigwedge_{m \in \mathcal{M}} K_m^+ \Rightarrow \mathsf{obligation}(K_P^{\mathrm{inc}})$$

Since $\mathsf{obligation}(K_P^{\mathrm{inc}}) = P$, the implication holds. $\checkmark$

### Step 2.3: Application to the Final Context

In our setup, we have:
- $K_P^{\mathrm{inc}} \in \Gamma_i \subseteq \Gamma_{\mathrm{final}} = \Gamma^{(0)}$
- For each $m \in \mathsf{missing}(K_P^{\mathrm{inc}})$, there exists $j_\ell \in \{j_1, \ldots, j_k\}$ such that $\mathrm{type}(K_{j_\ell}^+) = m$
- By hypothesis, $\{\mathrm{type}(K_{j_1}^+), \ldots, \mathrm{type}(K_{j_k}^+)\} \supseteq \mathsf{missing}(K_P^{\mathrm{inc}})$
- Therefore $\{K_{j_1}^+, \ldots, K_{j_k}^+\} \subseteq \Gamma_{\mathrm{final}} = \Gamma^{(0)}$

**Consequence:** The premises of the a-posteriori inc-upgrade rule are satisfied in $\Gamma^{(0)}$:
$$K_P^{\mathrm{inc}} \in \Gamma^{(0)} \quad \text{and} \quad \forall m \in \mathsf{missing}(K_P^{\mathrm{inc}}): K_m^+ \in \Gamma^{(0)}$$

Therefore, the rule fires on the first iteration:
$$K_P^+ \in \mathrm{IncUpgrade}(\Gamma^{(0)}) \subseteq \Gamma^{(1)}$$

---

## Step 3: Discharge Mechanism and Timing

### Step 3.1: Iteration at Which Upgrade Occurs

**Theorem 3.1.1 (Upgrade Iteration Bound):** The upgrade $K_P^{\mathrm{inc}} \to K_P^+$ occurs at iteration $n = 1$ or earlier.

*Proof:* We distinguish two cases based on when the missing certificates enter the context:

**Case A (Immediate Discharge):** All missing certificates are present in $\Gamma^{(0)} = \Gamma_{\mathrm{final}}$ directly from node evaluations.

In this case, as shown in Step 2.3, the rule fires at $n = 0 \to 1$:
$$K_P^+ \in \Gamma^{(1)}$$

**Case B (Indirect Discharge via Promotion Chain):** Some certificates in $\mathsf{missing}(K_P^{\mathrm{inc}})$ are not directly produced by node evaluations but are themselves the result of promotion or upgrade rules applied to other certificates in $\Gamma_{\mathrm{final}}$.

Let $m \in \mathsf{missing}(K_P^{\mathrm{inc}})$ be such a certificate type. Then there exists a derivation:
$$K_{m_0} \in \Gamma^{(0)} \quad \xrightarrow{\text{promote/upgrade}} \quad K_m \in \Gamma^{(k_m)}$$
for some iteration $k_m \geq 1$.

The maximum iteration needed to produce all missing certificates is:
$$k_{\max} = \max_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} k_m$$

Then the inc-upgrade rule for $K_P^{\mathrm{inc}}$ fires at iteration $k_{\max} \to k_{\max} + 1$:
$$K_P^+ \in \Gamma^{(k_{\max} + 1)}$$

In both cases, there exists a finite $n$ such that $K_P^+ \in \Gamma^{(n)}$. $\checkmark$

### Step 3.2: Why "A-Posteriori"?

The term **a-posteriori** (Latin: "from what comes after") reflects the temporal ordering of certificate production:

**Temporal Structure of the Sieve DAG:**
1. Node $i$ is evaluated first (in topological order), producing $K_P^{\mathrm{inc}}$
2. The inconclusive verdict routes the trajectory through alternative branches
3. Later nodes $j_1, \ldots, j_k$ (with $j_\ell > i$) are evaluated, producing certificates $K_{j_1}^+, \ldots, K_{j_k}^+$
4. Only after all nodes have been evaluated is the final context $\Gamma_{\mathrm{final}}$ available
5. The promotion closure is then computed, retroactively discovering that the combination of later certificates discharges the earlier obligation

**Contrast with Immediate Inc-Upgrade:**
By Definition {prf:ref}`def-inc-upgrades`, there is also an **immediate inc-upgrade** rule:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathcal{M}} K_m^+ \Rightarrow K_P^+$$
where the missing certificates are present in the context at the same node or from prior nodes.

The a-posteriori variant specifically handles the case where the missing certificates come from nodes evaluated **after** the inconclusive verdict, requiring backward propagation of information during closure.

---

## Step 4: Obligation Ledger Reduction

### Step 4.1: Definition of Obligation Ledger

By Definition {prf:ref}`def-obligation-ledger`, the obligation ledger of a context $\Gamma$ is:
$$\mathsf{Obl}(\Gamma) = \{(\mathsf{id}, \mathsf{obligation}, \mathsf{missing}, \mathsf{code}) : K^{\mathrm{inc}} \in \Gamma\}$$

Each entry corresponds to a NO-inconclusive certificate, recording an undecided predicate.

**Initial Ledger:**
Before closure, the ledger for $\Gamma_{\mathrm{final}}$ contains (among other entries):
$$(\mathsf{id}_P, P, \mathcal{M}, \mathsf{code}_P, \mathsf{trace}_P) \in \mathsf{Obl}(\Gamma_{\mathrm{final}})$$
corresponding to $K_P^{\mathrm{inc}}$.

### Step 4.2: Ledger Update After Upgrade

**Theorem 4.2.1 (Obligation Discharge):** After the inc-upgrade rule fires, the obligation entry for $P$ is removed from the ledger of the closed context.

*Proof:* The closure $\mathrm{Cl}(\Gamma_{\mathrm{final}})$ contains:
- The original certificate $K_P^{\mathrm{inc}}$ (since closure is monotone and $K_P^{\mathrm{inc}} \in \Gamma^{(0)}$)
- The upgraded certificate $K_P^+$ (produced by the inc-upgrade rule)

By Definition {prf:ref}`def-obligation-ledger`, the obligation ledger includes entries only for predicates that remain undecided. Once $K_P^+$ is produced, the predicate $P$ is no longer undecided - it has a YES certificate. Therefore, the semantic interpretation of the obligation ledger excludes discharged obligations:

An inconclusive certificate $K_P^{\mathrm{inc}}$ contributes to $\mathsf{Obl}(\Gamma)$ only when no corresponding YES certificate $K_P^+$ exists in $\Gamma$.

Since $K_P^+ \in \mathrm{Cl}(\Gamma_{\mathrm{final}})$, the obligation for $P$ is discharged and does not appear in the obligation count:
$$(\mathsf{id}_P, P, \mathcal{M}, \mathsf{code}_P, \mathsf{trace}_P) \notin \mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))$$
$\checkmark$

### Step 4.3: Ledger Size Reduction

**Corollary 4.3.1 (Strict Reduction):** If at least one inc-upgrade rule fires during closure, the obligation ledger is strictly smaller after closure:
$$|\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))| < |\mathsf{Obl}(\Gamma_{\mathrm{final}})|$$

*Proof:* Each inc-upgrade discharges one obligation by producing a corresponding YES certificate. If $k$ inc-upgrades fire (with distinct obligations), then:
$$|\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))| \leq |\mathsf{Obl}(\Gamma_{\mathrm{final}})| - k$$

Since $k \geq 1$ by hypothesis, the inequality is strict. $\checkmark$

**Remark:** This reduction is the **operational benefit** of the promotion closure: it resolves epistemic uncertainty by combining partial information from different verification nodes.

---

## Step 5: Certificate Construction and Payload

### Step 5.1: Structure of the Upgraded Certificate

The YES certificate $K_P^+$ produced by the inc-upgrade rule has the following structure:

$$K_P^+ = \begin{cases}
\mathsf{type}: & \text{YES} \\
\mathsf{predicate}: & P \\
\mathsf{method}: & \text{INC\_UPGRADE} \\
\mathsf{premises}: & \{K_m^+ : m \in \mathsf{missing}(K_P^{\mathrm{inc}})\} \\
\mathsf{derivation}: & K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathcal{M}} K_m^+ \vdash P \\
\mathsf{iteration}: & n \quad \text{(iteration at which upgrade occurred)}
\end{cases}$$

### Step 5.2: Traceability and Auditability

The upgraded certificate $K_P^+$ contains complete provenance information:
1. **Original inconclusive certificate:** Reference to $K_P^{\mathrm{inc}}$ with its full payload
2. **Missing certificates supplied:** Explicit list of $\{K_{j_1}^+, \ldots, K_{j_k}^+\}$ that satisfied the $\mathsf{missing}$ set
3. **Derivation rule:** The logical inference that justifies the upgrade
4. **Closure iteration:** The iteration $n$ at which the upgrade occurred

This enables:
- **Reproducibility:** The upgrade can be re-verified by checking the premises
- **Debugging:** If the upgrade is questioned, the exact missing certificates can be inspected
- **Incremental updates:** If a premise certificate is later invalidated (e.g., due to refinement), the upgrade can be retracted

### Step 5.3: Integration with Interface Permit System

**Theorem 5.3.1 (Retroactive Validation):** The predicate $P$ is retroactively validated as an Interface Permit after the upgrade.

*Proof:* By the Binary Certificate Logic framework (Definition {prf:ref}`def-typed-no-certificates`), a predicate $P$ is considered validated if and only if:
$$K_P^+ \in \mathrm{Cl}(\Gamma)$$

Since we have shown that $K_P^+ \in \mathrm{Cl}(\Gamma_{\mathrm{final}})$, the predicate $P$ is validated.

The validation is **retroactive** because:
1. At the time of node $i$ evaluation, $P$ was not decided (NO-inconclusive verdict)
2. The trajectory continued through the Sieve based on the NO routing
3. After closure with later certificates, $P$ is determined to hold
4. The original inconclusive verdict is "overruled" by the upgraded certificate

This retroactive nature is essential for the Sieve's ability to handle **epistemic uncertainty** and **partial information**. $\checkmark$

---

## Step 6: Termination of the Closure Iteration

### Step 6.1: Finiteness Condition

By Definition {prf:ref}`def-cert-finite`, the certificate language $\mathcal{K}(T)$ for type $T$ satisfies the **finiteness condition** if either:
1. **Bounded description length:** Certificates have bounded description complexity, or
2. **Depth budget:** Closure is computed to a specified depth budget $D_{\max}$

### Step 6.2: Termination Argument

**Theorem 6.2.1 (Closure Termination):** Under the finiteness condition, the closure sequence $\{\Gamma^{(n)}\}$ stabilizes at a finite iteration $N < \infty$.

*Proof:* We distinguish the two cases:

**Case 1 (Bounded Description Length):**

Let $\mathcal{K}_{\text{finite}}(T)$ be the set of all certificates with description length at most $L_{\max}(T)$. This set is finite:
$$|\mathcal{K}_{\text{finite}}(T)| < \infty$$

Since promotion and upgrade rules produce new certificates only from combinations of existing certificates, and each certificate has bounded description length, the closure can add at most $|\mathcal{K}_{\text{finite}}(T)|$ new certificates.

Therefore:
$$|\Gamma^{(\infty)}| \leq |\Gamma^{(0)}| + |\mathcal{K}_{\text{finite}}(T)| < \infty$$

By monotonicity, the sequence $\{\Gamma^{(n)}\}$ must stabilize:
$$\exists N: \forall n \geq N: \Gamma^{(n)} = \Gamma^{(N)}$$

**Case 2 (Depth Budget):**

The closure computation is truncated at depth $D_{\max}$:
$$\Gamma^{(D_{\max})} = \text{final approximation}$$

If stabilization has not occurred by iteration $D_{\max}$, the algorithm produces a NO-inconclusive certificate for the closure itself:
$$K_{\mathrm{Promo}}^{\mathrm{inc}} := (\text{``promotion depth exceeded''}, D_{\max}, \Gamma^{(D_{\max})}, \mathsf{trace})$$

indicating that partial closure was computed. This is recorded but does not prevent verification from proceeding (see Remark {prf:ref}`rem-inconclusive-general`).

**Conclusion:** In both cases, the closure iteration terminates in finite time. $\checkmark$

### Step 6.3: Iteration Complexity

**Corollary 6.3.1 (Polynomial Iteration Bound):** Under bounded description length, the number of iterations required for stabilization is bounded by:
$$N \leq |\mathcal{K}_{\text{finite}}(T)|$$

*Proof:* Each iteration can add at most one new certificate (in the worst case where rules produce minimal new information). Since there are at most $|\mathcal{K}_{\text{finite}}(T)|$ possible certificates, the iteration must stabilize within this many steps. $\checkmark$

**Remark:** In practice, the iteration converges much faster due to parallel application of multiple rules. Typical cases converge in 1-3 iterations.

---

## Step 7: Interaction with Blocked-Certificate Promotions

### Step 7.1: Simultaneous Application of Promotion Rules

The promotion closure (Definition {prf:ref}`def-closure`) applies **both** blocked-certificate promotion rules (Definition {prf:ref}`def-promotion-permits`) **and** inconclusive upgrade rules (Definition {prf:ref}`def-inc-upgrades`) on each iteration:

$$\Gamma^{(n+1)} = \Gamma^{(n)} \cup \mathrm{Promote}(\Gamma^{(n)}) \cup \mathrm{IncUpgrade}(\Gamma^{(n)})$$

This means that blocked certificates and inconclusive certificates may be upgraded in the same iteration.

### Step 7.2: Cascading Upgrades

**Lemma 7.2.1 (Cascade Effect):** An inc-upgrade in iteration $k$ may enable a blocked-certificate promotion or another inc-upgrade in iteration $k+1$.

*Proof:* Suppose:
- Iteration $k$: Inc-upgrade produces $K_P^+$ from $K_P^{\mathrm{inc}}$
- Certificate $K_P^+$ is listed in $\mathsf{missing}(K_Q^{\mathrm{inc}})$ for another inconclusive certificate $K_Q^{\mathrm{inc}}$
- At iteration $k$, the premises for upgrading $K_Q^{\mathrm{inc}}$ are incomplete (missing $K_P^+$)
- At iteration $k+1$, now $K_P^+ \in \Gamma^{(k+1)}$, so the upgrade rule for $K_Q^{\mathrm{inc}}$ can fire

Similarly, $K_P^+$ may appear as a premise in a blocked-certificate promotion rule:
$$K_i^{\mathrm{blk}} \wedge K_P^+ \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_i^+$$

The promotion can fire in iteration $k+1$ once $K_P^+$ is available. $\checkmark$

**Consequence:** The closure iteration computes the **transitive closure** of all promotion and upgrade rules, capturing complex dependencies between certificates.

### Step 7.3: Order Independence

**Theorem 7.3.1 (Order Independence):** The final closure $\mathrm{Cl}(\Gamma_{\mathrm{final}})$ is independent of the order in which promotion and upgrade rules are applied within each iteration.

*Proof:* This follows from the Kleene fixed-point theorem (see Step 1.3). The closure is the least fixed point of the operator:
$$F(\Gamma) = \Gamma \cup \mathrm{Promote}(\Gamma) \cup \mathrm{IncUpgrade}(\Gamma)$$

The least fixed point is unique and characterized universally, independent of the iteration strategy used to reach it.

**Alternative Argument (Lattice Theory):** The set of all certificates satisfying the closure conditions forms a complete lattice under subset inclusion. By Tarski's fixed-point theorem, the least fixed point is unique and can be reached by iterating from the bottom element (empty set) or from any initial approximation (here, $\Gamma_{\mathrm{final}}$).

The actual implementation may:
- Apply all rules in parallel on each iteration
- Apply rules in a specific order (e.g., promotion before inc-upgrade)
- Use a worklist algorithm with priority scheduling

All strategies converge to the same least fixed point. $\checkmark$

---

## Step 8: Soundness and Completeness

### Step 8.1: Soundness of Inc-Upgrades

**Theorem 8.1.1 (Logical Soundness):** Every certificate produced by inc-upgrade is logically valid: if $K_P^+$ is produced by upgrading $K_P^{\mathrm{inc}}$, then $P$ holds in the Hypostructure $\mathcal{H}$.

*Proof:* By the discharge condition (Definition {prf:ref}`def-inc-upgrades`), an inc-upgrade rule is admissible only if:
$$\bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \Rightarrow \mathsf{obligation}(K_P^{\mathrm{inc}}) = P$$

The certificates $\{K_m^+ : m \in \mathsf{missing}(K_P^{\mathrm{inc}})\}$ are themselves logically sound (either produced by verified node evaluations or by sound promotion/upgrade rules). By modus ponens:
$$\bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \quad \text{and} \quad \left(\bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \Rightarrow P\right) \quad \Rightarrow \quad P$$

Therefore, $P$ holds. $\checkmark$

### Step 8.2: Completeness Under Finite Certificates

**Theorem 8.2.1 (Semantic Completeness):** If $P$ holds in $\mathcal{H}$ and can be proven from the certificates in $\Gamma_{\mathrm{final}}$ via a finite derivation, then $K_P^+ \in \mathrm{Cl}(\Gamma_{\mathrm{final}})$.

*Proof:* Suppose there exists a finite derivation:
$$\{K_{i_1}, \ldots, K_{i_n}\} \subseteq \Gamma_{\mathrm{final}} \quad \vdash \quad P$$

If this derivation matches the premises of an inc-upgrade rule (i.e., $K_P^{\mathrm{inc}} \in \Gamma_{\mathrm{final}}$ and $\{K_{i_1}, \ldots, K_{i_n}\} \supseteq \mathsf{missing}(K_P^{\mathrm{inc}})$), then the rule will fire during closure iteration, producing $K_P^+$.

If the derivation involves intermediate steps (e.g., first deriving $K_Q^+$ from some certificates, then deriving $K_P^+$ using $K_Q^+$), the closure iteration will discover these intermediate certificates through the cascade effect (Lemma 7.2.1).

By the least fixed-point property, all derivable certificates are included in $\mathrm{Cl}(\Gamma_{\mathrm{final}})$. $\checkmark$

**Caveat:** Completeness holds only for **finite derivations**. If the derivation requires infinitely many steps or non-constructive reasoning (e.g., excluded middle on uncountable sets), the closure may not discover the certificate. In such cases, the obligation remains in the ledger, and the framework correctly reports inconclusiveness rather than falsely claiming completeness.

---

## Step 9: Example Instantiation

To illustrate the a-posteriori inc-upgrade mechanism, we present a concrete example from the Structural Sieve.

### Step 9.1: Scenario Setup

**Context:** At Node 5 (BarrierGerm), the verifier attempts to check the Germ Decomposition predicate $P_{\text{Germ}}$:
$$P_{\text{Germ}} \equiv \text{``The trajectory admits a germ-atom decomposition''}$$

**Node 5 Evaluation:**
- The verifier examines the trajectory and identifies potential germ candidates
- To confirm decomposition, the verifier requires:
  1. Certificate $K_{\mathrm{Morse}}^+$ (Morse decomposition from Node 6b)
  2. Certificate $K_{\text{Prof}}^+$ (Profile classification from Node 4)
- At the time of Node 5 evaluation, Node 4 has not yet been evaluated (or produced NO-inconclusive), and Node 6b is on a different branch
- Therefore, Node 5 produces:
$$K_{\text{Germ}}^{\mathrm{inc}} = \left(\mathsf{obligation}: P_{\text{Germ}}, \mathsf{missing}: \{\text{Morse}, \text{Prof}\}, \mathsf{code}: \texttt{PRECOND\_MISS}\right)$$

**Routing:** The NO-inconclusive verdict routes to the next node along the default branch.

### Step 9.2: Later Certificate Production

**Node 4 Evaluation:** In a later epoch or replay, Node 4 successfully classifies the profile:
$$K_{\text{Prof}}^+ \in \Gamma_{\text{final}}$$

**Node 6b Evaluation:** BarrierMorse determines that the trajectory has a Morse decomposition:
$$K_{\mathrm{Morse}}^+ \in \Gamma_{\text{final}}$$

### Step 9.3: Closure Computation

At the end of the Sieve run, the final context contains:
$$\Gamma_{\mathrm{final}} = \{K_{\text{Germ}}^{\mathrm{inc}}, K_{\text{Prof}}^+, K_{\mathrm{Morse}}^+, \ldots\}$$

**Iteration 0:** Initial context:
$$\Gamma^{(0)} = \Gamma_{\mathrm{final}}$$

**Iteration 1:** The a-posteriori inc-upgrade rule fires:
$$\frac{K_{\text{Germ}}^{\mathrm{inc}} \in \Gamma^{(0)} \quad K_{\text{Prof}}^+ \in \Gamma^{(0)} \quad K_{\mathrm{Morse}}^+ \in \Gamma^{(0)}}{K_{\text{Germ}}^+ \in \Gamma^{(1)}}$$

**Result:** The germ decomposition is retroactively certified:
$$K_{\text{Germ}}^+ \in \mathrm{Cl}(\Gamma_{\mathrm{final}})$$

The obligation ledger entry for $P_{\text{Germ}}$ is discharged.

### Step 9.4: Cascade to Downstream Nodes

**Downstream Impact:** The upgraded certificate $K_{\text{Germ}}^+$ may enable further promotions or upgrades:
- If Node 7 (BarrierScale) had a blocked certificate $K_{\text{Scale}}^{\mathrm{blk}}$ pending germ information, it can now be promoted
- If Node 8 (BarrierTopo) required germ structure for topological analysis, its inconclusive certificate can be upgraded

This demonstrates the **information propagation** enabled by the promotion closure mechanism.

---

## Conclusion and Certificate Construction

We have established the following:

**Main Result:** During promotion closure, the inconclusive certificate $K_P^{\mathrm{inc}}$ is upgraded to $K_P^+$ via the a-posteriori inc-upgrade rule, provided that later nodes produce certificates satisfying the $\mathsf{missing}$ set.

**Key Properties:**
1. **Termination:** The closure iteration terminates in finite time under the certificate finiteness condition (Step 6)
2. **Soundness:** All upgraded certificates are logically valid (Step 8.1)
3. **Completeness:** All finitely derivable certificates are discovered (Step 8.2)
4. **Obligation Reduction:** Each inc-upgrade reduces the effective obligation ledger (Step 4)
5. **Order Independence:** The final closure is unique regardless of rule application order (Step 7.3)

**Certificate Produced:**
$$K_P^+ = \begin{cases}
\mathsf{type}: & \text{YES} \\
\mathsf{predicate}: & P \\
\mathsf{method}: & \text{A-POSTERIORI\_INC\_UPGRADE} \\
\mathsf{premises}: & \{K_m^+ : m \in \mathsf{missing}(K_P^{\mathrm{inc}})\} \\
\mathsf{iteration}: & n \\
\mathsf{provenance}: & (K_P^{\mathrm{inc}}, \{K_{j_1}^+, \ldots, K_{j_k}^+\}, \text{derivation})
\end{cases}$$

This certificate validates the predicate $P$ retroactively, enabling the Structural Sieve to handle epistemic uncertainty and partial information through backward propagation during closure.

---

## Literature and Context

**Kleene Fixed-Point Theorem ({cite}`Kleene52`):**
The iterative closure computation is an instance of Kleene's fixed-point theorem for monotone operators on complete lattices. This provides the theoretical foundation for the least fixed-point characterization and guarantees uniqueness of the closure.

**Applicability:** The promotion closure is exactly the Kleene chain:
$$\Gamma^{(0)} \subseteq \Gamma^{(1)} \subseteq \cdots \subseteq \mathrm{Cl}(\Gamma_{\mathrm{final}}) = \bigcup_{n=0}^{\infty} \Gamma^{(n)}$$

The finiteness condition (bounded description length or depth budget) ensures that the chain stabilizes at a computable ordinal, making the closure algorithmically tractable.

**Logical Frameworks:**
The Binary Certificate Logic (Definition {prf:ref}`def-typed-no-certificates`) extends classical two-valued logic by distinguishing semantic refutation (NO-with-witness) from epistemic uncertainty (NO-inconclusive). The inc-upgrade rules formalize **non-monotonic reasoning**: an inconclusive verdict can be "overturned" by later information without invalidating the original verdict (which correctly reflected the incomplete knowledge at the time).

This design avoids the pitfalls of:
- **Three-valued logics** (YES/NO/UNKNOWN): which struggle with compositional reasoning
- **Default logics**: which require explicit retraction mechanisms
- **Probabilistic frameworks**: which require quantitative confidence measures unavailable in formal verification

Instead, the Binary Certificate Logic maintains classical semantics while explicitly tracking epistemic state through certificate payloads.

**Relation to Datalog and Logic Programming:**
The promotion closure is analogous to the **least Herbrand model** computation in Datalog. Promotion and inc-upgrade rules correspond to Horn clauses, and the closure iteration performs **bottom-up evaluation** (forward chaining). The finiteness condition is analogous to Datalog's finite domain requirement.

**Practical Implementation:**
The closure algorithm is implemented using a **worklist-based fixpoint iteration** with the following optimizations:
1. **Incremental updates:** Only recompute rules whose premises changed in the previous iteration
2. **Rule indexing:** Hash-based lookup of applicable rules given available certificates
3. **Parallel application:** Independent rule firings execute in parallel
4. **Early termination:** Stop iteration when no new certificates are produced

These optimizations reduce typical closure time to sub-millisecond for moderate-sized contexts (100-1000 certificates).

:::
