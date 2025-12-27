---
title: "UP-CausalBarrier - AI/RL/ML Translation"
---

# UP-CausalBarrier: Causal Inference Limits in Learning

## Overview

The causal barrier theorem establishes fundamental limits on what can be learned about causal relationships from observational data alone. Without interventional data, certain causal quantities are unidentifiable, creating irreducible uncertainty barriers.

**Original Theorem Reference:** {prf:ref}`mt-up-causal-barrier`

---

## AI/RL/ML Statement

**Theorem (Causal Inference Barrier, ML Form).**
For a causal model with graph $\mathcal{G}$ and observational distribution $P(X)$:

1. **Identifiability Barrier:** The interventional distribution $P(Y | \text{do}(X = x))$ is not identifiable from $P(X)$ if:
   $$\exists \text{ unobserved confounder } U: X \leftarrow U \to Y$$

2. **Adjustment Barrier:** No purely statistical adjustment can recover causal effects under confounding.

3. **Pearl's Hierarchy:** Each level requires data from that level:
   - **Association:** $P(Y|X)$ from observations
   - **Intervention:** $P(Y|\text{do}(X))$ from experiments
   - **Counterfactual:** $P(Y_x|X=x')$ from structural knowledge

---

## Terminology Translation Table

| Hypostructure Concept | AI/RL/ML Analog | Formal Correspondence |
|-----------------------|-----------------|------------------------|
| Causal barrier | Identifiability limit | Unidentified causal effects |
| Horizon | Counterfactual boundary | What cannot be learned |
| Information bound | Data requirements | What data is needed |
| Intervention | do-operator | $P(Y|\text{do}(X))$ |
| Confounding | Hidden common cause | $X \leftarrow U \to Y$ |
| Instrumental variable | Exogenous variation | Lever for identification |

---

## Causal Learning Framework

### Pearl's Causal Hierarchy

| Level | Question | Data Needed | Example |
|-------|----------|-------------|---------|
| 1: Association | $P(Y|X)$? | Observational | Correlation |
| 2: Intervention | $P(Y|\text{do}(X))$? | Experimental | Treatment effect |
| 3: Counterfactual | $P(Y_x|X=x')$? | Structural model | What if |

### Barrier Types

| Barrier | Cause | Resolution |
|---------|-------|------------|
| Confounding | Hidden common cause | RCT, IV, adjustment |
| Selection | Biased sampling | Weighting, bounds |
| Mediator | Pathway ambiguity | Mediation analysis |
| Measurement | Proxy variables | Error modeling |

---

## Proof Sketch

### Step 1: Simpson's Paradox

**Claim:** Association can reverse under conditioning.

**Example:** Treatment-outcome association:
$$P(Y=1|X=1) > P(Y=1|X=0) \quad \text{(marginal)}$$

But stratified:
$$P(Y=1|X=1, Z=z) < P(Y=1|X=0, Z=z) \quad \forall z$$

**Barrier:** Observational $P(Y|X) \neq P(Y|\text{do}(X))$.

**Reference:** Pearl, J. (2009). *Causality*. Cambridge.

### Step 2: Confounding Barrier

**Claim:** Unobserved confounders block identification.

**Confounded Model:**
$$U \to X, \quad U \to Y, \quad X \to Y$$

**Identifiability Failure:**
$$P(Y|\text{do}(X)) \neq P(Y|X)$$

and no adjustment formula exists.

**Bound:** At best, bounds on causal effect:
$$P(Y|\text{do}(X)) \in [\text{lower}, \text{upper}]$$

**Reference:** Manski, C. (1990). Nonparametric bounds on treatment effects. *AER*.

### Step 3: Backdoor Criterion

**Claim:** Adjustment possible if backdoor paths blocked.

**Backdoor Criterion:** $Z$ satisfies backdoor relative to $(X, Y)$ if:
1. $Z$ blocks all backdoor paths $X \leftarrow \cdots \to Y$
2. $Z$ contains no descendants of $X$

**Adjustment Formula:**
$$P(Y|\text{do}(X)) = \sum_z P(Y|X, Z=z)P(Z=z)$$

**Barrier:** Criterion fails if confounders unobserved.

**Reference:** Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*.

### Step 4: Frontdoor Criterion

**Claim:** Identification possible through mediators.

**Frontdoor:** $M$ satisfies frontdoor if:
1. $X \to M \to Y$ (mediated path)
2. No unblocked backdoor $X \to M$
3. All backdoors $M \to Y$ blocked by $X$

**Frontdoor Formula:**
$$P(Y|\text{do}(X)) = \sum_m P(M=m|X)\sum_{x'}P(Y|M=m, X=x')P(X=x')$$

**Reference:** Pearl, J. (2009). *Causality*. Cambridge.

### Step 5: Instrumental Variables

**Claim:** Instruments enable identification under confounding.

**Instrument $Z$:** Valid if:
1. $Z \to X$ (relevance)
2. $Z \not\to Y$ except through $X$ (exclusion)
3. $Z \perp U$ (independence)

**IV Estimator:**
$$\text{ATE} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(X, Z)}$$

**Barrier:** Instruments must be known valid (untestable from data).

**Reference:** Angrist, J., et al. (1996). Identification of causal effects using IV. *JASA*.

### Step 6: Counterfactual Barrier

**Claim:** Counterfactuals require strongest assumptions.

**Counterfactual Question:** "What would $Y$ have been had $X=x$, given we observed $X=x'$?"

**Notation:** $P(Y_x | X=x')$

**Barrier:** Generally requires structural equations, not just graph.

**Reference:** Pearl, J. (2009). *Causality*. Cambridge.

### Step 7: RL and Causal Learning

**Claim:** RL learns causal effects through intervention.

**Policy Optimization:**
$$\pi^* = \arg\max_\pi \mathbb{E}_{s \sim P, a \sim \pi}[R(s, a)]$$

**Intervention:** Actions $a$ are interventions: $P(s'|\text{do}(a))$.

**Barrier:** Off-policy learning from observations faces confounding:
$$P(s'|a) \neq P(s'|\text{do}(a))$$

**Reference:** Bareinboim, E., Pearl, J. (2016). Causal inference and the data-fusion problem. *PNAS*.

### Step 8: Causal Discovery

**Claim:** Learning causal structure from data has limits.

**Markov Equivalence:** Graphs with same d-separations are indistinguishable:
$$\mathcal{G}_1 \sim_{\text{obs}} \mathcal{G}_2$$

**Barrier:** Only equivalence class identifiable from observations.

**Orientation:** Some edges remain unoriented.

**Reference:** Spirtes, P., et al. (2000). *Causation, Prediction, and Search*. MIT Press.

### Step 9: Bounds and Sensitivity Analysis

**Claim:** When unidentified, bounds are possible.

**Manski Bounds:** Without assumptions:
$$P(Y=1|\text{do}(X=1)) \in [P(Y=1, X=1), P(Y=1, X=1) + P(X=0)]$$

**Sensitivity Analysis:** Bound how much unmeasured confounding could change estimates.

**Reference:** Rosenbaum, P. (2002). *Observational Studies*. Springer.

### Step 10: Compilation Theorem

**Theorem (Causal Inference Barriers):**

1. **Confounding:** Unobserved common causes block identification
2. **Hierarchy:** Each causal level requires corresponding data
3. **Equivalence:** Multiple graphs explain same observations
4. **Bounds:** Without identification, only bounds are available

**Barrier Certificate:**
$$K_{\text{causal}} = \begin{cases}
\mathcal{G} & \text{causal graph} \\
\text{unobserved} & \text{confounders} \\
\text{criterion} & \text{backdoor/frontdoor/IV} \\
[\text{lower}, \text{upper}] & \text{identification bounds}
\end{cases}$$

**Applications:**
- Causal machine learning
- Off-policy RL
- Fairness in ML
- Medical AI

---

## Key AI/ML Techniques Used

1. **Do-Operator:**
   $$P(Y|\text{do}(X)) \neq P(Y|X)$$

2. **Backdoor Adjustment:**
   $$P(Y|\text{do}(X)) = \sum_z P(Y|X,z)P(z)$$

3. **IV Estimation:**
   $$\beta = \text{Cov}(Y,Z)/\text{Cov}(X,Z)$$

4. **Manski Bounds:**
   $$\text{ATE} \in [\text{lower}, \text{upper}]$$

---

## Literature References

- Pearl, J. (2009). *Causality*. Cambridge.
- Spirtes, P., et al. (2000). *Causation, Prediction, and Search*. MIT Press.
- Manski, C. (1990). Nonparametric bounds. *AER*.
- Angrist, J., et al. (1996). Identification using IV. *JASA*.
- Bareinboim, E., Pearl, J. (2016). Causal inference and data-fusion. *PNAS*.

