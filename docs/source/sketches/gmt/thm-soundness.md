# THM-Soundness: Soundness — GMT Translation

## Original Statement (Hypostructure)

The permit system is sound: any conclusion derivable from valid permits is geometrically valid.

## GMT Setting

**Derivation:** $\Pi \vdash T$ — $T$ is derivable from permits $\Pi$

**Validity:** $T$ satisfies all geometric constraints specified by $\Pi$

**Soundness:** Derivable implies valid

## GMT Statement

**Theorem (Soundness).** If $\Pi \vdash T$, then $T$ satisfies all geometric constraints encoded in $\Pi$:

1. **Mass Bounds:** $\mathbf{M}(T) \leq \Lambda$

2. **Regularity:** $T$ satisfies $\varepsilon$-regularity where specified

3. **Convergence:** Flow from $T$ converges as specified by $K_{\text{LS}_\sigma}^+$

4. **Surgery:** Surgical modifications preserve validity

## Proof Sketch

### Step 1: Axiom Soundness

**Base Case:** Each axiom of the permit system corresponds to a valid geometric statement.

**Energy Dissipation Axiom ($K_{D_E}^+$):**
$$\frac{d}{dt}\Phi(T_t) \leq 0$$

**Geometric Meaning:** For gradient flow of area/energy functional, this is the energy-dissipation identity.

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.

**Compactness Axiom ($K_{C_\mu}^+$):**
$$\sup_j \mathbf{M}(T_j) < \infty \implies \text{convergent subsequence}$$

**Geometric Meaning:** Federer-Fleming compactness theorem.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.

### Step 2: Rule Soundness

**Flow Rule:** If $T_0$ is valid and $T_t = \varphi_t(T_0)$ is the gradient flow:
$$T_0 \text{ valid} \implies T_t \text{ valid for all } t \geq 0$$

*Proof:* The gradient flow preserves:
- Mass bounds (by dissipation)
- Rectifiability (by smoothing)
- Boundary conditions (by Dirichlet/Neumann preservation)

**Surgery Rule:** If $T$ is valid and surgery is admissible:
$$T \text{ valid} \land \text{Adm}(T, V) \implies T' = \mathcal{O}_S(T, V) \text{ valid}$$

*Proof:* Surgery preserves:
- Mass (by conservation)
- Regularity (by profile regularity)
- Topology (by controlled excision)

### Step 3: Inductive Soundness

**Theorem:** Every derivable statement is valid.

*Proof by induction on derivation length:*

**Base Case (length 0):** Axioms are geometrically valid (Step 1).

**Inductive Step:** Assume all derivations of length $< n$ produce valid statements.

For derivation of length $n$:
- Final step uses a rule $R$ with premises $P_1, \ldots, P_k$
- Each $P_i$ has derivation of length $< n$
- By induction, each $P_i$ is valid
- By rule soundness (Step 2), conclusion is valid

### Step 4: Semantic Interpretation

**Model:** The geometric model is:
- Objects: Integral currents $T \in \mathbf{I}_k(M)$
- Predicates: Mass bounds, regularity conditions, convergence
- Operations: Flow, surgery, blow-up

**Satisfaction:** $T \models P$ if $T$ satisfies predicate $P$ in the geometric model.

**Soundness:** $\Pi \vdash P \implies \mathbf{I}_k \models P$ under any instantiation.

### Step 5: Preservation Under Limits

**Limit Soundness:** If $T_j$ are valid and $T_j \to T_\infty$:
$$T_j \text{ valid for all } j \implies T_\infty \text{ valid}$$

*Proof:* Each validity condition is closed under limits:
- Mass: $\mathbf{M}(T_\infty) \leq \liminf_j \mathbf{M}(T_j) \leq \Lambda$
- Boundary: $\partial T_\infty = \lim_j \partial T_j$
- Regularity: By Allard compactness

**Reference:** Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.

### Step 6: Preservation Under Surgery

**Surgery Soundness:** Valid permits before surgery yield valid state after surgery.

*Proof:* Surgery construction (FACT-Surgery) ensures:
1. Profile $V \in \mathcal{L}$ is regular
2. Gluing preserves mass bounds
3. Boundary conditions match

### Step 7: No Spurious Validity

**Claim:** The permit system does not derive false geometric statements.

*Proof:* Each rule is geometrically justified:
- Flow rule: based on PDE theory
- Surgery rule: based on GMT surgery theory
- Compactness rule: based on Federer-Fleming

No rule introduces non-geometric conclusions.

### Step 8: Error Bound Preservation

**Quantitative Soundness:** If $\Pi$ specifies bounds $(C_1, \ldots, C_m)$:
$$\Pi \vdash T \implies T \text{ satisfies bounds with same constants}$$

*Proof:* Derivation rules preserve quantitative bounds:
- Flow preserves or improves regularity constants
- Surgery uses explicit profile constants
- Limits preserve bounds by lower semicontinuity

### Step 9: Completeness Partial Converse

**Partial Completeness:** For decidable properties $P$:
$$T \models P \implies \Pi \vdash P(T)$$

(under appropriate permit instantiation)

**Reference:** For completeness theorems in geometric logic, see Gödel, K. (1930). Die Vollständigkeit der Axiome des logischen Funktionenkalküls. *Monatshefte für Mathematik und Physik*, 37.

### Step 10: Compilation Theorem

**Theorem (Soundness):**

1. **Derivation implies validity:** $\Pi \vdash T \implies T$ satisfies geometric constraints

2. **Rule preservation:** Each derivation rule preserves validity

3. **Quantitative:** Bounds are preserved through derivation

4. **Limit closure:** Validity is closed under limits

**Applications:**
- Regularity theorems derived from permits are geometrically true
- Convergence results derived from permits hold
- Surgery outcomes satisfy all stated properties

## Key GMT Inequalities Used

1. **Energy-Dissipation:**
   $$\frac{d}{dt}\Phi(T_t) = -|\nabla\Phi|^2 \leq 0$$

2. **Mass Semicontinuity:**
   $$\mathbf{M}(T_\infty) \leq \liminf_j \mathbf{M}(T_j)$$

3. **Allard Regularity:**
   $$\text{tilt excess} < \varepsilon \implies \text{regular}$$

4. **Surgery Bounds:**
   $$\mathbf{M}(T') \leq \mathbf{M}(T) - \epsilon_T$$

## Literature References

- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Allard, W. K. (1972). On the first variation of a varifold. *Ann. of Math.*, 95.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
