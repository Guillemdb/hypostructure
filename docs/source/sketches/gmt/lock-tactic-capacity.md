# LOCK-TacticCapacity: Capacity Barrier Lock — GMT Translation

## Original Statement (Hypostructure)

The capacity barrier lock shows that capacity bounds create barriers preventing flow into regions of insufficient capacity, locking out certain configurations.

## GMT Setting

**Capacity:** Sobolev $(1,p)$-capacity or Hausdorff content

**Barrier:** Region with capacity below threshold

**Lock:** Flow cannot enter capacity-deficient regions

## GMT Statement

**Theorem (Capacity Barrier Lock).** For gradient flow on $\mathbf{I}_k(M)$:

1. **Capacity Threshold:** $\text{Cap}_{1,p}(\text{sing}(T)) \leq \kappa$ required for regularity

2. **Barrier Region:** $\mathcal{B} = \{T : \text{Cap}_{1,p}(\text{sing}(T)) > \kappa\}$

3. **Lock:** Flow with $T_0 \notin \mathcal{B}$ cannot enter $\mathcal{B}$

## Proof Sketch

### Step 1: Sobolev Capacity

**Definition:** For $E \subset \mathbb{R}^n$, the $(1,p)$-capacity:
$$\text{Cap}_{1,p}(E) = \inf \left\{ \|u\|_{W^{1,p}}^p : u \geq 1 \text{ on } E, u \in C_c^\infty \right\}$$

**Reference:** Adams, D. R., Hedberg, L. I. (1996). *Function Spaces and Potential Theory*. Springer.

### Step 2: Capacity and Hausdorff Measure

**Relationship:** For $p < n$:
$$\text{Cap}_{1,p}(E) = 0 \iff \mathcal{H}^{n-p}(E) = 0$$

**Reference:** Reshetnyak, Yu. G. (1989). *Space Mappings with Bounded Distortion*. AMS.

**Consequence:** Capacity measures effective $(n-p)$-dimensional size.

### Step 3: Capacity and Singular Sets

**Regularity Theory:** For $T \in \mathbf{I}_k(M)$:
$$\text{dim}_H(\text{sing}(T)) \leq k - 2$$

**Capacity Bound:**
$$\text{Cap}_{1,2}(\text{sing}(T)) \leq C \cdot \mathcal{H}^{k-2}(\text{sing}(T))$$

**Reference:** Federer, H. (1970). The singular sets of area minimizing rectifiable currents. *Bull. AMS*, 76.

### Step 4: Capacity Monotonicity Under Flow

**Lemma:** For mean curvature flow, singular capacity is monotonic:
$$\frac{d}{dt} \text{Cap}_{1,p}(\text{sing}(T_t)) \leq 0$$

in weak sense.

*Sketch:*
- Flow smooths: singularities cannot spontaneously form
- Capacity decreases as singular set shrinks
- Only surgery can increase singular capacity

### Step 5: Barrier Definition

**Capacity Barrier:** Define:
$$\mathcal{B}_\kappa = \{T \in \mathbf{I}_k(M) : \text{Cap}_{1,p}(\text{sing}(T)) > \kappa\}$$

**Open Set:** $\mathcal{B}_\kappa$ is open in appropriate topology (capacity is upper semicontinuous).

### Step 6: Lock by Monotonicity

**Theorem:** If $T_0 \notin \mathcal{B}_\kappa$ and $\{T_t\}$ is smooth flow without surgery:
$$T_t \notin \mathcal{B}_\kappa \quad \text{for all } t \geq 0$$

*Proof:*
1. $\text{Cap}(\text{sing}(T_0)) \leq \kappa$
2. Monotonicity: $\text{Cap}(\text{sing}(T_t)) \leq \text{Cap}(\text{sing}(T_0))$
3. Therefore $\text{Cap}(\text{sing}(T_t)) \leq \kappa$
4. Hence $T_t \notin \mathcal{B}_\kappa$

### Step 7: Capacity and Removable Singularities

**Theorem (Serrin):** Singularities of capacity zero are removable for elliptic equations.

**Reference:** Serrin, J. (1964). Removable singularities of solutions of elliptic equations. *Arch. Rational Mech. Anal.*, 17, 67-78.

**GMT Version:** If $\text{Cap}_{1,2}(\text{sing}(T)) = 0$, then $T$ extends to regular current.

### Step 8: Capacity in Potential Theory

**Wiener Criterion:** Point $x$ is regular for Dirichlet problem iff:
$$\sum_{j=1}^\infty 2^{j(n-2)} \text{Cap}_{1,2}(B_{2^{-j}}(x) \cap E^c) = \infty$$

**Reference:** Wiener, N. (1924). The Dirichlet problem. *J. Math. Phys.*, 3, 127-146.

**Barrier Interpretation:** Capacity criterion determines accessibility.

### Step 9: Upper Semicontinuity

**Theorem:** The map $T \mapsto \text{Cap}_{1,p}(\text{sing}(T))$ is upper semicontinuous in flat topology.

*Sketch:*
- If $T_j \to T$ in flat norm
- Then $\limsup_{j} \text{sing}(T_j) \subset \text{sing}(T)$ in appropriate sense
- Capacity is outer measure, hence upper semicontinuous

### Step 10: Compilation Theorem

**Theorem (Capacity Barrier Lock):**

1. **Capacity:** $\text{Cap}_{1,p}$ measures effective size of singular set

2. **Monotonicity:** Capacity non-increasing under regular flow

3. **Barrier:** $\mathcal{B}_\kappa = \{\text{Cap} > \kappa\}$ is invariant region boundary

4. **Lock:** Flow cannot enter $\mathcal{B}_\kappa$ from outside

**Applications:**
- Regularity preservation
- Barrier construction for flows
- Singular set control

## Key GMT Inequalities Used

1. **Capacity-Hausdorff:**
   $$\text{Cap}_{1,p}(E) \sim \mathcal{H}^{n-p}(E)$$

2. **Monotonicity:**
   $$\text{Cap}(\text{sing}(T_t)) \leq \text{Cap}(\text{sing}(T_0))$$

3. **Removability:**
   $$\text{Cap}_{1,2}(E) = 0 \implies E \text{ removable}$$

4. **Singular Dimension:**
   $$\dim(\text{sing}(T)) \leq k-2 \implies \text{Cap}_{1,2}(\text{sing}(T)) < \infty$$

## Literature References

- Adams, D. R., Hedberg, L. I. (1996). *Function Spaces and Potential Theory*. Springer.
- Reshetnyak, Yu. G. (1989). *Space Mappings with Bounded Distortion*. AMS.
- Federer, H. (1970). Singular sets of area minimizing currents. *Bull. AMS*, 76.
- Serrin, J. (1964). Removable singularities. *Arch. Rational Mech. Anal.*, 17.
- Wiener, N. (1924). The Dirichlet problem. *J. Math. Phys.*, 3.
