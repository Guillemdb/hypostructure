# UP-Absorbing: Absorbing Boundary Promotion — GMT Translation

## Original Statement (Hypostructure)

The absorbing boundary promotion shows that flow eventually enters an absorbing region and never leaves, ensuring convergence to the attractor.

## GMT Setting

**Absorbing Set:** $B_0 \subset \mathbf{I}_k(M)$ such that flow enters and stays

**Trapping Region:** $\varphi_t(B_0) \subset B_0$ for $t \geq 0$

**Promotion:** Local absorption implies global

## GMT Statement

**Theorem (Absorbing Boundary Promotion).** If:

1. **Energy Bound:** $\Phi(T) \leq \Lambda$ for all initial data

2. **Dissipation:** $\frac{d}{dt}\Phi(T_t) \leq -c \mathfrak{D}(T_t)$

3. **Coercivity:** $\Phi(T) \to \infty$ as $\|T\| \to \infty$

Then:
- **Absorbing Set Exists:** $B_0 = \{T : \Phi(T) \leq R_0\}$ is absorbing
- **Absorption Time:** $t_{B_0} \leq C(\Lambda - R_0)/c$
- **Attractor:** $\omega(B_0)$ is the global attractor

## Proof Sketch

### Step 1: Absorbing Set Definition

**Definition:** $B_0 \subset X$ is **absorbing** if:
$$\forall B \text{ bounded}, \exists t_B : t \geq t_B \implies \varphi_t(B) \subset B_0$$

**Energy Ball:**
$$B_0 := \{T \in \mathbf{I}_k(M) : \Phi(T) \leq R_0\}$$

for appropriate $R_0 > \Phi_{\min}$.

**Reference:** Temam, R. (1988). *Infinite-Dimensional Dynamical Systems*. Springer.

### Step 2: Dissipation Implies Absorption

**Energy Decay:** By dissipation:
$$\Phi(T_t) \leq \Phi(T_0) - c \int_0^t \mathfrak{D}(T_s) \, ds$$

**Absorption Time:** For $\Phi(T_0) = \Lambda$:
$$\Phi(T_{t_B}) \leq R_0 \implies t_B \leq \frac{\Lambda - R_0}{c \inf \mathfrak{D}}$$

**Entry:** Once $\Phi(T_t) \leq R_0$, the trajectory stays in $B_0$ (energy monotone).

### Step 3: Positively Invariant Sets

**Positive Invariance:** $B_0$ is **positively invariant**:
$$\varphi_t(B_0) \subset B_0 \quad \forall t \geq 0$$

*Proof:* $T \in B_0 \Rightarrow \Phi(T) \leq R_0 \Rightarrow \Phi(\varphi_t(T)) \leq \Phi(T) \leq R_0 \Rightarrow \varphi_t(T) \in B_0$.

### Step 4: Trapping Region

**Trapping:** $B_0$ is a **trapping region**:
- Absorbing: all bounded sets enter
- Positively invariant: once in, stays in

**Compactness:** If $B_0$ is precompact (by $K_{C_\mu}^+$), the omega-limit set is well-defined.

### Step 5: Omega-Limit and Attractor

**Omega-Limit Set:**
$$\omega(B_0) := \bigcap_{s \geq 0} \overline{\bigcup_{t \geq s} \varphi_t(B_0)}$$

**Properties:**
- Non-empty (nested compact sets)
- Compact
- Invariant: $\varphi_t(\omega(B_0)) = \omega(B_0)$

**Global Attractor:** $\mathcal{A} = \omega(B_0)$ attracts all bounded sets.

**Reference:** Hale, J. K. (1988). *Asymptotic Behavior of Dissipative Systems*. AMS.

### Step 6: Quantitative Absorption

**Theorem:** If $\mathfrak{D}(T) \geq c_0 > 0$ for $\Phi(T) > R_0$:
$$t_{B_0}(B) \leq \frac{\sup_B \Phi - R_0}{c_0}$$

**Uniform Absorption:** For balls $B_R = \{\|T\| \leq R\}$:
$$t_{B_0}(B_R) \leq C(R)$$

with explicit dependence on $R$.

### Step 7: Lyapunov Function

**Lyapunov Function:** $V: X \to \mathbb{R}$ with:
1. $V \geq 0$
2. $V(T_t) \leq V(T_0)$ for $t \geq 0$
3. $V(T) \to \infty$ as $\|T\| \to \infty$

**Sublevel Sets:** $\{V \leq c\}$ are absorbing and positively invariant.

**Reference:** LaSalle, J. P. (1976). *The Stability of Dynamical Systems*. SIAM.

### Step 8: Nested Absorbing Sets

**Filtration:** Absorbing sets can be nested:
$$B_{R_1} \subset B_{R_2} \subset \cdots \subset B_0$$

with $R_1 < R_2 < \cdots < R_0$.

**Attractivity:** Flow moves inward through nested sets:
$$\varphi_t(B_0) \subset B_{R_1} \text{ for large } t$$

**Asymptotic:** $\mathcal{A} = \bigcap_{n} B_{R_n}$

### Step 9: Boundary Behavior

**Absorbing Boundary:** $\partial B_0 = \{T : \Phi(T) = R_0\}$

**Inward Pointing:** The gradient $\nabla \Phi$ points inward:
$$\langle -\nabla \Phi(T), \nu \rangle > 0 \text{ for } T \in \partial B_0$$

where $\nu$ is outward normal.

**No Escape:** Trajectories on $\partial B_0$ move into interior of $B_0$.

### Step 10: Compilation Theorem

**Theorem (Absorbing Boundary Promotion):**

1. **Existence:** Energy dissipation implies absorbing set exists

2. **Absorption Time:** $t_{B_0} \leq C(\Lambda - R_0)/c$

3. **Attractor:** $\mathcal{A} = \omega(B_0)$ is global attractor

4. **Quantitative:** Absorption time computable from energy bounds

**Applications:**
- Long-time behavior of geometric flows
- Convergence to equilibria
- Global attractor structure

## Key GMT Inequalities Used

1. **Energy Dissipation:**
   $$\frac{d}{dt}\Phi(T_t) \leq -c\mathfrak{D}(T_t)$$

2. **Absorption Time:**
   $$t_{B_0} \leq (\Phi_0 - R_0)/c$$

3. **Positive Invariance:**
   $$\varphi_t(B_0) \subset B_0$$

4. **Omega-Limit:**
   $$\omega(B_0) = \bigcap_{s \geq 0} \overline{\bigcup_{t \geq s} \varphi_t(B_0)}$$

## Literature References

- Temam, R. (1988). *Infinite-Dimensional Dynamical Systems*. Springer.
- Hale, J. K. (1988). *Asymptotic Behavior of Dissipative Systems*. AMS.
- LaSalle, J. P. (1976). *The Stability of Dynamical Systems*. SIAM.
- Raugel, G. (2002). Global attractors in partial differential equations. *Handbook of Dynamical Systems*, Vol. 2.
