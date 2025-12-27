# THM-EpochTermination: Epoch Termination — GMT Translation

## Original Statement (Hypostructure)

Each resolution epoch (period between surgeries or significant events) terminates in finite time, ensuring the flow makes definite progress.

## GMT Setting

**Epoch:** Time interval $[t_i, t_{i+1}]$ between consecutive surgeries

**Termination:** Flow reaches surgery threshold or equilibrium in finite time

**Progress:** Energy decreases by definite amount each epoch

## GMT Statement

**Theorem (Epoch Termination).** Under soft permits, each epoch $[t_i, t_{i+1}]$ satisfies:

1. **Finite Duration:** $t_{i+1} - t_i \leq T_{\max}(\Lambda, n, \theta)$

2. **Energy Progress:** $\Phi(T_{t_{i+1}}) \leq \Phi(T_{t_i}) - \epsilon_T$

3. **Alternative Termination:** Either surgery occurs or equilibrium is reached

## Proof Sketch

### Step 1: Energy Dissipation Rate

**Gradient Flow:** The flow satisfies:
$$\frac{d}{dt}\Phi(T_t) = -|\nabla \Phi|^2(T_t) = -\mathfrak{D}(T_t)$$

**Dissipation Bound:** By Łojasiewicz-Simon inequality:
$$\mathfrak{D}(T) \geq c |\Phi(T) - \Phi_*|^{2(1-\theta)}$$

for $T$ near equilibrium $T_*$.

**Reference:** Simon, L. (1983). Asymptotics for a class of nonlinear evolution equations. *Ann. of Math.*, 118, 525-571.

### Step 2: Time Bound Near Equilibrium

**Simon's Estimate (1983):** If $\Phi(T_0) - \Phi_* = \delta$ small:
$$t_{\text{equilibrium}} \leq C \delta^{-(1-2\theta)}$$

*Proof:* Integrate the Łojasiewicz inequality:
$$\int_0^{t_*} |\nabla \Phi| \, dt \leq C |\Phi_0 - \Phi_*|^{\theta}$$

**Reference:** Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.

### Step 3: Time Bound Away from Equilibrium

**Away from Critical Set:** If $d(T_t, \text{Crit}(\Phi)) \geq \delta_0$:
$$\mathfrak{D}(T_t) \geq c_0 > 0$$

by continuity of $|\nabla \Phi|$ on compact sets.

**Time to Surgery:** The time to reach surgery threshold $\Phi_{\text{surg}}$:
$$t_{\text{surg}} \leq \frac{\Phi(T_0) - \Phi_{\text{surg}}}{c_0}$$

### Step 4: Surgery Threshold Detection

**Curvature Blowup (Perelman, 2003):** For Ricci flow, surgery is triggered when:
$$\max_x |Rm|(x, t) \geq r_{\text{surg}}^{-2}$$

**Reference:** Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.

**Mean Curvature Flow (Huisken-Sinestrari, 2009):**
$$\max_x |A|(x, t) \geq H_{\text{surg}}$$

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.

**Threshold Crossing:** By continuity, threshold is crossed in finite time or never (equilibrium).

### Step 5: Dichotomy: Surgery or Equilibrium

**Trichotomy at Epoch End:**

1. **Surgery:** Curvature reaches threshold → perform surgery → new epoch

2. **Equilibrium:** Flow converges to critical point → epoch terminates forever

3. **Infinite Time:** Flow takes infinite time to reach threshold → excluded by energy bounds

**Exclusion of Case 3:** If flow takes infinite time without surgery:
$$\int_0^\infty \mathfrak{D}(T_t) \, dt = \Phi(T_0) - \Phi_\infty < \infty$$

By Łojasiewicz, this implies convergence to equilibrium (Case 2).

### Step 6: Energy Drop Per Epoch

**Minimum Energy Drop (Hamilton, 1997):**
$$\Phi(T_{t_i}) - \Phi(T_{t_{i+1}}) \geq \epsilon_T$$

where:
- $\epsilon_T = c(n) \cdot r_{\text{surg}}^n$ for surgery epochs
- $\epsilon_T = $ Łojasiewicz gap for equilibrium epochs

**Reference:** Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.

### Step 7: Epoch Count Bound

**Theorem:** The total number of epochs is bounded:
$$N_{\text{epochs}} \leq \frac{\Phi(T_0) - \Phi_{\min}}{\epsilon_T}$$

*Proof:* Each epoch drops energy by $\geq \epsilon_T$. Total available energy is $\Phi(T_0) - \Phi_{\min}$.

### Step 8: Uniform Time Bound

**Theorem:** Each epoch has duration $\leq T_{\max}$.

*Proof:* Combine:
1. Near equilibrium: Simon's time bound
2. Away from equilibrium: $t \leq (\Phi_0 - \Phi_{\text{surg}})/c_0$
3. Near surgery: curvature blowup in finite time (by scaling argument)

$$T_{\max} = \max\{C \delta^{-(1-2\theta)}, (\Phi_0 - \Phi_{\text{surg}})/c_0, T_{\text{blowup}}\}$$

### Step 9: Zeno Prevention

**No Accumulation:** Surgery times $\{t_i\}$ cannot accumulate:
$$\inf_i (t_{i+1} - t_i) \geq t_{\min} > 0$$

*Proof:* Each surgery requires:
1. Energy drop $\geq \epsilon_T$
2. Curvature to reach threshold (takes positive time from post-surgery state)

### Step 10: Compilation Theorem

**Theorem (Epoch Termination):**

1. **Finite Duration:** Each epoch $[t_i, t_{i+1}]$ has $t_{i+1} - t_i \leq T_{\max}$

2. **Definite Progress:** $\Phi(T_{t_{i+1}}) - \Phi(T_{t_i}) \leq -\epsilon_T$

3. **Finite Epochs:** $N_{\text{epochs}} \leq (\Phi_0 - \Phi_{\min})/\epsilon_T$

4. **Total Time:** $T_{\text{total}} \leq N_{\text{epochs}} \cdot T_{\max}$

**Constructive Content:**
- Algorithm to detect epoch boundaries
- Computable bounds on epoch duration
- Termination guarantee for resolution procedure

## Key GMT Inequalities Used

1. **Łojasiewicz Time Bound:**
   $$t_{\text{conv}} \leq C|\Phi_0 - \Phi_*|^{-(1-2\theta)}$$

2. **Energy Drop:**
   $$\Phi(T_{t_i}) - \Phi(T_{t_{i+1}}) \geq \epsilon_T$$

3. **Epoch Count:**
   $$N \leq \Phi_0/\epsilon_T$$

4. **Dissipation Lower Bound:**
   $$d(T, \text{Crit}) \geq \delta \implies \mathfrak{D}(T) \geq c_0$$

## Literature References

- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Colding, T., Minicozzi, W. (2012). Generic mean curvature flow I. *Ann. of Math.*, 175.
