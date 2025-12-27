# THM-FiniteRuns: Finite Complete Runs — GMT Translation

## Original Statement (Hypostructure)

Every resolution process terminates in finite time, producing a complete run from initial state to equilibrium or classified terminal state.

## GMT Setting

**Run:** Sequence of flow epochs and surgeries $T_0 \to T_1 \to \cdots \to T_N$

**Complete:** Run terminates at equilibrium or fully classified state

**Finite:** Run has finitely many steps

## GMT Statement

**Theorem (Finite Complete Runs).** Under soft permits, every resolution process from initial $T_0 \in \mathbf{I}_k(M)$ satisfies:

1. **Termination:** The process terminates after finitely many steps

2. **Completeness:** Terminal state is either equilibrium or classified

3. **Bound:** Number of steps $N \leq (\Phi(T_0) - \Phi_{\min})/\epsilon_T$

## Proof Sketch

### Step 1: Resolution Process Definition

**Process:**
$$T_0 \xrightarrow{\varphi_{[0,t_1]}} T_1^- \xrightarrow{\text{surg}} T_1 \xrightarrow{\varphi_{[t_1,t_2]}} T_2^- \xrightarrow{\text{surg}} \cdots \to T_N$$

**Components:**
- $\varphi_{[t_i, t_{i+1}]}$: Gradient flow
- $\text{surg}$: Surgery operation
- $T_i$: State after $i$-th surgery

### Step 2: Energy Monotonicity

**Flow Phase:** During $\varphi_{[t_i, t_{i+1}]}$:
$$\Phi(T_{i+1}^-) \leq \Phi(T_i)$$

**Surgery Phase:** Each surgery:
$$\Phi(T_{i+1}) \leq \Phi(T_{i+1}^-) - \epsilon_{\text{surg}}$$

where $\epsilon_{\text{surg}} > 0$ is minimum surgery energy drop.

**Combined:**
$$\Phi(T_N) \leq \Phi(T_0) - N \cdot \epsilon_T$$

### Step 3: Energy Lower Bound

**Minimum Energy:** There exists $\Phi_{\min}$ such that:
$$\Phi(T) \geq \Phi_{\min}$$

for all $T$ in the state space.

**Examples:**
- Mass functional: $\Phi = \mathbf{M} \geq 0$
- Willmore energy: $\Phi = \int |H|^2 \geq 0$
- Dirichlet energy: $\Phi = \int |\nabla u|^2 \geq 0$

### Step 4: Finite Step Count

**Theorem:** $N \leq (\Phi(T_0) - \Phi_{\min})/\epsilon_T$

*Proof:*
$$\Phi_{\min} \leq \Phi(T_N) \leq \Phi(T_0) - N \cdot \epsilon_T$$

Rearranging:
$$N \leq \frac{\Phi(T_0) - \Phi_{\min}}{\epsilon_T}$$

### Step 5: Terminal State Classification

**Terminal States:** The process terminates when:

1. **Equilibrium:** $\nabla \Phi(T_N) = 0$ — critical point reached

2. **Regular:** $\text{sing}(T_N) = \emptyset$ — all singularities removed

3. **Classified Singular:** $T_N$ has singular set entirely in $\mathcal{L}$ (library profiles)

**Completeness:** One of these three conditions must hold at termination.

### Step 6: No Infinite Loops

**Claim:** The process cannot loop infinitely.

*Proof:* Each step decreases energy:
- Flow strictly decreases $\Phi$ (except at equilibria)
- Surgery strictly decreases $\Phi$ by $\geq \epsilon_{\text{surg}}$

With finite energy and positive steps, only finitely many steps possible.

### Step 7: Termination Detection

**Detection Criteria:**

1. **Equilibrium Test:** $|\nabla \Phi(T_N)| < \varepsilon_{\text{equil}}$

2. **Regularity Test:** Allard's $\varepsilon$-regularity satisfied everywhere

3. **Classification Test:** All tangent cones belong to $\mathcal{L}$

**Algorithm:**
```
while not terminated(T):
    T = flow(T, Δt)
    if surgery_needed(T):
        T = surgery(T)
return T
```

### Step 8: Complete Run Structure

**Run Record:**
$$\mathcal{R} = \{(T_0, t_0), (T_1, t_1), \ldots, (T_N, t_N), \text{terminal state}\}$$

**Verification:** Each transition is valid:
- Flow transitions: verify energy dissipation
- Surgery transitions: verify admissibility

**Certificate:** The run record serves as proof of termination.

### Step 9: Time Bound

**Total Time:**
$$T_{\text{total}} \leq \sum_{i=0}^{N-1} T_{\max} = N \cdot T_{\max} \leq \frac{(\Phi_0 - \Phi_{\min}) \cdot T_{\max}}{\epsilon_T}$$

where $T_{\max}$ is maximum epoch duration (from THM-EpochTermination).

### Step 10: Compilation Theorem

**Theorem (Finite Complete Runs):**

1. **Termination:** Every run terminates in $N \leq (\Phi_0 - \Phi_{\min})/\epsilon_T$ steps

2. **Completeness:** Terminal state is equilibrium, regular, or classified

3. **Time Bound:** Total time $\leq N \cdot T_{\max}$

4. **Certificate:** Run record provides termination proof

**Applications:**
- Ricci flow with surgery (Perelman): finitely many surgeries
- MCF with surgery (Huisken-Sinestrari): finitely many surgeries
- Harmonic map heat flow: converges in finite time

## Key GMT Inequalities Used

1. **Energy Drop per Surgery:**
   $$\Phi(T_{i+1}) \leq \Phi(T_i) - \epsilon_T$$

2. **Step Count:**
   $$N \leq \Phi_0/\epsilon_T$$

3. **Energy Lower Bound:**
   $$\Phi(T) \geq \Phi_{\min}$$

4. **Time per Epoch:**
   $$t_{i+1} - t_i \leq T_{\max}$$

## Literature References

- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Perelman, G. (2003). Finite extinction time. arXiv:math/0307245.
- Hamilton, R. S. (1997). Four-manifolds with positive isotropic curvature. *Comm. Anal. Geom.*, 5.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- Kleiner, B., Lott, J. (2008). Notes on Perelman's papers. *Geom. Topol.*, 12.
- Simon, L. (1983). Asymptotics for nonlinear evolution equations. *Ann. of Math.*, 118.
