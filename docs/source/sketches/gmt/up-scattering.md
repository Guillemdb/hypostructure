# UP-Scattering: Scattering Promotion — GMT Translation

## Original Statement (Hypostructure)

The scattering promotion shows that solutions disperse to equilibrium at large times, with profiles separating and becoming asymptotically independent.

## GMT Setting

**Scattering:** Solution approaches free evolution as $t \to \infty$

**Asymptotic Freedom:** Profiles become non-interacting

**Dispersion:** Energy spreads out, preventing concentration

## GMT Statement

**Theorem (Scattering Promotion).** Under soft permits with $\Phi(T_0) < E_c$ (subcritical energy):

1. **Global Existence:** Solution $T_t$ exists for all $t \in [0, \infty)$

2. **Scattering:** There exists $T_+^\infty \in \mathbf{I}_k(M)$ such that:
$$d(T_t, \varphi_t^{\text{free}}(T_+^\infty)) \to 0 \text{ as } t \to \infty$$

3. **Profile Separation:** Profiles in the decomposition become asymptotically independent

## Proof Sketch

### Step 1: Subcritical Energy

**Critical Energy:** $E_c$ is the threshold below which scattering occurs.

**Subcritical Regime:** $\Phi(T_0) < E_c$ ensures:
- No concentration (energy too low to form singularity)
- Global existence
- Dispersion dominates

**Reference:** Kenig, C., Merle, F. (2006). Global well-posedness for energy-critical NLS. *Acta Math.*, 201, 147-212.

### Step 2: Dispersive Estimates

**Decay Estimate:** For dispersive equations:
$$\|T_t\|_{L^\infty} \leq C t^{-\alpha} \|T_0\|_{L^1}$$

for some $\alpha > 0$ depending on dimension.

**Strichartz Estimates:** Mixed space-time norms:
$$\|T\|_{L^q_t L^r_x} \leq C \|T_0\|_{L^2}$$

for admissible pairs $(q, r)$.

**Reference:** Strichartz, R. S. (1977). Restrictions of Fourier transforms to quadratic surfaces and decay of solutions of wave equations. *Duke Math. J.*, 44, 705-714.

### Step 3: Non-Concentration

**No Concentration Lemma:** For $\Phi(T_0) < E_c$:
$$\limsup_{t \to \infty} \sup_{x \in M} \int_{B_R(x)} |T_t|^{p^*} \, dx \to 0 \text{ as } R \to 0$$

*Proof:* If concentration occurred, profile decomposition would produce a profile with energy $\geq E_c$, contradiction.

**Reference:** Gérard, P. (1998). Description du défaut de compacité de l'injection de Sobolev. *ESAIM Control Optim. Calc. Var.*, 3, 213-233.

### Step 4: Scattering State Construction

**Duhamel Formula:** Write solution as:
$$T_t = \varphi_t^{\text{free}}(T_0) + \int_0^t \varphi_{t-s}^{\text{free}}(N(T_s)) \, ds$$

where $N$ is the nonlinearity.

**Scattering State:**
$$T_+^\infty := T_0 + \int_0^\infty \varphi_{-s}^{\text{free}}(N(T_s)) \, ds$$

**Convergence:** By decay estimates, the integral converges.

### Step 5: Asymptotic Independence

**Profile Decomposition:** $T_t = \sum_l V^l_t + w_t$

**Separation:** As $t \to \infty$:
$$\frac{|x_t^l - x_t^m|}{\sqrt{t}} \to \infty \quad \text{for } l \neq m$$

(profiles move apart at rate $\sqrt{t}$).

**Independence:** Interaction between profiles vanishes:
$$\int V^l_t \cdot V^m_t \to 0 \text{ as } t \to \infty$$

### Step 6: Energy Partition

**Asymptotic Energy:**
$$\lim_{t \to \infty} \Phi(T_t) = \sum_l \Phi(V^l) + \Phi_{\text{disp}}$$

where $\Phi_{\text{disp}}$ is dispersed energy (goes to infinity).

**Conservation Check:**
$$\Phi(T_0) = \sum_l \Phi(V^l) + \Phi_{\text{disp}}$$

(total energy conserved).

### Step 7: Scattering Norm Bounds

**Scattering Norm:** Define:
$$\|T\|_S := \|T\|_{L^{q}_t L^r_x([0, \infty) \times M)}$$

for appropriate $(q, r)$.

**Bound:** For $\Phi(T_0) < E_c$:
$$\|T\|_S \leq C(E_c - \Phi(T_0))^{-\beta}$$

**Small Data:** For $\Phi(T_0) \ll 1$:
$$\|T\|_S \leq C \Phi(T_0)^\gamma$$

**Reference:** Tao, T. (2006). *Nonlinear Dispersive Equations*. CBMS Regional Conference Series.

### Step 8: Scattering in GMT

**GMT Interpretation:** Scattering means:

1. **No New Singularities:** $\text{sing}(T_t) \subset \text{sing}(T_0)$ for $t > 0$
2. **Singularity Dispersion:** Singular set spreads out, density decreases
3. **Regular Limit:** $T_t \to T_\infty$ where $T_\infty$ is regular (or has simpler singularities)

**Example (MCF):** For convex hypersurfaces, MCF scatters to a point (round shrinking):
$$M_t \to \{p\} \text{ as } t \to T_{\text{ext}}$$

**Reference:** Huisken, G. (1984). Flow by mean curvature of convex surfaces. *J. Diff. Geom.*, 20, 237-266.

### Step 9: Asymptotic Completeness

**Wave Operators:** Define:
$$W_\pm: T_0 \mapsto T_\pm^\infty := \lim_{t \to \pm\infty} \varphi_{-t}^{\text{free}}(T_t)$$

**Completeness:** For subcritical energies:
$$\text{Range}(W_+) = \text{Range}(W_-) = \{T : \Phi(T) < E_c\}$$

**Inverse Scattering:** Given $T_+^\infty$, reconstruct $T_0$ via $W_+^{-1}$.

### Step 10: Compilation Theorem

**Theorem (Scattering Promotion):**

1. **Subcritical:** $\Phi(T_0) < E_c$ implies global existence + scattering

2. **Scattering State:** $T_t \to \varphi_t^{\text{free}}(T_+^\infty)$ as $t \to \infty$

3. **Profile Separation:** Profiles become asymptotically independent

4. **Completeness:** Wave operators are bijective on subcritical data

**Applications:**
- Global solutions for small initial data
- Long-time asymptotics
- Inverse scattering for profile reconstruction

## Key GMT Inequalities Used

1. **Dispersive Decay:**
   $$\|T_t\|_{L^\infty} \leq C t^{-\alpha} \|T_0\|_{L^1}$$

2. **Strichartz:**
   $$\|T\|_{L^q_t L^r_x} \leq C \|T_0\|_{L^2}$$

3. **Non-Concentration:**
   $$\sup_x \int_{B_R(x)} |T_t|^{p^*} \to 0$$

4. **Profile Separation:**
   $$|x_t^l - x_t^m| \to \infty$$

## Literature References

- Kenig, C., Merle, F. (2006). Global well-posedness for energy-critical NLS. *Acta Math.*, 201.
- Strichartz, R. S. (1977). Restrictions of Fourier transforms. *Duke Math. J.*, 44.
- Tao, T. (2006). *Nonlinear Dispersive Equations*. CBMS.
- Gérard, P. (1998). Défaut de compacité. *ESAIM COCV*, 3.
- Huisken, G. (1984). Flow by mean curvature. *J. Diff. Geom.*, 20.
- Nakanishi, K., Schlag, W. (2011). *Invariant Manifolds and Dispersive Hamiltonian Evolution Equations*. EMS.
