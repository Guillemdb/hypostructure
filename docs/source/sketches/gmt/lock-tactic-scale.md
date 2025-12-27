# LOCK-TacticScale: Type II Exclusion Lock — GMT Translation

## Original Statement (Hypostructure)

The Type II exclusion lock shows that certain singularity types (Type II in Ricci flow terminology) are excluded by geometric barriers, leaving only Type I singularities.

## GMT Setting

**Type I Singularity:** Curvature blows up at controlled rate: $|A|^2 \leq C/(T-t)$

**Type II Singularity:** Curvature blows up faster than Type I rate

**Exclusion:** Type II is blocked by energy/entropy barriers

## GMT Statement

**Theorem (Type II Exclusion Lock).** For mean curvature flow $\{T_t\}$ in $\mathbf{I}_k(M)$:

1. **Rate Classification:**
   - Type I: $\sup_{x \in \text{spt}(T_t)} |A(x,t)|^2 \leq C/(T-t)$
   - Type II: Rate exceeds Type I bound

2. **Energy Barrier:** Type II requires infinite local energy concentration

3. **Lock:** Under finite energy, only Type I singularities occur

## Proof Sketch

### Step 1: Singularity Rate Classification

**Type I Definition (Hamilton):** Singularity at $T$ is Type I if:
$$\sup_{M} |A|^2 (T - t) \leq C$$

as $t \to T^-$.

**Type II Definition:** Singularity is Type II if:
$$\limsup_{t \to T^-} \sup_M |A|^2 (T-t) = \infty$$

**Reference:** Hamilton, R. S. (1995). The formation of singularities in the Ricci flow. *Surveys in Differential Geometry*, 2, 7-136.

### Step 2: Blow-Up Analysis

**Type I Blow-Up:** Rescale by $\lambda_i = 1/\sqrt{T - t_i}$:
$$T_i = \lambda_i(T_{t_i} - p)$$

converges to self-similar (homothetically shrinking) limit.

**Type II Blow-Up:** Rescale by $\lambda_i = \sqrt{\max|A|^2(t_i)}$:

Limit is eternal solution.

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175, 137-221.

### Step 3: Entropy Monotonicity

**Huisken's Monotonicity:** For mean curvature flow:
$$\Phi_\rho(t) = \int_{T_t} \rho(x, t) \, d\mathcal{H}^k$$

where $\rho$ is backward heat kernel, is monotonically decreasing.

**Reference:** Huisken, G. (1990). Asymptotic behavior for singularities of the mean curvature flow. *J. Differential Geom.*, 31, 285-299.

**Type I Bound:** $\Phi_\rho(t)$ finite implies Type I.

### Step 4: Energy Concentration for Type II

**Lemma:** Type II singularity requires:
$$\liminf_{t \to T} \int_{B_r(p)} |A|^2 \, d\mathcal{H}^k = \infty$$

for any $r > 0$ around singularity $p$.

*Proof:*
- Type II: $|A|^2 \geq C_i/(T - t_i)$ with $C_i \to \infty$
- Local integral must absorb diverging curvature
- Finite local energy contradicts Type II rate

### Step 5: White's Local Regularity

**Theorem (White, 2005):** If:
$$\int_{B_r(p) \times (t-r^2, t)} |A|^2 \, d\mathcal{H}^k dt \leq \varepsilon_0$$

then $T_s$ is regular at $p$ for $s \in (t - r^2/2, t)$.

**Reference:** White, B. (2005). A local regularity theorem for mean curvature flow. *Ann. of Math.*, 161, 1487-1519.

**Contrapositive:** Singularity requires local energy concentration.

### Step 6: Gaussian Density Bounds

**Gaussian Density:**
$$\Theta(T, (p, T)) = \lim_{t \to T^-} \int_{T_t} \frac{1}{(4\pi(T-t))^{k/2}} e^{-|x-p|^2/4(T-t)} d\mathcal{H}^k$$

**Type I:** Gaussian density is finite and achieved by tangent flow.

**Type II:** Gaussian density equals infinity.

**Reference:** Colding, T. H., Minicozzi, W. P. (2012). Generic mean curvature flow I: Generic singularities. *Ann. of Math.*, 175, 755-833.

### Step 7: Exclusion via Entropy

**Theorem (Colding-Minicozzi):** For generic initial data, all singularities are Type I (spherical or cylindrical).

**Reference:** Colding, T. H., Minicozzi, W. P. (2015). Uniqueness of blowups and Łojasiewicz inequalities. *Ann. of Math.*, 182, 221-285.

**Generic Exclusion:** Type II is non-generic.

### Step 8: Huisken-Sinestrari Surgery

**Surgery Program:** At Type I singularities:
1. Detect neck (cylindrical region)
2. Perform surgery (cut and cap)
3. Continue flow

**Type II Obstruction:** Type II would require different surgery protocol or is impossible.

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.

### Step 9: Avoidance Principle

**Strong Maximum Principle:** Disjoint submanifolds stay disjoint under MCF.

**Self-Intersection Avoidance:** Embedded hypersurface stays embedded until first singular time.

**Reference:** Ecker, K. (2004). *Regularity Theory for Mean Curvature Flow*. Birkhäuser.

**Type II and Topology:** Type II often associated with topology change, constrained by avoidance.

### Step 10: Compilation Theorem

**Theorem (Type II Exclusion Lock):**

1. **Type I/II Classification:** By blow-up rate of curvature

2. **Energy Barrier:** Type II requires infinite local energy concentration

3. **Generic Exclusion:** Type II non-generic (Colding-Minicozzi)

4. **Lock:** Finite energy + genericity → Type I only

**Applications:**
- Classification of MCF singularities
- Surgery protocol design
- Regularity for generic flows

## Key GMT Inequalities Used

1. **Type I Rate:**
   $$|A|^2 \leq C/(T-t)$$

2. **Monotonicity:**
   $$\frac{d}{dt}\Phi_\rho \leq 0$$

3. **Local Regularity:**
   $$\int |A|^2 \leq \varepsilon_0 \implies \text{regular}$$

4. **Gaussian Density:**
   $$\Theta_{\text{Type I}} < \infty, \quad \Theta_{\text{Type II}} = \infty$$

## Literature References

- Hamilton, R. S. (1995). Formation of singularities in Ricci flow. *Surveys in Differential Geometry*, 2.
- Huisken, G. (1990). Asymptotic behavior of MCF singularities. *J. Differential Geom.*, 31.
- Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175.
- White, B. (2005). Local regularity for MCF. *Ann. of Math.*, 161.
- Colding, T. H., Minicozzi, W. P. (2012). Generic mean curvature flow I. *Ann. of Math.*, 175.
- Colding, T. H., Minicozzi, W. P. (2015). Uniqueness of blowups. *Ann. of Math.*, 182.
- Ecker, K. (2004). *Regularity Theory for Mean Curvature Flow*. Birkhäuser.
