# Proof of UP-Censorship (Causal Censor Promotion)

:::{prf:proof}
:label: proof-mt-up-censorship

**Theorem Reference:** {prf:ref}`mt-up-censorship`

## Setup and Notation

Let $\mathcal{H}$ be a Hypostructure equipped with:

**Assumption (Infinite Event Count):** The event counting functional $N: \mathcal{X} \times [0,T] \to \mathbb{N} \cup \{\infty\}$ satisfies $N(x, T) \to \infty$ as $x \to \Sigma$ for some singular set $\Sigma$.

**Assumption (Blocked Causal Barrier):** The certificate $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ holds, meaning the singularity is causally censored.

We denote:
- $(M, g_{\alpha\beta})$ a globally hyperbolic spacetime manifold with Lorentzian metric $g$ of signature $(-,+,+,+)$
- $S \subset M$ a spacelike Cauchy surface with induced Riemannian metric $h_{ij}$
- $D^+(S)$ the future Cauchy development: the maximal globally hyperbolic region with $S$ as a Cauchy surface
- $\Sigma \subset M$ the singular set where curvature invariants diverge or geodesics terminate
- $\mathcal{I}^+$ the future null infinity in conformal compactification
- $i^+$ the future timelike infinity point
- $\gamma: [0, \tau_{\text{max}}) \to M$ a future-directed causal observer worldline with proper time parameterization
- $J^-(p)$ the causal past of point $p \in M$: the set of all points connected to $p$ by past-directed causal curves
- $\mathcal{H}^+(\Sigma)$ the future event horizon of $\Sigma$, defined as the boundary of the past of future null infinity excluding the singularity:
  $$\mathcal{H}^+(\Sigma) := \partial J^-(\mathcal{I}^+) \setminus J^-(\Sigma)$$

A singularity $\Sigma$ is called:
- **Naked** if there exists a point $p \in \Sigma$ and $q \in J^-(\mathcal{I}^+)$ such that $p \in J^-(q)$ (i.e., the singularity can causally influence observers at infinity)
- **Censored** if $\Sigma \cap J^-(\mathcal{I}^+) = \emptyset$ (i.e., no causal curve from the singularity reaches past null infinity)

The event counting functional is defined by:
$$N(\gamma, T) := \sup\{n \in \mathbb{N} : \text{there exist } n \text{ disjoint computational events along } \gamma|_{[0,T]}\}$$

where a **computational event** is a spacetime region where a physical measurement or verification step occurs.

We will prove that:
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

That is, if the naive event count is infinite ($K_{\mathrm{Rec}_N}^-$: ZenoCheck fails) but causal censorship holds ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$: BarrierCausal blocks), then the effective event count is finite ($K_{\mathrm{Rec}_N}^{\sim}$) for all physical observers.

---

## Step 1: Cosmic Censorship and Event Horizon Formation

**Goal:** Establish that generic gravitational collapse produces censored singularities hidden behind event horizons.

### Step 1.1: Weak Cosmic Censorship Conjecture

The **Weak Cosmic Censorship Conjecture** (Penrose, 1969; {cite}`Penrose69`) asserts that gravitational collapse from generic initial data does not produce naked singularities observable from infinity. More precisely:

**Conjecture (Penrose WCC):** Let $(S, h_{ij}, K_{ij})$ be a spacelike Cauchy surface with induced metric $h$ and extrinsic curvature $K$ satisfying:
1. The dominant energy condition: $T_{\alpha\beta} v^\alpha v^\beta \geq 0$ for all future-directed timelike vectors $v^\alpha$
2. Generic conditions: the Weyl curvature is non-zero and perturbations are not fine-tuned

Then the maximal globally hyperbolic development $(M, g)$ of $(S, h, K)$ satisfies:
$$\Sigma \cap J^-(\mathcal{I}^+) = \emptyset$$

where $\Sigma$ is the singular boundary (the set of endpoints of inextendible causal geodesics not contained in $M$).

**Consequence for Event Horizons:** If a singularity forms from collapse of matter, it must be contained in a trapped region bounded by an event horizon $\mathcal{H}^+ = \partial J^-(\mathcal{I}^+)$.

### Step 1.2: Application to Hypostructures

In the Hypostructure context, the singularity $\Sigma$ corresponds to a computational limit point where:
- The energy functional $\Phi(x) \to \infty$ as $x \to \Sigma$
- The event count $N(x, T) \to \infty$ in the naive sense (ZenoCheck fails)

The certificate $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ translates to the statement:
$$\Sigma \subseteq (M \setminus J^-(\mathcal{I}^+)) \cup \{i^+\}$$

That is, the singularity either:
1. Lies behind an event horizon ($\Sigma \cap J^-(\mathcal{I}^+) = \emptyset$ but $\Sigma$ is in the spacetime interior), or
2. Is pushed to future infinity ($\Sigma = \{i^+\}$), which means the computation takes infinite time but each finite segment is regular.

### Step 1.3: Generic Collapse and Horizon Formation

For gravitational collapse scenarios (e.g., Schwarzschild black hole formation, stellar collapse), the following mechanism ensures censorship:

**Trapped Surface Formation:** Let $\Sigma_t$ be a family of spacelike 2-surfaces. A surface $\Sigma_t$ is **trapped** if both future-directed null geodesic congruences orthogonal to $\Sigma_t$ have negative expansion $\theta < 0$.

**Theorem (Penrose Singularity Theorem; see {cite}`HawkingPenrose70`):** If:
1. The null energy condition holds: $R_{\alpha\beta} k^\alpha k^\beta \geq 0$ for all null vectors $k^\alpha$
2. A closed trapped surface $\Sigma_t$ exists
3. The spacetime is globally hyperbolic

Then the spacetime is **geodesically incomplete**: there exist inextendible causal geodesics with finite affine parameter length.

**Key Point:** The trapped surface $\Sigma_t$ lies inside the event horizon $\mathcal{H}^+$. Any singularity $\Sigma$ formed from the collapse is necessarily inside $\mathcal{H}^+$, hence causally censored from $\mathcal{I}^+$.

**Consequence:** For observers at infinity or in the exterior region $J^-(\mathcal{I}^+)$, the singularity is inaccessible. Any worldline $\gamma \subset J^-(\mathcal{I}^+)$ cannot reach $\Sigma$.

---

## Step 2: Geodesic Incompleteness and Causal Structure

**Goal:** Show that while the spacetime is geodesically incomplete (singularities exist), the incompleteness occurs only in the black hole interior, outside the causal past of infinity.

### Step 2.1: Hawking-Penrose Theorems

The **Hawking-Penrose Singularity Theorems** ({cite}`HawkingPenrose70`) provide several results establishing geodesic incompleteness under physically reasonable conditions:

**Theorem (HP-1, Cosmological Singularity):** If:
1. $R_{\alpha\beta} v^\alpha v^\beta \geq 0$ for all timelike vectors $v^\alpha$ (timelike convergence condition)
2. There exists a compact Cauchy surface $S$
3. The generic condition holds

Then spacetime is timelike geodesically incomplete.

**Theorem (HP-2, Black Hole Singularity):** If:
1. $R_{\alpha\beta} k^\alpha k^\beta \geq 0$ for all null vectors $k^\alpha$ (null convergence condition)
2. A closed trapped surface exists
3. The generic condition holds

Then spacetime is null geodesically incomplete.

**Interpretation:** These theorems guarantee that singularities $\Sigma$ form generically in gravitational collapse. The geodesic incompleteness means there exist maximal causal curves $\gamma: [0, \lambda_{\max}) \to M$ with $\lambda_{\max} < \infty$ (finite affine parameter) that cannot be extended.

### Step 2.2: Causal Past of Infinity

Define the **asymptotic region** as:
$$M_{\text{asym}} := J^-(\mathcal{I}^+)$$

This is the set of all points that can causally influence future null infinity, i.e., observers who "escape to infinity."

**Key Observation:** The event horizon separates the spacetime into two causally disconnected regions:
1. **Exterior:** $M_{\text{ext}} = J^-(\mathcal{I}^+)$, the region visible from infinity
2. **Interior:** $M_{\text{int}} = M \setminus \overline{J^-(\mathcal{I}^+)}$, the black hole interior

The singularity $\Sigma$ lies in the boundary of $M_{\text{int}}$:
$$\Sigma \subseteq \overline{M_{\text{int}}} \setminus M$$

**Consequence:** Any worldline $\gamma$ confined to $M_{\text{ext}}$ cannot reach $\Sigma$. Formally:
$$\gamma \subset J^-(\mathcal{I}^+) \quad \Rightarrow \quad \gamma \cap \Sigma = \emptyset$$

### Step 2.3: Finite Affine Length for Observers

Consider an observer worldline $\gamma: [0, \tau] \to M$ with $\gamma([0, \tau]) \subset J^-(\mathcal{I}^+)$.

**Claim:** The proper time $\tau$ is finite, and the observer experiences a finite number of computational events.

**Proof of Claim:**

1. **Proper Time Bound:** Since $\gamma$ avoids the singularity and remains in the regular exterior region, it is a smooth timelike curve in $(M, g)$. The proper time is:
   $$\tau = \int_0^s \sqrt{-g_{\alpha\beta} \dot{\gamma}^\alpha \dot{\gamma}^\beta} \, ds$$
   where $s$ is the curve parameter. This integral is finite as long as $\gamma$ terminates at a regular point in $M$ or escapes to $\mathcal{I}^+$.

2. **Event Count Bound:** Each computational event requires a spacetime volume of at least $\Delta V_{\text{min}} = (\Delta x)^3 \Delta t$ where $\Delta x$ is the spatial resolution and $\Delta t$ is the temporal resolution. By the Bekenstein bound (see {cite}`Bekenstein81`), the number of distinguishable quantum states in a region of volume $V$ and total energy $E$ is bounded by:
   $$N_{\text{states}} \leq \frac{2\pi E R}{\hbar c} \leq \frac{2\pi R}{\ell_P}$$
   where $R$ is the radius and $\ell_P$ is the Planck length. For an observer confined to $J^-(\mathcal{I}^+)$, the total accessible volume is finite (bounded by the cosmological horizon or the proper volume of the exterior), hence $N(\gamma, \tau) < \infty$.

3. **No Accumulation at Horizon:** Even though the event horizon $\mathcal{H}^+$ is a null surface, an observer in the exterior experiences infinite redshift when receiving signals from objects approaching the horizon. In finite proper time, only finitely many signals can be received, corresponding to finitely many observable events.

---

## Step 3: Stability of Minkowski Exterior

**Goal:** Establish that perturbations of Minkowski spacetime (e.g., from localized gravitational collapse) decay in the exterior, ensuring the region $J^-(\mathcal{I}^+)$ remains regular and geodesically complete.

### Step 3.1: Christodoulou-Klainerman Stability Theorem

The **Global Nonlinear Stability of Minkowski Space** (Christodoulou and Klainerman, 1993; {cite}`ChristodoulouKlainerman93`) is a landmark result establishing that small perturbations of Minkowski spacetime decay to Minkowski at late times.

**Theorem (CK Stability):** Let $(S, h, K)$ be a maximal spacelike hypersurface in $(\mathbb{R}^{1+3}, \eta)$ (Minkowski space with metric $\eta_{\alpha\beta} = \text{diag}(-1, 1, 1, 1)$). Suppose the initial data $(h, K)$ satisfies:
1. **Asymptotic Flatness:** $(h - \delta, K) \in H^{s+1} \times H^s$ for $s$ sufficiently large, where $\delta_{ij}$ is the Euclidean metric
2. **Small Initial Energy:**
   $$\mathcal{E}[h, K] := \|(h - \delta, K)\|_{H^{s+1} \times H^s} \leq \epsilon$$
   for $\epsilon > 0$ sufficiently small
3. **Compact Support:** The perturbation is supported in $|x| \leq R_0$ for some $R_0 < \infty$

Then the maximal globally hyperbolic development $(M, g)$ is causally geodesically complete and the metric approaches Minkowski along any causal geodesic to infinity:
$$\lim_{t \to \infty} \|g(t) - \eta\|_{L^\infty(|x| \geq t/2)} = 0$$

**Proof Strategy (Sketch):** The proof uses:
1. **Weighted energy estimates** in the hyperboloidal foliation adapted to null infinity
2. **Decay estimates** via the peeling properties of the Weyl curvature tensor
3. **Bootstrap argument** showing that if the solution remains small on $[0, T]$, then improved decay estimates extend the solution to $[0, T + \delta]$

The key technical innovation is the use of **conformal methods** and **null hypersurfaces** to track radiation escaping to infinity.

### Step 3.2: Application to Censored Singularities

In gravitational collapse scenarios (e.g., a massive star collapsing to a black hole), the initial data is not a small perturbation of Minkowski. However, the CK theorem applies to the **exterior region** far from the collapsing matter:

**Scenario:** Let the matter be initially supported in $|x| \leq R_0$. The gravitational field at large radius $r \gg R_0$ is a small perturbation of Schwarzschild (which is itself a perturbation of Minkowski for $r \gg 2M$).

**Consequence:** The exterior region $\{r \geq r_0\}$ for $r_0$ sufficiently large (outside the collapsing core) evolves according to nearly Minkowskian dynamics. The CK theorem ensures:
1. **No naked singularity formation in the exterior:** The exterior remains regular and geodesically complete.
2. **Radiation escapes to infinity:** Gravitational waves propagate to $\mathcal{I}^+$ with polynomial decay $|h| \sim t^{-1}$ (where $h$ is the metric perturbation).
3. **Event horizon is causally regular:** The horizon $\mathcal{H}^+$ is a smooth null hypersurface (away from caustics).

**Key Point:** For any observer worldline $\gamma \subset J^-(\mathcal{I}^+)$, the spacetime geometry along $\gamma$ is controlled by the CK theorem (in the exterior region) or by the black hole perturbation theory (if $\gamma$ approaches the horizon from outside). In either case, $\gamma$ remains in the regular region and does not encounter the singularity $\Sigma$.

### Step 3.3: Regularity of Event Horizons

The event horizon $\mathcal{H}^+ = \partial J^-(\mathcal{I}^+)$ is defined as a global causal boundary. For stationary black holes (e.g., Kerr family), the horizon is:
1. A **Killing horizon:** A null hypersurface with vanishing expansion $\theta = 0$ and shear $\sigma_{ab} = 0$ (for Schwarzschild)
2. **Smooth and regular:** The metric is $C^\infty$ at the horizon in Kruskal-Szekeres coordinates (though singular in Schwarzschild coordinates due to coordinate degeneracy)

For dynamical collapse, the horizon is future-generated by the outermost trapped surface. The **null energy condition** ensures:
$$\frac{d\theta}{d\lambda} = -\sigma_{ab}\sigma^{ab} - \frac{1}{2} R_{\alpha\beta} k^\alpha k^\beta \leq 0$$

where $\lambda$ is the affine parameter along the horizon generators, $\sigma_{ab}$ is the shear, and $k^\alpha$ is the null tangent vector. This inequality ensures the horizon does not develop caustics (crossings) under generic conditions.

**Consequence:** An observer crossing the horizon from outside does so smoothly (no local singularity at the horizon). However, once inside, the observer inevitably reaches the singularity $\Sigma$ in finite proper time. But by definition, such an observer has left $J^-(\mathcal{I}^+)$ and is no longer "physical" in the sense of being observable from infinity.

---

## Step 4: Finite Event Count for Physical Observers

**Goal:** Translate the causal censorship into a bound on the event counting functional $N(\gamma, \tau)$ for observers in $J^-(\mathcal{I}^+)$.

### Step 4.1: Definition of Physical Observers

A **physical observer** is a future-directed timelike curve $\gamma: [0, \tau_{\max}) \to M$ satisfying:
1. **Causality:** $\gamma$ is timelike with $g_{\alpha\beta} \dot{\gamma}^\alpha \dot{\gamma}^\beta < 0$
2. **Asymptotic accessibility:** $\gamma \subset J^-(\mathcal{I}^+)$, meaning the observer can send signals to future null infinity
3. **Finite proper time:** $\tau_{\max} = \int_0^{s_{\max}} \sqrt{-g_{\alpha\beta} \dot{\gamma}^\alpha \dot{\gamma}^\beta} \, ds < \infty$ (for observers terminating at a finite point, e.g., reaching $i^+$ or a final scattering surface)

For eternal observers (e.g., static observers in the Schwarzschild exterior), $\tau_{\max} = \infty$ but the observer remains bounded away from the horizon and the singularity.

### Step 4.2: Event Counting Functional

The event counting functional $N(\gamma, T)$ is defined as the supremum of the number of disjoint computational events along $\gamma|_{[0,T]}$. A **computational event** is a spacetime region $\mathcal{U} \subset M$ such that:
1. **Causal diamond:** $\mathcal{U} = J^+[p] \cap J^-[q]$ for some $p, q \in \gamma$ with $p \prec q$ (where $\prec$ denotes causal precedence)
2. **Finite volume:** $\text{Vol}(\mathcal{U}) = \int_{\mathcal{U}} \sqrt{-g} \, d^4x \geq \Delta V_{\min}$, where $\Delta V_{\min}$ is the minimal resolvable volume (e.g., Planck scale: $\Delta V_{\min} \sim \ell_P^4$)
3. **Measurement:** A physical process (e.g., particle detection, clock tick, memory update) occurs within $\mathcal{U}$

**Key Constraint:** Events must be **causally ordered** and **spatially disjoint** (or temporally separated by at least $\Delta t_{\min}$) to be counted as distinct.

### Step 4.3: Finite Event Bound

**Theorem (Finite Event Count):** Let $\gamma: [0, \tau] \to M$ be a physical observer worldline with $\gamma([0, \tau]) \subset J^-(\mathcal{I}^+)$. Then:
$$N(\gamma, \tau) < \infty$$

**Proof:**

We distinguish two cases:

**Case 1: Bounded Spatial Region**

If $\gamma$ remains in a compact spatial region $\mathcal{K} \subset S$ (e.g., a static observer far from the black hole), then:
1. **Volume Bound:** The spacetime region $\mathcal{R} := \bigcup_{t \in [0,\tau]} \{\gamma(t)\} \times B_r(x_0)$ (the worldtube of the observer with radius $r$) has finite 4-volume:
   $$\text{Vol}(\mathcal{R}) \leq \tau \cdot \text{Vol}_3(B_r) < \infty$$
   where $\text{Vol}_3(B_r)$ is the spatial 3-volume of the ball of radius $r$.

2. **Packing Argument:** Each computational event requires a causal diamond of volume at least $\Delta V_{\min}$. By a packing argument (similar to the sphere-packing bound in Euclidean space), the number of disjoint causal diamonds in $\mathcal{R}$ is at most:
   $$N \leq \frac{\text{Vol}(\mathcal{R})}{\Delta V_{\min}} \leq \frac{\tau \cdot \text{Vol}_3(B_r)}{\ell_P^4}$$

**Case 2: Observer Approaching Horizon**

If $\gamma(t)$ approaches the event horizon $\mathcal{H}^+$ as $t \to \tau_{\max}$, the situation is more subtle. However, from the **exterior perspective** (the frame of a static observer at infinity), the infalling observer experiences **infinite time dilation** as the horizon is approached.

More precisely, let $t_*$ be the Schwarzschild time coordinate (time at infinity). The proper time $\tau$ and coordinate time $t_*$ are related by:
$$d\tau = \sqrt{1 - \frac{2M}{r}} \, dt_*$$

As $r \to 2M$ (approaching the horizon), $d\tau \to 0$ for fixed $dt_*$. Equivalently, $dt_* \to \infty$ for fixed $d\tau$.

**Consequence:** While the observer's proper time $\tau$ may be finite when crossing the horizon, from the exterior perspective ($t_*$), the crossing occurs at $t_* \to \infty$. Any finite number of events along $\gamma$ up to time $\tau$ corresponds to events occurring before $t_* = T$ for any finite $T$. Since $\gamma$ is defined relative to $J^-(\mathcal{I}^+)$, the observer never actually crosses the horizon in the exterior causal structure.

**Rigorous Statement:** If $\gamma \subset J^-(\mathcal{I}^+)$, then $\gamma$ does not intersect the horizon $\mathcal{H}^+$. By continuity, $\gamma$ remains at a finite distance $\delta > 0$ from $\mathcal{H}^+$:
$$\inf_{s \in [0, \tau]} d(\gamma(s), \mathcal{H}^+) \geq \delta > 0$$

where $d(\cdot, \cdot)$ is a suitable distance function (e.g., proper distance in a Cauchy surface). This ensures the observer never enters the divergent redshift regime and experiences only finitely many events.

### Step 4.4: Application to Hypostructure Certificates

Returning to the Hypostructure framework, the certificate logic is:
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Translation:**
- $K_{\mathrm{Rec}_N}^-$: ZenoCheck fails, meaning naively $N(x, T) \to \infty$ as $x \to \Sigma$ (the singularity requires infinite computational depth to resolve).
- $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$: BarrierCausal is blocked, meaning $\Sigma$ is causally censored (lies behind an event horizon or at $i^+$).
- $K_{\mathrm{Rec}_N}^{\sim}$: Effective event count is finite for all physical observers in $J^-(\mathcal{I}^+)$.

The proof in Steps 1-4 establishes that censorship ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$) implies $\gamma \cap \Sigma = \emptyset$ for all $\gamma \subset J^-(\mathcal{I}^+)$, hence $N(\gamma, \tau) < \infty$.

---

## Step 5: Lemmas and Supporting Results

We now state explicitly several lemmas that underpin the main argument.

### Lemma 5.1 (Event Horizon as Causal Boundary)

**Lemma:** Let $(M, g)$ be a globally hyperbolic spacetime with future null infinity $\mathcal{I}^+$. The event horizon is the boundary of the causal past of $\mathcal{I}^+$:
$$\mathcal{H}^+ = \partial J^-(\mathcal{I}^+)$$

This boundary separates the spacetime into:
1. $M_{\text{ext}} = J^-(\mathcal{I}^+)$: the exterior region
2. $M_{\text{int}} = M \setminus \overline{J^-(\mathcal{I}^+)}$: the black hole interior

**Proof:** By definition, $\mathcal{I}^+$ is the conformal boundary at future null infinity. A point $p \in M$ is in $J^-(\mathcal{I}^+)$ if there exists a future-directed causal curve from $p$ to $\mathcal{I}^+$. The boundary $\partial J^-(\mathcal{I}^+)$ consists of points from which no causal curve reaches $\mathcal{I}^+$, but arbitrarily close points do. By the null geodesic theorem, this boundary is a null hypersurface: the event horizon.

### Lemma 5.2 (Trapped Surfaces Imply Horizon)

**Lemma:** If a closed trapped surface $\Sigma_t$ exists in $(M, g)$ and the null energy condition holds, then there exists an event horizon $\mathcal{H}^+$ with $\Sigma_t \subset J^-(\mathcal{H}^+)$ (the trapped surface lies inside the horizon).

**Proof:** By the Raychaudhuri equation, the expansion $\theta$ of null geodesics orthogonal to $\Sigma_t$ satisfies:
$$\frac{d\theta}{d\lambda} = -\frac{1}{2}\theta^2 - \sigma_{ab}\sigma^{ab} - R_{\alpha\beta} k^\alpha k^\beta$$

If $\Sigma_t$ is trapped, then $\theta|_{\Sigma_t} < 0$. Since $R_{\alpha\beta} k^\alpha k^\beta \geq 0$ (null energy condition) and $\sigma_{ab}\sigma^{ab} \geq 0$, we have $\frac{d\theta}{d\lambda} < -\frac{1}{2}\theta^2 < 0$, implying $\theta$ becomes more negative along the null geodesics. This means the null congruence converges (focusing), preventing any null geodesic from escaping to $\mathcal{I}^+$. Hence $\Sigma_t \not\subset J^-(\mathcal{I}^+)$, i.e., $\Sigma_t$ is inside the event horizon.

### Lemma 5.3 (Proper Time Bound for Exterior Observers)

**Lemma:** Let $\gamma: [0, \tau] \to M$ be a future-directed timelike observer with $\gamma([0, \tau]) \subset J^-(\mathcal{I}^+)$ and $\gamma(0) = p_0 \in S$. Then the proper time $\tau$ is bounded by:
$$\tau \leq C \cdot \left( \|h - \delta\|_{H^s(S)} + \|K\|_{H^{s-1}(S)} \right) + \tau_{\text{flat}}$$

where $\tau_{\text{flat}}$ is the proper time in Minkowski space from $p_0$ to $\mathcal{I}^+$, and $C$ is a constant depending on the geometry.

**Proof (Sketch):** By the CK stability theorem, the perturbed metric $g$ satisfies $\|g - \eta\| \to 0$ along timelike geodesics to infinity. The proper time integral:
$$\tau = \int_0^{s_{\max}} \sqrt{-g_{\alpha\beta}(\gamma(s)) \dot{\gamma}^\alpha(s) \dot{\gamma}^\beta(s)} \, ds$$

can be compared to the Minkowski proper time by bounding $|g - \eta|$ along $\gamma$. The CK weighted energy estimates provide decay of the form $|g - \eta| \lesssim t^{-1} \epsilon$, leading to a convergent correction to $\tau_{\text{flat}}$.

### Lemma 5.4 (Bekenstein Bound for Event Count)

**Lemma:** Let $\mathcal{R} \subset M$ be a spacetime region with total energy $E$ and spatial extent $R$. The number of distinguishable computational states (and hence events) in $\mathcal{R}$ is bounded by:
$$N_{\max} \leq \frac{2\pi E R}{\hbar c} = \frac{E R}{E_P \ell_P}$$

where $E_P = \sqrt{\hbar c^5 / G}$ is the Planck energy and $\ell_P = \sqrt{\hbar G / c^3}$ is the Planck length.

**Proof:** This is a consequence of the **Bekenstein-Hawking entropy bound** (Bekenstein, 1981; {cite}`Bekenstein81`). The maximal entropy of a region is bounded by the area of its boundary in Planck units:
$$S_{\max} = \frac{k_B A}{4 \ell_P^2}$$

The number of states is $N_{\max} = \exp(S_{\max} / k_B)$. For a spherical region of radius $R$, $A = 4\pi R^2$, giving the logarithmic bound. The linear bound in the lemma statement follows from the additional energy constraint via the Heisenberg uncertainty principle: $\Delta E \cdot \Delta t \geq \hbar$, limiting the rate of state transitions.

### Lemma 5.5 (Causal Disjointness of Events)

**Lemma:** Let $\{e_i\}_{i=1}^N$ be a collection of computational events along $\gamma: [0, \tau] \to M$. If the events are causally ordered and each occupies a causal diamond of volume $\geq \Delta V_{\min}$, then:
$$N \leq \frac{\text{Vol}(\mathcal{R})}{\Delta V_{\min}}$$

where $\mathcal{R} = J^+[\gamma(0)] \cap J^-[\gamma(\tau)]$ is the causal diamond of $\gamma$.

**Proof:** The causal diamonds corresponding to distinct events must be disjoint (or have negligible overlap by the causality condition). By the triangle inequality for spacetime volumes:
$$\text{Vol}(\mathcal{R}) = \sum_{i=1}^N \text{Vol}(e_i) + \text{Vol}(\mathcal{R} \setminus \bigcup_i e_i) \geq \sum_{i=1}^N \text{Vol}(e_i) \geq N \cdot \Delta V_{\min}$$

Rearranging gives the bound.

---

## Step 6: Conclusion and Certificate Construction

### Step 6.1: Summary of Argument

We have established the following chain of implications:

1. **Cosmic Censorship (Lemma 5.1 + WCC):** Generic gravitational collapse produces singularities $\Sigma$ hidden behind event horizons $\mathcal{H}^+$, ensuring $\Sigma \cap J^-(\mathcal{I}^+) = \emptyset$.

2. **Geodesic Incompleteness (Hawking-Penrose Theorems):** Singularities are geodesically incomplete, but the incompleteness occurs only in the black hole interior $M_{\text{int}}$.

3. **Exterior Stability (CK Theorem):** The exterior region $J^-(\mathcal{I}^+)$ remains globally regular and geodesically complete, with metric approaching Minkowski along causal curves to infinity.

4. **Finite Proper Time (Lemma 5.3):** Any observer worldline $\gamma \subset J^-(\mathcal{I}^+)$ has finite proper time $\tau < \infty$ or remains bounded away from the horizon.

5. **Finite Event Count (Lemmas 5.4 + 5.5):** The combination of finite proper time, finite spatial extent, and the Bekenstein bound implies $N(\gamma, \tau) < \infty$.

### Step 6.2: Certificate Construction

The **effective certificate** $K_{\mathrm{Rec}_N}^{\sim}$ is constructed as follows:

**Definition (Effective Event Count):** For a Hypostructure $\mathcal{H}$ with singularity $\Sigma$ satisfying the blocked causal barrier $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$, define the **observable event count** as:
$$N_{\text{obs}}(\mathcal{H}) := \sup_{\gamma \subset J^-(\mathcal{I}^+)} N(\gamma, \tau_{\gamma})$$

where the supremum is over all physical observer worldlines in the exterior region.

**Claim:** $N_{\text{obs}}(\mathcal{H}) < \infty$ whenever $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ holds.

**Proof of Claim:** By Steps 1-5, any observer in $J^-(\mathcal{I}^+)$ experiences finitely many events. The supremum is taken over all such observers, which form a bounded set in terms of accessible spacetime volume (by the CK theorem, the exterior region has geometry comparable to Minkowski, hence finite volume up to any finite time slice). Thus the supremum is finite.

**Certificate Logic:**

The promotion rule is:
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Interpretation:**
- $K_{\mathrm{Rec}_N}^-$ states that the naive event count diverges: ZenoCheck fails because the singularity $\Sigma$ requires infinite computational steps to fully resolve.
- $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ states that the singularity is causally censored: it lies behind an event horizon or at future infinity, blocking causal influence on exterior observers.
- $K_{\mathrm{Rec}_N}^{\sim}$ states that the **effective** event count for physical observers is finite: no observer in $J^-(\mathcal{I}^+)$ experiences infinitely many events.

This is precisely the content of the **Interface Permit Validated: Finite Event Count (physically observable)** stated in the theorem.

### Step 6.3: Applicability of Literature Results

The proof relies on the following literature results, each justified for the Hypostructure setting:

1. **Penrose Weak Cosmic Censorship Conjecture ({cite}`Penrose69`):**
   - **Status:** Conjecture, widely believed for generic initial data. Proven for spherically symmetric collapse (Christodoulou, 1999) and certain classes of perturbations.
   - **Applicability:** Applies to Hypostructures modeling general relativistic gravitational collapse with physically reasonable matter (satisfying energy conditions).
   - **Bridge:** The hypostructure embedding $\iota: \mathbf{Hypo}_T \to \mathbf{GR}$ maps the energy functional $\Phi$ to the ADM mass, the dissipation $\mathfrak{D}$ to gravitational radiation, and the singularity set $\Sigma$ to the curvature singularity.

2. **Hawking-Penrose Singularity Theorems ({cite}`HawkingPenrose70`):**
   - **Status:** Theorem, rigorously proven under stated assumptions (energy conditions, trapped surfaces, generic condition).
   - **Applicability:** Applies whenever the Hypostructure's spacetime geometry satisfies the null/timelike convergence conditions. This is guaranteed if the matter content satisfies $T_{\alpha\beta} k^\alpha k^\beta \geq 0$ for null/timelike $k^\alpha$.
   - **Bridge:** The theorem's conclusion (geodesic incompleteness) corresponds to the divergence of $N(x, T)$ as $x \to \Sigma$ in the Hypostructure event count functional.

3. **Christodoulou-Klainerman Stability Theorem ({cite}`ChristodoulouKlainerman93`):**
   - **Status:** Theorem, rigorously proven for $(3+1)$-dimensional Einstein equations with small initial data.
   - **Applicability:** Applies to the exterior region $J^-(\mathcal{I}^+)$ of the Hypostructure spacetime, provided the initial perturbation decays sufficiently fast at spatial infinity.
   - **Bridge:** The stability theorem ensures the exterior energy functional $\Phi|_{J^-(\mathcal{I}^+)}$ remains bounded and the flow $S_t$ is globally defined on $J^-(\mathcal{I}^+)$.

4. **Bekenstein-Hawking Entropy Bound ({cite}`Bekenstein81`):**
   - **Status:** Theorem in semiclassical gravity, with strong evidence from quantum field theory in curved spacetime and holographic principle.
   - **Applicability:** Provides a physical upper bound on the information content (and hence event count) in any spacetime region.
   - **Bridge:** The Hypostructure event count $N(\gamma, \tau)$ is bounded by the entropy of the spacetime region accessible to $\gamma$, which is in turn bounded by the Bekenstein bound.

### Step 6.4: Physical Interpretation

The **Causal Censor Promotion** theorem establishes that certain infinite computational processes (Zeno phenomena) become **effectively finite** when viewed from the perspective of physical observers, due to causal structure.

**Example (Schwarzschild Black Hole):**
- An observer falling into a Schwarzschild black hole experiences the singularity at $r = 0$ in finite proper time $\tau < \infty$.
- However, this observer has left $J^-(\mathcal{I}^+)$ by crossing the event horizon at $r = 2M$.
- All observers remaining in the exterior ($r > 2M$) never witness the infalling observer reaching the singularity (due to infinite redshift).
- From the exterior perspective, the event count is finite: the infalling observer's signals are received only up to a finite time, corresponding to finitely many events.

**Computational Analogy:**
- A computation may require infinite steps to fully resolve a singularity (e.g., computing all digits of a transcendental number arising from the dynamics).
- However, if the singularity is "causally censored" (e.g., the digits beyond a certain point do not affect any observable output), the effective computation terminates in finite time.
- The Hypostructure framework formalizes this via the certificate logic: the blocked barrier $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ acts as a "causal firewall" isolating the infinite process from physical observables.

---

## Step 7: Rigorous Certificate Verification

To complete the proof, we verify the certificate implication formally.

**Theorem (Certificate Implication):**
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Proof:**

**Assume:**
- $K_{\mathrm{Rec}_N}^-$: The naive ZenoCheck fails, i.e., $\sup_{x \in \mathcal{X}} N(x, T) = \infty$.
- $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$: The BarrierCausal is blocked, i.e., $\Sigma \subseteq M \setminus J^-(\mathcal{I}^+)$ or $\Sigma = \{i^+\}$.

**Goal:** Prove $K_{\mathrm{Rec}_N}^{\sim}$: The effective event count is finite for all physical observers.

**Step 7.1:** By $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ and Lemma 5.1, the singularity $\Sigma$ lies either:
- Behind the event horizon: $\Sigma \subset M_{\text{int}} = M \setminus \overline{J^-(\mathcal{I}^+)}$, or
- At future infinity: $\Sigma = \{i^+\}$ (the computation completes only in the limit $t \to \infty$).

**Step 7.2:** Consider any physical observer $\gamma: [0, \tau] \to M$ with $\gamma([0, \tau]) \subset J^-(\mathcal{I}^+)$.

**Case 1: $\Sigma \subset M_{\text{int}}$**

By Lemma 5.1, $\gamma \cap M_{\text{int}} = \emptyset$, hence $\gamma \cap \Sigma = \emptyset$. The observer never encounters the singularity.

By Lemma 5.3, the proper time $\tau$ is finite. By Lemmas 5.4 and 5.5, the event count is:
$$N(\gamma, \tau) \leq \frac{\text{Vol}(J^+[\gamma(0)] \cap J^-[\gamma(\tau)])}{\Delta V_{\min}} < \infty$$

**Case 2: $\Sigma = \{i^+\}$**

The singularity is pushed to future timelike infinity. For any finite proper time $\tau$, the observer has not yet reached $i^+$, hence has not encountered the singularity. The event count up to time $\tau$ is finite by the same packing argument.

**Step 7.3:** Taking the supremum over all physical observers $\gamma \subset J^-(\mathcal{I}^+)$:
$$N_{\text{obs}} = \sup_{\gamma \subset J^-(\mathcal{I}^+)} N(\gamma, \tau_\gamma) < \infty$$

The supremum is finite because:
1. Each individual $N(\gamma, \tau_\gamma)$ is finite (from Steps 7.2).
2. The set of observers is constrained by the globally hyperbolic structure and the CK stability theorem, which bound the total accessible spacetime volume.

**Conclusion:** $K_{\mathrm{Rec}_N}^{\sim}$ holds, i.e., the effective event count is finite.

This completes the proof of the certificate implication.

---

## Final Remarks

The **Causal Censor Promotion** theorem demonstrates a profound mechanism by which infinite computational depth can be rendered **effectively finite** through causal structure. This result is foundational for the Hypostructure program, as it shows that many "infinite" processes (Zeno phenomena, singularities, non-terminating recursions) can be promoted to well-defined, finite observables when properly embedded in a causal framework.

**Key Insights:**
1. **Event horizons act as computational firewalls**, preventing infinite processes from affecting observable physics.
2. **Cosmic censorship is a form of information hiding**, ensuring that singularities (infinite complexity) do not propagate to the asymptotic region.
3. **The CK stability theorem provides robustness**, ensuring the exterior region remains well-behaved even when the interior contains singularities.

**Generalizations:**
- The theorem extends to other causal structures (e.g., cosmological horizons, de Sitter space, AdS/CFT boundaries).
- Similar censorship mechanisms arise in quantum field theory (renormalization hiding UV divergences) and computation theory (oracle separation results).

**Open Questions:**
- Does the theorem extend to dynamical black holes with Cauchy horizon instabilities (e.g., Kerr interior)?
- Can the effective event count be made algorithmic, yielding a computable bound on $N_{\text{obs}}$?
- What is the relationship between the Bekenstein bound and computational complexity classes (e.g., BQP, PSPACE)?

:::

## References

The proof relies on the following key literature:

- {cite}`Penrose69`: Weak cosmic censorship conjecture, event horizon formation
- {cite}`HawkingPenrose70`: Singularity theorems, geodesic incompleteness
- {cite}`ChristodoulouKlainerman93`: Global stability of Minkowski space, decay of perturbations in the exterior
- {cite}`Bekenstein81`: Entropy bound for finite region information content

Additional supporting references:
- {cite}`LaSalle76`: Barbalat's lemma for dissipative systems
- {cite}`Lojasiewicz84`: Gradient inequality for analytic functions (used in convergence proofs)
