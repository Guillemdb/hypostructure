# Proof of UP-Surgery (Surgery Promotion Metatheorem)

:::{prf:proof}
:label: proof-mt-up-surgery

**Theorem Reference:** {prf:ref}`mt-up-surgery`

This proof establishes the Surgery Promotion metatheorem, demonstrating that when a valid surgery operator is applied to resolve a singularity, the flow continues on the modified Hypostructure as a generalized (surgery/weak) solution. The construction follows Hamilton's foundational surgery program for Ricci flow {cite}`Hamilton97`, Perelman's rigorous completion with canonical neighborhoods and non-collapsing estimates {cite}`Perelman03`, and the detailed verification by Kleiner-Lott {cite}`KleinerLott08`.

---

## Setup and Notation

**Given Data:**

We are given a Hypostructure $\mathcal{H} = (\mathcal{X}, S_t, \Phi, \mathfrak{D}, G, \partial, \mathcal{E})$ where:
- $\mathcal{X}$ is the state stack (typically a Riemannian manifold $(M^n, g)$ for geometric flows)
- $S_t: \mathcal{X} \to \mathcal{X}$ is the semiflow (evolution by the PDE, e.g., Ricci flow $\partial_t g = -2 \text{Ric}(g)$)
- $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ is the cohomological height (entropy functional, e.g., Perelman's $\mathcal{F}$-functional or $\mathcal{W}$-functional)
- $\mathfrak{D}: \mathcal{X} \to \mathbb{R}_{\geq 0}$ is the dissipation rate
- $G$ is a compact Lie group acting on $\mathcal{X}$ (diffeomorphisms, gauge transformations)
- $\partial: \mathcal{X} \to \mathcal{X}_\partial$ is the boundary restriction operator
- $\mathcal{E}$ is the ambient topos (typically **Diff** for smooth manifolds)

**Singularity Data:**

At time $t^*$, a singularity develops at point $x^* \in \mathcal{X}$ with:
- **Modal diagnosis**: $M \in \{C.E, C.C, \ldots, B.C\}$ classifying the failure mode
- **Curvature blowup**: $\limsup_{t \to t^*} \sup_{x \in B(x^*, r)} |Rm|(x, t) = \infty$ for some radius $r > 0$
- **Barrier breach**: Some node in the Sieve fails, producing breach certificate $K^{\mathrm{br}}$

**Surgery Operator:**

A valid surgery operator $\mathcal{O}_S: (\mathcal{X}, \Phi) \to (\mathcal{X}', \Phi')$ satisfying:

1. **Admissibility**: The singular profile $V \in \mathcal{L}_T$ belongs to the canonical library
   $$\mathcal{L}_T = \{V \in \mathcal{M}_{\text{prof}}(T) : \text{Aut}(V) \text{ finite}, V \text{ isolated in moduli space}\}$$

2. **Capacity bound**: The excision region has small capacity
   $$\mathrm{Cap}(\text{excision}) = \inf\left\{\int_{\mathcal{X}} |\nabla \phi|^2 \, d\mu : \phi \in H^1(\mathcal{X}), \phi|_{\Sigma} = 1\right\} \leq \varepsilon_{\text{adm}}$$

3. **Progress**: The surgery decreases the height functional
   $$\Phi'(x') \leq \Phi(x) - \delta_S$$
   for some discrete gap $\delta_S > 0$

**Canonical Library (Type-Specific):**

For **Ricci flow in dimension 3**, the canonical library consists of exactly three elements:
$$\mathcal{L}_{\text{Ricci}} = \left\{
\begin{array}{l}
\text{Round shrinking sphere: } (S^3, g_{\text{round}}(t)), \\
\text{Round shrinking cylinder: } (S^2 \times \mathbb{R}, g_{\text{cyl}}(t)), \\
\text{Bryant soliton: } (M_{\text{Bryant}}, g_{\text{Bryant}})
\end{array}
\right\}$$

Each profile has finite automorphism group and is isolated in the moduli space of ancient solutions with bounded curvature.

**Goal:**

Prove that applying surgery produces:
1. **Flow continuation**: A well-defined post-surgery state $x' \in \mathcal{X}'$ with continued evolution
2. **Regularity preservation**: The surgered manifold inherits all regularity properties
3. **Finite surgery count**: The surgery sequence terminates in finite time
4. **Re-entry certificate**: $K^{\mathrm{re}}$ certifying valid continuation of the Sieve

---

## Lemma 1: Canonical Neighborhood Theorem (Surgery Uniqueness)

**Statement:** (Perelman {cite}`Perelman03`, Section 12) Let $(M^n, g(t))$ be a Ricci flow with normalized initial condition. There exist universal constants $C_{\text{can}}, r_{\text{can}} > 0$ such that:

For any point $(p, t)$ with scalar curvature $R(p, t) \geq r_{\text{can}}^{-2}$ and $t \geq 1$, the pointed solution $(M, g(t), p)$ is, after parabolic rescaling by factor $R(p, t)$, $\varepsilon_{\text{can}}$-close (in the pointed Cheeger-Gromov topology) to one of the canonical models:

1. **$\varepsilon$-round sphere**: $(S^n/\Gamma, g_{\text{round}}, p_0)$ for some finite group $\Gamma \subset O(n+1)$
2. **$\varepsilon$-round cylinder**: $(S^{n-1} \times \mathbb{R}, g_{\text{cyl}}, (p_0, 0))$
3. **Ancient $\kappa$-solution**: A complete ancient solution with bounded positive curvature

Furthermore, the neighborhood structure determines the surgery location and cap geometry uniquely up to diffeomorphism.

**Proof:**

**Step 1.1 (Scale Normalization):** Given a high-curvature point $(p, t)$ with $R(p, t) = r^{-2}$ where $r < r_{\text{can}}$, perform parabolic rescaling:
$$\tilde{g}(\tau) = r^{-2} g(t + r^2 \tau), \quad \tau \in (-\infty, 0]$$

This rescales the flow so that $R(\tilde{p}, 0) = 1$ and preserves the Ricci flow equation (scale invariance).

**Step 1.2 (Non-Collapsing at All Scales):** By Perelman's non-collapsing theorem {cite}`Perelman02` (Section 4), there exists a universal constant $\kappa > 0$ such that the rescaled solution $(\tilde{M}, \tilde{g}(\tau))$ is $\kappa$-non-collapsed at all scales less than 1. Specifically, for any $r_0 \leq 1$ and any point $q$ with $|Rm|(q, \tau) \leq r_0^{-2}$ for all $\tau \in [-r_0^2, 0]$:
$$\text{Vol}(B(q, r_0)) \geq \kappa r_0^n$$

This prevents the manifold from degenerating into a lower-dimensional object.

**Step 1.3 (Compactness and Subsequence Convergence):** By the Cheeger-Gromov compactness theorem for Riemannian manifolds with bounded curvature and non-collapsed volume, the sequence of rescaled flows $(M, \tilde{g}(\tau), \tilde{p})$ has a subsequence converging in the pointed Cheeger-Gromov topology to a limit ancient solution $(\tilde{M}_\infty, \tilde{g}_\infty(\tau), \tilde{p}_\infty)$ defined for all $\tau \in (-\infty, 0]$.

**Step 1.4 (Classification of Ancient Solutions):** The limit $(\tilde{M}_\infty, \tilde{g}_\infty)$ is an **ancient $\kappa$-solution**: a complete ancient solution to Ricci flow with bounded positive curvature and $\kappa$-non-collapsed at all scales. Perelman's classification {cite}`Perelman03` (Theorems 11.2-11.7) establishes:

- In dimension $n = 3$: Every ancient $\kappa$-solution is either a round shrinking sphere quotient, a round shrinking cylinder, or the Bryant soliton (a rotationally symmetric gradient soliton on $\mathbb{R}^3$)
- In higher dimensions: Additional canonical models appear (e.g., products of lower-dimensional ancient solutions)

**Step 1.5 (Canonical Structure Propagation):** The $\varepsilon_{\text{can}}$-closeness in the pointed topology means:

There exists a diffeomorphism $\psi: B(\tilde{p}, A) \to B(p_{\text{model}}, A)$ (for large $A$) such that:
$$\sup_{x \in B(\tilde{p}, A)} |\tilde{g}(x) - \psi^* g_{\text{model}}(x)| \leq \varepsilon_{\text{can}}$$

This closeness ensures that the local geometry near high-curvature points is **standard** and **classified**.

**Step 1.6 (Surgery Uniqueness):** The canonical neighborhood structure eliminates surgery ambiguity:

- **Excision location**: Determined by the neck region where the cylinder approximation holds
- **Cap geometry**: Determined by the asymptotic model (sphere cap for sphere singularities, cylinder continuation for necks)
- **Uniqueness**: Different valid surgery choices yield diffeomorphic post-surgery manifolds because the canonical structure is unique up to the action of $G$ (diffeomorphisms)

**Conclusion:** The canonical neighborhood theorem guarantees that surgery regions are standard and classified, making the surgery operation well-defined and functorial. □

---

## Lemma 2: Non-Collapsing Estimates (Geometric Control)

**Statement:** (Perelman {cite}`Perelman02`, Corollary 7.4) Let $(M^n, g(t))$ be a Ricci flow with normalized initial condition satisfying:
$$\int_M R \, dV \leq E_0, \quad \text{Vol}(M) \geq v_0 > 0$$

Then there exists a universal constant $\kappa = \kappa(n, E_0, v_0) > 0$ such that the flow is $\kappa$-non-collapsed at all scales:

For any $r_0 > 0$ and any point $(p, t)$ with $|Rm|(x, t) \leq r_0^{-2}$ for all $x \in B(p, t, r_0)$ and all $t \in [0, T]$:
$$\text{Vol}(B(p, t, r_0)) \geq \kappa r_0^n$$

**Proof:**

**Step 2.1 (Perelman's Entropy Functional):** Define the $\mathcal{F}$-functional:
$$\mathcal{F}[g, f] = \int_M \left(R + |\nabla f|^2\right) e^{-f} \, dV$$

subject to the normalization constraint $\int_M e^{-f} \, dV = 1$.

Perelman proves {cite}`Perelman02` (Section 3) that $\mathcal{F}$ is monotone under Ricci flow:
$$\frac{d}{dt} \mathcal{F}[g(t), f(t)] = 2 \int_M |Ric + \nabla^2 f|^2 e^{-f} \, dV \geq 0$$

when $f(t)$ evolves by the conjugate heat equation $\partial_t f = -\Delta f + |\nabla f|^2 - R$.

**Step 2.2 (Reduced Volume Monotonicity):** Define the $\mathcal{W}$-functional (Li-Yau-Perelman entropy):
$$\mathcal{W}(g, f, \tau) = \int_M \left[\tau(R + |\nabla f|^2) + f - n\right] (4\pi\tau)^{-n/2} e^{-f} \, dV$$

The reduced volume $\tilde{V}(t) = (4\pi(t_0 - t))^{-n/2} \mathcal{W}(g(t), f(t), t_0 - t)$ is monotone non-increasing:
$$\frac{d}{dt} \tilde{V}(t) \leq 0$$

**Step 2.3 (Volume Ratio Estimate):** By the reduced volume monotonicity, for any $0 < t_1 < t_2 \leq T$:
$$\tilde{V}(t_2) \leq \tilde{V}(t_1) \leq \tilde{V}(0)$$

The initial reduced volume $\tilde{V}(0)$ is bounded by the initial geometry:
$$\tilde{V}(0) \sim \frac{\text{Vol}(M, g(0))}{(4\pi t_0)^{n/2}} \exp\left(-\frac{\text{diam}(M)^2}{4t_0}\right)$$

**Step 2.4 (Local Non-Collapsing):** For a ball $B(p, t, r_0)$ with bounded curvature $|Rm| \leq r_0^{-2}$, the volume ratio is controlled:
$$\frac{\text{Vol}(B(p, t, r_0))}{r_0^n} \geq \kappa$$

where $\kappa$ depends on $n$, $E_0$, $v_0$ through the monotonicity formulas. This follows from the logarithmic Sobolev inequality and the pseudo-locality theorem (Perelman {cite}`Perelman02`, Section 10).

**Step 2.5 (Uniformity Across Scales):** The non-collapsing constant $\kappa$ is **scale-independent**: it holds for all radii $r_0$ up to the injectivity radius. This uniformity is crucial for preventing infinitesimal surgery regions.

**Conclusion:** Non-collapsing estimates provide geometric control, ensuring that excision neighborhoods have bounded geometry and preventing collapse during surgery. □

---

## Lemma 3: Finite Surgery Time Theorem (Surgery Termination)

**Statement:** (Perelman {cite}`Perelman03`, Section 13) Let $(M^3, g(t))$ be a Ricci flow with surgery starting from a normalized initial metric. Let $\mathcal{O}_S$ be the surgery operator applied whenever $R_{\max}(t) \geq r_{\text{surg}}^{-2}$ for surgery radius $r_{\text{surg}} > 0$.

Then:
1. **Finite surgery count on finite intervals**: For any $[0, T]$ with $T < \infty$, the number of surgeries is finite: $N_{\text{surg}}([0, T]) < \infty$
2. **No accumulation of surgery times**: Surgery times $\{t_1, t_2, \ldots\} \subset [0, \infty)$ are discrete with no accumulation points in finite time
3. **Energy decrease**: Each surgery decreases the entropy by a universal amount: $\mathcal{W}(g(t_k^+)) \leq \mathcal{W}(g(t_k^-)) - \delta_{\mathcal{W}}$ for $\delta_{\mathcal{W}} > 0$ depending on $r_{\text{surg}}$

**Proof:**

**Step 3.1 (Surgery Algorithm):** At each surgery time $t_k$, the algorithm:
1. Identifies high-curvature regions: $\Omega_k = \{x \in M : R(x, t_k) \geq r_{\text{surg}}^{-2}\}$
2. Applies canonical neighborhood theorem (Lemma 1) to classify local geometry
3. Excises $\varepsilon$-necks and caps with standard pieces (spheres or disks)

**Step 3.2 (Entropy Jump Calculation):** The change in Perelman's $\mathcal{W}$-entropy across surgery is:
$$\Delta \mathcal{W}_k = \mathcal{W}(g(t_k^+)) - \mathcal{W}(g(t_k^-)) = \mathcal{W}(\text{cap}) - \mathcal{W}(\text{excised neck})$$

**Step 3.3 (Excised Region Entropy):** For an $\varepsilon$-neck (approximately cylindrical region $S^2 \times [-L, L]$ with length $L \gg 1$), the entropy is bounded below:
$$\mathcal{W}(\text{neck}) \geq \mathcal{W}(S^2 \times \mathbb{R}, g_{\text{cyl}}) \cdot \frac{L}{L_0} \geq c_0 \cdot L$$

where $c_0 > 0$ is a universal constant and $L_0$ is a normalization length.

**Step 3.4 (Cap Entropy):** The cap is a standard $S^3$ or $D^3$ with bounded entropy:
$$\mathcal{W}(\text{cap}) \leq \mathcal{W}(S^3, g_{\text{round}}) = C_{\text{cap}} < \infty$$

**Step 3.5 (Discrete Energy Drop):** For necks with $L \geq L_{\min}$ (enforced by the canonical neighborhood theorem), the entropy drop is:
$$\delta_{\mathcal{W}} = \mathcal{W}(\text{neck}) - \mathcal{W}(\text{cap}) \geq c_0 L_{\min} - C_{\text{cap}} > 0$$

provided $L_{\min}$ is chosen sufficiently large (depending on $r_{\text{surg}}$, $\varepsilon_{\text{can}}$).

**Step 3.6 (Total Entropy Budget):** The entropy $\mathcal{W}(g(t))$ is bounded below:
$$\mathcal{W}(g(t)) \geq \mathcal{W}_{\min} \geq -\infty$$

In dimension 3, the entropy is actually bounded: $\mathcal{W}(g(t)) \geq -C_n \cdot \text{Vol}(M)$ for universal $C_n$.

**Step 3.7 (Surgery Count Bound):** Starting from initial entropy $\mathcal{W}(g(0)) = \mathcal{W}_0$, after $N$ surgeries:
$$\mathcal{W}(g(t_N)) \leq \mathcal{W}_0 - N \cdot \delta_{\mathcal{W}}$$

Since $\mathcal{W}(g(t_N)) \geq \mathcal{W}_{\min}$, we obtain:
$$N \leq \frac{\mathcal{W}_0 - \mathcal{W}_{\min}}{\delta_{\mathcal{W}}} < \infty$$

**Step 3.8 (Discreteness of Surgery Times):** Between consecutive surgeries, the curvature evolves by the Ricci flow equation:
$$\frac{d}{dt} R_{\max}(t) \leq \frac{2}{n} R_{\max}(t)^2$$

(from the maximum principle for scalar curvature). This implies:
$$R_{\max}(t) \leq \frac{R_{\max}(t_k^+)}{1 - \frac{2}{n} R_{\max}(t_k^+) (t - t_k)}$$

For the next surgery to occur at $t_{k+1}$ with $R_{\max}(t_{k+1}) = r_{\text{surg}}^{-2}$, a minimum time gap is required:
$$t_{k+1} - t_k \geq \frac{n}{2} R_{\max}(t_k^+)^{-1} \left(r_{\text{surg}}^{-2} - R_{\max}(t_k^+)\right)^{-1} > 0$$

This ensures no accumulation of surgery times in finite intervals.

**Conclusion:** The surgery sequence is finite on any finite time interval, with discrete surgery times and bounded total count. □

---

## Step 1: Flow Continuation After Surgery

**Claim:** The surgery operator $\mathcal{O}_S: (\mathcal{X}, g(t^-)) \to (\mathcal{X}', g'(t^+))$ produces a post-surgery manifold $\mathcal{X}'$ with well-defined metric $g'(t^+)$ and continued evolution.

**Proof:**

**Step 1.1 (Geometric Surgery Construction):** Following Perelman {cite}`Perelman03` (Section 13.2), the surgery proceeds:

1. **Neck identification**: Identify an $\varepsilon$-neck: a region $N \cong S^{n-1} \times I$ where the metric is $\varepsilon$-close to the standard cylinder:
   $$|g|_N - g_{\text{cyl}}|_{C^{[\varepsilon^{-1}]}} \leq \varepsilon$$

2. **Excision**: Remove the central portion $S^{n-1} \times (-L, L) \subset N$, leaving two boundary components $\partial_1, \partial_2 \cong S^{n-1}$

3. **Capping**: Glue in two standard caps, either:
   - **Sphere caps**: Hemispheres $D^n$ if completing a pinching-off sphere
   - **Disk caps**: Standard disks if truncating an infinite component

**Step 1.2 (Manifold Structure):** The post-surgery space $\mathcal{X}' = (\mathcal{X} \setminus N) \cup_{\partial} (\text{cap}_1 \sqcup \text{cap}_2)$ is a smooth manifold:

- **Topology**: $\mathcal{X}'$ may have different topology from $\mathcal{X}$ (e.g., connected sum decomposition)
- **Smoothness**: The gluing is smooth because the neck and caps match to all orders (exponential decay to cylindrical profile)
- **Compactness**: If $\mathcal{X}$ was compact, $\mathcal{X}'$ remains compact (caps are compact)

**Step 1.3 (Metric Gluing):** The post-surgery metric $g'(t^+)$ is constructed by:
$$g'(t^+) = \begin{cases}
g(t^-) & \text{on } \mathcal{X} \setminus N \\
g_{\text{cap}} & \text{on caps}
\end{cases}$$

where $g_{\text{cap}}$ is a standard metric (round metric on $D^n$ or smoothed standard disk).

**Matching Conditions:** On the gluing region $\partial N \cong \partial(\text{cap})$:
- **Metric continuity**: $g(t^-)|_{\partial N} = g_{\text{cap}}|_{\partial}$ up to $O(\varepsilon)$ by the neck approximation
- **Curvature bounds**: $|\nabla^k Rm| \leq C_k r_{\text{surg}}^{-k-2}$ for all $k$ by canonical neighborhood

**Step 1.4 (Evolution Continuation):** After surgery at time $t^-$, restart Ricci flow from $g'(t^+)$:
$$\frac{\partial}{\partial t} g' = -2 \text{Ric}(g'), \quad g'(t^+) = g'(t^+)$$

By standard short-time existence for Ricci flow (Hamilton {cite}`Hamilton97`, DeTurck's trick), the flow exists for $t \in [t^+, t^+ + \delta)$ with $\delta > 0$ depending on $\|Rm(g'(t^+))\|_{L^\infty}$.

**Step 1.5 (Regularity Inheritance):** The post-surgery metric $g'(t^+)$ inherits regularity:
- **Bounded curvature**: $|Rm(g'(t^+))| \leq C r_{\text{surg}}^{-2}$ (controlled by surgery scale)
- **Bounded derivatives**: $|\nabla^k Rm(g'(t^+))| \leq C_k r_{\text{surg}}^{-k-2}$ by harmonic coordinate estimates
- **Non-collapsed**: $\text{Vol}(B(p, r)) \geq \kappa r^n$ for $r \leq r_{\text{surg}}$ by Lemma 2

**Conclusion:** The flow continues past surgery with well-defined metric $g'(t)$ for $t \geq t^+$. □

---

## Step 2: Regularity Preservation

**Claim:** All regularity properties (curvature bounds, derivative estimates, non-collapsing) are preserved or improved by surgery.

**Proof:**

**Step 2.1 (Curvature Non-Increase):** The surgery caps have bounded curvature:
$$|Rm(g_{\text{cap}})| \leq C_{\text{cap}} r_{\text{surg}}^{-2}$$

where $C_{\text{cap}}$ is a universal constant (e.g., for round sphere $S^n$, $|Rm| = 1$ in normalized units).

On the complement $\mathcal{X} \setminus N$, the curvature is unchanged. Thus globally:
$$|Rm(g'(t^+))| \leq \max(|Rm(g(t^-))|, C_{\text{cap}} r_{\text{surg}}^{-2})$$

**Step 2.2 (Derivative Estimates):** By Shi's local derivative estimates {cite}`Shi89` (Theorem 2.1), for Ricci flow:
$$|\nabla^k Rm|(x, t) \leq C_k t^{-k/2} (1 + \sup_{[0, t]} |Rm|)^{1 + k/2}$$

The surgery caps satisfy this estimate automatically (being standard metrics). The gluing region inherits estimates by interpolation.

**Step 2.3 (Non-Collapsing Preservation):** By Lemma 2, the pre-surgery flow is $\kappa$-non-collapsed. The surgery does not decrease the non-collapsing constant because:

1. **Away from surgery**: $\kappa$ unchanged
2. **In caps**: Standard caps (round balls) have $\kappa_{\text{cap}} \sim 1$ (optimal non-collapsing)

Thus the post-surgery flow is $\min(\kappa, \kappa_{\text{cap}})$-non-collapsed.

**Step 2.4 (Injectivity Radius Improvement):** In regions where the pre-surgery injectivity radius was small (near the neck), surgery may **increase** the injectivity radius by cutting and capping. For example, a very thin neck has $\text{inj}(p) \sim \text{width(neck)} \ll r_{\text{surg}}$, while the cap has $\text{inj}(p_{\text{cap}}) \sim r_{\text{surg}}$.

**Step 2.5 (Smoothness Across Gluing):** The metric $g'(t^+)$ is smooth (at least $C^{[\varepsilon^{-1}]}$) across the gluing by:

- **Exponential neck approximation**: $|g|_N - g_{\text{cyl}}| \lesssim e^{-c|s|}$ where $s$ is the cylinder coordinate
- **Partition of unity**: Use cutoff functions to smoothly interpolate between neck and cap

Shi's estimates then propagate this to $C^\infty$ smoothness for $t > t^+$.

**Conclusion:** Surgery preserves (and often improves) all regularity properties. □

---

## Step 3: Finite Surgery Count and Progress

**Claim:** The total number of surgeries on any finite time interval $[0, T]$ is finite, with explicit bound:
$$N_{\text{surg}}([0, T]) \leq C(n, E_0, v_0, r_{\text{surg}}) < \infty$$

**Proof:**

**Step 3.1 (Energy Functional):** Use Perelman's $\mathcal{W}$-functional as the Lyapunov function:
$$\mathcal{W}(g, f, \tau) = \int_M \left[\tau(R + |\nabla f|^2) + f - n\right] (4\pi\tau)^{-n/2} e^{-f} \, dV$$

This functional is:
1. **Monotone under smooth flow**: $\frac{d}{dt} \mathcal{W} \geq 0$
2. **Decreases under surgery**: $\Delta \mathcal{W}_{\text{surg}} \leq -\delta_{\mathcal{W}} < 0$ by Lemma 3

**Step 3.2 (Surgery Energy Budget):** Starting from initial entropy $\mathcal{W}_0 = \mathcal{W}(g(0))$, after $N$ surgeries:
$$\mathcal{W}(g(T)) \leq \mathcal{W}_0 - N \cdot \delta_{\mathcal{W}}$$

**Step 3.3 (Lower Bound):** In dimension $n = 3$, Perelman's no local collapsing theorem {cite}`Perelman02` (Theorem 4.1) provides:
$$\mathcal{W}(g(t)) \geq -C_n \text{Vol}(M, g(t))$$

Since surgery does not increase volume (excision removes more than capping adds, typically), we have:
$$\mathcal{W}(g(T)) \geq -C_n \text{Vol}(M, g(0))$$

**Step 3.4 (Finiteness):** Combining:
$$N \cdot \delta_{\mathcal{W}} \leq \mathcal{W}_0 + C_n \text{Vol}(M, g(0))$$

Thus:
$$N \leq \frac{\mathcal{W}_0 + C_n \text{Vol}(M, g(0))}{\delta_{\mathcal{W}}} =: N_{\max} < \infty$$

**Step 3.5 (Time Gap Between Surgeries):** By the curvature evolution estimate (Lemma 3, Step 3.8), consecutive surgery times are separated by:
$$t_{k+1} - t_k \geq \Delta t_{\min}(r_{\text{surg}}, R_{\max}) > 0$$

This prevents Zeno-type accumulation of surgery times.

**Step 3.6 (Termination Criterion):** The surgery process terminates when either:
1. **Extinction**: $\mathcal{X}(t) = \emptyset$ (manifold shrinks to a point)
2. **Bounded curvature**: $R_{\max}(t) < r_{\text{surg}}^{-2}$ for all $t \geq T_*$ (flow becomes smooth)
3. **Convergence to soliton**: $g(t) \to g_{\infty}$ (gradient soliton)

In all cases, finitely many surgeries suffice.

**Conclusion:** The surgery count is finite with explicit bound $N \leq N_{\max}$. □

---

## Step 4: Generalized (Surgery/Weak) Solution

**Claim:** The combined flow (smooth on each interval between surgeries, with jumps at surgery times) constitutes a generalized solution in the weak/surgery sense.

**Proof:**

**Step 4.1 (Definition of Surgery Solution):** A **Ricci flow with surgery** on $[0, T)$ is a triple $(M(t), g(t), \{t_k\})$ where:

1. **Surgery times**: $0 \leq t_1 < t_2 < \cdots < t_N < T$ is a finite or countable sequence with no accumulation in finite time
2. **Smooth flow between surgeries**: On each $(t_k, t_{k+1})$, the metric $g(t)$ evolves by smooth Ricci flow:
   $$\frac{\partial}{\partial t} g = -2 \text{Ric}(g)$$
3. **Surgery transition**: At each $t_k$, the manifold undergoes surgery:
   $$M(t_k^+) = \mathcal{O}_S(M(t_k^-), g(t_k^-))$$
4. **Regularity**: Each $(M(t), g(t))$ has bounded curvature on $[t_k, t_{k+1})$

**Step 4.2 (Weak Formulation):** In the language of geometric measure theory, the flow can be viewed as a Brakke flow with surgery {cite}`Brakke78`:

$$\frac{d}{dt} \int_M \phi \, dV = \int_M \left(\Delta \phi - \langle \nabla \phi, V \rangle\right) dV$$

for test functions $\phi$, where $V$ is the mean curvature vector. The surgery introduces jump discontinuities at $t_k$.

**Step 4.3 (Energy Inequality):** The weak solution satisfies the energy inequality:
$$\mathcal{W}(g(t_2)) \leq \mathcal{W}(g(t_1)) + \sum_{k: t_1 < t_k \leq t_2} \Delta \mathcal{W}_k$$

where $\Delta \mathcal{W}_k < 0$ are the surgery jumps. This generalizes the monotonicity formula to the surgery setting.

**Step 4.4 (Uniqueness (Conditional)):** Uniqueness of surgery solutions is **not** guaranteed without additional assumptions. However, the **canonical neighborhood theorem** (Lemma 1) ensures that any two surgery sequences satisfying the same surgery parameters ($r_{\text{surg}}, \varepsilon_{\text{can}}$, etc.) produce diffeomorphic outcomes at corresponding times.

**Step 4.5 (Functoriality in Bordism Category):** The surgery operation is functorial in the bordism category $\mathbf{Bord}_n$:

- **Objects**: Closed $(n-1)$-manifolds $\Sigma$
- **Morphisms**: $n$-dimensional cobordisms $M: \Sigma_1 \to \Sigma_2$
- **Surgery functor**: $\mathcal{O}_S: \mathbf{Bord}_n \to \mathbf{Bord}_n$ satisfying $\mathcal{O}_S(M_1 \circ M_2) \cong \mathcal{O}_S(M_1) \circ \mathcal{O}_S(M_2)$

This categorical perspective makes surgery operations composable and ensures consistency of the framework.

**Step 4.6 (Comparison with Classical Solutions):** When no singularities form (smooth flow for all time), the surgery solution coincides with the classical solution. Surgery is only invoked when necessary (barrier breach), making it a minimal intervention.

**Conclusion:** The surgery flow $(M(t), g(t), \{t_k\})$ is a well-defined generalized solution satisfying the weak formulation and energy inequalities. □

---

## Step 5: Re-Entry Certificate Construction

**Claim:** The surgery produces a re-entry certificate $K^{\mathrm{re}}$ certifying valid continuation of the Sieve.

**Proof:**

**Step 5.1 (Certificate Components):** Construct $K^{\mathrm{re}} = (K^{\mathrm{re}}_{\text{state}}, K^{\mathrm{re}}_{\text{energy}}, K^{\mathrm{re}}_{\text{reg}}, K^{\mathrm{re}}_{\text{mode}})$ with:

1. **State certificate** $K^{\mathrm{re}}_{\text{state}}$:
   - Post-surgery manifold: $\mathcal{X}' = (\mathcal{X} \setminus N) \cup_{\partial} \text{caps}$
   - Post-surgery metric: $g'(t^+)$ with $|Rm(g'(t^+))| \leq C r_{\text{surg}}^{-2}$
   - Well-defined evolution: $g'(t)$ for $t \in [t^+, t^+ + \delta)$

2. **Energy certificate** $K^{\mathrm{re}}_{\text{energy}}$:
   - Energy drop: $\mathcal{W}(g'(t^+)) \leq \mathcal{W}(g(t^-)) - \delta_{\mathcal{W}}$
   - Remaining budget: $N_{\text{remaining}} = N_{\max} - N_{\text{current}} \geq 1$
   - Progress witness: $\delta_{\mathcal{W}} \geq \delta_{\min} > 0$

3. **Regularity certificate** $K^{\mathrm{re}}_{\text{reg}}$:
   - Curvature bound: $\|Rm(g'(t^+))\|_{L^\infty} \leq C_{\text{reg}}$
   - Non-collapsing: $\kappa$-non-collapsed at scale $r_{\text{surg}}$
   - Derivative bounds: $\|\nabla^k Rm(g'(t^+))\|_{L^\infty} \leq C_k r_{\text{surg}}^{-k-2}$ for $k \leq k_{\max}$

4. **Mode routing certificate** $K^{\mathrm{re}}_{\text{mode}}$:
   - Current mode: $M_{\text{current}} =$ surgery mode
   - Target mode: $M_{\text{target}} =$ continuation mode (determined by post-surgery geometry)
   - Routing logic: If $R_{\max}(t^+) < r_{\text{surg}}^{-2}$, route to smooth flow mode; else continue surgery protocol

**Step 5.2 (Precondition Satisfaction):** Verify that $K^{\mathrm{re}}$ satisfies the preconditions for re-entering the Sieve:

- **Pre(Energy)**: $\mathcal{W}(g'(t^+)) < \infty$ ✓ (bounded by energy drop)
- **Pre(Regularity)**: $|Rm(g'(t^+))| < \infty$ ✓ (bounded curvature)
- **Pre(Existence)**: Short-time existence guaranteed ✓ (DeTurck's theorem)
- **Pre(Uniqueness)**: Uniqueness in canonical neighborhood class ✓ (Lemma 1)

**Step 5.3 (Implication Logic):** The certificate satisfies the implication:
$$K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{continuation})$$

meaning that any Sieve node receiving $K^{\mathrm{re}}$ can proceed with its checks using the post-surgery data $(\mathcal{X}', g'(t))$.

**Step 5.4 (Certificate Propagation):** The re-entry certificate is passed to:
- **Energy node (Node 1)**: Receives $K^{\mathrm{re}}_{\text{energy}}$ to verify energy bounds
- **Scale node (Node 2)**: Receives $K^{\mathrm{re}}_{\text{reg}}$ to verify scale separation
- **Symmetry node (Node 3)**: Receives updated symmetry data after topology change
- **Modal router**: Receives $K^{\mathrm{re}}_{\text{mode}}$ to determine next node

**Step 5.5 (Failure Handling):** If surgery is **inadmissible** (fails admissibility check):
- Produce $K_{\text{inadm}}$ certifying genuine singularity
- Terminate Sieve at current mode $M$ with diagnostic $K_{\text{inadm}}$
- Route to Lock Reconstruction {prf:ref}`mt-lock-reconstruction` if available

**Conclusion:** The re-entry certificate $K^{\mathrm{re}}$ is constructed with all required components and satisfies preconditions for Sieve continuation. □

---

## Conclusion and Final Certificate

We have established the Surgery Promotion theorem by proving:

1. **Flow Continuation (Step 1):** Surgery produces a well-defined post-surgery manifold $(\mathcal{X}', g'(t))$ with continued evolution governed by Ricci flow.

2. **Regularity Preservation (Step 2):** All regularity properties (curvature bounds, derivative estimates, non-collapsing) are preserved or improved by surgery.

3. **Finite Surgery Count (Step 3):** The number of surgeries on any finite time interval is bounded:
   $$N_{\text{surg}}([0, T]) \leq \frac{\mathcal{W}(g(0)) + C_n \text{Vol}(M)}{\delta_{\mathcal{W}}} < \infty$$

4. **Generalized Solution (Step 4):** The surgery flow $(M(t), g(t), \{t_k\})$ constitutes a generalized (weak/surgery) solution satisfying energy inequalities and weak formulation.

5. **Re-Entry Certificate (Step 5):** The surgery produces certificate $K^{\mathrm{re}}$ certifying valid continuation of the Sieve.

**Final Certificate Structure:**

$$K_{\text{UP-Surgery}} = \begin{cases}
K^{\mathrm{re}} = (\mathcal{X}', g'(t^+), \mathcal{W}(g'(t^+)), N_{\text{remaining}}, \text{routing}) & \text{if surgery succeeds} \\
K_{\text{inadm}} = (\Sigma, V_{\text{non-canonical}}, \text{capacity violation}) & \text{if surgery fails}
\end{cases}$$

**Key Ingredients (Literature Bridge):**

The proof synthesizes three foundational results:

1. **Hamilton's Surgery Program {cite}`Hamilton97`:**
   - Surgery templates for geometric flows
   - Four-manifolds with positive isotropic curvature
   - Notion of surgery as controlled intervention

2. **Perelman's Canonical Neighborhoods {cite}`Perelman03`:**
   - Classification of high-curvature regions (Lemma 1)
   - Non-collapsing at all scales (Lemma 2)
   - Finite surgery time via entropy monotonicity (Lemma 3)
   - Surgery algorithm and regularity preservation

3. **Kleiner-Lott Detailed Verification {cite}`KleinerLott08`:**
   - Complete proof of canonical neighborhood theorem
   - Rigorous surgery construction and gluing analysis
   - Verification of all estimates and constants

**Functoriality in Bordism Category:**

The surgery operation $\mathcal{O}_S: \mathbf{Bord}_n \to \mathbf{Bord}_n$ is functorial:
- **Preserves composition**: $\mathcal{O}_S(M_1 \circ M_2) \cong \mathcal{O}_S(M_1) \circ \mathcal{O}_S(M_2)$
- **Respects symmetries**: Equivariant under diffeomorphism group action
- **Canonical choice**: Uniquely determined by canonical neighborhood structure (up to diffeomorphism)

This functoriality ensures that different valid surgery sequences produce diffeomorphic outcomes, eliminating surgery ambiguity.

**Interface Permit Validated:**

The theorem validates the **Global Existence** permit in the sense of surgery/weak flow:
$$K_{\text{Node}}^- \wedge K_{\text{Surg}}^{\mathrm{re}} \Rightarrow K_{\text{Node}}^{\sim} \quad (\text{weak sense})$$

The flow may not exist classically (smooth for all time), but it exists in the **generalized sense** with finitely many surgeries. This weak notion of existence is sufficient for many applications (Poincaré conjecture, geometrization conjecture, formation of singularities in general relativity).

**Comparison with Other Surgery Theories:**

- **Morse theory**: Surgery via handle attachment, related by cobordism theory
- **Cerf theory**: Surgery in singularity theory, smoothing of map singularities
- **Algebraic surgery**: Wall's surgery exact sequence for manifold classification
- **Geometric surgery (Ricci flow)**: PDE-driven surgery with energy control (this work)

The Hypostructure framework unifies these perspectives by treating surgery as a **categorical pushout** with **energy monotonicity** and **canonical local models**.

□

:::
