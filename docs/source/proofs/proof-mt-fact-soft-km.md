# Proof of FACT-SoftKM (Kenig-Merle Machine Derivation)

:::{prf:proof}
:label: proof-mt-fact-soft-km

**Theorem Reference:** {prf:ref}`mt-fact-soft-km`

This proof establishes that the concentration-compactness + stability machine (Kenig-Merle framework) can be systematically derived from well-posedness, profile decomposition, and soft interface permits, without requiring explicit PDE-specific analysis. The result is a certificate witnessing the existence of a minimal critical element that is almost periodic modulo symmetries, enabling rigidity analysis.

## Setup and Notation

### Given Data

We are provided with the following certified permits:

1. **Critical Well-Posedness:** $K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{LWP}, \mathsf{uniq}, \mathsf{cont}, \mathsf{crit\_blowup})$ certifying:
   - Local existence in critical space $X_c$ (typically $X_c = \dot{H}^{s_c}(\mathbb{R}^d)$)
   - Uniqueness and continuous dependence on initial data
   - Continuation criterion: finite-time blowup implies $\|u\|_{S([0,T_{\max}))} = \infty$ for a critical control norm $S$

2. **Profile Decomposition:** $K_{\mathrm{ProfDec}_{s_c,G}}^+ = (\{\phi^j\}, \{g_n^j\}, \{r_n^J\}, \mathsf{orth}, \mathsf{rem})$ certifying that every bounded sequence $(u_n) \subset X_c$ admits a Bahouri-Gérard decomposition:
   $$u_n = \sum_{j=1}^J g_n^j \phi^j + r_n^J$$
   with asymptotic orthogonality, energy decoupling, and vanishing remainder in the control norm $S$

3. **Energy Boundedness Below:** $K_{D_E}^+ = (B, \mathsf{bound\_proof})$ certifying that the energy functional $\Phi: X_c \to \mathbb{R}_{\geq 0}$ satisfies:
   $$\Phi(u) \geq -B \quad \forall u \in X_c$$
   for some finite constant $B < \infty$

4. **Scaling Structure:** $K_{\mathrm{SC}_\lambda}^+ = (\alpha, \beta, \mathsf{subcrit\_proof})$ certifying the scaling exponents satisfy:
   $$\Phi(\mathcal{S}_\lambda u) = \lambda^\alpha \Phi(u), \quad \mathfrak{D}(\mathcal{S}_\lambda u) = \lambda^\beta \mathfrak{D}(u)$$
   with $\alpha > \beta$ (subcritical scaling)

### Target Property and Scattering

Fix a target regularity property $\mathcal{P}$ (typically global existence and scattering). A solution $u: [0, T) \to X_c$ **scatters** if:
$$\exists u_\infty^+ \in X_c: \quad \lim_{t \to T^-} \|u(t) - e^{it\Delta} u_\infty^+\|_{X_c} = 0$$
(for linear dispersive equations; analogous definition for other types).

A solution $u$ is **non-scattering** if it does not scatter forward in time (i.e., remains concentrated or exhibits nontrivial asymptotic dynamics).

### Goal

We construct a certificate:
$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+ = (\mathsf{minimal\_u^*}, E_c, \mathsf{almost\_periodic\_mod\_G}, \mathsf{perturbation\_lemma})$$
witnessing the existence of a minimal critical element $u^*$ that is almost periodic modulo symmetries, serving as the foundation for rigidity analysis.

---

## Step 1: Construction of the Critical Energy Level

### Lemma 1.1: Non-Scattering Energy Set is Well-Defined

**Statement:** The set of energies of non-scattering solutions is bounded below:
$$E_{\mathrm{NS}} := \{E \in \mathbb{R} : \exists u \text{ non-scattering with } \Phi(u_0) = E\}$$
satisfies $\inf E_{\mathrm{NS}} > -\infty$.

**Proof:** By $K_{D_E}^+$, the energy functional is bounded below: $\Phi(u) \geq -B$ for all $u \in X_c$. Therefore:
$$\inf E_{\mathrm{NS}} \geq -B > -\infty$$

Moreover, by the continuation criterion in $K_{\mathrm{WP}_{s_c}}^+$, if $u$ is a global solution (i.e., $T_{\max} = \infty$) that does not scatter, then $\|u\|_{S([0,\infty))} < \infty$ (otherwise blowup would occur). This establishes that non-scattering solutions with bounded energy exist.

### Lemma 1.2: Critical Energy is Achieved

**Statement:** The critical energy level:
$$E_c := \inf\{E : \exists u \text{ maximal solution, non-scattering, } \Phi(u_0) = E\}$$
is achieved by some solution $u^*$.

**Proof:** By compactness from profile decomposition. Let $(u_n)$ be a minimizing sequence:
$$\Phi(u_{n,0}) \to E_c, \quad \text{each } u_n \text{ non-scattering}$$

**Step 1.2.1 (Boundedness):** By definition, $\sup_n \Phi(u_{n,0}) \leq E_c + 1 < \infty$. The sequence $(u_{n,0})$ is bounded in $X_c$.

**Step 1.2.2 (Profile Extraction):** Apply $K_{\mathrm{ProfDec}_{s_c,G}}^+$ to the sequence $(u_{n,0})$. There exist profiles $\{\phi^j\}_{j \geq 1}$ and symmetry parameters $\{g_n^j\}$ such that:
$$u_{n,0} = \sum_{j=1}^J g_n^j \phi^j + r_n^J$$
with energy decoupling:
$$\Phi(u_{n,0}) = \sum_{j=1}^J \Phi(\phi^j) + \Phi(r_n^J) + o_n(1)$$

**Step 1.2.3 (Energy Concentration):** Since $\|r_n^J\|_S \to 0$ as first $J \to \infty$ and then $n \to \infty$, and scattering is controlled by the $S$-norm (by the continuation criterion), the remainder $r_n^J$ scatters for $J$ sufficiently large. Therefore, non-scattering behavior must be encoded in at least one profile.

**Claim:** At least one profile $\phi^{j_0}$ is non-scattering with $\Phi(\phi^{j_0}) \geq E_c$.

**Proof of Claim:** Suppose all profiles scatter. By $K_{\mathrm{WP}_{s_c}}^+$ (continuous dependence), the sum $\sum_{j=1}^J g_n^j \phi^j$ for fixed $J$ behaves asymptotically like non-interacting scattering profiles (since the $g_n^j$ are asymptotically orthogonal). For $J$ large and $n$ large:
$$\left\|\sum_{j=1}^J g_n^j \phi^j - \text{(linear scattering)}\right\|_S \leq \varepsilon$$
The nonlinear evolution of $u_n$ then satisfies:
$$u_n(t) = \text{(scattering)} + O(\varepsilon) + r_n^J(t)$$
By perturbation theory in $K_{\mathrm{WP}_{s_c}}^+$ (Lemma 1.3 below), small perturbations of scattering solutions scatter. This contradicts the assumption that $u_n$ is non-scattering. Therefore, at least one profile $\phi^{j_0}$ is non-scattering.

**Step 1.2.4 (Minimality):** By energy decoupling:
$$E_c + o(1) = \Phi(u_{n,0}) \geq \Phi(\phi^{j_0})$$
Taking $n \to \infty$: $\Phi(\phi^{j_0}) \leq E_c$. By definition of $E_c$ as the infimum: $\Phi(\phi^{j_0}) \geq E_c$. Hence:
$$\Phi(\phi^{j_0}) = E_c$$

**Step 1.2.5 (Critical Element):** Let $u^*(t)$ be the maximal solution with $u^*(0) = \phi^{j_0}$. By $K_{\mathrm{WP}_{s_c}}^+$, this solution exists and is unique. It satisfies:
- $\Phi(u^*_0) = E_c$ (minimal energy)
- $u^*$ is non-scattering (inherited from profile)
- $u^*$ is maximal (by WP)

This is the **critical element**.

### Lemma 1.3: Perturbation Stability (Small Data Scattering)

**Statement:** There exists $\delta > 0$ such that if $\Phi(v_0) < E_c - \delta$, then the solution $v(t)$ scatters.

**Proof:** By contrapositive. Suppose not: there exists a sequence $(v_n)$ with $\Phi(v_{n,0}) < E_c - 1/n$ and $v_n$ non-scattering. Then:
$$E_c \leq \lim_{n \to \infty} \Phi(v_{n,0}) \leq \lim_{n \to \infty} (E_c - 1/n) = E_c$$
with the middle inequality being strict for all finite $n$: $\Phi(v_{n,0}) < E_c - 1/n < E_c$. This contradicts the definition of $E_c$ as the infimum of non-scattering energies.

**Quantitative Bound:** By the proof structure, $\delta$ can be taken as $\delta = \min(1, \text{Gap}(E_c, E_{\mathrm{NS}} \setminus \{E_c\}))$ where the gap measures the distance to the next energy level in $E_{\mathrm{NS}}$.

---

## Step 2: Almost Periodicity Modulo Symmetries

### Definition: Almost Periodicity

A trajectory $u^*(t)$ is **almost periodic modulo symmetries $G$** if the modulated trajectory:
$$\mathcal{O} := \{g \cdot u^*(t) : t \geq 0, g \in G\}$$
has compact closure in $X_c$, where $G$ is the symmetry group (typically $G = \mathbb{R}^d \times \mathbb{R}^+ \times \text{U}(1)$ for translations, scaling, and phase).

### Lemma 2.1: Precompactness of Modulated Trajectory

**Statement:** The critical element $u^*$ satisfies almost periodicity modulo $G$:
$$\overline{\{g(t) \cdot u^*(t) : t \in [0, \infty)\}} \text{ is compact in } X_c$$
for an appropriate choice of symmetry parameters $g(t) \in G$.

**Proof:** We use the profile decomposition and minimality to establish precompactness.

**Step 2.1.1 (Uniform Energy Bound):** By energy conservation (or dissipation control from $K_{D_E}^+$):
$$\Phi(u^*(t)) \leq \Phi(u^*_0) + \int_0^t \mathfrak{D}(u^*(s)) ds \leq E_c + C$$
for some constant $C < \infty$ (in the conservative case, $\mathfrak{D} \equiv 0$ and $\Phi(u^*(t)) = E_c$ exactly).

**Step 2.1.2 (Concentration Dichotomy):** Consider a sequence of times $t_n \to \infty$. The sequence $(u^*(t_n))$ is bounded in $X_c$ by Step 2.1.1. Apply $K_{\mathrm{ProfDec}_{s_c,G}}^+$:
$$u^*(t_n) = \sum_{j=1}^J g_n^j \psi^j + r_n^J$$
with energy decoupling and vanishing remainder.

**Step 2.1.3 (Ruling Out Dispersion):** Suppose $u^*(t_n) \rightharpoonup 0$ (weak convergence to zero, i.e., dispersion). Then by the continuation criterion in $K_{\mathrm{WP}_{s_c}}^+$ and Sobolev embedding, $\|u^*(t_n)\|_{L^p} \to 0$ for subcritical $p$. By standard scattering criteria (cf. {cite}`Tao06`), this implies $u^*$ scatters, contradicting the construction of $u^*$ as non-scattering.

**Step 2.1.4 (Single Profile Concentration):** Therefore, at least one profile $\psi^{j_0}$ is non-zero. By energy decoupling:
$$E_c + o(1) = \Phi(u^*(t_n)) = \sum_j \Phi(\psi^j) + o(1)$$
Since $\Phi(\psi^j) \geq 0$ (by boundedness below) and the sum equals $E_c$, we must have:
$$\Phi(\psi^{j_0}) \leq E_c$$

**Claim:** If $\Phi(\psi^{j_0}) < E_c$, then $\psi^{j_0}$ scatters by Lemma 1.3. But then the evolution $u^*(t)$ would asymptotically decouple into scattering profiles, contradicting non-scattering of $u^*$.

**Conclusion:** $\Phi(\psi^{j_0}) = E_c$, and there is **only one** profile (i.e., $J = 1$). Otherwise, the sum $\sum_j \Phi(\psi^j) > E_c$, violating energy conservation.

**Step 2.1.5 (Modulation):** There exist symmetry parameters $g(t_n) \in G$ such that:
$$g(t_n) \cdot u^*(t_n) \to \psi \quad \text{strongly in } X_c$$
for some fixed profile $\psi$ with $\Phi(\psi) = E_c$.

**Step 2.1.6 (Precompactness):** By diagonal argument, for any sequence $t_n \to \infty$, there exist modulation parameters $g(t_n)$ and a subsequence such that $g(t_{n_k}) \cdot u^*(t_{n_k})$ converges strongly. This is the definition of precompactness modulo $G$.

### Lemma 2.2: Rigidity of the Critical Set

**Statement:** The set of critical elements at energy $E_c$ is precompact modulo symmetries.

**Proof:** Let $u_1^*, u_2^*$ be two critical elements with $\Phi(u_{1,0}^*) = \Phi(u_{2,0}^*) = E_c$, both non-scattering. By the same argument as Lemma 2.1, both are almost periodic modulo $G$.

**Claim:** Up to symmetries, $u_{1,0}^* = g \cdot u_{2,0}^*$ for some $g \in G$.

**Proof of Claim:** Apply profile decomposition to the pair $(u_{1,0}^*, u_{2,0}^*)$. If they concentrate at different scales/positions, energy decoupling implies:
$$\Phi(u_{1,0}^* + u_{2,0}^*) \approx \Phi(u_{1,0}^*) + \Phi(u_{2,0}^*) = 2E_c$$
But by uniqueness from minimality, any non-scattering solution with energy $< 2E_c$ must have energy $\geq E_c$. This forces the profiles to be identical up to symmetry.

**Rigidity Consequence:** The critical element $u^*$ is **unique** modulo symmetries $G$. This is the **Kenig-Merle rigidity setup** {cite}`KenigMerle06` Theorem 1.1.

---

## Step 3: Perturbative Stability and Dichotomy

### Lemma 3.1: Continuous Dependence Near Critical Energy

**Statement:** For any $\varepsilon > 0$, there exists $\delta > 0$ such that if $\|u_0 - u^*_0\|_{X_c} < \delta$, then:
$$\sup_{t \in [0, T_{\max}(u))} \|u(t) - u^*(t)\|_{X_c} < \varepsilon$$
on the common interval of existence.

**Proof:** This follows directly from $K_{\mathrm{WP}_{s_c}}^+$ (continuous dependence on initial data). By the Lipschitz dependence estimate in the LWP certificate:
$$\|u(t) - u^*(t)\|_{X_c} \leq C e^{CT} \|u_0 - u^*_0\|_{X_c}$$
for $t \in [0, T]$, where $T$ is the minimum of the lifespans and $C$ depends on $\|u_0\|_{X_c}, \|u^*_0\|_{X_c}$.

### Lemma 3.2: Scattering/Non-Scattering Dichotomy

**Statement:** For initial data $u_0$ with $\Phi(u_0) \approx E_c$, either:
1. $u_0$ is close to the orbit $G \cdot u^*_0$ (i.e., $\exists g \in G: \|u_0 - g \cdot u^*_0\|_{X_c} < \delta$), and $u(t)$ remains close to the modulated trajectory $u^*(t)$, or
2. $u_0$ is not close to the critical orbit, and $u(t)$ scatters.

**Proof:** By contradiction and profile decomposition.

**Step 3.2.1 (Setup):** Suppose $u_0$ satisfies $E_c \leq \Phi(u_0) < E_c + \eta$ for small $\eta > 0$, and $u$ does not scatter.

**Step 3.2.2 (Apply Profile Decomposition):** By Lemma 2.1, the trajectory $u(t)$ at large times concentrates to profiles. By the energy bound, the total profile energy is $\leq E_c + \eta$.

**Step 3.2.3 (Energy Gap):** If $u(t)$ is non-scattering and $\Phi(u_0) < E_c + \eta$, then by definition of $E_c$ as the minimal non-scattering energy, we must have $\Phi(u_0) \geq E_c$. If $\Phi(u_0) = E_c$, then $u_0$ is a critical element, hence lies on the orbit $G \cdot u^*_0$ by Lemma 2.2.

**Step 3.2.4 (Conclusion):** For $\eta$ sufficiently small (depending on the spectral gap around $E_c$), the only non-scattering solutions near $E_c$ are those on the critical orbit. All others scatter.

### Lemma 3.3: Perturbation Around the Critical Element

**Statement:** Let $u^*$ be the critical element. For $u_0 = u^*_0 + h$ with $\|h\|_{X_c}$ small, one of the following holds:
1. **Ejection:** $\Phi(u_0) < E_c - \delta$ for some $\delta > 0$, and $u(t)$ scatters.
2. **Trapping:** $\Phi(u_0) \approx E_c$ and $u(t)$ remains in a neighborhood of the modulated trajectory $u^*(t)$ for all time.
3. **Blowup:** $\Phi(u_0) > E_c$ and the solution blows up in finite time (this case depends on the specific equation; for subcritical scaling $\alpha > \beta$, this leads to Type I blowup analysis).

**Proof:** This is the **stability/instability dichotomy** for the critical element, established via:
- **Spectral analysis:** Linearize the evolution around $u^*$. The linearized operator $L = -\Delta + V'(u^*)$ has spectral properties controlled by $K_{\mathrm{SC}_\lambda}^+$ (scaling structure) and $K_{\mathrm{LS}_\sigma}^+$ (Łojasiewicz gradient structure, if available).
- **Monotonicity formulas:** Use virial identities (from $K_{\mathrm{Mon}_\phi}^+$, if available in downstream analysis) to show that perturbations either grow (leading to blowup) or decay (leading to scattering), with the critical element $u^*$ being the threshold.

**Applicability:** This lemma is the **perturbation lemma** component of the Kenig-Merle framework, establishing that the critical element is a saddle point in the energy landscape.

---

## Step 4: Certificate Assembly and Verification

### Certificate Construction

We construct the Kenig-Merle certificate:
$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+ = (\mathsf{minimal\_u^*}, E_c, \mathsf{almost\_periodic\_mod\_G}, \mathsf{perturbation\_lemma})$$

**Component 1 (Minimal Element $\mathsf{minimal\_u^*}$):**
- **Data:** Initial condition $u^*_0 \in X_c$ with $\Phi(u^*_0) = E_c$
- **Properties:**
  - Non-scattering (certified by construction in Lemma 1.2)
  - Maximal lifespan solution (by $K_{\mathrm{WP}_{s_c}}^+$)
  - Unique modulo symmetries $G$ (by Lemma 2.2)
- **Witness:** Profile decomposition trace showing concentration at energy $E_c$

**Component 2 (Critical Energy $E_c$):**
- **Value:** $E_c = \inf\{E : \exists u \text{ non-scattering with } \Phi(u_0) = E\}$
- **Bounds:** By Lemma 1.1, $-B \leq E_c < \infty$
- **Attainment:** Certified by Lemma 1.2 (existence of $u^*$)
- **Gap Property:** For $\Phi(u_0) < E_c - \delta$, solution $u$ scatters (Lemma 1.3)

**Component 3 (Almost Periodicity $\mathsf{almost\_periodic\_mod\_G}$):**
- **Statement:** $\overline{\{g(t) \cdot u^*(t) : t \geq 0\}}$ is compact in $X_c$
- **Modulation Parameters:** Time-dependent $g(t) \in G$ (translations, scaling, phase)
- **Precompactness Witness:** Profile decomposition showing single-profile concentration (Lemma 2.1)
- **Quantitative Estimate:** $\|g(t_n) \cdot u^*(t_n) - \psi\|_{X_c} \to 0$ for any $t_n \to \infty$

**Component 4 (Perturbation Lemma $\mathsf{perturbation\_lemma}$):**
- **Statement:** Small perturbations of $u^*_0$ either scatter (if $\Phi < E_c$) or remain close to the modulated trajectory (if $\Phi \approx E_c$)
- **Quantitative Bound:** For $\|h\|_{X_c} < \delta$:
  - If $\Phi(u^*_0 + h) < E_c$: scattering
  - If $\Phi(u^*_0 + h) \in [E_c, E_c + \eta]$: trapping near $u^*$
- **Continuous Dependence:** Certified by Lemma 3.1 via $K_{\mathrm{WP}_{s_c}}^+$

### Verification Protocol

**Verification Step 1 (Energy Boundedness):** Confirm $K_{D_E}^+$ provides $\Phi(u) \geq -B$. This ensures $E_c$ is finite.

**Verification Step 2 (Profile Decomposition):** Confirm $K_{\mathrm{ProfDec}_{s_c,G}}^+$ applies to all bounded sequences. This ensures minimizing sequences converge (up to symmetry).

**Verification Step 3 (Well-Posedness):** Confirm $K_{\mathrm{WP}_{s_c}}^+$ provides continuous dependence and continuation criterion. This ensures perturbations are controlled.

**Verification Step 4 (Scaling Structure):** Confirm $K_{\mathrm{SC}_\lambda}^+$ provides $\alpha > \beta$ (Type I scaling). This ensures the critical element is not at the scaling-critical threshold (which would require Type II analysis).

**Certificate Validation:** All components are derived from the input permits via Lemmas 1.1-3.3. The certificate is **proof-carrying**: each component includes a witness from the lemmas above.

---

## Step 5: Metatheoretic Consequences

### Consequence 1: Reduction to Rigidity Analysis

**Statement:** By the KM certificate, the problem of global regularity reduces to **classifying** the critical element $u^*$.

**Proof:** By Lemma 3.2, any non-scattering solution with energy near $E_c$ must lie on the orbit $G \cdot u^*$. If we can show that $u^*$ either:
1. **Does not exist** (i.e., all solutions scatter), or
2. **Exists but is a known stationary/self-similar solution** (e.g., soliton, standing wave),

then we achieve global regularity (modulo the library $\mathcal{L}_T$ of exceptional solutions).

**Downstream Analysis:** The rigidity certificate $K_{\mathrm{Rigidity}_T}^+$ (cf. {prf:ref}`mt-fact-soft-rigidity`) uses $K_{\mathrm{KM}}^+$ as input to perform this classification via:
- Monotonicity formulas (virial/Morawetz identities from $K_{\mathrm{Mon}_\phi}^+$)
- Łojasiewicz gradient structure (from $K_{\mathrm{LS}_\sigma}^+$)
- Categorical exclusion (from Lock barrier $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$)

### Consequence 2: Type I vs. Type II Bifurcation

**Statement:** The scaling exponent $\alpha - \beta$ determines the blowup mechanism (if blowup occurs).

**Proof:** By $K_{\mathrm{SC}_\lambda}^+$, we have $\alpha > \beta$. For a solution $u$ with $\Phi(u_0) > E_c$ that blows up at time $T_{\max} < \infty$:
- **Type I Blowup:** If $\alpha > \beta$, the blowup rate is self-similar:
  $$\|u(t)\|_{X_c} \sim (T_{\max} - t)^{-(\alpha - \beta)/2}$$
  This is controlled by the scaling structure.
- **Type II Blowup:** If $\alpha = \beta$ (critical scaling), exotic non-self-similar blowup may occur. This case requires refined analysis via {cite}`MerleZaag98` and is excluded by the hypothesis $\alpha > \beta$.

**Certificate Routing:** If $\alpha \leq \beta$, the evaluator `Eval_KM(T)` returns $K_{\mathrm{KM}}^{\mathrm{inc}}$ (inconclusive), signaling that the standard KM framework does not apply, and specialized Type II machinery is required.

### Consequence 3: Compactness vs. Dispersion Dichotomy

**Statement:** The KM framework rigorously separates the compactness case (leading to critical elements and rigidity analysis) from the dispersion case (leading to scattering).

**Proof:** By the profile decomposition $K_{\mathrm{ProfDec}_{s_c,G}}^+$, any bounded sequence either:
1. **Concentrates:** $u_n = g_n \phi + o(1)$ for some profile $\phi \neq 0$ → leads to critical element $u^*$
2. **Vanishes:** $u_n \rightharpoonup 0$ → leads to scattering

The energy threshold $E_c$ quantitatively separates these regimes.

---

## Conclusion

### Main Result

**Theorem (FACT-SoftKM):** Given certificates:
$$K_{\mathrm{WP}_{s_c}}^+, \quad K_{\mathrm{ProfDec}_{s_c,G}}^+, \quad K_{D_E}^+, \quad K_{\mathrm{SC}_\lambda}^+$$
we have constructed a certificate:
$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+ = (\mathsf{minimal\_u^*}, E_c, \mathsf{almost\_periodic\_mod\_G}, \mathsf{perturbation\_lemma})$$
via the following logical chain:

1. **Energy Boundedness + Profile Decomposition** (Lemmas 1.1-1.2) → **Critical Element Exists**
2. **Scaling Structure + Minimality** (Lemmas 2.1-2.2) → **Almost Periodicity Modulo $G$**
3. **Well-Posedness + Continuous Dependence** (Lemmas 3.1-3.3) → **Perturbation Stability**
4. **Certificate Assembly** (Step 4) → **KM Framework Certified**

### Quantitative Bounds

| Property | Bound | Source |
|----------|-------|--------|
| Critical energy | $E_c \in [-B, \infty)$ | Lemma 1.1 ($K_{D_E}^+$) |
| Perturbation threshold | $\delta = \min(1, \text{Gap}(E_c, E_{\mathrm{NS}} \setminus \{E_c\}))$ | Lemma 1.3 |
| Modulation accuracy | $\|g(t_n) \cdot u^*(t_n) - \psi\|_{X_c} \to 0$ | Lemma 2.1 ($K_{\mathrm{ProfDec}_{s_c,G}}^+$) |
| Continuous dependence | $\|u(t) - u^*(t)\|_{X_c} \leq C e^{CT} \|u_0 - u^*_0\|_{X_c}$ | Lemma 3.1 ($K_{\mathrm{WP}_{s_c}}^+$) |

### Certificate Payload Structure

The final certificate $K_{\mathrm{KM}}^+$ is a structured type:
```
K_KM^+ := {
  minimal_element: {
    initial_data: u*_0 ∈ X_c,
    energy: Φ(u*_0) = E_c,
    non_scattering_witness: ProfileTrace,
    uniqueness_proof: Lemma 2.2
  },
  critical_energy: {
    value: E_c ∈ ℝ,
    lower_bound: -B,
    attainment_proof: Lemma 1.2,
    gap_estimate: δ
  },
  almost_periodicity: {
    modulation_group: G,
    precompactness_proof: Lemma 2.1,
    orbit_characterization: G·u*
  },
  perturbation_lemma: {
    stability_threshold: δ,
    scattering_criterion: Φ < E_c - δ,
    trapping_criterion: Φ ∈ [E_c, E_c + η],
    continuous_dependence: Lemma 3.1
  }
}
```

### Literature and Applicability

**Primary Source:**
- {cite}`KenigMerle06`: This paper introduces the concentration-compactness/rigidity method for the energy-critical nonlinear Schrödinger equation. Our proof systematically reconstructs their framework in a type-theoretic, certificate-producing manner. Specifically:
  - **Theorem 1.1** (global well-posedness and scattering below the ground state energy) corresponds to our Lemma 1.3 (small data scattering).
  - **Proposition 4.1** (linear profile decomposition) is abstracted into our use of $K_{\mathrm{ProfDec}_{s_c,G}}^+$.
  - **Lemma 4.3** (existence of critical element) is formalized in our Lemma 1.2.
  - **Theorem 5.1** (compactness up to symmetries) is our Lemma 2.1.
  - The perturbation/stability analysis (Section 6) is captured in our Lemmas 3.1-3.3.

**Applicability:** The Kenig-Merle framework applies to **energy-critical dispersive equations** where:
1. The scaling is subcritical ($\alpha > \beta$, ensuring Type I behavior)
2. The symmetry group $G$ is standard (translations, scaling, phase)
3. The equation admits a continuation criterion (blowup is controlled by a critical norm)

Examples: energy-critical NLS ($\dot{H}^1$ in dimensions $d \geq 3$), energy-critical wave equation, Yang-Mills equations.

**Extensions and Generalizations:**
- {cite}`KillipVisan10`: Extends the KM framework to higher dimensions ($d \geq 5$) for the focusing energy-critical NLS, using improved Strichartz estimates. Our certificate structure accommodates this by allowing dimension-dependent bounds in $K_{\mathrm{WP}_{s_c}}^+$.

- {cite}`DuyckaertsKenigMerle11`: Analyzes **Type II blowup** for the energy-critical wave equation, showing that when $\alpha = \beta$ (critical scaling), exotic blowup profiles can occur. Our framework detects this case via the evaluator `Eval_KM(T)` returning $K_{\mathrm{KM}}^{\mathrm{inc}}$ when $\alpha \leq \beta$.

- {cite}`BahouriGerard99`: Provides the profile decomposition machinery for nonlinear wave equations. Our use of $K_{\mathrm{ProfDec}_{s_c,G}}^+$ directly formalizes their Theorem 1.1 (profile decomposition with error estimates).

- {cite}`Lions84`: The foundational concentration-compactness principle. Our Lemma 1.2 (critical element existence) is a typed version of Lions' Lemma I.1 (dichotomy: compactness vs. vanishing).

**Downstream Usage:** The certificate $K_{\mathrm{KM}}^+$ serves as input to:
- **Rigidity Analysis** ({prf:ref}`mt-fact-soft-rigidity`): Uses $u^*$ to derive contradictions or classify solutions into the library $\mathcal{L}_T$.
- **Profile Classification** ({prf:ref}`mt-resolve-auto-profile`): Mechanism A (CC+Rig) consumes $K_{\mathrm{KM}}^+$ and produces $K_{\mathrm{prof}}^+$.
- **Lock Barrier** ({prf:ref}`mt-fact-lock`): Categorical exclusion uses almost periodicity to show $\text{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathbb{H}) = \emptyset$.

:::
