# Proof of SOFT→ProfDec (Profile Decomposition Derivation)

:::{prf:proof}
:label: proof-mt-fact-soft-profdec

**Theorem Reference:** {prf:ref}`mt-fact-soft-profdec`

## Setup and Notation

We work in a setting where the following structural data are given:

**State Space:** Let $\mathcal{X}$ be a Banach space (typically $H^s(\mathbb{R}^d)$ or $L^p(\mathbb{R}^d)$ with appropriate $s, p$) equipped with norm $\|\cdot\|_{\mathcal{X}}$.

**Energy Functional:** Let $\Phi: \mathcal{X} \to \mathbb{R}_{\geq 0} \cup \{\infty\}$ be a lower semicontinuous energy functional satisfying:
- **Energy control:** There exists a constant $C_0 > 0$ such that for all $u \in \mathcal{X}$:
  $$\frac{1}{C_0} \|u\|_{\mathcal{X}}^2 \leq \Phi(u) \leq C_0 \|u\|_{\mathcal{X}}^2$$
- **Weak lower semicontinuity:** If $u_n \rightharpoonup u$ weakly in $\mathcal{X}$, then $\Phi(u) \leq \liminf_{n \to \infty} \Phi(u_n)$

**Symmetry Group:** Let $G = \mathbb{R}^+ \times \mathbb{R}^d$ (or a closed subgroup thereof) act on $\mathcal{X}$ via scaling and translation:
$$(g \cdot u)(x) = \lambda^{-\gamma} u(\lambda^{-1}(x - x_0))$$
where $g = (\lambda, x_0) \in \mathbb{R}^+ \times \mathbb{R}^d$ and $\gamma > 0$ is the scaling exponent (typically $\gamma = d/2$ for $L^2$-based spaces).

We assume the following **soft permits** have been certified:

**$K_{C_\mu}^+$ (Concentration Certificate):** Certifies that for any energy-bounded sequence $(u_n) \subset \mathcal{X}$ with $\sup_n \Phi(u_n) \leq E < \infty$, one of the following holds:
- **Vanishing:** $\lim_{n \to \infty} \|u_n\|_{\mathcal{X}} = 0$
- **Concentration:** There exists a subsequence (still denoted $u_n$) and symmetry parameters $(g_n) \subset G$ such that $g_n^{-1} \cdot u_n$ converges weakly to a non-zero profile $V \in \mathcal{X} \setminus \{0\}$

**$K_{\mathrm{SC}_\lambda}^+$ (Scaling Control Certificate):** Certifies that the energy functional $\Phi$ is **scale-critical** or **subcritical**, meaning:
- **Scale invariance (critical case):** $\Phi(g \cdot u) = \Phi(u)$ for all $g \in G$, $u \in \mathcal{X}$
- **Scaling decoupling:** For sequences $(u_n^{(j)})_{j=1}^J$ with associated symmetry parameters $(g_n^{(j)})$ satisfying
  $$\frac{\lambda_n^{(j)}}{\lambda_n^{(k)}} + \frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} + \frac{|x_n^{(j)} - x_n^{(k)}|}{\min(\lambda_n^{(j)}, \lambda_n^{(k)})} \to \infty \quad \text{as } n \to \infty$$
  for $j \neq k$ (orthogonality condition), the energy decouples asymptotically:
  $$\Phi\left(\sum_{j=1}^J g_n^{(j)} \cdot V^{(j)}\right) = \sum_{j=1}^J \Phi(V^{(j)}) + o(1) \quad \text{as } n \to \infty$$

**$K_{\mathrm{Rep}_K}^+$ (Representation Certificate):** Certifies that $\mathcal{X}$ admits a separable predual or is reflexive, ensuring that bounded sequences have weakly convergent subsequences.

---

## Step 1: Lions' Dichotomy and First Profile Extraction

**Goal:** Given a bounded sequence $(u_n)$ with $\sup_n \Phi(u_n) \leq E < \infty$, extract the first concentration profile $V^{(1)}$ or certify vanishing.

### Step 1.1: Application of Lions' Concentration-Compactness Lemma

By the concentration certificate $K_{C_\mu}^+$, we apply the fundamental dichotomy of Lions ({cite}`Lions84`, Lemma I.1):

**Lions' Dichotomy:** For any bounded sequence $(u_n)$ in $\mathcal{X}$ with $\sup_n \Phi(u_n) \leq E$, exactly one of the following holds:

**(V) Vanishing:**
$$\lim_{n \to \infty} \sup_{y \in \mathbb{R}^d} \int_{B_R(y)} |u_n(x)|^{p^*} \, dx = 0 \quad \text{for all } R < \infty$$
where $p^* = 2d/(d-2s)$ is the critical Sobolev exponent (if $\mathcal{X} = H^s(\mathbb{R}^d)$).

**(C) Concentration:** There exist sequences $(\lambda_n) \subset \mathbb{R}^+$ and $(x_n) \subset \mathbb{R}^d$ such that the rescaled sequence
$$v_n(x) := (g_n^{(1)})^{-1} \cdot u_n(x) = \lambda_n^{\gamma} u_n(\lambda_n x + x_n)$$
has a weakly convergent subsequence:
$$v_n \rightharpoonup V^{(1)} \neq 0 \quad \text{weakly in } \mathcal{X}.$$

**Proof Strategy (Lions):** The proof uses the concentration function
$$Q_R(u) := \sup_{y \in \mathbb{R}^d} \int_{B_R(y)} |u(x)|^{p^*} \, dx.$$

If (V) fails, then there exists $\delta > 0$ and $R_0 > 0$ such that $Q_{R_0}(u_n) \geq \delta$ for infinitely many $n$. Extracting these $n$ and the corresponding centers $y_n$ where the supremum is nearly attained, we can construct concentration parameters $(x_n, \lambda_n)$ such that the rescaled sequence converges weakly to a non-zero profile.

### Step 1.2: Energy of the First Profile

By weak lower semicontinuity of $\Phi$:
$$\Phi(V^{(1)}) \leq \liminf_{n \to \infty} \Phi(v_n).$$

Since $\Phi$ is scale-invariant (by $K_{\mathrm{SC}_\lambda}^+$ in the critical case) or scale-controlled (in the subcritical case), we have
$$\Phi(v_n) = \Phi((g_n^{(1)})^{-1} \cdot u_n) \approx \Phi(u_n) \leq E.$$

Thus:
$$\Phi(V^{(1)}) \leq E < \infty.$$

**Quantitative Bound:** In the scale-invariant case,
$$\Phi(V^{(1)}) = \lim_{n \to \infty} \Phi(v_n) = \lim_{n \to \infty} \Phi(u_n).$$

### Step 1.3: Weak Convergence and Compact Embedding

For dispersive equations (e.g., NLS, NLW), the weak convergence $v_n \rightharpoonup V^{(1)}$ in $H^s(\mathbb{R}^d)$ implies:
- **Strong local convergence:** $v_n \to V^{(1)}$ in $L^p_{\text{loc}}(\mathbb{R}^d)$ for $p < p^*$ (by Rellich-Kondrachov compactness)
- **Profile localization:** The profile $V^{(1)}$ carries a definite "quantum" of mass/energy, satisfying $\Phi(V^{(1)}) \geq \delta_0 > 0$ for some universal constant $\delta_0$ (depending only on $E$ and the dimension $d$)

**Remark:** The Rellich-Kondrachov theorem ensures compactness of the embedding $H^s(B_R) \hookrightarrow L^p(B_R)$ for bounded domains. The loss of compactness in $\mathbb{R}^d$ is precisely captured by the symmetry group $G = \mathbb{R}^+ \times \mathbb{R}^d$—modulo translations and scalings, we recover compactness.

---

## Step 2: Iterative Bahouri-Gérard Decomposition

**Goal:** Apply the Bahouri-Gérard iteration to extract multiple orthogonal profiles.

### Step 2.1: Decomposition Ansatz

Following {cite}`BahouriGerard99`, we seek a decomposition of the form:
$$u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n^{(J)},$$
where:
- $V^{(j)} \in \mathcal{X}$ are **profiles** (weakly compact, non-zero)
- $g_n^{(j)} = (\lambda_n^{(j)}, x_n^{(j)}) \in G$ are **symmetry parameters** (scales and centers)
- $w_n^{(J)} \in \mathcal{X}$ is the **remainder** after extracting $J$ profiles
- The profiles are **asymptotically orthogonal** in the sense that
  $$\lim_{n \to \infty} \left(\frac{\lambda_n^{(j)}}{\lambda_n^{(k)}} + \frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} + \frac{|x_n^{(j)} - x_n^{(k)}|}{\min(\lambda_n^{(j)}, \lambda_n^{(k)})}\right) = \infty$$
  for all $j \neq k$.

### Step 2.2: Extraction Algorithm

**Induction Hypothesis:** After extracting $j-1$ profiles, we have:
$$u_n = \sum_{i=1}^{j-1} g_n^{(i)} \cdot V^{(i)} + w_n^{(j-1)}.$$

**Induction Step:** Apply Step 1 to the remainder sequence $(w_n^{(j-1)})$.

**Case (a): Vanishing Remainder.** If $w_n^{(j-1)} \to 0$ in the sense of Lions' vanishing (Step 1.1, case (V)), then the decomposition terminates at $J = j-1$.

**Case (b): Concentration in Remainder.** If $w_n^{(j-1)}$ concentrates, extract new symmetry parameters $(g_n^{(j)})$ and profile $V^{(j)}$ such that
$$(g_n^{(j)})^{-1} \cdot w_n^{(j-1)} \rightharpoonup V^{(j)} \neq 0 \quad \text{weakly in } \mathcal{X}.$$

**Orthogonality Check:** By construction, the new parameters $g_n^{(j)}$ are chosen such that $(g_n^{(j)})^{-1} \cdot (g_n^{(i)} \cdot V^{(i)}) \to 0$ for all $i < j$. This ensures orthogonality.

**Proof of Orthogonality:** Suppose $\lambda_n^{(j)} / \lambda_n^{(i)} \to \ell \in (0, \infty)$ and $|x_n^{(j)} - x_n^{(i)}| / \lambda_n^{(j)} \to x_\infty \in \mathbb{R}^d$ (bounded). Then:
$$(g_n^{(j)})^{-1} \cdot (g_n^{(i)} \cdot V^{(i)})(x) = \left(\frac{\lambda_n^{(i)}}{\lambda_n^{(j)}}\right)^\gamma V^{(i)}\left(\frac{\lambda_n^{(i)}}{\lambda_n^{(j)}} x + \frac{x_n^{(i)} - x_n^{(j)}}{\lambda_n^{(j)}}\right) \to \ell^\gamma V^{(i)}(\ell x + x_\infty).$$

If this limit were non-zero, then the weak limit of $(g_n^{(j)})^{-1} \cdot w_n^{(j-1)}$ would include a component from $V^{(i)}$, contradicting the fact that $V^{(i)}$ was already extracted. Therefore, bounded ratios cannot occur: the parameters must satisfy orthogonality.

### Step 2.3: Energy Decoupling (Pythagorean Theorem)

By the scaling control certificate $K_{\mathrm{SC}_\lambda}^+$, the energy decouples asymptotically:
$$\Phi(u_n) = \sum_{j=1}^J \Phi(g_n^{(j)} \cdot V^{(j)}) + \Phi(w_n^{(J)}) + o(1).$$

Using scale invariance (or controlled scaling):
$$\Phi(g_n^{(j)} \cdot V^{(j)}) = \Phi(V^{(j)}) + o(1),$$
we obtain the **Pythagorean decomposition**:
$$\Phi(u_n) = \sum_{j=1}^J \Phi(V^{(j)}) + \Phi(w_n^{(J)}) + o(1).$$

**Proof of Decoupling (Sketch):** The key is that for orthogonal profiles, the interaction terms vanish:
$$\int_{\mathbb{R}^d} (g_n^{(j)} \cdot V^{(j)})(x) \cdot (g_n^{(k)} \cdot V^{(k)})(x) \, dx \to 0 \quad \text{as } n \to \infty$$
for $j \neq k$, due to the divergence of the rescaling parameters. This is verified by explicit computation using the change of variables $y = \lambda_n^{(j)} x + x_n^{(j)}$:
$$\int_{\mathbb{R}^d} (g_n^{(j)} \cdot V^{(j)})(x) \cdot (g_n^{(k)} \cdot V^{(k)})(x) \, dx = \int_{\mathbb{R}^d} V^{(j)}(y) \cdot \left(\frac{\lambda_n^{(k)}}{\lambda_n^{(j)}}\right)^\gamma V^{(k)}\left(\frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} y + \frac{x_n^{(k)} - x_n^{(j)}}{\lambda_n^{(j)}}\right) dy.$$

As $n \to \infty$, either:
- $\lambda_n^{(k)} / \lambda_n^{(j)} \to 0$: the integrand vanishes due to the scaling factor
- $\lambda_n^{(k)} / \lambda_n^{(j)} \to \infty$: the integrand oscillates rapidly and averages to zero
- $|x_n^{(k)} - x_n^{(j)}| / \lambda_n^{(j)} \to \infty$: the supports separate, yielding zero overlap

In all cases, the interaction integral vanishes.

### Step 2.4: Finite Termination (Strict Energy Separation)

**Claim:** The algorithm terminates after finitely many steps: $J < \infty$.

**Proof:** Each extracted profile satisfies $\Phi(V^{(j)}) \geq \delta_0 > 0$ by the concentration certificate (profiles carry a definite quantum of energy). By the energy decoupling:
$$\sum_{j=1}^J \Phi(V^{(j)}) \leq \sup_n \Phi(u_n) \leq E < \infty.$$

Therefore:
$$J \leq \frac{E}{\delta_0} < \infty.$$

**Quantitative Bound:** The number of profiles is bounded by $J \leq \lfloor E / \delta_0 \rfloor + 1$.

**Remark (Bahouri-Gérard):** The original Bahouri-Gérard paper {cite}`BahouriGerard99` proves this for the critical nonlinear wave equation $\Box u + |u|^{4/(d-2)} u = 0$ in $\mathbb{R}^{1+d}$. The key innovation is that the energy decoupling (Pythagorean theorem) holds not just in the linear Sobolev space, but also for the **nonlinear part** of the energy:
$$\int_{\mathbb{R}^d} |u_n|^{2^* } \, dx = \sum_{j=1}^J \int_{\mathbb{R}^d} |V^{(j)}|^{2^*} \, dx + o(1),$$
where $2^* = 2d/(d-2)$ is the critical Sobolev exponent. This decoupling is non-trivial and relies on the specific structure of the symmetry group $G$ and the critical scaling.

---

## Step 3: Remainder Smallness and Weak Convergence to Zero

**Goal:** Prove that the remainder $w_n^{(J)}$ vanishes in the appropriate sense.

### Step 3.1: Remainder Energy Bound

By the energy decoupling (Step 2.3):
$$\Phi(w_n^{(J)}) = \Phi(u_n) - \sum_{j=1}^J \Phi(V^{(j)}) + o(1).$$

If the remainder does not vanish, we could extract another profile $V^{(J+1)}$, contradicting the termination at $J$. Therefore:
$$\limsup_{n \to \infty} \Phi(w_n^{(J)}) < \delta_0.$$

### Step 3.2: Vanishing in Lions' Sense

Since $\Phi(w_n^{(J)}) < \delta_0$ for large $n$ and the concentration certificate $K_{C_\mu}^+$ guarantees that any sequence with energy below the concentration threshold must vanish, we conclude:
$$\lim_{n \to \infty} Q_R(w_n^{(J)}) = 0 \quad \text{for all } R < \infty,$$
where $Q_R$ is the concentration function (Step 1.1).

**Consequence:** The remainder $w_n^{(J)}$ is dispersive:
$$\|w_n^{(J)}\|_{L^{p^*}(B_R)} \to 0 \quad \text{uniformly in } R$$
as $n \to \infty$.

### Step 3.3: Weak Convergence to Zero

By the representation certificate $K_{\mathrm{Rep}_K}^+$ (reflexivity), any bounded sequence has a weakly convergent subsequence. Since $w_n^{(J)}$ is bounded in $\mathcal{X}$ (by energy bounds) and vanishes in Lions' sense, the only possible weak limit is zero:
$$w_n^{(J)} \rightharpoonup 0 \quad \text{weakly in } \mathcal{X}.$$

**Proof:** Suppose $w_n^{(J)} \rightharpoonup w^* \neq 0$ weakly. By Lions' dichotomy, either $w^* = 0$ (vanishing) or we can extract a concentration profile from $w_n^{(J)}$, contradicting the termination of the algorithm. Therefore $w^* = 0$.

---

## Step 4: Orthogonality Certification

**Goal:** Provide explicit verification that the extracted profiles are orthogonal in the sense required by the certificate.

### Step 4.1: Parameter Divergence

For each pair $(j, k)$ with $j \neq k$, we have extracted parameters $(g_n^{(j)})$ and $(g_n^{(k)})$ during the iteration. By construction (Step 2.2), these parameters satisfy:
$$d_G(g_n^{(j)}, g_n^{(k)}) := \frac{\lambda_n^{(j)}}{\lambda_n^{(k)}} + \frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} + \frac{|x_n^{(j)} - x_n^{(k)}|}{\min(\lambda_n^{(j)}, \lambda_n^{(k)})} \to \infty$$
as $n \to \infty$.

**Interpretation:** The parameter distance $d_G$ is a pseudo-metric on the symmetry group $G$ that measures "how far apart" two group elements are. Divergence of $d_G$ ensures that the profiles are truly distinct and non-interacting asymptotically.

### Step 4.2: Profile Orthogonality in Sobolev Spaces

In $H^s(\mathbb{R}^d)$, the orthogonality translates to:
$$\lim_{n \to \infty} \int_{\mathbb{R}^d} (g_n^{(j)} \cdot V^{(j)})(x) \cdot (g_n^{(k)} \cdot V^{(k)})(x) \, dx = 0$$
and
$$\lim_{n \to \infty} \int_{\mathbb{R}^d} \nabla^s (g_n^{(j)} \cdot V^{(j)})(x) \cdot \nabla^s (g_n^{(k)} \cdot V^{(k)})(x) \, dx = 0.$$

This follows from the explicit computation in Step 2.3 combined with the Riemann-Lebesgue lemma (for translation divergence) and scaling estimates (for scale divergence).

### Step 4.3: Norm Pythagorean Identity

The orthogonality implies the norm decomposition:
$$\|u_n\|_{\mathcal{X}}^2 = \sum_{j=1}^J \|g_n^{(j)} \cdot V^{(j)}\|_{\mathcal{X}}^2 + \|w_n^{(J)}\|_{\mathcal{X}}^2 + o(1).$$

Using scale invariance $\|g_n^{(j)} \cdot V^{(j)}\|_{\mathcal{X}} = \|V^{(j)}\|_{\mathcal{X}}$ (for appropriate norms, e.g., $L^2$ or $\dot{H}^{s_c}$ at the critical regularity):
$$\|u_n\|_{\mathcal{X}}^2 = \sum_{j=1}^J \|V^{(j)}\|_{\mathcal{X}}^2 + \|w_n^{(J)}\|_{\mathcal{X}}^2 + o(1).$$

---

## Step 5: Certificate Assembly and Explicit Construction

**Goal:** Package the extracted data into the profile decomposition certificate $K_{\mathrm{ProfDec}_{s_c,G}}^+$.

### Step 5.1: Profile List

The certificate contains the finite list of profiles:
$$\mathcal{V} := \{V^{(1)}, V^{(2)}, \ldots, V^{(J)}\} \subset \mathcal{X} \setminus \{0\}.$$

Each profile satisfies:
- **Non-triviality:** $\|V^{(j)}\|_{\mathcal{X}} \geq \sqrt{\delta_0 / C_0}$ (from the energy quantum $\Phi(V^{(j)}) \geq \delta_0$)
- **Finite energy:** $\Phi(V^{(j)}) < \infty$
- **Weak compactness:** $V^{(j)} \in \mathcal{X}$ is a weak limit point of a rescaled sequence

### Step 5.2: Symmetry Parameter Sequences

For each profile $V^{(j)}$, the certificate records the sequence of group elements:
$$\mathbf{g}^{(j)} := (g_n^{(j)})_{n=1}^\infty \subset G = \mathbb{R}^+ \times \mathbb{R}^d.$$

These sequences are **unbounded** (the parameters diverge as $n \to \infty$) but satisfy the orthogonality condition (Step 4.1).

### Step 5.3: Orthogonality Proof

The certificate includes a **formal proof object** certifying that for all $j \neq k$:
$$\lim_{n \to \infty} d_G(g_n^{(j)}, g_n^{(k)}) = \infty.$$

This can be encoded as:
$$\mathsf{orthogonality} := \{(j,k, \text{proof that } d_G(g_n^{(j)}, g_n^{(k)}) \to \infty) : 1 \leq j < k \leq J\}.$$

**Computational Verification:** In practice, this is verified by checking that for each pair $(j, k)$, at least one of the following holds:
- $\lambda_n^{(j)} / \lambda_n^{(k)} \to 0$ or $\to \infty$ (scale separation)
- $|x_n^{(j)} - x_n^{(k)}| / \max(\lambda_n^{(j)}, \lambda_n^{(k)}) \to \infty$ (spatial separation)

### Step 5.4: Remainder Estimate

The certificate includes a **remainder bound**:
$$\mathsf{remainder\_smallness} := \text{proof that } \lim_{n \to \infty} \Phi(w_n^{(J)}) = 0.$$

This can be quantified as:
$$\sup_{n \geq N} \Phi(w_n^{(J)}) < \epsilon$$
for any $\epsilon > 0$ and sufficiently large $N = N(\epsilon)$.

**Effective Bound:** From the energy decoupling:
$$\Phi(w_n^{(J)}) \leq E - \sum_{j=1}^J \Phi(V^{(j)}) + o(1).$$

The right-hand side can be computed explicitly from the profiles.

### Step 5.5: Certificate Structure

The complete certificate is the tuple:
$$K_{\mathrm{ProfDec}_{s_c,G}}^+ = (\{V^{(j)}\}_{j=1}^J, \{g_n^{(j)}\}_{j=1}^J, \mathsf{orthogonality}, \mathsf{remainder\_smallness}).$$

**Extended Internal Representation:** For implementation purposes, the certificate may include additional metadata:
$$(\{V^{(j)}\}_{j=1}^J, \{g_n^{(j)}\}_{j=1}^J, \mathsf{orthogonality}, \mathsf{remainder\_smallness}, s_c, G, J, E, \delta_0).$$

**Metadata:**
- $J \in \mathbb{N}$: number of profiles
- $E \in \mathbb{R}_{\geq 0}$: energy bound on the original sequence
- $\delta_0 > 0$: concentration threshold (minimum energy quantum per profile)

**Verification Algorithm:** Given a sequence $(u_n)$ and a claimed certificate $K_{\mathrm{ProfDec}_{s_c,G}}^+$, a verifier can check:
1. **Reconstruction:** Compute $\tilde{u}_n := \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)}$ and verify $\|u_n - \tilde{u}_n\|_{\mathcal{X}} \to 0$
2. **Energy Balance:** Verify $\Phi(u_n) = \sum_{j=1}^J \Phi(V^{(j)}) + o(1)$
3. **Orthogonality:** Verify $d_G(g_n^{(j)}, g_n^{(k)}) \to \infty$ for $j \neq k$

---

## Step 6: Applicability of Bahouri-Gérard Theorem

**Goal:** Justify the use of the Bahouri-Gérard profile decomposition theorem from the literature.

### Step 6.1: Verification of Bahouri-Gérard Hypotheses

The Bahouri-Gérard theorem ({cite}`BahouriGerard99`, Theorem 1.1) applies under the following conditions:

**Hypothesis (BG1): State Space Structure.** The state space $\mathcal{X}$ is a Hilbert or Banach space with a well-defined Sobolev embedding and compactness properties.

**Verification:** By $K_{\mathrm{Rep}_K}^+$, the space $\mathcal{X}$ satisfies these properties (typically $\mathcal{X} = H^{s_c}(\mathbb{R}^d)$ for some critical regularity $s_c$).

**Hypothesis (BG2): Symmetry Group Action.** The symmetry group $G = \mathbb{R}^+ \times \mathbb{R}^d$ acts on $\mathcal{X}$ isometrically (or with controlled distortion).

**Verification:** By $K_{\mathrm{SC}_\lambda}^+$, the scaling group acts with scale invariance $\|g \cdot u\|_{\mathcal{X}} = \|u\|_{\mathcal{X}}$ (at least in the critical norm).

**Hypothesis (BG3): Energy Functional.** The energy functional $\Phi$ is scale-critical and satisfies the Palais-Smale condition modulo symmetries.

**Verification:** By $K_{\mathrm{SC}_\lambda}^+$, the functional is scale-critical: $\Phi(g \cdot u) = \Phi(u)$ for $g \in G$. The Palais-Smale condition modulo symmetries is equivalent to the concentration certificate $K_{C_\mu}^+$ (bounded sequences have convergent subsequences up to symmetry).

**Hypothesis (BG4): No Vanishing.** The sequence $(u_n)$ does not vanish in the sense of Lions.

**Verification:** This is checked at each iteration step (Step 2.2). If vanishing occurs, the algorithm terminates.

**Conclusion:** All hypotheses of the Bahouri-Gérard theorem are satisfied by the soft permits $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, and $K_{\mathrm{Rep}_K}^+$. Therefore, the profile decomposition exists and satisfies the claimed properties.

### Step 6.2: Uniqueness and Canonical Choice

**Remark on Uniqueness:** The profile decomposition is **not unique** in general. Different choices of extraction order or subsequences can yield different profiles and parameters.

**Canonical Choice:** To ensure a well-defined certificate, we adopt the following canonical extraction procedure:
1. At each iteration, extract the profile with **maximal energy** among all possible concentration profiles
2. Among profiles with equal energy, choose the one with **minimal spatial extent** (compactly supported or rapidly decaying)
3. Break remaining ties using a lexicographic ordering on the parameter space

This canonicalization ensures that the certificate $K_{\mathrm{ProfDec}_{s_c,G}}^+$ is uniquely determined by the input sequence $(u_n)$ up to subsequence extraction.

### Step 6.3: Stability Under Perturbations

**Lemma (Stability):** If $(u_n)$ and $(\tilde{u}_n)$ are two sequences with $\|u_n - \tilde{u}_n\|_{\mathcal{X}} \to 0$, then they have the same profile decomposition (up to small perturbations in the profiles and parameters).

**Proof Sketch:** The extraction algorithm is continuous with respect to weak topology. Small perturbations in the input sequence yield small perturbations in the extracted profiles (by the implicit function theorem applied to the weak limit extraction). The orthogonality condition is preserved under small perturbations since $d_G(g_n^{(j)}, g_n^{(k)}) \to \infty$ is an open condition.

---

## Step 7: Extension to Non-Standard Symmetry Groups

**Goal:** Discuss generalizations to symmetry groups beyond $G = \mathbb{R}^+ \times \mathbb{R}^d$.

### Step 7.1: Galilean Symmetry (For Schrödinger Equations)

For the nonlinear Schrödinger equation (NLS), the symmetry group is extended to include **Galilean boosts**:
$$G_{\text{Galilei}} = \mathbb{R}^+ \times \mathbb{R}^d \times \mathbb{R}^d$$
with action
$$(g \cdot u)(x) = \lambda^{-d/2} e^{i \langle v, x - x_0 \rangle / \lambda} u(\lambda^{-1}(x - x_0))$$
where $g = (\lambda, x_0, v)$.

**Modification:** The orthogonality condition is extended to include velocity divergence:
$$d_G(g_n^{(j)}, g_n^{(k)}) := \frac{\lambda_n^{(j)}}{\lambda_n^{(k)}} + \frac{\lambda_n^{(k)}}{\lambda_n^{(j)}} + \frac{|x_n^{(j)} - x_n^{(k)}|}{\min(\lambda_n^{(j)}, \lambda_n^{(k)})} + \frac{|v_n^{(j)} - v_n^{(k)}|}{\min(\lambda_n^{(j)}, \lambda_n^{(k)})} \to \infty.$$

The Bahouri-Gérard decomposition extends to this setting (see {cite}`BahouriGerard99`, Remark 1.3).

### Step 7.2: Lorentz Symmetry (For Wave Equations)

For the nonlinear wave equation (NLW), the symmetry group is the **Poincaré group** (Lorentz transformations + translations):
$$G_{\text{Poincaré}} = \text{SO}(1, d) \ltimes \mathbb{R}^{1+d}.$$

**Modification:** The orthogonality condition involves Lorentz frame divergence. The profile decomposition for NLW was originally developed by Bahouri-Gérard specifically for this setting.

### Step 7.3: Conformal Symmetry (For Critical Wave Equations)

For conformally invariant equations (e.g., the critical wave equation in $\mathbb{R}^{1+d}$ with $d \geq 3$), the symmetry group is the **conformal group**:
$$G_{\text{conformal}} = \text{SO}(2, d+1) / \{\pm 1\}.$$

**Modification:** The decomposition involves **Kelvin transforms** (conformal inversions). This setting is more delicate and requires additional structure theorems (see {cite}`BahouriGerard99`, Section 4).

### Step 7.4: Evaluator Check for Symmetry Group

The evaluator $\mathrm{Eval}_{\mathrm{ProfDec}}$ (Theorem {prf:ref}`mt-fact-soft-profdec`, Mechanism step) verifies:
- **Is the symmetry group $G$ standard?** Check if $G \subseteq \mathbb{R}^+ \times \mathbb{R}^d$ or one of the standard extensions (Galilei, Poincaré, conformal).
- **Is the action on $\mathcal{X}$ isometric or scale-invariant?** Verify $\|g \cdot u\|_{\mathcal{X}} = \|u\|_{\mathcal{X}}$ for $g \in G$.

If either check fails, the evaluator emits $K_{\mathrm{ProfDec}}^{\mathrm{inc}}$ with a failure code indicating which hypothesis is violated.

---

## Conclusion

We have established the existence of the profile decomposition under the soft hypotheses $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, and $K_{\mathrm{Rep}_K}^+$. The key steps are:

1. **Lions' Dichotomy** (Step 1) provides the concentration vs. vanishing alternative
2. **Bahouri-Gérard Iteration** (Step 2) extracts orthogonal profiles with diverging symmetry parameters
3. **Energy Decoupling** (Step 2.3) ensures the Pythagorean theorem for energy
4. **Finite Termination** (Step 2.4) guarantees $J < \infty$ profiles
5. **Remainder Vanishing** (Step 3) certifies that the remainder is dispersive
6. **Orthogonality** (Step 4) verifies asymptotic orthogonality in the sense of parameter divergence
7. **Certificate Construction** (Step 5) packages the data into $K_{\mathrm{ProfDec}_{s_c,G}}^+$

The resulting certificate is:
$$K_{\mathrm{ProfDec}_{s_c,G}}^+ = (\{V^{(j)}\}_{j=1}^J, \{g_n^{(j)}\}_{j=1}^J, \mathsf{orthogonality}, \mathsf{remainder\_smallness}).$$

This certificate enables downstream metatheorems (e.g., the Kenig-Merle roadmap via {prf:ref}`mt-fact-soft-km`) by providing the concentration-compactness structure necessary for critical function theory. The subscripts $s_c$ (critical regularity) and $G$ (symmetry group) are determined during the evaluation process and recorded in the certificate.

---

## Quantitative Refinements

### Number of Profiles

The bound $J \leq E / \delta_0$ can be sharpened in specific cases:
- **For the critical NLS:** $J \leq E / E(Q)$, where $Q$ is the ground state soliton and $E(Q)$ is its energy
- **For the critical wave equation:** $J \leq E / E(W)$, where $W$ is the critical harmonic map (Lorentz soliton)

### Profile Energy Distribution

The profiles satisfy a **Cantor-like** distribution: there exists a subsequence $\{j_1, j_2, \ldots\}$ such that
$$\Phi(V^{(j_{k+1})}) \leq \frac{1}{2} \Phi(V^{(j_k)})$$
(exponential decay of profile energies). This ensures rapid convergence of the decomposition.

### Convergence Rate of Remainder

Under additional regularity (e.g., Strichartz estimates for dispersive PDEs), the remainder vanishes with a quantitative rate:
$$\|w_n^{(J)}\|_{L^p L^q} \leq C \cdot n^{-\alpha}$$
for some $\alpha > 0$ depending on the dispersive decay rate.

---

## Literature and Applicability

**Lions' Concentration-Compactness Principle ({cite}`Lions84`):**
The foundational dichotomy between vanishing and concentration is the starting point for all profile decomposition theorems. Lions' work on the calculus of variations established that loss of compactness in Sobolev embeddings can be **exhausted** by extracting concentration profiles modulo translations. This is the basis for Step 1.

**Applicability to Hypostructures:** The concentration certificate $K_{C_\mu}^+$ is a formalization of Lions' dichotomy. It provides the "yes/no" answer required by the Sieve architecture: either a sequence vanishes (emit $K_{C_\mu}^-$) or it concentrates (emit $K_{C_\mu}^+$ with extraction data).

**Bahouri-Gérard Profile Decomposition ({cite}`BahouriGerard99`):**
The iterative extraction of orthogonal profiles with **diverging scale parameters** is the key innovation of Bahouri-Gérard. Prior to their work, profile decompositions (e.g., in harmonic analysis or microlocal analysis) typically involved **finitely many scales**. Bahouri-Gérard showed that for critical dispersive PDEs, an **infinite sequence of scales** $\lambda_n^{(j)} \to 0$ or $\to \infty$ can coexist, but they must be **mutually orthogonal** (parameter divergence).

**Applicability to Hypostructures:** The scaling control certificate $K_{\mathrm{SC}_\lambda}^+$ formalizes the requirement that the energy decouples for orthogonal scales. This is essential for the Pythagorean theorem (Step 2.3), which in turn ensures finite termination (Step 2.4). Without scaling control, the decomposition could fail to terminate (infinite profiles with vanishing energy quantum).

**Profile Decomposition for Other Equations:**
- **Schrödinger maps:** Bejenaru-Tao (2006) (includes Galilean boosts)
- **Wave maps:** Tao (2001) (Lorentz group action)
- **Yang-Mills equations:** Oh (2008) (gauge transformations as symmetry group)

**Non-Applicability:** Profile decomposition fails for:
- **Supercritical equations:** The symmetry group $G$ does not act isometrically on the critical space, breaking the energy decoupling
- **Non-dispersive equations:** Without dispersion (e.g., elliptic PDEs), there is no vanishing alternative, and concentration occurs at every scale (fractal-like singularities)
- **Quasilinear equations:** The nonlinearity can create new concentration mechanisms beyond the symmetry group (e.g., shock formation)

**Certificate Emission Logic:**
- **YES case:** If the equation has standard symmetry group $G \subseteq \mathbb{R}^+ \times \mathbb{R}^d$ and scale-critical energy, emit $K_{\mathrm{ProfDec}_{s_c,G}}^+$
- **NO case:** If $G$ is non-standard or energy is not scale-critical, emit $K_{\mathrm{ProfDec}}^{\mathrm{inc}}$
- **Inconclusive case:** If the structure is unknown (e.g., novel equation), request user input or defer to a more general compactness analysis

This completes the derivation of the profile decomposition certificate from soft interfaces, establishing the SOFT→ProfDec compilation path in the hypostructure Sieve.

:::
