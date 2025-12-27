# UP-Spectral: Spectral Gap Promotion — GMT Translation

## Original Statement (Hypostructure)

The spectral gap promotion upgrades local spectral bounds to global spectral control, ensuring uniform exponential decay of perturbations.

## GMT Setting

**Spectral Gap:** $\lambda_1 > 0$ — first non-zero eigenvalue of linearized operator

**Exponential Decay:** $\|T_t - T_*\| \leq C e^{-\lambda_1 t}$

**Promotion:** Local spectral gaps imply global

## GMT Statement

**Theorem (Spectral Gap Promotion).** If the linearized operator $L = -\nabla^2 \Phi|_{T_*}$ at equilibrium $T_*$ satisfies:

1. **Local Gap:** $\lambda_1(L|_{B_r(x)}) \geq \lambda_0 > 0$ for all $(x, r)$

2. **Uniformity:** Gap is uniform across locations

3. **Compactness:** Domain is compact or has controlled geometry

Then:
- **Global Gap:** $\lambda_1(L) \geq c \lambda_0$
- **Exponential Decay:** $\|T_t - T_*\| \leq C e^{-\lambda_1 t/2} \|T_0 - T_*\|$

## Proof Sketch

### Step 1: Linearized Operator

**Second Variation:** At critical point $T_*$:
$$L := \nabla^2 \Phi|_{T_*} = -\Delta + V$$

where $V$ is the potential from curvature of $\Phi$.

**Eigenvalue Problem:**
$$L \phi = \lambda \phi, \quad \phi|_{\partial M} = 0 \text{ (or other BC)}$$

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU. [Chapter 6]

### Step 2: Rayleigh Quotient

**Rayleigh Characterization:**
$$\lambda_1 = \inf_{u \perp \text{ker}(L)} \frac{\langle L u, u \rangle}{\|u\|^2}$$

**Poincaré Inequality:** For $u \perp 1$:
$$\int_M |\nabla u|^2 \geq \lambda_1 \int_M u^2$$

**Reference:** Chavel, I. (1984). *Eigenvalues in Riemannian Geometry*. Academic Press.

### Step 3: Local Spectral Bounds

**Local Gap:** On $B_r(x)$:
$$\lambda_1(L|_{B_r(x)}) \geq \lambda_0$$

**Neumann Problem:** Consider Neumann eigenvalues on balls:
$$L \phi = \lambda \phi \text{ in } B_r, \quad \partial_\nu \phi = 0 \text{ on } \partial B_r$$

**Scaling:** $\lambda_1(B_r) \sim r^{-2}$

### Step 4: Domain Decomposition

**Partition:** Decompose $M = \bigcup_i U_i$ with overlap.

**Local-to-Global (Li-Yau, 1980):**
$$\lambda_1(M) \geq c(n) \min_i \lambda_1(U_i)$$

**Reference:** Li, P., Yau, S. T. (1980). Estimates of eigenvalues of a compact Riemannian manifold. *Proc. Symp. Pure Math.*, 36, 205-239.

### Step 5: Cheeger Inequality

**Cheeger Constant:**
$$h(M) := \inf_{\Sigma} \frac{\mathcal{H}^{n-1}(\Sigma)}{\min(\text{Vol}(A), \text{Vol}(B))}$$

where $\Sigma$ separates $M = A \sqcup \Sigma \sqcup B$.

**Cheeger's Inequality (1970):**
$$\lambda_1 \geq \frac{h^2}{4}$$

**Reference:** Cheeger, J. (1970). A lower bound for the smallest eigenvalue of the Laplacian. *Problems in Analysis*, Princeton.

**Promotion:** Uniform local isoperimetric constant implies global spectral gap.

### Step 6: Spectral Gap and Curvature

**Lichnerowicz Theorem (1958):** If $\text{Ric} \geq (n-1)K > 0$:
$$\lambda_1 \geq nK$$

**Reference:** Lichnerowicz, A. (1958). *Géométrie des Groupes de Transformations*. Dunod.

**Promotion:** Curvature lower bounds promote to spectral gap.

### Step 7: Exponential Convergence

**Theorem:** If $\lambda_1(L) > 0$:
$$\|T_t - T_*\| \leq C e^{-\lambda_1 t/2} \|T_0 - T_*\|$$

*Proof:* The gradient flow satisfies:
$$\frac{d}{dt}(T_t - T_*) = -L(T_t - T_*) + O(\|T_t - T_*\|^2)$$

Linear part gives exponential decay:
$$\frac{d}{dt}\|T_t - T_*\|^2 \leq -2\lambda_1 \|T_t - T_*\|^2$$

### Step 8: Stability Analysis

**Stable Equilibrium:** $T_*$ is **stable** if $L \geq 0$ (all eigenvalues non-negative).

**Asymptotically Stable:** $T_*$ is **asymptotically stable** if $\lambda_1(L) > 0$.

**Spectral Gap Guarantee:** Under soft permits with unique equilibrium:
$$\lambda_1(L) \geq c(\theta) > 0$$

where $\theta$ is Łojasiewicz exponent.

### Step 9: Uniform Gap Across Family

**Family of Equilibria:** If $\{T_*^\alpha\}_{\alpha \in A}$ is a family:
$$\inf_\alpha \lambda_1(L|_{T_*^\alpha}) \geq \lambda_{\min}$$

**Promotion:** Uniform gap over compact parameter space.

### Step 10: Compilation Theorem

**Theorem (Spectral Gap Promotion):**

1. **Local-to-Global:** Local spectral gaps imply global gap

2. **Cheeger:** Isoperimetric constant controls gap

3. **Curvature:** Positive Ricci curvature implies gap

4. **Exponential Decay:** Gap implies exponential convergence

**Applications:**
- Stability of minimal surfaces
- Convergence rate for geometric flows
- Spectral characterization of equilibria

## Key GMT Inequalities Used

1. **Rayleigh Quotient:**
   $$\lambda_1 = \inf \frac{\langle Lu, u \rangle}{\|u\|^2}$$

2. **Cheeger:**
   $$\lambda_1 \geq h^2/4$$

3. **Lichnerowicz:**
   $$\text{Ric} \geq (n-1)K \implies \lambda_1 \geq nK$$

4. **Exponential Decay:**
   $$\|T_t - T_*\| \leq Ce^{-\lambda_1 t/2}$$

## Literature References

- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Chavel, I. (1984). *Eigenvalues in Riemannian Geometry*. Academic Press.
- Li, P., Yau, S. T. (1980). Estimates of eigenvalues. *Proc. Symp. Pure Math.*, 36.
- Cheeger, J. (1970). A lower bound for the smallest eigenvalue. *Problems in Analysis*.
- Lichnerowicz, A. (1958). *Géométrie des Groupes de Transformations*. Dunod.
