# LOCK-SpectralGen: Spectral Generator Lock — GMT Translation

## Original Statement (Hypostructure)

The spectral generator lock shows that spectral properties of the linearized operator generate obstructions to deformation, locking the configuration against certain perturbations.

## GMT Setting

**Linearized Operator:** Second variation of energy functional

**Spectral Data:** Eigenvalues and eigenfunctions of linearization

**Generator:** Spectral data generates constraint algebra

## GMT Statement

**Theorem (Spectral Generator Lock).** For critical $T_* \in \mathbf{I}_k(M)$ of functional $\Phi$:

1. **Linearization:** $L = D^2\Phi(T_*)$ is self-adjoint operator

2. **Kernel:** $\ker(L)$ generates infinitesimal symmetries/moduli

3. **Negative Eigenspace:** $\{v : \langle Lv, v \rangle < 0\}$ gives unstable directions

4. **Lock:** Deformations transverse to spectral decomposition are blocked

## Proof Sketch

### Step 1: Second Variation

**First Variation:** For $T$ critical:
$$\delta\Phi(T)[\xi] = 0 \quad \text{for all } \xi$$

**Second Variation:** The Hessian:
$$D^2\Phi(T)[\xi, \eta] = \left.\frac{d^2}{ds\, dt}\right|_{s=t=0} \Phi(T + s\xi + t\eta)$$

**Reference:** Simons, J. (1968). Minimal varieties in Riemannian manifolds. *Ann. of Math.*, 88, 62-105.

### Step 2: Jacobi Operator

**For Area Functional:** At minimal surface $\Sigma$, the Jacobi operator:
$$L_\Sigma = \Delta_\Sigma + |A|^2 + \text{Ric}_M(\nu, \nu)$$

acting on normal variations.

**Reference:** Schoen, R., Yau, S. T. (1979). On the structure of manifolds with positive scalar curvature. *Manuscripta Math.*, 28, 159-183.

**Eigenvalue Problem:** $L_\Sigma u = \lambda u$

### Step 3: Morse Index

**Definition:** Morse index of $T_*$:
$$\text{ind}(T_*) = \dim\{v : D^2\Phi(T_*)[v,v] < 0\}$$

**Stability:** $T_*$ is stable if $\text{ind}(T_*) = 0$.

**Reference:** Morse, M. (1934). *The Calculus of Variations in the Large*. AMS.

### Step 4: Kernel and Moduli

**Jacobi Fields:** $\ker(L)$ consists of infinitesimal deformations preserving criticality.

**Moduli Space:** Near non-degenerate critical point:
$$\dim(\text{Moduli}) = \dim(\ker(L))$$

**Reference:** Fischer-Colbrie, D., Schoen, R. (1980). The structure of complete stable minimal surfaces. *Comm. Pure Appl. Math.*, 33, 199-211.

### Step 5: Spectral Decomposition

**Self-Adjointness:** $L$ is self-adjoint in $L^2$ with appropriate boundary conditions.

**Spectral Theorem:**
$$L^2 = \bigoplus_{\lambda \in \sigma(L)} E_\lambda$$

where $E_\lambda$ is eigenspace for eigenvalue $\lambda$.

**Decomposition:** Any deformation $\xi = \sum_\lambda \xi_\lambda$ with $\xi_\lambda \in E_\lambda$.

### Step 6: Negative Eigenspace Lock

**Instability Directions:** Deformations in negative eigenspace decrease energy:
$$\Phi(T_* + \epsilon v) \approx \Phi(T_*) + \frac{\epsilon^2}{2} \langle Lv, v \rangle < \Phi(T_*)$$

for $v$ in negative eigenspace.

**Lock Mechanism:** If we seek minimizers, negative directions are forbidden at critical points.

### Step 7: Positive Eigenspace Stability

**Local Minimum:** If $\text{ind}(T_*) = 0$:
$$D^2\Phi(T_*)[v,v] \geq 0 \quad \text{for all } v$$

**Strong Stability:** If spectral gap $\lambda_1(L) > 0$:
$$D^2\Phi(T_*)[v,v] \geq \lambda_1 \|v\|^2$$

**Reference:** Schoen, R. (1983). Estimates for stable minimal surfaces in three-manifolds. *Annals of Mathematics Studies*, 103, 111-126.

### Step 8: Fredholm Theory

**Fredholm Operator:** $L$ is Fredholm with:
$$\text{ind}_F(L) = \dim\ker(L) - \dim\text{coker}(L)$$

**Perturbation:** Small perturbations of $T_*$ are controlled by Fredholm index.

**Reference:** Smale, S. (1965). An infinite dimensional version of Sard's theorem. *Amer. J. Math.*, 87, 861-866.

### Step 9: Constraint from Spectrum

**Obstruction Theory:** Deformation $T_* \leadsto T_\epsilon$ requires:
1. $\xi_\lambda = 0$ for $\lambda < 0$ (stability)
2. $\xi_0$ satisfies integrability conditions (if $\ker(L) \neq 0$)
3. $\xi_\lambda$ free for $\lambda > 0$

**Lock:** Negative and kernel directions impose constraints.

### Step 10: Compilation Theorem

**Theorem (Spectral Generator Lock):**

1. **Linearization:** $L = D^2\Phi(T_*)$ self-adjoint

2. **Index:** $\text{ind}(T_*) = $ dimension of negative eigenspace

3. **Kernel:** $\ker(L)$ generates moduli/symmetries

4. **Lock:** Deformations constrained by spectral decomposition

**Applications:**
- Stability analysis of minimal surfaces
- Moduli space dimension
- Bifurcation theory

## Key GMT Inequalities Used

1. **Second Variation:**
   $$D^2\Phi[v,v] = \langle Lv, v \rangle$$

2. **Stability:**
   $$\text{ind}(T_*) = 0 \iff D^2\Phi \geq 0$$

3. **Spectral Gap:**
   $$D^2\Phi[v,v] \geq \lambda_1 \|v\|^2$$

4. **Morse Index:**
   $$\text{ind}(T_*) = \#\{\lambda_i < 0\}$$

## Literature References

- Simons, J. (1968). Minimal varieties in Riemannian manifolds. *Ann. of Math.*, 88.
- Morse, M. (1934). *Calculus of Variations in the Large*. AMS.
- Schoen, R., Yau, S. T. (1979). Positive scalar curvature manifolds. *Manuscripta Math.*, 28.
- Fischer-Colbrie, D., Schoen, R. (1980). Complete stable minimal surfaces. *Comm. Pure Appl. Math.*, 33.
- Schoen, R. (1983). Estimates for stable minimal surfaces. *Annals Math. Studies*, 103.
- Smale, S. (1965). Infinite dimensional Sard's theorem. *Amer. J. Math.*, 87.
