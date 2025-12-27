# KRNL-Jacobi: Action Reconstruction — GMT Translation

## Original Statement (Hypostructure)

The canonical Lyapunov functional equals the minimal geodesic action from the state to the safe manifold with respect to the Jacobi metric induced by dissipation.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — complete Riemannian manifold

**Energy Functional:** $\Phi: M \to [0, \infty)$ — smooth function with $\nabla \Phi \neq 0$ except on $\text{Crit}(\Phi)$

**Dissipation:** $\mathfrak{D} := |\nabla \Phi|_g^2$ — squared gradient norm

**Jacobi Metric:** $g_\Phi := \mathfrak{D} \cdot g = |\nabla \Phi|^2 \cdot g$ — conformal rescaling

**Safe Manifold:** $\mathcal{M} := \{x \in M : \nabla \Phi(x) = 0\}$ — critical set

## GMT Statement

**Theorem (Jacobi Metric Reconstruction).** Let $(M, g)$ be a complete Riemannian manifold and $\Phi: M \to [0, \infty)$ a proper smooth function. Assume:

1. **(Non-Degeneracy)** $\nabla \Phi(x) \neq 0$ for $x \notin \mathcal{M}$
2. **(Gradient Consistency)** The gradient flow $\dot{x} = -\nabla \Phi(x)$ satisfies $|\dot{x}|_g = |\nabla \Phi(x)|_g$
3. **(Stiffness)** Łojasiewicz-Simon inequality holds near $\mathcal{M}$

Then the Lyapunov functional satisfies:
$$\mathcal{L}(x) = \Phi_{\min} + \text{dist}_{g_\Phi}(x, \mathcal{M})$$

where $\text{dist}_{g_\Phi}$ is the Riemannian distance in the Jacobi metric $g_\Phi = |\nabla \Phi|^2 \cdot g$.

## Proof Sketch

### Step 1: Jacobi Metric Structure

The Jacobi metric is the conformal rescaling:
$$g_\Phi := |\nabla \Phi|^2 \cdot g$$

**Length in Jacobi Metric:** For a curve $\gamma: [0, 1] \to M$:
$$L_{g_\Phi}(\gamma) = \int_0^1 |\dot{\gamma}|_{g_\Phi} \, dt = \int_0^1 |\nabla \Phi|(\gamma(t)) \cdot |\dot{\gamma}|_g(t) \, dt$$

**Geodesic Distance:**
$$d_\Phi(x, y) := \inf_{\gamma: x \to y} L_{g_\Phi}(\gamma) = \inf_{\gamma} \int_0^1 |\nabla \Phi|(\gamma) \cdot |\dot{\gamma}|_g \, dt$$

### Step 2: Gradient Flow as Jacobi Geodesic

**Claim:** The gradient flow $\gamma(t) = S_t x$ (solving $\dot{\gamma} = -\nabla \Phi(\gamma)$) is a minimizing geodesic in $(M, g_\Phi)$ connecting $x$ to $\mathcal{M}$.

*Proof:* Along the gradient flow:
$$|\dot{\gamma}|_g = |\nabla \Phi(\gamma)|_g \quad \text{(by Gradient Consistency)}$$

Hence:
$$L_{g_\Phi}(\gamma|_{[0,T]}) = \int_0^T |\nabla \Phi|(\gamma) \cdot |\dot{\gamma}|_g \, dt = \int_0^T |\nabla \Phi|^2(\gamma(t)) \, dt$$

By the Energy-Dissipation Identity:
$$\Phi(\gamma(0)) - \Phi(\gamma(T)) = \int_0^T |\nabla \Phi|^2(\gamma(t)) \, dt = L_{g_\Phi}(\gamma|_{[0,T]})$$

**Minimality:** Any other curve $\tilde{\gamma}$ from $x$ to $y$ with $y = \gamma(T)$ satisfies:
$$L_{g_\Phi}(\tilde{\gamma}) \geq \Phi(x) - \Phi(y) = L_{g_\Phi}(\gamma|_{[0,T]})$$

by the fundamental inequality for gradient flows.

### Step 3: Distance to Critical Set

**Claim:** $\text{dist}_{g_\Phi}(x, \mathcal{M}) = \Phi(x) - \Phi_{\min}$ when the gradient flow from $x$ reaches $\mathcal{M}$.

*Proof:* Let $\gamma: [0, \infty) \to M$ be the gradient flow from $x$. By Łojasiewicz-Simon, $\gamma(t) \to x^* \in \mathcal{M}$ as $t \to \infty$.

The total Jacobi length:
$$\int_0^\infty |\nabla \Phi|^2(\gamma(t)) \, dt = \Phi(x) - \lim_{t \to \infty} \Phi(\gamma(t)) = \Phi(x) - \Phi_{\min}$$

This equals $\text{dist}_{g_\Phi}(x, \mathcal{M})$ by the minimality of gradient flow curves.

### Step 4: Conformal Eikonal Equation

The distance function $u(x) := \text{dist}_{g_\Phi}(x, \mathcal{M})$ satisfies the **eikonal equation** in the Jacobi metric:
$$|\nabla_{g_\Phi} u|_{g_\Phi} = 1$$

**Translation to Original Metric:** Using the conformal relation $|\cdot|_{g_\Phi} = |\nabla \Phi| \cdot |\cdot|_g$:
$$|\nabla_g u|_g = |\nabla \Phi|$$

Squaring:
$$|\nabla_g u|_g^2 = |\nabla \Phi|^2 = \mathfrak{D}$$

This is the **Hamilton-Jacobi equation** $|\nabla \mathcal{L}|^2 = \mathfrak{D}$ with $\mathcal{L} = \Phi_{\min} + u$.

### Step 5: Uniqueness and Characterization

**Uniqueness:** The Lyapunov functional $\mathcal{L}$ is characterized as the unique viscosity solution to:
$$\begin{cases}
|\nabla \mathcal{L}|^2 = |\nabla \Phi|^2 & \text{on } M \setminus \mathcal{M} \\
\mathcal{L} = \Phi_{\min} & \text{on } \mathcal{M}
\end{cases}$$

*Proof of Uniqueness:* If $\mathcal{L}_1, \mathcal{L}_2$ both solve this equation with the same boundary values on $\mathcal{M}$, then $w := \mathcal{L}_1 - \mathcal{L}_2$ satisfies:
$$|\nabla \mathcal{L}_1|^2 - |\nabla \mathcal{L}_2|^2 = 0$$

Linearizing near any point: $\nabla(\mathcal{L}_1 + \mathcal{L}_2) \cdot \nabla w = 0$.

Since $\nabla(\mathcal{L}_1 + \mathcal{L}_2) \neq 0$ away from $\mathcal{M}$ (by Hopf boundary lemma), the level sets of $w$ coincide with characteristics of the eikonal equation.

By the boundary condition $w|_{\mathcal{M}} = 0$ and propagation along characteristics, $w \equiv 0$.

### Step 6: Rectifiable Currents Perspective

**Current Formulation:** Let $T_x$ be the 1-current representing the gradient flow line from $x$ to $\mathcal{M}$:
$$T_x := \int_0^\infty \llbracket S_t x \rrbracket \cdot |\nabla \Phi|(S_t x) \, dt$$

**Mass Computation:**
$$\mathbf{M}(T_x) = \int_0^\infty |\nabla \Phi|(S_t x) \, dt$$

**Relation to Jacobi Distance:** By the AGS characterization of curves of maximal slope:
$$\mathbf{M}(T_x) = \sqrt{\text{Total Dissipation}} = \sqrt{\Phi(x) - \Phi_{\min}}$$

Wait — this needs correction. The mass is:
$$\mathbf{M}(T_x) = \int_0^\infty |\nabla \Phi| \, dt$$

which by Cauchy-Schwarz relates to both the arc length and the energy drop. The exact relationship:
$$\left( \int_0^\infty |\nabla \Phi| \, dt \right)^2 \leq T \cdot \int_0^\infty |\nabla \Phi|^2 \, dt$$

with equality for finite $T$ only if $|\nabla \Phi|$ is constant along the flow.

## Key GMT Inequalities Used

1. **Conformal Metric Relation:**
   $$|\nabla_{g_\Phi} f|_{g_\Phi}^2 = |\nabla \Phi|^{-2} \cdot |\nabla_g f|_g^2$$

2. **Eikonal Equation:**
   $$|\nabla_g (\text{dist}_{g_\Phi}(\cdot, \mathcal{M}))|_g = |\nabla \Phi|_g$$

3. **Energy-Dissipation Identity:**
   $$\Phi(x) - \Phi(y) = \int_{\text{GF}(x \to y)} |\nabla \Phi|^2 \, dt$$

4. **Gradient Flow Optimality:**
   $$L_{g_\Phi}(\text{any curve } x \to y) \geq L_{g_\Phi}(\text{GF}(x \to y)) = \Phi(x) - \Phi(y)$$

## Literature References

- Mielke, A. (2011). A gradient structure for reaction-diffusion systems and for energy-drift-diffusion systems. *Nonlinearity*, 24, 1329-1346.
- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces*. Birkhäuser.
- Evans, L. C. (2010). *Partial Differential Equations*. 2nd ed. AMS. [Chapter 3: Hamilton-Jacobi]
- Crandall, M. G., Lions, P.-L. (1983). Viscosity solutions of Hamilton-Jacobi equations. *Trans. Amer. Math. Soc.*, 277, 1-42.
- Fathi, A. (2008). *Weak KAM Theorem in Lagrangian Dynamics*. Cambridge University Press.
