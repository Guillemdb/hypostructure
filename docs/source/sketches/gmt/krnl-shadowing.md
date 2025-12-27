# KRNL-Shadowing: Shadowing Metatheorem — GMT Translation

## Original Statement (Hypostructure)

For systems with stiffness (spectral gap), every pseudo-orbit is shadowed by a true orbit. This upgrades numerical simulations to existence proofs.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — compact Riemannian manifold

**Discrete Dynamics:** $f: M \to M$ — $C^1$ diffeomorphism

**Continuous Dynamics:** $\phi_t: M \to M$ — $C^1$ flow

**Hyperbolicity:** Uniform exponential dichotomy on tangent bundle

**Rectifiable Currents:** $\mathbf{I}_1(M)$ — 1-currents representing orbits

## GMT Statement

**Theorem (Shadowing Lemma for Currents).** Let $(M, g)$ be compact and $\phi_t$ a uniformly hyperbolic flow with spectral gap $\lambda > 0$. Let $\gamma: [0, T] \to M$ be a piecewise $C^1$ curve representing an $\varepsilon$-pseudo-orbit:
$$d(\gamma(t + h), \phi_h(\gamma(t))) < \varepsilon \quad \text{for all } t, h \text{ with small } |h|$$

Then there exists a true orbit $\sigma: [0, T'] \to M$ with $\sigma(0) \in M$, $\phi_{T'}(\sigma(0)) = \sigma(T')$, such that:
$$d_{\mathbf{I}_1}(\llbracket \gamma \rrbracket, \llbracket \sigma \rrbracket) \leq C(\lambda) \cdot \varepsilon$$

where $d_{\mathbf{I}_1}$ is the flat distance on 1-currents.

**Corollary (Rectifiability of Shadowing):** The shadowing orbit $\sigma$ is contained in the tubular neighborhood:
$$\sigma \subset \{x \in M : d(x, \text{Im}(\gamma)) < \delta(\varepsilon)\}$$

where $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

## Proof Sketch

### Step 1: Hyperbolicity and Exponential Dichotomy

**Uniform Hyperbolicity (Anosov, 1967):** The flow $\phi_t$ is uniformly hyperbolic if the tangent bundle splits:
$$TM = E^s \oplus E^c \oplus E^u$$

where $E^c$ is the flow direction and:
- $\|D\phi_t|_{E^s}\| \leq Ce^{-\lambda t}$ for $t > 0$ (stable)
- $\|D\phi_t|_{E^u}\| \leq Ce^{\lambda t}$ for $t < 0$ (unstable)

**Spectral Gap:** The gap $\lambda > 0$ is the minimum expansion/contraction rate.

**Reference:** Anosov, D. V. (1967). Geodesic flows on closed Riemannian manifolds with negative curvature. *Proc. Steklov Inst. Math.*, 90.

### Step 2: Pseudo-Orbit Definition

**$\varepsilon$-Pseudo-Orbit:** A sequence $(x_n)_{n=0}^N$ is an $\varepsilon$-pseudo-orbit for the time-1 map $f = \phi_1$ if:
$$d(f(x_n), x_{n+1}) < \varepsilon \quad \text{for } n = 0, \ldots, N-1$$

**Continuous Version:** A curve $\gamma: [0, T] \to M$ is an $\varepsilon$-pseudo-orbit for the flow if:
$$d(\gamma(t+s), \phi_s(\gamma(t))) < \varepsilon \cdot s \quad \text{for small } s$$

**Reference:** Bowen, R. (1975). ω-limit sets for Axiom A diffeomorphisms. *J. Diff. Eq.*, 18, 333-339.

### Step 3: Shadowing by Contraction Mapping

**Classical Shadowing (Anosov-Bowen):** For $\varepsilon$ sufficiently small (depending on $\lambda$ and $C$), there exists a unique true orbit $(y_n)$ with:
$$d(x_n, y_n) < \delta = \frac{C\varepsilon}{\lambda(1 - e^{-\lambda})}$$

**Proof Method (Palmer, 1988):** Define the operator $\Phi$ on sequences:
$$(\Phi(y))_n = f^{-1}(y_{n+1}) \text{ adjusted to stay near } x_n$$

The hyperbolicity provides contraction:
$$\|\Phi(y) - \Phi(z)\|_\infty \leq (1 - c\lambda)\|y - z\|_\infty$$

By Banach fixed point theorem, $\Phi$ has a unique fixed point — the shadowing orbit.

**Reference:** Palmer, K. (1988). Exponential dichotomies, the shadowing lemma and transversal homoclinic points. *Dynamics Reported*, 1, 265-306.

### Step 4: Current Formulation

**Pseudo-Orbit Current:** Represent the pseudo-orbit as a 1-current:
$$T_\gamma := \int_0^T \llbracket \gamma(t) \rrbracket \otimes \dot{\gamma}(t) \, dt \in \mathbf{I}_1(M)$$

**True Orbit Current:** The shadowing orbit defines:
$$T_\sigma := \int_0^{T'} \llbracket \sigma(t) \rrbracket \otimes \dot{\sigma}(t) \, dt$$

**Flat Distance Estimate:** By the pointwise shadowing bound:
$$\mathbb{F}(T_\gamma - T_\sigma) \leq \int_0^T d(\gamma(t), \sigma(t)) \cdot |\dot{\gamma}(t)| \, dt \leq \delta \cdot \text{Length}(\gamma)$$

### Step 5: Rectifiability of the Shadowing Correspondence

**Shadowing Map:** Define $\Sigma_\varepsilon: \{\varepsilon\text{-pseudo-orbits}\} \to \{\text{true orbits}\}$ sending $\gamma \mapsto \sigma$.

**Lipschitz Continuity (Pilyugin, 1999):** The shadowing map is Lipschitz:
$$d(\Sigma_\varepsilon(\gamma_1), \Sigma_\varepsilon(\gamma_2)) \leq L \cdot d(\gamma_1, \gamma_2)$$

with Lipschitz constant $L = O(1/\lambda)$.

**Rectifiable Graph:** The graph $\{(\gamma, \Sigma_\varepsilon(\gamma))\}$ in $C([0,T], M) \times C([0,T'], M)$ is a rectifiable subset.

**Reference:** Pilyugin, S. Yu. (1999). *Shadowing in Dynamical Systems*. Lecture Notes in Mathematics 1706, Springer.

### Step 6: Quantitative Shadowing Bounds

**Explicit Constants (Katok-Hasselblatt):** For Anosov diffeomorphisms with hyperbolicity constants $(C, \lambda)$:

$$\delta = \frac{C^2 \varepsilon}{1 - e^{-\lambda}} \leq \frac{2C^2 \varepsilon}{\lambda}$$

**Hölder Shadowing (Barreira-Valls):** Under Hölder continuous splitting, the shadowing is also Hölder:
$$d(\gamma(t), \sigma(t)) \leq C \varepsilon^\alpha$$

for some $\alpha \in (0, 1)$ depending on the Hölder exponent of $E^{s/u}$.

**Reference:**
- Katok, A., Hasselblatt, B. (1995). *Introduction to the Modern Theory of Dynamical Systems*. Cambridge University Press.
- Barreira, L., Valls, C. (2008). *Stability of Nonautonomous Differential Equations*. Springer.

### Step 7: Application to Numerical Verification

**Rigorous Numerics:** Given a numerically computed trajectory $\gamma^{\text{num}}$ with verified error bounds:
$$d(\gamma^{\text{num}}(t + h), \phi_h^{\text{num}}(\gamma^{\text{num}}(t))) < \varepsilon_{\text{num}}$$

the shadowing lemma guarantees existence of a true orbit $\sigma$ within $\delta(\varepsilon_{\text{num}})$ of the numerical trajectory.

**Reference:**
- Lohner, R. J. (1987). Enclosing the solutions of ordinary initial and boundary value problems. *Computer Arithmetic*, 255-286.
- Tucker, W. (2002). A rigorous ODE solver and Smale's 14th problem. *Found. Comput. Math.*, 2, 53-117.

## Key GMT Inequalities Used

1. **Anosov Exponential Bounds:**
   $$\|D\phi_t|_{E^s}\| \leq Ce^{-\lambda t}, \quad \|D\phi_{-t}|_{E^u}\| \leq Ce^{-\lambda t}$$

2. **Contraction Constant:**
   $$\|\Phi(y) - \Phi(z)\|_\infty \leq (1 - c\lambda)\|y - z\|_\infty$$

3. **Shadowing Bound:**
   $$d(\gamma(t), \sigma(t)) \leq \frac{C\varepsilon}{\lambda}$$

4. **Flat Distance for Curves:**
   $$\mathbb{F}(\llbracket \gamma \rrbracket - \llbracket \sigma \rrbracket) \leq \sup_t d(\gamma(t), \sigma(t)) \cdot \text{Length}(\gamma)$$

## Literature References

- Anosov, D. V. (1967). Geodesic flows on closed Riemann manifolds with negative curvature. *Trudy Mat. Inst. Steklov*, 90. [English: Proc. Steklov Inst. Math. 90]
- Bowen, R. (1975). ω-limit sets for Axiom A diffeomorphisms. *J. Differential Equations*, 18, 333-339.
- Palmer, K. (1988). Exponential dichotomies, the shadowing lemma and transversal homoclinic points. *Dynamics Reported*, 1, 265-306.
- Pilyugin, S. Yu. (1999). *Shadowing in Dynamical Systems*. Lecture Notes in Math. 1706, Springer.
- Katok, A., Hasselblatt, B. (1995). *Introduction to the Modern Theory of Dynamical Systems*. Cambridge University Press.
- Robinson, C. (1977). Stability theorems and hyperbolicity in dynamical systems. *Rocky Mountain J. Math.*, 7, 425-437.
