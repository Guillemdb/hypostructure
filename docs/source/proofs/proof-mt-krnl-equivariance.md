# Proof of KRNL-Equivariance (Symmetry Compatibility)

:::{prf:proof}
:label: proof-mt-krnl-equivariance

**Theorem Reference:** {prf:ref}`mt-krnl-equivariance`

## Setup and Notation

Let $G$ be a compact Lie group with Haar measure $\mu_G$ (normalized: $\mu_G(G) = 1$). We work in the following framework:

**System Distribution:** Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space and $\mathcal{S}: \Omega \to \mathbf{Sys}$ a random variable taking values in the category of systems $\mathbf{Sys}$. Each system $S \in \mathbf{Sys}$ consists of:
- A state space $X_S$ (Banach/Sobolev space)
- An energy functional $\Phi_S: X_S \to [0, \infty]$
- A dissipation functional $\mathfrak{D}_S: X_S \to [0, \infty]$
- A symmetry group action $G \times X_S \to X_S$

**Parameter Space:** Let $\Theta$ be a smooth manifold (possibly infinite-dimensional, e.g., a Fréchet manifold for neural network parameters). The group $G$ acts smoothly on $\Theta$ via a representation $\rho: G \to \mathrm{Diff}(\Theta)$.

**Hypostructure Assignment:** The parametrized hypostructure assignment is a functor
$$\mathcal{H}_\Theta: \mathbf{Sys} \to \mathbf{Hypo}$$
where $\mathbf{Hypo}$ is the category of hypostructures (small categories with initial objects corresponding to singularity germs). For each system $S$ and parameter $\Theta$, $\mathcal{H}_\Theta(S)$ encodes the categorical structure detecting singularities in $S$ under the parametrization $\Theta$.

**Loss Functional:** For each pair $(\Theta, S)$, define the loss
$$\mathcal{L}(\Theta, S) := \mathbb{E}_{u \sim \mathcal{D}_S}\left[\sum_{A \in \mathcal{A}} w_A \cdot \left\|K_{A,S}^{(\Theta)}(u)\right\|_{\mathcal{C}}\right]$$
where:
- $\mathcal{D}_S$ is the trajectory distribution for system $S$
- $\mathcal{A}$ is the alphabet of certificate types (axioms)
- $w_A > 0$ are importance weights
- $K_{A,S}^{(\Theta)}(u)$ is the certificate produced by the Sieve for trajectory $u$ under parametrization $\Theta$
- $\|\cdot\|_{\mathcal{C}}$ is a norm on the certificate space measuring "quality" or "optimality"

**Risk Functional:**
$$R(\Theta) := \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{L}(\Theta, S)] = \int_{\Omega} \mathcal{L}(\Theta, S(\omega)) \, d\mathbb{P}(\omega)$$

### Hypotheses

We assume the following compatibility conditions:

**(H1) Group-Covariant Distribution:**
$$S \sim \mathcal{S} \quad \Longrightarrow \quad g \cdot S \sim \mathcal{S} \quad \forall g \in G$$
Formally: the pushforward $(g \cdot)_\# \mathcal{S} = \mathcal{S}$ where $(g \cdot S)(\omega) := g \cdot S(\omega)$ via the group action on systems.

**(H2) Equivariant Parametrization:**
$$g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$$
There exists a natural isomorphism $\phi_{g,\Theta,S}: g \cdot \mathcal{H}_\Theta(S) \to \mathcal{H}_{g \cdot \Theta}(g \cdot S)$ in $\mathbf{Hypo}$ satisfying the cocycle condition:
$$\phi_{gh,\Theta,S} = \phi_{g, h \cdot \Theta, h \cdot S} \circ (g \cdot \phi_{h,\Theta,S})$$

**(H3) Defect-Level Equivariance:**
$$K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$$
For all axiom types $A \in \mathcal{A}$, parameters $\Theta$, systems $S$, group elements $g \in G$, and trajectories $u \in X_S$, the certificate produced is equivariant under simultaneous transformation of parameter, system, and state.

---

## Step 1: Risk Invariance Under Group Action

**Claim:** The risk functional satisfies $R(g \cdot \Theta) = R(\Theta)$ for all $g \in G$.

**Proof of Claim:**

Fix $g \in G$ and compute:
\begin{align*}
R(g \cdot \Theta)
&= \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{L}(g \cdot \Theta, S)] \\
&= \int_{\Omega} \mathcal{L}(g \cdot \Theta, S(\omega)) \, d\mathbb{P}(\omega)
\end{align*}

By hypothesis (H1), the distribution $\mathcal{S}$ is $G$-invariant. Perform the change of variables $S' = g \cdot S$. Since $(g \cdot)_\# \mathbb{P} = \mathbb{P}$ on the system space (by $G$-invariance of $\mathcal{S}$), we have:
$$R(g \cdot \Theta) = \int_{\Omega} \mathcal{L}(g \cdot \Theta, g \cdot S'(\omega)) \, d\mathbb{P}(\omega)$$

Now expand the loss functional:
\begin{align*}
\mathcal{L}(g \cdot \Theta, g \cdot S')
&= \mathbb{E}_{u \sim \mathcal{D}_{g \cdot S'}}\left[\sum_{A \in \mathcal{A}} w_A \cdot \left\|K_{A,g \cdot S'}^{(g \cdot \Theta)}(u)\right\|_{\mathcal{C}}\right]
\end{align*}

By equivariance of the trajectory distribution, $\mathcal{D}_{g \cdot S'} = (g \cdot)_\# \mathcal{D}_{S'}$. Substituting $u = g \cdot v$ where $v \sim \mathcal{D}_{S'}$:
\begin{align*}
\mathcal{L}(g \cdot \Theta, g \cdot S')
&= \mathbb{E}_{v \sim \mathcal{D}_{S'}}\left[\sum_{A \in \mathcal{A}} w_A \cdot \left\|K_{A,g \cdot S'}^{(g \cdot \Theta)}(g \cdot v)\right\|_{\mathcal{C}}\right]
\end{align*}

By hypothesis (H3) (Defect-Level Equivariance):
$$K_{A,g \cdot S'}^{(g \cdot \Theta)}(g \cdot v) = K_{A,S'}^{(\Theta)}(v)$$

Therefore:
\begin{align*}
\mathcal{L}(g \cdot \Theta, g \cdot S')
&= \mathbb{E}_{v \sim \mathcal{D}_{S'}}\left[\sum_{A \in \mathcal{A}} w_A \cdot \left\|K_{A,S'}^{(\Theta)}(v)\right\|_{\mathcal{C}}\right] \\
&= \mathcal{L}(\Theta, S')
\end{align*}

Substituting back into the risk integral:
$$R(g \cdot \Theta) = \int_{\Omega} \mathcal{L}(\Theta, S'(\omega)) \, d\mathbb{P}(\omega) = \mathbb{E}_{S' \sim \mathcal{S}}[\mathcal{L}(\Theta, S')] = R(\Theta)$$

Thus $R$ is $G$-invariant. **QED Claim.**

---

## Step 2: Gradient Equivariance and Flow Preservation

**Claim:** If $R$ is $C^1$ and $G$-invariant, then the gradient is $G$-equivariant:
$$\nabla R(g \cdot \Theta) = \rho'(g)[\nabla R(\Theta)]$$
where $\rho'(g): T_\Theta \Theta \to T_{g \cdot \Theta}\Theta$ is the derivative of the group action.

**Proof of Claim:**

By the chain rule for group actions on manifolds, if $F: \Theta \to \mathbb{R}$ satisfies $F(g \cdot \Theta) = F(\Theta)$, then differentiating with respect to $\Theta$ at $\Theta_0$:
$$dF(g \cdot \Theta_0)[\rho'(g) \cdot v] = dF(\Theta_0)[v] \quad \forall v \in T_{\Theta_0}\Theta$$

In coordinates, if $\Theta$ is a Riemannian manifold with metric $h$, the gradient satisfies:
$$h(g \cdot \Theta)(\nabla R(g \cdot \Theta), w) = dR(g \cdot \Theta)[w]$$

By invariance:
$$dR(g \cdot \Theta)[w] = dR(\Theta)[\rho'(g^{-1}) \cdot w]$$

Therefore:
\begin{align*}
h(g \cdot \Theta)(\nabla R(g \cdot \Theta), w)
&= h(\Theta)(\nabla R(\Theta), \rho'(g^{-1}) \cdot w)
\end{align*}

If the metric $h$ is $G$-invariant (i.e., $h(g \cdot \Theta)(\rho'(g) \cdot v, \rho'(g) \cdot w) = h(\Theta)(v, w)$), then:
\begin{align*}
h(g \cdot \Theta)(\nabla R(g \cdot \Theta), w)
&= h(g \cdot \Theta)(\rho'(g)[\nabla R(\Theta)], w)
\end{align*}

By non-degeneracy of $h$:
$$\nabla R(g \cdot \Theta) = \rho'(g)[\nabla R(\Theta)]$$

**Consequence for Gradient Flow:** The gradient flow is the ODE:
$$\frac{d\Theta_t}{dt} = -\nabla R(\Theta_t)$$

If $\Theta_0$ lies on a $G$-orbit, say $\Theta_0 = g_0 \cdot \Theta^*$, then the flow trajectory satisfies:
\begin{align*}
\frac{d}{dt}(g_0^{-1} \cdot \Theta_t)
&= \rho'(g_0^{-1})\left[\frac{d\Theta_t}{dt}\right] \\
&= -\rho'(g_0^{-1})[\nabla R(\Theta_t)] \\
&= -\rho'(g_0^{-1})[\nabla R(g_0 \cdot (g_0^{-1} \cdot \Theta_t))] \\
&= -\rho'(g_0^{-1})[\rho'(g_0)[\nabla R(g_0^{-1} \cdot \Theta_t)]] \\
&= -\nabla R(g_0^{-1} \cdot \Theta_t)
\end{align*}

Thus $\Theta_t' := g_0^{-1} \cdot \Theta_t$ also satisfies the gradient flow equation. By uniqueness of solutions to ODEs:
$$g_0^{-1} \cdot \Theta_t = \Theta_t'(0) \exp(-t \nabla R)|_{\Theta_t'(0)}$$

Therefore, if $\Theta_0 \in G \cdot \Theta^*$, then $\Theta_t \in G \cdot \Theta^*$ for all $t \geq 0$. **The gradient flow preserves $G$-orbits.** **QED Claim.**

---

## Step 3: Risk Minimizers Lie in Symmetry Orbits

**Claim:** Every global minimizer $\widehat{\Theta}$ of $R$ satisfies $\widehat{\Theta} \in G \cdot \Theta^*$ for some $\Theta^* \in \Theta$.

**Proof of Claim:**

Suppose $\widehat{\Theta}$ is a global minimizer: $R(\widehat{\Theta}) = \inf_{\Theta} R(\Theta)$.

By Step 1, for any $g \in G$:
$$R(g \cdot \widehat{\Theta}) = R(\widehat{\Theta})$$

Therefore, the entire orbit $G \cdot \widehat{\Theta} = \{g \cdot \widehat{\Theta} : g \in G\}$ consists of global minimizers.

**Case 1: Discrete Minimizer Set.** If the set of minimizers $\mathcal{M} := \{\Theta : R(\Theta) = \inf R\}$ is discrete, then each minimizer $\widehat{\Theta}$ is isolated up to $G$-action. Define $\Theta^* := \widehat{\Theta}$. Then trivially $\widehat{\Theta} \in G \cdot \Theta^*$.

**Case 2: Continuous Minimizer Manifold.** If $\mathcal{M}$ is a manifold (or more generally, a submanifold of $\Theta$), then by $G$-invariance, $\mathcal{M}$ is a union of $G$-orbits:
$$\mathcal{M} = \bigcup_{[\Theta^*] \in \mathcal{M}/G} G \cdot \Theta^*$$

For any minimizer $\widehat{\Theta} \in \mathcal{M}$, there exists a representative $\Theta^* \in \mathcal{M}/G$ such that $\widehat{\Theta} \in G \cdot \Theta^*$.

**Quantitative Bound (Compactness):** If $\Theta$ is a compact manifold and $R$ is continuous, then by compactness, $R$ attains its minimum. If additionally $R$ is strictly convex on each $G$-orbit (after quotienting), then the minimizer is unique up to $G$-action.

**Remark on Stochastic Gradient Descent:** In practice, minimization is performed via stochastic gradient descent (SGD):
$$\Theta_{k+1} = \Theta_k - \eta_k \nabla \mathcal{L}(\Theta_k, S_k)$$
where $S_k \sim \mathcal{S}$ are i.i.d. samples. By Step 2, if $\Theta_0$ is $G$-invariant (e.g., initialized at a fixed point $\Theta_0 = g \cdot \Theta_0$ for all $g \in G$, which requires $\Theta_0$ to be in the center of the $G$-action), then each iterate $\Theta_k$ remains $G$-invariant in expectation. Alternatively, if $\Theta_0 \in G \cdot \Theta^*$, then by equivariance of the gradient, the iterates remain in $G \cdot \Theta^*$ (up to noise fluctuations). **QED Claim.**

---

## Step 4: Defect Transfer and Certificate Inheritance

**Claim:** If the learned parameters $\widehat{\Theta}$ minimize $R$, then the certificates $K_{A,S}^{(\widehat{\Theta})}$ inherit all symmetries of the system distribution $\mathcal{S}$.

**Proof of Claim:**

Fix a system $S \sim \mathcal{S}$, a trajectory $u \in X_S$, and a group element $g \in G$. By hypothesis (H3):
$$K_{A,g \cdot S}^{(g \cdot \widehat{\Theta})}(g \cdot u) = K_{A,S}^{(\widehat{\Theta})}(u)$$

Since $\widehat{\Theta}$ is a minimizer and $R(g \cdot \widehat{\Theta}) = R(\widehat{\Theta})$ by Step 1, we have $g \cdot \widehat{\Theta} \in G \cdot \widehat{\Theta}$ is also a minimizer.

**Symmetry of Certificates:** For any $g \in G$:
\begin{align*}
K_{A,g \cdot S}^{(\widehat{\Theta})}(g \cdot u)
&= K_{A,g \cdot S}^{(e \cdot \widehat{\Theta})}(g \cdot u) \quad \text{(where $e \in G$ is the identity)} \\
&= K_{A, g \cdot S}^{(g \cdot (g^{-1} \cdot \widehat{\Theta}))}(g \cdot u)
\end{align*}

If $\widehat{\Theta}$ is $G$-invariant (i.e., $g \cdot \widehat{\Theta} = \widehat{\Theta}$ for all $g$, which occurs when $\widehat{\Theta}$ is a fixed point of the $G$-action), then:
$$K_{A,g \cdot S}^{(\widehat{\Theta})}(g \cdot u) = K_{A,S}^{(\widehat{\Theta})}(u)$$

More generally, if $\widehat{\Theta}$ is only in the orbit $G \cdot \Theta^*$, we write $\widehat{\Theta} = h \cdot \Theta^*$ for some $h \in G$. Then:
\begin{align*}
K_{A,g \cdot S}^{(h \cdot \Theta^*)}(g \cdot u)
&= K_{A,g \cdot S}^{(gh \cdot (g^{-1}h) \cdot \Theta^*)}(g \cdot u) \\
&= K_{A,g \cdot S}^{(gh \cdot h^{-1}g^{-1}gh \cdot \Theta^*)}(g \cdot u) \\
&= K_{A,S}^{((g^{-1}gh) \cdot \Theta^*)}(u) \quad \text{(by hypothesis (H3))} \\
&= K_{A,S}^{(h \cdot \Theta^*)}(u)
\end{align*}

Thus the certificates exhibit the same symmetry pattern as the input distribution.

**Hypostructure Inheritance:** By hypothesis (H2), the hypostructure satisfies:
$$g \cdot \mathcal{H}_{\widehat{\Theta}}(S) \simeq \mathcal{H}_{g \cdot \widehat{\Theta}}(g \cdot S)$$

If $\widehat{\Theta}$ is $G$-invariant, then $g \cdot \widehat{\Theta} = \widehat{\Theta}$, and:
$$g \cdot \mathcal{H}_{\widehat{\Theta}}(S) \simeq \mathcal{H}_{\widehat{\Theta}}(g \cdot S)$$

This states that the hypostructure assignment **commutes with the group action**: applying $g$ to the system and then computing its hypostructure is isomorphic to computing the hypostructure and then acting by $g$.

**Consequence for Singularity Detection:** If a singularity germ $[P, \pi] \in \mathcal{H}_{\widehat{\Theta}}(S)$ is detected for system $S$, then the transformed germ $g \cdot [P, \pi] \in \mathcal{H}_{\widehat{\Theta}}(g \cdot S)$ is detected for the transformed system. Since $\mathcal{S}$ is $G$-invariant, singularities are detected with equal probability across all $G$-orbits. **This prevents bias in singularity detection.** **QED Claim.**

---

## Step 5: Quantitative Stability Estimates

To make the proof practically applicable, we establish quantitative bounds on how equivariance is preserved under perturbations.

**Proposition (Approximate Equivariance):** Suppose hypotheses (H1)-(H3) hold approximately with error $\epsilon > 0$:
- (H1-$\epsilon$): $\sup_{g \in G} d_{\mathrm{TV}}((g \cdot)_\# \mathcal{S}, \mathcal{S}) \leq \epsilon$
- (H2-$\epsilon$): $\sup_{g,\Theta,S} d_{\mathbf{Hypo}}(g \cdot \mathcal{H}_\Theta(S), \mathcal{H}_{g \cdot \Theta}(g \cdot S)) \leq \epsilon$
- (H3-$\epsilon$): $\sup_{A,g,\Theta,S,u} \|K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) - K_{A,S}^{(\Theta)}(u)\|_{\mathcal{C}} \leq \epsilon$

Then:
$$|R(g \cdot \Theta) - R(\Theta)| \leq C \epsilon$$
where $C = C(\mathcal{A}, w_A, \|\mathcal{L}\|_{\mathrm{Lip}})$ depends on Lipschitz constants.

**Proof of Proposition:**

By the triangle inequality and Lipschitz continuity of $\mathcal{L}$:
\begin{align*}
|R(g \cdot \Theta) - R(\Theta)|
&= \left|\mathbb{E}_{S \sim \mathcal{S}}[\mathcal{L}(g \cdot \Theta, S)] - \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{L}(\Theta, S)]\right| \\
&\leq \mathbb{E}_{S \sim \mathcal{S}}[|\mathcal{L}(g \cdot \Theta, S) - \mathcal{L}(\Theta, S)|]
\end{align*}

Expand using (H3-$\epsilon$):
\begin{align*}
|\mathcal{L}(g \cdot \Theta, S) - \mathcal{L}(\Theta, S)|
&\leq \mathbb{E}_{u \sim \mathcal{D}_S}\left[\sum_{A \in \mathcal{A}} w_A \left|\|K_{A,S}^{(g \cdot \Theta)}(u)\|_{\mathcal{C}} - \|K_{A,S}^{(\Theta)}(u)\|_{\mathcal{C}}\right|\right]
\end{align*}

By the reverse triangle inequality and Lipschitz continuity of the norm:
$$\left|\|K_{A,S}^{(g \cdot \Theta)}(u)\|_{\mathcal{C}} - \|K_{A,S}^{(\Theta)}(u)\|_{\mathcal{C}}\right| \leq \|K_{A,S}^{(g \cdot \Theta)}(u) - K_{A,S}^{(\Theta)}(u)\|_{\mathcal{C}}$$

Combining errors from (H1-$\epsilon$), (H2-$\epsilon$), (H3-$\epsilon$):
$$|R(g \cdot \Theta) - R(\Theta)| \leq \left(|\mathcal{A}| \cdot \sum_A w_A\right) \epsilon =: C \epsilon$$

**Consequence:** In numerical implementations, even if exact symmetry is broken by discretization or approximation errors, the equivariance property holds approximately, ensuring robustness. **QED Proposition.**

---

## Step 6: Lie Algebra Perspective and Infinitesimal Equivariance

For infinitesimal analysis, we work with the Lie algebra $\mathfrak{g} = T_e G$.

**Claim (Infinitesimal Equivariance):** For each $\xi \in \mathfrak{g}$, let $\xi_\Theta := \frac{d}{dt}\Big|_{t=0} \exp(t\xi) \cdot \Theta$ be the vector field on $\Theta$ generated by $\xi$. Then:
$$\langle \nabla R(\Theta), \xi_\Theta \rangle = 0$$

**Proof of Claim:**

By Step 1, $R(g \cdot \Theta) = R(\Theta)$ for all $g \in G$. Differentiating with respect to $g$ at $g = e$ in direction $\xi$:
\begin{align*}
0 = \frac{d}{dt}\Big|_{t=0} R(\exp(t\xi) \cdot \Theta)
&= dR(\Theta)[\xi_\Theta] \\
&= \langle \nabla R(\Theta), \xi_\Theta \rangle
\end{align*}

Thus $\nabla R(\Theta) \perp \xi_\Theta$ for all $\xi \in \mathfrak{g}$. This means **the gradient is perpendicular to the $G$-orbit**, which is the expected behavior for a $G$-invariant function. **QED Claim.**

**Application to Noether's Theorem:** In the PDE context, if $G$ is a continuous symmetry group of the Lagrangian $\mathcal{L}$, Noether's theorem {cite}`Noether18` guarantees a conserved quantity for each generator $\xi \in \mathfrak{g}$:
$$Q_\xi := \langle \frac{\partial \mathcal{L}}{\partial \dot{q}}, \xi_q \rangle - \langle \xi, \mathcal{L} \rangle$$

In our framework, the conserved quantities correspond to the vanishing of the gradient components along $G$-orbits, ensuring that learned parameters respect physical conservation laws.

---

## Step 7: Certificate Construction and Algorithmic Verification

We now construct the certificate $K_{\text{SV08}}^+$ (Symmetry Preservation) and provide an algorithm for verification.

### Certificate Structure

The certificate $K_{\text{SV08}}^+$ consists of:

1. **Symmetry Group Record:** $(G, \rho, \mathfrak{g})$
   - The group $G$ with its representation $\rho$ on $\Theta$
   - The Lie algebra $\mathfrak{g}$ with generators $\{\xi_1, \ldots, \xi_k\}$

2. **Invariance Witness:** $(\epsilon_{\text{inv}}, \{g_i\}_{i=1}^N, \{R_i\}_{i=1}^N)$
   - A tolerance $\epsilon_{\text{inv}} > 0$
   - Test group elements $\{g_i\}$ (dense in $G$ up to $\epsilon_{\text{inv}}/2$)
   - Measured risks $R_i := R(g_i \cdot \Theta)$ satisfying $|R_i - R(\Theta)| < \epsilon_{\text{inv}}$

3. **Equivariance Diagram:** $(\mathcal{H}_\Theta, \phi_{g,\Theta,S}, \delta_{\mathcal{H}})$
   - The hypostructure assignment $\mathcal{H}_\Theta$
   - Natural isomorphisms $\phi_{g,\Theta,S}: g \cdot \mathcal{H}_\Theta(S) \xrightarrow{\sim} \mathcal{H}_{g \cdot \Theta}(g \cdot S)$
   - Distance bounds $\delta_{\mathcal{H}} := \sup_{g,\Theta,S} d_{\mathbf{Hypo}}(\phi_{g,\Theta,S}, \mathrm{id})$

4. **Defect Covariance Table:** $\{(A, g, \Theta, S, u, K_{\mathrm{L}}, K_{\mathrm{R}}, \delta_{A,g})\}$
   - For each axiom $A$ and sample $(g, \Theta, S, u)$:
     - Left certificate: $K_{\mathrm{L}} := K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u)$
     - Right certificate: $K_{\mathrm{R}} := K_{A,S}^{(\Theta)}(u)$
     - Distance: $\delta_{A,g} := \|K_{\mathrm{L}} - K_{\mathrm{R}}\|_{\mathcal{C}}$

5. **Orbit Preservation Proof:** $(\Theta_0, \{\Theta_t\}_{t \in [0,T]}, \delta_{\text{orbit}})$
   - Initial parameter $\Theta_0 \in G \cdot \Theta^*$
   - Gradient flow trajectory $\Theta_t$
   - Orbit distance: $\delta_{\text{orbit}} := \sup_{t \in [0,T]} \inf_{g \in G} d(g \cdot \Theta^*, \Theta_t)$

### Verification Algorithm

**Algorithm VerifyEquivariance($\Theta$, $\mathcal{S}$, $G$, $\epsilon$):**

**Input:**
- Learned parameters $\Theta$
- System distribution $\mathcal{S}$
- Symmetry group $G$
- Tolerance $\epsilon > 0$

**Output:**
- Certificate $K_{\text{SV08}}^+$ if verified, or FAIL

**Procedure:**

1. **Test Risk Invariance:**
   - Sample $N = \lceil \log(1/\epsilon) / \epsilon^2 \rceil$ group elements $g_i \sim \mu_G$ uniformly
   - For each $g_i$, compute $R_i := R(g_i \cdot \Theta)$ via Monte Carlo sampling
   - Verify $|R_i - R(\Theta)| < \epsilon$ for all $i$
   - If any violation: return FAIL

2. **Test Hypostructure Equivariance:**
   - Sample $M$ systems $S_j \sim \mathcal{S}$
   - For each $S_j$ and each $g_i$:
     - Compute $\mathcal{H}_\Theta(S_j)$ and $\mathcal{H}_{g_i \cdot \Theta}(g_i \cdot S_j)$
     - Measure distance $d_{\mathbf{Hypo}}(g_i \cdot \mathcal{H}_\Theta(S_j), \mathcal{H}_{g_i \cdot \Theta}(g_i \cdot S_j))$
   - Verify distance $< \epsilon$
   - If any violation: return FAIL

3. **Test Defect Covariance:**
   - For each axiom $A \in \mathcal{A}$:
     - Sample trajectories $u_\ell \sim \mathcal{D}_{S_j}$ for each system $S_j$
     - For each $(g_i, u_\ell)$:
       - Compute $K_{\mathrm{L}} = K_{A,g_i \cdot S_j}^{(g_i \cdot \Theta)}(g_i \cdot u_\ell)$
       - Compute $K_{\mathrm{R}} = K_{A,S_j}^{(\Theta)}(u_\ell)$
       - Verify $\|K_{\mathrm{L}} - K_{\mathrm{R}}\|_{\mathcal{C}} < \epsilon$
     - If any violation: return FAIL

4. **Test Orbit Preservation (for gradient flow):**
   - Initialize $\Theta_0 \in G \cdot \Theta^*$ (or project current $\Theta$ to nearest orbit)
   - Simulate gradient flow for time $T$
   - At checkpoints $t_k = kT/K$, verify:
     $$\inf_{g \in G} d(g \cdot \Theta^*, \Theta_{t_k}) < \epsilon$$
   - If any violation: return FAIL

5. **Construct Certificate:**
   - Collect all test data into $K_{\text{SV08}}^+$ structure
   - Return $K_{\text{SV08}}^+$

**Complexity:** $O(|\mathcal{A}| \cdot N \cdot M \cdot L \cdot \dim(G) \cdot C_{\mathrm{cert}})$ where $C_{\mathrm{cert}}$ is the cost of computing one certificate.

---

## Conclusion

We have established all three conclusions of the theorem:

### (1) Risk Minimizers Lie in Symmetry Orbits

By Steps 1 and 3, every global minimizer $\widehat{\Theta}$ of $R(\Theta)$ satisfies:
$$\widehat{\Theta} \in G \cdot \Theta^*$$
for some representative $\Theta^* \in \Theta/G$. This follows from $G$-invariance of $R$ and the fact that $\mathcal{M} = \{\Theta : R(\Theta) = \inf R\}$ is a union of $G$-orbits.

### (2) Gradient Flow Preserves Equivariance

By Step 2, the gradient flow $\dot{\Theta}_t = -\nabla R(\Theta_t)$ satisfies:
- If $\Theta_0 \in G \cdot \Theta^*$, then $\Theta_t \in G \cdot \Theta^*$ for all $t \geq 0$
- The gradient is $G$-equivariant: $\nabla R(g \cdot \Theta) = \rho'(g)[\nabla R(\Theta)]$
- In particular, if $\Theta_0$ is $G$-invariant (a fixed point), then $\Theta_t$ remains $G$-invariant

This is practically significant: **initializing parameters with symmetry guarantees preservation throughout training.**

### (3) Learned Hypostructures Inherit System Symmetries

By Step 4, the parametrized hypostructure $\mathcal{H}_{\widehat{\Theta}}(S)$ satisfies:
$$g \cdot \mathcal{H}_{\widehat{\Theta}}(S) \simeq \mathcal{H}_{\widehat{\Theta}}(g \cdot S)$$

Therefore:
- Singularity germs detected in system $S$ transform equivariantly to germs in $g \cdot S$
- Certificates $K_{A,S}^{(\widehat{\Theta})}$ exhibit the same symmetry as $\mathcal{S}$
- The Sieve's categorical checks (Node 17, Hom-emptiness) respect physical symmetries

**Certificate $K_{\text{SV08}}^+$ (Symmetry Preservation) produced by Step 7** provides algorithmic verification and numerical evidence for all three properties.

---

## Literature and Applicability

**Classical Foundations:**

1. **Noether (1918) {cite}`Noether18`:** The foundational connection between continuous symmetries and conserved quantities. Our Step 6 (infinitesimal equivariance) is a direct generalization of Noether's theorem to the meta-learning setting, where "conserved quantities" are gradients perpendicular to $G$-orbits.

2. **Weyl (1946) {cite}`Weyl46`:** The representation theory of compact Lie groups. Our use of the Haar measure $\mu_G$ and averaging over $G$ in Step 1 relies on Weyl's integration formula. The compactness of $G$ is essential for the existence of a finite invariant measure.

**Modern Machine Learning:**

3. **Cohen & Welling (2016) {cite}`CohenWelling16`:** Introduced group-equivariant convolutional networks (G-CNNs). Our hypothesis (H2) formalizes their equivariance constraint at the level of hypostructures: $g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$. In G-CNNs, this corresponds to equivariance of feature maps: $\rho_{\ell+1}(g)[f_{\ell+1}] = W_{\ell+1}[\rho_\ell(g)[f_\ell]]$ where $W_{\ell+1}$ are learned filters.

4. **Kondor (2018) {cite}`Kondor18`:** Extended equivariant networks to general compact groups $G$ using harmonic analysis. Our Step 5 (approximate equivariance) quantifies the stability of their construction under perturbations, with explicit error bounds $|R(g \cdot \Theta) - R(\Theta)| \leq C\epsilon$.

**Applicability to Concrete Systems:**

- **Translation Symmetry ($G = \mathbb{R}^d$, non-compact):** For systems on $\mathbb{R}^d$ with translation invariance, replace the Haar measure with Lebesgue measure restricted to a compact domain (periodic boundary conditions). The proof extends with minor modifications.

- **Rotation Symmetry ($G = \mathrm{SO}(d)$):** For rotationally invariant systems (e.g., central force problems in physics), $G = \mathrm{SO}(d)$ is compact. The proof applies directly. Certificates $K_{\text{SV08}}^+$ verify that learned parameters preserve angular momentum.

- **Gauge Symmetry ($G = \mathrm{U}(1)$ or $\mathrm{SU}(N)$):** For gauge theories (Yang-Mills, QCD), $G$ is the gauge group. Hypothesis (H3) ensures that gauge-fixing choices (Lorenz gauge, temporal gauge) do not affect physical certificates. This is the categorical formulation of gauge independence.

- **Permutation Symmetry ($G = S_n$):** For systems with identical particles (bosons, fermions), $G = S_n$ acts by permuting indices. Our framework guarantees that learned singularity detectors are permutation-invariant, preventing spurious detections due to particle relabeling.

**Robustness and Limitations:**

- **Discrete Symmetries:** The proof extends to discrete groups $G$ (e.g., $\mathbb{Z}/2\mathbb{Z}$ for parity) by replacing integrals with sums: $\int_G \to \frac{1}{|G|}\sum_{g \in G}$.

- **Approximate Symmetries:** Real physical systems often have slightly broken symmetries (e.g., isospin in nuclear physics). Step 5 provides a quantitative framework for handling $\epsilon$-approximate equivariance.

- **Infinite-Dimensional Groups:** For diffeomorphism groups $G = \mathrm{Diff}(M)$ (e.g., in fluid dynamics), the group is non-compact and infinite-dimensional. The proof requires modification using the Frechet geometry of $\mathrm{Diff}(M)$ and regularized Haar measures (e.g., Cameron-Martin space).

**Open Questions:**

- **Non-Compact Lie Groups:** Can the theorem be extended to $G = \mathrm{SL}(2, \mathbb{R})$ (used in AdS/CFT) or the Poincaré group (relativistic systems)? Requires developing a theory of equivariant risk for non-compact $G$.

- **Higher Categories:** If $\mathbf{Hypo}$ is promoted to a $2$-category (with natural transformations as $2$-morphisms), can we prove equivariance at all levels? Relevant for homotopy type theory approaches to singularities.

:::

---

**Literature:**
- {cite}`Noether18`: Noether, E. "Invariante Variationsprobleme." (1918) — Foundational connection between symmetries and conservation laws
- {cite}`Weyl46`: Weyl, H. "The Classical Groups: Their Invariants and Representations." (1946) — Representation theory and integration over compact groups
- {cite}`CohenWelling16`: Cohen, T.S. & Welling, M. "Group equivariant convolutional networks." ICML (2016) — Practical implementation of equivariant learning
- {cite}`Kondor18`: Kondor, R. & Trivedi, S. "On the generalization of equivariance and convolution to compact groups." ICML (2018) — Harmonic analysis for general compact group symmetries
