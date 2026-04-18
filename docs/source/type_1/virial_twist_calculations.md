# Virial and twist identity calculations for bounded ancient NS3D

Working notes. Two targets:
1. Gaussian virial identity for general bounded ancient solutions in self-similar variables.
2. Swirl-weighted energy identity for axisymmetric ancient solutions.

Goal: extract rigorous Liouville theorems for specific decay/structure classes.

## Setup

Bounded ancient solution of self-similar NS3D on $\mathbb{R}^3 \times \mathbb{R}$:
$$\partial_\tau V + (V \cdot \nabla)V + \nabla P = \Delta V - \tfrac{1}{2}(V + y \cdot \nabla V), \quad \nabla \cdot V = 0$$

with $\|V\|_{L^\infty(\mathbb{R}^3 \times \mathbb{R})} \leq M < \infty$.

The drift term $-\tfrac{1}{2}(V + y \cdot \nabla V)$ is the self-similar correction.

---

## Calculation 1: Gaussian virial identity

Define $\mathcal{E}(\tau) := \int |V(y,\tau)|^2 G(y)\,dy$ where $G(y) = e^{-|y|^2/4}$.

Note: $\nabla G = -\tfrac{y}{2} G$, $\Delta G = (\tfrac{|y|^2}{4} - \tfrac{3}{2})G$, $y \cdot \nabla G = -\tfrac{|y|^2}{2} G$.

Compute $\partial_\tau \mathcal{E}$:

$$\partial_\tau \mathcal{E} = 2 \int V \cdot \partial_\tau V \cdot G$$

Using the equation:
$$\partial_\tau V = \Delta V - (V \cdot \nabla)V - \nabla P - \tfrac{1}{2}(V + y \cdot \nabla V)$$

So:
$$\partial_\tau \mathcal{E} = 2 \int V \cdot [\Delta V - (V \cdot \nabla)V - \nabla P - \tfrac{1}{2}(V + y \cdot \nabla V)] G$$

### Term T1: $2\int V \cdot \Delta V \cdot G$

Integration by parts:
$$2\int V \cdot \Delta V \cdot G = -2\int |\nabla V|^2 G - 2\int \partial_i V_j \cdot V_j \cdot \partial_i G$$

For the second piece:
$$-2\int \partial_i V_j \cdot V_j \cdot \partial_i G = -\int \partial_i(V_j^2) \partial_i G = \int V_j^2 \Delta G = \int |V|^2 \Delta G$$

$$= \int |V|^2 (\tfrac{|y|^2}{4} - \tfrac{3}{2}) G$$

So T1 = $-2\int |\nabla V|^2 G + \int |V|^2 (\tfrac{|y|^2}{4} - \tfrac{3}{2}) G$.

### Term T2: $-2\int V \cdot (V \cdot \nabla)V \cdot G$

Use $V \cdot (V \cdot \nabla)V = \tfrac{1}{2}(V \cdot \nabla)|V|^2$:
$$-2\int V \cdot (V\cdot\nabla)V \cdot G = -\int (V \cdot \nabla)|V|^2 \cdot G = \int |V|^2 \nabla \cdot (VG) = \int |V|^2 V \cdot \nabla G$$

(using $\nabla \cdot V = 0$)

$$= \int |V|^2 V \cdot (-\tfrac{y}{2}) G = -\tfrac{1}{2}\int |V|^2 (V \cdot y) G$$

So T2 = $-\tfrac{1}{2}\int |V|^2 (V \cdot y) G$.

### Term T3: $-2\int V \cdot \nabla P \cdot G$

$$-2\int V \cdot \nabla P \cdot G = 2\int P \nabla \cdot (VG) = 2\int P (V \cdot \nabla G) = -\int P (V \cdot y) G$$

So T3 = $-\int P (V \cdot y) G$.

### Term T4: $-\int V \cdot (V + y \cdot \nabla V) G$ (the self-similar drift times 2 then halved)

Wait, re-check: the drift is $-\tfrac{1}{2}(V + y\cdot\nabla V)$, so the contribution to $\partial_\tau \mathcal{E}$ is:
$$-2 \cdot \tfrac{1}{2} \int V \cdot (V + y \cdot \nabla V) G = -\int V \cdot (V + y \cdot \nabla V) G$$

$$= -\int |V|^2 G - \int V \cdot (y \cdot \nabla V) G$$

For the second piece: $V \cdot (y \cdot \nabla V) = y_i V_j \partial_i V_j = \tfrac{1}{2} y_i \partial_i |V|^2 = \tfrac{1}{2} y \cdot \nabla |V|^2$.

$$-\int \tfrac{1}{2} (y \cdot \nabla |V|^2) G = \tfrac{1}{2} \int |V|^2 \nabla \cdot (y G) = \tfrac{1}{2} \int |V|^2 (3G + y \cdot \nabla G)$$

$$= \tfrac{3}{2} \int |V|^2 G + \tfrac{1}{2} \int |V|^2 (-\tfrac{|y|^2}{2}) G = \tfrac{3}{2}\int |V|^2 G - \tfrac{1}{4}\int |V|^2 |y|^2 G$$

So T4 = $-\int |V|^2 G + \tfrac{3}{2}\int |V|^2 G - \tfrac{1}{4}\int |V|^2 |y|^2 G = \tfrac{1}{2}\int |V|^2 G - \tfrac{1}{4}\int |V|^2 |y|^2 G$.

### Combining

$$\partial_\tau \mathcal{E} = T1 + T2 + T3 + T4$$

Quadratic $|V|^2$ terms:
From T1: $\int |V|^2 (\tfrac{|y|^2}{4} - \tfrac{3}{2}) G$
From T4: $\tfrac{1}{2}\int |V|^2 G - \tfrac{1}{4}\int |V|^2 |y|^2 G$

Sum of $|V|^2$ coefficients: $\tfrac{|y|^2}{4} - \tfrac{3}{2} + \tfrac{1}{2} - \tfrac{|y|^2}{4} = -1$

So quadratic terms collapse to $-\int |V|^2 G$.

Full identity:
$$\boxed{\partial_\tau \mathcal{E} = -2\int |\nabla V|^2 G - \int |V|^2 G - \tfrac{1}{2}\int |V|^2 (V \cdot y) G - \int P(V \cdot y) G}$$

### Sign analysis

The first two terms are strictly negative (dissipation + self-similar decay).

The nonlinear term $-\tfrac{1}{2}\int |V|^2 (V \cdot y) G$ has no definite sign. It's bounded:
$$|\tfrac{1}{2}\int |V|^2 (V \cdot y) G| \leq \tfrac{1}{2} \|V\|_{L^\infty} \int |V|^2 |y| G$$

By Cauchy-Schwarz: $\int |V|^2 |y| G \leq (\int |V|^2 G)^{1/2}(\int |V|^2 |y|^2 G)^{1/2}$... but this isn't quite what I want.

Actually simpler: $|y| G \leq C \cdot G^{1/2}$ on $\mathbb{R}^3$ (since $|y| e^{-|y|^2/4} \leq C e^{-|y|^2/8}$), so:
$$|\tfrac{1}{2}\int |V|^2 (V \cdot y) G| \leq \tfrac{C}{2} \|V\|_{L^\infty} \int |V|^2 G^{1/2} \cdot G^{1/2}$$

Hmm, need to be more careful. Let me redo:
$$|\tfrac{1}{2}\int |V|^2 (V \cdot y) G| \leq \tfrac{1}{2} \|V\|_{L^\infty}^3 \int |y| G \leq C M^3$$

This is a bound, not something that vanishes. Not helpful for showing $\mathcal{E} \to 0$.

Better: under the hypothesis that $\|V\|_{L^\infty} \leq M$ with $M$ sufficiently small, we have:
$$|\tfrac{1}{2}\int |V|^2 (V \cdot y) G| \leq \tfrac{1}{2} M \int |V|^2 |y| G \leq C M \mathcal{E}$$

Wait, that's not quite right either. Let me be careful.

$|V|^2 |V \cdot y| \leq |V|^3 |y|$. Then:
$$|\tfrac{1}{2}\int |V|^2 (V \cdot y) G| \leq \tfrac{1}{2} \int |V|^3 |y| G \leq \tfrac{M}{2} \int |V|^2 |y| G$$

And $|y| G \leq C G^{\alpha}$ for any $\alpha < 1$ (since the exponential decay dominates any polynomial). Actually, $|y| G(y) \leq C_1 \tilde G(y)$ where $\tilde G(y) = e^{-|y|^2/8}$, and $\tilde G \leq C_2 G^{1/2}$. So $|y| G \leq C G^{1/2}$ isn't quite right pointwise. Let me just bound directly:

$\int |V|^2 |y| G \leq \|V\|_{L^\infty} \int |V| |y| G \leq M \cdot (\int |V|^2 G)^{1/2} \cdot (\int |y|^2 G)^{1/2} = C M \mathcal{E}^{1/2}$

So: $|\tfrac{1}{2}\int |V|^2 (V\cdot y)G| \leq \tfrac{CM^2}{2} \mathcal{E}^{1/2}$.

This isn't a bound by $\mathcal{E}$ directly — it's by $\mathcal{E}^{1/2}$. Not useful for exponential decay.

Let me try a different bound:
$$|\tfrac{1}{2}\int |V|^2 (V\cdot y) G| \leq \tfrac{1}{2} \int |V|^3 |y| G$$

Using Young: $|V|^3 |y| \leq \tfrac{1}{2}|V|^4 + \tfrac{1}{2}|V|^2 |y|^2$. Then:
$$\leq \tfrac{1}{4}\int |V|^4 G + \tfrac{1}{4}\int |V|^2 |y|^2 G$$

Hmm, this produces $|V|^4$ which I can't control from $L^\infty$.

Better: $|V|^3 |y| \leq \tfrac{\epsilon}{2} |V|^2 |y|^2 / 2 + \tfrac{1}{2\epsilon} |V|^4$... still the $|V|^4$ problem.

Use $L^\infty$ bound directly: $|V|^3 \leq M \cdot |V|^2$, so $|V|^3 |y| \leq M |V|^2 |y|$. Then:
$$\tfrac{1}{2}\int |V|^3 |y| G \leq \tfrac{M}{2} \int |V|^2 |y| G$$

By Cauchy-Schwarz, $\int |V|^2 |y| G \leq (\int |V|^2 G)^{1/2}(\int |V|^2 |y|^2 G)^{1/2} = \mathcal{E}^{1/2} \cdot (\int |V|^2 |y|^2 G)^{1/2}$.

And $\int |V|^2 |y|^2 G \leq M^2 \int |y|^2 G = C M^2$, or bounding by the dissipation term... hmm.

Cleanest bound: $|y|^2 G \leq C G^{1/2}$ (since $|y|^2 e^{-|y|^2/4}$ is bounded), no wait, $|y|^2 e^{-|y|^2/4} \leq C e^{-|y|^2/8}$ and $e^{-|y|^2/8} \neq C G^{1/2} = C e^{-|y|^2/8}$. Actually $G^{1/2}(y) = e^{-|y|^2/8}$, so yes $|y|^2 G \leq C G^{1/2}$.

So $\int |V|^2 |y|^2 G \leq C \int |V|^2 G^{1/2} \leq C \|V\|_{L^\infty}^2 \int G^{1/2} = C M^2$ (since $G^{1/2}$ is integrable).

Therefore:
$$|\tfrac{1}{2}\int |V|^2 (V\cdot y)G| \leq \tfrac{CM^2}{2} \mathcal{E}^{1/2}$$

Hmm, still $\mathcal{E}^{1/2}$. Let me try yet another approach.

Actually, I want to get a bound by $\mathcal{E}$ times something small. Try:
$$\tfrac{1}{2}\int |V|^2 (V\cdot y)G \leq \tfrac{1}{2}\int |V|^2 |V| |y| G$$

Write $|V| |y| G = |V| \cdot |y|G$. Then $|y| G(y) \leq C G^{2/3}(y) = C e^{-|y|^2/6}$ (checking: $|y| e^{-|y|^2/4} \leq C e^{-|y|^2/6}$ requires $|y| \leq C e^{|y|^2/4 - |y|^2/6} = C e^{|y|^2/12}$, true).

So $\int |V|^2 |V| |y| G \leq \|V\|_{L^\infty} \int |V|^2 |y| G \leq CM \int |V|^2 G^{2/3}$... and $G^{2/3}(y) = e^{-|y|^2/6}$.

Hmm, but I want to bound by $\mathcal{E} = \int |V|^2 G$. Since $G^{2/3} \geq G$ (because $G \leq 1$), we have $\int |V|^2 G^{2/3} \geq \int |V|^2 G$. So this doesn't help.

### What the identity actually gives

Going in a different direction. From the identity:
$$\partial_\tau \mathcal{E} + \int |V|^2 G + 2\int|\nabla V|^2 G = -\tfrac{1}{2}\int |V|^2 (V\cdot y)G - \int P(V\cdot y) G$$

**Under small-data hypothesis $M \ll 1$:**

Using the nonlinear bound $|\tfrac{1}{2}\int|V|^2(V\cdot y)G| \leq CM^2 \mathcal{E}^{1/2}$ and crude pressure bound:
The pressure in self-similar variables: $P = P_{NS}$ satisfies $-\Delta P = \partial_i\partial_j(V_iV_j)$ on $\mathbb{R}^3$, so by Riesz:
$\|P\|_{L^\infty} \leq C\|V\|_{L^\infty}^2 = CM^2$.

Then $|\int P(V\cdot y) G| \leq CM^2 \int |V||y| G \leq CM^3 \int |y| G^{1/2} \leq CM^3$.

Hmm. The pressure term is bounded by $CM^3$, not by $\mathcal{E}$.

So the identity becomes:
$$\partial_\tau \mathcal{E} + \mathcal{E} + 2\mathcal{D} \leq CM^2 \mathcal{E}^{1/2} + CM^3$$

where $\mathcal{D} = \int |\nabla V|^2 G$.

For small $M$, RHS is small. But $\mathcal{E}$ for bounded $V$ is itself bounded: $\mathcal{E} \leq M^2 \int G \leq CM^2$. So $\mathcal{E}^{1/2} \leq CM$.

Then RHS $\leq CM^3 + CM^3 = C'M^3$, and LHS has $\mathcal{E} \leq CM^2$. For small $M$, $\mathcal{E}$ is uniformly controlled but the identity doesn't force $\mathcal{E} \to 0$.

**This suggests the Gaussian virial gives boundedness but not decay for the general case.** Not a small-data Liouville by this route.

Let me check: is there a way to get decay?

### Gronwall-type analysis

$\partial_\tau \mathcal{E} \leq -\mathcal{E} - 2\mathcal{D} + CM^2 \mathcal{E}^{1/2} + CM^3$

If $\mathcal{E}(\tau_0) = 0$ for some $\tau_0$, can we conclude $\mathcal{E} \equiv 0$?
From the above, $\partial_\tau \mathcal{E}|_{\mathcal{E}=0} \leq CM^3 \geq 0$, so we can't conclude.

What if we had a better bound on the RHS?

Try: control pressure differently. For axisymmetric, we might have stronger control.

For general bounded ancient, pressure Riesz estimate gives $|P - \bar P(\tau)| \leq C\|V\|_{L^\infty}^2$ pointwise only if $V$ has appropriate decay. Without decay, $P$ can include a constant-in-space term (the "pressure at infinity" in a sense).

Actually for bounded $V$ on $\mathbb{R}^3$, the Leray pressure is $P = R_iR_j(V_iV_j)$ which is bounded by $\|V\|_{L^\infty}^2$ pointwise via... wait, $R_iR_j$ is a bounded operator on $L^p$ for $p \in (1,\infty)$ but not on $L^\infty$. So pointwise control of $P$ from $V \in L^\infty$ alone fails.

This is a genuine subtlety: the pressure term $\int P(V \cdot y) G$ is not bounded by $M^3 \cdot C$ without additional structural hypothesis on $V$.

**Conclusion for Calculation 1: the Gaussian virial identity does not close to give a small-data Liouville without additional decay or structure hypotheses.**

This is actually consistent with what's known: small-data Liouville for bounded ancient NS3D is proved via mild formulation + Ornstein-Uhlenbeck semigroup decay, not via Gaussian virial identity. See e.g. Tsai's notes, or paper 2 of the ancient series.

---

## Calculation 2: Swirl-weighted energy identity

Now the axisymmetric swirl case. In cylindrical coordinates $(r, \theta, z)$, for axisymmetric $V$:
$V = V_r(r,z,\tau) e_r + V_\theta(r,z,\tau) e_\theta + V_z(r,z,\tau) e_z$

The swirl equation (self-similar form):
$$\partial_\tau V_\theta + V_r \partial_r V_\theta + V_z \partial_z V_\theta + \frac{V_r V_\theta}{r} = \Delta V_\theta - \frac{V_\theta}{r^2} - \frac{1}{2}V_\theta - \frac{1}{2}(r\partial_r + z\partial_z)V_\theta$$

where $\Delta = \partial_r^2 + \tfrac{1}{r}\partial_r + \partial_z^2$ is the axisymmetric Laplacian.

Let $\Gamma := r V_\theta$ (the "circulation"). The $\Gamma$ equation:

$\partial_\tau \Gamma = r \partial_\tau V_\theta + V_\theta \partial_\tau r = r \partial_\tau V_\theta$ (since $r$ doesn't depend on $\tau$ directly).

Actually in self-similar variables, $r$ is the spatial variable, so $\partial_\tau r = 0$. So:
$$\partial_\tau \Gamma = r\partial_\tau V_\theta$$

Substituting the swirl equation:
$\partial_\tau \Gamma = r[\Delta V_\theta - V_\theta/r^2 - V_r\partial_r V_\theta - V_z\partial_z V_\theta - V_r V_\theta/r - V_\theta/2 - (r\partial_r + z\partial_z)V_\theta/2]$

Let me compute each piece in terms of $\Gamma$:
- $V_\theta = \Gamma/r$, so $\partial_r V_\theta = (\partial_r\Gamma)/r - \Gamma/r^2$, and $\partial_z V_\theta = (\partial_z \Gamma)/r$.
- $r \Delta V_\theta$: $\Delta V_\theta = \partial_r^2 V_\theta + \tfrac{1}{r}\partial_r V_\theta + \partial_z^2 V_\theta - V_\theta/r^2$ (the $-V_\theta/r^2$ comes from the vector Laplacian in cylindrical coordinates).

Wait I need to be careful. The full scalar Laplacian in cylindrical coordinates is $\partial_r^2 + \tfrac{1}{r}\partial_r + \tfrac{1}{r^2}\partial_\theta^2 + \partial_z^2$. For axisymmetric functions ($\partial_\theta = 0$) the $\theta$-term drops. So $\Delta V_\theta = \partial_r^2 V_\theta + \tfrac{1}{r}\partial_r V_\theta + \partial_z^2 V_\theta$, but when $V_\theta$ is the $\theta$-component of a vector, the vector Laplacian adds a $-V_\theta/r^2$ term.

Hmm, I combined this into the swirl equation already. So the RHS of the swirl equation has $\Delta V_\theta - V_\theta/r^2$ where $\Delta$ is the *scalar* Laplacian. Let me verify by checking literature...

Actually the standard form: for axisymmetric with swirl,
$$\partial_t V_\theta + V_r \partial_r V_\theta + V_z \partial_z V_\theta + \frac{V_r V_\theta}{r} = (\Delta - \tfrac{1}{r^2})V_\theta$$
where $\Delta - 1/r^2$ applied to $V_\theta$ corresponds to $\partial_r^2 + \tfrac{1}{r}\partial_r - \tfrac{1}{r^2} + \partial_z^2$. This combines to $\partial_r[\tfrac{1}{r}\partial_r(r V_\theta)] + \partial_z^2 V_\theta$.

In terms of $\Gamma = r V_\theta$:
$\partial_r[\tfrac{1}{r}\partial_r \Gamma] = \partial_r(\Gamma/r)' = ...$

Let me just use the $\Gamma$ equation directly. Standard result: for axisymmetric NS with swirl,
$$\partial_t \Gamma + V_r \partial_r \Gamma + V_z \partial_z \Gamma = \Delta \Gamma - \frac{2}{r}\partial_r \Gamma$$

This is much cleaner. The $-2/r \cdot \partial_r \Gamma$ is the "modified" Laplacian that reflects the cylindrical geometry.

In self-similar variables, the $\Gamma$ equation becomes (let me derive):
Under $(y,\tau) \leftrightarrow (x,t)$ with $y = x/\sqrt{-t}$, $\tau = -\log(-t)$, and letting $R = r_y$ (cylindrical radius in self-similar variables):
$V(y,\tau) = \sqrt{-t} u(x,t)$, so $V_\theta(y,\tau) = \sqrt{-t} u_\theta(x,t)$.
$\Gamma_\text{ss}(y,\tau) = R V_\theta = R \cdot \sqrt{-t} u_\theta = \sqrt{-t} \cdot (R u_\theta)|_{x = y\sqrt{-t}}$

In physical variables, $r = R\sqrt{-t}$, so $r u_\theta = R\sqrt{-t} \cdot u_\theta = R \cdot \sqrt{-t} u_\theta = \Gamma_{ss}$.

So $\Gamma_{ss}(y,\tau) = \Gamma(x,t)|_{x=y\sqrt{-t}, t=-e^{-\tau}}$ — the circulation is invariant under this rescaling!

The self-similar equation for $\Gamma_{ss}$:
$$\partial_\tau \Gamma_{ss} + V_r \partial_R \Gamma_{ss} + V_z \partial_z \Gamma_{ss} = \Delta_y \Gamma_{ss} - \frac{2}{R}\partial_R \Gamma_{ss} - \frac{1}{2}(y \cdot \nabla_y)\Gamma_{ss}$$

(the $V/2$ drift term drops because $\Gamma_{ss}$ doesn't have scaling weight)

### Weighted $\Gamma$ energy

Define $\mathcal{G}(\tau) := \int \Gamma^2 w(y) \, dy$ for some weight $w > 0$ (integration over $\mathbb{R}^3$, where cylindrical $R = \sqrt{y_1^2 + y_2^2}$).

Compute:
$$\tfrac{1}{2}\partial_\tau \mathcal{G} = \int \Gamma \partial_\tau \Gamma \cdot w$$

$$= \int \Gamma [\Delta \Gamma - \tfrac{2}{R}\partial_R\Gamma - \tfrac{1}{2}(y\cdot\nabla)\Gamma - V_r \partial_R\Gamma - V_z\partial_z\Gamma] w$$

**Diffusion term:**
$\int \Gamma \Delta\Gamma \cdot w = -\int |\nabla\Gamma|^2 w + \tfrac{1}{2}\int \Gamma^2 \Delta w$ (by IBP twice).

**Cylindrical correction term:**
$-2\int \Gamma \tfrac{\partial_R \Gamma}{R} w = -\int \tfrac{\partial_R \Gamma^2}{R} w$

Integration by parts in $R$: need to be careful with the cylindrical volume element. $dy = R\, dR\, d\theta\, dz$. So $\int f \, dy = \int_0^\infty \int_{-\infty}^\infty f(R,z) \cdot 2\pi R \, dR\, dz$ for axisymmetric $f$.

Let $W(R,z) := 2\pi R \cdot w$ be the cylindrical density, so $\int f \, dy = \int f \cdot W \, dR\, dz$.

$-\int \tfrac{\partial_R \Gamma^2}{R} w \, dy = -\int \partial_R(\Gamma^2) \cdot \tfrac{w}{R} \cdot W \, dR\,dz = -\int \partial_R(\Gamma^2) \cdot 2\pi w \, dR\,dz$

IBP in $R$ (assume $\Gamma \to 0$ at $R=0$ and $R=\infty$):
$= \int \Gamma^2 \cdot 2\pi \partial_R w \, dR\,dz = \int \Gamma^2 \cdot \tfrac{\partial_R w}{R} \, dy$

Hmm wait, actually the sign: $-\int \partial_R(f) g \, dR = \int f \partial_R g \, dR$ assuming boundary terms vanish. But the density is $2\pi w$ (not $W = 2\pi R w$), so:
$-\int \partial_R(\Gamma^2) \cdot 2\pi w \, dR\,dz = +\int \Gamma^2 \cdot 2\pi \partial_R w \, dR\,dz - [\text{bdy at } R=0]$

Boundary at $R=0$: $\Gamma^2(0,z) \cdot 2\pi w(0,z)$. If $\Gamma(0,z) = 0$ (expected for smooth axisymmetric vector fields with $V_\theta$ finite on axis), this vanishes.

So: cylindrical correction = $\int \Gamma^2 \cdot \tfrac{2\pi \partial_R w}{W} W \, dR\,dz = \int \Gamma^2 \tfrac{\partial_R w}{R w} \cdot w \, dy$. That's messy. Let me just express as:

Cylindrical correction = $2\pi \int \Gamma^2 \partial_R w \, dR \, dz$

**Self-similar drift term:**
$-\tfrac{1}{2}\int \Gamma (y\cdot\nabla)\Gamma \cdot w = -\tfrac{1}{4}\int (y\cdot\nabla)\Gamma^2 \cdot w = \tfrac{1}{4}\int \Gamma^2 \nabla\cdot(yw) = \tfrac{1}{4}\int \Gamma^2 (3w + y\cdot\nabla w)$

**Transport term:**
$-\int \Gamma V_r \partial_R\Gamma \cdot w - \int \Gamma V_z \partial_z\Gamma \cdot w = -\tfrac{1}{2}\int V \cdot \nabla(\Gamma^2) \cdot w = \tfrac{1}{2}\int \Gamma^2 \nabla\cdot(Vw) = \tfrac{1}{2}\int \Gamma^2 V\cdot\nabla w$
(using $\nabla \cdot V = 0$)

### Putting it together

$$\tfrac{1}{2}\partial_\tau \mathcal{G} = -\int |\nabla\Gamma|^2 w + \tfrac{1}{2}\int \Gamma^2 \Delta w + 2\pi\int \Gamma^2 \partial_R w \, dR\,dz$$
$$+ \tfrac{1}{4}\int \Gamma^2(3w + y\cdot\nabla w) + \tfrac{1}{2}\int \Gamma^2 V\cdot\nabla w$$

### Choice of weight

Pick $w = G(y) = e^{-|y|^2/4}$ (Gaussian).
$\nabla w = -\tfrac{y}{2}w$, $\Delta w = (\tfrac{|y|^2}{4} - \tfrac{3}{2})w$, $y\cdot\nabla w = -\tfrac{|y|^2}{2}w$, $\partial_R w = -\tfrac{R}{2}w$ (since $|y|^2 = R^2 + z^2$).

Substituting:
- $\tfrac{1}{2}\int\Gamma^2\Delta w = \tfrac{1}{2}\int\Gamma^2(\tfrac{|y|^2}{4}-\tfrac{3}{2})w$
- $2\pi\int\Gamma^2 \partial_R w\, dR\,dz = 2\pi\int\Gamma^2 \cdot(-\tfrac{R}{2})w\, dR\,dz = -\tfrac{1}{2}\int\Gamma^2 w \, dy/R$ ... wait

Let me redo. $2\pi\int\Gamma^2 \partial_R w \, dR\,dz$ where the weight dropped the $R$ from the volume element. Converting back to full integral: $\int f \, dR\,dz = \int \tfrac{f}{2\pi R} \, dy$ for axisymmetric $f$. So:
$2\pi\int\Gamma^2 \partial_R w \, dR\,dz = \int \tfrac{\Gamma^2 \partial_R w}{R}\, dy = \int \tfrac{\Gamma^2 \cdot(-R/2)w}{R} dy = -\tfrac{1}{2}\int\Gamma^2 w\, dy$

- $\tfrac{1}{4}\int\Gamma^2(3w + y\cdot\nabla w) = \tfrac{1}{4}\int\Gamma^2(3w - \tfrac{|y|^2}{2}w) = \tfrac{3}{4}\int\Gamma^2 w - \tfrac{1}{8}\int\Gamma^2|y|^2 w$

- Nonlinear transport: $\tfrac{1}{2}\int\Gamma^2 V\cdot\nabla w = -\tfrac{1}{4}\int\Gamma^2 (V\cdot y)w$

### Summing the quadratic terms

Coefficient of $\Gamma^2 w$:
- from $\Delta w$: $-\tfrac{3}{4}$
- from cylindrical correction: $-\tfrac{1}{2}$
- from drift: $+\tfrac{3}{4}$
- net: $-\tfrac{1}{2}$

Coefficient of $\Gamma^2 |y|^2 w$:
- from $\Delta w$: $+\tfrac{1}{8}$
- from drift: $-\tfrac{1}{8}$
- net: $0$

### Final swirl identity

$$\boxed{\tfrac{1}{2}\partial_\tau \mathcal{G} = -\int|\nabla\Gamma|^2 G - \tfrac{1}{2}\int\Gamma^2 G - \tfrac{1}{4}\int\Gamma^2(V\cdot y)G}$$

Equivalently:
$$\partial_\tau \mathcal{G} + \mathcal{G} + 2\int|\nabla\Gamma|^2 G = -\tfrac{1}{2}\int\Gamma^2(V\cdot y)G$$

### Sign analysis

LHS: dissipation plus $\mathcal{G}$ (both nonnegative on the LHS is the "cost" side).

RHS: nonlinear transport term, no definite sign.

**Bound the RHS:**
$|\tfrac{1}{2}\int\Gamma^2(V\cdot y)G| \leq \tfrac{1}{2}\|V\|_{L^\infty}\int\Gamma^2|y|G$

And $|y|G \leq C G^{1/2}$... wait let me just bound by Cauchy-Schwarz:
$\int\Gamma^2|y|G \leq (\int\Gamma^2 G)^{1/2}(\int\Gamma^2|y|^2 G)^{1/2} = \mathcal{G}^{1/2}\cdot(\int\Gamma^2|y|^2 G)^{1/2}$

And $\int\Gamma^2|y|^2 G$ — this is controlled by $\|\Gamma\|_{L^\infty}^2 \cdot \int|y|^2 G \leq C\|\Gamma\|_{L^\infty}^2$.

But $\Gamma = RV_\theta$ doesn't have an immediate $L^\infty$ bound from $\|V\|_{L^\infty} \leq M$ — it grows linearly with $R$.

Hmm. This is an issue. $\Gamma \in L^\infty$ iff $RV_\theta \in L^\infty$, which requires either $V_\theta$ to decay like $1/R$ or $V_\theta$ bounded and $R$ bounded (compact support, which bounded ancient solutions need not have).

So the $\Gamma$-framework is useful only for solutions with $V_\theta = O(1/R)$ at infinity, i.e., swirl decaying like $1/r$ in physical variables.

**This matches Chen-Strain-Tsai-Yau 2008/2009!** Their controlled-swirl hypothesis is exactly $|u_\theta| \leq C/r$, i.e., $\Gamma = ru_\theta$ bounded. In self-similar variables this is $|V_\theta| \leq C/R$, i.e., $\Gamma_{ss}$ bounded.

So the weighted-$\Gamma$ identity is natural for the controlled-swirl class.

### Under controlled-swirl hypothesis $\|\Gamma\|_{L^\infty} \leq \Gamma_*$

$\int\Gamma^2|y|^2 G \leq \Gamma_*^2 \int|y|^2 G = C\Gamma_*^2$

$\int\Gamma^2|y| G \leq \Gamma_*^2 \int|y|G = C\Gamma_*^2$, or using Cauchy-Schwarz:
$\int\Gamma^2|y|G \leq \mathcal{G}^{1/2}\cdot(\int\Gamma^2|y|^2 G)^{1/2} \leq C\Gamma_* \mathcal{G}^{1/2}$

So RHS $\leq \tfrac{1}{2}M\cdot C\Gamma_* \mathcal{G}^{1/2} = CM\Gamma_* \mathcal{G}^{1/2}$.

Identity:
$$\partial_\tau \mathcal{G} + \mathcal{G} + 2\int|\nabla\Gamma|^2 G \leq CM\Gamma_*\mathcal{G}^{1/2}$$

**Solve this ODE inequality.** Let $\mathcal{G} = y^2$ (so $y = \mathcal{G}^{1/2}$):
$2y\partial_\tau y + y^2 \leq CM\Gamma_* y$
$2\partial_\tau y + y \leq CM\Gamma_*$
$\partial_\tau y + y/2 \leq CM\Gamma_*/2$

$y(\tau) \leq y(\tau_0) e^{-(\tau-\tau_0)/2} + CM\Gamma_*(1 - e^{-(\tau-\tau_0)/2})$

As $\tau_0 \to -\infty$ (ancient): first term vanishes if $y(\tau_0)$ is bounded (which it is, since $\mathcal{G}$ is bounded on bounded ancient).

So $y(\tau) \leq CM\Gamma_*$, i.e., $\mathcal{G}(\tau) \leq C^2 M^2 \Gamma_*^2$.

**This is a boundedness bound, not a vanishing bound.** Not yet a Liouville theorem.

### Pushing to Liouville

For Liouville, we need $\mathcal{G} \equiv 0$, i.e., $\Gamma \equiv 0$, i.e., no swirl.

The difficulty: the RHS transport term doesn't vanish unless either $V = 0$ or $\Gamma$ has additional structure.

**Key observation:** in a controlled-swirl setting with $V \to 0$ at spatial infinity, the transport $V\cdot y$ vanishes outside a bounded region. But the weight $G$ localizes to bounded region anyway.

Let me try: if additionally $M\Gamma_* < \epsilon$ for some small $\epsilon$, does the identity close to Liouville?

Specifically, using $|V\cdot y|G \leq \|V\|_{L^\infty}|y|G$ and $|y|G \leq C\cdot G^{2/3}$ (verify: $|y|e^{-|y|^2/4} \leq e^{-|y|^2/6}$ for $|y|^2 \geq 12\ln|y|$, approximate but ok):

$\tfrac{1}{2}\int\Gamma^2(V\cdot y)G \leq \tfrac{M}{2}\int\Gamma^2|y|G$

By Hölder with exponents 3 and 3/2:
$\int\Gamma^2|y|G = \int\Gamma^2 G^{2/3} \cdot |y|G^{1/3} \leq (\int|\Gamma|^3 G)^{2/3}(\int|y|^3 G)^{1/3}$

Hmm, this introduces $L^3$ of $\Gamma$ which we don't directly control.

Different approach: use the dissipation.
$\Gamma^2 \leq C(\epsilon)(|\nabla\Gamma|^2 + \epsilon^{-1}\Gamma^2 \cdot \text{weight ratio})$ — Poincaré-type. But on $\mathbb{R}^3$ without boundary we don't have Poincaré.

Hardy-type: for the Gaussian weight, $\int\Gamma^2 G \leq C\int|\nabla\Gamma|^2 G + C\int\Gamma^2 \cdot|\nabla G|^2/G = C\int|\nabla\Gamma|^2 G + C\int\Gamma^2 |y|^2 G/4$

Not obviously useful.

### What we actually get

The swirl identity
$$\partial_\tau \mathcal{G} + \mathcal{G} + 2\mathcal{D}_\Gamma \leq CM\Gamma_*\mathcal{G}^{1/2}$$

gives: **bounded ancient solutions with controlled swirl ($\Gamma \in L^\infty$) satisfy $\mathcal{G}(\tau) \leq CM^2\Gamma_*^2$ uniformly in $\tau$.**

This is *not* a Liouville theorem. It's a quantitative estimate showing that weighted swirl energy is controlled by the product of bounds.

**For actual Liouville**, we need to kill the RHS. Ways:
1. Add spatial decay of $V$ in $\infty$-direction — then $V\cdot y \to 0$ at infinity and integration against $G$ gives smaller bound.
2. Use small-data $M \leq M_*$ — then the $\mathcal{G}^{1/2}$ bound combined with the dissipation might close.
3. Different weight that makes the transport term dissipative.

Let me try (2). Under $M \leq M_*$ small, from $\mathcal{G} \leq C^2 M^2\Gamma_*^2$:
$\mathcal{G}^{1/2} \leq CM\Gamma_*$

RHS $\leq CM\Gamma_* \cdot CM\Gamma_* = C^2 M^2\Gamma_*^2$

Compared to LHS terms $\mathcal{G} \leq C^2 M^2\Gamma_*^2$. Same order! So for small $M$, boundedness but not forced decay.

Hmm. For decay, we'd need the RHS bound to be smaller than LHS terms.

**One more attempt:** notice that if $M\Gamma_* < 1/(2C)$, then the inequality
$\partial_\tau y + y/2 \leq CM\Gamma_*/2 = y_\infty/2$ where $y_\infty = CM\Gamma_*$
integrates to $y(\tau) \leq y_\infty + (y(\tau_0) - y_\infty) e^{-(\tau-\tau_0)/2}$

For ancient solutions, $y(\tau_0)$ as $\tau_0 \to -\infty$... but wait, the coefficient $(y(\tau_0) - y_\infty)$ doesn't have to be positive or go to anything special. For $\tau > \tau_0$:
$y(\tau) \leq y_\infty + |y(\tau_0) - y_\infty| e^{-(\tau-\tau_0)/2}$

If $y$ is bounded (which it is for bounded ancient), this gives $y(\tau) \leq y_\infty$ in the ancient limit by sending $\tau_0 \to -\infty$.

So $\mathcal{G}(\tau) \leq y_\infty^2 = C^2M^2\Gamma_*^2$. Same conclusion.

For $\mathcal{G} \to 0$ (i.e., Liouville), we'd need $y_\infty = 0$, which requires $M\Gamma_* = 0$.

**Conclusion for Calculation 2: The weighted-$\Gamma$ identity gives a boundedness estimate but not a Liouville theorem directly. To get Liouville, additional hypotheses or a different weight are needed.**

### What CAN be closed

The weighted-$\Gamma$ identity does give a concrete result:

**Proposition (provisional).** Let $V$ be a bounded axisymmetric ancient solution of the self-similar NS equation on $\mathbb{R}^3 \times \mathbb{R}$ with $\|V\|_{L^\infty} \leq M$. Suppose $\Gamma = RV_\theta \in L^\infty$ with $\|\Gamma\|_{L^\infty} \leq \Gamma_*$. Then for any $\tau \in \mathbb{R}$,
$$\int\Gamma^2(y,\tau)e^{-|y|^2/4}\,dy \leq C(M\Gamma_*)^2$$
where $C$ is a universal constant.

This is a quantitative a priori bound, useful as an input to other arguments. Not a Liouville theorem by itself.

### Where the actual swirl Liouville lives

Looking at the CSTY 2008/2009 papers, their approach to controlled-swirl uses:
- Maximum principle for $\Gamma$ (gives $\|\Gamma(\cdot,\tau)\|_{L^\infty} \leq \|\Gamma(\cdot,0)\|_{L^\infty}$)
- Exponential decay of $\Gamma$ from the drift-diffusion structure
- Combined with Biot-Savart for poloidal velocity

The weighted identity approach I tried is a different route and gives a weaker result. Closing swirl Liouville via weighted identities might require:
1. Weights singular at axis (Hardy-type) to exploit the $V_\theta/r^2$ viscous penalty.
2. Combined swirl-poloidal identities (coupled system).

---

## Honest summary

### Calculation 1 (Gaussian virial):
- Identity: $\partial_\tau\mathcal{E} + \mathcal{E} + 2\mathcal{D} = -\tfrac{1}{2}\int|V|^2(V\cdot y)G - \int P(V\cdot y)G$
- Does NOT close to Liouville for bounded ancient.
- Gives boundedness under small data, but small-data Liouville is known by better methods (mild formulation).
- **Honest conclusion: Gaussian virial not sufficient for general or small-data Liouville.**

### Calculation 2 (Weighted swirl):
- Identity: $\partial_\tau\mathcal{G} + \mathcal{G} + 2\mathcal{D}_\Gamma = -\tfrac{1}{2}\int\Gamma^2(V\cdot y)G$
- Under $\Gamma\in L^\infty$ (controlled swirl), gives $\mathcal{G}(\tau) \leq C(M\Gamma_*)^2$ uniformly.
- Does NOT close to Liouville.
- Known controlled-swirl Liouville (CSTY) uses different methods (maximum principle, not weighted energy).
- **Honest conclusion: weighted-$\Gamma$ identity gives a priori estimates but not Liouville; CSTY's approach is stronger for this class.**

### What is the actual status

Both calculations give **a priori bounds**, not **Liouville theorems**. The a priori bounds are real mathematical output but don't close the residual class of Paper 7.

For actual progress toward swirl Liouville beyond CSTY:
- Hardy-type weights near the axis to exploit $V_\theta/r^2$ cost.
- Combined poloidal-swirl weighted identities.
- Maximum principle refinements for $\Gamma$ in self-similar variables.

None of these is a "quick computation." They're research projects.

### What this does give you

- A clean a priori estimate for weighted swirl energy (Prop above), with explicit constant depending on $M\Gamma_*$.
- A precise diagnosis of why the weighted approach doesn't close Liouville for bounded ancient: the nonlinear transport term $\int \Gamma^2 (V \cdot y) G$ does not have a sign, and bounding it requires integrating against the Gaussian decay which matches the dissipation rate, so no margin for decay.
- A concrete research target: find a weight $w$ such that the analogous identity has margin. Candidate: power-weighted or inverse-power weighted, depending on the decay class.
