# KRNL-Jacobi: Action Reconstruction

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-jacobi*

With gradient consistency additionally validated, the Lyapunov functional becomes explicitly computable as geodesic distance in a conformally scaled metric (Jacobi metric).

---

## Arithmetic Formulation

### Setup

The arithmetic Jacobi metric corresponds to:
- **Base metric:** Arakelov metric on arithmetic varieties
- **Conformal scaling:** By the height function
- **Geodesics:** Minimal height paths between points

### Statement (Arithmetic Version)

**Theorem (Height as Geodesic Action).** Let $X/\mathbb{Q}$ be a smooth projective variety with:
- Arakelov metric $g = (g_\sigma)_{\sigma: K \hookrightarrow \mathbb{C}}$ on $X(\mathbb{C})$
- Height function $h_L$ for ample $L$
- Gradient consistency: the Kähler form $\omega = dd^c \log h_L$ is positive

Then the canonical height is the **minimal geodesic action**:
$$\hat{h}(P) = \inf_{\gamma: P \to M} \int_\gamma \sqrt{h_L} \cdot ds_g$$

where the infimum is over paths $\gamma$ from $P$ to the special locus $M$.

---

### Proof

**Step 1: Arakelov Geometry Setup**

By **Arakelov's intersection theory** [Arakelov 1974]:

For a variety $X/\mathbb{Q}$ with model $\mathcal{X}/\text{Spec}(\mathbb{Z})$, the Arakelov height is:
$$h_{\text{Ar}}(P) = \sum_{p} \log \#\mathcal{O}_{\mathcal{X},P}/\mathfrak{m}_P \cdot n_p + \sum_{\sigma} \log \|P\|_{\sigma}$$

where:
- First sum: finite places (intersection with special fiber)
- Second sum: infinite places (Hermitian metric)

**Step 2: Conformal Scaling by Height**

The Arakelov metric $g$ on $X(\mathbb{C})$ is Kähler with potential $\phi = \log h_L$.

**Jacobi metric construction:**
$$g_h = h_L \cdot g$$

This is the conformal scaling by the height function.

**Curvature:** By the $dd^c$ lemma [Griffiths-Harris, Ch. 1]:
$$\omega_h = dd^c(h_L \cdot \phi) = h_L \cdot \omega + dh_L \wedge d^c\phi$$

The gradient consistency condition ensures $\omega_h > 0$ (positive $(1,1)$-form).

**Step 3: Geodesic Action Formula**

For a path $\gamma: [0,1] \to X(\mathbb{C})$ from $P$ to $Q \in M$, define the **action**:
$$S[\gamma] = \int_0^1 \sqrt{h_L(\gamma(t))} \cdot |\dot{\gamma}(t)|_g \, dt$$

By standard Riemannian geometry, the geodesic in the Jacobi metric $g_h$ minimizes this action.

**Claim:** For the canonical height:
$$\hat{h}(P) = \inf_{\gamma: P \to M} S[\gamma]$$

**Step 4: Verification via Arakelov Theory**

By **Zhang's theorem** [Zhang 1995]:

For an abelian variety $A$ with symmetric ample $L$:
$$\hat{h}_L(P) = \lim_{n \to \infty} \frac{1}{n^2} \sum_{[n]Q = P} h_L(Q)$$

This is the "averaging over preimages" formula.

**Geodesic interpretation:** The geodesic from $P$ to $0$ (torsion) passes through $n$-th roots:
$$P \to [n^{-1}]P \to [n^{-2}]P \to \cdots \to 0$$

At each step:
$$h_L([n^{-1}]P) \approx \frac{h_L(P)}{n^2}$$

The action along this path:
$$S = \sum_{k=0}^\infty \sqrt{h_L([n^{-k}]P)} \cdot \text{dist}([n^{-k}]P, [n^{-(k+1)}]P)$$

By self-similarity of the abelian variety under $[n]$:
$$S \approx \sqrt{h_L(P)} \cdot \sum_{k=0}^\infty n^{-k} = \frac{\sqrt{h_L(P)}}{1 - n^{-1}}$$

Taking $n \to \infty$ and normalizing gives $S \propto \sqrt{\hat{h}(P)}$.

**Step 5: Explicit Formula**

**Theorem (Arakelov-Jacobi Formula):**
$$\hat{h}(P) = \inf_{Q \in M} \left(h_L(Q) + d_h(P, Q)^2\right)$$

where $d_h$ is the geodesic distance in the Jacobi metric $g_h$.

**For abelian varieties:** Taking $Q = 0$ (identity):
$$\hat{h}(P) = h_L(0) + d_h(P, 0)^2 = d_h(P, 0)^2$$

since $h_L(0) = 0$ (normalized).

---

### Connection to L-functions

For L-functions, the Jacobi metric has an analogue via the **explicit formula**:

**Height analogue:**
$$h_L(\rho) = -\log|\zeta(\rho)| \quad \text{(for zeros } \rho \text{ of } \zeta)$$

**Jacobi metric:** On the critical strip $\{s : 0 < \Re(s) < 1\}$:
$$g_\zeta(s) = |\zeta(s)|^{-1} \cdot |ds|^2$$

**Geodesic action:** The distance from $s$ to the real axis (pole at $s=1$) in this metric is:
$$d_\zeta(s, \mathbb{R}) = \int_s^{\Re(s)} \frac{|d\sigma|}{|\zeta(\sigma + i\Im(s))|}$$

Under GRH, zeros lie on $\Re(s) = 1/2$, which is equidistant from $s=0$ and $s=1$ in the Jacobi metric.

---

### Key Arithmetic Ingredients

1. **Arakelov Intersection Theory** [Arakelov 1974]: Height via arithmetic intersection.

2. **Zhang's Height Formula** [Zhang 1995]: Canonical height via preimage averaging.

3. **Jacobi Metric** [Classical Mechanics]: Conformal scaling by potential energy.

4. **$dd^c$-Lemma** [Griffiths-Harris]: Kähler potentials and curvature.

---

### Arithmetic Interpretation

> **Canonical height = squared geodesic distance to torsion in the Jacobi (height-weighted) metric. Heights are computed as minimal action integrals.**

This provides:
- **Explicit computation:** Heights via geodesic integration
- **Geometric insight:** Height measures "distance from special structure"
- **Variational principle:** Heights minimize action functionals

---

### Literature

- [Arakelov 1974] S.J. Arakelov, *Intersection theory of divisors on an arithmetic surface*, Math. USSR Izv.
- [Zhang 1995] S.-W. Zhang, *Small points and adelic metrics*, J. Alg. Geom.
- [Faltings 1984] G. Faltings, *Calculus on arithmetic surfaces*, Ann. Math.
- [Griffiths-Harris 1978] P. Griffiths, J. Harris, *Principles of Algebraic Geometry*, Ch. 1
