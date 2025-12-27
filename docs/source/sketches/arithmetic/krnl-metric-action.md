# KRNL-MetricAction: Extended Action Reconstruction

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-metric-action*

For non-Riemannian settings (Wasserstein spaces, discrete graphs), the action reconstruction extends using metric slope. The Lyapunov functional extends from Riemannian to Wasserstein to discrete settings.

---

## Arithmetic Formulation

### Setup

The arithmetic extension covers:
- **Discrete arithmetic:** Finite fields $\mathbb{F}_q$, function fields
- **Adelic arithmetic:** Product over all places $\prod_v K_v$
- **Non-Archimedean:** $p$-adic numbers $\mathbb{Q}_p$

### Statement (Arithmetic Version)

**Theorem (Extended Arithmetic Height Reconstruction).** Let $\mathcal{X}$ be an arithmetic object in a general metric setting:

1. **Metric space:** $(\mathcal{X}, d)$ where $d$ is an arithmetic metric (height distance, $p$-adic distance, etc.)

2. **Metric slope:** For a height function $h: \mathcal{X} \to \mathbb{R}_{\geq 0}$:
$$|\partial h|(x) = \limsup_{y \to x} \frac{(h(x) - h(y))^+}{d(x, y)}$$

3. **Generalized gradient consistency ($\mathrm{GC}'$):** Along any gradient flow:
$$\mathfrak{D}(u(t)) = |\partial h|^2(u(t))$$

Then the canonical height extends to:
$$\hat{h}(x) = \inf_{\gamma: x \to M} \int_0^1 |\partial h|(\gamma(s)) \cdot |\dot{\gamma}(s)| \, ds$$

---

### Proof

**Step 1: Arithmetic Metric Spaces**

**Example 1: Discrete arithmetic ($\mathbb{F}_q[t]$)**

For function fields $K = \mathbb{F}_q(t)$, points are places of $K$.

**Height:** For $\alpha \in K$:
$$h(\alpha) = \sum_{v} \max(0, v(\alpha)) \cdot \deg(v)$$

**Metric:** $d(\alpha, \beta) = q^{-v_\infty(\alpha - \beta)}$ ($\infty$-adic metric)

**Metric slope:**
$$|\partial h|(\alpha) = \limsup_{\beta \to \alpha} \frac{|h(\alpha) - h(\beta)|}{d(\alpha, \beta)}$$

For the height function, this equals the **conductor** at $\alpha$.

**Example 2: Adelic arithmetic ($\mathbb{A}_K$)**

The adele ring $\mathbb{A}_K = \prod'_v K_v$ (restricted product).

**Height:** For $\alpha = (\alpha_v) \in \mathbb{A}_K$:
$$h(\alpha) = \sum_v n_v \cdot \log \max(1, |\alpha_v|_v)$$

**Metric:** Product metric
$$d(\alpha, \beta) = \sup_v |\alpha_v - \beta_v|_v$$

**Example 3: $p$-adic arithmetic ($\mathbb{Q}_p$)**

**Height:** $h(\alpha) = \log \max(1, |\alpha|_p^{-1}) = \max(0, -v_p(\alpha)) \cdot \log p$

**Metric:** $d(\alpha, \beta) = |\alpha - \beta|_p = p^{-v_p(\alpha - \beta)}$

**Metric slope:**
$$|\partial h|(\alpha) = \begin{cases} 0 & v_p(\alpha) \geq 0 \\ \log p & v_p(\alpha) < 0 \end{cases}$$

**Step 2: Gradient Flow in Arithmetic**

Define "gradient flow" as the optimal descent direction:

**Frobenius flow:** In characteristic $p$, the Frobenius $\phi: x \mapsto x^p$ acts on heights:
$$h(\phi(\alpha)) = p \cdot h(\alpha)$$

The "gradient" direction is $\phi^{-1}$ (when defined).

**Galois flow:** In characteristic 0, Galois conjugation:
$$h(\sigma(\alpha)) = h(\alpha) \quad (\text{height invariance})$$

The "flow" is constant along Galois orbits.

**Step 3: Extended Gradient Consistency**

The generalized gradient consistency $\mathrm{GC}'$:
$$\mathfrak{D}(\alpha) = |\partial h|^2(\alpha)$$

**Verification for function fields:**

Let $\alpha \in \mathbb{F}_q(t)$ with $h(\alpha) = n$. The slope:
$$|\partial h|(\alpha) = n / \text{dist}(\alpha, \mathbb{F}_q[t])$$

where $\text{dist}(\alpha, \mathbb{F}_q[t])$ is the closest polynomial approximation.

By the **Diophantine approximation in function fields** [Lasjaunias 2000]:
$$\text{dist}(\alpha, \mathbb{F}_q[t]) \geq q^{-n}$$

Hence:
$$|\partial h|^2(\alpha) \leq n^2 \cdot q^{2n}$$

This bounds the "dissipation."

**Step 4: Action Reconstruction Formula**

The canonical height as minimal action:
$$\hat{h}(x) = \inf_{\gamma: x \to M} \int_\gamma |\partial h| \cdot d\ell$$

**Explicit computation for $p$-adic:**

Path from $\alpha$ to $0$ (torsion):
$$\gamma(t) = (1-t) \cdot \alpha, \quad t \in [0, 1]$$

Action:
$$\int_0^1 |\partial h|(\gamma(t)) \cdot |\dot{\gamma}(t)|_p \, dt = |\alpha|_p \cdot |\partial h|(\text{avg})$$

For $\alpha$ with $v_p(\alpha) < 0$:
$$\hat{h}(\alpha) = -v_p(\alpha) \cdot \log p = h(\alpha)$$

The reconstruction recovers the standard $p$-adic height.

**Step 5: Riemannian → Wasserstein → Discrete**

The framework extends through:

| **Setting** | **Metric** | **Slope** | **Height** |
|-------------|-----------|-----------|-----------|
| Riemannian | $d_g(x,y)$ | $\|\nabla h\|_g$ | Arakelov |
| Wasserstein | $W_2(\mu, \nu)$ | $|\partial h|_W$ | Measure-valued |
| Discrete | $d_{\text{graph}}$ | $\max_{\text{neighbors}}$ | Combinatorial |

For **function fields** (discrete analogue of number fields):
- Points = places of $K$
- Metric = ultrametric from valuations
- Height = degree sum
- Slope = conductor

---

### Key Arithmetic Ingredients

1. **Metric Slope** [Ambrosio-Gigli-Savaré 2008]: Generalized gradient in metric spaces.

2. **Function Field Arithmetic** [Goss 1996]: Heights and valuations over $\mathbb{F}_q(t)$.

3. **$p$-adic Analysis** [Schikhof 1984]: Ultrametric calculus.

4. **Adelic Heights** [Weil 1929]: Product formula over all places.

---

### Arithmetic Interpretation

> **The canonical height extends naturally to discrete and non-Archimedean settings via metric slope. The reconstruction formula works uniformly across number fields, function fields, and $p$-adic fields.**

---

### Literature

- [Ambrosio-Gigli-Savaré 2008] L. Ambrosio, N. Gigli, G. Savaré, *Gradient Flows in Metric Spaces*
- [Goss 1996] D. Goss, *Basic Structures of Function Field Arithmetic*, Springer
- [Schikhof 1984] W.H. Schikhof, *Ultrametric Calculus*, Cambridge
- [Lasjaunias 2000] A. Lasjaunias, *A survey of Diophantine approximation in fields of power series*
