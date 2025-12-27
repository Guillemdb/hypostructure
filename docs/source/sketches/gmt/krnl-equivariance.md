# KRNL-Equivariance: Equivariance Principle — GMT Translation

## Original Statement (Hypostructure)

If the symmetry group $G$ acts equivariantly on the system distribution and parameter space, then learned parameters preserve symmetry, gradient flow is equivariant, and hypostructures inherit all symmetries.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — Riemannian manifold with isometry group $\text{Isom}(M, g)$

**Compact Lie Group:** $G \subset \text{Isom}(M, g)$ — closed subgroup acting properly on $M$

**Current Space:** $\mathbf{I}_k^G(M) := \{T \in \mathbf{I}_k(M) : g_\# T = T \text{ for all } g \in G\}$ — $G$-invariant currents

**Varifold Space:** $\mathbf{V}_k^G(M) := \{V \in \mathbf{V}_k(M) : g_\# V = V \text{ for all } g \in G\}$ — $G$-invariant varifolds

**Energy Functional:** $\mathcal{E}: \mathbf{I}_k(M) \to [0, \infty]$ — $G$-invariant (i.e., $\mathcal{E}(g_\# T) = \mathcal{E}(T)$)

## GMT Statement

**Theorem (Equivariance of Gradient Flows and Minimizers).** Let $G$ be a compact Lie group acting by isometries on $(M, g)$. Let $\mathcal{E}: \mathbf{I}_k(M) \to [0, \infty]$ be a $G$-invariant functional satisfying:

1. **(Lower Semicontinuity)** $\mathcal{E}$ is l.s.c. in the flat topology
2. **(Coercivity)** $\{T : \mathcal{E}(T) \leq \Lambda\}$ is compact in the flat topology
3. **($G$-Invariance)** $\mathcal{E}(g_\# T) = \mathcal{E}(T)$ for all $g \in G$, $T \in \mathbf{I}_k(M)$

Then:

**(A) Equivariant Minimizers:** Every minimizer of $\mathcal{E}$ in a $G$-invariant constraint class lies in a $G$-orbit of minimizers.

**(B) Equivariant Gradient Flow:** If $(T_t)_{t \geq 0}$ is a gradient flow of $\mathcal{E}$, then $(g_\# T_t)_{t \geq 0}$ is also a gradient flow, and $G$-invariant initial data yields $G$-invariant solutions.

**(C) Symmetric Tangent Cones:** If $T$ is $G$-invariant and $x_0$ is a $G$-fixed point, then every tangent cone $C \in \text{VarTan}(T, x_0)$ is $G$-invariant.

## Proof Sketch

### Step 1: Orbit Method for Minimizers (Part A)

Let $T^* \in \mathbf{I}_k(M)$ minimize $\mathcal{E}$ in the constraint class $\mathcal{C} \subset \mathbf{I}_k(M)$. Assume $\mathcal{C}$ is $G$-invariant: $g_\# \mathcal{C} = \mathcal{C}$ for all $g \in G$.

**Claim:** For any $g \in G$, $g_\# T^*$ is also a minimizer.

*Proof:* By $G$-invariance of $\mathcal{E}$:
$$\mathcal{E}(g_\# T^*) = \mathcal{E}(T^*) = \inf_{T \in \mathcal{C}} \mathcal{E}(T)$$

Since $g_\# T^* \in g_\# \mathcal{C} = \mathcal{C}$, we have $g_\# T^* \in \mathcal{C}$ with the same minimal energy.

**$G$-Orbit Structure:** The set of minimizers $\mathcal{M} = \{T \in \mathcal{C} : \mathcal{E}(T) = \inf \mathcal{E}\}$ is $G$-invariant. If $G$ acts transitively on $\mathcal{M}$, all minimizers are related by symmetry.

### Step 2: Averaging and Symmetrization

For a current $T \in \mathbf{I}_k(M)$, define the **$G$-average**:
$$T^G := \int_G g_\# T \, d\mu_G(g)$$

where $\mu_G$ is the normalized Haar measure on $G$.

**Claim:** $T^G \in \mathbf{I}_k^G(M)$ is $G$-invariant, and $\mathcal{E}(T^G) \leq \mathcal{E}(T)$ by convexity (if $\mathcal{E}$ is convex).

*Proof of $G$-Invariance:* For $h \in G$:
$$h_\# T^G = h_\# \int_G g_\# T \, d\mu_G(g) = \int_G (hg)_\# T \, d\mu_G(g) = \int_G g_\# T \, d\mu_G(g) = T^G$$

using left-invariance of Haar measure.

**Energy Inequality (Jensen):** If $\mathcal{E}$ is convex in an appropriate sense:
$$\mathcal{E}(T^G) = \mathcal{E}\left(\int_G g_\# T \, d\mu_G\right) \leq \int_G \mathcal{E}(g_\# T) \, d\mu_G = \mathcal{E}(T)$$

### Step 3: Equivariant Gradient Flow (Part B)

Let $\mathcal{E}$ define a metric gradient flow via the **Wasserstein-type metric** on currents:
$$d_{\text{flat}}(T_1, T_2) := \inf \{\mathbf{M}(S) : \partial S = T_1 - T_2, \, S \in \mathbf{I}_{k+1}(M)\}$$

The gradient flow satisfies, in the metric sense:
$$\frac{d}{dt} T_t = -\nabla_{\text{flat}} \mathcal{E}(T_t)$$

**Equivariance Claim:** If $T_0 \in \mathbf{I}_k^G(M)$ is $G$-invariant, then $T_t \in \mathbf{I}_k^G(M)$ for all $t > 0$.

*Proof:* Define $\tilde{T}_t := g_\# T_t$ for fixed $g \in G$. Then:
1. $\tilde{T}_0 = g_\# T_0 = T_0$ (since $T_0$ is $G$-invariant)
2. $\frac{d}{dt} \tilde{T}_t = g_\# \frac{d}{dt} T_t = -g_\# \nabla_{\text{flat}} \mathcal{E}(T_t)$

By $G$-invariance of $\mathcal{E}$, the gradient is equivariant:
$$g_\# \nabla_{\text{flat}} \mathcal{E}(T_t) = \nabla_{\text{flat}} \mathcal{E}(g_\# T_t) = \nabla_{\text{flat}} \mathcal{E}(\tilde{T}_t)$$

Thus $\tilde{T}_t$ also solves the gradient flow with initial data $\tilde{T}_0 = T_0$. By uniqueness, $\tilde{T}_t = T_t$, i.e., $g_\# T_t = T_t$.

### Step 4: Symmetric Tangent Cones (Part C)

Let $T \in \mathbf{I}_k^G(M)$ and $x_0 \in M$ be a $G$-fixed point: $g \cdot x_0 = x_0$ for all $g \in G$.

**Blow-Up Sequence:** For $r_j \to 0$, define:
$$T_j := (\eta_{x_0, r_j})_\# T$$

where $\eta_{x_0, r}(y) = \exp_{x_0}^{-1}(y) / r$ in normal coordinates.

**$G$-Equivariance of Blow-Up:** Since $G$ fixes $x_0$, the $G$-action on $T_{x_0}M \cong \mathbb{R}^n$ is linear. The pushforward satisfies:
$$g_\# T_j = g_\# (\eta_{x_0, r_j})_\# T = (\eta_{x_0, r_j})_\# (g_\# T) = (\eta_{x_0, r_j})_\# T = T_j$$

**Limit Inheritance:** By Federer-Fleming compactness, $T_j \to C$ for some tangent cone $C$. Taking the limit of $g_\# T_j = T_j$:
$$g_\# C = C$$

for all $g \in G$. Hence $C$ is $G$-invariant.

### Step 5: Noether-Type Conservation Laws

The $G$-invariance of $\mathcal{E}$ implies **conservation laws** along the gradient flow. For each Killing field $X$ generating a one-parameter subgroup of $G$:

**Momentum Map:** Define $J_X: \mathbf{I}_k(M) \to \mathbb{R}$ by:
$$J_X(T) := \int_T \iota_X \omega$$

where $\omega$ is an appropriate $k$-form (e.g., calibration).

**Conservation:** $\frac{d}{dt} J_X(T_t) = 0$ along gradient flow.

*Proof:* By the equivariance of the flow:
$$J_X(T_t) = J_X((\exp(sX))_\# T_t)\big|_{s=0} = J_X(T_t)$$

is independent of $t$ by chain rule and $G$-invariance.

## Key GMT Inequalities Used

1. **Haar Measure Integration (Averaging):**
   $$T^G = \int_G g_\# T \, d\mu_G \in \mathbf{I}_k^G(M)$$

2. **Jensen's Inequality for Convex Functionals:**
   $$\mathcal{E}(T^G) \leq \int_G \mathcal{E}(g_\# T) \, d\mu_G = \mathcal{E}(T)$$

3. **Uniqueness of Gradient Flows:**
   $$T_0 = \tilde{T}_0, \, \dot{T} = -\nabla\mathcal{E}(T), \, \dot{\tilde{T}} = -\nabla\mathcal{E}(\tilde{T}) \implies T_t = \tilde{T}_t$$

4. **Federer-Fleming Limit of Equivariant Sequence:**
   $$g_\# T_j = T_j \, \forall j \implies g_\# T_\infty = T_\infty$$

## Literature References

- Hsiang, W.-Y. (1967). On the compact homogeneous minimal submanifolds. *Proc. Nat. Acad. Sci.*, 56, 5-6.
- Lawson, H. B. (1970). Complete minimal surfaces in $S^3$. *Annals of Mathematics*, 92, 335-374.
- Noether, E. (1918). Invariante Variationsprobleme. *Nachr. D. König. Gesellsch. D. Wiss. Göttingen*, 235-257.
- Palais, R. (1979). The principle of symmetric criticality. *Comm. Math. Phys.*, 69, 19-30.
- Weyl, H. (1946). *The Classical Groups: Their Invariants and Representations*. Princeton University Press.
- Cohen, T., Welling, M. (2016). Group equivariant convolutional networks. *ICML*, 2990-2999.
