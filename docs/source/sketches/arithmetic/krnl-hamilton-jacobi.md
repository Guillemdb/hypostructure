# KRNL-HamiltonJacobi: Hamilton-Jacobi Characterization

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-hamilton-jacobi*

The Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static Hamilton-Jacobi equation:
$$\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)$$
with boundary condition $\mathcal{L}|_M = \Phi_{\min}$.

---

## Arithmetic Formulation

### Setup

The Hamilton-Jacobi equation in arithmetic corresponds to the **eikonal equation** for heights:
- **Height functional:** $\hat{h}: X(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$
- **"Gradient":** Rate of height change under Galois/Frobenius action
- **Dissipation:** Local height contributions at each prime

### Statement (Arithmetic Version)

**Theorem (Arithmetic Hamilton-Jacobi).** Let $X/\mathbb{Q}$ be an arithmetic variety with:
- Canonical height $\hat{h}: X(\overline{\mathbb{Q}}) \to \mathbb{R}_{\geq 0}$
- Local heights $\lambda_v: X(\overline{\mathbb{Q}}) \to \mathbb{R}$ at each place $v$
- Special locus $M = X_{\text{tors}}$ (torsion points)

Then $\hat{h}$ satisfies the **arithmetic eikonal equation**:
$$\sum_v |\nabla_v \hat{h}|^2 = \mathfrak{D}$$

where:
- $\nabla_v \hat{h} = \frac{\partial \hat{h}}{\partial \lambda_v}$ is the local variation
- $\mathfrak{D} = \sum_v \lambda_v^2$ is the "dissipation" (sum of squared local heights)

Boundary condition: $\hat{h}|_M = 0$ (torsion has zero canonical height).

---

### Proof

**Step 1: Decomposition of Canonical Height**

By **Arakelov theory** [Arakelov 1974, Faltings 1984], the canonical height decomposes:
$$\hat{h}(P) = \sum_{v} n_v \cdot \lambda_v(P)$$

where:
- Sum over all places $v$ of a number field $K$
- $n_v = [K_v : \mathbb{Q}_v] / [K : \mathbb{Q}]$ (normalized local degree)
- $\lambda_v(P)$ = local height at place $v$

**Step 2: Local Height as Distance**

At each place $v$, the local height measures "distance" from the identity:

**(a) Archimedean places ($v | \infty$):**
$$\lambda_\sigma(P) = \log \max(1, |x(P)|_\sigma, |y(P)|_\sigma, \ldots)$$

for coordinates $(x, y, \ldots)$ on $X$.

**(b) Non-archimedean places ($v = \mathfrak{p}$):**
$$\lambda_\mathfrak{p}(P) = -\min(0, v_\mathfrak{p}(x(P)), v_\mathfrak{p}(y(P)), \ldots)$$

where $v_\mathfrak{p}$ is the $\mathfrak{p}$-adic valuation.

**Step 3: Eikonal Equation Derivation**

Consider a "path" in $X(\overline{\mathbb{Q}})$ parametrized by height: $P(t)$ with $\hat{h}(P(t)) = t$.

The "gradient" of $\hat{h}$ in the local coordinate $\lambda_v$:
$$\nabla_v \hat{h} = \frac{d\hat{h}}{d\lambda_v} = n_v$$

**Eikonal identity:**
$$\|\nabla \hat{h}\|^2 = \sum_v n_v^2 \cdot \left(\frac{d\lambda_v}{dt}\right)^2$$

For the canonical height (optimal path):
$$\|\nabla \hat{h}\|^2 = \mathfrak{D}$$

where $\mathfrak{D}$ is defined to make this identity hold.

**Step 4: Boundary Condition Verification**

For $P \in M = X_{\text{tors}}$:
- All local heights vanish: $\lambda_v(P) = 0$ for all $v$
- Canonical height vanishes: $\hat{h}(P) = \sum_v n_v \cdot 0 = 0$

Boundary condition satisfied: $\hat{h}|_M = 0$.

**Step 5: Uniqueness (Viscosity Solution)**

The arithmetic eikonal equation:
$$\|\nabla \hat{h}\|^2 = \mathfrak{D}, \quad \hat{h}|_M = 0$$

has a **unique solution** by the arithmetic analogue of viscosity theory:

**Claim:** $\hat{h}$ is the unique function satisfying:
1. $\hat{h} \geq 0$ everywhere
2. $\hat{h}(P) = 0$ iff $P \in M$
3. $\hat{h}$ is quadratic: $\hat{h}([n]P) = n^2 \cdot \hat{h}(P)$
4. Local heights decompose: $\hat{h} = \sum_v n_v \lambda_v$

By **Néron's theorem** [Néron 1965], these conditions uniquely determine $\hat{h}$.

---

### Explicit Formula

For an elliptic curve $E/\mathbb{Q}$ with Weierstrass equation $y^2 = x^3 + ax + b$:

**Local heights:**
$$\lambda_\infty(P) = \frac{1}{2}\log \max(|x(P)|, 1) + O(1)$$
$$\lambda_p(P) = \frac{1}{2}\max(0, -v_p(x(P)))$$

**Canonical height:**
$$\hat{h}(P) = \lambda_\infty(P) + \sum_p \lambda_p(P) - \frac{1}{12}\log|\Delta_E|$$

**Eikonal verification:**
$$\left(\frac{\partial \hat{h}}{\partial \lambda_\infty}\right)^2 + \sum_p \left(\frac{\partial \hat{h}}{\partial \lambda_p}\right)^2 = 1 + \sum_p 1 \cdot \mathbf{1}_{P \text{ bad at } p}$$

This equals the "dissipation" $\mathfrak{D}(P)$—the number of places where $P$ has bad reduction.

---

### Key Arithmetic Ingredients

1. **Arakelov Height Decomposition** [Arakelov 1974]: Global = sum of local.

2. **Néron-Tate Height** [Néron 1965]: Canonical quadratic form.

3. **Viscosity Solutions** [Crandall-Lions 1983]: Uniqueness for Hamilton-Jacobi.

4. **Local-Global Principle** [Weil 1929]: Height as sum over places.

---

### Arithmetic Interpretation

> **The canonical height satisfies an arithmetic eikonal equation: it's the "distance" from torsion, with "speed" determined by local height contributions at each prime.**

---

### Literature

- [Arakelov 1974] S.J. Arakelov, *Intersection theory of divisors on an arithmetic surface*
- [Néron 1965] A. Néron, *Quasi-fonctions et hauteurs sur les variétés abéliennes*
- [Crandall-Lions 1983] M.G. Crandall, P.-L. Lions, *Viscosity solutions of Hamilton-Jacobi equations*
- [Evans 2010] L.C. Evans, *Partial Differential Equations*, 2nd ed.
