# UP-SymmetryBridge: Symmetry-Gap Theorem — GMT Translation

## Original Statement (Hypostructure)

The symmetry-gap theorem shows that spectral gaps are preserved under symmetric perturbations, bridging local and global spectral control.

## GMT Setting

**Symmetry Group:** $G$ acting on state space

**Spectral Gap:** $\lambda_1 > 0$ — gap in linearized operator

**Bridge:** Symmetry constrains spectral structure

## GMT Statement

**Theorem (Symmetry-Gap Bridge).** If:

1. **$G$-Invariance:** Energy $\Phi$ is $G$-invariant: $\Phi(g \cdot T) = \Phi(T)$

2. **$G$-Equivariance:** Flow is $G$-equivariant: $g \cdot \varphi_t = \varphi_t \cdot g$

Then:
- **Spectrum Decomposition:** $\text{spec}(L) = \bigoplus_{\chi} \text{spec}(L|_{V_\chi})$ over irreps $\chi$
- **Gap Preservation:** Gap in fixed-point space implies global gap
- **Stability:** $G$-symmetric perturbations preserve gap

## Proof Sketch

### Step 1: Symmetric Energy

**$G$-Invariance:** For all $g \in G$:
$$\Phi(g \cdot T) = \Phi(T)$$

**Critical Points:** Critical set $\text{Crit}(\Phi)$ is $G$-invariant.

**Orbits:** Critical points form $G$-orbits: $G \cdot T_* \subset \text{Crit}(\Phi)$.

### Step 2: Equivariant Flow

**$G$-Equivariance:** The gradient flow satisfies:
$$\varphi_t(g \cdot T) = g \cdot \varphi_t(T)$$

*Proof:* $\nabla \Phi$ is $G$-equivariant when $\Phi$ is $G$-invariant.

**Consequence:** Symmetry is preserved along trajectories.

### Step 3: Representation Theory

**Irreducible Representations:** $G$ decomposes state space:
$$X = \bigoplus_{\chi \in \hat{G}} V_\chi \otimes W_\chi$$

where $V_\chi$ is the irrep and $W_\chi$ is the multiplicity space.

**Reference:** Serre, J.-P. (1977). *Linear Representations of Finite Groups*. Springer.

### Step 4: Spectral Decomposition

**Linearized Operator:** $L = \nabla^2 \Phi|_{T_*}$

**$G$-Equivariance of $L$:** $g L g^{-1} = L$ for $g \in G$.

**Schur's Lemma:** $L$ preserves each isotypic component:
$$L: V_\chi \to V_\chi$$

**Spectrum:** $\text{spec}(L) = \bigcup_\chi \text{spec}(L|_{V_\chi})$.

### Step 5: Fixed-Point Gap

**Fixed-Point Space:** $X^G = \{T : g \cdot T = T \text{ for all } g\}$

**Restricted Operator:** $L|_{X^G}$

**Gap in Fixed-Point Space:** If $\lambda_1(L|_{X^G}) \geq \lambda_0 > 0$:

**Bridge Theorem:** Gap in $X^G$ implies gap in non-trivial irreps:
$$\lambda_1(L|_{V_\chi}) \geq c(\chi) \lambda_0$$

for character-dependent constant $c(\chi)$.

### Step 6: Symmetric Perturbations

**$G$-Invariant Perturbation:** $\tilde{\Phi} = \Phi + \varepsilon \psi$ where $\psi$ is $G$-invariant.

**Gap Stability:** If $\lambda_1(L) \geq \lambda_0$:
$$\lambda_1(\tilde{L}) \geq \lambda_0 - C\varepsilon$$

for $\varepsilon$ small.

### Step 7: Branching Rules

**Subgroup Restriction:** For $H \subset G$:
$$V_\chi|_H = \bigoplus_{\psi \in \hat{H}} m_{\chi\psi} W_\psi$$

**Gap Inheritance:** Gap for $G$ implies gap for $H$:
$$\lambda_1(L|_{X^G}) \leq \lambda_1(L|_{X^H})$$

### Step 8: Symmetric Critical Points

**Palais Principle (1979):** If $G$ acts on $X$ and $\Phi$ is $G$-invariant:
$$\text{Crit}(\Phi) \cap X^G = \text{Crit}(\Phi|_{X^G})$$

**Reference:** Palais, R. S. (1979). The principle of symmetric criticality. *Comm. Math. Phys.*, 69, 19-30.

**Application:** Find critical points by restricting to symmetric subspace.

### Step 9: Gap Computation via Symmetry

**Algorithm:**
1. Identify symmetry group $G$
2. Decompose state space by irreps
3. Compute $\lambda_1(L|_{V_\chi})$ for each $\chi$
4. Global gap = $\min_\chi \lambda_1(L|_{V_\chi})$

**Efficiency:** Smaller matrices for each irrep.

### Step 10: Compilation Theorem

**Theorem (Symmetry-Gap Bridge):**

1. **Decomposition:** Spectrum decomposes by irreducible representations

2. **Fixed-Point Gap:** Gap in $X^G$ controls global gap

3. **Perturbation Stability:** $G$-symmetric perturbations preserve gap

4. **Computational:** Symmetry reduces spectral computation

**Applications:**
- Stability analysis with symmetry
- Pattern formation in symmetric systems
- Bifurcation with symmetry

## Key GMT Inequalities Used

1. **Schur Decomposition:**
   $$\text{spec}(L) = \bigcup_\chi \text{spec}(L|_{V_\chi})$$

2. **Fixed-Point Gap:**
   $$\lambda_1(L|_{X^G}) > 0 \implies \lambda_1(L) > 0$$

3. **Perturbation:**
   $$|\lambda_1(\tilde{L}) - \lambda_1(L)| \leq C\varepsilon$$

4. **Palais Principle:**
   $$\text{Crit}(\Phi) \cap X^G = \text{Crit}(\Phi|_{X^G})$$

## Literature References

- Serre, J.-P. (1977). *Linear Representations of Finite Groups*. Springer.
- Palais, R. S. (1979). The principle of symmetric criticality. *Comm. Math. Phys.*, 69.
- Golubitsky, M., Stewart, I. (2002). *The Symmetry Perspective*. Birkhäuser.
- Chossat, P., Lauterbach, R. (2000). *Methods in Equivariant Bifurcations and Dynamical Systems*. World Scientific.
