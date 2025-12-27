# FACT-SoftProfDec: Soft→ProfDec Compilation — GMT Translation

## Original Statement (Hypostructure)

The soft permits compile to a profile decomposition theorem: any bounded sequence decomposes into profiles plus remainder with orthogonality relations.

## GMT Setting

**Bounded Sequence:** $\{T_j\} \subset \mathbf{I}_k(M)$ with $\sup_j \mathbf{M}(T_j) \leq \Lambda$

**Profile Decomposition:**
$$T_j = \sum_{l=1}^L V^l_{j} + w_j^L$$

where $V^l_j$ are rescaled profiles and $w_j^L$ is remainder

**Orthogonality:** Profiles concentrate at asymptotically separated scales/locations

## GMT Statement

**Theorem (Soft→ProfDec Compilation).** Under soft permits, any bounded sequence $\{T_j\} \subset \mathbf{I}_k(M)$ admits profile decomposition:

$$T_j = \sum_{l=1}^L (\eta_{x_j^l, \lambda_j^l})_\# V^l + w_j^L$$

where:

1. **Profiles:** $V^l \in \mathbf{I}_k(\mathbb{R}^n)$ are non-trivial tangent cones

2. **Asymptotic Orthogonality:** For $l \neq m$:
$$\frac{\lambda_j^l}{\lambda_j^m} + \frac{\lambda_j^m}{\lambda_j^l} + \frac{|x_j^l - x_j^m|^2}{\lambda_j^l \lambda_j^m} \to \infty$$

3. **Remainder Decay:** $\|w_j^L\|_{L^p} \to 0$ as $L \to \infty$ then $j \to \infty$

4. **Mass Decoupling:**
$$\mathbf{M}(T_j) = \sum_{l=1}^L \mathbf{M}(V^l) + \mathbf{M}(w_j^L) + o(1)$$

## Proof Sketch

### Step 1: Concentration-Compactness Dichotomy

**Lions' Lemma (1984):** For bounded $\{u_j\} \subset \dot{H}^s(\mathbb{R}^n)$, one of three alternatives holds:

**(i) Compactness:** $\exists x_j$ such that $u_j(\cdot - x_j) \to u \neq 0$ strongly

**(ii) Vanishing:** $\sup_{y \in \mathbb{R}^n} \int_{B_R(y)} |u_j|^p \to 0$ for all $R > 0$

**(iii) Dichotomy:** Mass splits into two separated pieces

**Reference:** Lions, P.-L. (1984). The concentration-compactness principle. *Ann. Inst. H. Poincaré Anal. Non Linéaire*, 1, 109-145 and 223-283.

**GMT Version:** For $\{T_j\} \subset \mathbf{I}_k(M)$:
- (i) $T_j \to T$ in flat norm
- (ii) $\lim_j \sup_{x} \mathbf{M}(T_j \cap B_R(x)) = 0$
- (iii) $T_j = T_j^1 + T_j^2$ with separated supports

### Step 2: Profile Extraction Algorithm

**Iterative Extraction (Bahouri-Gérard, 1999):**

```
Initialize: R_j^0 = T_j, l = 0
While ||R_j^l||_{critical} > ε:
    l = l + 1
    Find concentration scale λ_j^l and point x_j^l
    Extract profile: V^l = lim_{j→∞} (η_{x_j^l, λ_j^l})_# R_j^{l-1}
    Update remainder: R_j^l = R_j^{l-1} - (η_{x_j^l, λ_j^l})_# V^l
Return: Profiles {V^l}, Remainder w_j^L = R_j^L
```

**Reference:** Bahouri, H., Gérard, P. (1999). High frequency approximation of solutions to critical nonlinear wave equations. *Amer. J. Math.*, 121, 131-175.

### Step 3: Orthogonality from Scale Separation

**Use of $K_{\text{SC}_\lambda}^+$:** Scale coherence ensures profiles live at distinct scales.

**Asymptotic Orthogonality Proof:** If $\lambda_j^l / \lambda_j^m \to c \in (0, \infty)$ and $|x_j^l - x_j^m|/\lambda_j^l \to 0$, then $V^l$ and $V^m$ would combine into a single profile, contradicting the extraction algorithm.

**Orthogonality Relation:** For $l \neq m$:
$$\lim_{j \to \infty} \left( \frac{\lambda_j^l}{\lambda_j^m} + \frac{\lambda_j^m}{\lambda_j^l} + \frac{|x_j^l - x_j^m|^2}{\lambda_j^l \lambda_j^m} \right) = \infty$$

### Step 4: Mass Decoupling via Locality

**Use of $K_{C_\mu}^+$:** Compactness ensures mass concentrates on rectifiable sets.

**Pythagorean Identity (GMT):** For orthogonal profiles:
$$\mathbf{M}(T_j) = \mathbf{M}\left(\sum_l V^l_j\right) + \mathbf{M}(w_j^L) = \sum_l \mathbf{M}(V^l_j) + \mathbf{M}(w_j^L) + o(1)$$

**Proof:** The supports of rescaled profiles become asymptotically disjoint:
$$\text{spt}(V^l_j) \cap \text{spt}(V^m_j) \to \emptyset$$

in Hausdorff distance as $j \to \infty$ for $l \neq m$.

### Step 5: Remainder Vanishing

**Critical Norm Decay:** The remainder $w_j^L$ satisfies:
$$\|w_j^L\|_{L^{p^*}} \to 0 \quad \text{as } L \to \infty \text{ then } j \to \infty$$

where $p^* = np/(n-k)$ is the critical Sobolev exponent.

**Proof:** Each extraction removes a quantum of critical norm:
$$\|R_j^{l-1}\|_{L^{p^*}} - \|R_j^l\|_{L^{p^*}} \geq c \cdot \|V^l\|_{L^{p^*}} > 0$$

Since $\sum_l \|V^l\|_{L^{p^*}} \leq C \cdot \sup_j \|T_j\|_{L^{p^*}} < \infty$, only finitely many profiles can exist.

### Step 6: Finite Profile Count

**Use of $K_{D_E}^+$:** Energy dissipation bounds the total mass:
$$\sum_{l=1}^\infty \mathbf{M}(V^l) \leq \limsup_j \mathbf{M}(T_j) \leq \Lambda$$

**Energy Gap (Struwe, 1984):** Each non-trivial profile has mass $\geq \varepsilon_0 > 0$.

**Reference:** Struwe, M. (1984). A global compactness result for elliptic boundary value problems involving limiting nonlinearities. *Math. Z.*, 187, 511-517.

**Profile Count Bound:**
$$L \leq \Lambda / \varepsilon_0$$

### Step 7: Profile Classification

**Use of $K_{\text{LS}_\sigma}^+$:** Each profile $V^l$ satisfies Łojasiewicz, hence is a critical point:
$$\nabla \Phi(V^l) = 0$$

**Consequence:** Profiles are:
- Tangent cones (blow-up limits)
- Solitons (traveling/scaling solutions)
- Bubble trees (hierarchical decomposition)

**Reference:** Lin, F., Rivière, T. (2002). A quantization property for static Ginzburg-Landau vortices. *CPAM*, 54, 206-228.

### Step 8: Compilation Theorem

**Theorem (Soft→ProfDec):** The compilation:
$$(K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+) \to \text{ProfDec}$$

is valid, producing:
- Finite list of profiles $\{V^1, \ldots, V^L\}$
- Concentration parameters $\{(x_j^l, \lambda_j^l)\}_{l,j}$
- Vanishing remainder $w_j^L$
- Mass decoupling identity

**Soundness:** Steps 1-7 above.

## Key GMT Inequalities Used

1. **Lions' Dichotomy:**
   $$\text{Compact or Vanishing or Dichotomy}$$

2. **Pythagorean Identity:**
   $$\mathbf{M}(T) = \sum_l \mathbf{M}(V^l) + \mathbf{M}(w^L) + o(1)$$

3. **Energy Gap:**
   $$V^l \neq 0 \implies \mathbf{M}(V^l) \geq \varepsilon_0$$

4. **Profile Count:**
   $$L \leq \Lambda / \varepsilon_0$$

## Literature References

- Lions, P.-L. (1984). Concentration-compactness I & II. *Ann. Inst. H. Poincaré*, 1.
- Bahouri, H., Gérard, P. (1999). High frequency approximation. *Amer. J. Math.*, 121.
- Struwe, M. (1984). Global compactness result. *Math. Z.*, 187.
- Lin, F., Rivière, T. (2002). Quantization property for Ginzburg-Landau. *CPAM*, 54.
- Kenig, C., Merle, F. (2006). Global well-posedness for energy-critical NLS. *Acta Math.*, 201.
- Tao, T. (2006). *Nonlinear Dispersive Equations*. CBMS Regional Conference Series.
