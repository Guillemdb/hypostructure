# FACT-Instantiation: Instantiation Metatheorem — GMT Translation

## Original Statement (Hypostructure)

The instantiation metatheorem establishes that every valid abstract permit configuration can be realized by concrete geometric/analytic data.

## GMT Setting

**Abstract Permit:** $\Pi = (K_{D_E}, K_{C_\mu}, K_{\text{SC}_\lambda}, K_{\text{LS}_\sigma}, \ldots)$

**Concrete Data:** $(M, g, T, \Phi, \mathfrak{D})$ — manifold, metric, current, energy, dissipation

**Instantiation:** $\mathcal{I}: \Pi \to (M, g, T, \Phi, \mathfrak{D})$

## GMT Statement

**Theorem (Instantiation Metatheorem).** For every consistent abstract permit configuration $\Pi$, there exists a valid instantiation:

1. **Existence:** $\exists \mathcal{I}: \Pi \to \text{GMT-Data}$ satisfying all permit requirements

2. **Uniqueness (up to equivalence):** Instantiations are unique modulo isometry/gauge

3. **Functoriality:** Permit morphisms induce instantiation morphisms

## Proof Sketch

### Step 1: Consistency of Permits

**Consistency Definition:** A permit configuration $\Pi$ is **consistent** if:
- No contradictory requirements
- All prerequisites for each certificate are satisfiable
- Energy bounds are compatible

**Verification:** Check:
1. $K_{D_E}^+$ requires dissipation $\mathfrak{D} \geq 0$
2. $K_{C_\mu}^+$ requires mass bound $\mathbf{M} \leq \Lambda$
3. $K_{\text{SC}_\lambda}^+$ requires homogeneous blow-ups
4. $K_{\text{LS}_\sigma}^+$ requires analytic energy

**Compatible:** These conditions are mutually compatible (no logical contradiction).

### Step 2: Existence via Construction

**Constructive Instantiation:** Given consistent $\Pi$, build:

1. **Ambient Space:** $M = \mathbb{R}^n$ or compact Riemannian manifold
2. **Current:** $T \in \mathbf{I}_k(M)$ with $\mathbf{M}(T) \leq \Lambda$
3. **Energy:** $\Phi(T) = \mathbf{M}(T)$ (mass functional)
4. **Dissipation:** $\mathfrak{D}(T) = |\nabla \Phi|^2(T)$

**Verification:** Check each certificate:
- $K_{D_E}^+$: Energy-dissipation identity holds for gradient flow
- $K_{C_\mu}^+$: Federer-Fleming compactness applies
- $K_{\text{SC}_\lambda}^+$: Tangent cones are homogeneous
- $K_{\text{LS}_\sigma}^+$: Mass functional is analytic on smooth currents

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.

### Step 3: Universal Instantiation

**Universal Example:** The space $(\mathbf{I}_k(\mathbb{R}^n), d_{\text{flat}}, \mathbf{M})$ instantiates all consistent permits.

**Universality:** Any other instantiation embeds into this one:
$$\mathcal{I}'(\Pi) \hookrightarrow (\mathbf{I}_k(\mathbb{R}^n), d_{\text{flat}}, \mathbf{M})$$

via Nash embedding theorem (for manifolds) and current embedding.

**Reference:** Nash, J. (1956). The imbedding problem for Riemannian manifolds. *Ann. of Math.*, 63, 20-63.

### Step 4: Uniqueness up to Equivalence

**Equivalence of Instantiations:** $\mathcal{I}_1 \sim \mathcal{I}_2$ if:
$$\exists \text{ isometry } g: M_1 \to M_2 \text{ with } g_\# T_1 = T_2$$

**Theorem:** If $\mathcal{I}_1, \mathcal{I}_2$ both instantiate $\Pi$:
$$\mathcal{I}_1 \sim \mathcal{I}_2$$

*Proof:* The permit data determines:
- Topology (via $K_{C_\mu}^+$)
- Local geometry (via $K_{\text{SC}_\lambda}^+$)
- Dynamics (via $K_{D_E}^+$)

These determine the instantiation up to isometry.

### Step 5: Functoriality

**Morphism of Permits:** $\phi: \Pi_1 \to \Pi_2$ is a map preserving:
- Certificate structure
- Compatibility relations
- Bound orderings

**Induced Morphism:** Given $\phi: \Pi_1 \to \Pi_2$:
$$\mathcal{I}(\phi): \mathcal{I}(\Pi_1) \to \mathcal{I}(\Pi_2)$$

is a geometric map respecting all structure.

**Proof:** The instantiation functor:
$$\mathcal{I}: \text{Permits} \to \text{GMT-Data}$$

is constructed by mapping permit morphisms to Lipschitz maps between current spaces.

### Step 6: Specific Instantiations

**Example 1: Minimal Surfaces**

Permits: $K_{D_E}^+$ (area decreasing), $K_{C_\mu}^+$ (compactness), $K_{\text{LS}_\sigma}^+$ (analyticity)

Instantiation:
- $M = \mathbb{R}^n$
- $T$ = area-minimizing current
- $\Phi = \mathbf{M}$ (area)
- $\mathfrak{D} = |H|^2$ (mean curvature squared)

**Example 2: Ricci Flow**

Permits: Perelman's entropy monotonicity, no local collapsing, surgery admissibility

Instantiation:
- $M$ = 3-manifold
- $g(t)$ = Riemannian metrics evolving by Ricci flow
- $\Phi = \mathcal{W}$ (Perelman's entropy)
- $\mathfrak{D} = |\text{Ric} + \nabla^2 f|^2$

**Reference:** Perelman, G. (2002). The entropy formula for the Ricci flow. arXiv:math/0211159.

**Example 3: Mean Curvature Flow**

Permits: Huisken's monotonicity, type I/II classification, surgery

Instantiation:
- $M_t \subset \mathbb{R}^{n+1}$ = evolving hypersurfaces
- $T = [M_t]$ (current associated to surface)
- $\Phi = \text{Area}(M_t)$
- $\mathfrak{D} = \int |H|^2$

**Reference:** Huisken, G. (1984). Flow by mean curvature of convex surfaces. *J. Diff. Geom.*, 20, 237-266.

### Step 7: Moduli of Instantiations

**Moduli Space:** Define:
$$\mathcal{M}(\Pi) := \{\mathcal{I}(\Pi)\} / \sim$$

(instantiations modulo equivalence).

**Finite Dimensionality:** Under soft permits:
$$\dim \mathcal{M}(\Pi) < \infty$$

**Proof:** By rigidity theorems (FACT-SoftRigidity), the moduli is discrete or finite-dimensional.

### Step 8: Obstruction Theory

**Non-Instantiability:** Some abstract configurations cannot be instantiated:

1. **Topological Obstruction:** Requested topology doesn't embed in $\mathbb{R}^n$
2. **Metric Obstruction:** Requested curvature bounds incompatible
3. **Energy Obstruction:** Requested energy configuration unstable

**Detection:** Check for obstructions before attempting instantiation.

### Step 9: Compilation Theorem

**Theorem (Instantiation):** The instantiation metatheorem:

1. **Inputs:** Consistent permit configuration $\Pi$
2. **Outputs:** Geometric instantiation $\mathcal{I}(\Pi)$
3. **Guarantees:**
   - Existence for consistent $\Pi$
   - Uniqueness up to isometry
   - Functorial in permit morphisms

**Constructive Content:**
- Algorithm to construct instantiation from permits
- Verification that constructed data satisfies permits
- Equivalence checking between instantiations

### Step 10: Meta-Level Significance

**Soundness:** If $\Pi$ is abstractly derivable, $\mathcal{I}(\Pi)$ is geometrically realizable.

**Completeness:** If geometric data $(M, T, \Phi)$ satisfies permits, it arises from some $\Pi$.

**Correspondence:**
$$\text{Abstract Permits} \xleftrightarrow{\mathcal{I}} \text{Concrete GMT Data}$$

is a bijection on equivalence classes.

## Key GMT Inequalities Used

1. **Federer-Fleming Compactness:**
   $$\sup \mathbf{M}(T_j) < \infty \implies \text{convergent subsequence}$$

2. **Nash Embedding:**
   $$(M^n, g) \hookrightarrow \mathbb{R}^{n + n(n+1)/2}$$ isometrically

3. **Tangent Cone Uniqueness:**
   $$\text{Isolated singularities have unique tangent cones}$$

4. **Rigidity:**
   $$\text{Permits determine geometry up to isometry}$$

## Literature References

- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Nash, J. (1956). The imbedding problem for Riemannian manifolds. *Ann. of Math.*, 63.
- Perelman, G. (2002). The entropy formula for the Ricci flow. arXiv:math/0211159.
- Huisken, G. (1984). Flow by mean curvature. *J. Diff. Geom.*, 20.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
