# ACT-Surgery: Structural Surgery Principle — GMT Translation

## Original Statement (Hypostructure)

The structural surgery principle shows how to systematically modify configurations at singularities, cutting and regluing to achieve regularity while preserving essential structure.

## GMT Setting

**Surgery:** Topological modification at singular points

**Cut-and-Paste:** Remove neighborhood, replace with regular piece

**Structure Preservation:** Mass, boundary, homology preserved up to control

## GMT Statement

**Theorem (Structural Surgery Principle).** For $T \in \mathbf{I}_k(M)$ with isolated singularities:

1. **Identification:** Locate singular set $\Sigma = \text{sing}(T)$

2. **Excision:** Remove tubular neighborhood $T \setminus T_\epsilon(\Sigma)$

3. **Replacement:** Insert standard model $T_{\text{std}}$ with matching boundary

4. **Control:** $\mathbf{M}(T') \leq \mathbf{M}(T) + C\epsilon$, $[\partial T'] = [\partial T]$

## Proof Sketch

### Step 1: Surgery in Differential Topology

**Classical Surgery:** Replace $S^k \times D^{n-k}$ with $D^{k+1} \times S^{n-k-1}$:
$$M' = (M \setminus \text{int}(S^k \times D^{n-k})) \cup_\phi (D^{k+1} \times S^{n-k-1})$$

**Reference:** Milnor, J. (1961). A procedure for killing the homotopy groups of differentiable manifolds. *Proc. Symp. Pure Math.*, 3, 39-55.

### Step 2: Perelman's Ricci Flow Surgery

**Surgery for Ricci Flow:** At neck singularities:
1. Identify $\epsilon$-neck (region close to $S^{n-1} \times \mathbb{R}$)
2. Cut along middle sphere
3. Glue in standard caps

**Reference:** Perelman, G. (2003). Ricci flow with surgery on three-manifolds. arXiv:math/0303109.

### Step 3: GMT Surgery

**For Currents:** Given $T \in \mathbf{I}_k(M)$ with singularity at $p$:

**Step 1:** Choose $r > 0$ small, define:
$$T_{\text{out}} = T \llcorner (M \setminus B_r(p))$$

**Step 2:** Boundary:
$$\partial(T \llcorner B_r(p)) = S_r \text{ (slice on sphere)}$$

**Step 3:** Find $R$ with $\partial R = S_r$ and $R$ regular.

### Step 4: Isoperimetric Estimate

**Deformation Theorem (Federer-Fleming):** For $S \in \mathbf{I}_{k-1}(M)$ with $\partial S = 0$:
$$\exists R \in \mathbf{I}_k(M): \partial R = S, \quad \mathbf{M}(R) \leq C \cdot \mathbf{M}(S)^{k/(k-1)}$$

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

**Application:** Fill slice $S_r$ with controlled mass.

### Step 5: Mass Control

**Surgery Mass Estimate:**
$$\mathbf{M}(T') = \mathbf{M}(T \llcorner (M \setminus B_r)) + \mathbf{M}(R)$$

$$\mathbf{M}(T') \leq \mathbf{M}(T) - \mathbf{M}(T \llcorner B_r) + C \cdot \mathbf{M}(S_r)^{k/(k-1)}$$

**Small $r$:** For $r$ small, surgery adds $O(r^k)$ mass.

### Step 6: Regularity of Replacement

**Standard Models:** Replace singularity with:
- Smooth minimal surface (for area-minimizing)
- Cone resolution (for tangent cone regularization)
- Product resolution (for product singularities)

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. Australian National University.

### Step 7: Homology Preservation

**Boundary:** $\partial T' = \partial T$ since surgery is internal.

**Homology:** $[T'] = [T]$ in $H_k(M)$ if surgery preserves relative position.

**Caveat:** Surgery may change homology if topology of region changes.

### Step 8: Huisken-Sinestrari MCF Surgery

**Mean Curvature Flow Surgery:** When singularity forms:
1. Detect cylindrical region
2. Cut neck
3. Cap with convex surfaces

**Reference:** Huisken, G., Sinestrari, C. (2009). Mean curvature flow with surgeries. *Invent. Math.*, 175, 137-221.

**Mass Control:** Each surgery adds controlled mass.

### Step 9: Iteration

**Theorem:** Finite surgeries suffice to regularize:

*Proof:*
1. Each surgery increases regularity or removes singularity
2. Energy/mass bounded
3. Each surgery has positive energy cost
4. Finitely many surgeries possible

### Step 10: Compilation Theorem

**Theorem (Structural Surgery Principle):**

1. **Identification:** Locate singularities via blow-up analysis

2. **Excision:** Remove controlled neighborhood

3. **Replacement:** Insert regular piece with matching boundary

4. **Control:** Mass increase bounded: $\Delta\mathbf{M} \leq C\epsilon^k$

**Applications:**
- Regularization of currents
- Continuation past singularities
- Resolution of geometric flows

## Key GMT Inequalities Used

1. **Isoperimetric:**
   $$\mathbf{M}(R) \leq C \cdot \mathbf{M}(\partial R)^{k/(k-1)}$$

2. **Mass Control:**
   $$\mathbf{M}(T') \leq \mathbf{M}(T) + C\epsilon^k$$

3. **Slice Bound:**
   $$\mathbf{M}(S_r) \leq C r^{k-1}$$

4. **Finite Surgeries:**
   $$N_{\text{surg}} \leq \mathbf{M}(T_0)/\delta$$

## Literature References

- Milnor, J. (1961). Killing homotopy groups. *Proc. Symp. Pure Math.*, 3.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Perelman, G. (2003). Ricci flow with surgery. arXiv:math/0303109.
- Huisken, G., Sinestrari, C. (2009). MCF with surgeries. *Invent. Math.*, 175.
- Simon, L. (1983). *Lectures on GMT*. ANU.
