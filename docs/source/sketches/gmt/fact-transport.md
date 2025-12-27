# FACT-Transport: Equivalence + Transport Factory — GMT Translation

## Original Statement (Hypostructure)

The transport factory constructs equivalence relations and transport maps between isomorphic configurations, preserving relevant structure.

## GMT Setting

**Equivalence:** $T \sim S$ if $T = g \cdot S$ for some $g \in G$ (symmetry group)

**Transport Map:** $\tau_{T \to S}: \text{Data}(T) \to \text{Data}(S)$ — structure-preserving map

**Factory:** Constructs $\tau$ from geometric data

## GMT Statement

**Theorem (Transport Factory).** There exists a factory $\mathcal{F}_{\text{trans}}$ that, given:
- Currents $T, S \in \mathbf{I}_k(M)$ with $T \sim S$
- Equivalence witness $g \in G$ with $T = g \cdot S$
- Structure type $\mathfrak{S}$ to transport

produces transport map $\tau_{T \to S}$ with:

1. **Structure Preservation:** $\tau$ preserves $\mathfrak{S}$-structure

2. **Functoriality:** $\tau_{T \to T} = \text{id}$, $\tau_{S \to R} \circ \tau_{T \to S} = \tau_{T \to R}$

3. **Naturality:** $\tau$ commutes with flow: $\tau \circ \varphi_t = \varphi_t \circ \tau$

## Proof Sketch

### Step 1: Symmetry Group Structure

**Isometry Group (Myers-Steenrod, 1939):** For Riemannian $(M, g)$:
$$\text{Isom}(M, g) := \{f: M \to M : f^* g = g\}$$

is a Lie group.

**Reference:** Myers, S. B., Steenrod, N. E. (1939). The group of isometries of a Riemannian manifold. *Ann. of Math.*, 40, 400-416.

**Action on Currents:** For $g \in \text{Isom}(M)$:
$$(g)_\# T := g_* T$$

(pushforward of current).

### Step 2: Equivalence via Group Action

**Definition:** $T \sim_G S$ iff $\exists g \in G$ such that $(g)_\# T = S$.

**Equivalence Classes:** $[T]_G = \{(g)_\# T : g \in G\}$ (orbit of $T$)

**Quotient Space:** $\mathbf{I}_k(M) / G$ — moduli space of currents

**Reference:** Michor, P. W. (2008). *Topics in Differential Geometry*. AMS. [Chapter X]

### Step 3: Transport of Mass

**Mass Transport:** For $T \sim S$ via $g$:
$$\tau_{\mathbf{M}}: \mathbf{M}(T) \mapsto \mathbf{M}(S) = \mathbf{M}(T)$$

(mass is invariant under isometry).

**Density Transport:**
$$\Theta_k(S, g(x)) = \Theta_k(T, x)$$

### Step 4: Transport of Tangent Cones

**Tangent Cone Transport:** If $C_x$ is tangent cone of $T$ at $x$:
$$C_{g(x)}(S) = (dg)_\# C_x$$

**Proof:** By definition of tangent cone as blow-up limit, and functoriality of pushforward.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU. [Section 8]

### Step 5: Transport of Curvature

**Second Fundamental Form:** For surfaces $\Sigma \subset M$:
$$A_S(g(x)) = g_* A_T(x)$$

(pushforward of bilinear form).

**Mean Curvature:**
$$H_S(g(x)) = H_T(x)$$

(scalar invariant).

**Reference:** do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.

### Step 6: Transport of Homology

**Induced Map on Homology:** For $g: M \to M$:
$$g_*: H_k(T) \to H_k(S)$$

is an isomorphism.

**Naturality:** The diagram commutes:
```
H_k(T) --g_*--> H_k(S)
  |               |
  |               |
H_k(M) =========  H_k(M)
```

**Reference:** Spanier, E. H. (1966). *Algebraic Topology*. McGraw-Hill.

### Step 7: Transport of Regularity

**Regular Set Transport:**
$$\text{reg}(S) = g(\text{reg}(T))$$

**Singular Set Transport:**
$$\text{sing}(S) = g(\text{sing}(T))$$

**Stratification Transport:**
$$S^{(j)}(S) = g(S^{(j)}(T))$$

for each stratum dimension $j$.

### Step 8: Factory Construction

**Transport Factory Algorithm:**

```
TransportFactory(T, S, g, structure_type):
    # Verify equivalence
    assert (g)_# T = S

    # Select transport based on structure type
    if structure_type == MASS:
        return λx: M(T)  # mass is invariant

    if structure_type == TANGENT_CONE:
        return λx: (dg)_# C_x(T)

    if structure_type == CURVATURE:
        return λx: g_* A_T(g^{-1}(x))

    if structure_type == HOMOLOGY:
        return g_*: H_*(T) → H_*(S)

    if structure_type == REGULARITY:
        return {
            reg: g(reg(T)),
            sing: g(sing(T)),
            strata: [g(S^j(T)) for j]
        }

    if structure_type == ENERGY:
        return Φ(S) = Φ(T)  # energy is invariant

    return TRANSPORT_NOT_IMPLEMENTED
```

### Step 9: Optimal Transport Interpretation

**Wasserstein Transport (Villani, 2003):** The mass transport problem:
$$W_2(\mu_T, \mu_S)^2 = \inf_{\gamma} \int |x - y|^2 \, d\gamma(x, y)$$

where $\mu_T = \|T\|$, $\mu_S = \|S\|$.

**Reference:** Villani, C. (2003). *Topics in Optimal Transportation*. AMS.

**Transport Map:** If $T \sim S$ via isometry $g$:
$$\tau = g \text{ is optimal transport map}$$

with $W_2(\mu_T, \mu_S) = 0$.

### Step 10: Compilation Theorem

**Theorem (Transport Factory):** The factory $\mathcal{F}_{\text{trans}}$:

1. **Inputs:** Equivalent currents $(T, S)$, witness $g$, structure type
2. **Outputs:** Transport map $\tau_{T \to S}$
3. **Guarantees:**
   - Preserves specified structure
   - Functorial (composable)
   - Natural (commutes with flow)

**Constructive Content:**
- Given equivalence, compute transport explicitly
- Transport is invertible: $\tau_{S \to T} = \tau_{T \to S}^{-1}$
- Transport respects soft certificates

**Parallel Transport (Ambrosio-Gigli, 2013):** For metric measure spaces, transport along curves:
$$\tau_\gamma: T_{\gamma(0)} \to T_{\gamma(1)}$$

via Wasserstein geodesics.

**Reference:** Ambrosio, L., Gigli, N. (2013). A user's guide to optimal transport. *Springer LNM*, 2062, 1-155.

## Key GMT Inequalities Used

1. **Isometry Invariance:**
   $$\mathbf{M}((g)_\# T) = \mathbf{M}(T)$$

2. **Pushforward of Tangent Cone:**
   $$C_{g(x)}(S) = (dg)_\# C_x(T)$$

3. **Wasserstein Distance:**
   $$T \sim S \implies W_2(\|T\|, \|S\|) = 0$$

4. **Functoriality:**
   $$\tau_{S \to R} \circ \tau_{T \to S} = \tau_{T \to R}$$

## Literature References

- Myers, S. B., Steenrod, N. E. (1939). Group of isometries. *Ann. of Math.*, 40.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Michor, P. W. (2008). *Topics in Differential Geometry*. AMS.
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS.
- Ambrosio, L., Gigli, N. (2013). Optimal transport. *Springer LNM*, 2062.
- do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.
