# FACT-GermDensity: Germ Set Density — GMT Translation

## Original Statement (Hypostructure)

The germ set (local behavior near singularities) is dense in the profile space, allowing local-to-global reconstruction.

## GMT Setting

**Germ:** $\text{germ}_x(T) = [(T, x)]$ — equivalence class of currents agreeing near $x$

**Germ Space:** $\mathcal{G}_x := \{\text{germ}_x(T) : T \in \mathbf{I}_k(M)\}$

**Density:** $\mathcal{G}$ is dense in profile space $\mathcal{P}$

## GMT Statement

**Theorem (Germ Set Density).** Let $\mathcal{P} = \{T_\infty : T_\infty \text{ is a blow-up limit}\}$ be the profile space. Then:

1. **Density:** For any $T_\infty \in \mathcal{P}$ and $\varepsilon > 0$, there exists a germ $\gamma \in \mathcal{G}$ with:
$$d_{\text{GH}}(\gamma, T_\infty|_{B_1}) < \varepsilon$$

2. **Local Reconstruction:** If $\text{germ}_x(T) = \text{germ}_x(S)$ for all $x \in \text{sing}$, then $T \approx S$

3. **Finite Germs Suffice:** The germ space is spanned by finitely many canonical germs from $\mathcal{L}$

## Proof Sketch

### Step 1: Germ Equivalence Relation

**Definition:** Two currents $T, S$ are **germ equivalent at $x$** if:
$$\exists r > 0 : T \llcorner B_r(x) = S \llcorner B_r(x)$$

**Germ:** $\text{germ}_x(T) := [T]_{\sim_x}$ (equivalence class)

**Local Information:** The germ captures all local properties:
- Tangent cone at $x$
- Multiplicity at $x$
- Local regularity at $x$

### Step 2: Germ Space Structure

**Metric on Germs:** Define:
$$d_{\mathcal{G}}(\gamma_1, \gamma_2) := \inf_{r > 0} d_{\text{flat}}(T_1 \llcorner B_r, T_2 \llcorner B_r)$$

for any representatives $T_1 \in \gamma_1$, $T_2 \in \gamma_2$.

**Completeness:** $(\mathcal{G}_x, d_{\mathcal{G}})$ is a complete metric space.

**Compactness (Simon, 1983):** Under mass bounds, $\mathcal{G}_x$ is compact.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU. [Chapter 5]

### Step 3: Profile Space as Limit of Germs

**Blow-Up Germs:** For $T \in \mathbf{I}_k(M)$ and $x \in \text{sing}(T)$:
$$\text{germ}_{x,\lambda}(T) := \text{germ}_0((\eta_{x,\lambda})_\# T)$$

**Tangent Cone as Limit:**
$$T_{x,\infty} = \lim_{\lambda \to 0} \text{germ}_{x,\lambda}(T)$$

exists by compactness.

**Profile = Limit Germ:** Every profile $T_\infty \in \mathcal{P}$ is the limit of germs.

### Step 4: Density via Approximation

**Theorem:** For any profile $T_\infty \in \mathcal{P}$, the sequence of germs:
$$\gamma_n := \text{germ}_{0,1/n}(T_n)$$

converges to $T_\infty$ as $n \to \infty$.

*Proof:* By definition, $T_\infty = \lim_n T_{x_n, \lambda_n}$ for some blow-up sequence. The germs approximate arbitrarily well.

**Density:** The germ space $\mathcal{G}$ is dense in $\mathcal{P}$ in the Gromov-Hausdorff topology.

### Step 5: Reconstruction from Germs

**Local-to-Global Principle (Federer, 1969):** An integral current $T$ is uniquely determined by its restriction to any open cover.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.1]

**Germ Reconstruction:** If we know $\text{germ}_x(T)$ for all $x$ in a dense set $D \subset \text{spt}(T)$:
$$T = \lim_{\varepsilon \to 0} \sum_{x \in D \cap \varepsilon\mathbb{Z}^n} \text{germ}_x(T) \cdot \chi_{V_x}$$

where $V_x$ is a Voronoi cell.

### Step 6: Finite Canonical Germs

**Library Germs:** The canonical library $\mathcal{L}$ provides finitely many germ types:
$$\mathcal{G}_{\text{can}} := \{\text{germ}_0(C) : C \in \mathcal{L}\}$$

**Theorem:** Every germ $\gamma \in \mathcal{G}$ is approximated by library germs:
$$d_{\mathcal{G}}(\gamma, \mathcal{G}_{\text{can}}) < \varepsilon$$

after appropriate rescaling.

*Proof:* By profile trichotomy:
- Library profiles give library germs
- Tame profiles are continuous deformations of library germs
- Wild profiles are excluded by soft permits

### Step 7: Germ Cohomology

**Germ Sheaf:** Define sheaf $\mathcal{F}$ on $M$ by:
$$\mathcal{F}(U) := \{\text{germs over } U\}$$

**Stalks:** $\mathcal{F}_x = \mathcal{G}_x$ (germ space at $x$)

**Čech Cohomology:** The global sections $H^0(M; \mathcal{F})$ reconstruct global currents.

**Reference:** Bredon, G. E. (1997). *Sheaf Theory*. Springer. [Chapter II]

### Step 8: Density Measure

**Density Function:** For $T \in \mathbf{I}_k(M)$ at $x$:
$$\Theta_k(T, x) := \lim_{r \to 0} \frac{\mathbf{M}(T \cap B_r(x))}{\omega_k r^k}$$

**Germ Determines Density:** $\text{germ}_x(T)$ determines $\Theta_k(T, x)$.

**Monotonicity (Almgren, 1983):** Density is upper semicontinuous and determined by local behavior.

**Reference:** Almgren, F. J. (1983). Q-valued functions minimizing Dirichlet's integral. Preprint (published 2000 by World Scientific).

### Step 9: Compilation Theorem

**Theorem (GermDensity):** Under soft permits:

1. **$\mathcal{G}$ dense in $\mathcal{P}$:** Every profile is approximated by germs

2. **Finite spanning:** $\mathcal{G}$ is spanned by library germs plus tame deformations

3. **Reconstruction:** Currents are determined by their singular germs:
$$T \cong \text{regular part} \cup \bigcup_{x \in \text{sing}} \text{germ}_x(T)$$

**Constructive Content:**
- Given $T$, compute $\text{sing}(T)$ and extract germs
- Verify germs match library elements
- Reconstruct from germ data

### Step 10: Applications

**Application 1: Removable Singularities**

If $\text{germ}_x(T) = \text{germ}_x(\text{regular})$, then $x$ is a removable singularity.

**Brakke's Regularity (1978):** For mean curvature flow, germs at regular points extend smoothly.

**Reference:** Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.

**Application 2: Uniqueness of Tangent Cones**

If the germ is unique, the tangent cone is unique:
$$|\mathcal{G}_x| = 1 \implies \text{unique tangent cone at } x$$

**Simon (1993):** At isolated singularities, tangent cones are unique.

**Reference:** Simon, L. (1993). Cylindrical tangent cones. *J. Diff. Geom.*, 38, 585-652.

## Key GMT Inequalities Used

1. **Germ Compactness:**
   $$\mathbf{M}(T) \leq \Lambda \implies \mathcal{G} \text{ is compact}$$

2. **Density via Blow-Up:**
   $$T_\infty = \lim \text{germ}_{x,\lambda}(T)$$

3. **Local-to-Global:**
   $$\text{germs at all } x \text{ determine } T$$

4. **Finite Spanning:**
   $$\mathcal{G} \subset \overline{\text{span}(\mathcal{G}_{\text{can}})}$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Simon, L. (1993). Cylindrical tangent cones. *J. Diff. Geom.*, 38.
- Almgren, F. J. (2000). *Almgren's Big Regularity Paper*. World Scientific.
- Bredon, G. E. (1997). *Sheaf Theory*. Springer.
- Brakke, K. (1978). *The Motion of a Surface by Its Mean Curvature*. Princeton.
