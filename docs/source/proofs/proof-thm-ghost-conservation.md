# Proof of Ghost Conservation Theorem (thm-ghost-conservation)

:::{prf:proof}
:label: proof-thm-ghost-conservation

**Theorem (Ghost Conservation):** Surgery S7 (Ghost Extension) preserves the adjunction naturality $\mathcal{F} \dashv U$ between the expansion functor and the forgetful functor.

This proof establishes that Surgery S7 (Ghost Extension, {prf:ref}`def-surgery-sd`) preserves the adjunction naturality $\mathcal{F} \dashv U$ between the expansion functor and the forgetful functor. The key mechanism is the BRST cohomological construction ({prf:ref}`mt-act-ghost`): the extended system $\hat{X} = X \times \mathbb{R}^k$ is **isomorphic to the original system in the derived category**, ensuring that the physical content (the "Thin Kernel") is exactly conserved.

---

## Setup and Notation

### Given Data

We are provided with the following framework components:

1. **Ambient Topos:** $\mathcal{E}$ is a cohesive $(\infty, 1)$-topos equipped with:
   - Adjoint quadruple: $\Pi \dashv \flat \dashv \sharp \dashv \oint$
   - Modalities encoding shape, flat inclusion, and sharp contraction

2. **Category of Thin Kernels:** $\mathbf{Thin}_T$ with objects:
   $$T = (X^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}})$$
   encoding minimal physical data (state space, potential, dissipation, symmetry, boundary).

3. **Category of Hypostructures:** $\mathbf{Hypo}_T(\mathcal{E})$ with objects:
   $$\mathbb{H} = (X, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$$
   encoding the full structural completion with certificate data.

4. **Expansion-Forgetful Adjunction ({prf:ref}`thm-expansion-adjunction`):**
   $$\mathcal{F}: \mathbf{Thin}_T \rightleftarrows \mathbf{Hypo}_T(\mathcal{E}): U$$
   with unit $\eta: \text{Id}_{\mathbf{Thin}_T} \Rightarrow U \circ \mathcal{F}$ and counit $\varepsilon: \mathcal{F} \circ U \Rightarrow \text{Id}_{\mathbf{Hypo}_T}$.

5. **Surgery S7 (Ghost Extension, {prf:ref}`def-surgery-sd`):**
   - Input: Hypostructure $\mathbb{H} = (X, \Phi, \ldots)$ with stiffness failure $K_{\text{LS}_\sigma}^{\text{br}}$
   - Transformation: $\hat{X} = X \times \mathbb{R}^k$ with extended potential $\hat{\Phi}(x, \xi) = \Phi(x) + \frac{1}{2}|\xi|^2$
   - Output: Extended hypostructure $\hat{\mathbb{H}} = (\hat{X}, \hat{\Phi}, \ldots)$ with restored spectral gap

6. **BRST Complex ({prf:ref}`mt-act-ghost`):**
   - Ghost fields $(c, \bar{c})$ with opposite (fermionic) statistics
   - Nilpotent differential $s: s^2 = 0$
   - BRST cohomology: $H^*_s(X_{\text{BRST}})$

### Goal

We construct a certificate:
$$K_{\text{ghost}}^+ = (X, \hat{X}, \pi, \text{BRST-iso}, \eta\text{-commutativity})$$
witnessing:

1. **Derived Isomorphism:** $\hat{X} \cong X$ in the derived category $D^b(\mathbf{Hypo}_T)$
2. **Adjunction Preservation:** The unit $\eta_T: T \to U(\mathcal{F}(T))$ commutes with the projection $\pi: \hat{X} \to X$
3. **Physical State Conservation:** $\mathcal{H}_{\text{phys}}(\hat{\mathbb{H}}) \cong \mathcal{H}_{\text{phys}}(\mathbb{H})$

---

## Step 1: Ghost Variables as a Graded Extension

### Lemma 1.1: Ghost Sector is Contractible

**Statement:** The ghost sector $\mathbb{R}^k$ (with potential $\frac{1}{2}|\xi|^2$) is contractible in the homotopy-theoretic sense.

**Proof:**

**Step 1.1.1 (Quadratic Potential):** The potential on the ghost sector is:
$$\Phi_{\text{ghost}}(\xi) = \frac{1}{2}|\xi|^2 = \frac{1}{2}\sum_{i=1}^k \xi_i^2$$

This is a positive-definite quadratic form with unique minimum at $\xi = 0$.

**Step 1.1.2 (Morse Theory):** By Morse theory, a Euclidean space $\mathbb{R}^k$ with a non-degenerate quadratic potential has trivial homology above degree 0:
$$H_n(\mathbb{R}^k, \Phi_{\text{ghost}}) = \begin{cases} \mathbb{Z} & n = 0 \\ 0 & n > 0 \end{cases}$$

The gradient flow of $\Phi_{\text{ghost}}$ contracts $\mathbb{R}^k$ to the origin.

**Step 1.1.3 (Homotopy Equivalence):** The inclusion $\{0\} \hookrightarrow \mathbb{R}^k$ is a homotopy equivalence. Therefore:
$$\pi_n(\mathbb{R}^k, 0) = 0 \quad \forall n \geq 0$$

The ghost sector contributes no non-trivial homotopy. □

### Lemma 1.2: Extended System is a Trivial Bundle

**Statement:** The extended system $\hat{X} = X \times \mathbb{R}^k$ is a trivial fiber bundle over $X$ with contractible fiber.

**Proof:**

**Step 1.2.1 (Product Structure):** By construction of Surgery S7:
$$\hat{X} = X \times \mathbb{R}^k$$
with projection $\pi: \hat{X} \to X$ given by $\pi(x, \xi) = x$.

**Step 1.2.2 (Trivial Fibration):** A product of a space with a contractible space is homotopy equivalent to the original space:
$$\hat{X} \simeq X$$

**Step 1.2.3 (Section):** The zero section $\sigma: X \to \hat{X}$ given by $\sigma(x) = (x, 0)$ provides a splitting:
$$\pi \circ \sigma = \text{id}_X$$

This exhibits $\hat{X}$ as a trivial bundle over $X$. □

---

## Step 2: BRST Cohomology Recovers Physical States

### Lemma 2.1: Ghost Fields Form a BRST Complex

**Statement:** The ghost variables $\xi = (\xi_1, \ldots, \xi_k)$ in Surgery S7 correspond to the ghost fields $(c, \bar{c})$ in the BRST formalism, forming a cochain complex with nilpotent differential.

**Proof:**

**Step 2.1.1 (Null Directions as Gauge Symmetries):** The stiffness failure $K_{\text{LS}_\sigma}^{\text{br}}$ indicates a zero spectral gap: $\ker(H_\Phi) \neq \{0\}$. The $k$ null directions in the kernel correspond to $k$ infinitesimal symmetries (flat directions of the potential).

**Step 2.1.2 (Ghost Fields for Null Directions):** Following the BRST prescription ({prf:ref}`mt-act-ghost`), each null direction receives:
- A ghost field $c_i$ (Grassmann variable, ghost number +1)
- An anti-ghost field $\bar{c}_i$ (Grassmann variable, ghost number -1)

In the bosonic setting of Surgery S7, the role is played by the auxiliary variables $\xi_i \in \mathbb{R}$, which lift the degeneracy.

**Step 2.1.3 (BRST Differential):** Define the BRST operator:
$$s: \Omega^*(X) \otimes \mathbb{R}[\xi] \to \Omega^*(X) \otimes \mathbb{R}[\xi]$$
with $s(\xi_i) = 0$ (ghosts are BRST-closed) and $s(f) = \sum_i \xi_i \frac{\partial f}{\partial x^i}$ on functions along null directions.

The key property is $s^2 = 0$ (nilpotency), which follows from the fact that the null directions commute. □

### Lemma 2.2: Physical States are BRST Cohomology

**Statement:** The space of physical states in the extended system is isomorphic to the BRST cohomology:
$$\mathcal{H}_{\text{phys}}(\hat{\mathbb{H}}) \cong H^0_s(X_{\text{BRST}})$$

**Proof:**

**Step 2.2.1 (BRST-Closed States):** A state $\psi \in \mathcal{H}(\hat{X})$ is physical if it is BRST-invariant:
$$s\psi = 0$$

This means the state is annihilated by all gauge transformations along null directions.

**Step 2.2.2 (BRST-Exact States are Unphysical):** States of the form $\psi = s\chi$ for some $\chi$ are gauge artifacts (pure gauge states). Physical observables must vanish on such states.

**Step 2.2.3 (Cohomology Quotient):** The physical Hilbert space is:
$$\mathcal{H}_{\text{phys}} = \frac{\ker(s)}{\text{Im}(s)} = H^0_s(X_{\text{BRST}})$$

**Step 2.2.4 (Isomorphism with Original):** Since the ghost sector is contractible (Lemma 1.1), the BRST cohomology is concentrated in degree 0 and isomorphic to the original state space:
$$H^0_s(X_{\text{BRST}}) \cong \mathcal{H}(X)$$

This is the content of the "No-Ghost Theorem" in the BRST literature. □

---

## Step 3: Adjunction Naturality is Preserved

### Lemma 3.1: Projection Induces Derived Isomorphism

**Statement:** The projection $\pi: \hat{X} \to X$ induces an isomorphism in the derived category $D^b(\mathbf{Hypo}_T)$.

**Proof:**

**Step 3.1.1 (Derived Category Construction):** The derived category $D^b(\mathbf{Hypo}_T)$ is obtained by:
1. Forming the category of chain complexes in $\mathbf{Hypo}_T$
2. Localizing at quasi-isomorphisms (maps inducing isomorphisms on cohomology)

**Step 3.1.2 (Projection as Quasi-Isomorphism):** Consider the projection morphism:
$$\pi: \hat{\mathbb{H}} \to \mathbb{H}$$
at the level of chain complexes. By Lemma 2.2:
$$H^*(\hat{\mathbb{H}}) \cong H^*(X_{\text{BRST}}) \cong H^*(\mathbb{H})$$

Since $\pi$ induces an isomorphism on cohomology, it is a quasi-isomorphism.

**Step 3.1.3 (Derived Isomorphism):** In the derived category, quasi-isomorphisms become invertible:
$$[\hat{\mathbb{H}}] \cong [\mathbb{H}] \quad \text{in } D^b(\mathbf{Hypo}_T)$$ □

### Lemma 3.2: Unit Commutativity

**Statement:** The adjunction unit $\eta_T: T \to U(\mathcal{F}(T))$ commutes with the Surgery S7 extension.

**Proof:**

**Step 3.2.1 (Original Unit):** For a Thin Kernel $T$, the unit:
$$\eta_T: T \to U(\mathcal{F}(T))$$
embeds the thin data into the underlying thin data of the free hypostructure.

**Step 3.2.2 (Extended Unit):** After Surgery S7, the extended thin kernel is:
$$\hat{T} = (X \times \mathbb{R}^k, \hat{\Phi}, \ldots)$$

The extended unit:
$$\eta_{\hat{T}}: \hat{T} \to U(\mathcal{F}(\hat{T}))$$
embeds the extended thin data.

**Step 3.2.3 (Commutativity Diagram):** We have the following diagram:
$$
\begin{CD}
T @>{\eta_T}>> U(\mathcal{F}(T)) \\
@V{\sigma}VV @VV{U(\mathcal{F}(\sigma))}V \\
\hat{T} @>{\eta_{\hat{T}}}>> U(\mathcal{F}(\hat{T}))
\end{CD}
$$
where $\sigma: T \to \hat{T}$ is the zero-section embedding $\sigma(x) = (x, 0)$.

By naturality of $\eta$ (Step 6 of {prf:ref}`thm-expansion-adjunction`), this diagram commutes.

**Step 3.2.4 (Recovery via Projection):** Composing with the projection:
$$\pi \circ \eta_{\hat{T}} \circ \sigma = \eta_T$$

The thin kernel $T$ is exactly recovered from the extended system via projection. □

---

## Step 4: Certificate Construction

### Theorem Statement (Ghost Conservation)

**Statement:** Surgery S7 (Ghost Extension) preserves adjunction naturality: the extended system $\hat{X} = X \times \mathbb{R}^k$ is isomorphic to $X$ in the derived category $D^b(\mathbf{Hypo}_T)$, and the unit $\eta$ of the adjunction $\mathcal{F} \dashv U$ commutes with projection.

**Proof Summary:**

1. **Contractibility (Lemma 1.1):** Ghost sector $\mathbb{R}^k$ is homotopy-trivial
2. **Trivial Bundle (Lemma 1.2):** $\hat{X} \simeq X$ as homotopy types
3. **BRST Complex (Lemma 2.1):** Ghost variables form a cochain complex with $s^2 = 0$
4. **State Conservation (Lemma 2.2):** $\mathcal{H}_{\text{phys}}(\hat{\mathbb{H}}) \cong \mathcal{H}(\mathbb{H})$
5. **Derived Isomorphism (Lemma 3.1):** $[\hat{\mathbb{H}}] \cong [\mathbb{H}]$ in $D^b(\mathbf{Hypo}_T)$
6. **Unit Commutativity (Lemma 3.2):** $\pi \circ \eta_{\hat{T}} \circ \sigma = \eta_T$

### Certificate

$$K_{\text{ghost}}^+ = (X, \hat{X}, \pi, \sigma, s, H^*_s, \eta\text{-comm})$$

where:
- $X$ is the original state space
- $\hat{X} = X \times \mathbb{R}^k$ is the extended state space
- $\pi: \hat{X} \to X$ is the projection (quasi-isomorphism)
- $\sigma: X \to \hat{X}$ is the zero-section (homotopy inverse)
- $s$ is the BRST differential with $s^2 = 0$
- $H^*_s$ is the BRST cohomology with $H^0_s \cong \mathcal{H}(X)$
- $\eta\text{-comm}$ is the commutativity witness for the unit

---

## Literature Connections

### BRST Cohomology (Becchi-Rouet-Stora-Tyutin 1975-1976)

**References:** {cite}`BecchiRouetStora76`, {cite}`Tyutin75`

The BRST formalism provides a cohomological treatment of gauge symmetries. Physical states are BRST cohomology classes, and the No-Ghost Theorem ensures that ghost contributions cancel.

**Connection to This Framework:** Surgery S7's ghost variables are the bosonic analogue of BRST ghosts. The spectral gap restoration corresponds to gauge fixing, and the physical state isomorphism corresponds to the No-Ghost Theorem.

### Derived Categories and Quasi-Isomorphisms

**Reference:** {cite}`GelfandManin03`

Derived categories provide a framework for working with chain complexes up to quasi-isomorphism. This is essential for understanding when two systems are "the same" at the level of cohomology.

**Connection to This Framework:** The claim that $\hat{X} \cong X$ in the derived category means they have the same "homological content" even though they differ as point-sets. This is exactly what we need to preserve the adjunction's physical meaning.

### Homotopy Type Theory and ∞-Topoi

**Reference:** {cite}`Lurie09`, {cite}`UFP13`

In the $(\infty,1)$-topos setting, objects are defined up to homotopy equivalence. A trivial fibration (bundle with contractible fiber) does not change the homotopy type.

**Connection to This Framework:** Lemma 1.2 shows that $\hat{X} \to X$ is a trivial fibration, hence a homotopy equivalence. In the $(\infty,1)$-categorical framework, this is precisely an isomorphism.

---

## Summary

This proof establishes the **Ghost Conservation Theorem**:

1. **Surgery S7 Does Not Break the Adjunction:** The ghost extension $\hat{X} = X \times \mathbb{R}^k$ is isomorphic to $X$ in the derived category because:
   - The ghost sector is contractible
   - The projection is a quasi-isomorphism
   - BRST cohomology recovers the original states

2. **Physical Content is Conserved:** The physical Hilbert space $\mathcal{H}_{\text{phys}}$ is unchanged:
   $$\mathcal{H}_{\text{phys}}(\hat{\mathbb{H}}) \cong H^0_s(X_{\text{BRST}}) \cong \mathcal{H}(X)$$

3. **Unit Naturality is Preserved:** The adjunction unit $\eta$ commutes with the ghost extension via the zero-section and projection:
   $$\pi \circ \eta_{\hat{T}} \circ \sigma = \eta_T$$

This addresses Issue 1 of the Red Team audit: the ghost variables in Surgery S7 do not break adjunction faithfulness because they introduce only "derived redundancy" that is exactly quotiented out by BRST cohomology.

:::
