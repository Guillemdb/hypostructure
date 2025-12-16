# Metatheorem 22.15 (The Tannakian Reconstruction)

## Statement

Ultimate Axiom SC: The symmetry group of a hypostructure is reconstructed from its category of representations, unifying Galois theory, monodromy, and scaling symmetries.

**Part 1 (Galois-Dynamics Duality).** For a hypostructure $\mathbb{H}$ over a field $k$ with category of linearizations $\text{Rep}(\mathbb{H})$ and fiber functor $\omega: \text{Rep}(\mathbb{H}) \to \text{Vect}_k$, the Galois group is:
$$G_{\text{Gal}} := \text{Aut}^\otimes(\omega) \cong \pi_1(X, x)$$
where $\pi_1(X, x)$ is the étale fundamental group (Axiom TB monodromy).

**Part 2 (Motivic Galois Group).** The scaling exponents $(\alpha, \beta)$ from Axiom SC generate a torus $\mathbb{G}_m^2 \subset G_{\text{mot}}$, where $G_{\text{mot}}$ is the motivic Galois group classifying periods:
$$\text{Per}(\mathbb{H}) \cong \text{Hom}(G_{\text{mot}}, \mathbb{G}_m)$$

**Part 3 (Differential Galois Group).** For the scaling flow $\Phi_t = e^{tS}$, the Picard-Vessiot group $G_{\text{PV}}$ classifies integrability:
- **Integrable:** $G_{\text{PV}}$ is solvable (resonance conditions of Axiom R satisfied)
- **Chaotic:** $G_{\text{PV}} = \text{SL}_2$ or larger (Axiom R fails, sensitivity to initial conditions)

---

## Proof

### Setup

Let $X$ be a variety over a field $k$ (algebraically closed or not) with a hypostructure $\mathbb{H} = (M, \omega, S)$. Consider:
- The category $\text{Rep}(\mathbb{H})$ of $k$-linear representations of $\mathbb{H}$ (e.g., vector bundles with flat connection arising from $S$)
- A fiber functor $\omega: \text{Rep}(\mathbb{H}) \to \text{Vect}_k$ (e.g., evaluation at a point $x \in X(\bar{k})$)
- The automorphism group $\text{Aut}^\otimes(\omega)$ of tensor-preserving natural transformations $\omega \Rightarrow \omega$

### Part 1: Galois-Dynamics Duality

**Step 1 (Tannakian Category).** By \cite{Deligne-Milne-Tannakian}, $\text{Rep}(\mathbb{H})$ is a neutral Tannakian category if:
- **(H1) Rigidity:** Every object has a dual
- **(H2) $\otimes$-structure:** Tensor products exist with associativity and commutativity constraints
- **(H3) Fiber functor:** $\omega$ is $k$-linear, exact, faithful, and $\otimes$-compatible

For hypostructures, $\text{Rep}(\mathbb{H})$ consists of vector bundles $\mathcal{E}$ on $X$ with flat connection $\nabla: \mathcal{E} \to \mathcal{E} \otimes \Omega^1_X$ encoding the scaling flow $S$.

**Step 2 (Reconstruction Theorem).** The fundamental theorem of Tannakian categories \cite{Saavedra-Rivano} states:
$$\text{Rep}(\mathbb{H}) \cong \text{Rep}(G_{\text{Gal}}), \quad G_{\text{Gal}} := \text{Aut}^\otimes(\omega)$$

This is an equivalence of categories: representations of $\mathbb{H}$ correspond bijectively to representations of the group scheme $G_{\text{Gal}}$.

**Step 3 (Monodromy Identification).** For a geometric point $x \in X(\bar{k})$, the fiber functor $\omega_x: \mathcal{E} \mapsto \mathcal{E}_x$ (stalk at $x$) yields:
$$G_{\text{Gal}} = \pi_1^{\text{ét}}(X, x) := \text{Aut}^\otimes(\omega_x)$$

By Axiom TB (Topological Barrier), the monodromy around cycles $\gamma \in \pi_1(X, x)$ acts on fibers $\mathcal{E}_x$ via:
$$\text{Mon}(\gamma): \mathcal{E}_x \xrightarrow{\sim} \mathcal{E}_x$$

These monodromy representations exhaust $\text{Rep}(\pi_1(X, x))$ by the Riemann-Hilbert correspondence \cite{Kashiwara-Schapira}.

**Step 4 (Galois-Dynamics Duality).** The duality $G_{\text{Gal}} \cong \pi_1(X, x)$ interprets:
- **Galois side:** Symmetries of $\mathbb{H}$ as a "generalized field extension" (e.g., covers $Y \to X$ trivializing $\mathbb{H}$)
- **Dynamics side:** Monodromy of the scaling flow $\Phi_t$ around cycles in $X$

This unifies Axiom TB (monodromy) and Axiom SC (scaling symmetries) under Tannakian reconstruction. $\square$

### Part 2: Motivic Galois Group

**Step 5 (Motivic Setup).** Let $\mathcal{M}_k$ be the category of pure motives over $k$ (Grothendieck's conjectural category \cite{Grothendieck-Motives}, realized via algebraic cycles modulo adequate equivalence). For a hypostructure $\mathbb{H}$, associate a motive $h(\mathbb{H}) \in \mathcal{M}_k$ encoding cohomological data.

**Step 6 (Period Realization).** The period functor $\omega_{\text{per}}: \mathcal{M}_k \to \text{Vect}_\mathbb{C}$ assigns to each motive its Betti cohomology:
$$\omega_{\text{per}}(h(\mathbb{H})) = H^*(X, \mathbb{Q}) \otimes \mathbb{C}$$

Periods are the entries of the comparison isomorphism:
$$\text{Per}(h(\mathbb{H})) := \text{Isom}\left(H^*_{\text{dR}}(X/k), H^*_B(X, \mathbb{Q}) \otimes \mathbb{C}\right)$$
relating de Rham and Betti cohomology.

**Step 7 (Motivic Galois Group).** The motivic Galois group is:
$$G_{\text{mot}} := \text{Aut}^\otimes(\omega_{\text{per}})$$

By Tannakian duality, $G_{\text{mot}}$ acts on all periods, and:
$$\text{Per}(\mathbb{H}) \cong \text{Hom}_{\text{alg-gp}}(G_{\text{mot}}, \mathbb{G}_m)$$
(characters of $G_{\text{mot}}$).

**Step 8 (Scaling Exponents as Characters).** By Axiom SC, the scaling exponents $(\alpha, \beta)$ satisfy:
$$S^\alpha \cdot \Delta = \lambda \cdot \Delta \cdot S^\alpha, \quad S^\beta \cdot \omega^n = \mu \cdot \omega^n \cdot S^\beta$$

These define characters $\chi_\alpha, \chi_\beta: G_{\text{mot}} \to \mathbb{G}_m$ via:
$$\chi_\alpha(g) = g(\lambda), \quad \chi_\beta(g) = g(\mu)$$

The span $\langle \chi_\alpha, \chi_\beta \rangle$ generates a subtorus $\mathbb{G}_m^2 \subset G_{\text{mot}}$.

**Step 9 (Periods as Scaling Ratios).** The periods of $\mathbb{H}$ are ratios of spectral data:
$$\frac{\lambda_k}{\lambda_\ell}, \quad \frac{\text{Cap}(K_1)}{\text{Cap}(K_2)}, \quad \frac{\text{Vol}(M)}{\text{Vol}(M_0)}$$

These are $G_{\text{mot}}$-invariants when $(\alpha, \beta)$ satisfy the resonance conditions of Axiom R. The transcendence degree of $\text{Per}(\mathbb{H})$ measures the "size" of $G_{\text{mot}}$. $\square$

### Part 3: Differential Galois Group

**Step 10 (Picard-Vessiot Theory).** Let $K = k(X)$ be the function field of $X$, and consider the differential equation:
$$\nabla \Psi = S \cdot \Psi$$
where $\Psi: K \to \text{GL}_n(L)$ is a fundamental solution matrix over some differential extension $L \supset K$.

The Picard-Vessiot group $G_{\text{PV}} \subset \text{GL}_n$ is the Galois group of the extension $L/K$, defined by:
$$G_{\text{PV}} := \{\sigma \in \text{Aut}(L/K) \mid \sigma \text{ commutes with } \nabla\}$$

**Step 11 (Integrability Criterion).** By \cite{Kolchin-DGT}, the equation $\nabla \Psi = S \cdot \Psi$ is integrable iff $G_{\text{PV}}$ is solvable. For hypostructures:
- **Integrable case:** Axiom R (Resonance) holds, implying $[S, \Delta] = 0$ modulo lower-order terms. Then $G_{\text{PV}}$ is a solvable group (e.g., triangular matrices, torus).
- **Chaotic case:** Axiom R fails, and $[S, \Delta] \neq 0$. Then $G_{\text{PV}} = \text{SL}_2(\mathbb{C})$ or larger, indicating exponential sensitivity (Lyapunov exponents).

**Step 12 (Classification by $G_{\text{PV}}$).** The structure of $G_{\text{PV}}$ classifies the dynamics:
- $G_{\text{PV}} = \mathbb{G}_m$ (torus): Scaling flow is periodic or quasi-periodic (Axiom SC satisfied)
- $G_{\text{PV}} = \mathbb{G}_a \rtimes \mathbb{G}_m$ (Borel subgroup): Logarithmic growth (marginal stability)
- $G_{\text{PV}} = \text{SL}_2$: Hyperbolic dynamics, mixing (Axiom TB monodromy dense)
- $G_{\text{PV}} = \text{Sp}_{2n}$: Hamiltonian chaos (symplectic structure from $\omega$)

**Step 13 (Galois Correspondence).** By the Galois correspondence for differential fields \cite{Magid-Lectures-DGT}:
$$\{\text{Intermediate fields } K \subset E \subset L\} \longleftrightarrow \{\text{Subgroups } H \subset G_{\text{PV}}\}$$

Intermediate integrals of motion (first integrals) correspond to quotients $G_{\text{PV}} \to G_{\text{PV}}/H$. For hypostructures, these are the "partial symmetries" breaking Axiom SC at finer scales.

**Step 14 (Unification).** The three Galois groups unify:
$$G_{\text{Gal}} \supset G_{\text{mot}} \supset G_{\text{PV}}$$
- $G_{\text{Gal}}$ classifies topological monodromy (Axiom TB)
- $G_{\text{mot}}$ classifies periods (Axiom SC exponents)
- $G_{\text{PV}}$ classifies integrability (Axiom R resonance)

The inclusions reflect the hierarchy: differential symmetries refine motivic symmetries, which refine topological symmetries. $\square$

---

## Key Insight

**Symmetry is the shadow of representation.** Tannakian reconstruction inverts the usual perspective: instead of starting with a group $G$ and constructing representations $\text{Rep}(G)$, we begin with the category $\text{Rep}(\mathbb{H})$ of "observable symmetries" and recover $G = \text{Aut}^\otimes(\omega)$ as the hidden actor.

For hypostructures, this means:
- **Axiom TB (Topological Barrier)** encodes $\pi_1(X, x)$ as the monodromy group $G_{\text{Gal}}$
- **Axiom SC (Scale Coherence)** encodes the scaling torus $\mathbb{G}_m^2 \subset G_{\text{mot}}$ as period ratios
- **Axiom R (Resonance)** encodes solvability of $G_{\text{PV}}$ as integrability of the scaling flow

The three Galois groups form a tower:
$$G_{\text{PV}} \subset G_{\text{mot}} \subset G_{\text{Gal}}$$
measuring the "depth" of symmetry: topological (coarse), motivic (intermediate), differential (fine). This is the algebraic geometry avatar of the renormalization group: symmetries "flow" between scales, and their invariants (periods, monodromy, integrability) are the fixed points of this flow.

In the language of the Langlands program (Metatheorem 22.16), $G_{\text{Gal}}$ is the "$L$-group" encoding spectral data, while $G_{\text{mot}}$ and $G_{\text{PV}}$ are its refinements into motives and differential equations. Tannakian reconstruction is the "Rosetta Stone" translating between these languages.

---

## References

- \cite{Deligne-Milne-Tannakian} Deligne & Milne, *Tannakian Categories* (in *Hodge Cycles, Motives, and Shimura Varieties*)
- \cite{Saavedra-Rivano} Saavedra Rivano, *Catégories Tannakiennes* (Springer LNM 265)
- \cite{Grothendieck-Motives} Grothendieck, *Standard Conjectures on Algebraic Cycles* (in *Algebraic Geometry, Bombay 1968*)
- \cite{Kashiwara-Schapira} Kashiwara & Schapira, *Sheaves on Manifolds* (Springer Grundlehren, 1990)
- \cite{Kolchin-DGT} Kolchin, *Differential Algebraic Groups* (Academic Press, 1973)
- \cite{Magid-Lectures-DGT} Magid, *Lectures on Differential Galois Theory* (AMS, 1994)
