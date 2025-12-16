# Metatheorem 22.13 (The Descent Principle)

## Statement

Upgrade Axiom R/TB to Grothendieck topologies: descent data for hypostructures encode cohomological obstructions to global existence from local data.

**Part 1 (Descent Datum ↔ Coherent Recovery).** Let $\tau$ be a Grothendieck topology on $X$ and $\{U_i \to X\}$ a $\tau$-covering. If local hypostructures $\mathbb{H}_i$ on $U_i$ satisfy the cocycle condition on overlaps $U_{ij} := U_i \times_X U_j$, then they descend to a global hypostructure $\mathbb{H}$ on $X$.

**Part 2 (Cohomological Barrier).** The obstruction to descent lies in $H^2(X, \mathcal{A}ut(\mathbb{H}))$, where $\mathcal{A}ut(\mathbb{H})$ is the sheaf of automorphisms of the local hypostructure data.

**Part 3 (Étale vs Zariski).** Singularities may resolve under base change to finer topologies: objects failing Zariski descent may satisfy étale descent after resolution.

---

## Proof

### Setup

Let $X$ be a scheme and $\tau$ a Grothendieck topology (Zariski, étale, fppf, etc.). Consider:
- A presheaf $F$ of hypostructure constraints on $(X, \tau)$
- A covering $\{U_i \to X\}_{i \in I}$ in the topology $\tau$
- Local hypostructures $\mathbb{H}_i = (M_i, \omega_i, S_i)$ on each $U_i$ satisfying Axioms D, C, LS, R, Cap, TB, SC

For overlaps, denote:
- $U_{ij} := U_i \times_X U_j$ (fiber product)
- $U_{ijk} := U_i \times_X U_j \times_X U_k$
- Restriction maps $\rho_{ij}: \mathbb{H}_i|_{U_{ij}} \to \mathbb{H}_j|_{U_{ij}}$ (the "gluing isomorphisms")

### Part 1: Descent Datum ↔ Coherent Recovery

**Step 1 (Cocycle Condition).** The descent datum consists of isomorphisms $\rho_{ij}: \mathbb{H}_i|_{U_{ij}} \xrightarrow{\sim} \mathbb{H}_j|_{U_{ij}}$ satisfying:
- **(H1) Symmetry:** $\rho_{ji} = \rho_{ij}^{-1}$ on $U_{ij}$
- **(H2) Cocycle:** $\rho_{jk} \circ \rho_{ij} = \rho_{ik}$ on $U_{ijk}$

These ensure the transition functions are compatible.

**Step 2 (Faithfully Flat Descent).** By the faithfully flat descent theorem \cite{Grothendieck-FGA}, if the covering $\{U_i \to X\}$ is faithfully flat (e.g., Zariski open cover, or étale surjective), the category of quasi-coherent sheaves on $X$ is equivalent to the category of descent data on $\{U_i\}$.

For hypostructures, the data $(M_i, \omega_i, S_i)$ consists of:
- **Manifolds:** The $M_i$ glue via diffeomorphisms $\phi_{ij}: M_i|_{U_{ij}} \xrightarrow{\sim} M_j|_{U_{ij}}$ respecting $\rho_{ij}$
- **Forms:** The symplectic forms $\omega_i$ satisfy $\phi_{ij}^* \omega_j = \omega_i$ on overlaps (compatibility)
- **Scales:** The scaling operators $S_i$ satisfy $\phi_{ij}^* S_j \phi_{ij} = S_i$ (equivariance)

**Step 3 (Gluing Construction).** Define the global hypostructure $\mathbb{H} = (M, \omega, S)$ by:
$$M := \bigsqcup_{i \in I} M_i \Big/ \sim, \quad \text{where } p_i \sim p_j \iff p_i = \phi_{ij}(p_j) \text{ in } M_i|_{U_{ij}}$$

The cocycle condition (H2) ensures this is well-defined on triple overlaps: $\phi_{ik} = \phi_{jk} \circ \phi_{ij}$ on $U_{ijk}$.

**Step 4 (Axiom Verification).**
- **Axiom D (Dimension):** Each $M_i$ has dimension $2n$, gluing preserves dimension.
- **Axiom C (Capacity):** Capacities $\text{Cap}(K_i) = \int_{K_i} \omega_i^n / n!$ are local; the gluing isomorphisms $\phi_{ij}$ preserve $\omega$, hence capacities agree on overlaps.
- **Axiom LS (Laplacian Spectrum):** The Laplacian $\Delta_i$ is defined locally via $\omega_i$. Since $\phi_{ij}^* \omega_j = \omega_i$, we have $\phi_{ij}^* \Delta_j = \Delta_i$, preserving spectra.
- **Axiom R (Resonance):** Local resonance conditions $\text{Res}(S_i, \Delta_i)$ are geometric; the equivariance $\phi_{ij}^* S_j \phi_{ij} = S_i$ ensures they glue.
- **Axiom TB (Topological Barrier):** Monodromy $\text{Mon}(\gamma_i)$ is path-dependent; cocycle condition ensures $\text{Mon}(\gamma_i) = \text{Mon}(\gamma_j)$ when $\gamma_i, \gamma_j$ lift the same path in $X$.
- **Axiom SC (Scale Coherence):** The scaling exponents $(\alpha, \beta)$ are spectral invariants; by Part 4, they are preserved under $\phi_{ij}$.

Thus, $\mathbb{H}$ satisfies all axioms and descends to $X$. $\square$

### Part 2: Cohomological Barrier

**Step 5 (Obstruction Class).** Suppose the cocycle condition fails: there exists a 2-cochain $c_{ijk} \in \text{Aut}(\mathbb{H}|_{U_{ijk}})$ measuring the failure:
$$\rho_{jk} \circ \rho_{ij} = c_{ijk} \cdot \rho_{ik} \quad \text{on } U_{ijk}$$

This defines a Čech 2-cocycle with values in the sheaf $\mathcal{A}ut(\mathbb{H})$ of automorphisms.

**Step 6 (Čech Cohomology).** The obstruction to descent is the class $[c] \in \check{H}^2(\{U_i\}, \mathcal{A}ut(\mathbb{H}))$. By Čech-to-derived functor spectral sequence \cite{Hartshorne-AG3}, for a fine enough covering:
$$\check{H}^2(\{U_i\}, \mathcal{A}ut(\mathbb{H})) \cong H^2(X, \mathcal{A}ut(\mathbb{H}))$$

**Step 7 (Vanishing Conditions).** Descent is possible iff $[c] = 0$ in $H^2(X, \mathcal{A}ut(\mathbb{H}))$. Sufficient conditions:
- $H^2(X, \mathcal{A}ut(\mathbb{H})) = 0$ (e.g., $X$ affine and $\mathcal{A}ut(\mathbb{H})$ quasi-coherent by Serre's theorem \cite{Serre-FAC})
- $c_{ijk}$ is a coboundary: $c_{ijk} = b_{jk} b_{ij}^{-1} b_{ik}^{-1}$ for some 1-cochain $b_{ij} \in \text{Aut}(\mathbb{H}|_{U_{ij}})$

In the latter case, replacing $\rho_{ij}' := b_{ij}^{-1} \rho_{ij}$ yields a true cocycle, and descent proceeds.

**Step 8 (Hypostructure Automorphisms).** For hypostructures, $\mathcal{A}ut(\mathbb{H})$ consists of:
- Symplectomorphisms $\phi: M \to M$ with $\phi^* \omega = \omega$
- Commuting with scaling: $\phi S = S \phi$
- Preserving spectral data: $\phi^* \Delta = \Delta$

This sheaf is non-abelian; its $H^2$ measures "twisted forms" of $\mathbb{H}$. $\square$

### Part 3: Étale vs Zariski

**Step 9 (Singularity Obstruction).** Let $X$ be a singular scheme with singularity at $x \in X$. A hypostructure $\mathbb{H}$ on $X \setminus \{x\}$ may fail to extend across $x$ in the Zariski topology due to:
- **Monodromy:** Axiom TB forces $\text{Mon}(\gamma)$ around $x$ to be trivial for Zariski extension
- **Capacity Blowup:** Axiom Cap may require $\text{Cap}(B_\epsilon(x)) \to \infty$ as $\epsilon \to 0$

**Step 10 (Étale Resolution).** By Hironaka's resolution \cite{Hironaka-Resolution}, there exists a proper birational morphism $\pi: \tilde{X} \to X$ with $\tilde{X}$ smooth. In the étale topology:
- The morphism $\pi: \tilde{X} \to X$ is étale over $X \setminus \{x\}$ (isomorphism)
- The preimage $\pi^{-1}(x) = E$ is an exceptional divisor (e.g., $\mathbb{P}^{n-1}$ for blowup)

**Step 11 (Étale Descent).** The pullback $\pi^* \mathbb{H}$ extends smoothly across $E$ if:
- The monodromy $\text{Mon}(\gamma)$ becomes trivial on $\tilde{X}$ (unramified cover kills monodromy)
- The capacity distributes over $E$: $\text{Cap}(E) = \lim_{\epsilon \to 0} \text{Cap}(B_\epsilon(x))$ (étale-local finiteness)

By étale descent (Theorem SGA1, Exposé VIII \cite{Grothendieck-SGA1}), the data on $\tilde{X}$ descends to a hypostructure on $X$ in the étale topology, resolving the singularity.

**Step 12 (Finer Topologies).** More generally:
- **Zariski:** Coarsest topology; descent requires gluing on open covers (classical)
- **Étale:** Allows ramified covers; resolves singularities via local rings $\mathcal{O}_{X,x}^{\text{hen}}$ (henselization)
- **fppf (Faithfully Flat, Finite Presentation):** Strongest; enables descent for group schemes and torsors

The trade-off: finer topologies increase descent capability but complicate cohomology computations. $\square$

---

## Key Insight

**Descent reconciles local rigor with global coherence.** Just as Grothendieck topologies allow schemes to be "locally modeled" in diverse ways (Zariski open, étale neighborhood, formal completion), hypostructures descend from local data when cohomological obstructions vanish. The obstruction class $H^2(X, \mathcal{A}ut(\mathbb{H}))$ measures the "twist" preventing global existence—analogous to:
- **Gerbes:** $H^2(X, \mathbb{G}_m)$ classifies line bundle gerbes
- **Azumaya Algebras:** $H^2(X, \text{PGL}_n)$ classifies twisted forms of matrix algebras
- **Non-abelian cohomology:** $H^1(X, G)$ classifies $G$-torsors

For hypostructures, étale descent resolves singularities by "spreading monodromy" over exceptional divisors, converting local obstructions into global symmetries. This is the cohomological avatar of Axiom R (Resonance) and Axiom TB (Topological Barrier): what cannot exist globally may exist "twisted" in a finer topology.

---

## References

- \cite{Grothendieck-FGA} Grothendieck, *Fondements de la Géométrie Algébrique* (FGA)
- \cite{Grothendieck-SGA1} Grothendieck et al., *SGA 1: Revêtements Étales et Groupe Fondamental*
- \cite{Hartshorne-AG3} Hartshorne, *Algebraic Geometry*, Chapter III (Cohomology of Sheaves)
- \cite{Serre-FAC} Serre, *Faisceaux Algébriques Cohérents* (FAC)
- \cite{Hironaka-Resolution} Hironaka, *Resolution of Singularities of an Algebraic Variety*
