### Metatheorem 22.7 (The Virtual Cycle Correspondence)

Axiom Cap (Capacity) extends naturally to virtual fundamental classes in moduli spaces with obstructions. This allows integration of permits over singular moduli spaces, connecting hypostructure defects to Gromov-Witten and Donaldson-Thomas invariants.

**Statement.** Let $\mathcal{M}$ be a moduli space of profiles (curves, sheaves, maps) with expected dimension $\text{vdim}(\mathcal{M}) = d$. Suppose $\mathcal{M}$ has a perfect obstruction theory:
$$\mathbb{E}^\bullet = [E^{-1} \to E^0] \to \mathbb{L}_{\mathcal{M}}$$
where $\mathbb{L}_{\mathcal{M}}$ is the cotangent complex. Then Axiom Cap upgrades to virtual capacity:

1. **Virtual Fundamental Class:** The singular locus $\mathcal{Y}_{\text{sing}} \subset \mathcal{M}$ (where profiles violate permits) admits a virtual fundamental class:
$$[\mathcal{Y}_{\text{sing}}]^{\text{vir}} = e(\text{Ob}^\vee) \cap [\mathcal{M}] \in A_d(\mathcal{M})$$
where $e(\text{Ob}^\vee)$ is the Euler class of the dual obstruction bundle and $A_d(\mathcal{M})$ is the Chow group.

2. **Permit Integration:** Permits integrate over the virtual class:
$$\int_{[\mathcal{Y}_{\text{sing}}]^{\text{vir}}} \Pi = \int_{[\mathcal{M}]^{\text{vir}}} \mathbb{1}_{\{\Pi = 0\}} = \text{Defect Count}.$$
This counts profiles satisfying $\Pi = 0$ (zero-permit locus) with virtual multiplicity.

3. **GW/DT Invariants:** Gromov-Witten invariants count Axiom R defects (curves violating energy concentration) integrated over moduli of stable maps:
$$\text{GW}_{g,n,\beta}(X) = \int_{[\overline{M}_{g,n}(X, \beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i).$$
Donaldson-Thomas invariants count Axiom Cap defects (sheaves violating capacity bounds) integrated over Hilbert schemes:
$$\text{DT}_n(X) = \int_{[\text{Hilb}^n(X)]^{\text{vir}}} 1.$$

*Proof.*

**Step 1 (Setup: Moduli Spaces with Obstructions).**

Let $\mathcal{M}$ be a moduli space parametrizing geometric objects (stable maps, coherent sheaves, instantons, etc.). The expected (virtual) dimension is:
$$\text{vdim}(\mathcal{M}) = \text{rank}(E^0) - \text{rank}(E^{-1})$$
where $[E^{-1} \to E^0]$ is the obstruction theory.

The deformation-obstruction theory gives:
- $T_{\mathcal{M}} = \ker(E^{-1} \to E^0)$ (tangent space = deformations),
- $\text{Ob}(E) = \text{coker}(E^{-1} \to E^0)$ (obstruction space).

When $\text{Ob}(E) \neq 0$, the moduli space is obstructed (singular), and its actual dimension exceeds the virtual dimension. A perfect obstruction theory allows constructing a virtual fundamental class $[\mathcal{M}]^{\text{vir}}$ of the correct dimension.

**Step 2 (Perfect Obstruction Theory).**

*Lemma 22.7.1 (Perfect Obstruction Theory).* A perfect obstruction theory on $\mathcal{M}$ is a morphism:
$$\phi: \mathbb{E}^\bullet \to \mathbb{L}_{\mathcal{M}}$$
in the derived category $D^b(\mathcal{M})$ where:
1. $\mathbb{E}^\bullet = [E^{-1} \to E^0]$ is a complex of vector bundles with cohomology in degrees $[-1, 0]$,
2. $h^0(\phi)$ is an isomorphism: $h^0(\mathbb{E}^\bullet) \cong T_{\mathcal{M}}$,
3. $h^{-1}(\phi)$ is surjective: $h^{-1}(\mathbb{E}^\bullet) \to \text{Ob}_{\mathcal{M}} \to 0$.

*Proof of Lemma.* This is the definition of Behrend-Fantechi \cite{BehrFant97}. The perfect obstruction theory provides a two-term complex controlling deformations and obstructions, allowing the construction of a virtual fundamental class via:
$$[\mathcal{M}]^{\text{vir}} = 0_E^! [\mathcal{M}] \in A_{\text{vdim}}(\mathcal{M})$$
where $0_E: \mathcal{M} \to E$ is the zero section and $0_E^!$ is the refined Gysin homomorphism. $\square$

**Step 3 (Virtual Fundamental Class from Euler Class).**

*Lemma 22.7.2 (Euler Class Construction).* The virtual fundamental class can be expressed as:
$$[\mathcal{M}]^{\text{vir}} = e(\text{Ob}^\vee) \cap [\mathcal{M}]$$
where:
- $\text{Ob}^\vee = (E^0)^\vee \to (E^{-1})^\vee$ is the dual obstruction bundle,
- $e(\text{Ob}^\vee)$ is the Euler class (top Chern class),
- $[\mathcal{M}]$ is the fundamental class of the ambient space.

*Proof of Lemma.* When $\mathcal{M}$ is smooth but has virtual dimension less than actual dimension (obstructed), the obstruction bundle $\text{Ob} = \text{coker}(E^{-1} \to E^0)$ has rank:
$$r = \text{rank}(\text{Ob}) = \dim(\mathcal{M}) - \text{vdim}(\mathcal{M}).$$

The zero locus of a section $s$ of $\text{Ob}^\vee$ has dimension $\dim(\mathcal{M}) - r = \text{vdim}(\mathcal{M})$. The virtual class is the Euler class of $\text{Ob}^\vee$:
$$[\mathcal{M}]^{\text{vir}} = s^{-1}(0) = e(\text{Ob}^\vee) \cap [\mathcal{M}].$$

When $\mathcal{M}$ is singular, the construction uses the intrinsic normal cone and virtual Gysin map (Behrend-Fantechi). $\square$

**Step 4 (Singular Locus as Zero-Permit Locus).**

*Lemma 22.7.3 (Permits as Sections).* Each hypostructure permit $\Pi_A$ (for axiom $A$) defines a section:
$$\Pi_A: \mathcal{M} \to \text{Ob}^\vee$$
where $\Pi_A(E) = 0$ if and only if the object $E$ satisfies the permit (is not singular under axiom $A$).

The singular locus is:
$$\mathcal{Y}_{\text{sing}} = \{E \in \mathcal{M} : \Pi_A(E) = 0 \text{ for some } A\}.$$

*Proof of Lemma.* For each axiom, the permit is a numerical constraint:
- **Axiom SC (Scaling):** $\Pi_{\text{SC}}(E) = \alpha(E) - \beta(E)$. Zero locus: $\alpha = \beta$ (critical scaling).
- **Axiom Cap (Capacity):** $\Pi_{\text{Cap}}(E) = \text{Cap}(\text{Supp}(E))$. Zero locus: support has zero capacity.
- **Axiom LS (Lojasiewicz):** $\Pi_{\text{LS}}(E) = \|\nabla \Phi(E)\| - c |\Phi(E)|^{1-\theta}$. Zero locus: gradient vanishes faster than Lojasiewicz bound.

Each permit $\Pi_A$ is a function on $\mathcal{M}$. When $\mathcal{M}$ has a perfect obstruction theory, $\Pi_A$ lifts to a section of $\text{Ob}^\vee$ (or a descendant class in cohomology).

The zero locus $\{\Pi_A = 0\}$ is the set of profiles where axiom $A$ fails, i.e., the singular locus for that axiom. The total singular locus is the union over all axioms. $\square$

**Step 5 (Integration of Permits).**

*Lemma 22.7.4 (Permit Integration Formula).* The count of singular profiles (with multiplicity) is:
$$\mathcal{N}_{\text{sing}} = \int_{[\mathcal{M}]^{\text{vir}}} \mathbb{1}_{\{\Pi = 0\}} = \int_{[\mathcal{M}]^{\text{vir}}} e(\Pi)$$
where $e(\Pi)$ is the Euler class of the permit section.

When $\Pi$ is transverse to the zero section, this counts points:
$$\mathcal{N}_{\text{sing}} = \sum_{E: \Pi(E) = 0} \frac{1}{|\text{Aut}(E)|}.$$

*Proof of Lemma.* The indicator function $\mathbb{1}_{\{\Pi = 0\}}$ is the Poincare dual of the zero locus:
$$\text{PD}(\{\Pi = 0\}) = e(\Pi) \in H^*(\mathcal{M}).$$
Integrating over the virtual class:
$$\int_{[\mathcal{M}]^{\text{vir}}} \mathbb{1}_{\{\Pi = 0\}} = \int_{[\mathcal{M}]^{\text{vir}}} e(\Pi) = \deg(e(\Pi) \cap [\mathcal{M}]^{\text{vir}}).$$

When $\Pi$ is a regular section (transverse to zero), the zero locus is a finite set of points, each with multiplicity:
$$\text{mult}(E) = \frac{1}{|\text{Aut}(E)|}$$
(automorphisms reduce multiplicity in moduli spaces). Summing gives the total count. $\square$

**Step 6 (Gromov-Witten Invariants as Axiom R Defects).**

*Lemma 22.7.5 (GW Invariants Count Energy Defects).* Let $\overline{M}_{g,n}(X, \beta)$ be the moduli space of genus $g$ stable maps to $X$ with $n$ marked points, representing the curve class $\beta \in H_2(X)$. The Gromov-Witten invariant is:
$$\text{GW}_{g,n,\beta}(X; \gamma_1, \ldots, \gamma_n) = \int_{[\overline{M}_{g,n}(X, \beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i)$$
where $\text{ev}_i: \mathcal{M} \to X$ evaluates the map at the $i$-th marked point, and $\gamma_i \in H^*(X)$ are cohomology insertions.

This counts curves violating Axiom R (Energy Concentration): the defect functional is:
$$\mathfrak{r}(f: C \to X) = \int_C f^*(\omega) - \text{const}$$
where $\omega$ is the Kahler form on $X$.

*Proof of Lemma.* The moduli space $\overline{M}_{g,n}(X, \beta)$ parametrizes stable maps $f: C \to X$ where $C$ is a genus $g$ nodal curve. The expected dimension is:
$$\text{vdim} = \int_\beta c_1(TX) + (1-g)(\dim X - 3) + n.$$

The obstruction theory is:
$$\mathbb{E}^\bullet = [R^1 f_* f^* TX \to R^0 f_* f^* TX]^\vee$$
where the deformations are infinitesimal variations of the map $f$, and obstructions are elements of $H^1(C, f^* TX)$.

The virtual class $[\overline{M}_{g,n}(X, \beta)]^{\text{vir}}$ has dimension $\text{vdim}$, even when the actual moduli space is singular or has higher dimension due to obstructed deformations.

Gromov-Witten invariants integrate cohomology classes over this virtual class. In hypostructure terms, each stable map $f$ represents a profile attempting to concentrate energy along the curve $C$. The defect is:
$$\mathfrak{r}(f) = \int_C f^*(\omega)$$
(total energy of the curve). The GW invariant counts curves with specified energy $\int_\beta \omega$ and insertion constraints $\gamma_i$, weighted by virtual multiplicity. $\square$

**Step 7 (Donaldson-Thomas Invariants as Axiom Cap Defects).**

*Lemma 22.7.6 (DT Invariants Count Capacity Defects).* Let $\text{Hilb}^n(X)$ be the Hilbert scheme of $n$ points on a Calabi-Yau threefold $X$, or more generally, the moduli space of ideal sheaves with Chern character $\text{ch} = (r, c_1, c_2, c_3)$. The Donaldson-Thomas invariant is:
$$\text{DT}_{\text{ch}}(X) = \int_{[\text{Hilb}(X, \text{ch})]^{\text{vir}}} 1$$
(integral of the constant function 1 over the virtual class).

This counts sheaves violating Axiom Cap: the capacity defect is:
$$\mathfrak{c}(\mathcal{I}) = \text{Cap}(\text{Supp}(\mathcal{I}))$$
where $\text{Supp}(\mathcal{I})$ is the support of the ideal sheaf $\mathcal{I}$ (a subscheme of $X$).

*Proof of Lemma.* The Hilbert scheme parametrizes ideal sheaves $\mathcal{I} \subset \mathcal{O}_X$ or equivalently, coherent sheaves $\mathcal{F}$ on $X$. The obstruction theory is:
$$\mathbb{E}^\bullet = R\text{Hom}(\mathcal{F}, \mathcal{F})_0$$
where the subscript 0 denotes the traceless part (Ext groups with zero trace).

For a Calabi-Yau threefold ($K_X \cong \mathcal{O}_X$), Serre duality gives:
$$\text{Ext}^i(\mathcal{F}, \mathcal{F}) \cong \text{Ext}^{3-i}(\mathcal{F}, \mathcal{F})^\vee.$$
The virtual dimension is:
$$\text{vdim} = \int_X \text{ch}(\mathcal{F}) \cdot \text{td}(X) = c_3(\mathcal{F}).$$

The DT invariant integrates the constant function 1, giving a count of sheaves (weighted by virtual multiplicity):
$$\text{DT}_{\text{ch}}(X) = \sum_{\mathcal{F}} \frac{1}{|\text{Aut}(\mathcal{F})|}.$$

In hypostructure terms, each sheaf $\mathcal{F}$ represents a profile attempting to concentrate energy on its support $\text{Supp}(\mathcal{F})$. Axiom Cap requires:
$$\text{Cap}(\text{Supp}(\mathcal{F})) > 0.$$
Sheaves with zero-capacity support (e.g., supported on a curve in a 3-fold) violate Axiom Cap. The DT invariant counts such violations, weighted by the obstruction theory. $\square$

**Step 8 (Virtual Capacity Bound).**

*Lemma 22.7.7 (Capacity on Virtual Classes).* For a moduli space $\mathcal{M}$ with perfect obstruction theory, the virtual capacity is:
$$\text{Cap}^{\text{vir}}(\mathcal{M}) = \sup\left\{\int_{[\mathcal{M}]^{\text{vir}}} \omega : \omega \text{ is a Kahler form on ambient space}\right\}.$$

If $\text{Cap}^{\text{vir}}(\mathcal{M}) = 0$, then the singular locus $\mathcal{Y}_{\text{sing}} \subset \mathcal{M}$ is empty (no profiles violate permits).

*Proof of Lemma.* The virtual fundamental class $[\mathcal{M}]^{\text{vir}}$ is a cycle in the Chow group $A_{\text{vdim}}(\mathcal{M})$. Its capacity is the supremum of integrals of positive forms.

When $[\mathcal{M}]^{\text{vir}} = 0$ (the virtual class vanishes), we have $\text{Cap}^{\text{vir}}(\mathcal{M}) = 0$, and no singular profiles exist (the count is zero).

Conversely, if $[\mathcal{M}]^{\text{vir}} \neq 0$, then $\text{Cap}^{\text{vir}}(\mathcal{M}) > 0$, and singular profiles are possible (but their count may still be zero if permits are satisfied). $\square$

**Step 9 (Obstruction Bundle and Defect Functional).**

*Lemma 22.7.8 (Defects as Obstruction Sections).* The hypostructure defect functional:
$$\mathcal{D}_A(E) = \max\{0, -\Pi_A(E)\}$$
(positive part of the negative permit) lifts to a section of the obstruction bundle $\text{Ob}^\vee$.

The total defect count is:
$$\mathcal{D}_{\text{total}}(\mathcal{M}) = \int_{[\mathcal{M}]^{\text{vir}}} \sum_A \mathcal{D}_A.$$

*Proof of Lemma.* Each axiom defect $\mathcal{D}_A$ measures the failure of permit $\Pi_A$. In moduli spaces, these defects are obstruction classes:
$$\mathcal{D}_A \in H^*(\mathcal{M}, \text{Ob}^\vee).$$

Integrating over the virtual class gives the total defect:
$$\mathcal{D}_{\text{total}} = \int_{[\mathcal{M}]^{\text{vir}}} \sum_A \mathcal{D}_A = \sum_A \int_{[\mathcal{M}]^{\text{vir}}} \mathcal{D}_A.$$

When all permits are satisfied ($\Pi_A \geq 0$ for all $A$), the defects vanish ($\mathcal{D}_A = 0$), and:
$$\mathcal{D}_{\text{total}} = 0.$$
This is the global regularity condition: zero total defect integrated over moduli space. $\square$

**Step 10 (Conclusion).**

The Virtual Cycle Correspondence establishes that Axiom Cap (Capacity) extends naturally to virtual fundamental classes in obstructed moduli spaces. The singular locus $\mathcal{Y}_{\text{sing}}$ (profiles violating permits) admits a virtual class $[\mathcal{Y}_{\text{sing}}]^{\text{vir}} = e(\text{Ob}^\vee) \cap [\mathcal{M}]$, allowing integration of permit defects with correct multiplicity. Gromov-Witten invariants count Axiom R defects (energy concentration along curves) integrated over moduli of stable maps, while Donaldson-Thomas invariants count Axiom Cap defects (capacity violations by sheaves) integrated over Hilbert schemes. This converts enumerative geometry (counting curves and sheaves) into hypostructure defect theory (measuring permit violations), unifying algebraic geometry and dynamical systems under the common language of virtual cycles. $\square$

**Key Insight.** Virtual fundamental classes are the natural setting for permit integration in singular moduli spaces. The obstruction bundle $\text{Ob}^\vee$ encodes the failure modes of the hypostructure: deformations (tangent space) correspond to allowed variations, while obstructions correspond to blocked directions (permit violations). The Euler class $e(\text{Ob}^\vee)$ measures the "signed count" of obstructions, giving the virtual class. Every enumerative invariant (GW, DT, Pandharipande-Thomas, Vafa-Witten) is a permit integral: a weighted count of geometric objects violating specific axioms. The framework reveals that enumerative geometry is the study of controlled permit violations in moduli spaces.
