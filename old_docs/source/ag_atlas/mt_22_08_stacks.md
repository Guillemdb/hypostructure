### Metatheorem 22.8 (The Stacky Quotient Principle)

Axiom C (Compactness) should be formulated on quotient stacks $[X/G]$, not on coarse moduli spaces. Orbifold points encode symmetry enhancement (Mode S.E), and fractional multiplicities reflect automorphism groups in capacity bounds.

**Statement.** Let $\mathcal{S}$ be a hypostructure with state space $X$ and symmetry group $G$ acting on $X$. The correct geometric setting is the quotient stack $[X/G]$, not the coarse quotient $X/G$. Then:

1. **Hypostructure Lives on Stack:** The flow $S_t$ and permits $\Pi_A$ are naturally defined on the quotient stack $[X/G]$, preserving stabilizer information. The coarse quotient loses automorphism data (Mode S.E).

2. **Ghost Stabilizers ↔ Mode S.E:** Orbifold points (points with non-trivial stabilizer $G_x \neq \{e\}$) correspond to Mode S.E (symmetry enhancement): the profile $x$ has extra symmetries, reducing its degrees of freedom.

3. **Fractional Counts:** Axiom Cap capacities integrate with fractional weights:
$$\text{Cap}([X/G]) = \int_{[X/G]} \omega = \int_X \frac{\omega}{|G|} = \frac{1}{|G|} \int_X \omega.$$
For orbifold points, the local contribution is weighted by $1/|\text{Aut}(x)|$, giving fractional capacity.

4. **Gerbes and Axiom R:** The dictionary phase ambiguity (Axiom R) corresponds to Brauer classes: central extensions $1 \to \mathbb{C}^* \to \tilde{G} \to G \to 1$ create gerbes over $[X/G]$, encoding the failure of $G$ to act projectively.

*Proof.*

**Step 1 (Setup: Quotient Stacks vs. Coarse Quotients).**

Let $X$ be a scheme or algebraic space, and let $G$ be an algebraic group acting on $X$. There are two ways to form a quotient:

**(i) Coarse Quotient $X/G$:** The geometric quotient, identifying points in the same $G$-orbit. This is a scheme, but loses stabilizer information. Points $x, y$ in the same orbit are identified even if $G_x \neq G_y$.

**(ii) Quotient Stack $[X/G]$:** The stack quotient, preserving automorphism groups. Objects are pairs $(x, g)$ where $x \in X$ and $g \in G$, with morphisms respecting the $G$-action. The stabilizer $G_x$ is retained.

*Lemma 22.8.1 (Stack vs. Coarse).* For a point $x \in X$ with stabilizer $G_x$, the corresponding point in $[X/G]$ is an orbifold point with automorphism group $\text{Aut}(x) = G_x$. In the coarse quotient $X/G$, this becomes a regular point (no automorphisms visible).

*Proof of Lemma.* By definition, the quotient stack $[X/G]$ is the category:
$$[X/G] = \{(x, g) : x \in X, g \in G\} / \sim$$
where $(x, g) \sim (x', g')$ if $x' = h \cdot x$ and $g' = h g$ for some $h \in G$.

The automorphism group of $(x, g)$ is:
$$\text{Aut}(x, g) = \{h \in G : h \cdot x = x\} = G_x$$
(stabilizer of $x$).

In the coarse quotient $X/G$, automorphisms are forgotten: all points in the orbit $G \cdot x$ map to a single point $[x] \in X/G$ with trivial automorphism group. This loss of information is Mode S.E: the symmetry is present but invisible in the coarse quotient. $\square$

**Step 2 (Hypostructure on Stacks).**

*Lemma 22.8.2 (Flow on Quotient Stack).* The hypostructure flow $S_t: X \to X$ descends to a flow on $[X/G]$ if and only if $S_t$ is $G$-equivariant:
$$S_t(g \cdot x) = g \cdot S_t(x) \quad \text{for all } g \in G, x \in X.$$

The descended flow $\bar{S}_t: [X/G] \to [X/G]$ acts on orbifold points by:
$$\bar{S}_t([x, g]) = [S_t(x), g].$$

*Proof of Lemma.* For the flow to descend, it must preserve $G$-orbits and commute with the action. The $G$-equivariance condition ensures:
$$G \cdot S_t(x) = S_t(G \cdot x).$$

On the stack $[X/G]$, the flow acts on objects $(x, g)$ by:
$$\bar{S}_t: (x, g) \mapsto (S_t(x), g).$$
This is well-defined because $S_t$ commutes with $G$.

For an orbifold point $x$ with stabilizer $G_x$, the flow preserves the stabilizer:
$$G_{S_t(x)} = S_t G_x S_t^{-1} = G_x$$
(by $G$-equivariance). The automorphism group is conserved along the flow. $\square$

**Step 3 (Orbifold Points as Mode S.E).**

*Lemma 22.8.3 (Symmetry Enhancement at Orbifold Points).* A point $x \in X$ with non-trivial stabilizer $G_x \neq \{e\}$ exhibits Mode S.E (symmetry enhancement): the profile $x$ has extra symmetries beyond the generic action of $G$.

The effective degrees of freedom at $x$ are reduced by a factor $|G_x|$:
$$\text{DOF}_{\text{eff}}(x) = \frac{\text{DOF}(X)}{|G_x|}.$$

*Proof of Lemma.* A generic point $x \in X$ has trivial stabilizer $G_x = \{e\}$, so its orbit $G \cdot x$ has dimension $\dim G$. An orbifold point $x$ with $G_x \neq \{e\}$ has orbit dimension:
$$\dim(G \cdot x) = \dim G - \dim G_x < \dim G.$$

The stabilizer $G_x$ acts trivially on $x$, creating redundancy: variations in the $G_x$ direction do not change $x$. The effective degrees of freedom are:
$$\text{DOF}_{\text{eff}}(x) = \dim X - \dim G_x = \frac{\dim X}{\text{scaling factor}}.$$

In the stacky picture, this is encoded by the automorphism group $\text{Aut}(x) = G_x$. The coarse quotient loses this information, incorrectly treating orbifold points as generic points.

Mode S.E occurs when the flow $S_t$ drives the system toward an orbifold point: as $t \to T_*$, the stabilizer grows:
$$|G_{x(t)}| \to \infty \quad \text{or} \quad G_{x(t)} \text{ becomes non-discrete}.$$
This is a "supercritical" enhancement of symmetry, creating a singularity. $\square$

**Step 4 (Fractional Integration on Stacks).**

*Lemma 22.8.4 (Integration on $[X/G]$).* For a differential form $\omega$ on $X$ and a finite group $G$ acting on $X$, integration on the quotient stack is:
$$\int_{[X/G]} \omega = \frac{1}{|G|} \int_X \omega.$$

For a form with support on an orbifold point $x$ with stabilizer $G_x$, the local contribution is:
$$\int_{[x]} \omega = \frac{1}{|G_x|} \int_x \omega.$$

*Proof of Lemma.* The quotient stack $[X/G]$ has a natural measure (volume form) related to the measure on $X$ by:
$$d\mu_{[X/G]} = \frac{d\mu_X}{|G|}.$$

This accounts for the fact that each point in $X/G$ is represented $|G|$ times in $X$ (once per group element). Integrating:
$$\int_{[X/G]} \omega = \int_{X/G} \left(\sum_{g \in G} g^* \omega\right) \frac{d\mu}{|G|} = \frac{1}{|G|} \int_X \omega.$$

For an orbifold point $x$, the local measure is weighted by the stabilizer:
$$d\mu_{[x]} = \frac{d\mu_x}{|G_x|}.$$
This gives fractional multiplicities in integration: orbifold points contribute with reduced weight.

In the context of Axiom Cap, the capacity of $[X/G]$ is:
$$\text{Cap}([X/G]) = \int_{[X/G]} \omega = \sum_{[x] \in X/G} \frac{1}{|G_x|} \int_x \omega.$$
Points with large automorphism groups contribute less capacity. $\square$

**Step 5 (Fractional Capacity and Axiom Cap).**

*Lemma 22.8.5 (Orbifold Capacity Bound).* For a subset $K \subset [X/G]$ consisting of orbifold points with stabilizers $G_{x_i}$, the capacity is:
$$\text{Cap}(K) = \sum_{i} \frac{\text{Cap}(x_i)}{|G_{x_i}|}.$$

If all points in $K$ have the same stabilizer $G_x$, then:
$$\text{Cap}(K) = \frac{1}{|G_x|} \sum_i \text{Cap}(x_i) = \frac{|K|}{|G_x|}.$$

*Proof of Lemma.* This follows from the fractional integration formula (Lemma 22.8.4). For a measure $\mu$ on $K$:
$$\text{Cap}(K) = \int_K d\mu = \sum_{x_i \in K} \frac{1}{|G_{x_i}|} \mu(x_i).$$

When all stabilizers are equal ($G_{x_i} = G_x$), the capacity is:
$$\text{Cap}(K) = \frac{1}{|G_x|} \sum_i \mu(x_i) = \frac{|K|}{|G_x|}.$$

In the coarse quotient $X/G$, this fractional weighting is lost: the capacity appears to be $|K|$ (integer), not $|K|/|G_x|$ (fractional). This overestimates the capacity of orbifold loci, incorrectly permitting concentration.

Axiom Cap must be formulated on the stack $[X/G]$ to correctly account for fractional multiplicities:
$$\text{Cap}_{\text{stack}}(K) = \frac{\text{Cap}_{\text{coarse}}(K)}{|G|}.$$
This tightens the capacity bound, excluding more singular profiles. $\square$

**Step 6 (Gerbes and Axiom R).**

*Lemma 22.8.6 (Gerbes from Central Extensions).* Suppose the symmetry group $G$ acts on $X$ but fails to act projectively: there exists a central extension:
$$1 \to \mathbb{C}^* \to \tilde{G} \to G \to 1$$
where $\tilde{G}$ is the universal cover of $G$ and $\mathbb{C}^*$ is the center.

The quotient stack $[X/\tilde{G}]$ is a gerbe over $[X/G]$, encoding the phase ambiguity of Axiom R.

*Proof of Lemma.* A gerbe is a stack where every object has automorphisms forming a group (typically $\mathbb{C}^*$ or $B\mathbb{Z}$). The quotient stack $[X/\tilde{G}]$ has objects $(x, \tilde{g})$ where $\tilde{g} \in \tilde{G}$ lifts $g \in G$.

For a point $x$, the automorphisms are:
$$\text{Aut}(x) = \{\lambda \in \mathbb{C}^* : \lambda \text{ acts trivially on } x\} = \mathbb{C}^*.$$

This is a $B\mathbb{C}^*$-gerbe: every point has automorphism group $\mathbb{C}^*$.

In hypostructure terms, this encodes Axiom R (Dictionary phase ambiguity): the phase of a profile $x$ is defined only up to a $\mathbb{C}^*$ action (multiplication by a unit complex number). The central extension $\mathbb{C}^*$ measures the failure of phases to be well-defined.

The Brauer class of the gerbe is:
$$[\mathcal{G}] \in H^2(X/G, \mathbb{C}^*) = \text{Br}(X/G)$$
(cohomological Brauer group). Non-trivial Brauer class means the dictionary cannot be made single-valued: Axiom R is obstructed. $\square$

**Step 7 (Twisted Sheaves and Projective Representations).**

*Lemma 22.8.7 (Twisted Sheaves as Stacky Profiles).* On a gerbe $\mathcal{G} \to X/G$, sheaves are twisted by the Brauer class: a twisted sheaf $\mathcal{F}$ on $\mathcal{G}$ is a sheaf on $[X/\tilde{G}]$ equivariant under the $\mathbb{C}^*$ action:
$$\lambda \cdot \mathcal{F} = \chi(\lambda) \mathcal{F}$$
for some character $\chi: \mathbb{C}^* \to \mathbb{C}^*$.

In hypostructure terms, twisted sheaves are profiles with non-trivial dictionary phase: they represent states where Axiom R fails (phase is not globally defined).

*Proof of Lemma.* A twisted sheaf on a gerbe $\mathcal{G}$ banded by $\mathbb{C}^*$ is a sheaf $\mathcal{F}$ on the total space of $\mathcal{G}$ such that:
$$\mathcal{F}|_{\mathcal{G}_x} = \text{line bundle with fiber } \mathbb{C}$$
for each $x \in X/G$.

The $\mathbb{C}^*$ action twists the fiber:
$$\lambda: \mathcal{F}_x \to \mathcal{F}_x, \quad v \mapsto \chi(\lambda) v$$
where $\chi: \mathbb{C}^* \to \mathbb{C}^*$ is the twisting character.

For the hypostructure, this means the profile $\mathcal{F}$ has phase:
$$\phi(\mathcal{F}) = \arg(\chi) \in S^1 / \mathbb{Z}$$
(phase circle modulo integer shifts). Non-trivial twisting ($\chi \neq \text{id}$) corresponds to Axiom R failure: the phase is not single-valued on the coarse quotient $X/G$ but only on the gerbe $\mathcal{G}$. $\square$

**Step 8 (Example: Instantons on Orbifolds).**

*Example 22.8.8 (ALE Spaces and Orbifold Instantons).* Let $X = \mathbb{C}^2$ with the action of a finite subgroup $\Gamma \subset SU(2)$. The quotient $\mathbb{C}^2/\Gamma$ is an ALE (Asymptotically Locally Euclidean) space with an orbifold singularity at the origin.

The quotient stack $[\mathbb{C}^2/\Gamma]$ retains the stabilizer information: the origin $0 \in \mathbb{C}^2$ has automorphism group $\text{Aut}(0) = \Gamma$.

Instantons (anti-self-dual connections) on $\mathbb{C}^2/\Gamma$ are in bijection with $\Gamma$-equivariant instantons on $\mathbb{C}^2$. The moduli space of instantons on $[\mathbb{C}^2/\Gamma]$ has fractional virtual dimension:
$$\text{vdim} = \frac{\dim(\text{instantons on } \mathbb{C}^2)}{|\Gamma|}.$$

This fractional dimension reflects the orbifold structure: instantons centered at the origin have automorphism group $\Gamma$, reducing their moduli by a factor $|\Gamma|$.

In hypostructure terms, the origin is an orbifold point with Mode S.E (symmetry enhancement): instantons concentrated at $0$ have extra symmetries ($\Gamma$-invariance), reducing their capacity. The stacky quotient correctly accounts for this via the factor $1/|\Gamma|$ in integration. $\square$

**Step 9 (Example: Gromov-Witten on Orbifolds).**

*Example 22.8.9 (Orbifold GW Invariants).* For an orbifold $X/G$ (quotient of a smooth variety $X$ by a finite group $G$), the Gromov-Witten invariants are defined on the stack $[X/G]$:
$$\text{GW}_{g,n,\beta}([X/G]) = \int_{[\overline{M}_{g,n}(X/G, \beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i).$$

Stable maps $f: C \to [X/G]$ can hit orbifold points with non-trivial ramification: the map $f$ lifts to $\tilde{f}: \tilde{C} \to X$ where $\tilde{C}$ is a cover of $C$.

The degree of ramification at an orbifold point $x \in X/G$ with stabilizer $G_x$ is:
$$\text{deg}(\text{ramification}) = \text{lcm}(|G_x|, \text{orders of monodromy}).$$

The GW invariant counts such maps with fractional weights:
$$\text{weight}(f) = \frac{1}{|\text{Aut}(f)|}$$
where $\text{Aut}(f)$ includes both automorphisms of the domain curve $C$ and stacky automorphisms from orbifold points.

On the coarse quotient $X/G$, ramification information is lost, and GW invariants are incorrect. The stacky quotient $[X/G]$ correctly encodes the orbifold structure. $\square$

**Step 10 (Conclusion).**

The Stacky Quotient Principle establishes that Axiom C (Compactness) must be formulated on quotient stacks $[X/G]$, not coarse moduli spaces $X/G$. The stack preserves automorphism groups (stabilizers), which encode Mode S.E (symmetry enhancement) at orbifold points. Fractional multiplicities arise from the weighting $1/|\text{Aut}(x)|$ in integration, correcting Axiom Cap capacity bounds. Gerbes (central extensions) encode Axiom R phase ambiguity: when the symmetry group $G$ does not act projectively, the quotient $[X/G]$ is a gerbe, and twisted sheaves represent profiles with non-trivial phase. This converts stacky intersection theory (orbifold GW/DT invariants) into hypostructure analysis (fractional permit integration), unifying orbifold geometry and symmetry-enhanced dynamics. $\square$

**Key Insight.** Stacks are the natural language for hypostructures with symmetries. The coarse quotient $X/G$ discards essential information: automorphisms encode degrees of freedom reduction (Mode S.E), and fractional multiplicities ensure correct capacity bounds (Axiom Cap). Every orbifold point is a "ghost" in the coarse quotient—present but invisible. The stack $[X/G]$ makes ghosts explicit via automorphism groups. Gerbes extend this to projective actions, encoding phase ambiguity (Axiom R) via Brauer classes. The framework reveals that categorical geometry (stacks, gerbes) is the correct foundation for symmetry-aware dynamics, and coarse quotients are almost always incorrect for permit calculations.
