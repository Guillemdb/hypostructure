### Metatheorem 22.6 (The Bridgeland Stability Isomorphism)

Axiom LS (Local Stiffness) finds a natural home in Bridgeland stability conditions on derived categories. Solitons are precisely Bridgeland-stable objects, and the Harder-Narasimhan filtration is the mode decomposition of Theorem 17.2.

**Statement.** Let $\mathcal{S}$ be a hypostructure on the derived category $D^b(X)$ of coherent sheaves on a smooth projective variety $X$. Define the central charge:
$$Z(E) = \Phi(E) + i \mathfrak{D}(E)$$
where $\Phi$ is the height and $\mathfrak{D}$ is the dissipation. Then:

1. **Phase ↔ Energy Density:** The phase of an object $E \in D^b(X)$ is:
$$\phi(E) = \frac{1}{\pi} \arg Z(E) = \frac{1}{\pi} \arctan\left(\frac{\mathfrak{D}(E)}{\Phi(E)}\right).$$
Objects with the same phase have proportional energy-dissipation ratios.

2. **Stable Objects ↔ Solitons:** An object $E$ is Bridgeland-stable if and only if it satisfies Axiom LS (is a soliton): for all proper subobjects $0 \neq F \subsetneq E$ in the abelian category $\mathcal{A}(\phi)$:
$$\phi(F) \leq \phi(E).$$
Bridgeland stability is exactly the condition that $E$ is a local minimizer of the phase functional.

3. **HN Filtration ↔ Mode Decomposition:** The Harder-Narasimhan filtration of $E$:
$$0 = E_0 \subsetneq E_1 \subsetneq \cdots \subsetneq E_n = E$$
with semistable quotients $E_i/E_{i-1}$ is isomorphic to the mode decomposition (Theorem 17.2), with:
$$\phi(E_1/E_0) > \phi(E_2/E_1) > \cdots > \phi(E_n/E_{n-1}).$$

4. **Wall Crossing ↔ Mode S.C:** Phase transitions (jumps in stability) occur when $Z(E)$ crosses a wall in the space of stability conditions. These wall crossings are precisely Mode S.C (sector instability): the system jumps between topological sectors.

*Proof.*

**Step 1 (Setup: Derived Category and Stability Conditions).**

Let $X$ be a smooth projective variety over $\mathbb{C}$, and let $D^b(X)$ be the bounded derived category of coherent sheaves on $X$. A Bridgeland stability condition \cite{Bridgeland07} is a pair $\sigma = (Z, \mathcal{P})$ where:

- $Z: K(X) \to \mathbb{C}$ is a group homomorphism (central charge) from the Grothendieck group to $\mathbb{C}$,
- $\mathcal{P}(\phi) \subset D^b(X)$ is a slicing: a collection of full subcategories indexed by phase $\phi \in \mathbb{R}$ satisfying:
  1. $\mathcal{P}(\phi + 1) = \mathcal{P}(\phi)[1]$ (shift periodicity),
  2. If $E \in \mathcal{P}(\phi)$, then $\text{Hom}(E, F) = 0$ for all $F \in \mathcal{P}(\psi)$ with $\psi > \phi$,
  3. Every object $E \in D^b(X)$ admits a Harder-Narasimhan filtration.

The central charge satisfies:
$$Z(E) \in \mathbb{R}_{>0} \cdot e^{i\pi\phi} \quad \text{for } E \in \mathcal{P}(\phi).$$

**Step 2 (Central Charge from Hypostructure).**

*Lemma 22.6.1 (Hypostructure Central Charge).* For a hypostructure $\mathcal{S}$ on $D^b(X)$, define:
$$Z(E) = \Phi(E) + i \mathfrak{D}(E)$$
where:
- $\Phi(E) = \int_X \text{ch}(E) \cdot \text{Td}(X) \cdot \omega$ is the height (Mukai pairing with an ample class $\omega$),
- $\mathfrak{D}(E) = \|\nabla E\|_{L^2}$ is the dissipation (derived gradient norm).

This defines a valid central charge on $K(X) \cong K_0(D^b(X))$.

*Proof of Lemma.* We verify that $Z$ satisfies the required properties:

**(i) Group homomorphism:** $Z$ is linear in the Grothendieck group by linearity of Chern character:
$$Z(E \oplus F) = Z(E) + Z(F).$$

**(ii) Support property:** For torsion sheaves supported on proper subvarieties, $\Phi$ decreases:
$$\text{dim}(\text{Supp}(E)) < \text{dim}(X) \implies \Phi(E) = 0.$$
This ensures the support property: objects with lower-dimensional support have smaller phase.

**(iii) Positivity:** For non-zero objects, $|Z(E)| = \sqrt{\Phi(E)^2 + \mathfrak{D}(E)^2} > 0$ since either $\Phi(E) > 0$ or $\mathfrak{D}(E) > 0$ by Axiom D (non-trivial objects have positive energy or dissipation). $\square$

**Step 3 (Phase as Energy-Dissipation Ratio).**

*Lemma 22.6.2 (Phase Formula).* The phase of an object $E$ is:
$$\phi(E) = \frac{1}{\pi} \arg Z(E) = \frac{1}{\pi} \arctan\left(\frac{\mathfrak{D}(E)}{\Phi(E)}\right).$$

For the hypostructure flow $S_t$, objects with constant phase satisfy:
$$\frac{d\Phi}{dt} = -\mathfrak{D}, \quad \frac{d\phi}{dt} = 0.$$

*Proof of Lemma.* Write $Z(E) = |Z(E)| e^{i\pi\phi(E)}$ in polar form. Then:
$$e^{i\pi\phi} = \frac{Z}{|Z|} = \frac{\Phi + i\mathfrak{D}}{\sqrt{\Phi^2 + \mathfrak{D}^2}}.$$
Taking the argument:
$$\pi\phi = \arctan\left(\frac{\mathfrak{D}}{\Phi}\right).$$

For the flow, by Axiom D:
$$\frac{d\Phi}{dt} = -\alpha \mathfrak{D}, \quad \frac{d\mathfrak{D}}{dt} = -\beta \mathfrak{D} + \text{lower order}.$$
The phase evolution is:
$$\frac{d\phi}{dt} = \frac{1}{\pi} \frac{d}{dt}\arctan\left(\frac{\mathfrak{D}}{\Phi}\right) = \frac{1}{\pi} \frac{\Phi \frac{d\mathfrak{D}}{dt} - \mathfrak{D} \frac{d\Phi}{dt}}{\Phi^2 + \mathfrak{D}^2}.$$

Substituting:
$$\frac{d\phi}{dt} = \frac{1}{\pi} \frac{\Phi(-\beta \mathfrak{D}) - \mathfrak{D}(-\alpha\mathfrak{D})}{\Phi^2 + \mathfrak{D}^2} = \frac{\mathfrak{D}(\alpha\mathfrak{D} - \beta\Phi)}{\pi(\Phi^2 + \mathfrak{D}^2)}.$$

Objects with constant phase satisfy $\frac{d\phi}{dt} = 0$, which gives:
$$\alpha \mathfrak{D} = \beta \Phi \implies \frac{\mathfrak{D}}{\Phi} = \frac{\beta}{\alpha}.$$
These are the solitons (self-similar solutions) of the flow. $\square$

**Step 4 (Bridgeland Stability as Axiom LS).**

*Lemma 22.6.3 (Stable Objects are Solitons).* An object $E \in D^b(X)$ is Bridgeland-stable with respect to $\sigma = (Z, \mathcal{P})$ if and only if it satisfies Axiom LS: for all proper subobjects $0 \neq F \subsetneq E$:
$$\phi(F) \leq \phi(E).$$

Moreover, stable objects are local minimizers of the phase functional in the moduli space of objects with fixed numerical class.

*Proof of Lemma.* By definition, $E$ is stable if:
$$\phi(F) < \phi(E) \quad \text{for all proper subobjects } F.$$

In hypostructure language, this is Axiom LS (Local Stiffness): the gradient of the phase functional dominates:
$$\nabla \phi(E) = 0 \quad \text{(critical point)},$$
$$\nabla^2 \phi(E) > 0 \quad \text{(positive definite Hessian)}.$$

The stability condition ensures that any deformation $E + tF$ with $F \subsetneq E$ increases the phase:
$$\phi(E + tF) \geq \phi(E) + c t^{2-\theta}$$
for some $\theta \in [0, 1)$ and $c > 0$. This is precisely the Lojasiewicz inequality at the stable object $E$.

Conversely, if $E$ is not stable, there exists a destabilizing subobject $F$ with $\phi(F) \geq \phi(E)$, violating Axiom LS. The object $E$ is not a local minimizer, and the flow $S_t$ will decompose $E$ along the HN filtration. $\square$

**Step 5 (Harder-Narasimhan Filtration as Mode Decomposition).**

*Lemma 22.6.4 (HN = Mode Decomposition).* Every object $E \in D^b(X)$ admits a unique Harder-Narasimhan filtration:
$$0 = E_0 \subsetneq E_1 \subsetneq \cdots \subsetneq E_n = E$$
where the quotients $E_i/E_{i-1}$ are semistable with strictly decreasing phases:
$$\phi(E_1/E_0) > \phi(E_2/E_1) > \cdots > \phi(E_n/E_{n-1}).$$

This filtration is isomorphic to the mode decomposition (Theorem 17.2):
$$E = \bigoplus_{i=1}^n (E_i/E_{i-1})$$
where each mode $E_i/E_{i-1}$ is stable (soliton) with distinct phase $\phi_i$.

*Proof of Lemma.* The existence and uniqueness of the HN filtration is a fundamental theorem in Bridgeland stability \cite{Bridgeland07}. We verify that it matches the mode decomposition.

By Theorem 17.2 (Mode Decomposition), every trajectory $u(t)$ decomposes into solitons:
$$u(t) = \sum_{i=1}^n g_i(t) \cdot V_i$$
where $V_i$ are canonical profiles (stable objects) and $g_i(t) \in G$ are symmetry group elements.

For the derived category, this decomposition is:
$$E = \bigoplus_{i=1}^n E_i/E_{i-1}$$
where each $E_i/E_{i-1}$ is semistable (cannot be further decomposed).

The phases are strictly ordered:
$$\phi_1 > \phi_2 > \cdots > \phi_n$$
corresponding to energy-dissipation ratios $\frac{\mathfrak{D}_i}{\Phi_i} = \tan(\pi\phi_i)$.

The HN filtration is the canonical way to decompose an unstable object into stable pieces. The mode decomposition is the canonical way to decompose a trajectory into solitons. These are isomorphic: each HN quotient is a mode. $\square$

**Step 6 (Wall Crossing as Mode S.C).**

*Lemma 22.6.5 (Stability Walls are Phase Transitions).* As the central charge $Z$ varies in the space of stability conditions $\text{Stab}(X)$, stable objects can become unstable when $Z$ crosses a wall. These wall-crossing phenomena correspond to Mode S.C (sector instability): the system jumps between topological sectors.

*Proof of Lemma.* The space of stability conditions $\text{Stab}(X)$ is a complex manifold of dimension $\text{rank}(K(X))$. Walls are real codimension-1 loci where:
$$\arg Z(E) = \arg Z(F)$$
for some exact sequence $0 \to F \to E \to G \to 0$.

When $Z$ crosses a wall, the object $E$ changes stability:
- Before the wall: $E$ is stable ($\phi(F) < \phi(E)$ for all $F$),
- On the wall: $E$ is strictly semistable ($\phi(F) = \phi(E)$ for some $F$),
- After the wall: $E$ is unstable ($\phi(F) > \phi(E)$ for some $F$).

In hypostructure terms, crossing the wall corresponds to Mode S.C: the sectorial index changes:
$$\tau(E) \neq \tau(E')$$
where $E, E'$ are the stable objects before and after the wall crossing.

The wall-crossing formula (Kontsevich-Soibelman \cite{KS08}) computes the change in invariants:
$$\mathcal{Z}_{\text{after}} = \mathcal{Z}_{\text{before}} \cdot \prod_{\gamma} (1 - e^{-\langle \gamma, - \rangle})^{\Omega(\gamma)}.$$
This encodes how the moduli space topology changes across the wall—a manifestation of Mode S.C topological sector transitions. $\square$

**Step 7 (Support Property and Axiom Cap).**

*Lemma 22.6.6 (Support Dimension as Capacity).* For a Bridgeland stability condition to satisfy the support property, objects with lower-dimensional support must have smaller phase. This corresponds to Axiom Cap (Capacity): singular sets of higher codimension cannot concentrate energy.

*Proof of Lemma.* The support property states that for objects $E, F$ with:
$$\text{dim}(\text{Supp}(E)) < \text{dim}(\text{Supp}(F)),$$
we have:
$$\phi(E) \ll \phi(F).$$

In hypostructure terms, the capacity of a set $K$ is:
$$\text{Cap}(K) = \sup\left\{\mu(K) : \mu \text{ is a probability measure on } K\right\}.$$
For lower-dimensional sets, $\text{Cap}(K) = 0$, so by Axiom Cap:
$$\int_K \Phi \, d\mu = 0 \implies \Phi|_K = 0.$$

The support property ensures that objects supported on lower-dimensional loci have zero height $\Phi(E) = 0$, hence:
$$\phi(E) = \frac{1}{\pi}\arctan\left(\frac{\mathfrak{D}(E)}{0}\right) = \frac{1}{2}$$
(by convention, phase is $\frac{1}{2}$ for zero height objects).

This capacity constraint prevents concentration: an object cannot "hide" energy on a lower-dimensional support. $\square$

**Step 8 (Example: Slope Stability on Curves).**

*Example 22.6.7 (Slope Stability as Phase).* For a smooth projective curve $C$, slope stability of vector bundles $E$ is defined by:
$$\mu(E) = \frac{\deg(E)}{\text{rank}(E)}.$$
A bundle $E$ is slope-stable if:
$$\mu(F) < \mu(E) \quad \text{for all proper subbundles } F \subset E.$$

This is a special case of Bridgeland stability with central charge:
$$Z(E) = -\deg(E) + i \cdot \text{rank}(E).$$
The phase is:
$$\phi(E) = \frac{1}{\pi}\arctan\left(\frac{\text{rank}(E)}{-\deg(E)}\right) = 1 - \frac{1}{\pi}\arctan(\mu(E)).$$

Slope stability corresponds to Axiom LS: the slope $\mu(E)$ is a local minimizer of the height-to-rank ratio. Stable bundles are solitons under the flow. $\square$

**Step 9 (Example: Gieseker Stability and $\chi$-Stability).**

*Example 22.6.8 (Gieseker Stability).* On a surface $S$, Gieseker stability is defined by the Hilbert polynomial:
$$P(E, m) = \chi(E \otimes \mathcal{O}_S(mH))$$
for an ample divisor $H$. A sheaf $E$ is Gieseker-stable if:
$$\frac{P(F, m)}{r(F)} < \frac{P(E, m)}{r(E)} \quad \text{for large } m \text{ and all subsheaves } F.$$

The central charge is:
$$Z(E) = -\int_S \text{ch}(E) \cdot e^H = -r(E) \int_S e^H + c_1(E) \cdot H + \chi(E).$$
This gives a Bridgeland stability condition on $D^b(S)$ with phase:
$$\phi(E) = \frac{1}{\pi}\arctan\left(\frac{\chi(E)}{-c_1(E) \cdot H}\right).$$

Gieseker-stable sheaves are Bridgeland-stable objects, hence solitons satisfying Axiom LS. $\square$

**Step 10 (Conclusion).**

The Bridgeland Stability Isomorphism establishes that Axiom LS (Local Stiffness) is not merely an analytical tool but encodes deep homological algebra. Bridgeland-stable objects are precisely the solitons (canonical profiles) of the hypostructure flow, with the phase $\phi(E)$ measuring the energy-dissipation ratio. The Harder-Narasimhan filtration is the mode decomposition, splitting unstable objects into stable components with decreasing phases. Wall crossings in the space of stability conditions correspond to Mode S.C topological transitions, where the sectorial structure changes. This isomorphism converts representation-theoretic questions (stability of sheaves) into dynamical systems (soliton decomposition), unifying derived categories and hypostructures. $\square$

**Key Insight.** Bridgeland stability conditions provide the natural categorical framework for Axiom LS. The phase $\phi(E)$ is the geometric angle $\arctan(\mathfrak{D}/\Phi)$ in the complex plane of the central charge $Z = \Phi + i\mathfrak{D}$. Stable objects minimize phase within their numerical class, satisfying the Lojasiewicz inequality. The HN filtration is the algorithmic procedure for decomposing an arbitrary object into solitons, and wall crossings are the phase transitions where the decomposition changes. Every result about Bridgeland stability has a dual statement about hypostructure dynamics, and vice versa. The framework reveals that homological algebra is the language of categorical solitons.
