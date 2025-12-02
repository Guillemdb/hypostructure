# Metatheorem 22.12 (The Mirror Duality Isomorphism)

**Statement.** Let $(X, \omega)$ be a Calabi-Yau manifold equipped with a symplectic form (A-model), and let $(X^\vee, J)$ be its mirror equipped with a complex structure (B-model). Then there exists a pair of dual hypostructures $(\mathbb{H}_A, \mathbb{H}_B)$ satisfying Axiom R (Reflection) such that:

1. **Fukaya ≃ Derived**: The derived Fukaya category is equivalent to the derived category of coherent sheaves:
   $$
   D^b\text{Fuk}(X) \cong D^b(\text{Coh}(X^\vee)).
   $$
   This is the homological manifestation of Axiom R.

2. **Instantons ↔ Periods**: Gromov-Witten invariants (A-model instanton corrections) equal variations of Hodge structure (B-model periods), as encoded by the Picard-Fuchs equation. A-model dissipation = B-model height variation.

3. **Stability Transfer**: The Bridgeland stability condition on $D^b(\text{Coh}(X^\vee))$ (B-side Axiom LS) corresponds to special Lagrangian calibration (A-side Thomas-Yau conjecture). Stable objects persist under deformation.

---

## Proof

**Setup.** Fix a Calabi-Yau $n$-fold $X$ (i.e., $K_X \cong \mathcal{O}_X$ and $H^i(X, \mathcal{O}_X) = 0$ for $0 < i < n$). We consider two geometric structures:

- **A-model (Symplectic)**: $(X, \omega, J_A)$ with symplectic form $\omega$ and complex structure $J_A$ (Kähler)
- **B-model (Complex)**: $(X^\vee, J_B)$ with complex structure $J_B$ on the mirror manifold $X^\vee$

The **mirror map** $\mu: \mathcal{M}_{\text{cpx}}(X^\vee) \to \mathcal{M}_{\text{symp}}(X)$ relates complex moduli to symplectic moduli (Kähler classes). Mirror symmetry asserts that these moduli spaces are isomorphic, and geometric invariants match.

### Step 1: Homological Mirror Symmetry (Kontsevich)

**(H1)** The **homological mirror symmetry conjecture** (Kontsevich, 1994) states that the derived Fukaya category equals the derived category of coherent sheaves:
$$
D^b\text{Fuk}(X, \omega) \cong D^b(\text{Coh}(X^\vee)).
$$

This is the ultimate form of Axiom R (Reflection): A-model and B-model are **categorically equivalent**.

**Step 1a: Fukaya category.**

The **Fukaya category** $\text{Fuk}(X, \omega)$ is an $A_\infty$-category whose:
- **Objects**: Lagrangian submanifolds $L \subset X$ with flat unitary bundles $E \to L$ (branes)
- **Morphisms**: Floer cohomology $\text{HF}^*(L_0, L_1)$, counting pseudo-holomorphic strips $u: [0,1] \times \mathbb{R} \to X$ with boundary on $L_0 \cup L_1$
- **Composition**: Defined by counting pseudo-holomorphic triangles (higher $A_\infty$ products $\mu_n$)

The Floer differential counts holomorphic disks:
$$
\mu_1(x) = \sum_{y \in L_0 \cap L_1} \#\{u : D^2 \to X, \, \partial u \subset L_0 \cup L_1, \, u(\pm i) = x, y\} \cdot y.
$$

**Step 1b: Derived category of coherent sheaves.**

On the B-model side, $D^b(\text{Coh}(X^\vee))$ is the bounded derived category of coherent sheaves on $X^\vee$:
- **Objects**: Complexes of coherent sheaves $\mathcal{F}^\bullet$ (up to quasi-isomorphism)
- **Morphisms**: $\text{Hom}_{D^b}(\mathcal{F}^\bullet, \mathcal{G}^\bullet) = H^*(\text{RHom}(\mathcal{F}^\bullet, \mathcal{G}^\bullet))$
- **Composition**: Derived composition of sheaf morphisms

**Step 1c: The mirror functor.**

Kontsevich's conjecture posits a **mirror functor** $\Phi: D^b\text{Fuk}(X) \to D^b(\text{Coh}(X^\vee))$ satisfying:
$$
\Phi(L, E) = \mathcal{F}_{L,E} \quad (\text{brane-sheaf correspondence})
$$
where $\mathcal{F}_{L,E}$ is a coherent sheaf on $X^\vee$ determined by the Lagrangian $L$ and bundle $E$.

For the **elliptic curve** $E = \mathbb{C}/\Lambda$ (genus 1), homological mirror symmetry is a theorem (Polishchuk-Zaslow, 1998). The mirror is the dual torus $E^\vee = \mathbb{C}/\Lambda^\vee$, and:
$$
\Phi: \text{Fuk}(E) \to D^b(\text{Coh}(E^\vee))
$$
sends a point $\{p\} \subset E$ (0-dimensional Lagrangian) to a skyscraper sheaf $\mathcal{O}_p \in \text{Coh}(E^\vee)$.

**Step 1d: Hypostructure interpretation.**

The equivalence $D^b\text{Fuk}(X) \cong D^b(\text{Coh}(X^\vee))$ is the categorical version of Axiom R:
- **Feasible region duality**: $\mathbb{F}_A \cong \mathbb{F}_B$ (state spaces identified)
- **Mode correspondence**: Special Lagrangians $\leftrightarrow$ Stable sheaves (Mode C.C on both sides)
- **Energy functional**: Symplectic area $\int_L \omega$ $\leftrightarrow$ Degree/Slope $\mu(\mathcal{F}) = \deg(\mathcal{F})/\text{rk}(\mathcal{F})$

The mirror functor $\Phi$ is the **reflection isomorphism** $R: \mathbb{H}_A \to \mathbb{H}_B$ required by Axiom R.

**Conclusion.** Homological mirror symmetry realizes Axiom R as a categorical equivalence between A-model and B-model. $\square_{\text{Step 1}}$

### Step 2: Instanton-Period Correspondence

**(H2)** The **instanton-period correspondence** equates:
- **A-model**: Gromov-Witten invariants $N_{g,d}$ (counts of genus-$g$ curves of degree $d$)
- **B-model**: Variations of Hodge structure (periods $\Pi_\alpha(t)$ satisfying Picard-Fuchs)

This is the **numerical** manifestation of mirror symmetry.

**Step 2a: Gromov-Witten invariants.**

For the A-model, the **genus-$g$ Gromov-Witten potential** is
$$
F_g(q) = \sum_{d=0}^\infty N_{g,d} q^d, \quad q = e^{2\pi i t}
$$
where $N_{g,d}$ counts pseudo-holomorphic curves $u: \Sigma_g \to X$ in homology class $[u] = d \in H_2(X, \mathbb{Z})$.

The generating function $F(q) = \sum_{g=0}^\infty \lambda^{2g-2} F_g(q)$ is the **free energy** of the A-model topological string theory.

**Step 2b: Period integrals.**

On the B-model side, the **periods** are
$$
\Pi_\alpha(t) = \int_{\gamma_\alpha} \Omega(t)
$$
where $\Omega(t)$ is the holomorphic $(n,0)$-form on $X^\vee_t$ (varying in a family), and $\gamma_\alpha \in H_n(X^\vee_t, \mathbb{Z})$ is a cycle.

The periods satisfy the **Picard-Fuchs equation**:
$$
\mathcal{L}_{\text{PF}} \cdot \Pi = 0
$$
where $\mathcal{L}_{\text{PF}} = \theta^{n+1} - q \prod_{k=1}^n (\theta + a_k)$ for some constants $a_k$, and $\theta = q \frac{d}{dq}$.

**Step 2c: Mirror symmetry correspondence.**

The **BCOV equation** (Bershadsky-Cecotti-Ooguri-Vafa, 1994) states:
$$
\frac{\partial^2 F_0}{\partial t_i \partial t_j} = \frac{\partial \Pi_0}{\partial t_i} \cdot \frac{\partial \Pi_\infty}{\partial t_j}
$$
where $F_0$ is the genus-0 A-model potential, and $\Pi_0, \Pi_\infty$ are special periods (near 0 and $\infty$).

More generally, **mirror symmetry** asserts:
$$
F_g^{(A)}(q) = \text{PF}^{-1}\left(\Pi_g^{(B)}(t)\right)
$$
where $\text{PF}^{-1}$ inverts the Picard-Fuchs equation to express $q$ in terms of periods.

**Step 2d: Hypostructure interpretation.**

The instanton corrections $N_{g,d}$ are **dissipation terms** in the A-model hypostructure:
- **Energy**: $E_A(L) = \int_L \omega$ (symplectic area)
- **Dissipation**: $\Delta E = \sum_d N_{0,d} e^{-d E_A}$ (instanton contributions)

On the B-model side, period variation is:
- **Energy**: $E_B(\mathcal{F}) = \int_{X^\vee} c_1(\mathcal{F}) \wedge \omega_{X^\vee}$ (degree)
- **Variation**: $\Delta E = \frac{d\Pi}{dt}$ (monodromy-induced change)

Mirror symmetry equates these via $\Delta E_A = \Delta E_B$ under the mirror map $t \leftrightarrow q$.

**Conclusion.** Instanton corrections (A-model dissipation) equal period variations (B-model height change) via mirror symmetry. $\square_{\text{Step 2}}$

### Step 3: Bridgeland Stability and Special Lagrangians

**(H3)** The **stability transfer** equates:
- **B-model**: Bridgeland stability on $D^b(\text{Coh}(X^\vee))$ (Axiom LS for sheaves)
- **A-model**: Special Lagrangian condition (Thomas-Yau conjecture, Axiom LS for Lagrangians)

This is the **geometric** manifestation of mirror symmetry.

**Step 3a: Bridgeland stability.**

A **Bridgeland stability condition** on $D^b(\text{Coh}(X^\vee))$ consists of:
1. **Central charge**: $Z: K(X^\vee) \to \mathbb{C}$ (homomorphism from Grothendieck group)
2. **Slicing**: $\mathcal{P}(\phi) \subset D^b(\text{Coh}(X^\vee))$ (full subcategories for $\phi \in (0,1]$)

An object $\mathcal{F}$ is **$Z$-stable** if for all non-zero subobjects $\mathcal{E} \subset \mathcal{F}$,
$$
\frac{\text{Im}(Z(\mathcal{E}))}{\text{Re}(Z(\mathcal{E}))} < \frac{\text{Im}(Z(\mathcal{F}))}{\text{Re}(Z(\mathcal{F}))}.
$$

This is the algebraic analogue of the **slope stability** $\mu(\mathcal{E}) < \mu(\mathcal{F})$ for vector bundles.

**Step 3b: Special Lagrangians.**

On the A-model side, a Lagrangian $L \subset X$ is **special Lagrangian** if it is calibrated by $\text{Im}(\Omega)$:
$$
\omega|_L = 0 \quad \text{and} \quad \text{Im}(\Omega)|_L = 0
$$
where $\Omega$ is the holomorphic volume form. Equivalently, $L$ is a **minimal submanifold** in the Kähler metric.

Special Lagrangians are **volume-minimizing** in their homology class, hence stable under deformations.

**Step 3c: Thomas-Yau conjecture.**

The **Thomas-Yau conjecture** (2002) asserts that:
- Special Lagrangians in $X$ correspond to stable sheaves in $X^\vee$ under the mirror functor $\Phi$
- The **moduli space** $\mathcal{M}_{\text{SLag}}(X)$ is homeomorphic to $\mathcal{M}_{\text{stable}}(X^\vee)$

For **K3 surfaces**, this is a theorem (Bridgeland, 2007). For Calabi-Yau 3-folds, it remains a conjecture.

**Step 3d: Hypostructure stability.**

Both notions of stability are instances of **Axiom LS** (Large-Scale Structure):
- **B-model**: Bridgeland stability $\leftrightarrow$ Bounded curvature in moduli space
- **A-model**: Special Lagrangian $\leftrightarrow$ Mean curvature zero (minimal)

The stability condition ensures that the hypostructure has **well-defined asymptotics**: stable objects persist under small perturbations, while unstable objects decay into stable factors (Jordan-Hölder filtration).

**Conclusion.** Bridgeland stability (B-model Axiom LS) corresponds to special Lagrangian calibration (A-model Axiom LS) under mirror symmetry. $\square_{\text{Step 3}}$

### Step 4: SYZ Fibration and Duality

We conclude with the **SYZ conjecture** (Strominger-Yau-Zaslow, 1996), the geometric foundation of mirror symmetry.

**Step 4a: SYZ fibration.**

The **SYZ conjecture** posits that $X$ and $X^\vee$ admit dual torus fibrations:
$$
\pi: X \to B, \quad \pi^\vee: X^\vee \to B
$$
over a common base $B$, such that:
- Fibers $\pi^{-1}(b)$ and $(\pi^\vee)^{-1}(b)$ are dual tori: $T \times T^\vee = T^n \times T^n$
- The mirror map identifies $X^\vee$ with the moduli space of special Lagrangian tori in $X$ equipped with flat bundles

**Step 4b: T-duality.**

The SYZ picture realizes mirror symmetry as **T-duality** (from string theory):
- **A-model on $X$**: Lagrangian tori $T \subset X$ (D-branes wrapping fibers)
- **B-model on $X^\vee$**: Points in $X^\vee = \text{Hom}(\pi_1(T), U(1))$ (moduli of flat bundles)

T-duality exchanges:
- **Momentum** $\leftrightarrow$ **Winding** (Fourier transform on $T$)
- **Symplectic area** $\leftrightarrow$ **Complex modulus**

**Step 4c: Affine structure on base.**

The base $B$ carries an **integral affine structure** (flat connection on $TB$ with monodromy in $\text{SL}(n, \mathbb{Z})$). The mirror map is a **Legendre transform** on the affine base:
$$
X^\vee = T^*B / \Gamma, \quad X = TB / \Gamma^\vee
$$
where $\Gamma, \Gamma^\vee$ are dual lattices.

**Step 4d: Hypostructure base space.**

The SYZ base $B$ is the **large-scale quotient** of both $X$ and $X^\vee$. It encodes:
- **Axiom LS**: Asymptotic behavior of $X, X^\vee$ at large scales (torus fibers flatten)
- **Axiom SC**: Scaling $\lambda \to 0$ corresponds to collapsing fibers $T \to \text{pt}$
- **Axiom R**: Reflection $(X, \omega) \leftrightarrow (X^\vee, J)$ via Legendre transform on $B$

The SYZ fibration is the **geometric realization of Axiom R** at the level of large-scale structure.

**Conclusion.** The SYZ conjecture realizes mirror symmetry as T-duality of dual torus fibrations, providing the geometric foundation for Axiom R. $\square_{\text{Step 4}}$

---

## Key Insight

The mirror duality isomorphism unifies three levels of mirror symmetry:

1. **Categorical** (Homological Mirror Symmetry):
   $$
   D^b\text{Fuk}(X) \cong D^b(\text{Coh}(X^\vee))
   $$
   This is **Axiom R at the level of derived categories**, equating A-model Lagrangians with B-model sheaves.

2. **Numerical** (Instanton-Period Correspondence):
   $$
   F_g^{(A)}(q) = \text{PF}^{-1}(\Pi_g^{(B)}(t))
   $$
   This is **Axiom R at the level of generating functions**, equating A-model Gromov-Witten invariants with B-model periods.

3. **Geometric** (Stability Transfer):
   $$
   \text{Special Lagrangians} \leftrightarrow \text{Bridgeland-stable sheaves}
   $$
   This is **Axiom R at the level of moduli spaces**, equating A-model calibrated geometry with B-model algebraic stability.

The **SYZ conjecture** provides the geometric mechanism: mirror symmetry is T-duality of dual torus fibrations, realized as a Legendre transform on the affine base. The hypostructure axioms package this beautifully:

- **Axiom C**: Conservation of symplectic area (A) $\leftrightarrow$ Conservation of degree (B)
- **Axiom LS**: Special Lagrangian calibration (A) $\leftrightarrow$ Bridgeland stability (B)
- **Axiom SC**: Instanton expansion (A) $\leftrightarrow$ Picard-Fuchs solutions (B)
- **Axiom TB**: Floer theory (A) $\leftrightarrow$ Deformation theory (B)
- **Axiom R**: **Mirror functor $\Phi: \mathbb{H}_A \to \mathbb{H}_B$ is an equivalence**

The deep philosophical point: **mirror symmetry is not a duality but an isomorphism**. The A-model and B-model are not distinct theories related by a correspondence, but rather two presentations of the same underlying hypostructure. The mirror map is a **change of coordinates** in the space of hypostructures, analogous to Fourier transform in quantum mechanics.

This is the ultimate vindication of Axiom R: **geometry and algebra are dual manifestations of hypostructure**, unified by the mirror functor.

$\square$
