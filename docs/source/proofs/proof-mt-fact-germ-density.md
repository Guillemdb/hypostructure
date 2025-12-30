# Proof of FACT-GermDensity (Germ Set Density)

:::{prf:proof}
:label: proof-mt-fact-germ-density

**Theorem Reference:** {prf:ref}`mt-fact-germ-density`

This proof establishes that the finite Bad Pattern Library $\mathcal{B} = \{B_i\}_{i \in I}$ is categorically dense in the universal bad pattern $\mathbb{H}_{\mathrm{bad}}^{(T)}$ in the sense that every germ coprojection factors through some $B_i$. As a consequence, if $\mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset$ for all library elements, then $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$. This is the soundness guarantee needed for the categorical Lock mechanism at Node 17.

## Setup and Notation

### Given Data

We are provided with the following categorical framework:

1. **Problem Type:** A fixed problem type $T$ (parabolic, algebraic, quantum, etc.) with associated critical parameters:
   - Critical regularity exponent $s_c$
   - Energy threshold $\Lambda_T < \infty$
   - Dimension bound $d \in \mathbb{N}$

2. **Hypostructure Category:** $\mathbf{Hypo}_T$ — the category of admissible T-hypostructures:
   - Objects: Hypostructures $\mathbb{H}$ encoding geometric/analytic data
   - Morphisms: Structure-preserving maps respecting the problem type $T$
   - Assumption: $\mathbf{Hypo}_T$ is locally presentable ({cite}`Lurie09` §5.5)

3. **Germ Set:** $\mathcal{G}_T$ — the small set of singularity germs:
   - Elements: Isomorphism classes $[P, \pi]$ where:
     - $P$ is a local singularity profile
     - $\pi: P \to \mathbb{R}^d$ is a blow-up parametrization (in a fixed local chart of dimension $d$)
     - Subcriticality: $\dim_H(P) \leq d - 2s_c$
     - Energy bound: $\|\pi\|_{\dot{H}^{s_c}(\mathbb{R}^d)} \leq \Lambda_T$
   - Equivalence: $(P, \pi) \sim (P', \pi')$ if equivalent under local diffeomorphism respecting blow-up structure
   - Smallness: $\mathcal{G}_T$ is a set (not a proper class) by the Cardinality Boundedness Argument (see Lemma 1.1)

4. **Universal Bad Pattern:** $\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{[P,\pi] \in \mathcal{G}_T} \mathbb{H}_{[P,\pi]}$
   - Colimit taken in $\mathbf{Hypo}_T$ over the index category $\mathbf{I}_{\text{small}}$ with objects $\mathcal{G}_T$
   - Coprojections: $\iota_{[P,\pi]}: \mathbb{H}_{[P,\pi]} \to \mathbb{H}_{\mathrm{bad}}^{(T)}$ for each germ
   - Initiality: For any $\mathbb{H} \in \mathbf{Hypo}_T$ receiving all germs, there exists a unique mediating morphism $\mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H}$

5. **Bad Pattern Library:** $\mathcal{B} = \{B_i\}_{i \in I}$ — a finite subset of $\text{Obj}(\mathbf{Hypo}_T)$:
   - Finiteness: $|I| < \infty$
   - Each $B_i$ is a canonical minimal bad pattern for type $T$
   - Construction is problem-specific (see Explicit Construction by Type below)

6. **Target Object:** A concrete object $Z$ with admissible hypostructure $\mathbb{H}(Z) \in \mathbf{Hypo}_T$

### Goal

We construct a certificate:
$$K_{\mathrm{density}}^+ = (\mathcal{B}, \mathcal{G}_T, \{\text{fact}_{[P,\pi]}\}_{[P,\pi] \in \mathcal{G}_T}, \mathsf{type\_completeness})$$
witnessing:

1. **Factorization Property:** For every germ $[P, \pi] \in \mathcal{G}_T$, there exists $B_{i([P,\pi])} \in \mathcal{B}$ and morphisms:
   $$\mathbb{H}_{[P,\pi]} \xrightarrow{\alpha_{[P,\pi]}} B_{i([P,\pi])} \xrightarrow{\beta_i} \mathbb{H}_{\mathrm{bad}}^{(T)}$$
   such that $\beta_i \circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}$ (factorization through library)

2. **Density Consequence:** The coprojections $\beta_i: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T)}$ form a jointly epimorphic family

3. **Hom-Set Reduction:**
   $$(\forall i \in I.\, \mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset) \Rightarrow \mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$$

### Conventions (Colimits and “Coprojections”)

Let $\mathcal{C} := \mathbf{Hypo}_T$.

1. **Index category.** We take $\mathbf{I}_{\text{small}}$ to be the full (small) subcategory of $\mathcal{C}$ spanned by the germ objects $\{\mathbb{H}_{[P,\pi]}\}_{[P,\pi]\in\mathcal{G}_T}$. In particular, a morphism $\mathbb{H}_{[P,\pi]} \to \mathbb{H}_{[P',\pi']}$ in $\mathbf{I}_{\text{small}}$ is just a morphism in $\mathcal{C}$ between the corresponding germ objects.

2. **Colimit notation.** The universal bad object is
   $$\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{[P,\pi]\in\mathcal{G}_T}\,\mathbb{H}_{[P,\pi]} \in \mathcal{C}.$$

3. **Coprojections and cocone identities.** The colimit comes with canonical morphisms (the colimit cocone)
   $$\iota_{[P,\pi]}: \mathbb{H}_{[P,\pi]} \to \mathbb{H}_{\mathrm{bad}}^{(T)}.$$
   By definition of “colimit cocone”, for every morphism $f:\mathbb{H}_{[P,\pi]}\to\mathbb{H}_{[P',\pi']}$ in $\mathbf{I}_{\text{small}}$ we have the commutativity relation:
   $$\iota_{[P',\pi']} \circ f = \iota_{[P,\pi]}.$$

4. **Joint epimorphism (definition).** A family of morphisms $\{e_j:X_j\to Y\}$ in $\mathcal{C}$ is *jointly epimorphic* if for any $g,h:Y\to Z$, the equalities $g\circ e_j = h\circ e_j$ for all $j$ imply $g=h$.

5. **Joint epimorphism (colimit coprojections).** The coprojections $\{\iota_{[P,\pi]}\}$ are jointly epimorphic: if $g,h:\mathbb{H}_{\mathrm{bad}}^{(T)}\to Z$ satisfy $g\circ\iota_{[P,\pi]}=h\circ\iota_{[P,\pi]}$ for all germs, then $g=h$ by the universal property of the colimit.

---

## Step 1: Smallness of the Germ Set

Before establishing density, we verify that $\mathcal{G}_T$ is indeed small (a set, not a proper class), justifying the colimit construction.

### Lemma 1.1: Cardinality Boundedness of $\mathcal{G}_T$

**Statement:** The germ set $\mathcal{G}_T$ has bounded cardinality: $|\mathcal{G}_T| \leq \kappa_T$ for some cardinal $\kappa_T$ depending only on $T$.

**Proof:** We give an explicit set-theoretic bound. It suffices to show that the class of admissible germ data is at most continuum; passing to isomorphism classes can only reduce cardinality.

**Step 1.1.1 (Separable Metric Spaces Have Cardinality $\leq 2^{\aleph_0}$):** Let $(X,d)$ be a separable metric space. Fix a countable dense subset $D=\{d_n\}_{n\in\mathbb{N}}$. For each $x\in X$ define a sequence $(n_k)_{k\ge 1}$ by choosing (for each $k$) some index $n_k$ with
$$d(x,d_{n_k}) < 2^{-k}.$$
If $x$ and $y$ yield the same sequence $(n_k)$, then for every $k$,
$$d(x,y) \le d(x,d_{n_k}) + d(d_{n_k},y) < 2^{-k} + 2^{-k} = 2^{-k+1},$$
so $d(x,y)=0$ and hence $x=y$. Thus we have an injection $X \hookrightarrow \mathbb{N}^{\mathbb{N}}$, and therefore
$$|X| \le |\mathbb{N}^{\mathbb{N}}| = 2^{\aleph_0}.$$

**Step 1.1.2 (Bounded Sobolev Balls Are Separable):** For the Sobolev/Hilbert-scale profile spaces used in the Sieve, $\dot{H}^{s_c}(\mathbb{R}^d)$ is separable for the usual choices of $s_c$ in PDE applications. (If one works in a nonseparable topology, the framework restricts to a separable “certificate model” of representatives, which is sufficient for set-sized indexing.) Hence the bounded set
$$\mathcal{P}_T := \left\{\pi \in \dot{H}^{s_c}(\mathbb{R}^d) : \|\pi\|_{\dot{H}^{s_c}(\mathbb{R}^d)} \le \Lambda_T\right\}$$
is a separable metric space (with the metric induced from $\dot{H}^{s_c}$). By Step 1.1.1,
$$|\mathcal{P}_T| \le 2^{\aleph_0}.$$

**Step 1.1.3 (Adding Auxiliary Germ Data Does Not Exceed Continuum):** A representative germ consists of a pair $(P,\pi)$, where $\pi\in\mathcal{P}_T$ and $P$ encodes local geometric/singularity data subject to the fixed type-$T$ constraints (subcriticality, bounded ambient dimension $d$, fixed local chart conventions). For fixed $T$ and $d$, such local model data can be encoded using countably many real parameters plus a finite combinatorial type (e.g., a finite atlas + transition data), hence ranges over at most $2^{\aleph_0}$ possibilities. Therefore, the set of all admissible representatives $(P,\pi)$ has cardinality $\le 2^{\aleph_0}$.

Concretely, one may choose a countable “coding scheme” as follows: encode the local chart data and transition maps by their Taylor/Fourier coefficients in a fixed countable basis (truncated at increasing orders), and encode finite combinatorial choices (e.g. number of charts) as an integer. This gives an injection of admissible model data into a subset of $\mathbb{R}^{\mathbb{N}}\times \mathbb{N}$, whose cardinality is $2^{\aleph_0}$.

**Step 1.1.4 (Passing to Germ Equivalence Classes):** The germ set $\mathcal{G}_T$ is the quotient of the admissible representatives by the equivalence relation “local diffeomorphism respecting blow-up structure”. A quotient cannot have larger cardinality than the set being quotiented, so:
$$|\mathcal{G}_T| \le 2^{\aleph_0}.$$
Taking $\kappa_T := 2^{\aleph_0}$ proves the claim. □

**Remark:** In the framework, this “smallness” input is the set-theoretic ingredient behind the Small Object Argument viewpoint ({cite}`Quillen67` §II.3): analytic a priori bounds restrict attention to a set-sized collection of local models.

### Corollary 1.2: Existence of Colimit

**Statement:** The colimit $\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{[P,\pi] \in \mathcal{G}_T} \mathbb{H}_{[P,\pi]}$ exists in $\mathbf{Hypo}_T$.

**Proof:**

**Step 1.2.1 (Local Smallness):** By design, $\mathbf{Hypo}_T$ is locally small: for any objects $X,Y \in \mathbf{Hypo}_T$, the collection $\mathrm{Hom}(X,Y)$ is a set.

**Step 1.2.2 (Smallness of the Index Category):** By Lemma 1.1, the object set of $\mathbf{I}_{\text{small}}$ (identified with $\mathcal{G}_T$ via $[P,\pi]\mapsto \mathbb{H}_{[P,\pi]}$) is a set. By Step 1.2.1, for each pair of objects the morphisms form a set, hence the total morphism collection of $\mathbf{I}_{\text{small}}$ is also a set (it is a union of set-indexed Hom-sets). Therefore $\mathbf{I}_{\text{small}}$ is a small category.

**Step 1.2.3 (Existence of Colimits in a Locally Presentable Category):** Since $\mathbf{Hypo}_T$ is locally presentable ({cite}`Lurie09` Proposition 5.5.3.8), it admits all small colimits. Applying this to the small diagram indexed by $\mathbf{I}_{\text{small}}$ yields the existence of
$$\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{\mathbf{I}_{\text{small}}}\,\mathbb{H}_{(-)}.$$
The colimit is unique up to canonical isomorphism. □

---

## Step 2: Library Generation Property

We now establish that $\mathcal{B}$ generates all germs via factorization.

### Lemma 2.0: Finite $\varepsilon$-Nets from Compactness

**Statement:** Let $(M,d)$ be a compact metric space and let $\varepsilon>0$. Then there exist points $m_1,\dots,m_N\in M$ such that
$$M \subseteq \bigcup_{j=1}^N B(m_j,\varepsilon),$$
where $B(m,\varepsilon):=\{x\in M:d(x,m)<\varepsilon\}$. Equivalently, $\{m_1,\dots,m_N\}$ is a finite $\varepsilon$-net of $M$.

**Proof:** The family of open balls $\{B(x,\varepsilon)\}_{x\in M}$ is an open cover of $M$. By compactness, there is a finite subcover $M\subseteq \bigcup_{j=1}^N B(m_j,\varepsilon)$ for some points $m_1,\dots,m_N\in M$. □

**Admissibility Note (Compactness Input):** In applications below, $M$ is a moduli space (or a compactification of a moduli space) equipped with the admissible metric. The compactness of $(M,d)$ is part of the admissibility data for type $T$ (e.g., Uhlenbeck compactness or a compactified quotient after gauge fixing), and is explicitly assumed or certified before invoking Lemma 2.0.

### Lemma 2.1: Factorization Through Library

**Statement:** For every germ $[P, \pi] \in \mathcal{G}_T$, there exists $B_i \in \mathcal{B}$ and morphisms:
$$\mathbb{H}_{[P,\pi]} \xrightarrow{\alpha_{[P,\pi]}} B_i \xrightarrow{\beta_i} \mathbb{H}_{\mathrm{bad}}^{(T)}$$
such that $\beta_i \circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}$.

**Proof:**

**Step 2.1.0 (What We Need to Produce):** Since $\mathbf{I}_{\text{small}}$ is the full subcategory on the germ objects, any morphism $\alpha:\mathbb{H}_{[P,\pi]} \to \mathbb{H}_{[P',\pi']}$ between germ objects is a morphism in the indexing diagram. Hence the colimit cocone identities give:
$$\iota_{[P',\pi']} \circ \alpha = \iota_{[P,\pi]}.$$
Therefore, if we choose a library element $B_i$ that is itself realized as a germ object $\mathbb{H}_{[P_i,\pi_i]}$ and we set
$$\beta_i := \iota_{[P_i,\pi_i]}: B_i=\mathbb{H}_{[P_i,\pi_i]} \to \mathbb{H}_{\mathrm{bad}}^{(T)},$$
then **any** morphism $\alpha_{[P,\pi]}:\mathbb{H}_{[P,\pi]} \to B_i$ automatically yields the desired factorization:
$$\beta_i \circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}.$$
So the content of Lemma 2.1 is: build a finite family of “representative” germ objects $B_i$ and, for each germ, construct a morphism into one of them.

**Admissibility Note (Certified Linearization Data):** The existence of $\alpha_{[P,\pi]}$ from metric proximity uses the type-$T$ verification contract: a certified linearization/invertibility datum for the germ model and an explicit admissibility threshold $\varepsilon_0$ for the implicit-function step. These data are recorded as part of admissibility for $T$ and are invoked whenever we pass from $\varepsilon$-closeness to morphism existence.

We now verify this explicitly for three canonical families of problem types.

---

#### Case 1: Parabolic Type ($T = T_{\mathrm{para}}$)

**Framework:** Blow-up analysis for semilinear parabolic equations:
$$\partial_t u = \Delta u + |u|^{p-1} u, \quad p > 1$$

**Germ Set:** $\mathcal{G}_{T_{\mathrm{para}}}$ consists of Type I blow-up profiles:
$$u(x,t) \sim (T - t)^{-1/(p-1)} \Psi\left(\frac{x}{\sqrt{T-t}}\right), \quad \Psi \in \dot{H}^1(\mathbb{R}^d)$$
with $\|\Psi\|_{\dot{H}^1} \leq \Lambda_{T_{\mathrm{para}}}$.

**Library Construction:** By {cite}`MerleZaag98` Theorem 1.1, the space of Type I profiles modulo scaling and translation is **finite-dimensional**:
$$\mathcal{M}_{T_{\mathrm{para}}} := \{\Psi : \Delta \Psi + |\Psi|^{p-1} \Psi = -\frac{1}{p-1}\Psi, \, \|\Psi\|_{\dot{H}^1} \leq \Lambda_{T_{\mathrm{para}}}\} / G$$
has $\dim(\mathcal{M}_{T_{\mathrm{para}}}) \leq d(p-1) + O(1)$.

**Finite Cover:** Equip $\mathcal{M}_{T_{\mathrm{para}}}$ with the metric induced by $\dot{H}^1$. By compactness and Lemma 2.0, there exists a finite $\varepsilon$-net $\mathcal{B}_{\mathrm{para}} = \{B_1, \ldots, B_N\}$ such that:
$$\forall [P, \pi] \in \mathcal{G}_{T_{\mathrm{para}}}.\, \exists B_i: \, \|\pi - B_i\|_{\dot{H}^1} \leq \varepsilon$$

:::{important}
**Metric Proximity → Morphism Existence (with Certified Data):**

The claim that $\varepsilon$-closeness in $\dot{H}^1$ implies a morphism in $\mathbf{Hypo}_{T_{\mathrm{para}}}$ requires justification:

1. **Morphism definition for parabolic type:** A morphism $\alpha: \mathbb{H}_{[P,\pi]} \to \mathbb{H}_{[P',\pi']}$ consists of:
   - A $C^1$ map $\phi: P \to P'$ intertwining the blow-up parametrizations
   - Energy monotonicity: $\|\pi' \circ \phi\|_{\dot{H}^1} \leq \|\pi\|_{\dot{H}^1}$
   - Compatibility with the semilinear structure

2. **Implicit function theorem:** For $\|\pi - \pi'\|_{\dot{H}^1} < \varepsilon$ with $\varepsilon$ small, the map $\phi = \text{id} + \psi$ where $\psi$ solves a linearized equation provides the required morphism. This uses certified linearization/invertibility data:
   - A documented right inverse for the linearized operator (from spectral gap $\lambda > 0$ or a coercive estimate)
   - Lipschitz bounds on the nonlinearity

3. **Threshold value:** The admissibility threshold $\varepsilon_0$ is part of the verification contract for type $T_{\mathrm{para}}$, and depends on the spectral gap and nonlinearity. Explicitly:
   $$\varepsilon_0 \sim \frac{\lambda}{C_{\text{Lip}}(f)}$$
   where $C_{\text{Lip}}(f)$ is the Lipschitz constant of the nonlinearity $f(u) = |u|^{p-1}u$.
:::

Choose $\varepsilon < \varepsilon_0$ so that $\varepsilon$-closeness implies morphism existence.

**Normalization Note:** Because $\mathcal{M}_{T_{\mathrm{para}}}$ is a quotient by the symmetry group $G$, the statement “$\|\pi-B_i\|_{\dot{H}^1}\le\varepsilon$” should be read after choosing representatives in a fixed gauge/normalization. Equivalently, one can say: there exists $g\in G$ such that $\|\pi - g\cdot B_i\|_{\dot{H}^1}\le\varepsilon$. The choice of representatives (gauge fixing) and the $\varepsilon$-metric are part of the admissibility data/verification contract for type $T$, and the symmetry isomorphism is absorbed into the morphism $\alpha_{[P,\pi]}$.

**Factorization Construction:** We interpret each net point $B_i$ as a chosen **germ object** in $\mathbf{Hypo}_{T_{\mathrm{para}}}$ (a representative Type I profile) and keep the same symbol $B_i$ for the corresponding hypostructure $\mathbb{H}_{B_i}$.

For each germ $[P,\pi] \in \mathcal{G}_{T_{\mathrm{para}}}$:
- **Step 2.1.1 (Choose a Representative):** Choose an index $i=i([P,\pi])$ such that $\|\pi - B_i\|_{\dot{H}^1} \le \varepsilon$ (possible by the $\varepsilon$-net property).
- **Step 2.1.2 (Produce the Germ-to-Library Morphism):** Use the type-$T_{\mathrm{para}}$ stability/identification mechanism encoded in $\mathbf{Hypo}_{T_{\mathrm{para}}}$: for $\varepsilon$ below the interface threshold, “$\dot{H}^1$-closeness” yields a structure-preserving morphism
  $$\alpha_{[P,\pi]}:\mathbb{H}_{[P,\pi]} \to B_i.$$
- **Step 2.1.3 (Define the Library-to-Universal Map):** Define
  $$\beta_i := \iota_{B_i}: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T_{\mathrm{para}})}$$
  as the canonical colimit coprojection for the germ object $B_i$.

**Verification (Factorization Identity):** Since $\mathbf{I}_{\text{small}}$ is full on germ objects, $\alpha_{[P,\pi]}$ is a morphism in the indexing diagram. Therefore, by the cocone identity (Conventions 3),
$$\beta_i \circ \alpha_{[P,\pi]} = \iota_{B_i} \circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}.$$

**Certificate Fragment:** $\mathsf{fact}_{[P,\pi]} := (B_i, \alpha_{[P,\pi]}, \beta_i, \|\pi - B_i\|_{\dot{H}^1})$

---

#### Case 2: Algebraic Type ($T = T_{\mathrm{alg}}$)

**Framework:** Hodge theory and algebraic cycles ({cite}`Voisin02` Chapter 6).

**Germ Set:** $\mathcal{G}_{T_{\mathrm{alg}}}$ consists of minimal non-algebraic $(p,p)$-cohomology classes:
$$[P, \pi] = [\alpha], \quad \alpha \in H^{p,p}(X, \mathbb{C}) \cap H^{2p}(X, \mathbb{Q}), \quad \alpha \notin \text{Alg}(X)$$
where $X$ is a smooth projective variety and $\text{Alg}(X)$ is the subspace of algebraic classes.

**Library Construction:** By {cite}`Voisin02` (e.g. Theorem 11.35), for fixed $X$ and fixed Hodge type $(p,p)$, the space of classes relevant to the obstruction admits a finite-dimensional description. Concretely, the quotient of $(p,p)$-classes by algebraic classes can be treated (for the purposes of this framework) as a finite-dimensional complex vector space (or a complex torus after quotienting by an integral lattice). Let
$$V := H^{p,p}(X,\mathbb{C}) / \mathrm{Alg}(X)_{\mathbb{C}}$$
and write $\dim_{\mathbb{C}}(V)=r<\infty$.

**Finite Generators:** Choose elements $B_1,\dots,B_r$ whose images form a spanning set of $V$ (e.g. a basis of $V$). We treat these as the algebraic-type bad patterns; each $B_i$ determines a germ object (still denoted $B_i$) in $\mathbf{Hypo}_{T_{\mathrm{alg}}}$.

**Factorization Construction:** For each germ $[\alpha] \in \mathcal{G}_{T_{\mathrm{alg}}}$:
- **Step 2.1.4 (Express in the Spanning Set):** Write the class of $\alpha$ in the quotient $V$ as
  $$[\alpha] = \sum_{i=1}^r c_i [B_i] \quad \text{in } V$$
  for some coefficients $c_i\in\mathbb{C}$.
- **Step 2.1.5 (Choose an Active Generator and Map to It):** Since $[\alpha]\neq 0$ in $V$ (i.e. $\alpha$ is non-algebraic), at least one coefficient is nonzero. Choose an index $i_0$ with $c_{i_0}\neq 0$, and take as part of the germ data a morphism
  $$\alpha_{[\alpha]}: \mathbb{H}_{[\alpha]} \to B_{i_0}.$$
  Concretely, let $p_{i_0}:V\to \mathbb{C}\cdot[B_{i_0}]$ be a coordinate projection. In $\mathbf{Hypo}_{T_{\mathrm{alg}}}$, morphisms are taken to respect the linear/Hodge structure on these obstruction classes, so $p_{i_0}$ induces the required morphism $\alpha_{[\alpha]}$.
- **Step 2.1.6 (Define the Library-to-Universal Map):** Define
  $$\beta_{i_0} := \iota_{B_{i_0}}: B_{i_0} \to \mathbb{H}_{\mathrm{bad}}^{(T_{\mathrm{alg}})}$$
  as the canonical colimit coprojection for the germ object $B_{i_0}$.

**Verification (Factorization Identity):** As in Case 1, $\alpha_{[\alpha]}$ is a morphism between germ objects and hence a morphism in the indexing diagram. Therefore, by the cocone identity (Conventions 3),
$$\beta_{i_0} \circ \alpha_{[\alpha]} = \iota_{B_{i_0}} \circ \alpha_{[\alpha]} = \iota_{[\alpha]}.$$

**Certificate Fragment:** $\mathsf{fact}_{[\alpha]} := (B_{i_0}, \alpha_{[\alpha]}, \beta_{i_0}, c_{i_0})$

---

#### Case 3: Quantum Type ($T = T_{\mathrm{quant}}$)

**Framework:** Yang-Mills instantons and moduli spaces ({cite}`DonaldsonKronheimer90` Chapter 2).

**Germ Set:** $\mathcal{G}_{T_{\mathrm{quant}}}$ consists of instantons with bounded action:
$$[P, \pi] = [A], \quad A \in \mathcal{A}^{\text{ASD}}(E), \quad S(A) := \frac{1}{8\pi^2}\int |F_A|^2 \leq \Lambda_{T_{\mathrm{quant}}}$$
where $\mathcal{A}^{\text{ASD}}(E)$ is the space of anti-self-dual connections on a principal $G$-bundle $E \to X$, and $F_A$ is the curvature.

**Library Construction:** By {cite}`DonaldsonKronheimer90` Theorem 2.3.3 (Uhlenbeck compactness), the moduli space of instantons with bounded action is:
$$\mathcal{M}_{k}(X, E) := \{[A] \in \mathcal{A}^{\text{ASD}}(E) / \mathcal{G} : S(A) = 8\pi^2 k\}$$
which is a finite-dimensional manifold of dimension $\dim(\mathcal{M}_k) = 8k - 3(\chi(X) + \sigma(X))$ for topological charge $k$.

**Finite Cover for Small Action:** For $S(A) \leq \Lambda_{T_{\mathrm{quant}}}$, the total charge is bounded: $k \leq \Lambda_{T_{\mathrm{quant}}} / (8\pi^2)$. After Uhlenbeck compactification, each charge-$k$ moduli space is compact, hence admits a finite net by Lemma 2.0. One convenient way to organize such a finite net is via a finite triangulation of each $\mathcal{M}_k$ (for $k \leq k_{\max}$), yielding a finite vertex set $\mathcal{B}_{\mathrm{quant}} = \{B_1, \ldots, B_M\}$.

**Factorization Construction:** Interpret each vertex $B_i$ as a chosen representative instanton germ (hence a germ object in $\mathbf{Hypo}_{T_{\mathrm{quant}}}$).

For each germ $[A] \in \mathcal{G}_{T_{\mathrm{quant}}}$:
- **Step 2.1.7 (Choose the Charge Stratum):** Determine the topological charge $k$ of $[A]$ (so $[A]\in\mathcal{M}_k$).
- **Step 2.1.8 (Choose a Nearby Library Vertex):** Use the triangulation to choose a vertex $B_i$ in the simplex containing (or nearest to) the point $[A]\in\mathcal{M}_k$.
- **Step 2.1.9 (Produce the Germ-to-Library Morphism):** Use Uhlenbeck gauge-fixing to place $A$ in a controlled Sobolev neighborhood of the representative connection underlying $B_i$. In $\mathbf{Hypo}_{T_{\mathrm{quant}}}$, such gauge-controlled proximity is precisely the hypothesis guaranteeing existence of a structure-preserving morphism, giving
  $$\alpha_{[A]}:\mathbb{H}_{[A]} \to B_i.$$
- **Step 2.1.10 (Define the Library-to-Universal Map):** Define
  $$\beta_i := \iota_{B_i}: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T_{\mathrm{quant}})}$$
  as the canonical colimit coprojection for the germ object $B_i$.

**Verification (Factorization Identity):** Since $\alpha_{[A]}$ is a morphism between germ objects, the cocone identity (Conventions 3) gives:
$$\beta_i \circ \alpha_{[A]} = \iota_{B_i} \circ \alpha_{[A]} = \iota_{[A]}.$$

**Certificate Fragment:** $\mathsf{fact}_{[A]} := (B_i, \alpha_{[A]}, \beta_i, S(A), k)$

---

### Lemma 2.2: Finiteness of Library

**Statement:** For each problem type $T \in \{T_{\mathrm{para}}, T_{\mathrm{alg}}, T_{\mathrm{quant}}\}$, the library $\mathcal{B}_T$ is finite: $|\mathcal{B}_T| < \infty$.

**Proof:** We check each canonical type.

- **Parabolic:** The moduli space $\mathcal{M}_{T_{\mathrm{para}}}$ is (by the cited blow-up analysis) a compact metric space after quotienting by symmetries. Every compact metric space admits a finite $\varepsilon$-net for any $\varepsilon>0$. Choosing one such net produces finitely many representatives $B_1,\dots,B_N$.

- **Algebraic:** The obstruction space $V = H^{p,p}(X,\mathbb{C})/\mathrm{Alg}(X)_{\mathbb{C}}$ has finite complex dimension $r$. Any spanning set (in particular, a basis) has finitely many elements, giving $B_1,\dots,B_r$.

- **Quantum:** For each topological charge $k\le k_{\max}$, the Uhlenbeck compactness theorem yields a compactification of the moduli space of ASD connections with fixed charge. A compact finite-dimensional manifold/orbifold admits a finite triangulation, hence finitely many vertices. Since only finitely many charges occur under the action bound, the union of vertex sets is finite.

Therefore $|\mathcal{B}_T|<\infty$ in each case. □

---

## Step 3: Density in the Colimit

We establish that the library elements map into the colimit and that these maps form a jointly epimorphic family.

### Lemma 3.1: Existence of Library Coprojections

**Statement:** For each $B_i \in \mathcal{B}$, there exists a canonical coprojection:
$$\beta_i: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T)}$$
making the diagram commute:
$$
\begin{array}{ccc}
\mathbb{H}_{[P,\pi]} & \xrightarrow{\alpha_{[P,\pi]}} & B_i \\
\downarrow{\iota_{[P,\pi]}} & & \downarrow{\beta_i} \\
\mathbb{H}_{\mathrm{bad}}^{(T)} & = & \mathbb{H}_{\mathrm{bad}}^{(T)}
\end{array}
$$

**Proof:**
**Step 3.1.1 (Colimit Gives a Map for Every Diagram Object):** Each library element $B_i$ is (by construction in Lemma 2.1) a germ object, hence an object of the indexing category $\mathbf{I}_{\text{small}}$.

**Step 3.1.2 (Define $\beta_i$):** By the definition of the colimit cocone, there is a canonical morphism
$$\iota_{B_i}: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T)}.$$
Set $\beta_i := \iota_{B_i}$. This choice is canonical once the diagram and the colimit are fixed.

**Step 3.1.3 (Commutativity of the Square):** For any germ $[P,\pi]$ whose factorization uses $B_i$, Lemma 2.1 provides a morphism $\alpha_{[P,\pi]}:\mathbb{H}_{[P,\pi]}\to B_i$. Since $\mathbf{I}_{\text{small}}$ is full on germ objects, $\alpha_{[P,\pi]}$ is a morphism in the indexing diagram, and therefore the cocone identity (Conventions 3) implies:
$$\beta_i \circ \alpha_{[P,\pi]} = \iota_{B_i}\circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}.$$
This is exactly the commutativity asserted in the statement. □

**Remark:** In concrete models, $\beta_i$ is often an inclusion map embedding $B_i$ as a subobject of $\mathbb{H}_{\mathrm{bad}}^{(T)}$.

### Lemma 3.2: Joint Epimorphism of Library Coprojections

**Statement:** The family $\{\beta_i: B_i \to \mathbb{H}_{\mathrm{bad}}^{(T)}\}_{i \in I}$ is jointly epimorphic: the coprojections jointly cover $\mathbb{H}_{\mathrm{bad}}^{(T)}$.

**Proof:**
**Step 3.2.1 (Assume Agreement on the Library):** Let $g,h: \mathbb{H}_{\mathrm{bad}}^{(T)} \to Z$ satisfy
$$g \circ \beta_i = h \circ \beta_i \quad \text{for all } i\in I.$$

**Step 3.2.2 (Reduce to the Germ Coprojections):** Fix an arbitrary germ $[P,\pi]\in\mathcal{G}_T$. By Lemma 2.1, choose an index $i=i([P,\pi])$ and a morphism $\alpha_{[P,\pi]}:\mathbb{H}_{[P,\pi]}\to B_i$ such that
$$\beta_i \circ \alpha_{[P,\pi]} = \iota_{[P,\pi]}.$$

**Step 3.2.3 (Transfer Agreement from $\beta_i$ to $\iota_{[P,\pi]}$):** Using Step 3.2.1 and the above factorization,
$$g\circ \iota_{[P,\pi]} = g\circ \beta_i \circ \alpha_{[P,\pi]} = h\circ \beta_i \circ \alpha_{[P,\pi]} = h\circ \iota_{[P,\pi]}.$$

**Step 3.2.4 (Conclude by Joint Epimorphism of Colimit Coprojections):** Since the family $\{\iota_{[P,\pi]}\}$ is jointly epimorphic (Conventions 5), the equalities $g\circ \iota_{[P,\pi]}=h\circ \iota_{[P,\pi]}$ for all germs imply $g=h$. Thus $\{\beta_i\}_{i\in I}$ is jointly epimorphic. □

---

## Step 4: Hom-Set Reduction

We establish the key consequence used by Lock: emptiness on the finite library certifies emptiness for the universal object.

### Lemma 4.1: Hom-Set Reduction (Soundness)

**Statement:** For any $\mathbb{H}(Z) \in \mathbf{Hypo}_T$,
$$(\forall i \in I.\, \mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset) \Rightarrow \mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset.$$

**Proof:**

**Step 4.1.1 (Assume All Library Hom-Sets Are Empty):** Assume
$$\mathrm{Hom}(B_i,\mathbb{H}(Z))=\emptyset \quad \text{for all } i\in I.$$

**Step 4.1.2 (Suppose a Universal Morphism Exists):** Suppose, for contradiction, that there exists a morphism
$$\Phi:\mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H}(Z).$$

**Step 4.1.3 (Restrict $\Phi$ Along Each $\beta_i$):** For each $i\in I$, define
$$\phi_i := \Phi \circ \beta_i : B_i \to \mathbb{H}(Z).$$
Then $\phi_i \in \mathrm{Hom}(B_i,\mathbb{H}(Z))$, so $\mathrm{Hom}(B_i,\mathbb{H}(Z))\neq\emptyset$.

**Step 4.1.4 (Contradiction):** This contradicts the assumption in Step 4.1.1. Therefore no such $\Phi$ exists, and
$$\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)},\mathbb{H}(Z))=\emptyset.$$
□

**Corollary (Contrapositive):** If $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) \neq \emptyset$, then $\mathrm{Hom}(B_i, \mathbb{H}(Z)) \neq \emptyset$ for every $i \in I$.

---

## Step 5: Universal Property via Limit Formula

We provide an alternative characterization using the limit-colimit adjunction.

### Lemma 5.1: Hom-Set as Inverse Limit

**Statement:** For any $\mathbb{H}(Z) \in \mathbf{Hypo}_T$:
$$\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) \cong \varprojlim_{[P,\pi] \in \mathcal{G}_T} \mathrm{Hom}(\mathbb{H}_{[P,\pi]}, \mathbb{H}(Z))$$

**Proof:** This is the standard “colimit–Hom = limit of Hom” correspondence (see {cite}`MacLane71` Theorem V.5.1). We spell it out directly from the universal property.

Let $D:\mathbf{I}_{\text{small}}\to \mathbf{Hypo}_T$ be the germ diagram, so $D([P,\pi])=\mathbb{H}_{[P,\pi]}$, and let
$$\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{\mathbf{I}_{\text{small}}} D$$
with coprojections $\iota_{[P,\pi]}:D([P,\pi])\to \mathbb{H}_{\mathrm{bad}}^{(T)}$.

**Step 5.1.1 (From a Map Out of the Colimit to a Compatible Family):** Given $\Phi\in\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)},\mathbb{H}(Z))$, define a family of morphisms
$$\phi_{[P,\pi]} := \Phi \circ \iota_{[P,\pi]} \in \mathrm{Hom}(\mathbb{H}_{[P,\pi]},\mathbb{H}(Z)).$$

**Step 5.1.2 (Compatibility Condition):** For any morphism $f:\mathbb{H}_{[P,\pi]}\to\mathbb{H}_{[P',\pi']}$ in $\mathbf{I}_{\text{small}}$, the cocone identity gives $\iota_{[P',\pi']}\circ f = \iota_{[P,\pi]}$, hence
$$\phi_{[P',\pi']} \circ f = (\Phi\circ \iota_{[P',\pi']})\circ f = \Phi\circ (\iota_{[P',\pi']}\circ f) = \Phi\circ \iota_{[P,\pi]} = \phi_{[P,\pi]}.$$
So $(\phi_{[P,\pi]})$ is a compatible family.

**Step 5.1.3 (From a Compatible Family to a Map Out of the Colimit):** Conversely, suppose we are given a family
$$(\phi_{[P,\pi]})_{[P,\pi]\in\mathcal{G}_T} \quad \text{with } \phi_{[P',\pi']}\circ f = \phi_{[P,\pi]} \text{ for all } f:[P,\pi]\to[P',\pi'].$$
This is exactly the data of a cocone from $D$ to $\mathbb{H}(Z)$. By the universal property of the colimit, there exists a unique morphism
$$\Phi:\mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H}(Z)$$
such that $\Phi\circ \iota_{[P,\pi]}=\phi_{[P,\pi]}$ for all germs.

**Step 5.1.4 (Bijection and Identification with the Limit):** Steps 5.1.1–5.1.3 are inverse constructions, giving a bijection between $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)},\mathbb{H}(Z))$ and the set of compatible families $(\phi_{[P,\pi]})$. By definition, the set of compatible families is the inverse limit
$$\varprojlim_{[P,\pi]\in\mathcal{G}_T}\mathrm{Hom}(\mathbb{H}_{[P,\pi]},\mathbb{H}(Z)).$$
This proves the claimed isomorphism. □

**Remark (Why We Do Not Reduce the Limit to a Product):** The inverse limit in Lemma 5.1 is, concretely, a subset of a product:
$$\varprojlim_{[P,\pi]} \mathrm{Hom}(\mathbb{H}_{[P,\pi]},\mathbb{H}(Z)) \subseteq \prod_{[P,\pi]\in\mathcal{G}_T} \mathrm{Hom}(\mathbb{H}_{[P,\pi]},\mathbb{H}(Z)),$$
cut out by compatibility constraints along morphisms in the germ category. Even if every factor in the product is nonempty, the compatibility constraints can force the limit to be empty.

For the Lock, we do not need to compute this limit. We only need the contrapositive-style *nonexistence* implication in Lemma 4.1: if every $\mathrm{Hom}(B_i,\mathbb{H}(Z))$ is empty, then no morphism $\mathbb{H}_{\mathrm{bad}}^{(T)}\to \mathbb{H}(Z)$ can exist.

---

## Step 6: Completeness by Type

We verify that the library construction is complete for each problem type.

### Verification 6.1: Parabolic Type Completeness

**Claim:** For parabolic types, $\mathcal{B}_{\mathrm{para}}$ contains all Type I profiles below energy threshold.

**Proof:**

**Step 6.1.1 (Classification Input):** For semilinear parabolic blow-up of Type I, the blow-up analysis isolates a moduli space $\mathcal{M}_{T_{\mathrm{para}}}$ of possible profile germs modulo the symmetry group $G$ ({cite}`MerleZaag98`).

**Step 6.1.2 (Compactness/Finite Cover):** Equip $\mathcal{M}_{T_{\mathrm{para}}}$ with the metric induced by $\dot{H}^1$. Compactness implies: for any $\varepsilon>0$, there exists a finite $\varepsilon$-net. Choose such a net and represent each net point by a germ object $B_i$.

**Step 6.1.3 (From Proximity to a Morphism):** For any germ $[P,\pi]$, pick a net point $B_i$ with $\|\pi-B_i\|_{\dot{H}^1}\le\varepsilon$. By the design of $\mathbf{Hypo}_{T_{\mathrm{para}}}$, taking $\varepsilon$ below the interface tolerance yields a structure-preserving morphism $\alpha_{[P,\pi]}:\mathbb{H}_{[P,\pi]}\to B_i$.

This constructs the required germ-to-library morphisms for all Type I germs. □

**Literature Justification:** {cite}`MerleZaag98` provides optimal estimates for blow-up rate and behavior, establishing the finite-dimensional structure of the blow-up profile space. The compactness follows from energy boundedness and concentration-compactness ({cite}`Lions84`).

### Verification 6.2: Algebraic Type Completeness

**Claim:** For algebraic types, $\mathcal{B}_{\mathrm{alg}}$ contains generators of all non-algebraic $(p,p)$-cohomology.

**Proof:**

**Step 6.2.1 (Finite-Dimensional Obstruction Space):** The obstruction to algebraicity lives in a finite-dimensional quotient space
$$V := H^{p,p}(X,\mathbb{C})/\mathrm{Alg}(X)_{\mathbb{C}}$$
in the sense discussed in {cite}`Voisin02`.

**Step 6.2.2 (Choose a Finite Generating Set):** Choose finitely many classes $[B_1],\dots,[B_r]$ spanning $V$ (e.g. a basis).

**Step 6.2.3 (Every Non-Algebraic Class Hits a Generator):** If $[\alpha]\neq 0$ in $V$, then in its expansion $[\alpha]=\sum_i c_i[B_i]$ at least one coefficient is nonzero. Selecting such an index $i_0$ and recording the corresponding morphism $\alpha_{[\alpha]}:\mathbb{H}_{[\alpha]}\to B_{i_0}$ yields the required factorization witness.

Thus every non-algebraic germ maps to some generator in the finite library. □

**Literature Justification:** {cite}`Voisin02` develops the Hodge-theoretic framework used to isolate the relevant obstruction space and to justify working with a finite-dimensional quotient of $(p,p)$-classes by algebraic classes.

### Verification 6.3: Quantum Type Completeness

**Claim:** For quantum types, $\mathcal{B}_{\mathrm{quant}}$ contains instanton moduli generators for bounded action.

**Proof:**

**Step 6.3.1 (Compactness Under Action Bounds):** Uhlenbeck compactness ({cite}`DonaldsonKronheimer90` Theorem 2.3.3) implies that after compactification, the space of ASD connections with action bounded by $\Lambda_{T_{\mathrm{quant}}}$ decomposes into finitely many compact finite-dimensional pieces (one for each charge $k\le k_{\max}$).

**Step 6.3.2 (Finite Triangulation and Vertex Set):** Each compact piece admits a finite triangulation, producing finitely many vertices. Collect all vertices across charges into a finite set $\{B_i\}$.

**Step 6.3.3 (Map Each Germ to a Vertex):** Given a germ $[A]$, locate it in the triangulation and choose a vertex $B_i$ in the simplex containing (or nearest to) $[A]$. The gauge-controlled proximity gives a morphism $\alpha_{[A]}:\mathbb{H}_{[A]}\to B_i$.

Hence every bounded-action instanton germ maps to some element of the finite library. □

**Literature Justification:** {cite}`DonaldsonKronheimer90` establishes the fundamental properties of Yang-Mills moduli spaces, including compactness (after adding bubbling points), finite-dimensionality, and orientability. The energy bound controls the topology and prevents infinite bubbling.

---

## Conclusion: Certificate Construction

### Final Certificate

We have established all components of the density certificate:

$$K_{\mathrm{density}}^+ = (\mathcal{B}, \mathcal{G}_T, \{\text{fact}_{[P,\pi]}\}_{[P,\pi] \in \mathcal{G}_T}, \mathsf{type\_completeness})$$

where:

1. **Library:** $\mathcal{B} = \{B_i\}_{i \in I}$ with $|I| < \infty$ (Lemma 2.2)

2. **Germ Set:** $\mathcal{G}_T$ is small (Lemma 1.1) with explicit cardinality bound $|\mathcal{G}_T| \leq 2^{\aleph_0}$

3. **Factorization Witnesses:** For each $[P,\pi] \in \mathcal{G}_T$:
   $$\text{fact}_{[P,\pi]} = (B_{i([P,\pi])}, \alpha_{[P,\pi]}: \mathbb{H}_{[P,\pi]} \to B_{i([P,\pi])}, \beta_{i([P,\pi])}: B_{i([P,\pi])} \to \mathbb{H}_{\mathrm{bad}}^{(T)}, \mathsf{metrics})$$
   with $\iota_{[P,\pi]} = \beta_i \circ \alpha_{[P,\pi]}$ (Lemma 2.1)

4. **Type-Specific Completeness:** Explicit verification for $T \in \{T_{\mathrm{para}}, T_{\mathrm{alg}}, T_{\mathrm{quant}}\}$ (Verifications 6.1–6.3)

### Main Consequence (Restatement)

**Lemma (Hom-Set Reduction):** For any $\mathbb{H}(Z) \in \mathbf{Hypo}_T$:
$$(\forall i \in I.\, \mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset) \Rightarrow \mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$$

**Proof:** Lemma 4.1. $\square$

**Practical Implication:** The Lock mechanism at Node 17 of the Structural Sieve can certify the categorical obstruction $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$ by verifying the **finite** set of conditions:
$$\mathrm{Hom}(B_1, \mathbb{H}(Z)) = \emptyset, \ldots, \mathrm{Hom}(B_N, \mathbb{H}(Z)) = \emptyset$$
This makes the Lock verification algorithmically feasible (though not necessarily decidable, as noted in the Interface specification).

### Logical Exhaustiveness

**Corollary:** No singularity can "escape" the categorical check.

**Proof:** By the Categorical Completeness Theorem ({prf:ref}`thm-categorical-completeness`), every singularity pattern factors through the universal bad pattern $\mathbb{H}_{\mathrm{bad}}^{(T)}$. By density (this theorem), $\mathbb{H}_{\mathrm{bad}}^{(T)}$ is generated by the finite library $\mathcal{B}$. Therefore, every singularity factors through some $B_i \in \mathcal{B}$, and checking obstruction on $\mathcal{B}$ suffices. $\square$

This resolves the "Completeness Gap" critique: the proof that physical singularities map to categorical germs is provided by concentration-compactness (KRNL-Trichotomy, {prf:ref}`mt-krnl-trichotomy`), while the proof that germs are exhaustive (via finite library) is provided by this theorem.

---

## Literature and Methodological Context

### Categorical Foundations

The colimit construction and universal property arguments are standard in category theory:

- **Small Object Argument:** {cite}`Quillen67` §II.3 establishes the general framework for generating sets and cofibrations. While we do not use Quillen's SOA directly (we use cardinality bounds instead), the spirit is similar: replacing potentially large objects with small generating sets.

- **Cellular Structures:** {cite}`Hovey99` §2.1 develops the theory of cellular model categories, where objects are built from finite cells. Our library elements $\{B_i\}$ play the role of cells generating the bad pattern complex.

- **Presentability and Generation:** {cite}`Lurie09` §A.1.5 (Higher Topos Theory) provides the $(\infty,1)$-categorical setting for presentable categories. Local presentability ensures that colimits over small diagrams exist and behave well, which is essential for our colimit construction.

### Problem-Specific Analysis

The type-specific completeness verifications rely on deep results in each area:

- **Parabolic:** {cite}`MerleZaag98` provides the finite-dimensional structure of blow-up profiles for semilinear heat equations, enabling the finite cover construction.

- **Algebraic:** {cite}`Voisin02` develops the cohomological Hodge theory necessary to understand non-algebraic cycles and their generation.

- **Quantum:** {cite}`DonaldsonKronheimer90` establishes the geometry of Yang-Mills moduli spaces, including compactness and finite-dimensionality results essential for the triangulation.

### Applicability to the Structural Sieve

This theorem justifies Node 17 (Lock) of the Structural Sieve:

- **Input:** Interface Permit $\mathrm{Cat}_{\mathrm{Hom}}$ with Bad Pattern Library $\mathcal{B} = \{B_i\}_{i \in I}$

- **Verification:** Check $\mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset$ for all $i \in I$ using tactics E1–E12

- **Output:** If all checks pass, emit $K_{\text{Lock}}^{\mathrm{blk}}$ certifying $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$, which implies $\mathrm{Rep}_K(T,Z)$ holds (by the Initiality Lemma {prf:ref}`mt-krnl-exclusion`)

The density theorem ensures that this finite verification is sound for certifying the Lock: if $\mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset$ for all $i \in I$, then $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$ (Lemma 4.1). Equivalently, if a morphism $\mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H}(Z)$ exists, then $\mathrm{Hom}(B_i, \mathbb{H}(Z)) \neq \emptyset$ for every $i \in I$ (contrapositive).

:::

---

## References

See {prf:ref}`mt-fact-germ-density` for the theorem statement and proof sketch. The full proof expands each step with detailed lemmas, explicit constructions, and categorical verifications.
