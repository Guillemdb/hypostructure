# UP-OMinimal: O-Minimal Promotion — GMT Translation

## Original Statement (Hypostructure)

The o-minimal promotion shows that definable sets in o-minimal structures have tame topology, excluding wild (pathological) behavior.

## GMT Setting

**O-Minimal Structure:** Expansion of $(\mathbb{R}, <, +, \cdot)$ with tame definable sets

**Tame:** No pathological sets (Cantor, fractals) are definable

**Promotion:** Local tameness implies global

## GMT Statement

**Theorem (O-Minimal Promotion).** If the energy functional $\Phi$ and singular set $\Sigma$ are definable in an o-minimal structure:

1. **Finite Stratification:** $\Sigma = \bigsqcup_{i=1}^N S_i$ (finitely many strata)

2. **Tameness:** Each $S_i$ is a smooth submanifold

3. **Curve Selection:** Limits along curves are definable

4. **No Wild Profiles:** Profile trichotomy excludes Case 3 (wild)

## Proof Sketch

### Step 1: O-Minimal Structures

**Definition (van den Dries, 1998):** An o-minimal structure on $\mathbb{R}$ is a collection $\mathcal{S} = \{\mathcal{S}_n\}_{n \geq 1}$ where:
1. $\mathcal{S}_n$ is a Boolean algebra of subsets of $\mathbb{R}^n$
2. $\{(x_1, \ldots, x_n) : x_i = x_j\} \in \mathcal{S}_n$
3. Projection: if $A \in \mathcal{S}_{n+1}$, then $\pi(A) \in \mathcal{S}_n$
4. $\mathcal{S}_1$ consists exactly of finite unions of points and intervals

**Reference:** van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.

### Step 2: Examples of O-Minimal Structures

**Semialgebraic:** Sets defined by polynomial inequalities

**Subanalytic:** Sets definable in $\mathbb{R}_{\text{an}}$ (restricted analytic functions)

**Exponential:** Sets definable in $\mathbb{R}_{\exp}$ (with exponential function)

**Reference:** Wilkie, A. (1996). Model completeness results for expansions of the ordered field of real numbers by restricted Pfaffian functions and the exponential function. *J. AMS*, 9, 1051-1094.

### Step 3: Cell Decomposition

**Theorem (Cell Decomposition):** Every definable set $X \subset \mathbb{R}^n$ is a finite disjoint union of cells:
$$X = \bigsqcup_{i=1}^N C_i$$

where each $C_i$ is homeomorphic to an open ball $B^{d_i}$.

**Reference:** van den Dries, L. (1998). *Tame Topology*. Cambridge. [Chapter 3]

**Consequence:** No Cantor sets, no fractals, no pathological topology.

### Step 4: Łojasiewicz Inequality in O-Minimal

**Kurdyka-Łojasiewicz Inequality (1998):** For definable $\Phi$ near critical point $x_*$:
$$|\nabla(\psi \circ \Phi)|(x) \geq 1$$

on $\{a < \Phi(x) < b\} \setminus \{x_*\}$, for some definable desingularizing function $\psi$.

**Reference:** Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Ann. Inst. Fourier*, 48, 769-783.

**Consequence:** Gradient flows converge to equilibria in finite length.

### Step 5: Curve Selection Lemma

**Theorem (Curve Selection):** If $x_0 \in \overline{X}$ for definable $X$, there exists definable curve $\gamma: [0, 1] \to \mathbb{R}^n$ with:
$$\gamma(0) = x_0, \quad \gamma((0, 1]) \subset X$$

**Reference:** Lojasiewicz, S. (1965). Ensembles semi-analytiques. IHES preprint.

**Application:** Limits of sequences in definable sets are reached along definable curves.

### Step 6: Stratification of Singular Sets

**Whitney Stratification:** For definable $\Sigma$:
$$\Sigma = S_0 \sqcup S_1 \sqcup \cdots \sqcup S_k$$

where:
- Each $S_i$ is a smooth manifold of dimension $i$
- Whitney conditions (a) and (b) hold

**Reference:** Whitney, H. (1965). Local properties of analytic varieties. *Differential and Combinatorial Topology*, Princeton.

**Consequence:** Singular sets have controlled topology, no accumulation of strata.

### Step 7: Dimension Bounds

**Definable Dimension:** For definable $X$:
$$\dim(X) = \max\{d : \text{cells of dimension } d \text{ in decomposition}\}$$

**Monotonicity:** Projection does not increase dimension:
$$\dim(\pi(X)) \leq \dim(X)$$

**Hausdorff Dimension:** For definable $X$:
$$\dim_{\mathcal{H}}(X) = \dim(X)$$

(Hausdorff and topological dimensions agree).

### Step 8: Profile Trichotomy in O-Minimal

**Theorem:** In o-minimal setting, Profile Trichotomy reduces to Cases 1 and 2:
- **Case 1 (Library):** Profiles in finite list
- **Case 2 (Tame):** Profiles in definable family

**Case 3 (Wild) Excluded:** Wild profiles have:
- Fractal dimension (not integer) — impossible for definable
- Cantor set structure — impossible in o-minimal
- Undecidable properties — impossible for definable

### Step 9: Tameness of Flows

**Theorem (Kurdyka-Parusiński, 2000):** Gradient flows of definable functions are definable.

**Reference:** Kurdyka, K., Parusiński, A. (2000). $w_f$-stratification of subanalytic functions and the Łojasiewicz inequality. *C. R. Acad. Sci. Paris*, 318, 129-133.

**Consequence:**
- Flow trajectories are definable curves
- Limit sets are definable
- Surgery regions are definable

### Step 10: Compilation Theorem

**Theorem (O-Minimal Promotion):**

1. **Cell Decomposition:** Definable sets have finite cell decomposition

2. **Stratification:** Singular sets have Whitney stratification

3. **Łojasiewicz:** Gradient inequality holds with definable desingularizing function

4. **Wild Exclusion:** Case 3 of profile trichotomy is empty

**Applications:**
- All singular sets are tame
- All blow-up limits are definable
- Surgery is always possible (no wild obstructions)

## Key GMT Inequalities Used

1. **Cell Decomposition:**
   $$X = \bigsqcup_{i=1}^N C_i$$

2. **Kurdyka-Łojasiewicz:**
   $$|\nabla(\psi \circ \Phi)| \geq 1$$

3. **Dimension Agreement:**
   $$\dim_{\mathcal{H}}(X) = \dim_{\text{top}}(X)$$

4. **Curve Selection:**
   $$x_0 \in \overline{X} \implies \exists \gamma: \gamma(0) = x_0, \gamma((0,1]) \subset X$$

## Literature References

- van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.
- Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Ann. Inst. Fourier*, 48.
- Wilkie, A. (1996). Model completeness results. *J. AMS*, 9.
- Łojasiewicz, S. (1965). Ensembles semi-analytiques. IHES.
- Whitney, H. (1965). Local properties of analytic varieties. Princeton.
- Kurdyka, K., Parusiński, A. (2000). $w_f$-stratification. *C. R. Acad. Sci. Paris*, 318.
