# Local Exterior Regularity Stratification

This note records the local replacement for `paper8:ass:exterior-regularity`.
The relevant assertion is not an all-branch exterior smoothness theorem.
Instead, each fixed exterior annulus is treated by one of two local outcomes:

1. the annulus has no terminal exterior concentration, so standard local
   regularity estimates make the exterior component perturbative in the
   selected core frame;
2. the annulus carries terminal exterior concentration, so that concentration is
   extracted as an exterior, separated-core, or multicore concentration branch.

This is the form needed by Section 8.  The retained single-core branch obtains
exterior regularity only after the exterior concentration tests pass.  Failure
of those tests is retained as an adjacent concentration alternative.

## Manuscript Status

The previous assumption label

```text
paper8:ass:exterior-regularity
```

is no longer used as a hypothesis in the manuscript.  The local annular package
is

```text
paper8:def:exterior-ckn-quantity
paper8:def:no-exterior-concentration-stratum
paper8:def:exterior-concentration-branch
paper8:lem:exterior-test-dichotomy
paper8:thm:exterior-stratification
paper8:lem:exterior-ckn-cover
paper8:lem:exterior-pressure-localization
paper8:thm:exterior-regularity-no-concentration
paper8:cor:exterior-annulus-alternative
```

The later exterior lemmas use the no-exterior-concentration branch of this
stratification:

```text
paper8:lem:core-exterior-decomposition
paper8:lem:exterior-velocity-vanishes
paper8:lem:exterior-pressure-vanishes
paper8:lem:positive-distance-exterior
paper8:thm:exterior-regular-removal
```

## Guiding Principle

Work only on compact annuli and compact terminal windows.

Fix a selected core point \((x_*,T^*)\).  For
\[
  0<r<R<\infty,
\]
write
\[
  A_{r,R}:=\overline{B_R(x_*)}\setminus B_r(x_*).
\]

All estimates are made inside cylinders whose spatial support stays in
slightly enlarged annuli, for example
\[
  A_{r/2,2R}.
\]

No estimate is required away from the chosen annular stratum.

## Local Concentration Quantity

Use the standard Caffarelli--Kohn--Nirenberg scale-invariant local quantity.
For \(z_0=(x_0,t_0)\) and \(\rho>0\), define
\[
  Q_\rho(z_0):=B_\rho(x_0)\times(t_0-\rho^2,t_0)
\]
and
\[
  \mathcal C(u,p;z_0,\rho)
  :=
  \rho^{-2}\int_{t_0-\rho^2}^{t_0}\int_{B_\rho(x_0)} |u|^3\,dx\,dt
  +
  \rho^{-2}\int_{t_0-\rho^2}^{t_0}\int_{B_\rho(x_0)}
  |p-(p)_{B_\rho(x_0)}(t)|^{3/2}\,dx\,dt.
\]

The pressure is measured modulo functions of time.  This matches the pressure
normalization already used in the repaired-gauge parts of the manuscript.

Let \(\varepsilon_{\rm CKN}\) be a fixed epsilon-regularity threshold:
if
\[
  \mathcal C(u,p;z_0,\rho)\le \varepsilon_{\rm CKN},
\]
then \(u\) is bounded, and then smooth by interior parabolic estimates, on a
smaller cylinder \(Q_{\rho/2}(z_0)\).

## Exterior State Test

For each annulus \(A_{r,R}\), define the exterior test near \(T^*\).

The branch passes the exterior test on \(A_{r,R}\) if there are
\[
  \delta=\delta(r,R)>0,\qquad \rho_*=\rho_*(r,R)>0
\]
such that every cylinder \(Q_\rho(z_0)\) satisfying
\[
  x_0\in A_{r,R},\qquad
  0<\rho\le \rho_*,
  \qquad
  t_0\in(T^*-\delta,T^*),
\]
and
\[
  Q_\rho(z_0)\subset A_{r/2,2R}\times(T^*-\delta,T^*)
\]
obeys
\[
  \mathcal C(u,p;z_0,\rho)\le\varepsilon_{\rm CKN}.
\]

The branch fails the exterior test on \(A_{r,R}\) if there are sequences
\[
  x_n\in A_{r,R},\qquad t_n\uparrow T^*,\qquad \rho_n\downarrow0
\]
with
\[
  Q_{\rho_n}(x_n,t_n)\subset A_{r/2,2R}\times(0,T^*)
\]
and
\[
  \mathcal C(u,p;(x_n,t_n),\rho_n)\ge \varepsilon_{\rm CKN}.
\]

This is the complementary exterior concentration alternative.

## Main Replacement Theorem

### Theorem 1: Exterior Concentration Stratification

Let \(u,p\) be the suitable Navier--Stokes solution considered in the Type II
branch, and let \((x_*,T^*)\) be the selected core.  Fix an annulus
\[
  A_{r,R}.
\]
Then exactly one of the following alternatives is reached after
passing to a subsequence of terminal times:

1. **No exterior concentration on \(A_{r,R}\).**  
   The exterior test passes on \(A_{r,R}\).  Consequently \(u\) and \(p\),
   modulo time functions for the pressure, satisfy local \(C^k\) bounds on
   every compact subannulus
   \[
     A_{r',R'}\Subset A_{r,R}
   \]
   on a terminal time interval \((T^*-\delta',T^*)\).

2. **Exterior concentration.**  
   The exterior test fails.  Then there is an exterior concentration sequence
   \[
     (x_n,t_n,\rho_n)
   \]
   with \(x_n\in A_{r,R}\), \(t_n\uparrow T^*\), \(\rho_n\downarrow0\), and
   \[
     \mathcal C(u,p;(x_n,t_n),\rho_n)\ge\varepsilon_{\rm CKN}.
   \]
   The branch belongs to a named adjacent exterior stratum: exterior profile,
   separated core, or multicore branch, depending on the relation between
   \(\rho_n\) and the selected core scale.

This theorem replaces `paper8:ass:exterior-regularity` on the retained
single-core/no-exterior-concentration stratum.  It does not eliminate exterior
concentration; it includes that case as an adjacent alternative.

## Relation Between Exterior Scale and Core Scale

Let \(\lambda_n\) be the selected core scale at the terminal times \(t_n\).
For an exterior concentration sequence \((x_n,t_n,\rho_n)\), compare
\(\rho_n\) with \(\lambda_n\).

### Case A: \(\rho_n/\lambda_n\to0\) or \(\lambda_n/\rho_n\to0\)

The exterior concentration is scale-separated from the core and belongs to a
separated exterior profile or scale-separated multicore stratum.

Required lemma:

```text
paper8:lem:exterior-scale-separated-routing
```

Statement: if \(x_n\) stays a positive physical distance from \(x_*\) and the
concentration scale is asymptotically different from the selected core scale,
then the corresponding rescaled fields are orthogonal to the core frame and
define a separated exterior profile branch.

### Case B: \(\rho_n/\lambda_n\to c\in(0,\infty)\)

The exterior concentration scale is comparable to the selected core scale but
its center stays a positive physical distance from \(x_*\).  In the core frame,
the center escapes to spatial infinity:
\[
  \frac{x_n-x_*}{\lambda_n}\to\infty.
\]

Required lemma:

```text
paper8:lem:comparable-scale-exterior-routing
```

Statement: comparable-scale exterior concentration at positive physical
distance becomes an escaping-center profile in the selected core coordinates.
It is therefore not part of the retained compact core and belongs to the
exterior-profile or escaping multicore stratum.

### Case C: \(x_n\to x_*\) but \(x_n\notin B_r(x_*)\) for fixed \(r>0\)

This case cannot occur for a fixed annulus \(A_{r,R}\).  It appears only when
one lets \(r\downarrow0\).  That limit belongs to the inner-core/multicore
analysis, not to the fixed exterior annulus proof.

This separation is important: the exterior theorem is proved for fixed annuli.
The \(r\downarrow0\) collapse belongs to the inner-core and multicore analysis.

## Required Lemmas

### Lemma 2: Failure of Exterior Smallness Produces a Concentration Sequence

If the exterior test fails on \(A_{r,R}\), then there exist
\[
  x_n\in A_{r,R},\quad t_n\uparrow T^*,\quad \rho_n\downarrow0
\]
with
\[
  \mathcal C(u,p;(x_n,t_n),\rho_n)\ge\varepsilon_{\rm CKN}.
\]

Proof strategy:

Use only the negation of the exterior test.  No bound outside the annulus is
introduced.  If the bad radii did not satisfy \(\rho_n\downarrow0\), then a
positive-radius cylinder would carry persistent concentration away from the
selected core and would already be an exterior branch at a fixed physical
scale.

### Lemma 3: No-Exterior-Concentration Gives a CKN Cover

Assume the exterior test passes on \(A_{r,R}\).  Then for every compact
subannulus
\[
  A_{r',R'}\Subset A_{r,R}
\]
there is a finite parabolic cover of
\[
  A_{r',R'}\times(T^*-\delta',T^*)
\]
by cylinders on which
\[
  \mathcal C(u,p;z_j,\rho_j)\le\varepsilon_{\rm CKN}.
\]

Proof strategy:

Use compactness of \(A_{r',R'}\) and the local terminal time interval produced
by the exterior test.  The finite cover is annulus-local.  Constants may depend
on \(r,r',R,R'\) and the chosen cover.

### Lemma 4: CKN Cover Gives Local Velocity Bounds

On the no-exterior-concentration stratum, for every compact subannulus
\[
  A_{r',R'}\Subset A_{r,R}
\]
there are \(\delta'>0\) and \(C_{r',R'}\) such that
\[
  \|u\|_{L^\infty(A_{r',R'}\times(T^*-\delta',T^*))}
  \le C_{r',R'}.
\]

Proof strategy:

Apply CKN epsilon regularity on each cylinder in the finite cover.  Patch the
finite bounds.  The constant is only a local annular constant.

### Lemma 5: Local Pressure Decomposition on Exterior Annuli

On \(A_{r',R'}\Subset A_{r,R}\), decompose
\[
  p = p_{\rm loc}+p_{\rm harm}+c(t)
\]
on a slightly larger annulus, where
\[
  -\Delta p_{\rm loc}
  =
  \partial_i\partial_j\bigl(u_i u_j \eta\bigr)
\]
for a cutoff \(\eta\) supported in the larger annulus, and \(p_{\rm harm}\) is
harmonic on the smaller annulus.

Required conclusion:

If \(u\) is locally bounded on the larger annulus, then for every compact
subannulus and every \(q<\infty\),
\[
  p-c(t)
  \quad\text{is locally bounded in}\quad
  L^\infty_t L^q_x
\]
and then in local Hölder/Schauder classes after bootstrapping.

Proof strategy:

Use Calderon--Zygmund estimates for \(p_{\rm loc}\), harmonic estimates for
\(p_{\rm harm}\), and pressure normalization modulo \(c(t)\).  This is
annulus-local because the cutoff is supported away from the core.

### Lemma 6: Exterior Parabolic Bootstrapping

On the no-exterior-concentration stratum, for every integer \(k\ge0\),
\[
  \sup_{t\in(T^*-\delta_k,T^*)}
  \left(
  \|u(t)\|_{C^k(A_{r',R'})}
  +
  \|p(t)-c(t)\|_{C^k(A_{r',R'})}
  \right)
  \le C_{r',R',k}.
\]

Proof strategy:

Once \(u\) is locally bounded and \(p\) is locally controlled modulo time
functions, apply interior Stokes/Navier--Stokes estimates on nested annuli.
Use a standard bootstrap:

1. \(u\in L^\infty\) and \(p\in L^q\) locally;
2. local energy and equation give \(u\) in parabolic \(W^{2,1}_q\);
3. parabolic embedding gives Hölder continuity;
4. Schauder or \(L^q\)-bootstrap gives all \(C^k\) bounds on smaller annuli.

All constants are allowed to depend on the fixed annulus and the number of
derivatives.

### Lemma 7: Exterior Regular Field Vanishes in Compact Core Frames

Assume the no-exterior-concentration branch on \(A_{r,R}\).  Let
\[
  x_n\to x_*,\qquad \lambda_n\to0
\]
be a selected core frame.  If \(u_{\rm ext}\) is supported away from
\[
  B_{r'}(x_*)
\]
for some \(r'>0\), then
\[
  \lambda_n u_{\rm ext}(x_n+\lambda_n y,t_n+\lambda_n^2\tau)\to0
\]
in every compact core cylinder.  In the specific decomposition already used in
the manuscript, it is eventually identically zero on each compact core
cylinder.

Proof strategy:

This is essentially the existing `paper8:lem:exterior-velocity-vanishes`, but
the input is the no-exterior-concentration theorem rather than the old
assumption.

### Lemma 8: Exterior Pressure Vanishes Modulo Time Functions

Assume the exterior pressure source is supported a positive distance from the
selected core.  In a compact core frame,
\[
  \lambda_n^2 p_{\rm ext}(x_n+\lambda_n y,t_n+\lambda_n^2\tau)
  -
  c_n(\tau)
  \to0
\]
locally, with convergence at least in
\[
  L^\infty_\tau C^1_y
\]
on compact core cylinders.

Proof strategy:

This is the existing `paper8:lem:exterior-pressure-vanishes`.  The new proof
uses Lemma 5 and Lemma 6 to justify the local smoothness of the exterior
pressure on the no-exterior-concentration stratum.

## Extraction of the Exterior Branch

The complementary stratum is explicitly represented by the exterior
concentration sequence from Lemma 2:

```text
paper8:def:exterior-concentration-branch
paper8:thm:exterior-stratification
```

### Definition: Exterior Concentration Branch

An exterior concentration branch consists of sequences
\[
  x_n\in A_{r,R},\qquad t_n\uparrow T^*,\qquad \rho_n\downarrow0
\]
such that
\[
  \mathcal C(u,p;(x_n,t_n),\rho_n)\ge\varepsilon_{\rm CKN},
\]
with \(Q_{\rho_n}(x_n,t_n)\) staying inside the exterior annular region.

The associated rescaled fields are
\[
  U_n(y,s):=\rho_n u(x_n+\rho_n y,t_n+\rho_n^2 s),
\]
\[
  P_n(y,s):=\rho_n^2 p(x_n+\rho_n y,t_n+\rho_n^2 s).
\]

### Exterior Branch Alternatives

After passing to a subsequence, the exterior concentration branch satisfies one
of the following:

1. it is scale-center orthogonal to the selected core and belongs to the exterior
   profile branch;
2. it is one of finitely many separated active centers and belongs to the multicore
   branch;
3. it has scale-collapse behavior inside the exterior profile and belongs to the
   scale-collapse/cost branch;
4. it fails compactness or tightness in the exterior frame and belongs to the
   corresponding noncompact exterior stratum.

This is a classification of the complementary concentration alternative, not
an exclusion theorem.  Failure of exterior regularity is therefore represented
inside the local concentration-compactness decomposition.

## Implemented Package

### Step 1: Local Exterior State Definitions

The package introduces:

```text
paper8:def:exterior-ckn-quantity
paper8:def:no-exterior-concentration-stratum
paper8:def:exterior-concentration-branch
```

These definitions are completely local to fixed annuli.

### Step 2: Dichotomy

The dichotomy is:

```text
paper8:lem:exterior-test-dichotomy
paper8:thm:exterior-stratification
```

The theorem states that each fixed annulus either passes the CKN test or
produces an exterior concentration branch.

### Step 3: Regularity on the No-Concentration Stratum

The regularity package is:

```text
paper8:lem:exterior-ckn-cover
paper8:lem:exterior-pressure-localization
paper8:thm:exterior-regularity-no-concentration
```

This theorem replaces the old assumption for all later lemmas.

### Step 4: Vanishing in Core Coordinates

Updated exterior lemmas:

```text
paper8:lem:core-exterior-decomposition
paper8:lem:exterior-velocity-vanishes
paper8:lem:exterior-pressure-vanishes
paper8:lem:positive-distance-exterior
paper8:thm:exterior-regular-removal
```

Each assumes the no-exterior-concentration stratum, or cites
`paper8:thm:exterior-regularity-no-concentration`.

### Step 5: Complementary Stratum

The complementary alternative is:

```text
paper8:cor:exterior-annulus-alternative
```

The corollary says:

For every fixed exterior annulus, either the exterior contribution is regular
and perturbative in the selected core frame, or the branch belongs to a named
exterior concentration alternative.

## What This Proves and What It Does Not Prove

It proves:

* exterior regularity on fixed annuli after the no-exterior-concentration test
  passes;
* pressure control modulo time functions on the same fixed annuli;
* perturbative disappearance of exterior regular fields in compact core frames;
* explicit placement of failed exterior tests in exterior/multicore strata.

It does not prove:

* that exterior concentration never occurs;
* that the selected core is the only possible active center;
* any bound outside the fixed annular stratum under consideration;
* any uniform statement over all annuli without running the annular tests.

Those questions belong to the adjacent exterior, separated-core, multicore,
scale-collapse, or terminal-profile alternatives.

## Why This Gives Leverage

With this local exterior alternative in place, the following assumptions become
more accessible:

1. `paper7:ass:uniform-cost-exclusion-data`  
   A major non-retained exit from the compact cost channel becomes a named
   exterior branch.

2. `paper6a:ass:critical-ns-profile-decomposition`  
   Exterior concentration sequences are now explicitly tied to terminal profile
   extraction, so the no-hidden-terminal-mass statement has a concrete target.

3. `paper6a:ass:cost-divergence-exclusion`  
   The local cost criterion can be applied on retained compact branches without
   absorbing exterior leakage into the cost error terms.

4. `paper3:ass:scale-rigidity`  
   Some apparent rigid or degenerate branches may be identified as exterior or
   multicore concentration once exterior tests are made explicit.

## Minimal First Milestone

The first milestone is:

```text
paper8:thm:exterior-regularity-no-concentration
```

with the following dependency chain:

```text
paper8:def:exterior-ckn-quantity
paper8:def:no-exterior-concentration-stratum
paper8:lem:exterior-ckn-cover
paper8:lem:exterior-pressure-localization
paper8:thm:exterior-regularity-no-concentration
```

This milestone removes the analytic content of the old exterior
regularity assumption on the retained no-exterior-concentration stratum.  The
second milestone is the classification theorem for the complementary exterior
concentration branch.
