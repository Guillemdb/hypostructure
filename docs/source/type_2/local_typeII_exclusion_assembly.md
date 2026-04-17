# Local Type II exclusion assembly

This note adapts the previously established good-window criterion and
multibubble/cascade theorems to the local CKN entry used in
[local_ckn_typeII_bridge.md](local_ckn_typeII_bridge.md).  The conclusion is a
Type II exclusion theorem relative only to the upstream Type I criterion.  It does
not use whole-space critical-norm normalization, whole-space tightness, or a
terminal profile decomposition as starting assumptions.

Throughout, the starting point is a suitable weak solution and a singular point
with positive local CKN concentration supplied by the local concentration theorem.  The
Type I criterion is assumed already available.

For this note, the Type I criterion is used in its local tangent form: it excludes
any blow-up branch whose good-window rescalings have a nonzero bounded
parabolic tangent on a compact cylinder.  In physical variables such a tangent
has the self-similar amplitude \(|u|\sim r^{-1}\) on cylinders of radius \(r\),
so it belongs to the local Type I scale class.

## 1. Local single-core criterion

A local single-core Type II branch consists of parabolic blow-ups
\((u_n,p_n)\) at a singular point such that, after passing to a subsequence:

1. the local suitable package \(A+E+C+D\) is bounded on every compact backward
   cylinder;
2. a single retained core carries a fixed positive CKN density on a fixed
   cylinder;
3. localized scale and centering gauges are nondegenerate on that core;
4. local pressure is represented by the compact-ball decomposition
   \(P=P_{\mathrm{loc}}+H\);
5. local Caccioppoli estimates give windowed \(H^1\) control on compact
   cylinders;
6. the localized Type II cost is finite.

::::{prf:theorem} Local single-core Type II criterion
:label: thm-local-single-core-typeII-criterion

Assume the local Type I tangent criterion.  No local single-core Type II branch
satisfying the six conditions above can occur.

::::

:::{prf:proof}
Finite localized Type II cost gives good windows on which the localized
renormalized dissipation tends to zero.  The local Caccioppoli bounds and the
compact-cylinder suitable bounds give compactness on those same windows.
After passing to a subsequence, the good-window solutions converge on a compact
spacetime cylinder strongly in local \(L^2\) and in subcritical local spaces,
and weakly in the local suitable class.

The vanishing dissipation implies that the spatial gradient of the limiting
velocity vanishes on the retained spacetime core.  Hence the limiting velocity
is spatially constant on that cylinder.

There are two cases.

If the constant is zero, then strong local convergence forces the retained CKN
velocity density on a smaller cylinder to vanish.  In the spacetime suitable
limit the velocity is zero on that cylinder, so the local equation gives a
spatially constant pressure there; after subtracting spatial means, the pressure
density also vanishes.  This contradicts the positive retained CKN density.

If the constant is nonzero, then the good-window rescalings have a nonzero
bounded parabolic tangent on the retained core.  By the local tangent form of
the Type I criterion fixed above, this branch is Type I and is excluded upstream.

Both alternatives contradict the hypotheses.  Therefore the local single-core
Type II branch is impossible.  \(\square\)
:::

This is the local replacement for the old good-window compact criterion.  The
old proof used whole-space normalization and whole-space tightness to rule out a
constant low-dissipation limit.  In the local CKN formulation the correct
replacement is the zero/nonzero constant dichotomy above: zero contradicts
retained CKN mass; nonzero is a Type I tangent and is removed by the Type I
criterion.

## 2. Local multibubble and cascade exclusion

A local multibubble/cascade branch is any positive-concentration Type II branch
for which at least one single-core hypothesis fails in the following structural
way:

1. the compact-cylinder suitable package \(A+E+C+D\) is unbounded on some fixed
   rescaled cylinder;
2. positive CKN mass splits into two or more comparable retained cores;
3. a strict same-point scale cascade appears inside bounded rescaled cylinders;
4. separated physical points remain active in different local rescaled frames;
5. the localized scale or centering gauge degenerates because no unique core is
   selected.

The local multibubble/cascade analysis consists of the following proved local
reductions.

- Non-single-core Type II residue is reduced to multibubble concentration.
- Regular same-point cascades are removed, and the remaining same-point cascade
  case is reduced to nonlinear no-splitting.
- Separated-point multibubbles are reduced to same-point and separated-frame
  decoupling theorems.
- Those decoupling theorems are reduced to nonlinear profile-evolution
  decoupling.
- The nonlinear decoupling is proved in terminal active local rescaled frames
  using local suitable compactness, exterior discard, repaired-gauge
  representation, and Caccioppoli regularity.

::::{prf:theorem} Local multibubble/cascade exclusion
:label: thm-local-multibubble-cascade-exclusion

Assume the local Type I tangent criterion.  Then the local multibubble/cascade
decoupling and scale-rigidity theorems exclude every local multibubble or
cascade Type II branch.

::::

:::{prf:proof}
If compact-cylinder bounds fail, there are a fixed radius \(R<\infty\) and a
subsequence on which at least one component of the local suitable package
diverges in \(Q_R\).  By the standard local energy/Caccioppoli estimate for suitable weak
solutions, boundedness of \(C+D\) on a slightly larger cylinder controls
\(A+E\) on a smaller concentric cylinder.  Hence failure of the compact-cylinder
suitable package forces failure of the local \(C+D\) bound on a comparable
cylinder, after possibly changing the radius by a fixed factor.  The local
positive-concentration alternative then gives another active local concentration
core.  After passing to a further subsequence, that core is either comparable
to the original core, lies at a strict same-point subscale, or is separated in
physical space.  These are the cases in the same-point/separated-point classification.

Comparable same-point cores are combined into a compound active core.  If the
compound core is single, it returns to the single-core criterion.  If it is not
single, the localized gauge matrix is degenerate and the branch remains in the
multibubble class.

Strict same-point cascades are treated by the same-point cascade theorem.
Regular cascades are excluded by passing to the innermost active rescaled frame
and applying the scale-rigidity theorem.
The remaining same-point cascades are precisely nonlinear no-splitting failures,
which the decoupling reduction reduces to the nonlinear profile-decoupling theorem.

Separated physical points are treated by the separated-frame decoupling theorem.
In the rescaled frame of any active point,
all other active points are locally invisible in velocity and pressure after the
local decoupling theorems.  The selected rescaled frame therefore reduces to a single-core or
scale-rigidity branch.  The single-core branch is excluded by Theorem
{prf:ref}`thm-local-single-core-typeII-criterion`; the scale-rigidity branch is
excluded by the scale-rigidity theorem and the Type I criterion.

The terminal nonlinear decoupling theorem supplies the nonlinear
profile-decoupling theorem in terminal active local rescaled frames.  Hence
neither same-point nor separated-point multibubble/cascade configurations
remain.  \(\square\)
:::

## 3. Cost and rough-core alternatives

::::{prf:lemma} Local cost alternatives on a represented single core
:label: lem-local-cost-alternatives

For a represented local single-core Type II branch, exactly one of the following
holds on the tail of the renormalized time interval.

1. The localized Type II cost is finite.  Then the good-window argument in
   Theorem {prf:ref}`thm-local-single-core-typeII-criterion` applies.
2. The localized scale-drift contribution has infinite negative accumulation
   against a nonvanishing localized core.  This is the scale-collapse
   alternative and is excluded by the local scale-rigidity theorem.
3. The local windowed \(H^1\) control required for the good-window argument
   fails.  By the compact-cylinder Caccioppoli theorem, this forces failure of
   the local \(C+D\) bound on a comparable cylinder and hence enters the
   multibubble/cascade concentration alternative.

::::

:::{prf:proof}
The localized cost is nonnegative and measurable on finite windows.  If its tail
integral is finite, the standard vanishing-window selection gives good windows
with vanishing average cost, which is the input used in the single-core
criterion.  If the obstruction is the negative scale-drift term with a
nonvanishing localized \(L^2\) core, the scale-drift identity gives the
scale-collapse alternative handled by the scale-rigidity theorem.  The only
remaining way for the good-window compactness argument to fail on a represented
single core is loss of local windowed \(H^1\) control.  The compact-cylinder
Caccioppoli estimate bounds that quantity whenever the local \(C+D\) package,
pressure decomposition, and local modulation coefficients are bounded; pressure
and modulation are supplied by the representation theorem.  Therefore loss of
windowed control forces a comparable bounded-cylinder concentration branch.
\(\square\)
:::

## 4. Type II exclusion theorem

::::{prf:theorem} Local CKN Type II exclusion
:label: thm-local-ckn-typeII-exclusion

Assume the local Type I tangent criterion.  Let \((u,p)\) be a suitable weak
Navier-Stokes solution on \(\mathbb R^3\times(T-\delta,T)\).  Then no
singular point at time \(T\) can be a Type II singularity.

::::

:::{prf:proof}
Let \((x_0,T)\) be a singular point.  The local concentration theorem gives
positive local CKN concentration at \((x_0,T)\).  If the local Type I tangent
branch holds, it is excluded by the Type I criterion.  Otherwise the point
enters the Type II branch.

Apply the local compactness dichotomy from
[local_ckn_typeII_bridge.md](local_ckn_typeII_bridge.md).  If the local suitable
package is bounded on compact cylinders and the localized gauges are
nondegenerate, the branch is a represented local single-core Type II branch.
Lemma {prf:ref}`lem-local-cost-alternatives` then applies.  In the finite-cost
case, Theorem {prf:ref}`thm-local-single-core-typeII-criterion` excludes the
branch.  In the scale-drift or rough-core cases, Lemma
{prf:ref}`lem-local-cost-alternatives` sends the branch to scale-rigidity or to
multibubble/cascade concentration.

If compact-cylinder bounds fail, mass splits, scales cascade, or gauges
degenerate, the branch is multibubble/cascade and is excluded by Theorem
{prf:ref}`thm-local-multibubble-cascade-exclusion`.

These alternatives exhaust all positive local Type II concentration branches.
Therefore no Type II singularity occurs.  \(\square\)
:::

## 5. Local inputs proved or reduced

The theorem uses only local inputs after the Type I criterion.  The local
multibubble/cascade decoupling and scale-rigidity theorems provide the
subproof used in Section 2:

- suitability and the local energy inequality;
- CKN epsilon regularity and positive local CKN concentration;
- compact-cylinder suitable compactness or its multibubble/cascade negation;
- compact-ball pressure decomposition;
- localized repaired gauges on the single-core branch;
- local Caccioppoli estimates;
- the local multibubble/cascade decoupling and scale-rigidity theorems.

The former whole-space critical-norm normalization, whole-space tightness, and
terminal profile-decomposition starting assumptions are not used.
