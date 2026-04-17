# Local CKN bridge into the Type II exclusion theorem

This note replaces the former whole-space critical-norm starting point by a local
Caffarelli-Kohn-Nirenberg starting point.  It proves the initial reduction from a
suitable weak singularity to positive local concentration and records exactly
which local Type II statements must be used downstream.  No whole-space
critical-norm bound, whole-space tightness statement, or terminal profile
decomposition is assumed from this reduction.

Throughout, \((u,p)\) is a suitable weak solution of the three-dimensional
Navier-Stokes equations on \(\mathbb R^3\times(T-\delta,T)\).  For
\(z_0=(x_0,T)\), set
\[
Q_r(z_0)=B_r(x_0)\times(T-r^2,T),
\]
and define the CKN quantities
\[
C(z_0,r)=r^{-2}\int_{Q_r(z_0)}|u|^3\,dx\,dt,
\]
\[
D(z_0,r)=r^{-2}\int_{T-r^2}^{T}\int_{B_r(x_0)}
 |p(x,t)-(p)_{B_r(x_0)}(t)|^{3/2}\,dx\,dt .
\]
The pressure is always normalized modulo functions of time by subtracting its
spatial mean on the ball appearing in the local estimate.

## 1. Local concentration at singular points

::::{prf:theorem} Singular points have positive CKN concentration
:label: thm-local-ckn-singularity-concentration

If \((x_0,T)\) is singular, then
\[
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)>0.
\]
Equivalently, there are \(\eta>0\) and \(r_n\downarrow0\) such that
\[
C((x_0,T),r_n)+D((x_0,T),r_n)\ge \eta
\qquad\text{for all }n.
\]

::::

:::{prf:proof}
This is the contrapositive of CKN epsilon regularity, in the pressure-normalized
form stated in the local concentration theorem notes.  If the displayed limsup were zero, then
for some sufficiently small \(r\) the quantity \(C((x_0,T),r)+D((x_0,T),r)\)
would be below the universal epsilon-regularity threshold.  The point
\((x_0,T)\) would then be regular, contrary to hypothesis.  \(\square\)
:::

This proves that the no-concentration branch is closed before any Type I or
Type II analysis begins.

## 2. Rescaling and the local compactness dichotomy

Let \(r_n\downarrow0\) be a sequence supplied by Theorem
{prf:ref}`thm-local-ckn-singularity-concentration`, and define
\[
u_n(y,s)=r_n u(x_0+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_0+r_ny,T+r_n^2s).
\]
Then \((u_n,p_n)\) is suitable on every fixed backward cylinder
\(B_R\times(-R^2,0)\), once \(n\) is large enough, and
\[
\int_{Q_1}\left(|u_n|^3+
 |p_n-(p_n)_{B_1}(s)|^{3/2}\right)\,dy\,ds\ge \eta .
\]
This lower bound gives nontriviality only.  Compactness requires the full local
suitable package.  For a suitable solution \((v,q)\) on \(Q_R\), write
\[
A_{v}(R)=R^{-1}\operatorname*{ess\,sup}_{-R^2<s<0}
  \int_{B_R}|v(y,s)|^2\,dy,
\]
\[
E_{v}(R)=R^{-1}\int_{Q_R}|\nabla v|^2\,dy\,ds,
\]
and keep \(C_{v,q}(R)\), \(D_{v,q}(R)\) for the corresponding scaled
\(L^3\) and pressure quantities.  The correct local statement is the
following dichotomy.

::::{prf:theorem} Local compactness dichotomy
:label: thm-local-compactness-dichotomy

After passing to a subsequence, exactly one of the following alternatives holds.

1. For every fixed \(R<\infty\),
   \[
   \sup_n\left(A_{u_n}(R)+E_{u_n}(R)+C_{u_n,p_n}(R)+D_{u_n,p_n}(R)\right)<\infty .
   \]
   In this case the standard compactness theorem for suitable weak solutions
   gives an ancient suitable weak limit \((U,P)\) on
   \(\mathbb R^3\times(-\infty,0)\), after extracting a subsequence.

2. There is a fixed \(R<\infty\) for which the preceding supremum is
   infinite.  Then additional local energy, dissipation, velocity, or pressure
   concentration occurs on a bounded rescaled cylinder.  This is a local
   cascade or multibubble alternative, not a single-core compact branch.

::::

:::{prf:proof}
If the bounds in the first alternative hold, the local energy bounds, local
dissipation bounds, local pressure bounds, and the local energy inequality are
stable under the parabolic rescaling.  The usual compactness theorem for
suitable weak solutions gives a subsequence converging locally in the weak suitable class, with strong
convergence in subcritical local spaces and pressure convergence modulo spatial
means.  If the first alternative fails, its negation is exactly the second
alternative.  \(\square\)
:::

Thus local compactness is not an independent assumption.  It is the single-core
side of a local dichotomy.  Its failure is converted, by the local Caccioppoli
contrapositive below, into another bounded-cylinder concentration branch.


::::{prf:lemma} Compactness failure gives bounded-cylinder concentration
:label: lem-local-compactness-failure-concentration

If the first alternative in Theorem
{prf:ref}`thm-local-compactness-dichotomy` fails, then after changing the
radius by a fixed factor there is a sequence of bounded cylinders on which
\(C_{u_n,p_n}+D_{u_n,p_n}\) is unbounded, or at least bounded from below by a
new positive constant at smaller scales.  Hence the failure is a local
multibubble or cascade concentration branch.

::::

:::{prf:proof}
The local Caccioppoli inequality for suitable weak solutions gives, for
concentric cylinders with \(0<r<R\),
\[
A_{u_n}(r)+E_{u_n}(r)
 \le C(r,R)\,F\left(C_{u_n,p_n}(R),D_{u_n,p_n}(R)\right),
\]
where \(F\) is finite on bounded sets, after subtracting spatial pressure
means.  Therefore boundedness of
\(C+D\) on every comparable larger cylinder implies boundedness of \(A+E\) on
smaller cylinders.  If the full suitable package is unbounded on a fixed
compact cylinder, then \(C+D\) must fail to be bounded on a comparable cylinder.
A further parabolic rescaling at scales where \(C+D\) is positive gives another
bounded-cylinder concentration branch.  \(\square\)
:::

## 3. Type I/Type II separation

Assume the local Type I tangent criterion has already been proved in the form
used by [local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md):
any blow-up branch whose rescaled good-window limit is a nonzero bounded
parabolic tangent on a compact cylinder is excluded.  A positive-concentration
singularity not removed by that criterion is called a local Type II branch in
this note.

This separation uses only compact parabolic cylinders and the CKN density.  It
does not infer any whole-space critical-norm bound.

## 4. Local pressure representation

All pressure estimates used downstream are local.  On \(B_R\), choose
\(\zeta\in C_c^\infty(B_{2R})\) equal to one on \(B_R\), and write
\[
P=P_{\mathrm{loc}}+H,
\]
where
\[
-\Delta P_{\mathrm{loc}}
 =\partial_i\partial_j(\zeta V_iV_j)
\]
in \(\mathbb R^3\), and \(H\) is harmonic in \(B_R\).  Calderon-Zygmund
estimates control \(P_{\mathrm{loc}}\) by the local \(L^3\) norm of
\(V\), while interior harmonic estimates control \(H-(H)_{B_R}\) on
smaller balls.  This is the same pressure normalization used in CKN epsilon
regularity and in local Caccioppoli inequalities.

Therefore the whole-space Riesz-transform pressure formula is not an entry
hypothesis for Type II exclusion.

## 5. Local repaired gauges on the single-core branch

On the compact single-core branch one may impose compactly supported gauges. For
fixed large \(R\), let \(\chi_R\in C_c^\infty(B_{2R})\) equal one on
\(B_R\), and define
\[
G_{\mathrm{sc},R}(V)=
\int \chi_R(y)|y|^{-p}|V(y,\tau)|^3\,dy-\Theta_0,
\qquad 0<p<3,
\]
\[
G_{j,R}(V)=\int y_j\chi_R(y)|V(y,\tau)|^2\,dy,
\qquad j=1,2,3.
\]
The derivative of the scale row in the scale direction is
\[
D G_{\mathrm{sc},R}(V)[V+y\cdot\nabla V]
 =p\int \chi_R|y|^{-p}|V|^3
  -\int (y\cdot\nabla\chi_R)|y|^{-p}|V|^3 .
\]
If the retained core occupies \(B_R\) and the cutoff-annulus contribution is a
strictly smaller fraction of the core integral, this row is transverse.  The
centering rows are controlled by the corresponding localized moment matrix.
Failure of the localized transversality or moment invertibility means that a
unique core has not been selected at that scale; this is the gauge-degenerate
part of the multibubble alternative.

Thus repaired-gauge solvability is a local implicit-function theorem on the
single-core branch.  It is not a whole-space starting assumption.

## 6. Local consequences used downstream

The preceding sections prove the following local inputs for the Type II
argument.

1. A singular point supplies positive local CKN concentration.
2. The no-concentration case is closed by CKN epsilon regularity.
3. Local pressure estimates use compact-ball pressure decomposition, not a
   whole-space pressure representation.
4. On a genuine single-core branch, repaired gauges are obtained by a local
   implicit-function theorem; failure of transversality is a gauge-degenerate
   multibubble alternative.
5. Local compactness is a dichotomy: either compact-cylinder upper bounds for
   the suitable package give an ancient suitable weak limit, or their failure
   produces another bounded-cylinder CKN concentration branch.

These statements are local consequences of suitability, CKN regularity,
parabolic scaling, local pressure theory, and the single-core transversality
condition.

## 7. Local Type II criteria

The local criteria used after this bridge are established in
[local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md).  They
adapt the earlier good-window compact argument and the multibubble/cascade
decoupling theorems to the local CKN entry.

The two local Type II criteria are:

1. **Single-core criterion.** A locally compact single-core Type II branch with
   positive retained CKN mass, nondegenerate localized gauges, local
   Caccioppoli estimates, local windowed \(H^1\) control, and finite localized
   Type II cost cannot occur.  The zero-dissipation limit is handled by the
   local zero/nonzero constant dichotomy: zero contradicts retained CKN mass;
   nonzero is a Type I tangent and is excluded by the local Type I tangent criterion.
2. **Multibubble/cascade criterion.** Failure of compact-cylinder bounds, mass
   splitting, strict same-point cascades, separated active points, localized
   gauge degeneracy, infinite-cost scale drift, and rough-core loss of local
   windowed control are excluded by the local decoupling, scale-rigidity, and
   Caccioppoli theorems.

## 8. Type II exclusion theorem

::::{prf:theorem} Local entry and Type II exclusion
:label: thm-local-entry-to-local-typeII-bridge

Assume the local Type I tangent criterion.  Let \((u,p)\) be a suitable weak Navier-Stokes
solution with a finite-time singularity at \(T\).  Then the singularity is not a
Type II singularity.

::::

:::{prf:proof}
Let \(x_0\) be a singular point at time \(T\).  By Theorem
{prf:ref}`thm-local-ckn-singularity-concentration`, \((x_0,T)\) has positive
local CKN concentration.  If the local Type I tangent branch holds, the local
Type I tangent criterion excludes it.  Otherwise the point enters the Type II branch.

Apply the local compactness dichotomy.  If compact-cylinder upper bounds for
the suitable package fail, the point is in the cascade/multibubble alternative.
If they hold, extract an ancient suitable weak limit.  If the localized
single-core gauges are nondegenerate, the local single-core Type II criterion in
[local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md)
excludes the branch.  If they are not nondegenerate, the branch is
gauge-degenerate and hence belongs to the multibubble/cascade alternative,
which is excluded by the local multibubble/cascade theorem in the same assembly
note.

These alternatives exhaust all positive local Type II concentration branches.
Therefore no Type II singularity occurs.  \(\square\)
:::

The theorem shows that the former whole-space starting assumptions have been
removed from the Type II exclusion route.  After the local Type I tangent criterion, the local
CKN bridge plus the adapted single-core and multibubble/cascade criteria exclude
all Type II branches.
