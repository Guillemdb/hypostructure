# C7: certificate composition for the windowed \(H^1\) bridge

This note composes the existing representation, modulation, pressure, and
Caccioppoli results into the certificate
\[
K_{\mathrm{WinH1}}^+.
\]

The point is not to introduce a new PDE estimate. The PDE estimate is C6/T13.
This note identifies the upstream certificate payload needed to invoke C6/T13
and records the ordered defects when one payload is unavailable.

---

## C7.1 (Windowed \(H^1\) input package)

For a declared Type II candidate, define the **windowed \(H^1\) input package**
\[
K_{\mathrm{WinH1Input}}^+
\]
to mean that the repaired-gauge orbit \((V,P,a,b)\) satisfies:

1. represented repaired-gauge orbit:
   \[
   K_{\mathrm{RepBridge}}^+;
   \]
2. bounded critical norm:
   \[
   K_{L^3\mathrm{Bd}}^+:
   \qquad
   \sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty;
   \]
3. pressure reconstruction:
   \[
   K_{\mathrm{PressureRep}}^+:
   \qquad
   -\Delta P=\partial_i\partial_j(V_iV_j)
   \quad\text{in }\mathbb R^3
   \]
   for the pressure in the represented orbit;
4. bounded modulation:
   \[
   K_{\mathrm{ModBd}}^+:
   \qquad
   \sup_{\tau\ge\tau_0}(|a(\tau)|+|b(\tau)|)<\infty;
   \]
5. Caccioppoli regularity:
   \[
   K_{\mathrm{CaccioppoliReg}}^+,
   \]
   meaning local smoothness or approximation regularity sufficient to apply
   the renormalized Caccioppoli estimate on compact cylinders.


---

## C7.1a (C3 discharges bounded critical norm)

The bounded critical norm input in C7.1 is supplied by C3. Specifically,
[cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md)
proves
\[
K_{L^3\mathrm{Norm}}^+
\Longrightarrow
K_{L^3\mathrm{Bd}}^+.
\]
Thus, on the C-series route where C3 has emitted
\(K_{L^3\mathrm{Norm}}^+\), the defect \(K_{L^3\mathrm{Bd}}^-\) is discharged
and is not an independent rough-core survivor.

---

## C7.2 (Bounded modulation certificate from T5)

Assume the repaired-gauge modulation system satisfies
\[
M(V(\tau))
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-F(V(\tau))
\]
and the two certificates
\[
K_{\mathrm{ModMatrixInv}}^+:
\quad
\sup_{\tau\ge\tau_0}\|M(V(\tau))^{-1}\|_{\infty\to\infty}<\infty,
\]
\[
K_{\mathrm{ModForceBd}}^+:
\quad
\sup_{\tau\ge\tau_0}\|F(V(\tau))\|_\infty<\infty
\]
hold. Then
\[
K_{\mathrm{ModBd}}^+
\]
holds.

### Proof

This is exactly T5 in
[cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md). Namely,
\[
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-M(V(\tau))^{-1}F(V(\tau)),
\]
so
\[
\max\{|a(\tau)|,\|b(\tau)\|_\infty\}
\le
\|M(V(\tau))^{-1}\|_{\infty\to\infty}\|F(V(\tau))\|_\infty.
\]
Taking the supremum in \(\tau\) gives bounded modulation.

\(\square\)

The forcing certificate admits the T14 reduction
\[
K_{\mathrm{TransForceBd}}^+
\wedge
K_{\mathrm{ScaleForceBd}}^+
\Longrightarrow
K_{\mathrm{ModForceBd}}^+.
\]
The translation forcing certificate is supplied by local \(L^3\), local
\(H^1\), and local pressure control for the compactly supported centering rows,
provided this local \(H^1\) input is available independently of the C7
conclusion. The weighted scale-forcing certificate is decomposed in
[cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md):
\[
K_{\mathrm{ScaleLapBd}}^+
\wedge K_{\mathrm{ScaleL4Mom}}^+
\wedge K_{\mathrm{ScalePressureBd}}^+
\Longrightarrow
K_{\mathrm{ScaleForceBd}}^+.
\]
That note discharges the nonlinear scale-row term from the weighted fourth
moment \(K_{\mathrm{ScaleL4Mom}}^+\), leaving the weighted Laplacian and pressure
rows as the pointwise scale-force analytic inputs.

The modulation-matrix inverse certificate is discharged by
[cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md):
\[
K_{\mathrm{ModMatrixPayload}}^+
\Longrightarrow
K_{\mathrm{ModMatrixInv}}^+.
\]
Here \(K_{\mathrm{ModMatrixPayload}}^+\) is the repaired-gauge nondegeneracy
package supplied by T2--T4: scale transversality, translation-block inverse,
mixed-block bounds, and a Schur-complement gap.

---

## C7.2a (Good-window modulation route)

The pointwise modulation route in C7.2 can be replaced, for good-window
arguments, by the integrated route
\[
K_{\mathrm{ModMatrixInv}}^+
\wedge
K_{\mathrm{TransForceL^1Win}}^+
\wedge
K_{\mathrm{ScaleForceL^1Win}}^+
\Longrightarrow
K_{\mathrm{ModL^1Win}}^+.
\]
This is T17.3 in
[cluster_t15_t16_integrated_modulation.md](cluster_t15_t16_integrated_modulation.md).
It implies that every sequence of unit windows with vanishing average cost
contains selected times at which the cost tends to zero and
\[
|a(\tau)|+\|b(\tau)\|_\infty
\]
is uniformly bounded.

This route does not by itself emit the pointwise certificate
\(K_{\mathrm{ModBd}}^+\). It is a good-window substitute for arguments whose
compactness step only uses selected good times. If a Caccioppoli estimate is
formulated with \(L^1\)-window modulation coefficients, then this route may be
used as an input to that estimate; otherwise C7.5 uses the pointwise
modulation route.

---

## C7.3 (Representation bridge supplies the PDE orbit hypotheses)

Assume
\[
K_{\mathrm{RepBridge}}^+.
\]
Then the candidate is represented by a repaired-gauge renormalized
Navier-Stokes orbit \((V,P,a,b)\) satisfying:

1. the repaired-gauge renormalized Navier-Stokes equation on compact
   cylinders;
2. the pressure reconstruction payload \(K_{\mathrm{PressureRep}}^+\);
3. the modulation parameters \(a,b\) appearing in the represented equation;
4. the gauge realization payload needed to define the modulation matrix.

### Proof

This is the content of the C2 representation bridge. By definition,
\(K_{\mathrm{RepBridge}}^+\) is emitted from
\[
K_{\mathrm{RawOrb}}^+,\quad
K_{\mathrm{GaugeReal}}^+,\quad
K_{\mathrm{PressureRep}}^+,\quad
K_{\mathrm{ModParams}}^+.
\]
The output object is the tuple
\[
(V,P,a,b,G_{\mathrm{sc}},G_1,G_2,G_3,\tau_0),
\]
which is precisely the PDE object used by C6/T13.

\(\square\)

---

## C7.4 (Windowed \(H^1\) certificate from the input package)

Assume
\[
K_{\mathrm{WinH1Input}}^+.
\]
Then for every \(R>0\),
\[
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty.
\]
Equivalently, the candidate emits
\[
K_{\mathrm{WinH1}}^+
\]
on the renormalized tail.

### Proof

The represented repaired-gauge orbit, pressure reconstruction, bounded
critical norm, bounded modulation, and \(K_{\mathrm{CaccioppoliReg}}^+\) are
exactly the hypotheses of T13.2 in
[cluster_c6_windowed_h1_bridge.md](cluster_c6_windowed_h1_bridge.md). T13.2
gives the windowed gradient estimate, and T13.3 adds the local \(L^2\) part to
give the displayed windowed \(H^1\) estimate for every \(R>0\).

\(\square\)

---

## C7.5 (Composed rough-core suppression bridge)

Assume a declared Type II candidate satisfies:

1. \(K_{\mathrm{RepBridge}}^+\);
2. \(K_{L^3\mathrm{Bd}}^+\);
3. \(K_{\mathrm{ModMatrixInv}}^+\);
4. \(K_{\mathrm{ModForceBd}}^+\);
5. \(K_{\mathrm{CaccioppoliReg}}^+\).

Then
\[
K_{\mathrm{WinH1}}^+
\]
holds. Consequently the rough-core alternative
\[
\exists m\ge1:
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty
\]
is excluded for this represented branch.

### Proof

By C7.2, assumptions 3 and 4 imply \(K_{\mathrm{ModBd}}^+\). By C7.3,
assumption 1 supplies the represented orbit, pressure reconstruction, and gauge
payloads. Together with assumptions 2 and 5, these are exactly
\(K_{\mathrm{WinH1Input}}^+\). C7.4 gives \(K_{\mathrm{WinH1}}^+\). The
rough-core alternative is the negation of this tail windowed \(H^1\) bound on
some bounded core, so it is excluded.

\(\square\)

---

## Ordered defects for the rough-core bridge

If \(K_{\mathrm{WinH1}}^+\) is not emitted by C7.5, the first failed input
emits one of the following ordered defects:

1. \(K_{\mathrm{RepBridge}}^-\): the repaired-gauge renormalized orbit is not
   available;
2. \(K_{L^3\mathrm{Bd}}^-\): the global critical \(L^3\) bound is not
   available. This defect is discharged on the C-series route by
   \(K_{L^3\mathrm{Norm}}^+\Rightarrow K_{L^3\mathrm{Bd}}^+\) in
   [cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md);
3. \(K_{\mathrm{ModMatrixInv}}^-\): the repaired-gauge modulation matrix is not
   uniformly invertible. This defect is discharged by
   [cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md)
   whenever the repaired-gauge nondegeneracy payload \(K_{\mathrm{ModMatrixPayload}}^+\)
   is imported;
4. \(K_{\mathrm{ModForceBd}}^-\): the modulation forcing vector is not
   uniformly bounded. When the independent local hypotheses of T14 are
   available, this defect can be refined into failure of
   \(K_{\mathrm{TransForceBd}}^+\) or failure of
   \(K_{\mathrm{ScaleForceBd}}^+\). The latter is further decomposed by
   [cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md)
   into weighted diffusion, singular-weight convective integration by parts,
   weighted fourth-moment, or weighted pressure failure;
5. \(K_{\mathrm{CaccioppoliReg}}^-\): the compact-cylinder regularity needed
   to apply Caccioppoli is not available.

For good-window arguments, the fourth defect may be replaced by the integrated
forcing defect
\[
K_{\mathrm{TransForceL^1Win}}^-
\quad\text{or}\quad
K_{\mathrm{ScaleForceL^1Win}}^-,
\]
provided the compactness step uses the T15 selected times rather than a global
pointwise modulation bound.

The scale-force defect is further decomposed by T18 into
\[
K_{\mathrm{ScaleDiffL^1Win}}^-,
\qquad
K_{\mathrm{AnnConvReg}}^-,
\qquad
K_{\mathrm{ScaleV4L^1Win}}^-,
\qquad
K_{\mathrm{ScalePressL^1Win}}^-.
\]

This converts the rough-core survivor into a finite list of concrete upstream
certificate defects.

---

## C7.6 (Rough-core blocker certificate)

Define the rough-core blocker certificate
\[
K_{\mathrm{RoughCoreBlk}}^+
\]
to be the assertion that every represented Type II branch in the declared
backend satisfies the tail windowed \(H^1\) estimate
\[
\forall R>0:\qquad
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty.
\]
Equivalently,
\[
K_{\mathrm{RoughCoreBlk}}^+
\quad\Longleftrightarrow\quad
K_{\mathrm{WinH1}}^+
\]
on represented repaired-gauge Type II branches.

If the hypotheses of C7.5 hold for every represented Type II branch in the
declared backend, then
\[
K_{\mathrm{RoughCoreBlk}}^+
\]
holds.

### Proof

For a fixed represented branch, C7.5 emits \(K_{\mathrm{WinH1}}^+\). By the
definition above, this is exactly the exclusion of the rough-core alternative
on that branch. If the hypotheses of C7.5 hold for every represented branch,
the universal rough-core blocker \(K_{\mathrm{RoughCoreBlk}}^+\) holds in the
declared backend.

\(\square\)

---

## C7.7 (A-posteriori rough-core update)

Assume a Type II candidate has reached the C5 two-bucket classification, so a
finite-cost non-suppressed candidate must emit either
\[
K_{L^3\mathrm{Tight}}^-
\qquad\text{or}\qquad
K_{\mathrm{WinH1}}^-.
\]
If the C7.5 hypotheses are subsequently supplied for that candidate, then
\[
K_{\mathrm{WinH1}}^+
\]
is emitted and the rough-core branch is removed from the candidate's survivor
classification.

If, in addition, \(K_{\mathrm{RadBlk}}^+\) is available, so that
\[
K_{L^3\mathrm{Tight}}^+
\]
holds, then the candidate cannot remain finite-cost and non-suppressed.

### Proof

C7.5 gives \(K_{\mathrm{WinH1}}^+\), which contradicts the rough-core
certificate \(K_{\mathrm{WinH1}}^-\). Thus the rough-core alternative is
removed. If \(K_{\mathrm{RadBlk}}^+\) is also available, then the radiative
alternative \(K_{L^3\mathrm{Tight}}^-\) is also removed. The C5
classification has no remaining finite-cost non-suppressed alternative for the
candidate.

\(\square\)

---

## C7.8 (Full rough-core branch exclusion theorem)

Assume the declared Type II backend satisfies the following universal payload:
for every represented Type II branch, the certificates
\[
K_{\mathrm{RepBridge}}^+,\quad
K_{L^3\mathrm{Bd}}^+,\quad
K_{\mathrm{ModMatrixInv}}^+,\quad
K_{\mathrm{ModForceBd}}^+,\quad
K_{\mathrm{CaccioppoliReg}}^+
\]
are available. Then the declared backend emits
\[
K_{\mathrm{RoughCoreBlk}}^+.
\]

Equivalently, no represented Type II branch in the declared backend can satisfy
\[
\exists m\ge1:
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty.
\]

### Proof

The universal payload is exactly the universal version of the C7.5 hypotheses.
C7.6 therefore emits \(K_{\mathrm{RoughCoreBlk}}^+\). The displayed rough-core
condition is the negation of the windowed \(H^1\) estimate defining
\(K_{\mathrm{RoughCoreBlk}}^+\), so it cannot hold for any represented branch.

\(\square\)

---

## C7 status

C7 fully implements the rough-core branch exclusion route at certificate level:
\[
K_{\mathrm{RepBridge}}^+
\wedge K_{L^3\mathrm{Bd}}^+
\wedge K_{\mathrm{ModMatrixInv}}^+
\wedge K_{\mathrm{ModForceBd}}^+
\wedge K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+
\Longleftrightarrow
K_{\mathrm{RoughCoreBlk}}^+.
\]

On the C-series route, the bounded-critical-norm obligation is already
discharged by
[cluster_c7_l3bd_defect_discharge.md](cluster_c7_l3bd_defect_discharge.md)
from \(K_{L^3\mathrm{Norm}}^+\), and the modulation-matrix inverse obligation is
discharged by
[cluster_c7_modmatrix_inverse_discharge.md](cluster_c7_modmatrix_inverse_discharge.md)
from the repaired-gauge nondegeneracy payload. After importing C2/C5
representation and that gauge payload, the remaining nontrivial rough-core
obligations are therefore
\[
K_{\mathrm{ModForceBd}}^+,\qquad
K_{\mathrm{CaccioppoliReg}}^+.
\]
Once these are discharged universally, the rough-core bucket is not an
admissible survivor for represented Type II branches in the declared backend.
