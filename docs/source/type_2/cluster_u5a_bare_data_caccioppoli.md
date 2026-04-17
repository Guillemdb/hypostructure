# U5a bare-data Caccioppoli discharge for represented suitable branches

This note strengthens S11.  S11 proved the Caccioppoli regularity certificate
from physical suitability plus repaired-gauge representation, pressure
reconstruction, and smooth gauge functions.  U5a removes the last two as
independent inputs on represented branches.

Inside the declared NS3D Type II barrier backend, C2.R gives the repaired-gauge
representation and proves pressure reconstruction and the final modulation
coefficients from the raw chart plus the AC repaired-gauge solve.  Therefore a
represented suitable branch has enough structure to justify the renormalized
local energy inequality and the Caccioppoli estimate used by C6/T13.

The output is

```{math}
K_{\mathrm{CaccioppoliReg}}^+.
```

## Scope of the certificate

In U5a, \(K_{\mathrm{CaccioppoliReg}}^+\) means the **local** renormalized
Caccioppoli estimate on each compact repaired-gauge cylinder.  The estimate's
constant may depend on that cylinder, the cutoff, and the local upper/lower
bounds for the AC chart.  It is not the uniform windowed \(H^1\) certificate
\(K_{\mathrm{WinH1}}^+\), and it carries no local CKN, tightness,
bounded-modulation, finite-energy-at-infinity, or global compactness content.
Those are separate C6/C7 inputs.

## Bare-data input

::::{prf:definition} Represented suitable Type II branch
:label: def-u5a-represented-suitable-branch

A **represented suitable Type II branch** is a routed candidate \(\omega\) in
the declared NS3D Type II barrier backend such that:

1. \(K_{\mathrm{TypeIIRoute}}^+(\omega)\) holds;
2. \(K_{\mathrm{RepBridge}}^+(\omega)\) holds through C2.R, so the branch has
   an AC raw chart, an AC repaired-gauge solve, pressure reconstruction, and
   final modulation coefficients;
3. the physical branch \((u,p)\) is a local suitable weak solution on the
   terminal interval.  Equivalently, it emits
   \(K_{\mathrm{PhysSuitable}}^+\):
   \[
   u\in L^\infty_tL^2_{\mathrm{loc}}
   \cap L^2_tH^1_{\mathrm{loc}},
   \qquad
   p\in L^{3/2}_{\mathrm{loc}},
   \qquad
   \nabla\cdot u=0,
   \]
   the Navier-Stokes equations hold distributionally, and the physical local
   energy inequality holds for every nonnegative smooth compactly supported
   test function.

::::

No separate \(K_{\mathrm{PressureRep}}^+\), \(K_{\mathrm{GaugeReg}}^+\), or
\(C^1\)-gauge input is part of U5a.  Pressure reconstruction is a C2.R theorem,
and AC gauge regularity is enough for the pullback of the local energy
inequality.

## AC pullback of the local energy inequality

::::{prf:lemma} Local energy inequality is stable under AC Type II charts
:label: lem-u5a-ac-pullback-lei

Let \((u,p)\) be physically suitable on a compact physical cylinder, and let
\((V,P)\) be obtained from an AC represented chart

```{math}
u(x,t)=\Lambda(s)^{-1}V(Y,s),
\qquad
p(x,t)=\Lambda(s)^{-2}P(Y,s)+c(t),
\qquad
Y=\frac{x-X(s)}{\Lambda(s)},
```

with \(s=\tau\), \(dt=\Lambda(s)^2\,ds\), \(\Lambda>0\), and
\(X,\Lambda\in W^{1,1}_{\mathrm{loc}}\).  Then \((V,P)\) satisfies the
renormalized local energy inequality on compact \((Y,s)\)-cylinders:

```{math}
\begin{aligned}
&\int \frac{|V(s_2)|^2}{2}\phi(s_2)
+\nu\int_{s_1}^{s_2}\int |\nabla V|^2\phi \\
&\le
\int \frac{|V(s_1)|^2}{2}\phi(s_1)
+\int_{s_1}^{s_2}\int
\frac{|V|^2}{2}(\partial_s\phi+\nu\Delta\phi) \\
&\quad
+\int_{s_1}^{s_2}\int
\left(\frac{|V|^2}{2}+P\right)V\cdot\nabla\phi \\
&\quad
+\int_{s_1}^{s_2}\int
 a(s)\frac{|V|^2}{2}(-\phi-Y\cdot\nabla\phi)
-\int_{s_1}^{s_2}\int
 \frac{|V|^2}{2}b(s)\cdot\nabla\phi,
\end{aligned}
```

for every nonnegative \(\phi\in C_c^\infty(\mathbb R^3\times(s_1,s_2))\).  Here
\(a,b\) are the final modulation coefficients supplied by C2.R.

::::

:::{prf:proof}
First suppose \(X,\Lambda\) are \(C^1\).  Use the physical test function
\[
\psi(x,t)=\phi\left(\frac{x-X(s)}{\Lambda(s)},s\right),
\qquad t=t(s),
\]
where \(dt=\Lambda^2ds\).  The support of \(\phi\) is compact and
\(\Lambda\) is bounded above and below on the selected compact time interval,
so \(\psi\) is an admissible compactly supported test function in the physical
local energy inequality.  Changing variables
\(x=X(s)+\Lambda(s)Y\), \(dt=\Lambda^2ds\), and
\(u=\Lambda^{-1}V\), \(p=\Lambda^{-2}P+c(t)\), gives the displayed
renormalized inequality.  The pressure constant drops out because
\[
\int c(t)V\cdot\nabla\phi=-\int c(t)\phi\,\nabla\cdot V=0.
\]
The terms involving \(\Lambda_s\) and \(X_s\) are exactly the scale and
translation modulation terms in the represented equation; C2.R identifies their
coefficients as \(a,b\).

For \(X,\Lambda\in W^{1,1}_{\mathrm{loc}}\), do not approximate the
solution or invoke any global compactness.  On the selected compact time
interval, \(\Lambda\) has positive lower and finite upper bounds.  The pulled
back test
\[
\psi(x,t)=\phi((x-X(s))/\Lambda(s),s)
\]
is compactly supported, nonnegative, bounded, has bounded spatial derivatives
on the compact cylinder, and has time derivative in \(L^1_tL^\infty_x\)
because \(X',\Lambda'\in L^1\) and \(\phi\) is smooth compactly supported.
The physical local energy inequality, initially stated for
\(C_c^\infty\) tests, extends to this class by standard mollification of the
test function: the suitability classes give exactly the local integrability
needed for every term in the inequality, and the mollified tests keep support
inside a slightly larger compact cylinder.  Applying the extended inequality to
\(\psi\) and changing variables gives the displayed renormalized inequality.
The only derivatives of the chart that appear are the \(L^1\)-functions
\(X',\Lambda'\), hence the modulation terms are locally integrable against
local \(|V|^2\).  \(\square\)
:::

## Bare-data pressure and modulation closure

::::{prf:lemma} Represented branches have the pressure and modulation needed for Caccioppoli
:label: lem-u5a-pressure-modulation-automatic

For a represented suitable branch in the declared backend, the pressure
identity

```{math}
-\Delta P=\partial_i\partial_j(V_iV_j)
```

holds modulo functions of \(\tau\), and the final coefficients \(a,b\) in the
renormalized equation are measurable and locally integrable on compact
renormalized windows.

::::

:::{prf:proof}
This is exactly the pressure and modulation part of C2.R.  Pressure pullback
comes from the physical pressure equation and the chart/gauge scaling.  The
final coefficients are forced by the AC scale/translation gauge transform:
\[
a=\mu^2A+\frac{\mu_\tau}{\mu},
\qquad
b=\mu B+\mu A q+\frac{q_\tau}{\mu}.
\]
Since the chart and gauge parameters are AC on compact windows, these
coefficients are locally integrable.  \(\square\)
:::

## Caccioppoli regularity discharge

::::{prf:theorem} U5a bare-data Caccioppoli theorem
:label: thm-u5a-bare-data-caccioppoli

Every represented suitable NS3D Type II branch in the declared backend emits

```{math}
K_{\mathrm{CaccioppoliReg}}^+.
```

Equivalently,

```{math}
K_{\mathrm{PhysSuitable}}^+
\wedge
K_{\mathrm{TypeIIRoute}}^+
\wedge
K_{\mathrm{NS3DTypeIIBackend}}^+
\Longrightarrow
K_{\mathrm{CaccioppoliReg}}^+.
```

::::

:::{prf:proof}
By the declared backend and C2.R, the routed branch has the AC represented
chart, the admissible AC repaired-gauge solve, pressure reconstruction, and the
final modulation coefficients.  Lemma
{prf:ref}`lem-u5a-ac-pullback-lei` pulls the physical local energy inequality
to the repaired variables.  Lemma
{prf:ref}`lem-u5a-pressure-modulation-automatic` supplies the pressure and
coefficient identities needed to interpret the renormalized inequality in the
same PDE class used by C6/T13.

Choose a standard compactly supported Caccioppoli cutoff
\(\phi(Y,\tau)=\eta(\tau)\zeta(Y)^2\).  Inserting it into the renormalized
local energy inequality gives the Caccioppoli estimate on every compact
renormalized cylinder, with constants allowed to depend on that compact
cylinder and the chosen cutoff.

No global estimate is used in this step.  On the compact cylinder under
consideration, the AC chart has
\(0<\Lambda_-\leq \Lambda(\tau)\leq \Lambda_+<\infty\).  Pulling back
physical suitability therefore gives, for every compact spatial set \(K\),

```{math}
V\in L^\infty_\tau L^2_Y(K)\cap L^2_\tau H^1_Y(K),
\qquad
P\in L^{3/2}_{\tau,Y}(K).
```

Local Sobolev interpolation on the finite cylinder gives
\(V\in L^{10/3}_{\tau,Y}(K)\), hence \(V\in L^3_{\tau,Y}(K)\).  Thus
\(PV\cdot\nabla\phi\) is integrable by Holder.  C2.R gives
\(a,b\in L^1_{\mathrm{loc}}(d\tau)\), and
\(\tau\mapsto\int_K |V(Y,\tau)|^2\,dY\) is locally essentially bounded;
therefore the scale and translation modulation terms in the pulled-back local
energy inequality are integrable on the same compact cylinder.  This proves
exactly the local certificate \(K_{\mathrm{CaccioppoliReg}}^+\), not a global
windowed \(H^1\) or critical-norm bound.  \(\square\)
:::

## Consequence for the rough-core bridge

::::{prf:corollary} U5a removes the Caccioppoli defect from represented suitable branches
:label: cor-u5a-removes-caccioppoli-defect

On represented suitable branches in the declared backend,
\(K_{\mathrm{CaccioppoliReg}}^-\), \(K_{\mathrm{PressureRep}}^-\), and
\(K_{\mathrm{GaugeReg}}^-\) are not independent rough-core defects.  The
remaining inputs for the C6/C7 windowed \(H^1\) route are the separate bounded
critical-norm and modulation bounds required by that route.

::::

:::{prf:proof}
Theorem {prf:ref}`thm-u5a-bare-data-caccioppoli` emits
\(K_{\mathrm{CaccioppoliReg}}^+\).  C2.R emits pressure reconstruction and the
final modulation coefficients from the represented chart and AC gauge solve.
Thus the listed negative certificates cannot occur on represented suitable
branches inside the declared backend.  C6/C7 may still require bounded
critical norm and bounded modulation to upgrade Caccioppoli regularity to the
uniform windowed \(H^1\) certificate.  \(\square\)
:::

## Status

U5a is discharged at the certificate level.  It proves the bare-data
Caccioppoli theorem for represented suitable branches using only compact-cylinder
physical suitability and the local AC represented chart supplied by C2.R.  It
does not assume or prove bounded modulation, whole-space critical control,
whole-space tightness, finite energy at infinity, or any whole-space compactness
input.
Those remain separate inputs in the C6/C7 routes that upgrade local
Caccioppoli regularity to \(K_{\mathrm{WinH1}}^+\).
