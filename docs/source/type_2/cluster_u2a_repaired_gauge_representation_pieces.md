# U2a: automatic repaired-gauge representation pieces

This note isolates the representation components that are automatic once an
absolutely continuous Navier-Stokes concentration chart and an absolutely
continuous repaired-gauge path are present.  It does not prove extraction of
the chart from arbitrary bare data and it does not prove existence of the
repaired gauge root.  Those are the genuine remaining U2 inputs.

The output is the certificate
\[
K_{\mathrm{RepAuto},NS3D}^+.
\]
It consists of the final physical chart identity, the repaired-gauge
renormalized Navier-Stokes equation, pressure reconstruction modulo constants,
modulation-coefficient realization, critical \(L^3\)-norm invariance, and the
basic AC gauge-regularity inheritance used by C2.R, U5a, S13, and S14.

---

## U2a.1 (Raw AC chart)

Let \(u,p\) solve NS3D on a terminal interval \((t_0,T^*)\):
\[
\partial_tu+(u\cdot\nabla)u+\nabla p=\nu\Delta u,
\qquad
\nabla\cdot u=0.
\]
The raw absolutely continuous chart certificate
\[
K_{\mathrm{ACChart},NS3D}^+
\]
means that there are absolutely continuous functions
\[
x_c:(t_0,T^*)\to\mathbb R^3,\qquad
\lambda:(t_0,T^*)\to(0,\infty),
\]
and a strictly increasing absolutely continuous raw time \(\rho\) satisfying
\[
\rho_t=\lambda^{-2}\quad\text{a.e.},
\]
such that
\[
y=\frac{x-x_c(t)}{\lambda(t)},
\qquad
u(x,t)=\lambda(t)^{-1}U(y,\rho(t)),
\]
and
\[
p(x,t)=\lambda(t)^{-2}Q(y,\rho(t))+c(t)
\]
for some scalar function \(c(t)\).  The chart is assumed to have enough local
integrability for the distributional chain rule on compact \(y\)-sets.  This
is the analytic content of the chart part of the C2.R input
\[
K_{\mathrm{Chart},NS3D}^+.
\]

Define the raw modulation coefficients
\[
A(\rho(t)):=\lambda(t)\lambda_t(t),
\qquad
B(\rho(t)):=\lambda(t)x_c'(t).
\]
Equivalently,
\[
A=\partial_\rho\log\lambda,
\qquad
B=\lambda^{-1}\partial_\rho x_c,
\]
where \(\lambda,x_c\) are regarded as functions of \(\rho\).

### Lemma U2a.1

Assume \(K_{\mathrm{ACChart},NS3D}^+\).  Then \(U,Q\) satisfy
\[
\partial_\rho U+(U\cdot\nabla)U+\nabla Q
=
\nu\Delta U+A(\rho)(U+y\cdot\nabla U)+B(\rho)\cdot\nabla U,
\qquad
\nabla\cdot U=0
\]
distributionally on compact \(y\)-sets.

#### Proof

For smooth functions the computation is pointwise.  The distributional case
follows by testing against compactly supported functions and using the assumed
chart chain rule.

The spatial derivatives are
\[
\nabla_xu=\lambda^{-2}\nabla_yU,\qquad
\Delta_xu=\lambda^{-3}\Delta_yU,
\qquad
(u\cdot\nabla_x)u=\lambda^{-3}(U\cdot\nabla_y)U.
\]
Since
\[
y_t=-\frac{x_c'}{\lambda}-\frac{\lambda_t}{\lambda}y,
\qquad
\rho_t=\lambda^{-2},
\]
one has
\[
\partial_tu
=
\lambda^{-3}\partial_\rho U
-\lambda^{-2}\lambda_t(U+y\cdot\nabla U)
-\lambda^{-2}x_c'\cdot\nabla U.
\]
Also
\[
\nabla_xp=\lambda^{-3}\nabla_yQ.
\]
Substitution into the physical Navier-Stokes equation and multiplication by
\(\lambda^3\) gives
\[
\partial_\rho U
-\lambda\lambda_t(U+y\cdot\nabla U)
-\lambda x_c'\cdot\nabla U
+(U\cdot\nabla)U+\nabla Q
=
\nu\Delta U.
\]
Moving the two chart-velocity terms to the right gives the displayed equation.
Finally,
\[
\nabla_x\cdot u=\lambda^{-2}\nabla_y\cdot U,
\]
so \(\nabla_y\cdot U=0\).

\(\square\)

---

## U2a.2 (AC repaired-gauge path)

The AC repaired-gauge path certificate
\[
K_{\mathrm{ACGaugePath},NS3D}^+
\]
means that, on every compact final time interval, there are absolutely
continuous functions
\[
\mu(\tau)>0,\qquad q(\tau)\in\mathbb R^3,\qquad \rho=\rho(\tau),
\]
such that
\[
\rho_\tau=\mu(\tau)^2\quad\text{a.e.},
\]
and
\[
V(Y,\tau)=\mu(\tau)U(\mu(\tau)Y+q(\tau),\rho(\tau)),
\]
\[
P(Y,\tau)=\mu(\tau)^2Q(\mu(\tau)Y+q(\tau),\rho(\tau)).
\]
The gauge equations are
\[
G_{\mathrm{sc}}(V(\tau))=0,
\qquad
G_j(V(\tau))=0,\quad j=1,2,3,
\]
with the repaired scale functional and centering functionals used throughout
the Type II stack.

This certificate is weaker than the full repaired-gauge solve.  It assumes the
AC path is already selected.  The root-existence and AC-selection problem is
kept as a genuine remaining representation payload.

### Lemma U2a.2

Assume \(K_{\mathrm{ACChart},NS3D}^+\) and
\(K_{\mathrm{ACGaugePath},NS3D}^+\).  Let \(t=t(\rho(\tau))\), and define
\[
\Lambda(\tau):=\lambda(t(\rho(\tau)))\mu(\tau),
\]
\[
X(\tau):=x_c(t(\rho(\tau)))+\lambda(t(\rho(\tau)))q(\tau).
\]
Then, on every compact final time interval, \(\Lambda\) and \(X\) are
absolutely continuous, \(\Lambda>0\), and
\[
\frac{d\tau}{dt}=\Lambda(\tau)^{-2}.
\]
Moreover,
\[
x=X(\tau)+\Lambda(\tau)Y
\]
is exactly the final physical chart and
\[
u(x,t)=\Lambda(\tau)^{-1}V(Y,\tau),
\qquad
p(x,t)=\Lambda(\tau)^{-2}P(Y,\tau)+c(t).
\]

#### Proof

The raw time \(\rho\) is strictly increasing because
\(\rho_t=\lambda^{-2}>0\) a.e.; hence it has an absolutely continuous inverse
on compact subintervals of its range.  The compositions
\(\lambda(t(\rho(\tau)))\) and \(x_c(t(\rho(\tau)))\) are therefore
absolutely continuous on compact final windows.  Products and sums with the
AC functions \(\mu,q\) are AC, so \(\Lambda\) and \(X\) are AC.

The identity \(\rho_\tau=\mu^2\) implies
\[
\frac{d\tau}{d\rho}=\mu^{-2}.
\]
Since \(d\rho/dt=\lambda^{-2}\),
\[
\frac{d\tau}{dt}
=
\frac{d\tau}{d\rho}\frac{d\rho}{dt}
=
\mu^{-2}\lambda^{-2}
=
(\lambda\mu)^{-2}
=
\Lambda^{-2}.
\]
If \(x=X+\Lambda Y\), then
\[
x=x_c+\lambda q+\lambda\mu Y,
\]
and hence the raw coordinate is
\[
y=\frac{x-x_c}{\lambda}=q+\mu Y.
\]
Therefore
\[
u(x,t)
=\lambda^{-1}U(q+\mu Y,\rho)
=\lambda^{-1}\mu^{-1}V(Y,\tau)
=\Lambda^{-1}V(Y,\tau),
\]
and similarly
\[
p(x,t)
=\lambda^{-2}Q(q+\mu Y,\rho)+c(t)
=\lambda^{-2}\mu^{-2}P(Y,\tau)+c(t)
=\Lambda^{-2}P(Y,\tau)+c(t).
\]

\(\square\)

---

## U2a.3 (Final modulation coefficients)

Define
\[
a(\tau):=\partial_\tau\log\Lambda(\tau),
\qquad
b(\tau):=\Lambda(\tau)^{-1}X_\tau(\tau)
\]
where derivatives are understood a.e. on compact final windows.

### Lemma U2a.3

Under the hypotheses of Lemma U2a.2,
\[
a(\tau)=\mu(\tau)^2A(\rho(\tau))+\frac{\mu_\tau(\tau)}{\mu(\tau)}
\]
and
\[
b(\tau)=\mu(\tau)B(\rho(\tau))
+\mu(\tau)A(\rho(\tau))q(\tau)
+\frac{q_\tau(\tau)}{\mu(\tau)}.
\]
In particular \(a,b\in L^1_{\mathrm{loc}}\) on every compact final time
interval.

#### Proof

Since \(\Lambda=\lambda(\rho(\tau))\mu(\tau)\),
\[
\partial_\tau\log\Lambda
=
\rho_\tau\partial_\rho\log\lambda+\partial_\tau\log\mu
=
\mu^2A+\frac{\mu_\tau}{\mu}.
\]
This proves the formula for \(a\).

Next
\[
X=x_c(\rho(\tau))+\lambda(\rho(\tau))q(\tau).
\]
Using
\[
\partial_\rho x_c=\lambda B,\qquad
\partial_\rho\lambda=\lambda A,\qquad
\rho_\tau=\mu^2,
\]
we get
\[
X_\tau
=
\mu^2\lambda B+\mu^2\lambda A q+\lambda q_\tau.
\]
Division by \(\Lambda=\lambda\mu\) gives
\[
\Lambda^{-1}X_\tau
=
\mu B+\mu Aq+\frac{q_\tau}{\mu}.
\]

Local integrability follows from AC regularity and positivity of \(\mu\).
Indeed, on every compact final interval, \(\mu\) is bounded above and below
away from zero, \(q\) is bounded, and \(\mu_\tau,q_\tau\in L^1\).  The terms
with \(A,B\) are locally integrable because
\[
\int |A(\rho(\tau))|\mu(\tau)^2\,d\tau
=
\int |A(\rho)|\,d\rho,
\]
and the remaining factors \(\mu^{-1}\), \(\mu\), and \(q\) are bounded on the
same compact window.

\(\square\)

### Lemma U2a.4

The final variables \(V,P\) satisfy
\[
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V+a(\tau)(V+Y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0
\]
distributionally on compact \(Y\)-sets.

#### Proof

This follows either by applying Lemma U2a.1 to the final physical chart of
Lemma U2a.2, or by direct transformation of the raw \(U,Q\) equation.
For the direct computation, set
\[
z=\mu Y+q.
\]
Then
\[
\nabla_YV=\mu^2\nabla_zU,\qquad
\Delta_YV=\mu^3\Delta_zU,
\]
\[
(V\cdot\nabla_Y)V=\mu^3(U\cdot\nabla_z)U,
\qquad
\nabla_YP=\mu^3\nabla_zQ.
\]
Differentiation in \(\tau\), using \(\rho_\tau=\mu^2\), gives
\[
\begin{aligned}
V_\tau
&=
\mu_\tau U
+\mu\mu^2U_\rho
+\mu(\mu_\tau Y+q_\tau)\cdot\nabla_zU .
\end{aligned}
\]
Insert the raw equation for \(U_\rho\), multiply the raw equation terms by
\(\mu^3\), and rewrite every occurrence of \(U\) and \(\nabla_zU\) in terms
of \(V\) and \(\nabla_YV\).  The coefficient multiplying
\(V+Y\cdot\nabla_YV\) is
\[
\frac{\mu_\tau}{\mu}+\mu^2A,
\]
and the coefficient multiplying \(\nabla_YV\) is
\[
\mu B+\mu Aq+\frac{q_\tau}{\mu}.
\]
By Lemma U2a.3 these are exactly \(a\) and \(b\).  Incompressibility follows
from
\[
\nabla_Y\cdot V=\mu^2\nabla_z\cdot U=0.
\]

\(\square\)

---

## U2a.4 (Pressure reconstruction)

### Lemma U2a.5

Assume the physical pressure satisfies
\[
-\Delta_xp=\partial_i\partial_j(u_iu_j)
\]
in distributions.  Then the raw pressure satisfies
\[
-\Delta_yQ=\partial_i\partial_j(U_iU_j)
\]
modulo functions of \(\rho\), and the final pressure satisfies
\[
-\Delta_YP=\partial_i\partial_j(V_iV_j)
\]
modulo functions of \(\tau\).

#### Proof

The additive function \(c(t)\) in
\[
p=\lambda^{-2}Q+c(t)
\]
has zero spatial derivatives.  Since
\[
\Delta_xp=\lambda^{-4}\Delta_yQ,
\qquad
\partial_{x_i}\partial_{x_j}(u_iu_j)
=
\lambda^{-4}\partial_{y_i}\partial_{y_j}(U_iU_j),
\]
the raw pressure equation follows.

For the repaired variables,
\[
P(Y)=\mu^2Q(\mu Y+q),
\qquad
V(Y)=\mu U(\mu Y+q).
\]
Therefore
\[
-\Delta_YP
=
\mu^4(-\Delta_zQ)(z)
=
\mu^4\partial_{z_i}\partial_{z_j}(U_iU_j)(z)
=
\partial_{Y_i}\partial_{Y_j}(V_iV_j)(Y).
\]
Adding functions of time to \(Q\) or \(P\) does not change the identity.

\(\square\)

---

## U2a.5 (Critical norm and local norm transformation)

### Lemma U2a.6

For every \(1\le r<\infty\) for which the norms are finite,
\[
\|V(\tau)\|_{L^r_Y}^r
=
\Lambda(\tau)^{r-3}\|u(t(\tau))\|_{L^r_x}^r.
\]
In particular,
\[
\|V(\tau)\|_{L^3_Y}=\|u(t(\tau))\|_{L^3_x}.
\]
Thus \(L^3\)-domain well-definedness, finiteness, infiniteness, and positivity
are invariant under the represented chart.

#### Proof

Using
\[
u(X+\Lambda Y,t)=\Lambda^{-1}V(Y,\tau),
\qquad
dx=\Lambda^3\,dY,
\]
we obtain
\[
\|u(t)\|_{L^r_x}^r
=
\int |\Lambda^{-1}V(Y,\tau)|^r\Lambda^3\,dY
=
\Lambda^{3-r}\|V(\tau)\|_{L^r_Y}^r.
\]
Rearranging gives the stated formula.  The case \(r=3\) is exactly
scale-invariant.

\(\square\)

---

## U2a.6 (AC gauge regularity inheritance)

### Lemma U2a.7

If \(\lambda,x_c,\mu,q\) are AC and \(\mu>0\), then on compact final time
intervals the final scale \(\Lambda\), center \(X\), and coefficients \(a,b\)
are locally integrable.  This emits the AC gauge-regularity certificate
\[
K_{\mathrm{GaugeReg},AC}^+.
\]

#### Proof

This follows from Lemmas U2a.2 and U2a.3.  Positivity of \(\mu\) and
compactness of the time window give a positive lower bound for \(\mu\), so
the divisions by \(\mu\) in the formula for \(a,b\) are legitimate.

\(\square\)

---

## U2a.7 (Automatic representation-pieces theorem)

### Theorem U2a.8

Assume
\[
K_{\mathrm{ACChart},NS3D}^+
\wedge
K_{\mathrm{ACGaugePath},NS3D}^+.
\]
Then
\[
K_{\mathrm{RepAuto},NS3D}^+
\]
holds.  More explicitly, the branch emits:
\[
K_{\mathrm{RawOrb}}^+,\quad
K_{\mathrm{PressureRep}}^+,\quad
K_{\mathrm{ModParams}}^+,\quad
K_{\mathrm{GaugeReg},AC}^+,
\]
It also emits critical \(L^3\)-norm invariance along the represented orbit.

#### Proof

Lemma U2a.1 gives the raw renormalized equation and hence the raw orbit
identity.  Lemmas U2a.2--U2a.4 give the final chart and the final
repaired-gauge equation with the correct coefficients.  Lemma U2a.5 gives
pressure reconstruction modulo constants.  Lemma U2a.6 gives the critical
norm invariance.  Lemma U2a.7 gives the gauge-regularity inheritance.  These
are exactly the automatic representation pieces collected in
\(K_{\mathrm{RepAuto},NS3D}^+\).

\(\square\)

### Corollary U2a.9

After C2.R and Theorem U2a.8, the following are not independent
representation defects:
\[
K_{\mathrm{PressureRep}}^-,
\qquad
K_{\mathrm{ModCoeff}}^-,
\qquad
K_{\mathrm{GaugeReg},AC}^-,
\qquad
K_{L^3\mathrm{ChartInv}}^-.
\]
If any of them appears, it means the assumed AC chart or AC gauge path was not
actually available with the stated regularity.

The exact remaining U2 representation payloads on the current bare-data route
are:

1. \(K_{\mathrm{ChartExtract},NS3D}^+\): extraction of an AC concentration
   chart from the retained Type II branch.
2. \(K_{\mathrm{GaugeRoot},NS3D}^+\): solvability of the repaired scale and
   centering equations on the admissible profile class.
3. \(K_{\mathrm{GaugeACSel},NS3D}^+\): selection of the gauge roots as an AC
   path satisfying \(\rho_\tau=\mu^2\).
4. \(K_{\mathrm{GaugeTerm},NS3D}^+\): terminal admissibility of the final
   scale and time, including \(\tau\to\infty\) and preservation of the Type II
   core scale.

Thus the repaired-gauge representation row has been reduced to chart
extraction and the genuine repaired-gauge solve.  Pressure pullback,
modulation formulas, AC regularity, and critical \(L^3\)-invariance are
theorems after those inputs are supplied.

#### Proof

The first statement is immediate from Theorem U2a.8.  The remaining list is
the complementary part of the representation construction: before Theorem
U2a.8 can be applied, one must still produce the raw chart and an admissible
AC repaired-gauge path.  Producing such a path requires pointwise roots of the
gauge equations, an AC selection of those roots, and terminal admissibility of
the resulting scale-time chart.

\(\square\)
