# T18: scale-force decomposition

This note decomposes the integrated weighted scale-force payload into
diffusive, convective, and pressure contributions. It proves the convective
piece from annular convective regularity plus a weighted \(L^4\)-window
certificate and isolates the remaining diffusion and pressure scale-force
defects.

Let
\[
w(y):=|y|^{-p},\qquad 0<p<3,
\]
and
\[
H(V,P):=\nu\Delta V-(V\cdot\nabla)V-\nabla P.
\]
The repaired scale forcing is
\[
F_0(V)
:=
3\int_{\mathbb R^3}w|V|V\cdot H(V,P)\,dy.
\]

All weighted pairings below are understood in the same sense as
\(K_{\mathrm{ScaleForceL^1Win}}^+\): either as absolutely integrable
functions of \(\tau\), or as specified distributional pairings represented by
\(L^1_{\mathrm{loc}}(d\tau)\) functions. In the distributional case, the
three pairings are required to be compatible with the decomposition of
\(H(V,P)\) into diffusion, transport, and pressure.

---

## T18.1 (Scale-force decomposition)

Define
\[
F_0^\Delta(V)
:=
3\nu\int_{\mathbb R^3}w|V|V\cdot\Delta V\,dy,
\]
\[
F_0^{\mathrm{nl}}(V)
:=
-3\int_{\mathbb R^3}w|V|V\cdot (V\cdot\nabla V)\,dy,
\]
and
\[
F_0^P(V,P)
:=
-3\int_{\mathbb R^3}w|V|V\cdot\nabla P\,dy.
\]
Then, for every \(\tau\) at which all three pairings are defined,
\[
F_0(V(\tau))
=
F_0^\Delta(V(\tau))
+F_0^{\mathrm{nl}}(V(\tau))
+F_0^P(V(\tau),P(\tau)).
\]

### Proof

Substitute
\[
H(V,P)=\nu\Delta V-(V\cdot\nabla)V-\nabla P
\]
into
\[
F_0(V)=3\int w|V|V\cdot H(V,P).
\]
Linearity of the pairing gives exactly the three displayed terms.

\(\square\)

---

## T18.2 (Subpayloads imply integrated scale-force control)

Define the three subpayloads
\[
K_{\mathrm{ScaleDiffL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}|F_0^\Delta(V(\tau))|\,d\tau<\infty,
\]
\[
K_{\mathrm{ScaleConvL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}|F_0^{\mathrm{nl}}(V(\tau))|\,d\tau<\infty,
\]
and
\[
K_{\mathrm{ScalePressL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}|F_0^P(V(\tau),P(\tau))|\,d\tau<\infty.
\]
Then
\[
K_{\mathrm{ScaleDiffL^1Win}}^+
\wedge
K_{\mathrm{ScaleConvL^1Win}}^+
\wedge
K_{\mathrm{ScalePressL^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleForceL^1Win}}^+.
\]

### Proof

By T18.1,
\[
|F_0(V(\tau))|
\le
|F_0^\Delta(V(\tau))|
+|F_0^{\mathrm{nl}}(V(\tau))|
+|F_0^P(V(\tau),P(\tau))|
\]
for almost every \(\tau\). Integrating on \([T,T+1]\), taking the supremum in
\(T\), and applying the three subpayload bounds gives
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}|F_0(V(\tau))|\,d\tau<\infty.
\]
This is \(K_{\mathrm{ScaleForceL^1Win}}^+\).

\(\square\)

---

## T18.3 (Convective scale-force bound from weighted \(L^4\) and IBP)

Define the singular-weight convective integration-by-parts certificate
\[
K_{\mathrm{ScaleConvIBP}}^+
\]
to mean that, for almost every \(\tau\), the identity
\[
-\int_{\mathbb R^3}w\,V\cdot\nabla(|V|^3)\,dy
=
\int_{\mathbb R^3}|V|^3\,V\cdot\nabla w\,dy
\]
holds with no boundary contribution from \(y=0\) or \(|y|=\infty\), either
classically or by convergence of smooth compactly supported divergence-free
approximants.

Define
\[
K_{\mathrm{ScaleV4L^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}\int_{\mathbb R^3}
|y|^{-p-1}|V(y,\tau)|^4\,dy\,d\tau
<\infty.
\]
Then
\[
K_{\mathrm{ScaleConvIBP}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleConvL^1Win}}^+.
\]
More precisely,
\[
|F_0^{\mathrm{nl}}(V(\tau))|
\le
p\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^4\,dy
\]
for almost every \(\tau\).

### Proof

For fixed \(\tau\), suppress \(\tau\) from the notation. Since
\[
\nabla(|V|^3)=3|V|(V\cdot\nabla)V
\]
componentwise in the sense
\[
V\cdot\nabla(|V|^3)
=
3|V|\,V_iV_j\partial_iV_j,
\]
one has
\[
w|V|V\cdot(V\cdot\nabla V)
=
\frac13 w\,V\cdot\nabla(|V|^3).
\]
Therefore
\[
F_0^{\mathrm{nl}}(V)
=
-3\int w|V|V\cdot(V\cdot\nabla V)
=
-\int w\,V\cdot\nabla(|V|^3).
\]
Using \(\nabla\cdot V=0\) and \(K_{\mathrm{ScaleConvIBP}}^+\),
\[
-\int w\,V\cdot\nabla(|V|^3)
=
\int |V|^3\,V\cdot\nabla w.
\]
Since
\[
\nabla w(y)=-p|y|^{-p-2}y
\]
for \(y\ne0\),
\[
|V\cdot\nabla w|
\le
p|y|^{-p-1}|V|.
\]
Thus
\[
|F_0^{\mathrm{nl}}(V)|
\le
p\int |y|^{-p-1}|V|^4\,dy.
\]
Integrating over \([T,T+1]\) and taking the supremum in \(T\) proves
\(K_{\mathrm{ScaleConvL^1Win}}^+\).

\(\square\)

---

## T18.4 (Annular regularity discharges singular-weight IBP)

Define the annular convective regularity certificate
\[
K_{\mathrm{AnnConvReg}}^+
\]
to mean that, for almost every \(\tau\), \(V(\tau)\) is divergence free and
the identity
\[
V\cdot\nabla(|V|^3)=3|V|\,V_iV_j\partial_iV_j
\]
holds on every compact annulus \(0<r<|y|<R<\infty\), either classically or as
the limit of smooth divergence-free approximants in \(L^4_{\mathrm{loc}}\) on
annuli.

Then
\[
K_{\mathrm{AnnConvReg}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleConvIBP}}^+.
\]

### Proof

Fix a time \(\tau\) for which the weighted \(L^4\) density
\[
\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^4\,dy
\]
is finite and \(K_{\mathrm{AnnConvReg}}^+\) holds. Such times form a
full-measure set on every unit window.

Let \(\eta\in C^\infty([0,\infty))\) satisfy \(0\le\eta\le1\),
\(\eta(s)=0\) for \(s\le1\), and \(\eta(s)=1\) for \(s\ge2\). Let
\[
\chi_{\varepsilon,R}(y):=\eta(|y|/\varepsilon)\eta(R/|y|).
\]
Then \(\chi_{\varepsilon,R}\) is supported in
\(\{\varepsilon<|y|<R\}\), equals one on
\(\{2\varepsilon<|y|<R/2\}\), and satisfies
\[
|\nabla\chi_{\varepsilon,R}(y)|
\le
C\varepsilon^{-1}\mathbf 1_{\{\varepsilon<|y|<2\varepsilon\}}
+CR^{-1}\mathbf 1_{\{R/2<|y|<R\}}.
\]
By \(K_{\mathrm{AnnConvReg}}^+\), integration by parts on the annulus gives
\[
-\int w\chi_{\varepsilon,R}V\cdot\nabla(|V|^3)
=
\int |V|^3V\cdot\nabla(w\chi_{\varepsilon,R}).
\]
Expanding the derivative,
\[
-\int w\chi_{\varepsilon,R}V\cdot\nabla(|V|^3)
=
\int \chi_{\varepsilon,R}|V|^3V\cdot\nabla w
+\int w|V|^3V\cdot\nabla\chi_{\varepsilon,R}.
\]
The cutoff-error term satisfies
\[
\left|
\int w|V|^3V\cdot\nabla\chi_{\varepsilon,R}
\right|
\le
C\int_{\{\varepsilon<|y|<2\varepsilon\}}
|y|^{-p-1}|V|^4\,dy
+C\int_{\{R/2<|y|<R\}}
|y|^{-p-1}|V|^4\,dy.
\]
The first term tends to zero as \(\varepsilon\downarrow0\), and the second
term tends to zero as \(R\uparrow\infty\), because
\(|y|^{-p-1}|V|^4\in L^1(\mathbb R^3)\) at the fixed time.

Also,
\[
|\chi_{\varepsilon,R}|V|^3V\cdot\nabla w|
\le
p|y|^{-p-1}|V|^4\in L^1(\mathbb R^3),
\]
so dominated convergence gives
\[
\int \chi_{\varepsilon,R}|V|^3V\cdot\nabla w
\to
\int |V|^3V\cdot\nabla w.
\]
Passing first \(\varepsilon\downarrow0\) and then \(R\uparrow\infty\) gives
\[
-\int_{\mathbb R^3}w\,V\cdot\nabla(|V|^3)\,dy
=
\int_{\mathbb R^3}|V|^3V\cdot\nabla w\,dy,
\]
with no boundary contribution at \(y=0\) or at infinity. This is
\(K_{\mathrm{ScaleConvIBP}}^+\).

\(\square\)

---

## T18.5 (Reduced integrated scale-force route)

In any setting where \(K_{\mathrm{AnnConvReg}}^+\) is supplied by the local
regularity or approximation payload, the integrated scale-force payload is
reduced to
\[
K_{\mathrm{ScaleDiffL^1Win}}^+,
\qquad
K_{\mathrm{ScaleV4L^1Win}}^+,
\qquad
K_{\mathrm{ScalePressL^1Win}}^+.
\]
Precisely,
\[
K_{\mathrm{AnnConvReg}}^+
\wedge
K_{\mathrm{ScaleDiffL^1Win}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\wedge
K_{\mathrm{ScalePressL^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleForceL^1Win}}^+.
\]

### Proof

T18.4 gives
\[
K_{\mathrm{AnnConvReg}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleConvIBP}}^+.
\]
Then T18.3 gives
\[
K_{\mathrm{ScaleConvL^1Win}}^+.
\]
Combining this with \(K_{\mathrm{ScaleDiffL^1Win}}^+\) and
\(K_{\mathrm{ScalePressL^1Win}}^+\), T18.2 gives
\(K_{\mathrm{ScaleForceL^1Win}}^+\).

\(\square\)

---

## T18.6 (Remaining scale-force defects)

Before discharging the singular-weight integration by parts, the integrated
scale-force payload is reduced to
\[
K_{\mathrm{ScaleDiffL^1Win}}^+,
\qquad
K_{\mathrm{ScaleConvIBP}}^+,
\qquad
K_{\mathrm{ScaleV4L^1Win}}^+,
\qquad
K_{\mathrm{ScalePressL^1Win}}^+.
\]
Equivalently,
\[
K_{\mathrm{ScaleDiffL^1Win}}^+
\wedge
K_{\mathrm{ScaleConvIBP}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\wedge
K_{\mathrm{ScalePressL^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleForceL^1Win}}^+.
\]

After T18.4 discharges \(K_{\mathrm{ScaleConvIBP}}^+\) from
\(K_{\mathrm{AnnConvReg}}^+\) and \(K_{\mathrm{ScaleV4L^1Win}}^+\), the
remaining scale-row defects in the regularized C7 setting are:

1. \(K_{\mathrm{ScaleDiffL^1Win}}^-\): failure to control the weighted
   diffusive scale-force contribution;
2. \(K_{\mathrm{AnnConvReg}}^-\): failure of the annular convective
   chain-rule/approximation payload;
3. \(K_{\mathrm{ScaleV4L^1Win}}^-\): failure of the weighted \(L^4\)-window
   control needed for the convective scale-force contribution;
4. \(K_{\mathrm{ScalePressL^1Win}}^-\): failure to control the weighted
   pressure scale-force contribution.

Before that discharge is applied, the unresolved scale-row defects are:

1. \(K_{\mathrm{ScaleDiffL^1Win}}^-\): failure to control the weighted
   diffusive scale-force contribution;
2. \(K_{\mathrm{ScaleConvIBP}}^-\): failure of the singular-weight
   convective integration-by-parts identity;
3. \(K_{\mathrm{ScaleV4L^1Win}}^-\): failure of the weighted \(L^4\)-window
   control needed for the convective scale-force contribution;
4. \(K_{\mathrm{ScalePressL^1Win}}^-\): failure to control the weighted
   pressure scale-force contribution.

These are strictly more explicit than the single opaque defect
\(K_{\mathrm{ScaleForceL^1Win}}^-\).
