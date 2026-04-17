# S5: same-point multiscale cascade exclusion

This note proves the conditional exclusion of same-point multiscale cascades in
the NS3D Type II branch. The result has two logically separate parts.

First, a bounded critical \(L^3\) norm forbids an infinite cascade when every
nonzero bubble carries a fixed positive critical mass. This is a direct
consequence of \(L^3\) profile decoupling.

Second, a finite strict same-point cascade reduces, in the innermost
renormalized variables, to a single-bubble repaired-gauge orbit plus compactly
small perturbations. Once the perturbative S3/NRŠ payload is available, the
innermost bubble is impossible. Hence the same-point cascade is impossible.

The only multibubble residue not removed here is the genuinely multi-point
case, or a same-point case in which the no-splitting/profile-compatibility or
perturbative S3 payload fails.

---

## S5.1 (Critical \(L^3\) profiles and orthogonality)

For \(\lambda>0\), \(x_0\in\mathbb R^3\), and
\(\phi\in L^3(\mathbb R^3)\), define the critical NS scaling operator
\[
(\Lambda_{\lambda,x_0}\phi)(x)
:=
\lambda^{-1}\phi\left(\frac{x-x_0}{\lambda}\right).
\]
Then
\[
\|\Lambda_{\lambda,x_0}\phi\|_{L^3(\mathbb R^3)}
=
\|\phi\|_{L^3(\mathbb R^3)}.
\]

Two parameter sequences \((\lambda_i^n,x_i^n)\) and
\((\lambda_j^n,x_j^n)\) are asymptotically orthogonal if
\[
\frac{\lambda_i^n}{\lambda_j^n}
+
\frac{\lambda_j^n}{\lambda_i^n}
+
\frac{|x_i^n-x_j^n|}{\max(\lambda_i^n,\lambda_j^n)}
\longrightarrow \infty .
\]

### Lemma S5.1 (finite \(L^3\) decoupling)

Let \(K<\infty\). Let \(\phi^1,\ldots,\phi^K\in L^3(\mathbb R^3)\), and let
\[
g_n^j:=\Lambda_{\lambda_j^n,x_j^n}\phi^j .
\]
Assume the parameter sequences are pairwise asymptotically orthogonal. Then
\[
\left\|\sum_{j=1}^K g_n^j\right\|_{L^3}^3
=
\sum_{j=1}^K \|\phi^j\|_{L^3}^3+o_n(1).
\]

#### Proof

It suffices first to prove the statement for
\(\phi^j\in C_c^\infty(\mathbb R^3)\). The general \(L^3\) case follows by
density. Indeed, for each \(\varepsilon>0\) choose
\(\psi^j\in C_c^\infty\) with
\[
\|\phi^j-\psi^j\|_3<\varepsilon .
\]
Since each \(\Lambda_{\lambda_j^n,x_j^n}\) is an \(L^3\) isometry,
\[
\left\|
\sum_{j=1}^K\Lambda_{\lambda_j^n,x_j^n}(\phi^j-\psi^j)
\right\|_3
\le K\varepsilon .
\]
On bounded \(L^3\) sets the map \(f\mapsto \|f\|_3^3\) is locally Lipschitz
from \(L^3\) to \(\mathbb R\):
\[
\bigl|\|f\|_3^3-\|h\|_3^3\bigr|
\le
C(\|f\|_3+\|h\|_3)^2\|f-h\|_3 .
\]
Thus the decoupling identity for smooth compact profiles implies the same
identity for \(L^3\) profiles after sending \(\varepsilon\downarrow0\).

Assume now \(\phi^j\in C_c^\infty\). We use induction on \(K\). The case
\(K=1\) is the scaling invariance of the \(L^3\) norm.

Assume the result for \(K-1\), and write
\[
G_n:=\sum_{j=1}^{K-1}\Lambda_{\lambda_j^n,x_j^n}\phi^j,
\qquad
g_n^K:=\Lambda_{\lambda_K^n,x_K^n}\phi^K .
\]
Apply the inverse scaling of the \(K\)-th profile:
\[
(\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n)(y)
=
\sum_{j=1}^{K-1}
\frac{\lambda_K^n}{\lambda_j^n}
\phi^j\left(
\frac{\lambda_K^n y+x_K^n-x_j^n}{\lambda_j^n}
\right).
\]
For each fixed \(j<K\), asymptotic orthogonality gives one of the following
alternatives along the chosen subsequence:

1. \(\lambda_K^n/\lambda_j^n\to0\);
2. \(\lambda_j^n/\lambda_K^n\to0\);
3. the scale ratio stays bounded above and below while
   \(|x_K^n-x_j^n|/\lambda_j^n\to\infty\).

In the first case the displayed term is bounded pointwise by
\[
\frac{\lambda_K^n}{\lambda_j^n}\|\phi^j\|_\infty\to0.
\]
In the second case, for each fixed \(y\), the argument of \(\phi^j\) tends to
infinity for almost every \(y\). More explicitly, if
\(\operatorname{supp}\phi^j\subset B_A\), then the set on which the displayed
term is nonzero is
\[
E_n^j
:=
\left\{
y:
\left|
\frac{\lambda_K^n y+x_K^n-x_j^n}{\lambda_j^n}
\right|\le A
\right\}.
\]
Because \(\lambda_j^n/\lambda_K^n\to0\), the diameter of \(E_n^j\) is
\(O(\lambda_j^n/\lambda_K^n)\to0\). Therefore
\(\mathbf 1_{E_n^j}\to0\) pointwise outside at most one limiting point after
passing to a subsequence, and the term converges to zero a.e. In the third
case, the spatial translation tends to infinity in the rescaled variables, so
for every fixed compact \(B_R\) and all large \(n\) the argument of \(\phi^j\)
lies outside \(\operatorname{supp}\phi^j\); compact support again gives
pointwise convergence to zero. Hence
\[
\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n\to0
\quad\text{a.e. on }\mathbb R^3.
\]
The sequence \(\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n\) is bounded in \(L^3\),
because the inverse scaling is an \(L^3\) isometry and the finite sum is
bounded in \(L^3\).

The Brézis--Lieb lemma, applied to
\[
\phi^K+\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n,
\]
gives
\[
\left\|
\phi^K+\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n
\right\|_3^3
=
\|\phi^K\|_3^3
+
\left\|
\Lambda_{\lambda_K^n,x_K^n}^{-1}G_n
\right\|_3^3
+o_n(1).
\]
Using the \(L^3\)-isometry again,
\[
\|G_n+g_n^K\|_3^3
=
\|\phi^K\|_3^3+\|G_n\|_3^3+o_n(1).
\]
The induction hypothesis gives
\[
\|G_n\|_3^3
=
\sum_{j=1}^{K-1}\|\phi^j\|_3^3+o_n(1).
\]
Combining the two identities proves the claim for \(K\), and the induction is
complete.

\(\square\)

---

## S5.2 (Admissible same-point cascade decompositions)

Let \(u_n:=u(\cdot,t_n)\), with \(t_n\uparrow T^*\). A same-point critical
cascade decomposition at \(x_*\) consists of profiles
\(\{\phi^j\}_{j\in\mathcal J}\subset L^3(\mathbb R^3)\), scales
\(\lambda_j^n\downarrow0\), centers \(x_j^n\to x_*\), and remainders
\(r_n^J\) such that, for every finite \(J\subset\mathcal J\),
\[
u_n
=
\sum_{j\in J}\Lambda_{\lambda_j^n,x_j^n}\phi^j+r_n^J
\]
and the following hold.

1. Pairwise asymptotic orthogonality:
   \[
   \frac{\lambda_i^n}{\lambda_j^n}
   +
   \frac{\lambda_j^n}{\lambda_i^n}
   +
   \frac{|x_i^n-x_j^n|}{\max(\lambda_i^n,\lambda_j^n)}
   \to\infty
   \quad (i\ne j).
   \]
2. Critical \(L^3\) decoupling with remainder:
   \[
   \|u_n\|_3^3
   =
   \sum_{j\in J}\|\phi^j\|_3^3
   +
   \|r_n^J\|_3^3
   +o_n(1)
   \]
   for every finite \(J\subset\mathcal J\).
3. Uniform nontriviality of active bubbles:
   \[
   \|\phi^j\|_3\ge\eta>0
   \quad\text{for every }j\in\mathcal J.
   \]

The second item is the \(L^3\) version of the usual profile-decomposition
Pythagorean expansion. Lemma S5.1 proves the atom-atom part of this identity;
the atom-remainder decoupling is the standard Brézis--Lieb remainder condition
in a strong \(L^3\) profile decomposition.

---

## S5.3 (Critical-mass exhaustion of cascades)

### Theorem S5.2 (uniform bound on cascade length)

Assume
\[
\sup_n\|u_n\|_{L^3(\mathbb R^3)}\le M<\infty .
\]
Assume \(u_n\) admits an admissible same-point critical cascade decomposition
with every active profile satisfying
\[
\|\phi^j\|_3\ge\eta>0.
\]
Then the number of active profiles is finite and satisfies
\[
\#\mathcal J\le
\left\lfloor \frac{M^3}{\eta^3}\right\rfloor .
\]

#### Proof

Let \(J\subset\mathcal J\) be finite. By \(L^3\) decoupling with remainder,
\[
\|u_n\|_3^3
=
\sum_{j\in J}\|\phi^j\|_3^3+\|r_n^J\|_3^3+o_n(1).
\]
The remainder contribution is nonnegative, so
\[
\limsup_{n\to\infty}\|u_n\|_3^3
\ge
\sum_{j\in J}\|\phi^j\|_3^3 .
\]
Since \(\|u_n\|_3\le M\),
\[
M^3
\ge
\sum_{j\in J}\|\phi^j\|_3^3
\ge
(\#J)\eta^3 .
\]
Therefore
\[
\#J\le \frac{M^3}{\eta^3}
\]
for every finite \(J\subset\mathcal J\). If \(\mathcal J\) had more than
\(\lfloor M^3/\eta^3\rfloor\) elements, choosing \(J\) with
\[
\#J=\left\lfloor \frac{M^3}{\eta^3}\right\rfloor+1
\]
would contradict the preceding inequality. Hence
\[
\#\mathcal J\le
\left\lfloor \frac{M^3}{\eta^3}\right\rfloor .
\]

\(\square\)

---

## S5.4 (Regular strict same-point cascades)

The critical-mass bound rules out infinite cascades. To rule out a finite
same-point cascade, one must reduce the innermost scale to a perturbed
single-bubble repaired-gauge orbit.

Assume there are \(J\ge2\) active profiles ordered by scale:
\[
\lambda_1^n\ll\lambda_2^n\ll\cdots\ll\lambda_J^n,
\qquad
\lambda_j^n\to0,
\qquad
x_j^n\to x_*.
\]
Set
\[
\mu_j^n:=\frac{\lambda_1^n}{\lambda_j^n}\to0
\quad (j\ge2).
\]
The cascade is called regular in the innermost coordinates if, after passing
to a subsequence, the following hold.

1. Outer centers have finite outer-scale offsets:
   \[
   \xi_j^n:=\frac{x_*-x_j^n}{\lambda_j^n}
   \longrightarrow \xi_j\in\mathbb R^3
   \quad (j\ge2).
   \]
2. Each outer profile is \(C^2\) in a neighborhood of \(\xi_j\).
3. For every fixed \(R<\infty\),
   \[
   \sup_{|z-\xi_j|\le R\mu_j^n}
   \bigl(|\phi^j(z)|+|\nabla\phi^j(z)|+|\nabla^2\phi^j(z)|\bigr)
   <\infty
   \]
   uniformly for large \(n\).
4. The innermost repaired-gauge representation is available:
   \[
   V_1^n(y,\tau)
   =
   \lambda_1(t)u(x_*(t)+\lambda_1(t)y,t)
   \]
   solves the repaired-gauge NS equation on compact \((y,\tau)\)-cylinders,
   modulo the outer-bubble perturbation described below.

If \(|\xi_j^n|\to\infty\), the corresponding outer bubble is not regular in
the above finite-offset sense, but it is locally invisible in the innermost
variables by Lemma S5.5. Under \(K_{\mathrm{EscPressDec}}^+\), it is
perturbative for the innermost S3 branch. Thus escaping-offset outer profiles
are not an independent same-point residue once the pressure-decoupling payload
is supplied.

---

## S5.5 (Inner expansion of outer bubbles)

For \(j\ge2\), write the \(j\)-th outer bubble in the innermost variables:
\[
W_j^n(y)
:=
\lambda_1^n
(\lambda_j^n)^{-1}
\phi^j\left(\frac{\lambda_1^n y+x_*-x_j^n}{\lambda_j^n}\right)
=
\mu_j^n\phi^j(\xi_j^n+\mu_j^n y).
\]

### Lemma S5.3 (outer bubbles are constant drifts to first order)

For every fixed \(R<\infty\),
\[
W_j^n(y)
=
\mu_j^n \phi^j(\xi_j)
+
(\mu_j^n)^2\,D\phi^j(\xi_j)y
+
\mathcal R_j^n(y)
\quad\text{on }B_R,
\]
where
\[
\|\mathcal R_j^n\|_{L^\infty(B_R)}
\le
C_{j,R}\Bigl((\mu_j^n)^3+ \mu_j^n|\xi_j^n-\xi_j|^2
+(\mu_j^n)^2|\xi_j^n-\xi_j|\Bigr)
=o(\mu_j^n)+O((\mu_j^n)^3).
\]
In particular,
\[
W_j^n
=
c_j^n+A_j^n y+o_{L^\infty(B_R)}(\mu_j^n),
\]
with
\[
c_j^n:=\mu_j^n\phi^j(\xi_j),
\qquad
A_j^n:=(\mu_j^n)^2D\phi^j(\xi_j),
\qquad
\|A_j^n\|=O((\mu_j^n)^2).
\]

#### Proof

Fix \(R\). For \(y\in B_R\), Taylor's theorem at \(\xi_j\) gives
\[
\phi^j(\xi_j^n+\mu_j^n y)
=
\phi^j(\xi_j)
+
D\phi^j(\xi_j)(\xi_j^n-\xi_j+\mu_j^n y)
+
\frac12
(\xi_j^n-\xi_j+\mu_j^n y)^T
D^2\phi^j(\zeta_{j,n,y})
(\xi_j^n-\xi_j+\mu_j^n y)
\]
for some \(\zeta_{j,n,y}\) on the segment between
\(\xi_j\) and \(\xi_j^n+\mu_j^n y\). Multiplying by \(\mu_j^n\) yields
\[
W_j^n(y)
=
\mu_j^n\phi^j(\xi_j)
+
\mu_j^n D\phi^j(\xi_j)(\xi_j^n-\xi_j)
+
(\mu_j^n)^2D\phi^j(\xi_j)y
+
O\!\left(
\mu_j^n|\xi_j^n-\xi_j+\mu_j^n y|^2
\right).
\]
The term
\(\mu_j^nD\phi^j(\xi_j)(\xi_j^n-\xi_j)\) is a constant-in-\(y\) vector and
can be included in \(c_j^n\) if one uses
\[
c_j^n:=\mu_j^n\phi^j(\xi_j^n)
\]
instead. With the displayed choice \(c_j^n=\mu_j^n\phi^j(\xi_j)\), it is part
of the stated remainder. Since \(\xi_j^n\to\xi_j\) and \(\mu_j^n\to0\), the
remainder estimate follows.

\(\square\)

For the dynamical reduction it is convenient to use the exact constant
\[
c_j^n:=\mu_j^n\phi^j(\xi_j^n),
\]
so that
\[
W_j^n(y)
=
c_j^n
+
(\mu_j^n)^2D\phi^j(\xi_j^n)y
+
O_{L^\infty(B_R)}((\mu_j^n)^3R^2).
\]
Because Theorem S5.2 gives \(J<\infty\),
\[
c_{\mathrm{out}}^n:=\sum_{j=2}^J c_j^n
=O\left(\sum_{j=2}^J\mu_j^n\right)
\to0
\]
on every fixed cascade sequence, provided the profiles are locally bounded at
their offsets.

---

## S5.6 (Absorption of the constant ambient velocity)

Consider the repaired-gauge equation
\[
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V+a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0.
\]
Suppose, on a compact cylinder, the total inner field has the form
\[
V=V_{\mathrm{in}}+c(\tau)+S,
\]
where \(c(\tau)\) is independent of \(y\), and \(S\) is a perturbation. Ignoring
for the moment the perturbation \(S\), one has
\[
(V_{\mathrm{in}}+c)\cdot\nabla(V_{\mathrm{in}}+c)
=
(V_{\mathrm{in}}\cdot\nabla)V_{\mathrm{in}}
+
c\cdot\nabla V_{\mathrm{in}}.
\]
Thus the constant ambient velocity changes only the translation row:
\[
\partial_\tau V_{\mathrm{in}}
+
(V_{\mathrm{in}}\cdot\nabla)V_{\mathrm{in}}
+
\nabla P_{\mathrm{in}}
=
\nu\Delta V_{\mathrm{in}}
+
a(V_{\mathrm{in}}+y\cdot\nabla V_{\mathrm{in}})
+
(b-c)\cdot\nabla V_{\mathrm{in}}.
\]
Terms depending only on \(\tau\), such as \(\partial_\tau c-a c\), are spatially
constant forces. They are removed by the equivalent translation-gauge
description, i.e. by choosing the center velocity so that the ambient constant
flow is part of the moving frame. With the sign convention in the displayed
equation, the effective translation parameter is
\[
b_{\mathrm{eff}}=b-c .
\]

Therefore the leading constant part of every outer bubble in the inner
coordinates is absorbed into \(b_{\mathrm{eff}}\). Since
\[
|c_{\mathrm{out}}^n|
\le
\sum_{j=2}^J
\mu_j^n|\phi^j(\xi_j^n)|,
\]
finite cascade length and local boundedness of the profiles imply
\[
\sup_n |c_{\mathrm{out}}^n|<\infty,
\qquad
c_{\mathrm{out}}^n\to0
\]
along a strict cascade. Hence bounded modulation is preserved whenever the
unperturbed repaired-gauge \(b\)-parameter is bounded.

---

## S5.7 (Perturbative single-bubble reduction)

Let
\[
E_n
\]
denote the remaining inner-coordinate error after subtracting the innermost
bubble and absorbing \(c_{\mathrm{out}}^n\) into \(b_{\mathrm{eff}}\). This
error contains:

1. the linear shear terms
   \[
   \sum_{j=2}^J(\mu_j^n)^2D\phi^j(\xi_j^n)y;
   \]
2. Taylor remainders of size \(O((\mu_j^n)^3)\) on fixed \(B_R\);
3. the rescaled critical remainder from the profile decomposition;
4. pressure cross-terms generated by the interaction of the innermost profile
   with the outer perturbation.

The perturbative S3 payload is the following assertion:
\[
K_{\mathrm{S3PertRob}}^+.
\]
It consists of these estimates on every compact cylinder
\[
B_R\times[\tau_0,\tau_0+L]:
\]

1. local forcing convergence:
   \[
   E_n\to0
   \quad\text{in }L^1_\tau H^{-1}_y(B_R)
   \]
   or in any stronger topology sufficient for the S3 compactness passage;
2. pressure compatibility:
   the pressure generated by the cross-terms converges to zero in the pressure
   topology used by T1--T6 and S3;
3. compactness stability:
   the local compactness, autonomous modulation, and stationary-limit
   extraction in S3 are unchanged by adding \(E_n\to0\);
4. rigidity stability:
   the limiting stationary equation is exactly the NRŠ-covered self-similar
   profile equation, not a perturbed equation.

Under \(K_{\mathrm{S3PertRob}}^+\), the innermost bubble emits a genuine S3
single-bubble branch with bounded modulation and positive critical mass.

---

## S5.8 (Exclusion theorem for regular same-point cascades)

### Theorem S5.4

Let \(u\) be a Leray--Hopf solution on
\(\mathbb R^3\times[0,T^*)\) with finite initial energy and
\[
\sup_{t<T^*}\|u(t)\|_{L^3(\mathbb R^3)}\le M<\infty .
\]
Let \(t_n\uparrow T^*\) be a concentrating sequence at \(x_*\). Assume:

1. \(u_n=u(\cdot,t_n)\) admits an admissible same-point critical cascade
   decomposition in the sense of S5.2;
2. every active bubble satisfies
   \[
   \|\phi^j\|_3\ge\eta>0;
   \]
3. the active bubbles form a regular strict same-point cascade in the sense
   of S5.4;
4. the innermost scale admits the repaired-gauge representation with bounded
   modulation, pressure reconstruction, Caccioppoli regularity, and positive
   finite critical mass;
5. the perturbative S3 robustness payload \(K_{\mathrm{S3PertRob}}^+\) holds;
6. the S3 nonzero-extraction and NRŠ rigidity payload
   \(K_{\mathrm{S3NRSPayload}}^+\) holds for the innermost branch.

Then the cascade cannot occur.

#### Proof

By Theorem S5.2, the number of active bubbles is finite:
\[
J\le \left\lfloor \frac{M^3}{\eta^3}\right\rfloor .
\]
Thus all outer-bubble contributions to the innermost variables are finite
sums.

Let \(\lambda_1^n\) be the smallest scale. For \(j\ge2\), set
\[
\mu_j^n=\lambda_1^n/\lambda_j^n\to0.
\]
Lemma S5.3 gives, on every fixed \(B_R\),
\[
\lambda_1^n
(\lambda_j^n)^{-1}
\phi^j\left(\frac{\lambda_1^n y+x_*-x_j^n}{\lambda_j^n}\right)
=
c_j^n
+
(\mu_j^n)^2D\phi^j(\xi_j^n)y
+
O_{L^\infty(B_R)}((\mu_j^n)^3R^2).
\]
Summing over \(2\le j\le J\), the constant part
\[
c_{\mathrm{out}}^n=\sum_{j=2}^J c_j^n
\]
is bounded and tends to zero. By S5.6 it is absorbed into the repaired-gauge
translation parameter:
\[
b_{\mathrm{eff}}^n=b^n-c_{\mathrm{out}}^n .
\]
The bounded-modulation hypothesis for \(b^n\), together with boundedness of
\(c_{\mathrm{out}}^n\), gives
\[
\sup_n |b_{\mathrm{eff}}^n|<\infty .
\]

The nonconstant outer contribution is
\[
\sum_{j=2}^J(\mu_j^n)^2D\phi^j(\xi_j^n)y
+
O_{L^\infty(B_R)}\left(\sum_{j=2}^J(\mu_j^n)^3R^2\right),
\]
which converges to zero uniformly on each fixed \(B_R\). The rescaled
profile-decomposition remainder and pressure cross-terms are included in
\(E_n\). By \(K_{\mathrm{S3PertRob}}^+\),
\[
E_n\to0
\]
in the topology required by the S3 compactness and rigidity passage.

Consequently, the innermost branch satisfies the repaired-gauge NS equation
with bounded effective modulation, positive finite critical mass, pressure
reconstruction, Caccioppoli regularity, and a perturbation that vanishes in
the S3 limit. The scale \(\lambda_1(t)\to0\) gives the scale-collapse drift
input for S3. Therefore
\[
K_{\mathrm{S3NRSPayload}}^+
\]
extracts a nonzero stationary \(L^3(\mathbb R^3)\) profile in the
NRŠ-covered self-similar class.

The Nečas--Růžička--Šverák rigidity theorem rules out every nonzero
\(L^3(\mathbb R^3)\) stationary profile in that class. This contradicts the
nonzero extraction, which ultimately comes from
\[
\|\phi^1\|_3\ge\eta>0.
\]
Hence the assumed regular strict same-point cascade cannot occur.

\(\square\)

---

## S5.9 (Endpoint classification contribution)

S5 removes the same-point regular strict cascade under the following exact
payload package:
\[
K_{\mathrm{L^3ProfDec}}^+
\wedge
K_{\mathrm{BubbleMassFloor}}^+
\wedge
K_{\mathrm{RegularNestedCascade}}^+
\wedge
K_{\mathrm{InnerRepBridge}}^+
\wedge
K_{\mathrm{S3PertRob}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+ .
\]
Here:

1. \(K_{\mathrm{L^3ProfDec}}^+\) is the strong \(L^3\) profile decomposition
   with Brézis--Lieb decoupling and remainder decoupling;
2. \(K_{\mathrm{BubbleMassFloor}}^+\) is the uniform lower bound
   \(\|\phi^j\|_3\ge\eta\);
3. \(K_{\mathrm{RegularNestedCascade}}^+\) is the finite-offset \(C^2\)
   nested-cascade regularity needed for the inner Taylor expansion;
4. \(K_{\mathrm{InnerRepBridge}}^+\) is the repaired-gauge representation of
   the innermost branch with pressure, Caccioppoli regularity, bounded
   modulation, and positive finite critical mass;
5. \(K_{\mathrm{S3PertRob}}^+\) says the \(O(\mu_j^2)\) shear terms, Taylor
   remainders, profile remainder, and pressure cross-terms vanish in the S3
   limiting topology;
6. \(K_{\mathrm{S3NRSPayload}}^+\) is the nonzero-extraction and NRŠ rigidity
   package from S3.

Thus bounded critical mass alone proves that a cascade has uniformly bounded
length. The full exclusion of the regular same-point cascade follows only
after the inner-decoupling and perturbative S3 payloads are supplied.

The remaining same-point multibubble residue is therefore precisely the
failure of one of these payloads, equivalently a no-splitting or
profile-compatibility defect. The genuinely multi-point residue is not treated
by S5.

---

## S5.10 (Hypothesis audit and discharge ledger)

This section records which hypotheses in Theorem S5.4 are proved inside S5,
which are standard NS3D backend payloads, and which failures are reclassified
as multibubble residue.

### Discharged inside S5

The following implications are proved in this note.

1. Atom-atom \(L^3\) decoupling:
   \[
   \text{pairwise profile orthogonality}
   \Longrightarrow
   \left\|\sum_{j=1}^K\Lambda_{\lambda_j^n,x_j^n}\phi^j\right\|_3^3
   =
   \sum_{j=1}^K\|\phi^j\|_3^3+o_n(1).
   \]
2. Critical-mass cascade exhaustion:
   \[
   \sup_n\|u_n\|_3\le M,\quad
   \|\phi^j\|_3\ge\eta
   \Longrightarrow
   \#\mathcal J\le\lfloor M^3/\eta^3\rfloor .
   \]
3. Inner expansion of every regular outer bubble:
   \[
   \mu_j^n\phi^j(\xi_j^n+\mu_j^n y)
   =
   c_j^n+(\mu_j^n)^2D\phi^j(\xi_j^n)y
   +O_{L^\infty(B_R)}((\mu_j^n)^3R^2).
   \]
4. Absorption of the constant ambient velocity:
   \[
   c_{\mathrm{out}}^n
   \quad\text{changes only}\quad
   b^n\mapsto b^n-c_{\mathrm{out}}^n .
   \]
5. Finite regular cascades have perturbations of size
   \(O(\sum_{j=2}^J(\mu_j^n)^2)\) on fixed inner balls after the constant
   ambient velocity is absorbed.

These steps are no longer hypotheses.

### Backend profile-decomposition payload

The only profile-decomposition input not proved from first principles in this
note is the strong \(L^3\) atom-remainder decoupling:
\[
K_{\mathrm{L^3ProfDec},NS3D}^+.
\]
It consists of:
\[
u_n
=
\sum_{j=1}^J\Lambda_{\lambda_j^n,x_j^n}\phi^j+r_n^J,
\]
pairwise parameter orthogonality, and
\[
\|u_n\|_3^3
=
\sum_{j=1}^J\|\phi^j\|_3^3+\|r_n^J\|_3^3+o_n(1).
\]
Lemma S5.1 proves the profile-profile part of this formula. The
profile-remainder term is part of the NS3D concentration-compactness backend.
If \(K_{\mathrm{L^3ProfDec},NS3D}^+\) fails, the failure is a backend
exhaustion/profile-decomposition defect, not a new singularity class.

### Active-bubble mass floor

The mass floor is discharged for active singular bubbles by the standard
small-data critical regularity threshold. Let
\[
\varepsilon_{\mathrm{sd}}>0
\]
be an \(L^3\) small-data threshold for the 3D Navier-Stokes flow, meaning that
initial data with \(L^3\) norm below \(\varepsilon_{\mathrm{sd}}\) generate a
global regular mild solution with perturbative critical bounds.

Assume the nonlinear profile stability payload
\[
K_{\mathrm{ProfStab},NS3D}^+
\]
which says that a profile whose critical norm is below
\(\varepsilon_{\mathrm{sd}}\) cannot be an active singular profile in the
Type II decomposition; its nonlinear evolution remains perturbative and is
absorbed into the regular remainder ledger.

Then every active singular bubble satisfies
\[
\|\phi^j\|_3\ge\varepsilon_{\mathrm{sd}}.
\]
Thus in Theorem S5.2 one may take
\[
\eta=\varepsilon_{\mathrm{sd}}
\]
for active singular bubbles. Small profiles are regular perturbative pieces;
they are not counted as singular cascade bubbles.

### Regular nested-cascade dichotomy

For a same-point strict cascade and an outer bubble \(j\ge2\), define
\[
\xi_j^n=\frac{x_*-x_j^n}{\lambda_j^n}.
\]
After passing to a subsequence, either:
\[
\xi_j^n\to\xi_j\in\mathbb R^3,
\]
or
\[
|\xi_j^n|\to\infty.
\]

If \(\xi_j^n\to\xi_j\) and the profile is locally \(C^2\) near \(\xi_j\), then
Lemma S5.3 discharges the inner expansion hypothesis.

If \(|\xi_j^n|\to\infty\), the outer bubble is locally invisible in the
innermost variables. This is discharged by Lemma S5.5 below, provided the
pressure interaction with the innermost profile satisfies the stated
cross-pressure decoupling payload.

If the local \(C^2\) regularity near \(\xi_j\) is missing, the failure is a
profile-regularity defect in the backend representation. For NS3D profile
evolutions this is discharged at any positive profile time by parabolic
smoothing; at the initial profile time it remains part of
\[
K_{\mathrm{InnerRepBridge}}^+
\quad\text{or}\quad
K_{\mathrm{ProfReg},NS3D}^+.
\]

### Escaping-offset outer profiles

Let
\[
W_j^n(y)=\mu_j^n\phi^j(\xi_j^n+\mu_j^n y),
\qquad
\mu_j^n=\frac{\lambda_1^n}{\lambda_j^n}\to0.
\]

#### Lemma S5.5 (escaping offsets vanish locally)

Assume \(\phi^j\in L^3(\mathbb R^3)\) and
\[
|\xi_j^n|\to\infty .
\]
Then, for every fixed \(R<\infty\),
\[
\|W_j^n\|_{L^3(B_R)}\to0.
\]

If additionally the profile tail is locally smooth along the escaping centers
in the sense that
\[
\|\nabla W_j^n\|_{L^3(B_R)}
+
\|D^2 W_j^n\|_{L^3(B_R)}
\to0
\]
for every fixed \(R\), then \(W_j^n\to0\) in \(W^{2,3}_{\mathrm{loc}}\).

#### Proof

By the change of variables
\[
z=\xi_j^n+\mu_j^n y,
\qquad
dy=(\mu_j^n)^{-3}dz,
\]
we have
\[
\|W_j^n\|_{L^3(B_R)}^3
=
\int_{B_R}(\mu_j^n)^3
|\phi^j(\xi_j^n+\mu_j^n y)|^3\,dy
=
\int_{B(\xi_j^n,\mu_j^n R)}|\phi^j(z)|^3\,dz.
\]
Since \(|\xi_j^n|\to\infty\) and \(\mu_j^nR\to0\), the balls
\[
B(\xi_j^n,\mu_j^n R)
\]
eventually lie in \(\{|z|>A\}\) for every fixed \(A<\infty\). Because
\(\phi^j\in L^3(\mathbb R^3)\),
\[
\int_{|z|>A}|\phi^j(z)|^3\,dz\to0
\quad\text{as }A\to\infty.
\]
Therefore
\[
\|W_j^n\|_{L^3(B_R)}^3
\le
\int_{|z|>A}|\phi^j(z)|^3\,dz
\]
for all sufficiently large \(n\), and then the right-hand side is made
arbitrarily small by taking \(A\) large. This proves the \(L^3_{\mathrm{loc}}\)
vanishing.

The \(W^{2,3}_{\mathrm{loc}}\) conclusion is exactly the additional derivative
tail hypothesis.

\(\square\)

The pressure contribution of an escaping-offset bubble is not automatic from
local velocity vanishing alone, because pressure is nonlocal. We therefore
define the escaping-offset pressure-decoupling payload
\[
K_{\mathrm{EscPressDec}}^+
\]
to mean that, for every compact \(B_R\) and compact time window, the pressure
generated by the cross source
\[
U_n\odot W_j^n+W_j^n\odot U_n+W_j^n\otimes W_j^n
\]
converges to zero modulo constants in the pressure topology used by T1--T6.
Under \(K_{\mathrm{EscPressDec}}^+\), escaping-offset outer profiles are
perturbative for the innermost S3 branch.

### Perturbative S3 robustness

The perturbative payload \(K_{\mathrm{S3PertRob}}^+\) is discharged under the
following explicit analytic estimates. Let \(U_n\) denote the innermost
renormalized branch after absorbing \(c_{\mathrm{out}}^n\) into \(b^n\), and
let \(S_n\) denote the remaining outer perturbation on
\[
B_R\times I.
\]
Assume:
\[
\|S_n\|_{L^\infty_\tau W^{2,\infty}_y(B_R)}
+
\|\partial_\tau S_n\|_{L^1_\tau H^{-1}_y(B_R)}
\longrightarrow0,
\]
\[
\|U_n\|_{L^\infty_\tau L^3_y(B_{2R})}
+
\|U_n\|_{L^2_\tau H^1_y(B_{2R})}
\le C_R,
\]
and the pressure reconstruction satisfies the local Calderon-Zygmund estimate
for cross terms.

Then every perturbative term produced by replacing \(U_n+S_n\) with \(U_n\)
vanishes in \(L^1_\tau H^{-1}_y(B_R)\). Indeed:
\[
\|\partial_\tau S_n\|_{L^1H^{-1}}\to0,
\qquad
\|\Delta S_n\|_{L^1H^{-1}}
\le C_R\|S_n\|_{L^1W^{2,\infty}}\to0.
\]
The linear transport terms satisfy
\[
\|(U_n\cdot\nabla)S_n\|_{L^1H^{-1}(B_R)}
\le
C_R\|U_n\|_{L^2L^2(B_R)}
\|\nabla S_n\|_{L^2L^\infty(B_R)}
\to0,
\]
and
\[
\|(S_n\cdot\nabla)U_n\|_{L^1H^{-1}(B_R)}
\le
C_R\|S_n\|_{L^\infty L^\infty(B_R)}
\|\nabla U_n\|_{L^1L^2(B_R)}
\to0.
\]
The quadratic perturbation obeys
\[
\|(S_n\cdot\nabla)S_n\|_{L^1H^{-1}(B_R)}
\le
C_R\|S_n\|_{L^\infty L^\infty}
\|\nabla S_n\|_{L^1L^\infty}
\to0.
\]
The modulation terms satisfy
\[
\|a_n(S_n+y\cdot\nabla S_n)+b_n\cdot\nabla S_n\|_{L^1H^{-1}(B_R)}
\le
C_R(M_{ab})
\|S_n\|_{L^1W^{1,\infty}(B_R)}
\to0.
\]
For the pressure, the cross source has the form
\[
2U_n\odot S_n+S_n\otimes S_n.
\]
On \(B_{2R}\),
\[
\|U_n\odot S_n\|_{L^1_\tau L^{3/2}_y}
\le
\|S_n\|_{L^\infty_{\tau,y}}
\|U_n\|_{L^1_\tau L^3_y}
\to0,
\]
and
\[
\|S_n\otimes S_n\|_{L^1_\tau L^{3/2}_y}\to0.
\]
The local pressure estimate then gives convergence of the cross pressure
modulo constants in the pressure topology used by T1--T6; hence its gradient
vanishes in \(L^1_\tau H^{-1}_y(B_R)\).

For a regular finite strict cascade, Lemma S5.3 and bounded relative
modulation give the required \(S_n\to0\) estimates. Therefore
\[
K_{\mathrm{RegularNestedCascade}}^+
\wedge
K_{\mathrm{ModBd}}^+
\wedge
K_{\mathrm{PressureRep}}^+
\wedge
K_{\mathrm{WinH1}}^+
\Longrightarrow
K_{\mathrm{S3PertRob}}^+ .
\]

### Final S5 discharge theorem

Assume the NS3D backend supplies
\[
K_{\mathrm{L^3ProfDec},NS3D}^+,
\quad
K_{\mathrm{ProfStab},NS3D}^+,
\quad
K_{\mathrm{InnerRepBridge}}^+,
\quad
K_{\mathrm{S3NRSPayload}}^+ .
\]
Then every regular strict same-point cascade of active singular profiles is
ruled out.

If such a same-point cascade is not ruled out by this theorem, then at least
one of the following occurs:

1. the profile decomposition/remainder decoupling fails;
2. an alleged active bubble is below the critical small-data threshold and is
   therefore not singular;
3. an escaping-offset outer bubble fails the cross-pressure decoupling payload
   \(K_{\mathrm{EscPressDec}}^+\);
4. the local profile regularity or innermost representation fails;
5. the S3 nonzero-extraction/NRŠ payload fails.

Items 1, 3, 4, and 5 are technical bridge or interaction-decoupling defects.
Item 2 is regular and is removed from the singular ledger.

Thus S5 leaves no separate same-point cascade singularity class. After the
technical payloads are supplied, the only unresolved same-point case is the
multibubble no-splitting/profile-compatibility residue.

---

## S5.11 (Conditional full same-point cascade exclusion from nonlinear decoupling)

The preceding sections close regular nested cascades and escaping-offset
subcases under explicit perturbative estimates. The remaining same-point
multibubble case is precisely the failure of nonlinear no-splitting. We record
the conditional theorem that would close all same-point cascades if that
payload is supplied.

Define the same-point nonlinear decoupling payload
\[
K_{\mathrm{SamePointNLDec}}^+
\]
as the conjunction of the following assertions for every finite same-point
strict scale-separated profile family:

1. strong critical decomposition:
   \[
   u_n
   =
   \sum_{j=1}^J\Lambda_{\lambda_j^n,x_j^n}\phi^j+r_n,
   \qquad
   \|u_n\|_3^3
   =
   \sum_{j=1}^J\|\phi^j\|_3^3+\|r_n\|_3^3+o_n(1);
   \]
2. nonlinear profile stability:
   profiles below the critical small-data threshold are perturbative and every
   active singular profile has \(L^3\)-mass at least
   \(\varepsilon_{\mathrm{sd}}\);
3. innermost velocity decoupling:
   after rescaling by the smallest scale and absorbing all constant ambient
   velocities into the translation gauge, the sum of all outer profiles and
   the rescaled remainder is a perturbation \(S_n\) with
   \[
   S_n\to0
   \]
   in the local velocity topology required by \(K_{\mathrm{S3PertRob}}^+\);
4. pressure decoupling:
   all pressure cross-terms between the innermost profile, outer profiles, and
   the remainder vanish modulo constants in the T1--T6/S3 pressure topology;
5. modulation compatibility:
   the effective repaired-gauge parameters of the innermost branch remain
   bounded after ambient constants are absorbed;
6. S3 topology compatibility:
   the full interaction error vanishes in the compactness and stationarity
   passage used by S3.

### Theorem S5.6 (conditional no same-point multibubble theorem)

Assume a declared NS3D Type II candidate has all technical bridge payloads
needed to enter the repaired-gauge Type II ledger, and assume:
\[
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{InnerRepBridge}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+.
\]
Then no same-point multibubble cascade of active singular profiles can occur.

#### Proof

Let a same-point strict scale-separated cascade be given. By
\(K_{\mathrm{SamePointNLDec}}^+\), the decomposition is strongly decoupled in
critical \(L^3\), and every active singular bubble carries at least
\(\varepsilon_{\mathrm{sd}}\) critical mass. Since the ambient Type II branch
has bounded critical mass,
\[
\sup_n\|u_n\|_3\le M,
\]
Theorem S5.2 gives a finite number of active bubbles:
\[
J\le \left\lfloor \frac{M^3}{\varepsilon_{\mathrm{sd}}^3}\right\rfloor .
\]

Choose the innermost scale \(\lambda_1^n\). The nonlinear decoupling payload
says that, in the innermost variables and after translation-gauge absorption
of constant ambient velocities,
\[
V_n=U_n+S_n,
\qquad
S_n\to0
\]
in the full S3 perturbative topology, including pressure and modulation
compatibility. Hence the innermost branch \(U_n\) satisfies the repaired-gauge
NS equation with bounded effective modulation, positive finite critical mass,
pressure reconstruction, and Caccioppoli regularity, up to an error that
vanishes in the S3 limiting passage.

The scale \(\lambda_1^n\to0\) gives the scale-collapse input. Therefore
\[
K_{\mathrm{S3NRSPayload}}^+
\]
extracts a nonzero stationary \(L^3(\mathbb R^3)\) profile in the
NRŠ-covered self-similar class. Nonzero extraction follows from the active
bubble mass floor:
\[
\|\phi^1\|_3\ge\varepsilon_{\mathrm{sd}}>0.
\]
The Nečas--Růžička--Šverák rigidity theorem rules out such a nonzero
\(L^3\) stationary self-similar profile. This contradiction excludes the
same-point cascade.

\(\square\)

Consequently, after S5.6 the only same-point obstruction left in the folder is
the failure of
\[
K_{\mathrm{SamePointNLDec}}^+,
\]
which is exactly the no-splitting/profile-compatibility multibubble problem.
