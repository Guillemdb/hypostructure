# S7: discharge of multibubble decoupling payloads

This note proves the same-point and separated-point camera-decoupling payloads
from explicit nonlinear profile-evolution estimates. It separates what follows
from static scale separation from what requires genuine nonlinear
Navier-Stokes profile control.

The output is:
\[
K_{\mathrm{NLProfDec},NS3D}^+
\Longrightarrow
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{MultiPointCamDec}}^+ .
\]
Combined with S6, this eliminates the multibubble residue under the terminal
nonlinear profile-decoupling theorem \(K_{\mathrm{NLProfDec},NS3D}^+\).

The theorem is conditional on \(K_{\mathrm{NLProfDec},NS3D}^+\). That payload
is the precise terminal-camera no-splitting/profile-compatibility theorem for
NS3D critical profiles. S8 proves this payload from terminal windowed
local compactness, scattering removal, exterior-regular discard,
repaired-gauge representation, and Caccioppoli regularity. S7 proves that no
further multibubble mechanism remains after that payload is supplied.

---

## S7.1 (Local perturbation criterion)

Let \(I\subset\mathbb R\) be a compact renormalized-time interval and
\[
Q_R:=B_R\times I.
\]
Let \(U_n\) be an S3-admissible repaired-gauge branch on \(Q_{2R}\), with
\[
\|U_n\|_{L^\infty_\tau L^3_y(B_{2R})}
+
\|U_n\|_{L^2_\tau H^1_y(B_{2R})}
\le C_R.
\]
Let \(S_n\) be an exterior or outer-profile perturbation. Assume:
\[
\nabla\cdot U_n=0,
\qquad
\nabla\cdot S_n=0
\quad\text{in distributions on }Q_{2R},
\]
\[
S_n\to0
\quad\text{in }L^\infty_\tau L^3_y(B_{2R}),
\]
\[
U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n
\to0
\quad\text{in }L^1_\tau L^2_y(B_R),
\]
\[
\mathcal R_n
:=
\partial_\tau S_n-\nu\Delta S_n
+a_n(S_n+y\cdot\nabla S_n)
+b_n\cdot\nabla S_n
\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R),
\]
where \(|a_n|+|b_n|\le M_{ab}\), and assume the pressure cross-term satisfies
\[
\nabla\mathcal P_{\mathrm{cross},n}\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]

### Lemma S7.1

Under the preceding assumptions, replacing \(U_n+S_n\) by \(U_n\) changes the
repaired-gauge Navier-Stokes equation on \(Q_R\) by an error tending to zero in
\[
L^1_\tau H^{-1}_y(B_R).
\]

#### Proof

The linear part is exactly \(\mathcal R_n\), which tends to zero by
assumption.

The nonlinear difference is
\[
(U_n+S_n)\cdot\nabla(U_n+S_n)-U_n\cdot\nabla U_n
=
\operatorname{div}(U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n),
\]
using \(\nabla\cdot U_n=\nabla\cdot S_n=0\) distributionally. By the assumed
\(L^1_\tau L^2_y\) stress convergence, this divergence tends to zero in
\[
L^1_\tau H^{-1}_y(B_R),
\]
because for \(\varphi\in H^1_0(B_R;\mathbb R^3)\),
\[
\left|
\int_{B_R}
(U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n):\nabla\varphi\,dy
\right|
\le
\|U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n\|_{L^2(B_R)}
\|\nabla\varphi\|_{L^2(B_R)}.
\]
The pressure contribution vanishes by the assumed cross-pressure convergence.
Therefore the total perturbative error tends to zero in
\[
L^1_\tau H^{-1}_y(B_R).
\]

\(\square\)

---

## S7.2 (Static same-point outer profiles vanish locally)

Let
\[
W_n(y)=\mu_n\phi(\xi_n+\mu_n y),
\qquad
\mu_n\to0,
\qquad
\phi\in L^3(\mathbb R^3).
\]

### Lemma S7.2

For every fixed \(R<\infty\),
\[
\|W_n\|_{L^3(B_R)}\to0.
\]

#### Proof

Changing variables \(z=\xi_n+\mu_n y\),
\[
\|W_n\|_{L^3(B_R)}^3
=
\int_{B(\xi_n,\mu_nR)}|\phi(z)|^3\,dz.
\]
The measure of \(B(\xi_n,\mu_nR)\) tends to zero. Since
\(|\phi|^3\in L^1(\mathbb R^3)\), the absolute continuity of the integral
gives
\[
\int_{B(\xi_n,\mu_nR)}|\phi(z)|^3\,dz\to0,
\]
uniformly in the centers \(\xi_n\). Hence \(\|W_n\|_{L^3(B_R)}\to0\).

\(\square\)

This proves the local velocity part of same-point decoupling for all outer
scales, including finite-offset and escaping-offset cases. It does not by
itself control \(\partial_\tau W_n\), diffusion, modulation forcing, or
nonlocal pressure. Those are supplied by the nonlinear profile-evolution
payload below.

---

## S7.3 (Static separated-point profiles vanish locally)

Let \(x_p^*\ne x_q^*\), and let
\[
W_{q\to p}^n(y)
:=
\lambda_p^n(\lambda_q^n)^{-1}
\phi^q\left(
\frac{x_p^n+\lambda_p^n y-x_q^n}{\lambda_q^n}
\right),
\qquad
\phi^q\in L^3(\mathbb R^3).
\]

### Lemma S7.3

For every fixed \(R<\infty\),
\[
\|W_{q\to p}^n\|_{L^3(B_R)}\to0.
\]

#### Proof

The proof is the same escaping-ball calculation as in S6. With
\[
c_n=\frac{x_p^n-x_q^n}{\lambda_q^n},
\qquad
\rho_n=\frac{\lambda_p^nR}{\lambda_q^n},
\]
one has
\[
\|W_{q\to p}^n\|_{L^3(B_R)}^3
=
\int_{B(c_n,\rho_n)}|\phi^q(z)|^3\,dz.
\]
Since \(|x_p^*-x_q^*|>0\),
\[
|c_n|\to\infty,
\qquad
\rho_n/|c_n|\to0.
\]
Thus \(B(c_n,\rho_n)\) eventually lies outside every fixed ball. The
\(L^1\)-tail of \(|\phi^q|^3\) tends to zero, so the displayed integral tends
to zero.

\(\square\)

---

## S7.4 (Pressure decoupling from kernel tails)

Let \(K_{ij}\) denote the Calderon-Zygmund pressure kernel for
\[
P=\mathcal R_i\mathcal R_j(F_{ij}).
\]
For \(y\in B_R\), split a source \(F_n\) into a local part
\[
F_n^{\mathrm{loc}}:=F_n\mathbf 1_{B_A}
\]
and a far part
\[
F_n^{\mathrm{far}}:=F_n\mathbf 1_{\mathbb R^3\setminus B_A},
\]
with \(A>2R\).

### Lemma S7.4 (local \(H^{-1}\)-pressure decoupling criterion)

Assume:

1. \(F_n\to0\) in \(L^1_\tau L^2_y(B_A)\) for every fixed \(A\);
2. \(\sup_n\|F_n\|_{L^\infty_\tau L^{3/2}_y(\mathbb R^3)}<\infty\).

Then the pressure generated by \(F_n\) satisfies
\[
P_n-c_{n,R}\to0
\quad\text{in }L^1_\tau L^2_y(B_R)
\]
for suitable constants \(c_{n,R}(\tau)\), and therefore
\[
\nabla P_n\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]

#### Proof

For the local part, Calderon-Zygmund boundedness gives
\[
\|P_n^{\mathrm{loc}}\|_{L^1_\tau L^2_y(B_A)}
\le
C_A\|F_n\|_{L^1_\tau L^2_y(B_A)}
\to0.
\]

For the far part, define
\[
c_{n,R}(\tau):=
\int_{|z|>A}K_{ij}(-z)F_{n,ij}(z,\tau)\,dz
\]
whenever the integral is first taken for compactly supported approximations
and then passed to the \(L^{3/2}\) limit. For \(y\in B_R\) and \(|z|>A>2R\),
the kernel difference satisfies
\[
|K_{ij}(y-z)-K_{ij}(-z)|
\le
C_R |z|^{-4}.
\]
Hence
\[
|P_n^{\mathrm{far}}(y,\tau)-c_{n,R}(\tau)|
\le
C_R\int_{|z|>A}|z|^{-4}|F_n(z,\tau)|\,dz.
\]
By Hölder with exponents \(3/2\) and \(3\),
\[
\int_{|z|>A}|z|^{-4}|F_n|\,dz
\le
\|F_n(\tau)\|_{L^{3/2}}
\left(\int_{|z|>A}|z|^{-12}\,dz\right)^{1/3}
\le
C A^{-3}\|F_n(\tau)\|_{L^{3/2}}.
\]
Therefore
\[
\|P_n^{\mathrm{far}}-c_{n,R}\|_{L^1_\tau L^\infty_y(B_R)}
\le
C_{R,I}A^{-3}
\sup_n\|F_n\|_{L^\infty_\tau L^{3/2}_y}.
\]
Since \(B_R\) has finite measure, this estimate also controls the far part in
\[
L^1_\tau L^2_y(B_R).
\]
Taking \(n\to\infty\) in the local term and then \(A\to\infty\) in the far
term yields
\[
P_n-c_{n,R}\to0
\quad\text{in }L^1_\tau L^2_y(B_R).
\]
Since, for vector test functions
\(\varphi\in H^1_0(B_R;\mathbb R^3)\),
\[
|\langle\nabla(P_n-c_{n,R}),\varphi\rangle|
\le
\left|
\int_{B_R}(P_n-c_{n,R})\nabla\cdot\varphi\,dy
\right|
\le
\|P_n-c_{n,R}\|_{L^2(B_R)}
\|\nabla\varphi\|_{L^2(B_R)},
\]
the gradient converges to zero in \(L^1_\tau H^{-1}_y(B_R)\).

\(\square\)

---

## S7.5 (Nonlinear profile-evolution decoupling payload)

Define
\[
K_{\mathrm{NLProfDec},NS3D}^+
\]
to be the following profile-evolution theorem on every compact camera cylinder
\[
B_R\times I.
\]

For every finite active profile package and every terminal active camera,
after comparable profiles at the same physical point have been grouped into
compound profiles and after constant ambient velocities have been absorbed
into the translation gauge, the sum of all profiles not belonging to the
terminal camera plus the rescaled remainder is a perturbation \(S_n\)
satisfying:

1. divergence-free compatibility:
   \[
   \nabla\cdot S_n=0
   \quad\text{distributionally on every compact camera cylinder};
   \]
2. velocity decoupling:
   \[
   S_n\to0
   \quad\text{in }L^\infty_\tau L^3_y(B_{2R});
   \]
3. nonlinear stress decoupling:
   \[
   U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n
   \to0
   \quad\text{in }L^1_\tau L^2_y(B_R);
   \]
4. linear residual decoupling:
   \[
   \partial_\tau S_n-\nu\Delta S_n
   +a_n(S_n+y\cdot\nabla S_n)
   +b_n\cdot\nabla S_n
   \to0
   \quad\text{in }L^1_\tau H^{-1}_y(B_R);
   \]
5. pressure-source decoupling:
   every pressure source involving at least one factor of \(S_n\) satisfies
   the \(L^2\)-strength criterion of Lemma S7.4, or directly emits
   \[
   \nabla\mathcal P_{\mathrm{cross},n}\to0
   \quad\text{in }L^1_\tau H^{-1}_y(B_R);
   \]
6. localization-error decoupling:
   cutoff errors introduced in isolating the selected camera vanish in
   \(L^1_\tau H^{-1}_y(B_R)\);
7. modulation compatibility:
   the effective modulation parameters of the selected camera remain bounded;
8. S3 admissibility:
   after discarding \(S_n\), the selected camera branch has positive finite
   critical mass, repaired-gauge representation, pressure reconstruction,
   Caccioppoli regularity, compactness/tightness, and the S3 limiting inputs.

The static velocity convergence in item 2 follows from Lemma S7.2 for
same-point outer scales and Lemma S7.3 for separated physical points. The
remaining items are the nonlinear profile-evolution content. The terminal
camera quantifier is essential: a nonterminal same-point camera can still see
a smaller active profile and therefore need not satisfy this payload.

---

## S7.6 (Discharge of \(K_{\mathrm{SamePointNLDec}}^+\))

### Theorem S7.5

Assume \(K_{\mathrm{NLProfDec},NS3D}^+\). Then
\[
K_{\mathrm{SamePointNLDec}}^+
\]
holds.

#### Proof

Consider a same-point active profile package. Group comparable scales into
compound profiles as in S6.3. For strict scale separation, choose the
innermost active scale as the selected camera.

By Lemma S7.2, every outer scale is locally invisible in the innermost
velocity topology. By \(K_{\mathrm{NLProfDec},NS3D}^+\), the full perturbation
formed by all outer profiles and the rescaled remainder satisfies the
nonlinear-stress, linear-residual, pressure-source, localization, modulation,
and S3-admissibility requirements on every compact camera cylinder. Lemma S7.1
then shows that the outer profiles and remainder alter the selected camera
equation only by an error tending to zero in \(L^1_\tau H^{-1}_y(B_R)\).

This is exactly the definition of \(K_{\mathrm{SamePointNLDec}}^+\): the
same-point multibubble reduces, in the selected innermost or compound camera,
to a single S3-admissible branch with perturbative interactions.

\(\square\)

---

## S7.7 (Discharge of \(K_{\mathrm{MultiPointCamDec}}^+\))

### Theorem S7.6

Assume \(K_{\mathrm{NLProfDec},NS3D}^+\). Then
\[
K_{\mathrm{MultiPointCamDec}}^+
\]
holds.

#### Proof

Choose an active physical point \(x_p^*\) and its active camera. For every
profile centered at a different physical point \(x_q^*\ne x_p^*\), Lemma S7.3
gives local velocity convergence to zero in the \(p\)-camera. If the pure
profile pressure is represented in \(L^2\), the same escaping-ball calculation
gives pure pressure convergence in \(L^2_{\mathrm{loc}}\), hence gradient
convergence in \(H^{-1}_{\mathrm{loc}}\). With only \(L^{3/2}\)-pressure one
gets \(L^{3/2}_{\mathrm{loc}}\) convergence, which is not sufficient for the
Hilbert \(H^{-1}\) pressure topology used here. For mixed pressure sources,
\(K_{\mathrm{NLProfDec},NS3D}^+\) supplies the \(L^2\)-strength hypotheses of
Lemma S7.4 or a direct \(L^1_\tau H^{-1}_y\) pressure-gradient convergence
certificate.

The same payload supplies localization-error convergence, modulation
compatibility, and S3 admissibility for the selected point after all exterior
profiles are discarded. Lemma S7.1 then proves that separated physical-point
profiles change the selected camera equation only by an error tending to zero
in \(L^1_\tau H^{-1}_{\mathrm{loc}}\).

This is precisely \(K_{\mathrm{MultiPointCamDec}}^+\).

\(\square\)

---

## S7.8 (Full multibubble exclusion under nonlinear profile decoupling)

Combining Theorems S7.5 and S7.6 with S6 gives:
\[
K_{\mathrm{TechTypeII}}^+
\wedge
K_{\mathrm{NLProfDec},NS3D}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+
\Longrightarrow
\text{no active multibubble Type II candidate}.
\]

Thus the two payloads
\[
K_{\mathrm{SamePointNLDec}}^+,
\qquad
K_{\mathrm{MultiPointCamDec}}^+
\]
are discharged by the single nonlinear profile-decoupling theorem
\[
K_{\mathrm{NLProfDec},NS3D}^+.
\]

The remaining problem is therefore not a new multibubble mechanism. S8 proves
\(K_{\mathrm{NLProfDec},NS3D}^+\) in the terminal-camera sense from terminal
windowed local compactness, scattering removal, exterior-regular discard,
repaired-gauge representation, and Caccioppoli regularity. Hence any failure
of the S7 payload is an upstream profile-completeness, exterior-discard,
representation, Caccioppoli, or S3-rigidity defect.
