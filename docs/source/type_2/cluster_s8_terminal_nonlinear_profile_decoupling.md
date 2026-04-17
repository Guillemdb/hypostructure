# S8: terminal nonlinear profile decoupling

This note proves the nonlinear profile-decoupling payload used in S7 in the
only form in which it is correct: for terminal active cameras. A terminal
camera is attached to a minimal-scale active cluster after comparable profiles
at the same physical point have been grouped into one compound profile.

The output is
\[
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\wedge
K_{\mathrm{ScattBranch}}^-
\wedge
K_{\mathrm{ExtRegDiscard}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{NLProfDec},NS3D}^+ .
\]
Here
\[
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\]
denotes the terminal windowed nonlinear form of local compactness: the
profile extractor is valid on compact terminal-camera windows, all omitted
non-scattering profiles are emitted, and the scattering remainder is small in
the critical stability topology on those windows. If the backend's
\(K_{\mathrm{ProfComplete},NS3D}^+\) already includes this windowed nonlinear
content, the two notations are identified.

The certificate \(K_{\mathrm{NLProfDec},NS3D}^+\) is understood in the
terminal-camera sense. The stronger assertion for arbitrary nonterminal
cameras is false, because a larger-scale camera can still see a smaller
same-point bubble.

---

## S8.1 (Critical profile package and active profiles)

Let \(u\) be a declared NS3D Type II candidate satisfying
\[
\sup_{t<T^*}\|u(t)\|_{X_{\mathrm{crit}}}\le M<\infty .
\]
Let \(t_n\uparrow T^*\) be a concentrating sequence past the scattering
branch. The terminal nonlinear profile-completeness payload
\[
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\]
provides a Navier-Stokes critical profile package
\[
u(\cdot,t_n)
=
\sum_{j\in J}
\Lambda_{\lambda_j^n,x_j^n}\phi^j+r_n^J
\]
for every finite set \(J\), where
\[
(\Lambda_{\lambda,x_0}\phi)(x)
=
\lambda^{-1}\phi\left(\frac{x-x_0}{\lambda}\right),
\qquad
\nabla\cdot\phi^j=0.
\]
The profile parameters are pairwise orthogonal after comparable parameters
have been grouped, and the Brézis-Lieb critical mass decoupling holds:
\[
\|u(t_n)\|_3^3
=
\sum_{j\in J}\|\phi^j\|_3^3+\|r_n^J\|_3^3+o_n(1).
\]

The small-data scattering threshold is denoted by
\[
\varepsilon_{\mathrm{sd}}>0.
\]
A profile is active if its nonlinear Navier-Stokes evolution is not removed by
the small-data stability/scattering ledger. Since we are past the scattering
branch, at least one active profile exists, and every active profile satisfies
\[
\|\phi^j\|_3\ge\varepsilon_{\mathrm{sd}}.
\]

### Lemma S8.1 (finite active profile count)

The active index set is finite and satisfies
\[
\#J_{\mathrm{act}}
\le
\left\lfloor
\frac{M^3}{\varepsilon_{\mathrm{sd}}^3}
\right\rfloor .
\]

#### Proof

Let \(J\subset J_{\mathrm{act}}\) be finite. By critical mass decoupling,
\[
M^3
\ge
\limsup_{n\to\infty}\|u(t_n)\|_3^3
\ge
\sum_{j\in J}\|\phi^j\|_3^3.
\]
Since \(\|\phi^j\|_3\ge\varepsilon_{\mathrm{sd}}\) for active profiles,
\[
M^3\ge (\#J)\varepsilon_{\mathrm{sd}}^3.
\]
The inequality holds for every finite active subset \(J\). Hence the active
set is finite and has the asserted cardinality bound.

\(\square\)

---

## S8.2 (Terminal active cameras)

Partition active profiles by physical concentration point:
\[
x_j^n\to x_\alpha^*.
\]
At a fixed physical point \(x_\alpha^*\), group profiles with comparable
scales:
\[
\frac{\lambda_i^n}{\lambda_j^n}\to\rho_{ij}\in(0,\infty).
\]
Each comparable-scale group is one compound profile. If the group is observed
in a representative scale \(\lambda_\mathcal C^n\) and center
\(x_\mathcal C^n\), its camera profile is
\[
\Phi_\mathcal C(y)
=
\sum_{j\in\mathcal C}
\rho_j^{-1}\phi^j\left(\frac{y-z_j}{\rho_j}\right),
\]
with the notation of S6.

A compound active cluster \(\mathcal C\) is terminal at \(x_\alpha^*\) if no
active cluster at the same physical point has strictly smaller scale:
\[
\lambda_{\mathcal D}^n/\lambda_{\mathcal C}^n\to0.
\]
Equivalently, for every other active cluster \(\mathcal D\) at the same
physical point, either \(\mathcal D=\mathcal C\) after grouping, or
\[
\mu_{\mathcal D}^n
:=
\lambda_{\mathcal C}^n/\lambda_{\mathcal D}^n
\to0.
\]

The terminal camera is
\[
y=\frac{x-x_\mathcal C^n}{\lambda_\mathcal C^n},
\qquad
\tau=\int^t(\lambda_\mathcal C(s))^{-2}\,ds,
\]
and
\[
V_n(y,\tau)
=
\lambda_\mathcal C^n
u(x_\mathcal C^n+\lambda_\mathcal C^n y,
t_n+(\lambda_\mathcal C^n)^2\tau).
\]
The retained branch \(U_n\) is the repaired-gauge nonlinear evolution of the
compound cluster \(\mathcal C\) in this camera. The discarded field is
\[
S_n:=V_n-U_n.
\]

The proof of \(K_{\mathrm{NLProfDec},NS3D}^+\) is the proof that \(S_n\) is
perturbative on every compact terminal-camera cylinder
\[
Q_R:=B_R\times I.
\]

---

## S8.3 (Local compactness excludes hidden terminal-window mass)

For a sequence \(f_n\) and a camera \((x_n,\lambda_n)\), define the local
critical mass observed in \(B_R\) by
\[
\mathfrak m_R(f_n;x_n,\lambda_n)
:=
\limsup_{n\to\infty}
\left\|
\lambda_n f_n(x_n+\lambda_n\cdot)
\right\|_{L^3(B_R)}.
\]
The terminal-window version is
\[
\mathfrak M_{R,I}(f_n;x_n,\lambda_n)
:=
\limsup_{n\to\infty}
\sup_{\tau\in I}
\left\|
\lambda_n f_n(x_n+\lambda_n\cdot,t_n+\lambda_n^2\tau)
\right\|_{L^3(B_R)}.
\]

The profile-completeness payload means the following: if
\[
\mathfrak M_{R,I}(f_n;x_n,\lambda_n)>0
\]
for some sequence left in the remainder ledger, then, after passing to a
subsequence, the profile extractor emits a nonzero profile with physical
center \(x_n+O(\lambda_n)\) and scale comparable to \(\lambda_n\). If the
emitted nonlinear profile is above the small-data threshold, it is active. If
it is below the threshold, it belongs to the scattering ledger, whose critical
stability norm can be made arbitrarily small before the limit \(n\to\infty\)
is taken.

This is the terminal windowed form of
\(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\).
If the C1 profile-completeness payload is stated only for single time slices,
this windowed statement follows by contradiction: a failure on \(I\) supplies
times \(\tau_n\in I\) with nonvanishing local critical mass, and applying the
time-slice extractor to the sequence
\[
u(\cdot,t_n+\lambda_n^2\tau_n)
\]
emits the omitted profile. The nonlinear stability part of the scattering
ledger controls subthreshold profiles uniformly on compact windows. Therefore
this step uses either the terminal windowed profile-completeness certificate
directly, or the conjunction of time-slice local compactness with nonlinear
profile stability of the scattering remainder.

### Lemma S8.2 (terminal local velocity decoupling)

Let \(\mathcal C\) be a terminal active cluster, and let \(S_n\) be the sum of
all discarded active profiles plus the scattering remainder in the
\(\mathcal C\)-camera. Then, for every compact terminal-camera cylinder
\[
B_R\times I,
\]
one has
\[
S_n\to0
\quad\text{in }L^\infty_\tau L^3_y(B_R).
\]

#### Proof

Assume the conclusion fails. Then there exist \(R<\infty\), a compact interval
\(I\), a number \(\delta>0\), a subsequence, and times
\(\tau_n\in I\) such that
\[
\|S_n(\tau_n)\|_{L^3(B_R)}\ge\delta.
\]
In physical variables this says that, after the retained terminal cluster has
been removed, the remainder carries at least \(\delta\) critical \(L^3\)-mass
inside a ball of radius \(O(\lambda_\mathcal C^n)\) centered at
\[
x_\mathcal C^n+O(\lambda_\mathcal C^n)
\]
at the time
\[
t_n+(\lambda_\mathcal C^n)^2\tau_n.
\]
By local terminal compactness, this nonvanishing mass emits a profile whose
physical center is the same point \(x_\alpha^*\) and whose scale is comparable
to \(\lambda_\mathcal C^n\).

If the emitted profile is active, then it belongs to the same comparable-scale
cluster as \(\mathcal C\). But all comparable active profiles at that physical
point were grouped into \(\mathcal C\), so it cannot remain in \(S_n\). This
contradicts the definition of the retained compound branch \(U_n\).

If the emitted profile is below the small-data threshold, it is part of the
scattering ledger. The small-data nonlinear stability norm of the scattering
ledger can be chosen smaller than \(\delta/2\) before passing to the
subsequence. Such a component cannot contribute \(\delta\) to
\(\|S_n(\tau_n)\|_{L^3(B_R)}\). This is again a contradiction.

Therefore no such \(\delta\) exists, and
\[
S_n\to0
\quad\text{in }L^\infty_\tau L^3_y(B_R).
\]

\(\square\)

The same argument covers same-point larger-scale profiles and separated-point
profiles. In the same-point larger-scale case, any nonzero mass seen in the
terminal camera would generate an omitted profile at the terminal scale. In
the separated-point case, S6 already gives local escape in the selected
camera; a failure of escape would again produce a profile at the selected
physical point, contradicting the partition by physical centers.

---

## S8.4 (Uniform local \(H^1\) bounds)

The repaired-gauge representation and Caccioppoli payloads give, for every
compact terminal-camera cylinder \(B_{2R}\times I\),
\[
\sup_n
\|V_n\|_{L^\infty_\tau L^3_y(B_{2R})}
+
\sup_n
\|V_n\|_{L^2_\tau H^1_y(B_{2R})}
\le C_R.
\]
The retained branch \(U_n\) is itself a repaired-gauge nonlinear NS branch
with positive finite critical mass and bounded modulation on the same compact
cylinder. This is part of the terminal nonlinear profile-completeness and
representation payload. The same Caccioppoli estimate gives
\[
\sup_n
\|U_n\|_{L^\infty_\tau L^3_y(B_{2R})}
+
\sup_n
\|U_n\|_{L^2_\tau H^1_y(B_{2R})}
\le C_R.
\]
Since \(S_n=V_n-U_n\),
\[
\sup_n
\|S_n\|_{L^\infty_\tau L^3_y(B_{2R})}
+
\sup_n
\|S_n\|_{L^2_\tau H^1_y(B_{2R})}
\le C_R.
\]
In particular, Sobolev on \(B_{2R}\) gives
\[
U_n,S_n
\quad\text{bounded in}\quad
L^2_\tau L^6_y(B_{2R}).
\]

---

## S8.5 (Nonlinear stress decoupling)

Define the cross stress
\[
F_n
:=
U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n.
\]

### Lemma S8.3

For every \(R<\infty\) and compact \(I\),
\[
F_n\to0
\quad\text{in }L^1_\tau L^2_y(B_R).
\]

#### Proof

By Lemma S8.2,
\[
\|S_n\|_{L^\infty_\tau L^3_y(B_R)}\to0.
\]
Using Hölder in space and then Hölder in time,
\[
\|U_n\otimes S_n\|_{L^1_\tau L^2_y(B_R)}
\le
\|S_n\|_{L^\infty_\tau L^3_y(B_R)}
\|U_n\|_{L^1_\tau L^6_y(B_R)}
\]
and
\[
\|U_n\|_{L^1_\tau L^6_y(B_R)}
\le
|I|^{1/2}
\|U_n\|_{L^2_\tau L^6_y(B_R)}
\le C_{R,I}.
\]
Thus
\[
\|U_n\otimes S_n\|_{L^1_\tau L^2_y(B_R)}\to0.
\]
The same estimate gives
\[
\|S_n\otimes U_n\|_{L^1_\tau L^2_y(B_R)}\to0.
\]
Finally,
\[
\|S_n\otimes S_n\|_{L^1_\tau L^2_y(B_R)}
\le
\|S_n\|_{L^\infty_\tau L^3_y(B_R)}
\|S_n\|_{L^1_\tau L^6_y(B_R)}
\to0,
\]
because \(\|S_n\|_{L^1_\tau L^6_y(B_R)}\le C_{R,I}\). Summing the three
terms proves the claim.

\(\square\)

Since \(U_n\), \(S_n\), and \(V_n\) are uniformly bounded in global
\(L^\infty_\tau L^3_y\), one also has
\[
\sup_n\|F_n\|_{L^\infty_\tau L^{3/2}_y(\mathbb R^3)}<\infty.
\]

---

## S8.6 (Pressure decoupling)

Let \(P_{\mathrm{cross},n}\) be the pressure generated by the cross stress:
\[
-\Delta P_{\mathrm{cross},n}
=
\partial_i\partial_j(F_n)_{ij}.
\]

### Lemma S8.4

For every \(R<\infty\), there are constants \(c_{n,R}(\tau)\) such that
\[
P_{\mathrm{cross},n}-c_{n,R}
\to0
\quad\text{in }L^1_\tau L^2_y(B_R),
\]
and therefore
\[
\nabla P_{\mathrm{cross},n}
\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]

#### Proof

Lemma S8.3 gives
\[
F_n\to0
\quad\text{in }L^1_\tau L^2_y(B_A)
\]
for every fixed \(A\). The retained local compact-cylinder bound gives
\[
\sup_n\|F_n\|_{L^\infty_\tau L^{3/2}_y(\mathbb R^3)}<\infty.
\]
The hypotheses of the S7 pressure kernel-tail lemma are therefore satisfied.
That lemma gives the asserted local \(L^2\) pressure convergence modulo
constants and the \(L^1_\tau H^{-1}_y(B_R)\) convergence of the pressure
gradient.

\(\square\)

---

## S8.7 (Linear residual decoupling)

Let \(V_n=U_n+S_n\) solve the repaired-gauge equation
\[
\partial_\tau V_n+(V_n\cdot\nabla)V_n+\nabla P_n
=
\nu\Delta V_n
+a_n(V_n+y\cdot\nabla V_n)
+b_n\cdot\nabla V_n.
\]
The retained compound branch \(U_n\) is the nonlinear evolution of the
terminal cluster in the same repaired gauge. After the constant ambient drift
has been absorbed into the translation parameter, it satisfies
\[
\partial_\tau U_n+(U_n\cdot\nabla)U_n+\nabla P_{U,n}
=
\nu\Delta U_n
+a_n(U_n+y\cdot\nabla U_n)
+b_n\cdot\nabla U_n
+e_n,
\]
where
\[
e_n\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R)
\]
on every compact terminal-camera cylinder. The error \(e_n\) contains only
the vanishing gauge-normalization, finite truncation, and localization
commutators supplied by the repaired-gauge representation and terminal
nonlinear profile-stability payloads. This is the equation-compatibility part
of \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\), not a consequence of a
static one-time profile decomposition alone.

Subtracting the \(U_n\)-equation from the \(V_n\)-equation gives
\[
\partial_\tau S_n-\nu\Delta S_n
=
-\operatorname{div}F_n
-\nabla P_{\mathrm{cross},n}
-e_n
+a_n(S_n+y\cdot\nabla S_n)
+b_n\cdot\nabla S_n.
\]
Equivalently,
\[
\partial_\tau S_n-\nu\Delta S_n
-a_n(S_n+y\cdot\nabla S_n)
-b_n\cdot\nabla S_n
=
-\operatorname{div}F_n
-\nabla P_{\mathrm{cross},n}
-e_n.
\]
By Lemma S8.3,
\[
F_n\to0
\quad\text{in }L^1_\tau L^2_y(B_R),
\]
hence
\[
\operatorname{div}F_n\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]
By Lemma S8.4,
\[
\nabla P_{\mathrm{cross},n}\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]
Together with \(e_n\to0\), this proves
\[
\partial_\tau S_n-\nu\Delta S_n
-a_n(S_n+y\cdot\nabla S_n)
-b_n\cdot\nabla S_n
\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]
The S7 definition uses the same residual with the opposite sign convention for
the modulation terms. The two residuals differ by
\[
2a_n(S_n+y\cdot\nabla S_n)+2b_n\cdot\nabla S_n.
\]
This difference tends to zero in \(L^1_\tau H^{-1}_y(B_R)\). Indeed, by
Lemma S8.2 and finite measure,
\[
S_n\to0
\quad\text{in }L^\infty_\tau L^2_y(B_R).
\]
For \(\varphi\in H^1_0(B_R)\),
\[
|\langle y\cdot\nabla S_n,\varphi\rangle|
=
\left|
\int_{B_R}S_n(3\varphi+y\cdot\nabla\varphi)\,dy
\right|
\le
C_R\|S_n\|_{L^2(B_R)}\|\varphi\|_{H^1(B_R)},
\]
and
\[
|\langle b_n\cdot\nabla S_n,\varphi\rangle|
\le
|b_n|\|S_n\|_{L^2(B_R)}\|\nabla\varphi\|_{L^2(B_R)}.
\]
The term \(S_n\) itself is controlled by the same local \(L^2\) convergence.
Since \(a_n\) and \(b_n\) are bounded, the S7 residual also tends to zero in
\[
L^1_\tau H^{-1}_y(B_R).
\]

---

## S8.8 (Modulation compatibility)

Same-point larger-scale profiles in a terminal camera have the form
\[
W_j^n(y,\tau)
=
\mu_j^n
U^j\!\left(\theta_j^n(\tau),\xi_j^n+\mu_j^n y\right),
\qquad
\mu_j^n\to0.
\]
When the profile is regular at the observed offset, its leading term is a
spatially constant ambient velocity. Such constants are absorbed by replacing
\[
b_n
\quad\text{with}\quad
b_n-b_{\mathrm{amb},n}.
\]
The remaining terms vanish on compact terminal-camera cylinders by Lemma
S8.2 and the \(H^{-1}\) product estimates above. If the offset escapes, the
entire profile is locally invisible and no gauge absorption is needed.

Separated physical-point profiles are locally invisible in the selected
camera by S6 and Lemma S8.2. Their pressures and cutoff commutators vanish by
the pressure and residual arguments above.

Thus the effective terminal-camera modulation parameters remain bounded:
\[
\sup_n\sup_{\tau\in I}
\left(|a_n(\tau)|+|b_n(\tau)|\right)
<\infty .
\]

---

## S8.9 (S3 admissibility of terminal cameras)

The terminal cluster is active, so its retained branch has positive critical
mass. More precisely, after the small scattering remainder and discarded
profiles have been removed,
\[
\inf_{\tau\in I}\|U_n(\tau)\|_{L^3(B_R)}
\ge
c_{\mathrm{act}}>0
\]
on the S3 windows, after passing to the same subsequences used in the profile
ledger. The upper bound follows from the retained local compact-cylinder bound:
\[
\sup_{\tau\in I}\|U_n(\tau)\|_{L^3(B_R)}
\le C(M).
\]
The terminal-complete profile ledger also supplies the tightness needed by
S3. If \(U_n\) were not uniformly \(L^3\)-tight on the selected windows, then
there would be \(\varepsilon>0\), radii \(R_k\to\infty\), and a subsequence of
times for which
\[
\int_{|y|>R_k}|U_n(y,\tau_n)|^3\,dy\ge\varepsilon.
\]
Applying the profile extractor to this escaping sequence produces either an
exterior-regular component, which is removed by the S4 exterior-discard
ledger, or an additional active profile at a different physical point or
larger same-point scale, which belongs to the discarded field \(S_n\) and is
handled by terminal decoupling. Hence no untight terminal branch remains in
the post-S4 active ledger.

The repaired-gauge representation gives pressure reconstruction and bounded
modulation, and the Caccioppoli bridge gives the local windowed \(H^1\)
control. The T7/Aubin-Lions compactness layer gives the compactness input used
by S3 on the selected windows. Therefore the retained terminal branch is
S3-admissible.

---

## S8.10 (Terminal nonlinear profile-decoupling theorem)

### Theorem S8.5

Assume:
\[
\sup_{t<T^*}\|u(t)\|_{X_{\mathrm{crit}}}\le M<\infty,
\]
\[
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+,
\qquad
K_{\mathrm{ScattBranch}}^-,
\qquad
K_{\mathrm{ExtRegDiscard}}^+,
\qquad
K_{\mathrm{RepBridge}}^+,
\qquad
K_{\mathrm{CaccioppoliReg}}^+.
\]
Then the terminal-camera nonlinear profile-decoupling payload holds:
\[
K_{\mathrm{NLProfDec},NS3D}^+.
\]

#### Proof

Let \(\mathcal C\) be any terminal active compound cluster, and let \(U_n\) be
the retained repaired-gauge nonlinear branch in its terminal camera. Let
\[
S_n:=V_n-U_n
\]
be the discarded field.

Lemma S8.2 proves
\[
S_n\to0
\quad\text{in }L^\infty_\tau L^3_y(B_{2R})
\]
on every compact terminal-camera cylinder. The divergence-free compatibility
follows from the divergence-free profile package and from the divergence-free
Navier-Stokes evolution of each nonlinear profile.

Lemma S8.3 proves the nonlinear stress convergence
\[
U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n
\to0
\quad\text{in }L^1_\tau L^2_y(B_R).
\]
Lemma S8.4 proves the pressure-source decoupling:
\[
\nabla P_{\mathrm{cross},n}
\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R),
\]
after subtracting harmless local pressure constants.

Section S8.7 proves the linear residual convergence
\[
\partial_\tau S_n-\nu\Delta S_n
+a_n(S_n+y\cdot\nabla S_n)
+b_n\cdot\nabla S_n
\to0
\quad\text{in }L^1_\tau H^{-1}_y(B_R).
\]
Section S8.8 gives modulation compatibility, and Section S8.9 gives S3
admissibility of the retained terminal branch.

These are exactly the items in the S7 definition of
\[
K_{\mathrm{NLProfDec},NS3D}^+
\]
with the corrected terminal-camera quantifier. Hence
\[
K_{\mathrm{NLProfDec},NS3D}^+
\]
holds.

\(\square\)

---

## S8.11 (Consequence for multibubbles)

Combining Theorem S8.5 with S7 gives
\[
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\wedge
K_{\mathrm{ScattBranch}}^-
\wedge
K_{\mathrm{ExtRegDiscard}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{MultiPointCamDec}}^+ .
\]
Combining this with S6 and the S3 rigidity payload gives
\[
K_{\mathrm{TechTypeII}}^+
\wedge
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\wedge
K_{\mathrm{ScattBranch}}^-
\wedge
K_{\mathrm{ExtRegDiscard}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+
\Longrightarrow
\text{no active multibubble Type II candidate}.
\]

Thus multibubble failure is no longer a separate singularity mechanism inside
the declared terminal-complete backend. It can occur only if profile
completeness, scattering removal, exterior-regular discard, repaired-gauge
representation, Caccioppoli regularity, or the S3 rigidity payload fails.
