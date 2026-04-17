# S6: multibubble camera reduction

This note records the multibubble reduction obtained by using one
renormalized camera for each active concentration structure. The conclusion is
conditional and precise:

1. bounded critical \(L^3\) mass gives only finitely many active bubbles;
2. comparable same-point bubbles are one compound profile in a common camera;
3. strict same-point cascades are handled by the S5 innermost-camera argument;
4. separated physical points are locally invisible in one another's cameras;
5. after the same-point nonlinear decoupling and multi-point camera-decoupling
   payloads are supplied, every multibubble candidate reduces to a single
   S3/NRŠ camera contradiction.

Thus the remaining multibubble obstruction before S8 is exactly the failure
of terminal nonlinear profile decoupling:
\[
K_{\mathrm{SamePointNLDec}}^-
\quad\text{or}\quad
K_{\mathrm{MultiPointCamDec}}^-.
\]

---

## S6.1 (Renormalized cameras)

A camera is a choice of center \(x_c(t)\), scale \(\lambda(t)>0\), and
renormalized time
\[
\frac{d\tau}{dt}=\lambda(t)^{-2}.
\]
The camera observes the physical solution through
\[
V(y,\tau)
=
\lambda(t)u(x_c(t)+\lambda(t)y,t).
\]
If the repaired-gauge conditions are imposed on \(V\), then \(V\) solves
\[
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0.
\]
Here
\[
a(\tau)=\partial_\tau\log\lambda(\tau),
\]
and \(b\) is the translation modulation parameter determined by the chosen
centering gauge.

The point of the camera is structural: once a camera is fixed, every other
profile either:

1. remains at bounded position and comparable scale, hence belongs to the same
   compound profile;
2. moves to infinity in camera variables, hence is a radiative or multipoint
   term;
3. has much larger scale at the same point, hence appears as a perturbative
   ambient field in the innermost camera;
4. has much smaller scale, in which case the chosen camera was not the
   innermost active camera.

---

## S6.2 (Active multibubble profile package)

Let \(t_n\uparrow T^*\). An active multibubble profile package consists of a
finite or countable set of divergence-free profiles
\[
\phi^j\in L^3(\mathbb R^3),
\]
scales \(\lambda_j^n\to0\), centers \(x_j^n\to x_j^*\), and finite truncation
remainders \(r_n^J\) such that, for every finite index set \(J\),
\[
u(\cdot,t_n)
=
\sum_{j\in J}
\Lambda_{\lambda_j^n,x_j^n}\phi^j+r_n^J,
\]
where
\[
(\Lambda_{\lambda,x_0}\phi)(x)
=
\lambda^{-1}\phi\left(\frac{x-x_0}{\lambda}\right).
\]
The package has:

1. pairwise profile orthogonality after comparable parameters have been
   grouped into compound profiles;
2. critical \(L^3\) decoupling:
   \[
   \|u(t_n)\|_3^3
   =
   \sum_{j\in J}\|\phi^j\|_3^3+\|r_n^J\|_3^3+o_n(1)
   \]
   for every finite \(J\);
3. active-bubble mass floor:
   \[
   \|\phi^j\|_3\ge\varepsilon_{\mathrm{sd}}>0
   \]
   for every active singular profile.

The small-data threshold \(\varepsilon_{\mathrm{sd}}\) is the critical
\(L^3\) threshold below which the corresponding nonlinear Navier-Stokes
profile is perturbative. Profiles below this threshold are not active singular
bubbles and are assigned to the regular remainder ledger.

### Lemma S6.1 (finite active bubble count)

Assume
\[
\sup_{t<T^*}\|u(t)\|_3\le M<\infty .
\]
Then the set of active bubbles is finite and has cardinality
\[
\#\mathcal J\le
\left\lfloor
\frac{M^3}{\varepsilon_{\mathrm{sd}}^3}
\right\rfloor .
\]

#### Proof

Let \(J\) be any finite set of active indices. By critical \(L^3\) decoupling,
\[
\|u(t_n)\|_3^3
=
\sum_{j\in J}\|\phi^j\|_3^3+\|r_n^J\|_3^3+o_n(1).
\]
Taking the limsup and using \(\|r_n^J\|_3^3\ge0\),
\[
M^3
\ge
\sum_{j\in J}\|\phi^j\|_3^3
\ge
(\#J)\varepsilon_{\mathrm{sd}}^3.
\]
Therefore
\[
\#J\le M^3/\varepsilon_{\mathrm{sd}}^3
\]
for every finite active \(J\). If the active index set had more than
\(\lfloor M^3/\varepsilon_{\mathrm{sd}}^3\rfloor\) elements, choosing a finite
subset with one more element would contradict the preceding inequality. This
proves the stated integer bound.

\(\square\)

---

## S6.3 (Comparable same-point profiles are compound profiles)

Let a set \(\mathcal C\) of active profiles have the same physical center
\[
x_j^*=x_*
\quad (j\in\mathcal C),
\]
and let their scales be mutually comparable:
\[
\frac{\lambda_j^n}{\lambda_*^n}\to\rho_j\in(0,\infty).
\]
Assume also that their relative centers converge:
\[
\frac{x_j^n-x_*^n}{\lambda_*^n}\to z_j\in\mathbb R^3 .
\]
In the camera \((x_*^n,\lambda_*^n)\), the \(j\)-th profile becomes
\[
\lambda_*^n
(\lambda_j^n)^{-1}
\phi^j
\left(
\frac{x_*^n+\lambda_*^n y-x_j^n}{\lambda_j^n}
\right)
\longrightarrow
\rho_j^{-1}
\phi^j\left(\frac{y-z_j}{\rho_j}\right)
\]
strongly in \(L^3_{\mathrm{loc}}\), and globally in \(L^3\) if the profile
convergence is global.

To justify this convergence, set
\[
\rho_j^n:=\frac{\lambda_j^n}{\lambda_*^n},
\qquad
z_j^n:=\frac{x_j^n-x_*^n}{\lambda_*^n}.
\]
Then the camera expression is
\[
T_n\phi^j(y)
:=
(\rho_j^n)^{-1}
\phi^j\left(\frac{y-z_j^n}{\rho_j^n}\right),
\]
and the proposed limit is
\[
T\phi^j(y)
:=
\rho_j^{-1}
\phi^j\left(\frac{y-z_j}{\rho_j}\right).
\]
The map
\[
T_{\rho,z}\phi(y):=\rho^{-1}\phi\left(\frac{y-z}{\rho}\right)
\]
is strongly continuous on \(L^3(\mathbb R^3)\) for
\(\rho\in(0,\infty)\), \(z\in\mathbb R^3\). For
\(\phi\in C_c^\infty\), the convergence
\[
T_{\rho_j^n,z_j^n}\phi\to T_{\rho_j,z_j}\phi
\]
is uniform and the supports lie in one compact set for all large \(n\), hence
it is convergence in \(L^3\). For a general \(\phi\in L^3\), choose
\(\psi\in C_c^\infty\) with \(\|\phi-\psi\|_3<\varepsilon\). Since every
\(T_{\rho,z}\) is an \(L^3\)-isometry,
\[
\|T_{\rho_j^n,z_j^n}\phi-T_{\rho_j,z_j}\phi\|_3
\le
2\varepsilon
+
\|T_{\rho_j^n,z_j^n}\psi-T_{\rho_j,z_j}\psi\|_3.
\]
Letting \(n\to\infty\) and then \(\varepsilon\downarrow0\) proves global
\(L^3\) convergence. Local \(L^3\) convergence is immediate.

Define the compound profile
\[
\Phi_{\mathcal C}(y)
:=
\sum_{j\in\mathcal C}
\rho_j^{-1}
\phi^j\left(\frac{y-z_j}{\rho_j}\right).
\]
If
\[
\|\Phi_{\mathcal C}\|_3\ge\varepsilon_{\mathrm{sd}},
\]
then \(\mathcal C\) is a single active bubble for camera purposes. If
\[
\|\Phi_{\mathcal C}\|_3<\varepsilon_{\mathrm{sd}},
\]
then the divergence-free compound profile is perturbative and is removed from
the active singular ledger by small-data stability.

Thus comparable same-point bubbles are not a separate multibubble obstruction.
They are either one compound single-bubble profile or a perturbative profile.

---

## S6.4 (Strict same-point scale separation)

Suppose the active profiles have one physical center \(x_*\) and distinct
scale classes. Let \(\lambda_1^n\) be the smallest active scale. Then for every
outer scale \(\lambda_j^n\gg\lambda_1^n\),
\[
\mu_j^n:=\lambda_1^n/\lambda_j^n\to0.
\]
In the innermost camera, the \(j\)-th outer profile is
\[
W_j^n(y)
=
\mu_j^n
\phi^j\left(
\frac{x_*^n-x_j^n}{\lambda_j^n}
+\mu_j^n y
\right).
\]

S5 proves:

1. finite-offset regular outer profiles are constant drifts plus
   \(O((\mu_j^n)^2)\) shear on fixed inner balls;
2. the constant drifts are absorbed into the translation modulation parameter;
3. escaping-offset profiles are locally \(L^3\)-invisible in the innermost
   variables;
4. under the pressure and nonlinear decoupling payloads, the remaining
   interaction is perturbative in the S3 topology.

Consequently, under
\[
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{InnerRepBridge}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+,
\]
S5 gives
\[
\text{no same-point multibubble cascade}.
\]

The only same-point obstruction left without attacking no-splitting is
\[
K_{\mathrm{SamePointNLDec}}^-.
\]

---

## S6.5 (Separated physical points are locally invisible in one camera)

Let \(p\ne q\) be two physical concentration points:
\[
x_p^*\ne x_q^*,
\qquad
d_{pq}:=|x_p^*-x_q^*|>0.
\]
Choose the \(p\)-camera
\[
y=\frac{x-x_p^n}{\lambda_p^n}.
\]
Let the \(q\)-profile contribution in the \(p\)-camera be
\[
W_{q\to p}^n(y)
:=
\lambda_p^n(\lambda_q^n)^{-1}
\phi^q
\left(
\frac{x_p^n+\lambda_p^n y-x_q^n}{\lambda_q^n}
\right).
\]

### Lemma S6.2 (separated bubbles vanish locally in another camera)

Assume \(\phi^q\in L^3(\mathbb R^3)\). Then, for every fixed
\(R<\infty\),
\[
\|W_{q\to p}^n\|_{L^3(B_R)}\to0.
\]

#### Proof

By the change of variables \(x=x_p^n+\lambda_p^n y\),
\[
\|W_{q\to p}^n\|_{L^3(B_R)}^3
=
\int_{B(x_p^n,\lambda_p^nR)}
(\lambda_q^n)^{-3}
\left|
\phi^q\left(\frac{x-x_q^n}{\lambda_q^n}\right)
\right|^3\,dx .
\]
Set \(z=(x-x_q^n)/\lambda_q^n\). Then
\[
\|W_{q\to p}^n\|_{L^3(B_R)}^3
=
\int_{B(c_n,\rho_n)}
|\phi^q(z)|^3\,dz,
\]
where
\[
c_n:=\frac{x_p^n-x_q^n}{\lambda_q^n},
\qquad
\rho_n:=\frac{\lambda_p^nR}{\lambda_q^n}.
\]
The center satisfies
\[
|c_n|\sim d_{pq}/\lambda_q^n\to\infty.
\]
Moreover
\[
\frac{\rho_n}{|c_n|}
=
\frac{\lambda_p^nR}{|x_p^n-x_q^n|}
\to0.
\]
Therefore the balls \(B(c_n,\rho_n)\) escape to spatial infinity: for every
\(A<\infty\), they are contained in \(\{|z|>A\}\) for all sufficiently large
\(n\). Since \(\phi^q\in L^3\),
\[
\int_{|z|>A}|\phi^q(z)|^3\,dz\to0
\quad\text{as }A\to\infty.
\]
Hence
\[
\|W_{q\to p}^n\|_{L^3(B_R)}^3\to0.
\]

\(\square\)

The corresponding pure \(q\)-profile pressure contribution is controlled by
the same argument if the profile pressure \(\Pi^q\) satisfies
\[
\Pi^q\in L^{3/2}(\mathbb R^3).
\]
Indeed, in the \(p\)-camera the pure \(q\)-profile pressure has the form
\[
P_{q\to p}^n(y)
=
(\lambda_p^n)^2(\lambda_q^n)^{-2}
\Pi^q\left(
\frac{x_p^n+\lambda_p^n y-x_q^n}{\lambda_q^n}
\right).
\]
Therefore
\[
\|P_{q\to p}^n\|_{L^{3/2}(B_R)}^{3/2}
=
\int_{B(c_n,\rho_n)}|\Pi^q(z)|^{3/2}\,dz,
\]
with the same \(c_n\) and \(\rho_n\) as above. Since these balls escape to
spatial infinity and \(\Pi^q\in L^{3/2}\),
\[
\|P_{q\to p}^n\|_{L^{3/2}(B_R)}\to0.
\]
For pressure cross-terms and cut-off localization errors, define the
multi-point camera-decoupling payload
\[
K_{\mathrm{MultiPointCamDec}}^+.
\]
It consists of:

1. local velocity decoupling in each active point's camera;
2. local pressure decoupling modulo constants in the T1--T6/S3 topology;
3. localization errors from spatial cutoffs vanish in \(L^1_\tau H^{-1}_y\);
4. the repaired-gauge modulation parameters in each camera remain bounded
   after exterior ambient terms are absorbed;
5. every active point has positive finite critical mass in its own camera;
6. after profiles centered at other physical points are discarded as
   perturbative exterior terms, the selected camera branch satisfies the S3
   admissibility package: repaired-gauge representation, pressure
   reconstruction, Caccioppoli regularity, bounded modulation, and the
   compactness/tightness inputs used by the S3 limiting passage.

Lemma S6.2 proves the local velocity vanishing part of this payload for
separated profiles. The remaining pressure and localization terms are the
nonlocal interaction part of the multi-point no-splitting problem. S8
discharges them in terminal active cameras from terminal windowed profile
completeness, scattering removal, exterior-regular discard, repaired-gauge
representation, and Caccioppoli regularity.

---

## S6.6 (Generic non-simultaneous multi-point candidates)

If a physical point \(x_q^*\ne x_p^*\) does not become singular at the first
blowup time \(T^*\), then it lies in a spacetime region where the physical
solution remains regular up to \(T^*\). Such a contribution is exterior-regular
relative to the \(p\)-camera and is removed from the Type II core ledger by
\[
K_{\mathrm{SinglePointBlowup}}^+
\wedge
K_{\mathrm{ExtRegDiscard}}^+.
\]
Thus the only genuine multi-point case is simultaneous multi-point
concentration: at least two separated physical points carry active critical
profiles at the same blowup time.

---

## S6.7 (Camera reduction theorem for multibubbles)

### Theorem S6.3

Assume a declared NS3D Type II candidate satisfies the technical Type II
payloads needed to enter the repaired-gauge profile ledger:
\[
K_{\mathrm{TechTypeII}}^+.
\]
Assume also:
\[
K_{\mathrm{SamePointNLDec}}^+,
\qquad
K_{\mathrm{MultiPointCamDec}}^+,
\qquad
K_{\mathrm{S3NRSPayload}}^+.
\]
Then no active multibubble Type II candidate can occur.

#### Proof

By Lemma S6.1, there are only finitely many active bubbles. Partition them by
physical concentration point \(x_\alpha^*\). Within each physical point,
partition the profiles by comparable scale class.

If a scale class contains several comparable profiles, S6.3 combines them into
one compound profile. If the compound profile is below the small-data
threshold, it is perturbative. If it is active, it is one single-bubble profile
for that camera.

If a physical point contains strict scale separation, S6.4 and
\(K_{\mathrm{SamePointNLDec}}^+\) reduce the innermost active scale to a
single repaired-gauge branch with perturbative outer interactions. S5 then
rules out that same-point cascade through the S3/NRŠ payload.

It remains to consider separated physical points. Choose any active point
\(x_\alpha^*\) and place a camera at its active scale and center. By
Lemma S6.2, every profile centered at a different physical point is locally
\(L^3\)-invisible in this camera. By \(K_{\mathrm{MultiPointCamDec}}^+\), its
pressure, cutoff, and modulation interactions are also perturbative in the S3
topology. Therefore the \(x_\alpha^*\)-camera sees a single active
repaired-gauge branch with positive finite critical mass, bounded modulation,
pressure reconstruction, Caccioppoli regularity, and the compactness/tightness
inputs required by S3.

The scale of that active branch tends to zero, so the S3 scale-collapse input
is present. The payload \(K_{\mathrm{S3NRSPayload}}^+\) extracts a nonzero
stationary \(L^3(\mathbb R^3)\) profile in the NRŠ-covered class. NRŠ rules
out such a profile. This contradiction excludes the active point. Since the
chosen point was arbitrary among active points, no active multibubble
candidate can occur.

\(\square\)

---

## S6.8 (Residual ledger)

S6 does not itself prove the nonlinear no-splitting payloads. It identifies
them as the exact multibubble payloads needed before applying S3. S7 reduces
both payloads to \(K_{\mathrm{NLProfDec},NS3D}^+\), and S8 proves
\(K_{\mathrm{NLProfDec},NS3D}^+\) for terminal active cameras from the
terminal-complete profile backend.

After S6, a multibubble candidate satisfies one of:

1. same-point nonlinear decoupling failure:
   \[
   K_{\mathrm{SamePointNLDec}}^-;
   \]
2. separated-point camera-decoupling failure:
   \[
   K_{\mathrm{MultiPointCamDec}}^-;
   \]
3. S3 rigidity payload failure:
   \[
   K_{\mathrm{S3NRSPayload}}^-;
   \]
4. a technical bridge defect inside \(K_{\mathrm{TechTypeII}}^+\).

If
\[
K_{\mathrm{TechTypeII}}^+
\wedge
K_{\mathrm{SamePointNLDec}}^+
\wedge
K_{\mathrm{MultiPointCamDec}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+
\]
is supplied, then every multibubble Type II candidate is ruled out.

Thus, at the level of S6 alone, the multibubble problem is reduced to the two
no-splitting/profile compatibility payloads:
\[
K_{\mathrm{SamePointNLDec}}^+,
\qquad
K_{\mathrm{MultiPointCamDec}}^+.
\]
After S7 and S8, these payloads are supplied by terminal nonlinear profile
decoupling. Any remaining multibubble failure is therefore an upstream failure
of terminal profile completeness, scattering removal, exterior-regular
discard, repaired-gauge representation, Caccioppoli regularity, or the S3
rigidity payload.
