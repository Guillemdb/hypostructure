# S2: rough-core status and vorticity-controlled subtype exclusion

This note records the status of the rough-core stratum. The rough-core stratum
is not closed by a standalone Biot-Savart or enstrophy-growth argument. The
unconditional 3D vorticity equation contains the stretching term
\[
(\Omega\cdot\nabla)V,
\]
and the standard localized enstrophy estimate requires control of gradient
quantities whose failure is precisely the rough-core defect.

The actual rough-core closure in this folder is the direct Caccioppoli bridge
C6/T13: bounded critical \(L^3\), pressure reconstruction, bounded modulation,
and Caccioppoli regularity imply \(K_{\mathrm{WinH1}}^+\).

The vorticity argument below is a valid conditional subtype exclusion. It says
that a rough core cannot have uniformly controlled local vorticity on unit
windows. Thus it is a diagnostic reduction, not the primary rough-core
closure.

---

## S2.0 (No autonomous vorticity self-closure)

Let
\[
\Omega:=\nabla\times V.
\]
For the repaired-gauge equation
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V+a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\]
the vorticity equation is
\[
\partial_\tau\Omega+(V\cdot\nabla)\Omega-(\Omega\cdot\nabla)V
=
\nu\Delta\Omega+a(\tau)(2\Omega+y\cdot\nabla\Omega)
+b(\tau)\cdot\nabla\Omega.
\]

Consequently, at the smooth approximation level, a localized enstrophy
estimate contains the vortex-stretching term
\[
\int(\Omega\cdot\nabla V)\cdot\Omega\,\chi_R^2.
\]
The standard smooth-level bound is
\[
\left|
\int(\Omega\cdot\nabla V)\cdot\Omega\,\chi_R^2
\right|
\le
C\|\Omega\chi_R\|_{L^2}^{1/2}
\|\nabla(\Omega\chi_R)\|_{L^2}^{3/2}
\|\nabla V\|_{L^2(B_{2R})}.
\]
Young's inequality then gives a differential inequality with coefficient
\[
C_\nu\|\nabla V\|_{L^2(B_{2R})}^4
\]
multiplying the local enstrophy. Therefore the localized vorticity equation
does not close a uniform enstrophy-window estimate without an independent
gradient-window input. That input is exactly what \(K_{\mathrm{WinH1}}^-\)
denies.

Thus S2 is not an autonomous state-space stratification proof of rough-core
exclusion. The local vorticity route only excludes the subtype in which
local vorticity-window control is already available.

---

## S2.1 (Local div-curl estimate)

Let \(B_R\subset\mathbb R^3\). There is a constant \(C_R\) such that every
vector field \(U\in L^2(B_{2R};\mathbb R^3)\) with
\[
\nabla\cdot U=0
\]
in the sense of distributions on \(B_{2R}\), and with
\[
\nabla\times U\in L^2(B_{2R};\mathbb R^3),
\]
satisfies \(U\in H^1(B_R;\mathbb R^3)\) and
\[
\|\nabla U\|_{L^2(B_R)}
\le
C_R\left(
\|\nabla\times U\|_{L^2(B_{2R})}
+\|U\|_{L^2(B_{2R})}
\right).
\]

### Proof

Choose \(\chi\in C_c^\infty(B_{2R})\) with \(\chi\equiv1\) on \(B_R\).
Set
\[
W:=\chi U.
\]
Then \(W\in L^2(\mathbb R^3)\) is compactly supported after extension by zero.
Moreover,
\[
\nabla\times W
=
\chi\nabla\times U+\nabla\chi\times U\in L^2(\mathbb R^3),
\]
and, using \(\nabla\cdot U=0\),
\[
\nabla\cdot W
=
\nabla\chi\cdot U\in L^2(\mathbb R^3).
\]
The Fourier div-curl identity for compactly supported distributions with
\(W\), \(\nabla\times W\), and \(\nabla\cdot W\) in \(L^2\) gives
\[
\|\nabla W\|_{L^2(\mathbb R^3)}^2
=
\|\nabla\times W\|_{L^2(\mathbb R^3)}^2
+\|\nabla\cdot W\|_{L^2(\mathbb R^3)}^2.
\]
Indeed, Plancherel gives
\[
|\xi|^2|\widehat W(\xi)|^2
=
|\xi\times\widehat W(\xi)|^2
+|\xi\cdot\widehat W(\xi)|^2,
\]
and integration in \(\xi\) yields the identity. Hence
\(W\in H^1(\mathbb R^3)\).
Since \(W=U\) on \(B_R\),
\[
\|\nabla U\|_{L^2(B_R)}
\le
\|\nabla W\|_{L^2(\mathbb R^3)}.
\]
Thus
\[
\|\nabla W\|_{L^2(\mathbb R^3)}
\le
C_R\left(
\|\nabla\times U\|_{L^2(B_{2R})}
+\|U\|_{L^2(B_{2R})}
\right).
\]
Combining the previous inequalities proves the estimate.

\(\square\)

---

## S2.2 (Critical \(L^3\) gives local \(L^2\))

Assume
\[
K_{L^3\mathrm{Bd}}^+:
\qquad
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}\le M.
\]
Then, for every \(R>0\),
\[
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^2(B_R)}^2
\le
C R M^2.
\]

### Proof

By Holder's inequality,
\[
\int_{B_R}|V(y,\tau)|^2\,dy
\le
|B_R|^{1/3}
\left(\int_{B_R}|V(y,\tau)|^3\,dy\right)^{2/3}
\le
C R M^2.
\]
Taking the supremum in \(\tau\) proves the estimate.

\(\square\)

---

## S2.3 (Local vorticity-window control implies windowed \(H^1\))

Let
\[
\Omega:=\nabla\times V.
\]
Assume:

1. represented divergence-free orbit:
   \[
   K_{\mathrm{RepBridge}}^+;
   \]
2. bounded critical norm:
   \[
   K_{L^3\mathrm{Bd}}^+;
   \]
3. local vorticity-window control:
   \[
   K_{\mathrm{VortL^2Win}}^+:
   \qquad
   \forall R>0:\quad
   \sup_{T\ge\tau_0+1}
   \int_T^{T+1}\|\Omega(\tau)\|_{L^2(B_{2R})}^2\,d\tau<\infty;
   \]
4. the vorticity \(\Omega=\nabla\times V\) is the distributional curl of the
   represented velocity, and \(V\) is distributionally divergence free.

Then
\[
K_{\mathrm{WinH1}}^+
\]
holds:
\[
\forall R>0:\qquad
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty.
\]

### Proof

Fix \(R>0\). By S2.1, for almost every \(\tau\),
\[
\|\nabla V(\tau)\|_{L^2(B_R)}^2
\le
C_R\left(
\|\Omega(\tau)\|_{L^2(B_{2R})}^2
+\|V(\tau)\|_{L^2(B_{2R})}^2
\right).
\]
Integrate over \([T,T+1]\). The vorticity term is uniformly bounded in
\(T\) by \(K_{\mathrm{VortL^2Win}}^+\). The local \(L^2\) term is uniformly
bounded by S2.2:
\[
\int_T^{T+1}\|V(\tau)\|_{L^2(B_{2R})}^2\,d\tau
\le
C_R M^2.
\]
Hence
\[
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|\nabla V(\tau)\|_{L^2(B_R)}^2\,d\tau<\infty.
\]
The same local \(L^2\) bound gives
\[
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{L^2(B_R)}^2\,d\tau<\infty.
\]
Combining the two estimates gives the asserted local windowed \(H^1\) bound.

\(\square\)

---

## S2.4 (Vorticity-controlled rough-core exclusion)

Assume a declared Type II candidate is represented and has positive finite
critical mass, so that C3 emits \(K_{L^3\mathrm{Bd}}^+\). If the branch also
emits \(K_{\mathrm{VortL^2Win}}^+\), then the rough-core defect is impossible:
\[
K_{\mathrm{WinH1}}^-
\quad\text{cannot hold}.
\]
Equivalently,
\[
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{VortL^2Win}}^+
\Longrightarrow
K_{\mathrm{RoughCoreBlk}}^+.
\]

### Proof

The represented divergence-free structure, bounded critical norm, and
\(K_{\mathrm{VortL^2Win}}^+\) are precisely the hypotheses of S2.3. Therefore
\(K_{\mathrm{WinH1}}^+\) holds. On represented repaired-gauge Type II
branches, C7 identifies
\[
K_{\mathrm{WinH1}}^+
\Longleftrightarrow
K_{\mathrm{RoughCoreBlk}}^+.
\]
Thus \(K_{\mathrm{WinH1}}^-\) is excluded.

\(\square\)

---

## S2.4a (Relation to the Caccioppoli rough-core closure)

The primary rough-core closure remains C6/T13:
\[
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{PressureRep}}^+
\wedge
K_{\mathrm{ModBd}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
\]
This is a direct local energy/Caccioppoli estimate. It does not rely on an
autonomous vorticity-enstrophy Grönwall closure.

S2.4 is therefore subordinate to the rough-core ledger: it shows that if
one supplies \(K_{\mathrm{VortL^2Win}}^+\), then the rough-core defect is
already impossible by div-curl. If \(K_{\mathrm{VortL^2Win}}^+\) is not
available, rough-core exclusion must proceed through C6/T13 or through the
modulation/Caccioppoli payloads in C7.

---

## S2.5 (Remaining rough-core mechanism after S2)

S2 kills the vorticity-controlled rough-core subtype. Therefore any
represented, bounded-critical-norm rough-core survivor must emit at least one
of the following defects:

1. \(K_{\mathrm{VortL^2Win}}^-\): failure of local vorticity-window control;
2. a representation or critical-mass defect upstream of S2.

Thus a rough-core branch is no longer allowed to be merely a bounded
divergence-free \(L^3\)-critical core with controlled vorticity. It must carry
a genuine local enstrophy-window defect, or else be handled by the remaining
C6/C7 payload defects:
\[
K_{\mathrm{ModForceBd}}^-,
\qquad
K_{\mathrm{CaccioppoliReg}}^-,
\]
or by their integrated refinements.
