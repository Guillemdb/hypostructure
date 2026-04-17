# S11: Caccioppoli regularity discharge

This note proves the Caccioppoli-regularity certificate used by C7, S8, and
the rough-core bridge. The certificate
\[
K_{\mathrm{CaccioppoliReg}}^+
\]
means that the represented repaired-gauge branch satisfies the local energy
inequality, or an approximation scheme strong enough to justify the
renormalized Caccioppoli estimate on compact cylinders.

---

## S11.1 (Suitable physical input)

Let \(u,p\) be a local suitable weak solution of NS3D on
\(\mathbb R^3\times(0,T^*)\). Thus
\[
u\in L^\infty_tL^2_{\mathrm{loc}}
\cap L^2_tH^1_{\mathrm{loc}},
\qquad
p\in L^{3/2}_{\mathrm{loc}},
\qquad
\nabla\cdot u=0,
\]
the Navier-Stokes equations hold distributionally, and for every nonnegative
\(\psi\in C_c^\infty(\mathbb R^3\times(0,T^*))\),
\[
\begin{aligned}
&\int \frac{|u(t_2)|^2}{2}\psi(t_2)
+\nu\int_{t_1}^{t_2}\int |\nabla u|^2\psi \\
&\le
\int \frac{|u(t_1)|^2}{2}\psi(t_1)
+\int_{t_1}^{t_2}\int
\frac{|u|^2}{2}(\partial_t\psi+\nu\Delta\psi)
+\int_{t_1}^{t_2}\int
\left(\frac{|u|^2}{2}+p\right)u\cdot\nabla\psi .
\end{aligned}
\]
This is the physical certificate
\[
K_{\mathrm{PhysSuitable}}^+.
\]

---

## S11.2 (Renormalized local energy inequality)

Let the repaired-gauge representation be
\[
u(x,t)=\lambda(t)^{-1}
V\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right),
\qquad
\frac{d\tau}{dt}=\lambda(t)^{-2},
\]
with
\[
P(y,\tau)=\lambda(t)^2p(x_c(t)+\lambda(t)y,t).
\]
Assume the gauge functions are \(C^1\), \(\lambda(t)>0\), and the
representation emits the repaired-gauge equation
\[
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V+a(V+y\cdot\nabla V)+b\cdot\nabla V,
\qquad
\nabla\cdot V=0.
\]

### Lemma S11.1

The renormalized field \(V,P\) satisfies the local energy inequality
\[
\begin{aligned}
&\int \frac{|V(\tau_2)|^2}{2}\phi(\tau_2)
+\nu\int_{\tau_1}^{\tau_2}\int |\nabla V|^2\phi \\
&\le
\int \frac{|V(\tau_1)|^2}{2}\phi(\tau_1)
+\int_{\tau_1}^{\tau_2}\int
\frac{|V|^2}{2}(\partial_\tau\phi+\nu\Delta\phi)\\
&\quad
+\int_{\tau_1}^{\tau_2}\int
\left(\frac{|V|^2}{2}+P\right)V\cdot\nabla\phi \\
&\quad
+\int_{\tau_1}^{\tau_2}\int
a\frac{|V|^2}{2}(-\phi-y\cdot\nabla\phi)
-\int_{\tau_1}^{\tau_2}\int
\frac{|V|^2}{2}b\cdot\nabla\phi
\end{aligned}
\]
for every nonnegative
\(\phi\in C_c^\infty(\mathbb R^3\times(\tau_0,\infty))\).

#### Proof

For smooth solutions, multiply the repaired-gauge equation by \(V\phi\) and
integrate by parts. With \(e=|V|^2/2\),
\[
V\cdot(V\cdot\nabla V)=V\cdot\nabla e=\nabla\cdot(eV),
\]
\[
V\cdot\nabla P=\nabla\cdot(PV),
\]
and
\[
V\cdot(y\cdot\nabla V)=y\cdot\nabla e,
\qquad
V\cdot(b\cdot\nabla V)=b\cdot\nabla e.
\]
The scaling contribution satisfies
\[
\int a(2e+y\cdot\nabla e)\phi
=
\int a e(-\phi-y\cdot\nabla\phi),
\]
because
\[
\int y\cdot\nabla e\,\phi
=
-\int e\,\nabla\cdot(y\phi)
=
-\int e(3\phi+y\cdot\nabla\phi).
\]
The translation contribution satisfies
\[
\int b\cdot\nabla e\,\phi
=
-\int e\,b\cdot\nabla\phi.
\]
The viscous term gives
\[
\int \nu\Delta V\cdot V\phi
=
-\nu\int|\nabla V|^2\phi
+\nu\int e\Delta\phi.
\]
Combining these identities gives the displayed equality in the smooth case.

For suitable weak solutions, apply the physical local energy inequality with
test function
\[
\psi(x,t)=\phi\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right),
\]
multiply by the Jacobian factors from
\[
x=x_c(t)+\lambda(t)y,
\qquad
dt=\lambda(t)^2\,d\tau,
\]
and pass to the renormalized variables. The \(C^1\) regularity of
\(\lambda,x_c\) justifies the chain rule for the test function. The resulting
inequality is the same as the smooth identity with equality replaced by
\(\le\).

\(\square\)

---

## S11.3 (Renormalized Caccioppoli estimate is justified)

### Lemma S11.2

Assume Lemma S11.1, bounded modulation on a compact cylinder, and local
\[
V\in L^\infty_\tau L^3_y,
\qquad
P\in L^{3/2}_{\mathrm{loc}}.
\]
Then the Caccioppoli estimate used in C6/T13 is valid on compact
renormalized cylinders.

#### Proof

Choose a standard cutoff
\[
\phi(y,\tau)=\eta(\tau)\zeta(y)^2
\]
with \(\eta,\zeta\ge0\), \(\zeta\equiv1\) on the inner ball, and
\(\zeta\) supported in the outer ball. Insert this \(\phi\) into Lemma S11.1.
The left-hand side contains
\[
\nu\int |\nabla V|^2\eta\zeta^2.
\]
Every right-hand-side term is finite under the stated hypotheses:
\[
\int |V|^2|\partial_\tau\phi+\Delta\phi|
\]
is controlled by local \(L^\infty_\tau L^3_y\) and finite measure;
\[
\int |V|^3|\nabla\phi|
\]
is controlled by local \(L^\infty_\tau L^3_y\);
\[
\int |P-c(\tau)|\,|V|\,|\nabla\phi|
\]
is controlled by \(P-c(\tau)\in L^{3/2}_{\mathrm{loc}}\) and
\(V\in L^3_{\mathrm{loc}}\), with constants disappearing because
\[
\int c(\tau)V\cdot\nabla\phi
=
-\int c(\tau)\phi\,\nabla\cdot V=0;
\]
and the modulation terms are bounded by
\[
\|a\|_{L^\infty}+\|b\|_{L^\infty}
\]
times local \(L^2\)-mass, which is again controlled by local \(L^3\)-mass on
bounded balls. This is exactly the renormalized Caccioppoli inequality used in
C6/T13.

\(\square\)

---

## S11.4 (Caccioppoli regularity certificate)

### Theorem S11.3

Assume:
\[
K_{\mathrm{PhysSuitable}}^+,
\qquad
K_{\mathrm{RepBridge}}^+,
\qquad
K_{\mathrm{PressureRep}}^+,
\]
and \(C^1\) repaired-gauge scale and center functions on compact
renormalized windows. Then
\[
K_{\mathrm{CaccioppoliReg}}^+
\]
holds.

#### Proof

The representation bridge converts the physical suitable weak solution into
the repaired-gauge variables \(V,P,a,b\). Lemma S11.1 gives the renormalized
local energy inequality. Lemma S11.2 shows that this inequality justifies the
renormalized Caccioppoli estimate on every compact cylinder. This is precisely
the meaning of \(K_{\mathrm{CaccioppoliReg}}^+\).

\(\square\)

---

## S11.5 (Connection to the rough-core bridge)

Combining Theorem S11.3 with C6/T13 gives
\[
K_{\mathrm{PhysSuitable}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{PressureRep}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{ModBd}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
\]
Thus \(K_{\mathrm{CaccioppoliReg}}^-\) is not an independent rough-core
singularity mechanism whenever the represented branch is suitable in the
physical variables and the gauge is \(C^1\) on compact windows.
