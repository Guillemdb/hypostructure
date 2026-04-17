# C6 / T13: windowed local \(H^1\) bridge from Caccioppoli

This note proves the rough-core bridge used by the compact Type II barrier.
The output certificate is
\[
K_{\mathrm{WinH1}}^+:
\qquad
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty
\quad\text{for every }R>0.
\]

The bridge is deliberately stated inside the repaired-gauge renormalized
Navier-Stokes variables. It does not rely directly on physical energy
dissipation, because the physical dissipation controls
\[
\int \lambda(\tau)\|\nabla_y V(\tau)\|_{L^2_y}^2\,d\tau,
\]
which is weaker than the unweighted windowed estimate when
\(\lambda(\tau)\to0\). The renormalized Caccioppoli estimate supplies the
correct unweighted local estimate.

Throughout, \(V,P,a,b\) solve the repaired-gauge renormalized Navier-Stokes
equation
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V+a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0.
\]

---

## T13.1 (Uniform local \(L^{3/2}\) pressure control from global \(L^3\))

Assume
\[
M_3:=\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty
\]
and
\[
-\Delta P=\partial_i\partial_j(V_iV_j)
\quad\text{in }\mathbb R^3
\]
for each \(\tau\ge\tau_0\). Then for every \(R>0\) there is a measurable
choice of constants \(c_R(\tau)\in\mathbb R\) such that
\[
\sup_{\tau\ge\tau_0}
\|P(\tau)-c_R(\tau)\|_{L^{3/2}(B_R)}
\le C_R M_3^2.
\]

### Proof

Lemma 9 in the master note gives
\[
\|P(\tau)-c_R(\tau)\|_{L^{3/2}(B_R)}
\le
C\left(
\|V(\tau)\|_{L^3(B_{4R})}^2
+R^2\mathcal T_R(\tau)
\right),
\]
where
\[
\mathcal T_R(\tau)
:=
\sup_{x\in B_R}\int_{|z|>2R}
\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz.
\]
Lemma 9.1 gives
\[
\sup_{\tau\ge\tau_0}\mathcal T_R(\tau)\le C R^{-4}M_3^2.
\]
Also
\[
\|V(\tau)\|_{L^3(B_{4R})}\le M_3.
\]
Combining the two bounds yields
\[
\sup_{\tau\ge\tau_0}
\|P(\tau)-c_R(\tau)\|_{L^{3/2}(B_R)}
\le C_RM_3^2.
\]
The constants \(c_R(\tau)\) can be chosen, for instance, by the construction
in Lemma 9, hence measurably after fixing the displayed pressure
decomposition.

\(\square\)

---

## T13.2 (Renormalized Caccioppoli gives uniform windowed gradient control)

Assume:

1. \(V,P\) are smooth on compact renormalized cylinders and solve the
   renormalized equation above;
2. global critical control:
   \[
   M_3:=\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty;
   \]
3. bounded modulation:
   \[
   M_{ab}:=\sup_{\tau\ge\tau_0}(|a(\tau)|+|b(\tau)|)<\infty;
   \]
4. the pressure is normalized modulo constants as in T13.1.

Then for every \(R>0\) there exists
\[
C_R=C(R,\nu,M_3,M_{ab})
\]
such that
\[
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\int_{B_R}|\nabla V(y,\tau)|^2\,dy\,d\tau
\le C_R.
\]

### Proof

Fix \(R>0\) and \(T\ge\tau_0+1\). Use the proof of the renormalized
Caccioppoli estimate, Lemma 5 in the master note, with outer radius \(2R\),
inner radius \(R\), and time intervals
\[
I_{2R}:=(T-1,T+2),
\qquad
I_R:=[T,T+1].
\]
Thus \(R_{\mathrm{out}}-R_{\mathrm{in}}=R\) and
\(\sigma_2-\sigma_1=1\). Repeating the proof of Lemma 5 with the pressure
term kept before taking absolute values gives
\[
\begin{aligned}
\nu\int_T^{T+1}\int_{B_R}|\nabla V|^2
\le\;&
C_R\iint_{(T-1,T+2)\times B_{2R}}|V|^2
+C_R\iint_{(T-1,T+2)\times B_{2R}}|V|^3 \\
&+
C_R\mathcal P_{T,R}
+C_RM_{ab}\iint_{(T-1,T+2)\times B_{2R}}|V|^2 .
\end{aligned}
\]
Here \(C_R\) absorbs the powers of \(R^{-1}\), \(R^{-2}\), and \(\nu\).
The pressure contribution is kept in its proof-level form
\[
\mathcal P_{T,R}
:=
\left|
\iint_{(T-1,T+2)\times B_{2R}}P\,V\cdot\nabla\zeta
\right|,
\]
where \(\zeta\) is the cutoff used in Lemma 5. This is essential because the
pressure is defined modulo time-dependent constants.

The \(L^2\)-term is controlled by the global \(L^3\)-bound and boundedness of
\(B_{2R}\):
\[
\int_{B_{2R}}|V(\tau)|^2\,dy
\le
|B_{2R}|^{1/3}\|V(\tau)\|_{L^3(\mathbb R^3)}^2
\le C_RM_3^2.
\]
Since the outer time interval has length \(3\),
\[
\iint_{(T-1,T+2)\times B_{2R}}|V|^2\le C_RM_3^2.
\]
Similarly,
\[
\iint_{(T-1,T+2)\times B_{2R}}|V|^3
\le 3M_3^3.
\]

It remains to estimate \(\mathcal P_{T,R}\).
For any time-dependent constant \(c(\tau)\),
\[
\int_{B_{2R}}c(\tau)V\cdot\nabla\zeta\,dy
=
-\int_{B_{2R}}c(\tau)\zeta\,\nabla\cdot V\,dy
=0.
\]
Therefore \(P\) may be replaced by \(P-c_{2R}(\tau)\). By Holder,
\[
\mathcal P_{T,R}
\le
C_R
\iint_{(T-1,T+2)\times B_{2R}}
|P-c_{2R}(\tau)|\,|V|\,dy\,d\tau,
\]
because \(|\nabla\zeta|\le C_R\). For almost every \(\tau\),
\[
\int_{B_{2R}}|P-c_{2R}(\tau)|\,|V(\tau)|\,dy
\le
\|P(\tau)-c_{2R}(\tau)\|_{L^{3/2}(B_{2R})}
\|V(\tau)\|_{L^3(B_{2R})}.
\]
Here \(c_{2R}(\tau)\) is the constant supplied by T13.1 with radius \(2R\).
T13.1 gives
\[
\|P(\tau)-c_{2R}(\tau)\|_{L^{3/2}(B_{2R})}
\le C_RM_3^2,
\]
and the global \(L^3\)-bound gives
\[
\|V(\tau)\|_{L^3(B_{2R})}\le M_3.
\]
Hence
\[
\mathcal P_{T,R}
\le C_RM_3^3.
\]

Substituting these estimates into Caccioppoli gives
\[
\nu\int_T^{T+1}\int_{B_R}|\nabla V|^2
\le
C_R\left(M_3^2+M_3^3+M_{ab}M_3^2\right),
\]
uniformly in \(T\ge\tau_0+1\). Dividing by \(\nu\) proves the claim after
renaming the constant.

\(\square\)

---

## T13.3 (Windowed local \(H^1\) bridge)

Under the hypotheses of T13.2, for every \(R>0\),
\[
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau
<\infty.
\]
Equivalently, the repaired-gauge renormalized orbit emits the tail version of
the windowed \(H^1\) certificate
\[
K_{\mathrm{WinH1}}^+.
\]

### Proof

The gradient part is T13.2. The \(L^2\)-part follows from the same bounded-set
Holder estimate used above:
\[
\int_T^{T+1}\|V(\tau)\|_{L^2(B_R)}^2\,d\tau
\le
C_R\int_T^{T+1}\|V(\tau)\|_{L^3(\mathbb R^3)}^2\,d\tau
\le C_RM_3^2.
\]
Adding the two bounds gives the asserted \(H^1(B_R)\) estimate.

\(\square\)

---

## T13.4 (Rough-core suppression under bounded critical norm and modulation)

Assume the represented Type II branch satisfies:

1. repaired-gauge renormalized Navier-Stokes equations on compact cylinders;
2. bounded critical norm:
   \[
   \sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty;
   \]
3. bounded modulation:
   \[
   \sup_{\tau\ge\tau_0}(|a(\tau)|+|b(\tau)|)<\infty;
   \]
4. pressure reconstruction by
   \[
   -\Delta P=\partial_i\partial_j(V_iV_j).
   \]

Then the rough-core alternative
\[
\exists m\ge1:
\sup_{T\ge\tau_0+1}
\int_T^{T+1}\|V(\tau)\|_{H^1(B_m)}^2\,d\tau=\infty
\]
is impossible.

### Proof

Apply T13.3 with \(R=m\). The resulting finite bound contradicts the displayed
rough-core alternative.

\(\square\)

---

## Status

This note proves a conditional rough-core suppression bridge:
\[
\text{bounded }L^\infty_\tau L^3_y
\quad+\quad
\text{bounded modulation}
\quad+\quad
\text{pressure reconstruction}
\quad\Longrightarrow\quad
K_{\mathrm{WinH1}}^+.
\]

The pressure input is not circular: it uses the \(L^{3/2}\)-pressure estimate
from T1 and the pressure-tail control from global \(L^3\), not the local
\(L^2\)-pressure estimate T1.5, which itself uses local \(H^1\).

The remaining structural task is to derive the bounded modulation and
represented-orbit hypotheses from the upstream Type II branch certificates.
