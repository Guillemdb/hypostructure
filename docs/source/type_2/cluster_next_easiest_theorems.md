# Immediate high-yield cluster: pressure tail, gauge matrix, modulation bounds, and time regularity

This note records the theorem-level closures T1--T6 from
[type2_roadmap.md](type2_roadmap.md) for the renormalized three-dimensional
Navier--Stokes system
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=\nu\Delta V+a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0.
\]

The repaired scale gauge is
\[
G_0(V)=\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0,\qquad 0<p<3,
\]
and the centering gauges are
\[
G_j(V)=\int_{\mathbb R^3}y_j|V(y)|^2\psi_R(y)\,dy,\qquad j=1,2,3.
\]
Here \(\psi_R\in C_c^\infty(\mathbb R^3)\) is radial,
\[
0\le \psi_R\le1,\qquad
\psi_R\equiv1\text{ on }B_R,\qquad
\operatorname{supp}\psi_R\subset B_{2R}.
\]
Set
\[
A_R:=\{R\le |y|\le 2R\}.
\]

For a matrix \(B\in\mathbb R^{3\times3}\), use
\[
\|B\|_{\infty\to\infty}:=\max_{1\le j\le3}\sum_{\ell=1}^3|B_{j\ell}|.
\]

The repaired modulation matrix is written in block form as
\[
M(V)=
\begin{pmatrix}
\alpha(V)&u(V)^T\\
v(V)&A(V)
\end{pmatrix},
\]
where
\[
\alpha(V)=M_{00}(V),\qquad
u_\ell(V)=M_{0\ell}(V),\qquad
v_j(V)=M_{j0}(V),\qquad
A(V)=(M_{j\ell}(V))_{1\le j,\ell\le3}.
\]

---

## T1. Uniform local pressure-tail control

For \(R>0\), define
\[
\mathcal T_R(\tau):=
\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz.
\]

Assume
\[
M_3:=\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty.
\]
Then, for each fixed \(R>0\),
\[
\sup_{\tau\ge\tau_0}\mathcal T_R(\tau)\le C R^{-4}M_3^2,
\]
where \(C\) is an absolute constant.

### Proof

Fix \(\tau\ge\tau_0\) and \(x\in B_R\). Decompose
\[
\{|z|>2R\}=\bigcup_{k=1}^{\infty}A_k,\qquad
A_k:=\{2^kR<|z|\le2^{k+1}R\}.
\]
For \(z\in A_k\),
\[
|x-z|\ge |z|-|x|\ge 2^kR-R\ge 2^{k-1}R.
\]
Hence
\[
\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz
\le
C\sum_{k\ge1}(2^kR)^{-5}\int_{A_k}|V(z,\tau)|^2\,dz.
\]
By Holder on \(A_k\),
\[
\int_{A_k}|V|^2
\le |A_k|^{1/3}\left(\int_{A_k}|V|^3\right)^{2/3}
\le C\,2^kR\,a_k(\tau)^{2/3},
\]
where
\[
a_k(\tau):=\int_{A_k}|V(z,\tau)|^3\,dz.
\]
Therefore
\[
\mathcal T_R(\tau)
\le C R^{-4}\sum_{k\ge1}2^{-4k}a_k(\tau)^{2/3}.
\]
Applying Holder on the counting measure with exponents \(3\) and \(3/2\),
\[
\sum_{k\ge1}2^{-4k}a_k^{2/3}
\le
\left(\sum_{k\ge1}2^{-12k}\right)^{1/3}
\left(\sum_{k\ge1}a_k\right)^{2/3}
\le C\|V(\tau)\|_{L^3(\mathbb R^3)}^2.
\]
Taking the supremum in \(x\) and \(\tau\) gives the claim.

\(\square\)

### Corollary T1.1. Pressure bound modulo constants

If \(P\) satisfies the pressure equation used in Lemma 9 of
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md), then for each fixed \(R>0\) there is \(c_R(\tau)\in\mathbb R\) such that
\[
\|P(\tau)-c_R(\tau)\|_{L^{3/2}(B_R)}
\le C_R\left(M_3^2+\mathcal T_R(\tau)\right).
\]
Consequently,
\[
\sup_{\tau\ge\tau_0}\inf_{c\in\mathbb R}
\|P(\tau)-c\|_{L^{3/2}(B_R)}<\infty.
\]

---

## T1.4. Verification of the T1.5 pressure hypotheses

Let \(I\subset\mathbb R\) be bounded and \(R>0\). Assume
\[
M_3:=\sup_{\tau\in I}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty
\]
and
\[
V\in L^2(I;H^1(B_{4R})).
\]
Then
\[
V\in L^4(I;L^4(B_{4R})),
\]
and
\[
\mathcal H_R(\tau):=
\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz
\]
satisfies
\[
\|\mathcal H_R\|_{L^\infty(I)}
\le C R^{-3}M_3^2.
\]
Consequently, \(\mathcal H_R\in L^2(I)\), with
\[
\|\mathcal H_R\|_{L^2(I)}
\le C |I|^{1/2}R^{-3}M_3^2.
\]

### Proof

First prove local \(L^4\) control. For almost every \(\tau\in I\), interpolation gives
\[
\|V(\tau)\|_{L^4(B_{4R})}
\le
\|V(\tau)\|_{L^3(B_{4R})}^{1/2}
\|V(\tau)\|_{L^6(B_{4R})}^{1/2}.
\]
Sobolev on \(B_{4R}\) gives
\[
\|V(\tau)\|_{L^6(B_{4R})}
\le C_R\|V(\tau)\|_{H^1(B_{4R})}.
\]
Thus
\[
\|V(\tau)\|_{L^4(B_{4R})}^4
\le C_R
\|V(\tau)\|_{L^3(B_{4R})}^2
\|V(\tau)\|_{H^1(B_{4R})}^2
\le C_RM_3^2\|V(\tau)\|_{H^1(B_{4R})}^2.
\]
Integrating in \(\tau\) yields
\[
\|V\|_{L^4(I;L^4(B_{4R}))}^4
\le C_RM_3^2
\|V\|_{L^2(I;H^1(B_{4R}))}^2<\infty.
\]

It remains to estimate \(\mathcal H_R\). Fix \(\tau\in I\) and \(x\in B_R\). Decompose
\[
\{|z|>2R\}=\bigcup_{k=1}^\infty A_k,\qquad
A_k:=\{2^kR<|z|\le2^{k+1}R\}.
\]
For \(z\in A_k\),
\[
|x-z|\ge |z|-|x|\ge 2^kR-R\ge 2^{k-1}R.
\]
Therefore
\[
\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz
\le
C\sum_{k\ge1}(2^kR)^{-4}\int_{A_k}|V(z,\tau)|^2\,dz.
\]
As in T1, Holder gives
\[
\int_{A_k}|V|^2
\le |A_k|^{1/3}\left(\int_{A_k}|V|^3\right)^{2/3}
\le C\,2^kR\,a_k(\tau)^{2/3},
\]
where \(a_k(\tau):=\int_{A_k}|V(z,\tau)|^3\,dz\). Hence
\[
\mathcal H_R(\tau)
\le C R^{-3}\sum_{k\ge1}2^{-3k}a_k(\tau)^{2/3}.
\]
Applying Holder on the counting measure with exponents \(3\) and \(3/2\),
\[
\sum_{k\ge1}2^{-3k}a_k(\tau)^{2/3}
\le
\left(\sum_{k\ge1}2^{-9k}\right)^{1/3}
\left(\sum_{k\ge1}a_k(\tau)\right)^{2/3}
\le C\|V(\tau)\|_{L^3(\mathbb R^3)}^2.
\]
Taking the supremum over \(x\in B_R\) gives
\[
\mathcal H_R(\tau)\le C R^{-3}M_3^2.
\]
Taking the \(L^\infty(I)\) and \(L^2(I)\) norms proves the claim.

\(\square\)

---

## T1.5. Local \(L^2\) pressure estimate modulo constants

For \(R>0\), define the fourth-order far-field pressure tail
\[
\mathcal H_R(\tau):=
\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz.
\]

Let \(I\subset\mathbb R\) be a bounded interval. Assume
\[
V\in L^4(I;L^4(B_{4R})),
\]
and
\[
\mathcal H_R\in L^2(I).
\]
Let \(P\) solve
\[
-\Delta P=\partial_i\partial_j(V_iV_j)
\quad\text{in }\mathbb R^3
\]
in the distributional sense for almost every \(\tau\in I\). Then there exists a measurable function
\[
c_R:I\to\mathbb R
\]
such that
\[
P-c_R\in L^2(I;L^2(B_R)),
\]
and
\[
\|P-c_R\|_{L^2(I;L^2(B_R))}
\le
C_R\left(
\|V\|_{L^4(I;L^4(B_{4R}))}^2
+\|\mathcal H_R\|_{L^2(I)}
\right).
\]
More explicitly, one may take \(C_R=C(1+R^{5/2})\), where \(C\) depends only on the dimension and on the fixed cutoff used in the pressure decomposition.

### Proof

Choose \(\eta\in C_c^\infty(B_{4R})\) with
\[
0\le\eta\le1,\qquad \eta\equiv1\text{ on }B_{2R}.
\]
For almost every \(\tau\in I\), decompose
\[
V_iV_j=\eta V_iV_j+(1-\eta)V_iV_j.
\]
Define
\[
P_{\mathrm{loc}}(\tau):=\mathcal R_i\mathcal R_j(\eta V_iV_j)(\tau),
\]
and
\[
P_{\mathrm{far}}(\tau):=P(\tau)-P_{\mathrm{loc}}(\tau).
\]

For the local term, Calderon-Zygmund boundedness on \(L^2(\mathbb R^3)\) gives
\[
\|P_{\mathrm{loc}}(\tau)\|_{L^2(\mathbb R^3)}
\le C\|\eta V_iV_j(\tau)\|_{L^2(\mathbb R^3)}
\le C\|V(\tau)\|_{L^4(B_{4R})}^2.
\]
Therefore
\[
\|P_{\mathrm{loc}}\|_{L^2(I;L^2(B_R))}
\le C\|V\|_{L^4(I;L^4(B_{4R}))}^2.
\]

For the far term, observe that the source
\[
\partial_i\partial_j((1-\eta)V_iV_j)
\]
vanishes in \(B_{2R}\). Hence \(P_{\mathrm{far}}(\tau)\) is harmonic in \(B_{2R}\) modulo an additive constant. Let
\[
K_{ij}(x):=\partial_i\partial_j\left(\frac{1}{4\pi|x|}\right).
\]
Up to an additive constant,
\[
P_{\mathrm{far}}(x,\tau)
=\int_{\mathbb R^3}K_{ij}(x-z)(1-\eta(z))V_i(z,\tau)V_j(z,\tau)\,dz.
\]
Fix \(x_0=0\in B_R\). Since pressure is defined modulo additive constants, choose \(c_R(\tau)\) so that
\[
P_{\mathrm{far}}(x,\tau)-c_R(\tau)
=
\int_{|z|>2R}
\bigl(K_{ij}(x-z)-K_{ij}(x_0-z)\bigr)
(1-\eta(z))V_i(z,\tau)V_j(z,\tau)\,dz.
\]
This formula defines the normalized far-field representative on \(B_R\); the integral is absolutely convergent by the estimate below and the assumption \(\mathcal H_R(\tau)<\infty\).
The kernel satisfies
\[
|\nabla K_{ij}(\xi)|\le C|\xi|^{-4},\qquad \xi\ne0.
\]
For \(x,x_0\in B_R\) and \(|z|>2R\), the segment between \(x-z\) and \(x_0-z\) remains a fixed positive fraction of \(|x-z|\) away from the origin. Therefore the mean-value theorem gives
\[
|K_{ij}(x-z)-K_{ij}(x_0-z)|
\le C|x-x_0|\,|x-z|^{-4}
\le CR\,|x-z|^{-4}.
\]
Since \(|1-\eta|\le1\),
\[
|P_{\mathrm{far}}(x,\tau)-c_R(\tau)|
\le
CR\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz
\le CR\,\mathcal H_R(\tau)
\]
for every \(x\in B_R\). Taking the \(L^2(B_R)\)-norm gives
\[
\|P_{\mathrm{far}}(\tau)-c_R(\tau)\|_{L^2(B_R)}
\le C R |B_R|^{1/2}\mathcal H_R(\tau)
\le C R^{5/2}\mathcal H_R(\tau).
\]
Therefore
\[
\|P_{\mathrm{far}}-c_R\|_{L^2(I;L^2(B_R))}
\le C R^{5/2}\|\mathcal H_R\|_{L^2(I)}.
\]

Combining the local and far estimates yields
\[
\|P-c_R\|_{L^2(I;L^2(B_R))}
\le
C\|V\|_{L^4(I;L^4(B_{4R}))}^2
+CR^{5/2}\|\mathcal H_R\|_{L^2(I)}.
\]
Absorbing dimensional and cutoff constants into \(C_R\) proves the stated estimate.

\(\square\)

### Corollary T1.6. Pressure input for T6

Under the hypotheses of T1.5,
\[
M_{P,2}:=\left\|\inf_{c\in\mathbb R}
\|P(\tau)-c\|_{L^2(B_R)}\right\|_{L^2_\tau(I)}
<\infty.
\]
Thus the pressure assumption in T6 is supplied by local \(L^4_{t,x}\) velocity control and the fourth-order far-field tail \(\mathcal H_R\in L^2(I)\).

---

## T2. Translation-block invertibility for the centering gauge

Assume
\[
C_\psi:=\max_{1\le j,\ell\le3}
\|y_j\partial_\ell\psi_R\|_{L^\infty(\mathbb R^3)}<\infty.
\]
The translation block entries are
\[
A_{j\ell}(V):=M_{j\ell}(V)
=-\delta_{j\ell}\int_{\mathbb R^3}|V|^2\psi_R\,dy
-\int_{\mathbb R^3}|V|^2y_j\partial_\ell\psi_R\,dy.
\]

Assume that, uniformly for \(\tau\ge\tau_0\),
\[
m_\psi(\tau):=\int_{\mathbb R^3}|V(y,\tau)|^2\psi_R(y)\,dy\ge m_0>0,
\]
\[
\int_{A_R}|V(y,\tau)|^2\,dy\le\varepsilon_0,
\]
and
\[
3C_\psi\varepsilon_0\le \frac{m_0}{2}.
\]
Then \(A(V(\tau))\) is invertible for every \(\tau\ge\tau_0\), and
\[
\sup_{\tau\ge\tau_0}\|A(V(\tau))^{-1}\|_{\infty\to\infty}
\le \frac{2}{m_0}.
\]

### Proof

Define
\[
E_{j\ell}(\tau):=-\int_{\mathbb R^3}|V(y,\tau)|^2y_j\partial_\ell\psi_R(y)\,dy.
\]
Then
\[
A(V(\tau))=-m_\psi(\tau)I_3+E(\tau).
\]
Since \(\partial_\ell\psi_R\) is supported in \(A_R\),
\[
|E_{j\ell}(\tau)|
\le
\|y_j\partial_\ell\psi_R\|_{L^\infty}
\int_{A_R}|V(y,\tau)|^2\,dy
\le C_\psi\varepsilon_0.
\]
Thus
\[
\|E(\tau)\|_{\infty\to\infty}
\le \max_j\sum_{\ell=1}^3|E_{j\ell}(\tau)|
\le 3C_\psi\varepsilon_0
\le \frac{m_0}{2}.
\]
Since \(m_\psi(\tau)\ge m_0\),
\[
\|m_\psi(\tau)^{-1}E(\tau)\|_{\infty\to\infty}\le\frac12.
\]
Therefore
\[
A(V(\tau))
=-m_\psi(\tau)\left(I_3-m_\psi(\tau)^{-1}E(\tau)\right),
\]
and the inverse exists by the Neumann series
\[
\left(I_3-m_\psi(\tau)^{-1}E(\tau)\right)^{-1}
=\sum_{n=0}^{\infty}\left(m_\psi(\tau)^{-1}E(\tau)\right)^n.
\]
Moreover,
\[
\left\|\left(I_3-m_\psi(\tau)^{-1}E(\tau)\right)^{-1}\right\|_{\infty\to\infty}
\le \sum_{n=0}^{\infty}2^{-n}=2.
\]
Multiplying by \(m_\psi(\tau)^{-1}\le m_0^{-1}\) gives
\[
\|A(V(\tau))^{-1}\|_{\infty\to\infty}\le \frac{2}{m_0}.
\]

\(\square\)

---

## T3. Cross-term control for the repaired modulation matrix

Assume the centering constraints
\[
\int_{\mathbb R^3}y_j|V(y,\tau)|^2\psi_R(y)\,dy=0,\qquad j=1,2,3.
\]
Assume also
\[
C_\nabla:=\|\,|y|\,\nabla\psi_R\|_{L^\infty(\mathbb R^3)}<\infty,
\]
\[
\int_{A_R}|V(y,\tau)|^2\,dy\le\varepsilon_0,
\]
and
\[
C_u:=\sup_{\tau\ge\tau_0}\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^3\,dy<\infty.
\]
Then the mixed blocks satisfy
\[
\|v(V(\tau))\|_\infty\le 2RC_\nabla\varepsilon_0,
\]
and
\[
\|u(V(\tau))\|_\infty\le p C_u.
\]
Consequently, if T2 gives
\[
\|A(V(\tau))^{-1}\|_{\infty\to\infty}\le C_A
\]
and
\[
\alpha(V(\tau))\ge \alpha_0>0,
\]
then
\[
|u(V(\tau))^T A(V(\tau))^{-1}v(V(\tau))|
\le 6p\,C_A\,C_u\,R C_\nabla\varepsilon_0.
\]
In particular, if
\[
6p\,C_A\,C_u\,R C_\nabla\varepsilon_0\le \frac{\alpha_0}{2},
\]
then
\[
\alpha(V(\tau))-u(V(\tau))^T A(V(\tau))^{-1}v(V(\tau))
\ge\frac{\alpha_0}{2}.
\]

### Proof

The repaired scale-row formula gives
\[
u_\ell(V)=M_{0\ell}(V)
=p\int_{\mathbb R^3}y_\ell |y|^{-p-2}|V|^3\,dy.
\]
Since \(|y_\ell|\le |y|\),
\[
|u_\ell(V(\tau))|
\le p\int_{\mathbb R^3}|y|^{-p-1}|V(y,\tau)|^3\,dy
\le pC_u.
\]
Therefore \(\|u(V(\tau))\|_\infty\le pC_u\).

For the translation-scale entries,
\[
v_j(V)=M_{j0}(V)
=2\int_{\mathbb R^3}y_jV\cdot(V+y\cdot\nabla V)\psi_R\,dy.
\]
Using \(V\cdot(y\cdot\nabla V)=\frac12 y\cdot\nabla(|V|^2)\),
\[
M_{j0}
=2\int y_j|V|^2\psi_R
+\int y_j\,y\cdot\nabla(|V|^2)\psi_R.
\]
Integrating by parts in the second term,
\[
\int y_j\,y\cdot\nabla(|V|^2)\psi_R
=-\int |V|^2\nabla\cdot(y_jy\psi_R).
\]
Since
\[
\nabla\cdot(y_jy\psi_R)=4y_j\psi_R+y_j(y\cdot\nabla\psi_R),
\]
one obtains
\[
M_{j0}
=-2\int y_j|V|^2\psi_R
-\int |V|^2y_j(y\cdot\nabla\psi_R).
\]
The centering condition cancels the first term, so
\[
M_{j0}
=-\int_{\mathbb R^3}|V|^2y_j(y\cdot\nabla\psi_R)\,dy.
\]
Because \(\nabla\psi_R\) is supported in \(A_R\), and \(|y_j|\le 2R\) on \(A_R\),
\[
|M_{j0}(V(\tau))|
\le 2R\,C_\nabla\int_{A_R}|V(y,\tau)|^2\,dy
\le 2R\,C_\nabla\varepsilon_0.
\]
This proves \(\|v(V(\tau))\|_\infty\le2RC_\nabla\varepsilon_0\).

Finally, using the duality of \(\ell^\infty\) and \(\ell^1\),
\[
|u^TA^{-1}v|
\le \|u\|_\infty\|A^{-1}v\|_1
\le 3\|u\|_\infty\|A^{-1}v\|_\infty.
\]
Also
\[
\|A^{-1}v\|_\infty
\le \|A^{-1}\|_{\infty\to\infty}\|v\|_\infty.
\]
Combining the bounds gives
\[
|u^TA^{-1}v|
\le
3(pC_u)C_A(2RC_\nabla\varepsilon_0)
=6p\,C_A\,C_u\,R C_\nabla\varepsilon_0.
\]
The Schur lower bound follows from the stated smallness condition.

\(\square\)

---

## T4. Full repaired-gauge nondegeneracy

Assume that, uniformly for \(\tau\ge\tau_0\),
\[
\|A(V(\tau))^{-1}\|_{\infty\to\infty}\le C_A,
\]
\[
\alpha(V(\tau))-u(V(\tau))^TA(V(\tau))^{-1}v(V(\tau))\ge \sigma_0>0,
\]
\[
\|u(V(\tau))\|_\infty\le C_u',\qquad
\|v(V(\tau))\|_\infty\le C_v'.
\]
Then \(M(V(\tau))\) is invertible for every \(\tau\ge\tau_0\), and
\[
\sup_{\tau\ge\tau_0}\|M(V(\tau))^{-1}\|_{\infty\to\infty}
\le C(C_A,\sigma_0,C_u',C_v'),
\]
where one may take
\[
C(C_A,\sigma_0,C_u',C_v')
=
\sigma_0^{-1}
+3\sigma_0^{-1}C_u'C_A
+\sigma_0^{-1}C_A C_v'
+C_A
+3C_A^2C_v'C_u'\sigma_0^{-1}.
\]

### Proof

Let
\[
S(\tau):=\alpha(V(\tau))-u(V(\tau))^TA(V(\tau))^{-1}v(V(\tau)).
\]
The Schur complement formula gives
\[
M^{-1}
=
\begin{pmatrix}
S^{-1} & -S^{-1}u^TA^{-1}\\
-A^{-1}vS^{-1} & A^{-1}+A^{-1}vS^{-1}u^TA^{-1}
\end{pmatrix}.
\]
The scalar block satisfies \(|S^{-1}|\le\sigma_0^{-1}\). For the upper-right block,
\[
\|S^{-1}u^TA^{-1}\|_{\infty\to\infty}
\le 3\sigma_0^{-1}\|u\|_\infty\|A^{-1}\|_{\infty\to\infty}
\le 3\sigma_0^{-1}C_u'C_A.
\]
For the lower-left block,
\[
\|A^{-1}vS^{-1}\|_{\infty\to\infty}
\le \sigma_0^{-1}C_A C_v'.
\]
For the lower-right block,
\[
\|A^{-1}+A^{-1}vS^{-1}u^TA^{-1}\|_{\infty\to\infty}
\le C_A+3C_A^2C_v'C_u'\sigma_0^{-1}.
\]
Combining these block estimates yields the displayed uniform bound after increasing the constant if necessary.

\(\square\)

---

## T5. Uniform boundedness of \((a(\tau),b(\tau))\)

Assume the gauge constraints are differentiated along the renormalized flow and give the finite-dimensional system
\[
M(V(\tau))
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-F(V(\tau)),
\]
where \(F(V(\tau))\in\mathbb R^4\). Assume
\[
\sup_{\tau\ge\tau_0}\|M(V(\tau))^{-1}\|_{\infty\to\infty}\le C_M,
\]
and
\[
\sup_{\tau\ge\tau_0}\|F(V(\tau))\|_\infty\le C_F.
\]
Then
\[
\sup_{\tau\ge\tau_0}\max\{|a(\tau)|,\|b(\tau)\|_\infty\}\le C_M C_F.
\]
In particular,
\[
\sup_{\tau\ge\tau_0}(|a(\tau)|+|b(\tau)|)<\infty.
\]

### Proof

For each \(\tau\),
\[
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-M(V(\tau))^{-1}F(V(\tau)).
\]
Taking the \(\ell^\infty\) norm in \(\mathbb R^4\) gives
\[
\max\{|a(\tau)|,\|b(\tau)\|_\infty\}
\le
\|M(V(\tau))^{-1}\|_{\infty\to\infty}\|F(V(\tau))\|_\infty
\le C_MC_F.
\]
Taking the supremum in \(\tau\) proves the claim.

\(\square\)

---

## T6. Local \(L^2_\tau H^{-1}_{y,\mathrm{loc}}\) time regularity

Let \(I\subset\mathbb R\) be bounded. Assume on \(B_{2R}\times I\):
\[
V\in L^2(I;H^1(B_{2R})),
\]
\[
M_3:=\sup_{\tau\in I}\|V(\tau)\|_{L^3(B_{2R})}<\infty,
\]
\[
M_{P,2}:=\left\|\inf_{c\in\mathbb R}
\|P(\tau)-c\|_{L^2(B_R)}\right\|_{L^2_\tau(I)}<\infty,
\]
and
\[
M_{ab}:=\sup_{\tau\in I}(|a(\tau)|+|b(\tau)|)<\infty.
\]
By Corollary T1.6, the pressure hypothesis follows from T1.5.
Then
\[
\partial_\tau V\in L^2(I;H^{-1}(B_R)),
\]
and
\[
\|\partial_\tau V\|_{L^2(I;H^{-1}(B_R))}
\le
C(R,\nu,M_3,M_{ab})
\left(M_{P,2}
+\|V\|_{L^2(I;H^1(B_{2R}))}\right).
\]

### Proof

Let \(\varphi\in H^1_0(B_R)\) with \(\|\varphi\|_{H^1(B_R)}\le1\). The equation gives
\[
\partial_\tau V=\nu\Delta V-(V\cdot\nabla)V-\nabla P
+aV+a\,y\cdot\nabla V+b\cdot\nabla V.
\]
Each term is estimated in \(H^{-1}(B_R)\).

For the Laplacian,
\[
|\langle \Delta V,\varphi\rangle|
=\left|\int_{B_R}\nabla V:\nabla\varphi\,dy\right|
\le \|\nabla V\|_{L^2(B_R)}.
\]

For the transport term, using \(\nabla\cdot V=0\),
\[
(V\cdot\nabla)V=\nabla\cdot(V\otimes V),
\]
so
\[
\|(V\cdot\nabla)V\|_{H^{-1}(B_R)}
\le \|V\otimes V\|_{L^2(B_R)}
=\|V\|_{L^4(B_R)}^2.
\]
By interpolation and Sobolev on \(B_{2R}\),
\[
\|V\|_{L^4(B_R)}
\le \|V\|_{L^3(B_R)}^{1/2}\|V\|_{L^6(B_R)}^{1/2}
\le C_R M_3^{1/2}\|V\|_{H^1(B_{2R})}^{1/2},
\]
hence
\[
\|(V\cdot\nabla)V\|_{H^{-1}(B_R)}
\le C_RM_3\|V\|_{H^1(B_{2R})}.
\]

For pressure, choose a measurable \(c(\tau)\) realizing the infimum up to an arbitrarily small error. Since \(\nabla c(\tau)=0\),
\[
\|\nabla P\|_{H^{-1}(B_R)}
=\|\nabla(P-c(\tau))\|_{H^{-1}(B_R)}
\le \|P-c(\tau)\|_{L^2(B_R)}.
\]

For modulation terms,
\[
\|aV\|_{H^{-1}(B_R)}
\le |a|\,\|V\|_{L^2(B_R)}
\le M_{ab}\|V\|_{H^1(B_{2R})},
\]
\[
\|a\,y\cdot\nabla V\|_{H^{-1}(B_R)}
\le |a|\,R\|\nabla V\|_{L^2(B_R)}
\le R M_{ab}\|V\|_{H^1(B_{2R})},
\]
and, for each component,
\[
\|\partial_\ell V\|_{H^{-1}(B_R)}
\le \|V\|_{L^2(B_R)}.
\]
Therefore
\[
\|b\cdot\nabla V\|_{H^{-1}(B_R)}
\le C M_{ab}\|V\|_{H^1(B_{2R})}.
\]
Combining the estimates gives, for almost every \(\tau\in I\),
\[
\|\partial_\tau V(\tau)\|_{H^{-1}(B_R)}
\le
C(R,\nu,M_3,M_{ab})
\left(\inf_{c\in\mathbb R}\|P(\tau)-c\|_{L^2(B_R)}
+\|V(\tau)\|_{H^1(B_{2R})}\right).
\]
Taking the \(L^2(I)\) norm yields the stated estimate.

\(\square\)

### Remark T6.1. Pressure regularity required for \(H^{-1}\)

The pressure-tail estimate in T1 gives local \(L^{3/2}\) pressure control modulo constants. That control is sufficient for a \(W^{-1,3/2}\)-type pressure estimate, but it does not by itself imply
\(\nabla P\in H^{-1}(B_R)\). The \(H^{-1}\) theorem above therefore explicitly assumes local \(L^2_\tau L^2_y\) pressure control modulo constants.

Indeed, for \(\varphi\in H^1_0(B_R)\),
\[
\langle \nabla P,\varphi\rangle
=-\int_{B_R}(P-c)\,\nabla\cdot\varphi.
\]
Since \(\nabla\cdot\varphi\in L^2(B_R)\), the direct \(H^{-1}\) estimate requires
\[
P-c\in L^2(B_R).
\]
The embedding \(H^1_0(B_R)\hookrightarrow L^6(B_R)\) does not improve this pairing, because the derivative falls on \(\varphi\). Thus replacing T1.5 by only
\[
P-c\in L^\infty_\tau L^{3/2}_y(B_R)
\]
would weaken the conclusion to a non-Hilbert negative Sobolev estimate rather than the \(L^2_\tau H^{-1}_y\) estimate used in the Aubin--Lions closure.

---

## Combined conclusion

Under T1--T6, the pressure tail is controlled, the translation block is invertible, the mixed entries satisfy a Schur-compatible smallness criterion, the full repaired gauge matrix is uniformly invertible, the modulation parameters are bounded, and the local \(H^{-1}\) time-regularity estimate required by Aubin--Lions--Simon is available.
