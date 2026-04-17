# T15--T17: integrated modulation forcing and good-time selection

This note gives the non-circular modulation route used by the good-window
Type II closure. It replaces pointwise modulation forcing bounds by
window-integrated forcing bounds and extracts good times with bounded
modulation.

The pointwise route remains:
\[
K_{\mathrm{ModMatrixInv}}^+
\wedge
K_{\mathrm{ModForceBd}}^+
\Longrightarrow
K_{\mathrm{ModBd}}^+.
\]
The integrated route is weaker. It gives bounded modulation on selected good
times, which is the form compatible with the good-window compactness mechanism.

---

## T15.1 (Window-integrated forcing gives window-integrated modulation)

Let the repaired-gauge modulation system be
\[
M(V(\tau))
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-F(V(\tau)).
\]
Assume
\[
K_{\mathrm{ModMatrixInv}}^+:
\qquad
\sup_{\tau\ge\tau_0}
\|M(V(\tau))^{-1}\|_{\infty\to\infty}\le C_M<\infty
\]
and
\[
K_{\mathrm{ModForceL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}\|F(V(\tau))\|_\infty\,d\tau
\le C_F<\infty.
\]
Then
\[
K_{\mathrm{ModL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}
\left(|a(\tau)|+\|b(\tau)\|_\infty\right)\,d\tau
<\infty.
\]
More precisely,
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}
\left(|a(\tau)|+\|b(\tau)\|_\infty\right)\,d\tau
\le
2C_MC_F.
\]

### Proof

For almost every \(\tau\),
\[
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=-M(V(\tau))^{-1}F(V(\tau)).
\]
Hence
\[
\max\{|a(\tau)|,\|b(\tau)\|_\infty\}
\le
C_M\|F(V(\tau))\|_\infty.
\]
Since
\[
|a(\tau)|+\|b(\tau)\|_\infty
\le
2\max\{|a(\tau)|,\|b(\tau)\|_\infty\},
\]
integration over \([T,T+1]\) gives
\[
\int_T^{T+1}
\left(|a(\tau)|+\|b(\tau)\|_\infty\right)\,d\tau
\le
2C_M\int_T^{T+1}\|F(V(\tau))\|_\infty\,d\tau
\le
2C_MC_F.
\]
Taking the supremum in \(T\) proves the claim.

\(\square\)

---

## T15.2 (Good-time modulation selection)

Let \(I_n=[T_n,T_n+1]\), with \(T_n\to\infty\). Let
\[
d_n:I_n\to[0,\infty),
\qquad
q_n:I_n\to[0,\infty)
\]
be measurable functions satisfying
\[
\int_{I_n}d_n(\tau)\,d\tau\to0
\]
and
\[
\sup_n\int_{I_n}q_n(\tau)\,d\tau\le Q<\infty.
\]
Then, after choosing measurable representatives, there exist times
\(\tau_n\in I_n\) such that
\[
d_n(\tau_n)\to0
\]
and
\[
q_n(\tau_n)\le 2Q
\]
for every \(n\) if \(Q>0\). If \(Q=0\), then one may choose
\(\tau_n\in I_n\) with \(q_n(\tau_n)=0\) and \(d_n(\tau_n)\to0\).

### Proof

Assume \(Q>0\). Define
\[
E_n:=\{\tau\in I_n:q_n(\tau)\le2Q\}.
\]
By Chebyshev's inequality,
\[
|I_n\setminus E_n|
\le
\frac{1}{2Q}\int_{I_n}q_n(\tau)\,d\tau
\le\frac12.
\]
Thus \(|E_n|\ge1/2\). Since \(d_n\ge0\),
\[
\inf_{\tau\in E_n}d_n(\tau)
\le
\frac{1}{|E_n|}\int_{E_n}d_n(\tau)\,d\tau
\le
2\int_{I_n}d_n(\tau)\,d\tau.
\]
Choose \(\tau_n\in E_n\) outside the null set on which the representatives are
undefined and such that
\[
d_n(\tau_n)
\le
2\int_{I_n}d_n(\tau)\,d\tau+\frac1n.
\]
Then \(d_n(\tau_n)\to0\), and \(q_n(\tau_n)\le2Q\).

If \(Q=0\), then \(q_n=0\) almost everywhere on \(I_n\). Apply the same
averaging argument to \(d_n\) on the full-measure set where \(q_n=0\).

\(\square\)

---

## T15.3 (Good-window modulation certificate)

Assume \(K_{\mathrm{ModL^1Win}}^+\). Let
\[
q(\tau):=|a(\tau)|+\|b(\tau)\|_\infty.
\]
For every sequence of unit windows \(I_n=[T_n,T_n+1]\) and every nonnegative
cost density \(d_n\) satisfying
\[
\int_{I_n}d_n(\tau)\,d\tau\to0,
\]
there exist \(\tau_n\in I_n\) such that
\[
d_n(\tau_n)\to0
\]
and
\[
\sup_n\left(|a(\tau_n)|+\|b(\tau_n)\|_\infty\right)<\infty.
\]

### Proof

Apply T15.2 with \(q_n=q|_{I_n}\). The uniform \(L^1\)-window bound for
\(q\) supplies the constant \(Q\). The selected times satisfy both desired
properties.

\(\square\)

---

## T15.4 (Product-form variant)

The pointwise inverse bound in T15.1 may be replaced by the product-form
certificate
\[
K_{\mathrm{ModProdL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}
\|M(V(\tau))^{-1}\|_{\infty\to\infty}
\|F(V(\tau))\|_\infty\,d\tau
<\infty.
\]
Then \(K_{\mathrm{ModL^1Win}}^+\) holds.

### Proof

The pointwise linear-system estimate gives
\[
|a(\tau)|+\|b(\tau)\|_\infty
\le
2\|M(V(\tau))^{-1}\|_{\infty\to\infty}
\|F(V(\tau))\|_\infty.
\]
Integrating on unit windows gives the result.

\(\square\)

---

## T16.1 (Integrated translation forcing bound)

Let \(G_j\), \(\Phi_j\), \(H(V,P)\), and \(F_j(V)\) be as in T14.1. Assume
that, for some \(R>0\),
\[
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(B_{2R})}\le M_3,
\]
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}\|\nabla V(\tau)\|_{L^2(B_{2R})}^2\,d\tau
\le A_R,
\]
and
\[
\sup_{\tau\ge\tau_0}
\inf_{c\in\mathbb R}
\|P(\tau)-c\|_{L^{3/2}(B_{2R})}\le M_P.
\]
Then
\[
K_{\mathrm{TransForceL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}\max_{1\le j\le3}|F_j(V(\tau))|\,d\tau
<\infty.
\]
More precisely,
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}\max_{1\le j\le3}|F_j(V(\tau))|\,d\tau
\le
C(R,\nu,\psi_R)(A_R+M_3^2+M_3^3+M_PM_3).
\]

### Proof

Fix \(j\). The proof of T14.1 gives, for almost every \(\tau\),
\[
|F_j(V(\tau))|
\le
C(R,\nu,\psi_R)
\left(
\|\nabla V(\tau)\|_{L^2(B_{2R})}^2
+M_3\|\nabla V(\tau)\|_{L^2(B_{2R})}
+M_3^3
+M_PM_3
\right).
\]
Integrate this estimate over \(I_T=[T,T+1]\). The first term contributes at
most \(A_R\). For the mixed term, Cauchy's inequality on the unit interval
gives
\[
\int_{I_T}M_3\|\nabla V(\tau)\|_{L^2(B_{2R})}\,d\tau
\le
M_3
\left(
\int_{I_T}\|\nabla V(\tau)\|_{L^2(B_{2R})}^2\,d\tau
\right)^{1/2}
\le
M_3A_R^{1/2}.
\]
Using \(M_3A_R^{1/2}\le (M_3^2+A_R)/2\), the mixed term is bounded by
\(C(A_R+M_3^2)\). The terms \(M_3^3\) and \(M_PM_3\) are constant on the unit
window. Therefore
\[
\int_T^{T+1}|F_j(V(\tau))|\,d\tau
\le
C(R,\nu,\psi_R)(A_R+M_3^2+M_3^3+M_PM_3).
\]
Taking the maximum over the three centering rows only changes the constant.

\(\square\)

---

## T16.2 (Non-circular source of the translation certificate)

The hypothesis
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}\|\nabla V(\tau)\|_{L^2(B_{2R})}^2\,d\tau<\infty
\]
in T16.1 must be supplied independently of the final \(K_{\mathrm{WinH1}}^+\)
conclusion whenever T16 is used to feed C7. Valid sources include:

1. a preliminary Caccioppoli estimate whose constants depend only on already
   available data;
2. an approximation or regularity payload that directly emits local
   \(L^2_\tau H^1_y\) control on the centering support;
3. an a-posteriori branch check after \(K_{\mathrm{WinH1}}^+\) has already
   been emitted by another route.

Thus T16 is non-circular exactly when the windowed gradient input is not taken
from the C7 conclusion being proved.

---

## T17.1 (Integrated scale-force payload)

Define
\[
K_{\mathrm{ScaleForceL^1Win}}^+:
\qquad
\sup_{T\ge\tau_0}
\int_T^{T+1}
\left|
\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\right|\,d\tau
<\infty.
\]
This certificate includes the assertion that the weighted pairing
\[
\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\]
is well defined for almost every \(\tau\) as an absolutely integrable function
or as a specified distributional pairing whose value is represented by an
\(L^1_{\mathrm{loc}}(d\tau)\) function.
Then the scale component
\[
F_0(V)=3\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\]
satisfies
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}|F_0(V(\tau))|\,d\tau<\infty.
\]

### Proof

This is immediate from the definition of \(F_0\) and the definition of
\(K_{\mathrm{ScaleForceL^1Win}}^+\).

\(\square\)

The decomposition and partial discharge of this payload are given in
[cluster_t18_scale_force_decomposition.md](cluster_t18_scale_force_decomposition.md):
\[
K_{\mathrm{ScaleDiffL^1Win}}^+
\wedge
K_{\mathrm{AnnConvReg}}^+
\wedge
K_{\mathrm{ScaleV4L^1Win}}^+
\wedge
K_{\mathrm{ScalePressL^1Win}}^+
\Longrightarrow
K_{\mathrm{ScaleForceL^1Win}}^+.
\]

---

## T17.2 (Full integrated modulation forcing)

Assume:

1. \(K_{\mathrm{TransForceL^1Win}}^+\);
2. \(K_{\mathrm{ScaleForceL^1Win}}^+\).

Then
\[
K_{\mathrm{ModForceL^1Win}}^+
\]
holds.

### Proof

The vector \(F(V)\in\mathbb R^4\) consists of \(F_0\) and the three
translation components \(F_j\), \(1\le j\le3\). Since
\[
\|F(V(\tau))\|_\infty
\le
|F_0(V(\tau))|+\max_{1\le j\le3}|F_j(V(\tau))|,
\]
integration on a unit window is bounded by the sum of the scale and
translation window integrals. T16.1 gives the window-integrated bound for the
translation components, while T17.1 gives the window-integrated bound for the
scale component. Therefore
\[
\sup_{T\ge\tau_0}
\int_T^{T+1}\|F(V(\tau))\|_\infty\,d\tau<\infty.
\]

\(\square\)

---

## T17.3 (Integrated good-window modulation route)

The following implication is valid:
\[
K_{\mathrm{ModMatrixInv}}^+
\wedge
K_{\mathrm{TransForceL^1Win}}^+
\wedge
K_{\mathrm{ScaleForceL^1Win}}^+
\Longrightarrow
K_{\mathrm{ModL^1Win}}^+.
\]
Consequently, by T15.3, every vanishing-cost sequence of unit windows contains
selected times with vanishing cost and uniformly bounded modulation.

### Proof

T17.2 gives \(K_{\mathrm{ModForceL^1Win}}^+\). T15.1 combines this with
\(K_{\mathrm{ModMatrixInv}}^+\) to give \(K_{\mathrm{ModL^1Win}}^+\). T15.3
then gives the selected good times.

\(\square\)
