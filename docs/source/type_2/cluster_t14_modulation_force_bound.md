# T14: modulation forcing bounds

This note proves certificate-level sufficient conditions for
\[
K_{\mathrm{ModForceBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}\|F(V(\tau))\|_\infty<\infty,
\]
where the modulation forcing vector is defined by differentiating the repaired
scale and centering gauge constraints:
\[
F_k(V)
:=
DG_k(V)\big[\nu\Delta V-(V\cdot\nabla)V-\nabla P\big],
\qquad k=0,1,2,3.
\]

The centering rows are compactly supported and are estimated directly. The
translation estimate requires an independent pointwise local \(H^1\) input. It
is therefore not a substitute for the C6/C7 windowed \(H^1\) conclusion unless
that local gradient input is supplied by a separate regularity payload. The
weighted scale row requires an explicit weighted forcing hypothesis because the
functional
\[
G_{\mathrm{sc}}(V)=\int |y|^{-p}|V|^3\,dy-\Theta_0
\]
has a singular weight and cubic derivative.

---

## T14.1 (Translation-gauge forcing bound)

Let
\[
G_j(V)=\int_{\mathbb R^3}y_j|V(y)|^2\psi_R(y)\,dy,
\qquad j=1,2,3,
\]
where \(\psi_R\in C_c^\infty(B_{2R})\). Define
\[
\Phi_j(y):=y_j\psi_R(y).
\]
For
\[
H(V,P):=\nu\Delta V-(V\cdot\nabla)V-\nabla P,
\]
set
\[
F_j(V):=DG_j(V)[H(V,P)]
=2\int_{\mathbb R^3}\Phi_j\,V\cdot H(V,P)\,dy.
\]

Assume, uniformly for \(\tau\ge\tau_0\),
\[
\|V(\tau)\|_{L^3(B_{2R})}\le M_3,
\]
\[
\|\nabla V(\tau)\|_{L^2(B_{2R})}\le M_{\nabla},
\]
and
\[
\inf_{c\in\mathbb R}\|P(\tau)-c\|_{L^{3/2}(B_{2R})}\le M_P.
\]
Then
\[
\sup_{\tau\ge\tau_0}\max_{1\le j\le3}|F_j(V(\tau))|
\le
C(R,\nu,\psi_R)\left(
M_{\nabla}^2+M_3^2+M_3^3+M_PM_3
\right).
\]

### Proof

Fix \(j\). Since \(\Phi_j\) is compactly supported in \(B_{2R}\),
\[
F_j=2\nu I_\Delta-2I_{\mathrm{nl}}-2I_P,
\]
where
\[
I_\Delta:=\int \Phi_j V\cdot\Delta V,\qquad
I_{\mathrm{nl}}:=\int \Phi_j V\cdot (V\cdot\nabla V),
\qquad
I_P:=\int \Phi_j V\cdot\nabla P.
\]

For the Laplacian term, integrate by parts:
\[
I_\Delta
=-\int \Phi_j|\nabla V|^2-\int (\nabla\Phi_j\cdot\nabla V)\cdot V.
\]
Thus
\[
|I_\Delta|
\le
\|\Phi_j\|_{L^\infty}\|\nabla V\|_{L^2(B_{2R})}^2
+\|\nabla\Phi_j\|_{L^\infty}
\|V\|_{L^2(B_{2R})}\|\nabla V\|_{L^2(B_{2R})}.
\]
By Holder on the bounded set \(B_{2R}\),
\[
\|V\|_{L^2(B_{2R})}\le C_R\|V\|_{L^3(B_{2R})}\le C_RM_3.
\]
Hence
\[
|I_\Delta|\le C_R(M_{\nabla}^2+M_3M_{\nabla})
\le C_R(M_{\nabla}^2+M_3^2).
\]

For the nonlinear term, use
\[
V\cdot(V\cdot\nabla V)=\frac12 V\cdot\nabla(|V|^2)
\]
and \(\nabla\cdot V=0\):
\[
I_{\mathrm{nl}}
=\frac12\int \Phi_j V\cdot\nabla(|V|^2)
=-\frac12\int |V|^2 V\cdot\nabla\Phi_j.
\]
Therefore
\[
|I_{\mathrm{nl}}|
\le
\frac12\|\nabla\Phi_j\|_{L^\infty}\|V\|_{L^3(B_{2R})}^3
\le C_RM_3^3.
\]

For the pressure term, replace \(P\) by \(P-c(\tau)\). Since
\(\nabla\cdot V=0\),
\[
I_P
=\int \Phi_j V\cdot\nabla(P-c)
=-\int (P-c)V\cdot\nabla\Phi_j.
\]
Thus
\[
|I_P|
\le
\|\nabla\Phi_j\|_{L^\infty}
\|P-c\|_{L^{3/2}(B_{2R})}\|V\|_{L^3(B_{2R})}
\le C_RM_PM_3.
\]
Taking the infimum in \(c\) and then the supremum in \(\tau\) proves the
claim, with the bound
\[
C(R,\nu,\psi_R)\left(
M_{\nabla}^2+M_3^2+M_3^3+M_PM_3
\right).
\]

\(\square\)

The local gradient hypothesis in this theorem is an input, not a consequence
of the C7 rough-core conclusion. Thus T14.1 gives either an independent
translation-force certificate or an a-posteriori check on branches where
pointwise local \(H^1\) control is already known.

---

## T14.2 (Scale-gauge forcing bound under weighted forcing control)

Let
\[
G_{\mathrm{sc}}(V)=\int_{\mathbb R^3}|y|^{-p}|V|^3\,dy-\Theta_0,
\qquad 0<p<3.
\]
For
\[
H(V,P):=\nu\Delta V-(V\cdot\nabla)V-\nabla P,
\]
define
\[
F_0(V):=DG_{\mathrm{sc}}(V)[H(V,P)]
=3\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy.
\]
Assume the weighted forcing integrability certificate
\[
K_{\mathrm{ScaleForceBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}
\left|
\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\right|<\infty
\]
holds. Then
\[
\sup_{\tau\ge\tau_0}|F_0(V(\tau))|<\infty.
\]

### Proof

The displayed formula for \(F_0\) is the derivative formula for the repaired
weighted scale gauge. Therefore
\[
|F_0(V(\tau))|
\le
3\left|
\int_{\mathbb R^3}|y|^{-p}|V|V\cdot H(V,P)\,dy
\right|.
\]
Taking the supremum in \(\tau\) gives the result.

\(\square\)

---

## T14.3 (Full modulation forcing bound)

Assume:

1. the translation-gauge hypotheses of T14.1 hold for the centering radius
   \(R\);
2. \(K_{\mathrm{ScaleForceBd}}^+\) holds for the repaired weighted scale row.

Then
\[
K_{\mathrm{ModForceBd}}^+:
\qquad
\sup_{\tau\ge\tau_0}\|F(V(\tau))\|_\infty<\infty
\]
holds.

### Proof

The vector \(F(V)\in\mathbb R^4\) has the scale component \(F_0\) and the
three translation components \(F_j\), \(1\le j\le3\). T14.2 bounds the scale
component. T14.1 bounds the translation components. Taking the maximum of the
four bounds gives the asserted \(\ell^\infty\)-bound on \(F(V)\).

\(\square\)

---

## T14.4 (Certificate route for \(K_{\mathrm{ModForceBd}}^+\))

The modulation forcing certificate is emitted by the implication
\[
K_{\mathrm{TransForceBd}}^+
\wedge
K_{\mathrm{ScaleForceBd}}^+
\Longrightarrow
K_{\mathrm{ModForceBd}}^+,
\]
where \(K_{\mathrm{TransForceBd}}^+\) denotes the uniform translation-row
bound supplied by T14.1 under its independent local hypotheses.

The remaining nontrivial payload is \(K_{\mathrm{ScaleForceBd}}^+\). This is
the weighted scale-row forcing estimate and is separate from the purely
algebraic scale transversality theorem
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0.
\]
The pointwise scale-row payload is decomposed in
[cluster_c7_scale_force_bound.md](cluster_c7_scale_force_bound.md):
\[
K_{\mathrm{ScaleLapBd}}^+
\wedge K_{\mathrm{ScaleL4Mom}}^+
\wedge K_{\mathrm{ScalePressureBd}}^+
\Longrightarrow
K_{\mathrm{ScaleForceBd}}^+.
\]
There the nonlinear scale-row term is discharged from the weighted fourth-moment
certificate \(K_{\mathrm{ScaleL4Mom}}^+\).

Thus T14 reduces modulation-force boundedness to:

- local \(L^3\), local \(H^1\), and local \(L^{3/2}\) pressure control for the
  compactly supported centering rows, with the local \(H^1\) input supplied
  independently of the C7 conclusion;
- weighted Laplacian, weighted pressure, and weighted fourth-moment control for
  the repaired scale row.
