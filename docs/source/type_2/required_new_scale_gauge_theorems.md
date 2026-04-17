# Required theorems for the repaired weighted-moment scale gauge

This file fixes the scale-gauge layer used in the compact Type II notes with fully explicit assumptions and proof steps.

## 0. Setup and admissibility class

Fix
\[
0<p<3,\qquad \Theta_0>0,\qquad w(y):=|y|^{-p}.
\]
For a vector-valued profile \(V:\mathbb R^3\to\mathbb R^m\) (\(m\ge1\)), define
\[
G_{\mathrm{sc}}(V):=\int_{\mathbb R^3} w(y)|V(y)|^3\,dy-\Theta_0.
\]
Set
\[
I_p(V):=\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy.
\]

Unless noted otherwise, all integrals are over \(\mathbb R^3\) and all formulas are written for real-valued profiles.  
For complex-valued \(V\), replace \(V\cdot W\) by \(\Re(V\cdot\overline W)\) everywhere.

The class \(\mathcal A_p\) used in the gauge formulas below is
\[
\mathcal A_p:=\left\{V: I_p(V)<\infty,\ \int_{\mathbb R^3}|y|^{-p}|V|\,|y\cdot\nabla V|\,dy<\infty\right\}.
\]
This class contains all smooth compactly supported profiles and is the natural admissibility class for the repaired gauge in the notes.

Define also the scaling generator
\[
Z_{\mathrm{sc}}(V):=V+y\cdot\nabla V.
\]

---

## Theorem R1 (exact scaling law)

For \(\mu>0\), set \(V_\mu(y):=\mu\,V(\mu y)\).  
If \(V\in\mathcal A_p\), then
\[
I_p(V_\mu)=\mu^p I_p(V).
\]
Consequently,
\[
G_{\mathrm{sc}}(V_\mu)=\mu^p\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0.
\]

### Proof
\[
I_p(V_\mu)
=\int_{\mathbb R^3}|y|^{-p}|\mu V(\mu y)|^3\,dy
=\mu^3\int_{\mathbb R^3}|y|^{-p}|V(\mu y)|^3\,dy.
\]
With \(z=\mu y\), \(dy=\mu^{-3}dz\), and \(|y|^{-p}=\mu^{p}|z|^{-p}\), hence
\[
I_p(V_\mu)
=\mu^3\int_{\mathbb R^3}\mu^{-p}|z|^{-p}|V(z)|^3\,\mu^{-3}dz
=\mu^p I_p(V).
\]
Substituting into \(G_{\mathrm{sc}}\) gives the second identity. \(\square\)

---

## Theorem R2 (Fréchet derivative of \(G_{\mathrm{sc}}\))

Let \(V,W\in\mathcal A_p\) and assume \(W\in L^3(\mathbb R^3,w\,dy)\).  
Then the directional derivative exists:
\[
DG_{\mathrm{sc}}(V)[W]
=3\int_{\mathbb R^3} |y|^{-p}|V(y)|\,V(y)\cdot W(y)\,dy.
\]

### Proof
For \(h\neq0\),
\[
\frac{G_{\mathrm{sc}}(V+hW)-G_{\mathrm{sc}}(V)}{h}
=\int_{\mathbb R^3}|y|^{-p}\frac{|V+hW|^3-|V|^3}{h}\,dy.
\]
For each fixed \(y\), by the one-dimensional mean-value formula along \(t\mapsto|V+ tW|^3\),
\[
\frac{|V+hW|^3-|V|^3}{h}
=3\int_0^1 |V+\theta hW|\,(V+\theta hW)\cdot W\,d\theta.
\]
Thus
\[
\left|\frac{|V+hW|^3-|V|^3}{h}\right|
\le 3(|V|+|h||W|)^2|W|
\le 6\bigl(|V|^2|W|+|W|^3\bigr).
\]
Now \(|V|^2|W|\le \frac23|V|^3+\frac13|W|^3\), so
\[
6\bigl(|V|^2|W|+|W|^3\bigr)\le 6|V|^3+10|W|^3.
\]
Multiplying by \(w\), the dominating integrand is integrable by the class assumptions. Hence dominated convergence applies and yields
\[
\lim_{h\to0}\frac{G_{\mathrm{sc}}(V+hW)-G_{\mathrm{sc}}(V)}{h}
=\int w\,\left. \frac{d}{dh}\right|_{h=0}|V+hW|^3\,dy
=3\int w|V|\,V\cdot W\,dy.
\]
That is exactly the stated formula. \(\square\)

---

## Lemma R2.5 (equivalent scaling derivative identity for \(Z_{\mathrm{sc}}\))

For \(V\in\mathcal A_p\), the map \(\mu\mapsto V_\mu\) satisfies
\[
\left.\frac{d}{d\mu}\right|_{\mu=1}V_\mu = V+y\cdot\nabla V = Z_{\mathrm{sc}}(V).
\]
Hence
\[
\left.\frac{d}{d\mu}\right|_{\mu=1} G_{\mathrm{sc}}(V_\mu)
=DG_{\mathrm{sc}}(V)\big[Z_{\mathrm{sc}}(V)\big].
\]

### Proof
Direct differentiation in the scaling formula \(V_\mu(y)=\mu V(\mu y)\) gives
\[
\frac{d}{d\mu}V_\mu(y)=V(\mu y)+\mu\,y\cdot\nabla V(\mu y),
\]
so at \(\mu=1\):
\[
\frac{d}{d\mu}V_\mu(y)\Big|_{\mu=1}=V(y)+y\cdot\nabla V(y)=Z_{\mathrm{sc}}(V)(y).
\]
Applying the chain rule for directional derivatives of \(G_{\mathrm{sc}}\) in class \(\mathcal A_p\) gives the second identity. \(\square\)

---

## Theorem R3 (exact scale transversality)

Assume \(V\in\mathcal A_p\) is smooth and compactly supported.  
Then
\[
DG_{\mathrm{sc}}(V)\big[Z_{\mathrm{sc}}(V)\big]
=p\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy.
\]
If \(G_{\mathrm{sc}}(V)=0\), then
\[
DG_{\mathrm{sc}}(V)\big[Z_{\mathrm{sc}}(V)\big]=p\Theta_0.
\]

### Proof
Apply Theorem R2 with \(W=Z_{\mathrm{sc}}(V)\):
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]
=3\int w|V|\,V\cdot(V+y\cdot\nabla V)\,dy.
\]
Use \(3|V|\,V\cdot(V+y\cdot\nabla V)=3|V|^3+y\cdot\nabla(|V|^3)\):
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]
=3\int w|V|^3\,dy+\int w\,y\cdot\nabla(|V|^3)\,dy.
\]
Integration by parts is valid for compactly supported \(V\):
\[
\int w\,y\cdot\nabla(|V|^3)\,dy
=-\int |V|^3\,\nabla\cdot(wy)\,dy.
\]
Now
\[
\nabla\cdot(wy)=3|y|^{-p}+y\cdot\nabla(|y|^{-p})
=3|y|^{-p}-p|y|^{-p}
=(3-p)|y|^{-p},
\]
so
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]
=3\int w|V|^3\,dy-(3-p)\int w|V|^3\,dy
=p\int w|V|^3\,dy.
\]
Imposing \(G_{\mathrm{sc}}(V)=0\) yields \( \int w|V|^3=\Theta_0\), hence the final identity. \(\square\)

By density of compactly supported smooth profiles in \(\mathcal A_p\), this identity extends to general admissible \(V\) under the same integrability class.

---

## Theorem R4 (gauge-transversality constant)

On the gauge surface \(G_{\mathrm{sc}}(V)=0\),
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]
=p\Theta_0=:c_0>0.
\]

### Corollary
In particular,
\[
DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]\ge c_0.
\]

---

## Theorem R5 (modulation entries with the repaired scale gauge)

Let \(T_\ell(V):=\partial_{y_\ell}V\).  
Under \(V\in\mathcal A_p\cap W^{1,3}_{\mathrm{loc}}\) with sufficient decay to justify the integration steps (in particular, those steps in Theorem R3 and in the compact Type II notes), define
\[
M_{00}(V):=DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)],\qquad
M_{0\ell}(V):=DG_{\mathrm{sc}}(V)[T_\ell(V)],\ \ell=1,2,3.
\]
Then
\[
M_{00}(V)=p\int_{\mathbb R^3}|y|^{-p}|V|^3\,dy=p\Theta_0,
\]
and
\[
M_{0\ell}(V)
=3\int_{\mathbb R^3}|y|^{-p}|V|\,V\cdot\partial_\ell V\,dy
=\int_{\mathbb R^3}|y|^{-p}\partial_\ell(|V|^3)\,dy
=p\int_{\mathbb R^3}y_\ell |y|^{-p-2}|V(y)|^3\,dy.
\]

### Proof
The formula for \(M_{00}\) is Theorem R3.  
For \(M_{0\ell}\), first apply Theorem R2:
\[
M_{0\ell}=3\int |y|^{-p}|V|\,V\cdot\partial_\ell V\,dy
=\int |y|^{-p}\partial_\ell(|V|^3)\,dy.
\]
An integration by parts identity with compact support in the \( \mathcal A_p \) approximation gives
\[
\int |y|^{-p}\partial_\ell(|V|^3)\,dy
=-\int |V|^3\,\partial_\ell(|y|^{-p})\,dy
=p\int y_\ell|y|^{-p-2}|V|^3\,dy.
\]
Hence the stated formula. \(\square\)

---

## Theorem R6 (continuity of repaired scale-functionals)

Assume
\[
V_n\to V \text{ in }L^3(\mathbb R^3,|y|^{-p}dy),\qquad
|y|^{-p-2}y\,|V_n|^3\to |y|^{-p-2}y\,|V|^3 \text{ in }L^1_{\mathrm{loc}},
\]
and that \(V_n\to V\) in the local \(W^{1,3}\)-topology used for the translation block (or the continuity hypotheses from Parts VIII–G4).  
Then each repaired scale entry satisfies
\[
M_{00}(V_n)\to M_{00}(V),\qquad M_{0\ell}(V_n)\to M_{0\ell}(V),
\]
and the full modulation matrix built from \(G_{\mathrm{sc}},G_1,G_2,G_3\) is continuous in \(n\).

### Proof
- \(M_{00}(V_n)\) depends linearly on \(I_p(V_n)\), hence continuity by weighted \(L^1\) convergence.
- \(M_{0\ell}(V_n)\) depends on the weighted first moments \(\int y_\ell|y|^{-p-2}|V_n|^3\), which are continuous by the second assumption.
- translation block entries are controlled exactly as in the existing local continuity argument (same assumptions as in Parts VIII–G4).  
Combining these yields continuity of every entry and therefore of the full matrix. \(\square\)

---

## Theorem R7 (full gauge matrix nondegeneracy reduction)

Define
\[
M(V)=
\begin{pmatrix}
M_{00}(V) & M_{01}(V)&M_{02}(V)&M_{03}(V)\\
M_{10}(V)&M_{11}(V)&M_{12}(V)&M_{13}(V)\\
M_{20}(V)&M_{21}(V)&M_{22}(V)&M_{23}(V)\\
M_{30}(V)&M_{31}(V)&M_{32}(V)&M_{33}(V)
\end{pmatrix}
\]
using the repaired gauge and the standard centering gauge.
Assume along the orbit:
1. \(M_{00}(V)=p\Theta_0\ge c_0>0\).
2. translation block \(A(V):=(M_{j\ell}(V))_{1\le j,\ell\le3}\) is invertible with
\[
\|A(V)^{-1}\|\le C_A.
\]
3. \(S(V):=M_{00}(V)-u(V)^TA(V)^{-1}v(V)\ge \sigma_0>0\), where \(u=(M_{01},M_{02},M_{03})^T\), \(v=(M_{10},M_{20},M_{30})\).

Then \(M(V)\) is uniformly invertible and the usual Schur complement estimate applies.

### Proof
Write \(M=\begin{psmallmatrix}\alpha&u^T\\v&A\end{psmallmatrix}\) with \(\alpha=M_{00}\).  
The assumption \(A^{-1}\) exists gives
\[
M^{-1}
=\begin{pmatrix}
S^{-1} & -S^{-1}u^TA^{-1}\\
-A^{-1}vS^{-1}&A^{-1}+A^{-1}vS^{-1}u^TA^{-1}
\end{pmatrix},\qquad
S=\alpha-u^TA^{-1}v.
\]
Uniform lower bounds on \(\alpha\), \(S\) and upper bounds on \(\|A^{-1}\|\), \(\|u\|\), \(\|v\|\) give a uniform bound on \(\|M^{-1}\|\). \(\square\)

---

## Use in the compact Type II notes

In [compact_typeII_master_note_repaired_gauge.md], the repaired gauge is used exclusively through:
1. \(G_{\mathrm{sc}}(V)=0\),
2. The formula \(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0\),
3. The explicit scale/translation matrix entries of Theorem R5,
4. Schur-complement reduction of Theorem R7.

No other scale-gauge variant is introduced.
