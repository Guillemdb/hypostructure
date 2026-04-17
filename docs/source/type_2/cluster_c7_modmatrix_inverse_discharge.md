# C7 defect discharge: modulation-matrix inverse from repaired-gauge nondegeneracy

This note discharges the C7 modulation-matrix inverse defect by packaging the
already proved repaired-gauge nondegeneracy inputs T2--T4 into the rough-core
certificate language.

The C7 rough-core bridge needs

```{math}
K_{\mathrm{ModMatrixInv}}^+:
\qquad
\sup_{\tau\ge\tau_0}\|M(V(\tau))^{-1}\|_{\infty\to\infty}<\infty,
```

where the repaired modulation matrix is written in block form

```{math}
M(V)=
\begin{pmatrix}
\alpha(V)&u(V)^T\\
v(V)&A(V)
\end{pmatrix}.
```

Here \(\alpha=M_{00}\) is the repaired scale-scale row, \(A\) is the
translation block, and \(u,v\) are the scale-translation and translation-scale
mixed blocks.

## Modulation inverse payload

::::{prf:definition} Modulation inverse payload
:label: def-c7-modmatrix-payload

For a represented repaired-gauge Type II orbit, define
\(K_{\mathrm{ModMatrixPayload}}^+\) to be the conjunction of the following
uniform-in-\(\tau\) certificates.

1. Repaired scale transversality:

```{math}
K_{\mathrm{ScaleTransv}}^+:
\qquad
\alpha(V(\tau))=DG_{\mathrm{sc}}(V(\tau))[Z_{\mathrm{sc}}(V(\tau))]
\ge \alpha_0>0.
```

2. Translation-block inverse:

```{math}
K_{\mathrm{TransBlockInv}}^+:
\qquad
\|A(V(\tau))^{-1}\|_{\infty\to\infty}\le C_A.
```

3. Mixed-block boundedness:

```{math}
K_{\mathrm{MixedBlockBd}}^+:
\qquad
\|u(V(\tau))\|_\infty\le C_u',
\qquad
\|v(V(\tau))\|_\infty\le C_v'.
```

4. Schur gap:

```{math}
K_{\mathrm{SchurGap}}^+:
\qquad
S(V(\tau))
:=\alpha(V(\tau))-u(V(\tau))^TA(V(\tau))^{-1}v(V(\tau))
\ge \sigma_0>0.
```

The ordered defect \(K_{\mathrm{ModMatrixInv}}^-\) is emitted only when this
payload is unavailable or the resulting full inverse bound fails.

::::

The scale-transversality certificate is supplied by the repaired weighted scale
gauge: on the gauge surface,
\(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0>0\). The translation-block,
mixed-block, and Schur-gap inputs are exactly the T2--T3 nondegeneracy payloads
recorded in [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md)
and [required_new_scale_gauge_theorems.md](required_new_scale_gauge_theorems.md).

## Schur-complement inverse bound

::::{prf:lemma} Uniform inverse from the modulation payload
:label: lem-c7-modmatrix-payload-implies-inv

For every represented repaired-gauge Type II orbit,

```{math}
K_{\mathrm{ModMatrixPayload}}^+
\Longrightarrow
K_{\mathrm{ModMatrixInv}}^+.
```

More explicitly, if the constants in Definition
{prf:ref}`def-c7-modmatrix-payload` are available, then

```{math}
\sup_{\tau\ge\tau_0}\|M(V(\tau))^{-1}\|_{\infty\to\infty}
\le
\sigma_0^{-1}
+3\sigma_0^{-1}C_u'C_A
+\sigma_0^{-1}C_A C_v'
+C_A
+3C_A^2C_v'C_u'\sigma_0^{-1}.
```

::::

:::{prf:proof}
For each \(\tau\), write

```{math}
M(V(\tau))=
\begin{pmatrix}
\alpha&u^T\\
v&A
\end{pmatrix},
\qquad
S=\alpha-u^TA^{-1}v.
```

By \(K_{\mathrm{TransBlockInv}}^+\), \(A^{-1}\) exists and is uniformly
bounded. By \(K_{\mathrm{SchurGap}}^+\), \(S^{-1}\) exists and
\(|S^{-1}|\le\sigma_0^{-1}\). The block inverse formula gives

```{math}
M^{-1}=
\begin{pmatrix}
S^{-1} & -S^{-1}u^TA^{-1}\\
-A^{-1}vS^{-1} & A^{-1}+A^{-1}vS^{-1}u^TA^{-1}
\end{pmatrix}.
```

Using the \(\ell^\infty\) matrix norm and the finite dimension of the mixed
blocks,

```{math}
\|S^{-1}u^TA^{-1}\|_{\infty\to\infty}
\le 3\sigma_0^{-1}C_u'C_A,
```

```{math}
\|A^{-1}vS^{-1}\|_{\infty\to\infty}
\le \sigma_0^{-1}C_A C_v',
```

and

```{math}
\|A^{-1}vS^{-1}u^TA^{-1}\|_{\infty\to\infty}
\le 3C_A^2C_v'C_u'\sigma_0^{-1}.
```

Adding the four block estimates gives the displayed uniform bound. This is
exactly \(K_{\mathrm{ModMatrixInv}}^+\). \(\square\)
:::

## T2--T4 route to the payload

::::{prf:lemma} Repaired-gauge nondegeneracy emits the payload
:label: lem-c7-t2-t4-emits-modmatrix-payload

Assume the represented orbit satisfies the repaired scale gauge and the T2--T3
hypotheses from [cluster_next_easiest_theorems.md](cluster_next_easiest_theorems.md):

1. the weighted scale gauge is on its surface \(G_{\mathrm{sc}}(V)=0\);
2. the centering translation block has the T2 core-mass lower bound and annular
   error smallness;
3. the mixed blocks satisfy the T3 weighted first-moment bound and
   Schur-compatible smallness condition.

Then \(K_{\mathrm{ModMatrixPayload}}^+\) holds.

::::

:::{prf:proof}
The repaired scale-gauge theorem gives
\(\alpha(V)=DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0\), hence
\(K_{\mathrm{ScaleTransv}}^+\). T2 gives
\(K_{\mathrm{TransBlockInv}}^+\). T3 gives the mixed-block bounds and the lower
bound

```{math}
\alpha(V(\tau))-u(V(\tau))^TA(V(\tau))^{-1}v(V(\tau))
\ge \frac{\alpha_0}{2},
```

which is \(K_{\mathrm{SchurGap}}^+\) with \(\sigma_0=\alpha_0/2\). These are
precisely the four parts of Definition {prf:ref}`def-c7-modmatrix-payload`.
\(\square\)
:::

## C7 consequence

::::{prf:corollary} C7 modulation inverse discharge
:label: cor-c7-modmatrix-inv-discharge

For every represented repaired-gauge Type II orbit,

```{math}
K_{\mathrm{ModMatrixPayload}}^+
\Longrightarrow
K_{\mathrm{ModMatrixInv}}^+.
```

Consequently, on the C-series route after C3 has discharged
\(K_{L^3\mathrm{Bd}}^-\), the rough-core bridge no longer treats
\(K_{\mathrm{ModMatrixInv}}^-\) as an independent remaining defect whenever the
T2--T4 repaired-gauge nondegeneracy payload is present.

::::

:::{prf:proof}
This is Lemma {prf:ref}`lem-c7-modmatrix-payload-implies-inv`, with the payload
supplied by Lemma {prf:ref}`lem-c7-t2-t4-emits-modmatrix-payload` when the
explicit T2--T3 hypotheses are used. \(\square\)
:::

## Updated ordered defects

After the C3 bounded-critical-norm discharge and this modulation-matrix inverse
discharge, the remaining nontrivial rough-core bridge defects on represented
C-series branches are:

```{math}
K_{\mathrm{ScaleForceBd}}^-,
\qquad
K_{\mathrm{CaccioppoliReg}}^-,
```

together with any failure of the independent local translation-force hypotheses
used by T14 to emit \(K_{\mathrm{TransForceBd}}^+\). If representation is not
imported from C2/C5, then \(K_{\mathrm{RepBridge}}^-\) remains the first ordered
C7 defect. If the repaired-gauge nondegeneracy payload above is not imported,
then \(K_{\mathrm{ModMatrixInv}}^-\) remains as the corresponding gauge-layer
defect.
