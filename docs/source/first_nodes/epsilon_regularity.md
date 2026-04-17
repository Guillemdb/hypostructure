# Epsilon Regularity

The local regularity input is the Caffarelli-Kohn-Nirenberg epsilon regularity
criterion, stated with the same pressure normalization as the quantities \(C\)
and \(D\).

::::{prf:theorem} Caffarelli-Kohn-Nirenberg epsilon regularity
:label: thm-ns-ckn-epsilon-regularity

There exists a universal constant \(\varepsilon_{\mathrm{CKN}}>0\) with the
following property.  Let \((u,p)\) be a suitable weak solution in \(Q_r(z_0)\).
If

```{math}
C(z_0,r)+D(z_0,r)<\varepsilon_{\mathrm{CKN}},
```

then \(u\) is locally bounded in a smaller cylinder, for instance in
\(Q_{r/2}(z_0)\).  Consequently \(z_0\) is a regular point.

::::

Here

```{math}
D(z_0,r)=r^{-2}\int_{t_0-r^2}^{t_0}\int_{B_r(x_0)}
|p(x,t)-(p)_{B_r(x_0)}(t)|^{3/2}\,dx\,dt.
```

The pressure normalization is part of the statement.  Equivalent formulations
using another pressure representative are reduced to this mean-subtracted form
by the standard pressure-normalization estimates.

This theorem is local.  It does not imply global regularity, scattering, or any
classification of blow-up rates.
