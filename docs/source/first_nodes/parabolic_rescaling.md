# Parabolic Rescaling At A Fixed Point

Let \(z_0=(x_0,T)\) and let \(r>0\).  Define the usual Navier-Stokes parabolic
rescaling

```{math}
u^{(r)}(y,s)=r u(x_0+r y,T+r^2s),
\qquad
p^{(r)}(y,s)=r^2 p(x_0+r y,T+r^2s).
```

The rescaled variables are defined for those \((y,s)\) for which
\((x_0+r y,T+r^2s)\) lies in the original domain.  If \(r\downarrow0\), fixed
cylinders \(B_R\times(-R^2,0)\) correspond to shrinking cylinders around
\((x_0,T)\).

::::{prf:proposition} Invariance under parabolic rescaling
:label: prop-ns-parabolic-rescaling

If \((u,p)\) is a suitable weak solution near \((x_0,T)\), then
\((u^{(r)},p^{(r)})\) is a suitable weak solution on the rescaled domain.  The
local energy inequality is preserved under this change of variables, and the
pressure remains defined modulo functions of the rescaled time.

::::

:::{prf:proof}
The distributional equations and the divergence condition follow from the
change of variables \(x=x_0+r y\), \(t=T+r^2s\).  The local energy inequality is
obtained by testing the original inequality with the corresponding rescaled
nonnegative test function.  The pressure transformation is the one dictated by
Navier-Stokes scaling, and addition of a function of \(t\) becomes addition of a
function of \(s\).  \(\square\)
:::

Compact sets in the rescaled variables represent the local behavior of the
original solution at the fixed physical point \((x_0,T)\).
