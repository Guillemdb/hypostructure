# Noncompact Escape In Rescaled Variables

When one rescales around a fixed physical point \((x_0,T)\), mass may leave
every fixed compact set in the rescaled variables.  Such behavior is not by
itself a local singularity mechanism at \((x_0,T)\).

The local regularity question at \((x_0,T)\) is governed by the quantities
\(C((x_0,T),r)\) and \(D((x_0,T),r)\).  If these quantities tend to zero, then
\((x_0,T)\) is regular by epsilon regularity, regardless of what a rescaled
sequence does outside fixed compact cylinders.

::::{prf:proposition} Escape outside compact rescaled cylinders does not replace local concentration
:label: prop-ns-noncompact-escape-local-regularity

Let \((u,p)\) be suitable near \((x_0,T)\).  Suppose

```{math}
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0.
```

Then \((x_0,T)\) is regular.  In particular, any loss of mass to spatial
infinity in the rescaled variables cannot produce a singularity at \((x_0,T)\)
in the absence of nonzero compact-cylinder CKN density.

::::

:::{prf:proof}
The hypothesis gives a scale at which \(C+D\) is below the universal CKN
threshold.  Epsilon regularity then gives local boundedness near \((x_0,T)\).
\(\square\)
:::

No dispersive scattering theorem, asymptotic completeness statement, or decay
at spatial infinity is used here.
