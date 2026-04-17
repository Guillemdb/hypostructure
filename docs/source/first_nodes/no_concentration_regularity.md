# No-Concentration Regularity Statement

The following theorem records the local conclusion in the no-concentration case.

::::{prf:theorem} The no-concentration case is regular
:label: thm-ns-no-concentration-regularity

Let \((u,p)\) be a suitable weak solution on
\(\mathbb R^3\times(T-\delta,T)\) for some \(\delta>0\).  If
the pressure-normalized CKN density vanishes at every singular point,
namely

```{math}
\forall x_0\in\Sigma(T),\qquad
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0,
```

then \(\Sigma(T)=\emptyset\).  Thus no finite-time singularity occurs at \(T\)
in the no-concentration case.

::::

:::{prf:proof}
This is exactly
[vanishing_implies_regularity.md](vanishing_implies_regularity.md),
which applies the Caffarelli-Kohn-Nirenberg epsilon regularity theorem at each
singular point.  \(\square\)
:::

This is a local regularity statement.  It is not a scattering theorem and does
not address Type I or Type II concentration, which begin only after positive
local concentration has been found.
