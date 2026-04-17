# Local Vanishing

The no-concentration condition at time \(T\) is the pointwise vanishing of
the CKN density at every singular point.

::::{prf:definition} Local vanishing at time \(T\)
:label: def-ns-local-vanishing

We say that local critical concentration vanishes at time \(T\) if, for every
\(x_0\in\Sigma(T)\),

```{math}
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0.
```

::::

::::{prf:proposition} Equivalent rescaled formulation
:label: prop-ns-vanishing-rescaled-equivalence

The preceding condition is equivalent to the following statement: for every
\(x_0\in\Sigma(T)\) and every fixed \(R<\infty\), the rescaled solutions at
\((x_0,T)\) satisfy

```{math}
\lim_{r\downarrow0}\int_{Q_R}
\left(|u^{(r)}|^3+|p^{(r)}-(p^{(r)})_{B_R}(s)|^{3/2}\right)\,dy\,ds=0.
```

::::

:::{prf:proof}
By the scaling identity in
[local_critical_quantities.md](local_critical_quantities.md),
the rescaled integral over \(Q_R\) equals

```{math}
R^2\bigl(C((x_0,T),rR)+D((x_0,T),rR)\bigr).
```

Thus local vanishing of \(C+D\) implies vanishing on every fixed rescaled
cylinder.  Conversely, taking \(R=1\) gives the original pointwise condition.
\(\square\)
:::

This is a local statement at fixed physical points.  It is not a statement
about global compactness of the solution, spatial decay, or scattering.
