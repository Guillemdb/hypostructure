# Initial Step for Blow-Up-Rate Analysis

The local argument has only two outcomes.  Either the pressure-normalized CKN
density vanishes at every singular point, in which case the solution
is regular at time \(T\), or there is a positive concentration sequence

```{math}
C((x_0,T),r_n)+D((x_0,T),r_n)\ge\eta,
\qquad r_n\downarrow0,
```

at some \(x_0\in\Sigma(T)\).

The second outcome is the starting point for blow-up-rate analysis.  The
rescaled sequence

```{math}
u_n(y,s)=r_n u(x_0+r_n y,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2 p(x_0+r_n y,T+r_n^2s)
```

has a nonzero critical local density on a fixed compact cylinder, after passing
to a suitable subsequence if the relevant compactness theorem is available.

## Type I And Type II Separation

The subsequent distinction is the usual one.  A Type I analysis assumes a
self-similar-rate bound, for example

```{math}
\sup_{t<T}\sqrt{T-t}\,\|u(t)\|_{L^\infty}<\infty,
```

and studies ancient blow-up limits obtained from self-similar scaling.  A Type
II analysis treats concentration not controlled by the Type I rate and requires
separate profile, compactness, and rigidity arguments.

The local result supplies only the preliminary fact needed by
both analyses: any genuine finite-time singularity must first have positive
local CKN concentration.  It does not prove either the Type I ancient-limit
classification or the Type II exclusion theorem.
