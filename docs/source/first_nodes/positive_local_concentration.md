# Positive Local Concentration

The complement of local vanishing is positive concentration along a sequence of
shrinking cylinders.

::::{prf:lemma} Failure of local vanishing gives a positive concentration sequence
:label: lem-ns-positive-local-concentration

Suppose local vanishing at time \(T\) fails.  Then there exist
\(x_0\in\Sigma(T)\), a number \(\eta>0\), and radii \(r_n\downarrow0\) such
that

```{math}
C((x_0,T),r_n)+D((x_0,T),r_n)\ge\eta
\qquad\text{for all }n.
```

Equivalently, the rescaled sequence around \((x_0,T)\) has nonzero critical
local mass on at least one fixed compact cylinder.

::::

:::{prf:proof}
Negating the definition of local vanishing gives a point \(x_0\in\Sigma(T)\)
with positive limsup of \(C((x_0,T),r)+D((x_0,T),r)\) as \(r\downarrow0\).
Choose \(\eta>0\) below that limsup and pass to a sequence \(r_n\downarrow0\)
along which the displayed lower bound holds.  The rescaled formulation follows
from the scaling identity for \(C\) and \(D\).  \(\square\)
:::

## Optional Compactness Input

A later blow-up analysis may require a compactness theorem for the rescaled
sequence.  A typical local statement is the following.

::::{prf:assumption} Local profile compactness for positive concentration
:label: ass-ns-local-profile-compactness

Let \((u_n,p_n)\) be suitable weak solutions on compact subsets of
\(\mathbb R^3\times(-\infty,0)\), obtained by parabolic rescaling around a fixed
point \((x_0,T)\).  Suppose that on some fixed cylinder \(Q_R\),

```{math}
\int_{Q_R}\left(|u_n|^3+|p_n-(p_n)_{B_R}(s)|^{3/2}\right)\,dy\,ds\ge\eta>0.
```

If the available compactness theorem gives, after passing to a subsequence,
convergence to a suitable weak limit \((U,P)\) strong enough to pass this local
critical density, then the limit is nonzero in the sense that for some compact
cylinder \(Q_{R_1}\),

```{math}
\int_{Q_{R_1}}\left(|U|^3+|P-(P)_{B_{R_1}}(s)|^{3/2}\right)\,dy\,ds>0.
```

::::

This compactness input is not part of the epsilon-regularity argument.  It is
only the entry point for subsequent Type I or Type II blow-up analysis.
