# Local Nonconcentration And Regularity For Navier-Stokes

This section records the standard local PDE argument used before any
classification by blow-up rate of a possible three-dimensional Navier-Stokes
singularity.  A finite-time singularity must carry a nonzero amount of the critical
Caffarelli-Kohn-Nirenberg density in arbitrarily small backward parabolic
cylinders.  If that density vanishes at a singular point, the point is
regular by epsilon regularity.

Throughout, let \((u,p)\) be a suitable weak solution of the three-dimensional
incompressible Navier-Stokes equations on a time interval ending at \(T\):

```{math}
\partial_t u+(u\cdot\nabla)u+\nabla p=\nu\Delta u,
\qquad \nabla\cdot u=0,
```

with

```{math}
u\in L^\infty_tL^2_{x,\mathrm{loc}}\cap L^2_tH^1_{x,\mathrm{loc}},
\qquad p\in L^{3/2}_{\mathrm{loc}},
```

and satisfying the local energy inequality.  The pressure is understood modulo
functions of time.

## Singular Set

The singular set at time \(T\) is

```{math}
\Sigma(T)=\{x\in\mathbb R^3:
 u\text{ is not locally bounded in any backward parabolic neighborhood of }(x,T)\}.
```

A point outside \(\Sigma(T)\) is regular in the usual local sense.  If a
finite-time singularity occurs at time \(T\), then \(\Sigma(T)\neq\emptyset\).

For \(z_0=(x_0,T)\) and \(r>0\), write

```{math}
Q_r(z_0)=B_r(x_0)\times(T-r^2,T).
```

Define the scale-invariant local quantities

```{math}
C(z_0,r)=r^{-2}\int_{Q_r(z_0)} |u|^3\,dx\,dt,
```

and

```{math}
D(z_0,r)=r^{-2}\int_{T-r^2}^T\int_{B_r(x_0)}
|p(x,t)-(p)_{B_r(x_0)}(t)|^{3/2}\,dx\,dt.
```

The spatial mean in \(D\) fixes the pressure ambiguity in the form used by the
Caffarelli-Kohn-Nirenberg epsilon regularity criterion.

## Main Local Theorem

::::{prf:theorem} No singularity without local critical concentration
:label: thm-ns-local-nonconcentration-regularity

Let \((u,p)\) be a suitable weak solution on
\(\mathbb R^3\times(T-\delta,T)\) for some \(\delta>0\).
Suppose that for every \(x_0\in\Sigma(T)\),

```{math}
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0.
```

Then \(\Sigma(T)=\emptyset\).  In particular, no finite-time singularity occurs
at time \(T\).

::::

:::{prf:proof}
Fix \(x_0\in\Sigma(T)\).  By the assumed vanishing, choose \(r>0\) so small
that

```{math}
C((x_0,T),r)+D((x_0,T),r)<\varepsilon_{\mathrm{CKN}},
```

where \(\varepsilon_{\mathrm{CKN}}\) is the universal epsilon-regularity
threshold.  The CKN criterion implies that \((x_0,T)\) is regular,
contradicting \(x_0\in\Sigma(T)\).  Thus \(\Sigma(T)=\emptyset\).
\(\square\)
:::

## Positive Concentration Alternative

If the preceding vanishing condition fails, then there are
\(x_0\in\Sigma(T)\), \(\eta>0\), and \(r_n\downarrow0\) such that

```{math}
C((x_0,T),r_n)+D((x_0,T),r_n)\ge \eta.
```

The corresponding parabolic rescalings

```{math}
u_n(y,s)=r_n u(x_0+r_n y,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2 p(x_0+r_n y,T+r_n^2s)
```

are suitable weak solutions on expanding backward cylinders and retain a
nonzero critical local density on a fixed compact cylinder.  Any later profile
or blow-up analysis must start from such a sequence.

## Relation To Type I And Type II Analysis

This local argument does not prove Type I or Type II exclusion.  It proves only
the preliminary local fact that a finite-time singularity must first produce a
nonzero local critical concentration.

Once such concentration is obtained, subsequent arguments may distinguish Type I
and Type II behavior by the relevant blow-up rate.  Type I analysis studies
self-similar-rate concentration and ancient limits.  Type II analysis studies
non-self-similar concentration.  Neither downstream analysis is responsible for
the vanishing-density case, because that case is already ruled out here by local
epsilon regularity.

## Files

| Note | Purpose |
|---|---|
| [Suitable weak solutions](suitable_weak_solutions.md) | The local solution class and the singular set at time \(T\). |
| [Parabolic rescaling](parabolic_rescaling.md) | The Navier-Stokes scaling centered at a singular point. |
| [Local critical quantities](local_critical_quantities.md) | Definition and scaling of \(C(z_0,r)\) and \(D(z_0,r)\). |
| [Local vanishing](local_vanishing.md) | Equivalent formulations of vanishing critical density. |
| [Positive local concentration](positive_local_concentration.md) | Failure of local vanishing gives a positive concentration sequence. |
| [The local alternative](local_alternative.md) | The elementary dichotomy: vanishing or positive concentration. |
| [Epsilon regularity](epsilon_regularity.md) | The CKN criterion in pressure-normalized form. |
| [Vanishing implies regularity](vanishing_implies_regularity.md) | Proof that local vanishing empties the singular set. |
| [Noncompact escape](noncompact_escape.md) | Escape outside compact rescaled cylinders is irrelevant when the CKN density vanishes. |
| [No-concentration regularity statement](no_concentration_regularity.md) | The local regularity theorem for the no-concentration case. |
| [Initial step to blow-up-rate analysis](blowup_analysis_entry.md) | How positive local concentration enters Type I or Type II analysis. |
