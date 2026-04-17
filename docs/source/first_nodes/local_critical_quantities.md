# Local Critical Quantities

For \(z_0=(x_0,T)\) and \(r>0\), set

```{math}
C(z_0,r)=r^{-2}\int_{Q_r(z_0)} |u|^3\,dx\,dt,
```

and

```{math}
D(z_0,r)=r^{-2}\int_{T-r^2}^{T}\int_{B_r(x_0)}
|p(x,t)-(p)_{B_r(x_0)}(t)|^{3/2}\,dx\,dt.
```

These are the standard pressure-normalized critical quantities appearing in
Caffarelli-Kohn-Nirenberg type epsilon regularity criteria.  The subtraction of
the spatial mean fixes the pressure ambiguity by a convention compatible with
the local criterion.

::::{prf:proposition} Local finiteness and scaling
:label: prop-ns-local-critical-quantities

Let \((u,p)\) be a suitable weak solution near \(z_0\).  Then \(C(z_0,r)\) and
\(D(z_0,r)\) are finite for every sufficiently small \(r\).  Under the parabolic
rescaling of [parabolic_rescaling.md](parabolic_rescaling.md),

```{math}
\int_{Q_R}\left(|u^{(r)}|^3+
|p^{(r)}-(p^{(r)})_{B_R}(s)|^{3/2}\right)\,dy\,ds
=R^2\bigl(C(z_0,rR)+D(z_0,rR)\bigr).
```

::::

:::{prf:proof}
Local finiteness of the velocity term follows from

```{math}
u\in L^\infty_tL^2_{x,\mathrm{loc}}\cap L^2_tH^1_{x,\mathrm{loc}}
\subset L^3_{\mathrm{loc}}
```

on finite cylinders.  The pressure term is finite because
\(p\in L^{3/2}_{\mathrm{loc}}\) and subtracting a spatial mean over the same
ball preserves local \(L^{3/2}\) integrability.  The displayed identity follows
by the change of variables \(x=x_0+r y\), \(t=T+r^2s\); spatial pressure means
transform as

```{math}
(p^{(r)})_{B_R}(s)=r^2(p)_{B_{rR}(x_0)}(T+r^2s).
```

\(\square\)
:::
