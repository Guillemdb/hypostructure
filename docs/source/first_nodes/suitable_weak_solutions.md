# Suitable Weak Solutions And Singular Points

We fix the local Navier-Stokes class used in the nonconcentration argument.
No global compactness, critical-space bound, or scale classification is used.

## Suitable Weak Solutions

Let \((u,p)\) solve the three-dimensional incompressible Navier-Stokes equations
on \(\mathbb R^3\times I\), where \(I\) is an open time interval.
We use the standard suitable weak solution class:

```{math}
u\in L^\infty_tL^2_{x,\mathrm{loc}}\cap L^2_tH^1_{x,\mathrm{loc}},
\qquad p\in L^{3/2}_{\mathrm{loc}},
\qquad \nabla\cdot u=0.
```

The pair \((u,p)\) solves the equations distributionally.  For a.e.
\(t_1<t_2\) in \(I\) and every nonnegative
\(\phi\in C_c^\infty(\mathbb R^3\times I)\), it satisfies

```{math}
\begin{aligned}
&\int_{\mathbb R^3} \frac{|u(x,t_2)|^2}{2}\phi(x,t_2)\,dx
+\nu\int_{t_1}^{t_2}\int_{\mathbb R^3}|\nabla u|^2\phi\,dx\,dt \\
&\le
\int_{\mathbb R^3} \frac{|u(x,t_1)|^2}{2}\phi(x,t_1)\,dx
+\int_{t_1}^{t_2}\int_{\mathbb R^3}
\frac{|u|^2}{2}(\partial_t\phi+\nu\Delta\phi)\,dx\,dt \\
&\quad
+\int_{t_1}^{t_2}\int_{\mathbb R^3}
\left(\frac{|u|^2}{2}+p\right)u\cdot\nabla\phi\,dx\,dt .
\end{aligned}
```

The pressure is determined only up to addition of a function of time.  All local
criteria below use spatially mean-subtracted pressure on the same ball as the
velocity integral.

## Singular Set At A Fixed Time

For a time \(T\), define

```{math}
\Sigma(T)=\{x\in\mathbb R^3:
 u\text{ is not locally bounded in any }Q_r(x,T)\}.
```

Here

```{math}
Q_r(x,T)=B_r(x)\times(T-r^2,T).
```

Equivalently, \(x\notin\Sigma(T)\) if there exist \(r>0\) and \(M<\infty\)
such that

```{math}
\|u\|_{L^\infty(Q_r(x,T))}\le M.
```

The set \(\Sigma(T)\) is the usual local singular set at time \(T\).  If
\(u\) has a finite-time singularity at \(T\), then \(\Sigma(T)\neq\emptyset\).

## Locality

All assertions in this argument are made at fixed physical points \((x,T)\).
Moving centers and modulated gauges belong to later blow-up or profile analysis,
after one has already shown that the critical local density does not vanish.
