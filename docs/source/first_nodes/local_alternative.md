# The Local Alternative

At a fixed time, the CKN density gives an elementary alternative:
either it vanishes at every singular point, or it is bounded below
along a sequence of shrinking cylinders at some singular point.

::::{prf:proposition} Local vanishing versus positive concentration
:label: prop-ns-local-vanishing-positive-alternative

Let \((u,p)\) be a suitable weak solution on
\(\mathbb R^3\times(T-\delta,T)\) for some \(\delta>0\).
Exactly one of the following two statements holds.

1.  For every \(x_0\in\Sigma(T)\),

    ```{math}
    \limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0.
    ```

2.  There exist \(x_0\in\Sigma(T)\), \(\eta>0\), and \(r_n\downarrow0\) such
    that

    ```{math}
    C((x_0,T),r_n)+D((x_0,T),r_n)\ge\eta
    \qquad\text{for all }n.
    ```

::::

:::{prf:proof}
The first statement is a universal assertion over the singular set.
If it fails, then for some \(x_0\in\Sigma(T)\) the limsup is positive; choosing
a positive number below this limsup and a realizing sequence gives the second
statement.  Conversely, the second statement contradicts the first.
\(\square\)
:::

The first case is closed by CKN epsilon regularity.  The second case is the only
case that can enter a blow-up profile analysis.
