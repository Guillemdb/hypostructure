# No-Concentration Implies Local Regularity

We prove the main local conclusion.

::::{prf:theorem} Vanishing CKN density empties the singular set at time \(T\)
:label: thm-ns-vanishing-density-regularity

Let \((u,p)\) be a suitable weak solution on
\(\mathbb R^3\times(T-\delta,T)\) for some \(\delta>0\).
Assume that for every \(x_0\in\Sigma(T)\),

```{math}
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)=0.
```

Then \(\Sigma(T)=\emptyset\).

::::

:::{prf:proof}
Assume for contradiction that \(x_0\in\Sigma(T)\).  By the hypothesis, there is
a radius \(r>0\) such that

```{math}
C((x_0,T),r)+D((x_0,T),r)<\varepsilon_{\mathrm{CKN}}.
```

The epsilon-regularity theorem of
[epsilon_regularity.md](epsilon_regularity.md) implies
that \((x_0,T)\) is regular.  This contradicts the definition of \(\Sigma(T)\).
Hence \(\Sigma(T)=\emptyset\).  \(\square\)
:::

The proof uses only suitable weak solutions, the pressure-normalized CKN
quantity, and local epsilon regularity.
