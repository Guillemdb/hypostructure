# S14: terminal sequence routing

S13 uses the routing certificate

```{math}
K_{\mathrm{TermSeqFromOrbit}}^+.
```

This note discharges that routing certificate from repaired-gauge
representation and the terminal-camera construction used in S8.

## Terminal camera construction payload

::::{prf:definition} Terminal camera construction certificate
:label: def-s14-terminal-camera-construction

The certificate

```{math}
K_{\mathrm{TermCamConstruct}}^+
```

means that every terminal active camera used by S8 is constructed from a
represented repaired-gauge Type II orbit \((V,P,a,b)\) by:

1. choosing sampling times \(\tau_n\to\infty\);
2. choosing Navier-Stokes symmetry parameters \((\lambda_n,x_n)\);
3. taking terminal data by applying the critical camera map to the sampled
   state \(V(\tau_n)\);
4. grouping comparable same-point profiles into compound profiles before
   selecting terminal cameras.

No data external to the represented orbit is inserted into the terminal
profile sequence.

::::

The fourth item is the S8 terminal-camera convention: nonterminal cameras may
still see smaller active bubbles, while terminal cameras are attached to
minimal-scale active clusters after comparable profiles are grouped.

::::{prf:theorem} Terminal camera construction is built into the declared terminal backend
:label: thm-s14-terminal-camera-construction-built-in

Inside the S8 terminal active-camera backend,

```{math}
K_{\mathrm{TermCamConstruct}}^+
```

holds by construction.

::::

:::{prf:proof}
S8 defines terminal active cameras only after active profiles have been
partitioned by physical concentration point, comparable same-point profiles
have been grouped into compound profiles, and a minimal-scale active cluster
has been selected. The camera parameters are exactly the center and scale of
that compound cluster, and the camera data are obtained by applying the
Navier-Stokes critical camera map to the represented branch on the chosen
sampling sequence.

Thus every terminal camera used by S8 satisfies the four clauses of Definition
{prf:ref}`def-s14-terminal-camera-construction`. No independent analytic
estimate is assumed here; this is the construction rule defining the declared
terminal backend. \(\square\)
:::

## Routing theorem

::::{prf:theorem} S14 terminal camera routing
:label: thm-s14-terminal-camera-routing

Assume the declared S8 terminal active-camera backend and

```{math}
K_{\mathrm{RepBridge}}^+.
```

Then

```{math}
K_{\mathrm{TermSeqFromOrbit}}^+.
```

::::

:::{prf:proof}
\(K_{\mathrm{RepBridge}}^+\) supplies the repaired-gauge represented orbit
\((V,P,a,b)\) and the physical-to-renormalized chart maps. Theorem
{prf:ref}`thm-s14-terminal-camera-construction-built-in` supplies
\(K_{\mathrm{TermCamConstruct}}^+\) inside the declared S8 terminal backend.
Therefore every terminal active-camera sequence in S8 is built by sampling
this orbit at times \(\tau_n\to\infty\) and applying only Navier-Stokes
critical symmetries. Hence each terminal sequence has
the form

```{math}
u_{0,n}=\Lambda_{\lambda_n,x_n}V(\tau_n)
```

or the equivalent inverse camera normalization. This is exactly
\(K_{\mathrm{TermSeqFromOrbit}}^+\). \(\square\)
:::

## Failure classification

::::{prf:corollary} Terminal sequence route defects
:label: cor-s14-terminal-sequence-route-defects

If a declared terminal active-camera sequence reaches S8/S12 but
\(K_{\mathrm{TermSeqFromOrbit}}^+\) is not emitted, then the upstream defect is

```{math}
K_{\mathrm{RepBridge}}^-.
```

Outside the declared S8 terminal backend, a malformed camera construction is
recorded separately as \(K_{\mathrm{TermCamConstruct}}^-\); it is not a Type II
singularity class.

::::

:::{prf:proof}
Theorem {prf:ref}`thm-s14-terminal-camera-routing` proves the positive route
from \(K_{\mathrm{RepBridge}}^+\) in the declared terminal backend. Therefore
failure of \(K_{\mathrm{TermSeqFromOrbit}}^+\) at this stage is exactly failure
of the repaired-gauge representation bridge. A construction failure means the
sequence was not one of the terminal cameras admitted by S8. \(\square\)
:::

## Consequence for S13 and C18

::::{prf:corollary} S14 discharges S13 routing
:label: cor-s14-discharges-s13-routing

In S13 and C18, the hypothesis

```{math}
K_{\mathrm{TermSeqFromOrbit}}^+
```

may be replaced by

```{math}
K_{\mathrm{RepBridge}}^+
```

inside the declared S8 terminal active-camera backend.

::::

:::{prf:proof}
Apply Theorem {prf:ref}`thm-s14-terminal-camera-routing`. \(\square\)
:::
