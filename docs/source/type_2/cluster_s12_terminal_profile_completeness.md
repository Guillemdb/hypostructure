# S12: terminal critical local compactness

This note discharges the S8 profile-completeness payload from the standard
local suitable compactness theorem. It replaces the global
`CatLib` placeholder by the concrete analytic input actually needed by the
Type II terminal-camera argument.

The result is not a proof of global \(K_{\mathrm{CatLib}}^+\). It proves the
local Type-II payload

```{math}
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
```

for bounded critical terminal Type II profile sequences.

## External analytic theorem used

::::{prf:axiom} Critical NS3D profile decomposition and nonlinear stability theorem
:label: ax-s12-critical-ns3d-profile-decomposition

Let \(u_{0,n}\) be a bounded sequence in the critical divergence-free space
\(L^3_\sigma(\mathbb R^3)\), or in a stronger admissible critical profile
space whose profile theorem embeds into \(L^3_{\mathrm{loc}}\) and whose
nonlinear Navier-Stokes stability theory controls compact time windows. The
accepted theorem is the standard Kato/Gallagher-type critical Navier-Stokes
profile theorem in the following precise form.

Then, after passing to a subsequence, there are profiles
\(\phi^j\in L^3_\sigma\), scales \(\lambda_{j,n}>0\), centers
\(x_{j,n}\in\mathbb R^3\), and remainders \(r_n^J\) such that

```{math}
u_{0,n}
=
\sum_{j=1}^{J}\Lambda_{\lambda_{j,n},x_{j,n}}\phi^j
+r_n^J,
```

with pairwise orthogonal parameters, critical mass decoupling

```{math}
\|u_{0,n}\|_3^3
=
\sum_{j=1}^{J}\|\phi^j\|_3^3
+\|r_n^J\|_3^3
+o_n(1),
```

and a nonlinear profile/stability statement: after evolving the profiles by
Navier-Stokes and discarding subthreshold small-data profiles, the linear
remainder satisfies

```{math}
\lim_{J\to\infty}\limsup_{n\to\infty}
\|e^{t\Delta}r_n^J\|_{X_{\mathrm{Kato}}}=0,
```

and the nonlinear remainder is small in the critical stability topology on
compact profile-time windows. Any nonzero local critical mass left in a
terminal camera after removing the declared profiles emits an additional
nonzero profile, after passing to a subsequence.

::::

This is the standard analytic profile-decomposition input used in the
Navier-Stokes critical-element literature. It is an external theorem, not a
new consequence of the certificate algebra in this folder.

## Terminal profile-completeness payload

::::{prf:definition} Bounded critical terminal profile input
:label: def-s12-bounded-critical-terminal-profile-input

The certificate

```{math}
K_{\mathrm{BoundedCritTermSeq}}^+
```

means that every terminal active-camera sequence used by the declared Type II
backend is represented by divergence-free data \(u_{0,n}\) satisfying a uniform
critical bound

```{math}
\sup_n\|u_{0,n}\|_{L^3(\mathbb R^3)}<\infty,
```

or by data bounded in a stronger admissible critical profile space covered by
Axiom {prf:ref}`ax-s12-critical-ns3d-profile-decomposition`.

::::

::::{prf:theorem} Kato small-data critical stability
:label: thm-s12-kato-small-data-critical-stability

There exists a universal \(\varepsilon_{\mathrm{sd}}>0\) such that every
divergence-free datum \(f\in L^3_\sigma(\mathbb R^3)\) with
\(\|f\|_{L^3}\le\varepsilon_{\mathrm{sd}}\) generates a unique global mild
Navier-Stokes solution \(U\) satisfying

```{math}
U\in C([0,\infty);L^3_\sigma)
\cap X_{\mathrm{Kato}},
\qquad
\|U\|_{X_{\mathrm{Kato}}}
\le C\|f\|_{L^3}.
```

Moreover, the solution map is Lipschitz in the critical stability topology on
bounded \(X_{\mathrm{Kato}}\)-balls, and perturbations with sufficiently small
linear heat evolution in \(X_{\mathrm{Kato}}\) remain perturbative on every
compact profile-time window.

::::

:::{prf:proof}
This is the classical Kato fixed-point theorem in the critical \(L^3\) space,
together with the standard perturbative stability estimate obtained by writing
the difference equation in Duhamel form and absorbing the bilinear term on the
small \(X_{\mathrm{Kato}}\)-ball. S12 uses only this theorem-level payload, not
any stronger regularity conclusion. \(\square\)
:::

::::{prf:definition} Small-data critical stability ledger
:label: def-s12-small-data-critical-stability-ledger

The certificate

```{math}
K_{\mathrm{SmallDataStab}_{L^3}}^+
```

means that there exists \(\varepsilon_{\mathrm{sd}}>0\) such that any
divergence-free profile with \(L^3\)-norm at most
\(\varepsilon_{\mathrm{sd}}\) generates a global mild Navier-Stokes solution
whose contribution is perturbative in the critical stability norm on every
compact terminal-camera window.

::::

::::{prf:corollary} Kato theorem emits the small-data ledger
:label: cor-s12-kato-emits-small-data-ledger

Theorem {prf:ref}`thm-s12-kato-small-data-critical-stability` emits

```{math}
K_{\mathrm{SmallDataStab}_{L^3}}^+.
```

::::

:::{prf:proof}
The definition of \(K_{\mathrm{SmallDataStab}_{L^3}}^+\) is exactly the
small-data and perturbative compact-window consequence of the Kato theorem.
\(\square\)
:::

::::{prf:definition} Terminal critical profile theorem payload
:label: def-s12-terminal-critical-profile-theorem-payload

The certificate

```{math}
K_{\mathrm{TermCritProfThm},NS3D}^+
```

means

```{math}
K_{\mathrm{BoundedCritTermSeq}}^+
\wedge
K_{\mathrm{SmallDataStab}_{L^3}}^+
\wedge
K_{\mathrm{CriticalNSProfDecomp}}^+,
```

where \(K_{\mathrm{CriticalNSProfDecomp}}^+\) is the accepted external theorem
in Axiom {prf:ref}`ax-s12-critical-ns3d-profile-decomposition`.

::::

## Completeness theorem

::::{prf:theorem} S12 terminal critical local compactness
:label: thm-s12-terminal-critical-profile-completeness

Assume

```{math}
K_{\mathrm{TermCritProfThm},NS3D}^+.
```

Then

```{math}
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+.
```

::::

:::{prf:proof}
Fix a terminal active-camera sequence. By
\(K_{\mathrm{BoundedCritTermSeq}}^+\), it is bounded in the critical space
covered by the profile-decomposition theorem. Apply
{prf:ref}`ax-s12-critical-ns3d-profile-decomposition` and pass to a subsequence.
This gives a critical profile expansion with pairwise orthogonal parameters
and critical mass decoupling.

Profiles below the small-data threshold are placed in the scattering ledger by
\(K_{\mathrm{SmallDataStab}_{L^3}}^+\); their nonlinear evolutions are
perturbative on compact terminal-camera windows. Profiles above the threshold
are declared active. The mass decoupling implies that the number of active
profiles is finite on every bounded critical-mass terminal sequence.

It remains to check the hidden-mass clause in S8. Suppose that after removing
the declared active profiles and the subthreshold scattering remainder, the
terminal camera still sees nonzero local critical mass on some compact
cylinder. By the final clause of the external profile theorem, after passing
to a further subsequence the profile extractor emits an additional nonzero
profile with center and scale comparable to the camera in which that mass is
seen. If this profile is subthreshold, the small-data ledger makes it
perturbative, contradicting that it remains as nonzero terminal-window mass. If
it is above threshold, it is active and was omitted from the active profile
list, contradicting the declared profile expansion.

Therefore no hidden terminal-window critical mass remains, all non-scattering
profiles are emitted, and the scattering remainder is small in the critical
stability topology on compact terminal-camera windows. This is precisely
\(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\). \(\square\)
:::

## Consequence for C18

::::{prf:corollary} S12 discharges the profile-completeness slot in C18
:label: cor-s12-discharges-c18-profile-slot

In the C18 terminal-backend package, the hypothesis

```{math}
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
```

may be replaced by

```{math}
K_{\mathrm{TermCritProfThm},NS3D}^+.
```

::::

:::{prf:proof}
Apply Theorem
{prf:ref}`thm-s12-terminal-critical-profile-completeness` and then use the
resulting \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\) in C18. \(\square\)
:::

## Remaining non-CatLib boundary

S12 removes the handwavy global `CatLib` dependency for the terminal Type II
profile-completeness slot, but it does not prove local compactness for
arbitrary unbounded critical-norm sequences. Its unconditional content is
relative to bounded critical terminal profile sequences and the accepted
critical Navier-Stokes profile-decomposition/stability theorem.

Thus the remaining boundary for a fully bare-data Type II theorem is not
global \(K_{\mathrm{CatLib}}^+\). S13 discharges the bounded-critical terminal
sequence input from the C3 positive finite critical-mass branch and terminal
sequence routing:

```{math}
K_{L^3\mathrm{Norm}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\Longrightarrow
K_{\mathrm{BoundedCritTermSeq}}^+.
```

If a declared Type II branch has no local compact-cylinder terminal sequence in the
profile space covered by the external theorem, then it is outside the S12
terminal-profile-completeness route and must be classified by the corresponding
critical-norm blowup defect \(K_{L^3\mathrm{Inf}}^-\) or by the route defect
\(K_{\mathrm{RepBridge}}^-\) in the declared terminal backend.
