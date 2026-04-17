# U3b: stratified critical-mass bookkeeping

This note discharges U3 in the form needed by the Type II state-space
stratification program. The critical-mass ledger is **not** a global estimate
for the full Navier-Stokes solution. It is a local ledger on each retained
terminal stratum/profile state produced by the exhaustive state-space
partition.

The object controlled here is a terminal stratum state
\(\Phi_{\mathfrak s}\), not the full solution \(u(t)\) and not the sum of all
profiles/radiation at once. This distinction is essential: global critical
norms can fail to be bounded while every retained local stratum has a finite
positive critical size.

## Stratified critical states

::::{prf:definition} Exhaustive terminal state-space partition
:label: def-u3b-exhaustive-terminal-state-partition

The certificate

```{math}
K_{\mathrm{StateStratExh}}^+
```

means that the terminal Type II backend partitions every retained terminal
configuration into typed strata

```{math}
\mathcal S_{\mathrm{term}}
=
\mathcal S_{\mathrm{core}}
\sqcup
\mathcal S_{\mathrm{scatt}}
\sqcup
\mathcal S_{\mathrm{ext}}
\sqcup
\mathcal S_{\mathrm{rad}}
\sqcup
\mathcal S_{\mathrm{rough}}
\sqcup
\mathcal S_{\mathrm{multi}},
```

and every terminal piece is assigned to exactly one stratum. The active
finite-critical terminal profile argument uses only retained active strata

```{math}
\mathfrak s\in\mathcal S_{\mathrm{act}}
\subset
\mathcal S_{\mathrm{core}}
\cup
\mathcal S_{\mathrm{multi}}.
```

Scattering strata are discharged by S9, exterior strata by S10, radiative
strata by C6/S4--S8, rough strata by C7/S11, and scale-collapse/multibubble
remainders by S3/S5--S8.

::::

::::{prf:definition} Retained terminal stratum state
:label: def-u3b-retained-terminal-stratum-state

For each retained active terminal stratum \(\mathfrak s\), the backend assigns
a profile or compound profile state

```{math}
\Phi_{\mathfrak s}\in L^3_\sigma(\mathbb R^3)
```

obtained by applying the terminal camera to that stratum only. Its stratum
critical mass is

```{math}
N_{\mathfrak s}:=\|\Phi_{\mathfrak s}\|_{L^3(\mathbb R^3)}.
```

This norm is local to the state-space stratum. It is not the global
\(L^3\)-norm of the full physical solution.

::::

::::{prf:definition} Stratified finite critical-mass certificate
:label: def-u3b-stratified-critical-mass-certificate

The certificate

```{math}
K_{\mathrm{StratCritMass}}^+
```

means that every retained active terminal stratum has a positive finite
critical mass:

```{math}
0<N_{\mathfrak s}<\infty
\qquad
\text{for every }\mathfrak s\in\mathcal S_{\mathrm{act}}.
```

When a uniform finite active-strata packet is needed, the stronger packet
certificate

```{math}
K_{\mathrm{StratCritPacket}}^+(\eta,M,J)
```

means that the retained active packet has at most \(J\) strata and

```{math}
\eta\le N_{\mathfrak s}\le M
\qquad
\text{for every active }\mathfrak s
\text{ in the packet}.
```

::::

## Local bounds for each active stratum

::::{prf:lemma} Nonzero active strata have positive critical mass
:label: lem-u3b-active-strata-positive

Assume \(\mathfrak s\) is retained as an active terminal stratum, rather than
being sent to the scattering, exterior, radiative, rough-core, or lost-profile
ledger. Then

```{math}
N_{\mathfrak s}>0.
```

If the small-data scattering threshold \(\varepsilon_{\mathrm{sd}}>0\) is used
as the active cutoff, then

```{math}
N_{\mathfrak s}\ge \varepsilon_{\mathrm{sd}}.
```

::::

:::{prf:proof}
A retained active stratum is, by definition, represented by a nonzero terminal
profile or compound profile after all scattering and exterior pieces have been
removed. If \(N_{\mathfrak s}=0\), then
\(\Phi_{\mathfrak s}=0\) in \(L^3\), so the stratum contains no critical
profile state and must be routed to the zero/lost-profile ledger rather than
retained as active. If the active ledger is defined by the small-data cutoff,
then every retained active profile is not in the small-data scattering class;
therefore its critical norm is at least \(\varepsilon_{\mathrm{sd}}\).
\(\square\)
:::

::::{prf:lemma} Terminal profile states have finite critical mass
:label: lem-u3b-active-strata-finite

Assume the terminal profile-completeness theorem used by S12 applies to the
terminal stratum. Then

```{math}
N_{\mathfrak s}<\infty
```

for every retained active stratum \(\mathfrak s\).

::::

:::{prf:proof}
S12's terminal profile theorem emits profiles in the critical divergence-free
profile space, here \(L^3_\sigma(\mathbb R^3)\) or a stronger admissible
critical profile space embedded in the required local profile topology. A
retained compound profile is a finite sum of comparable-scale profiles in the
same terminal camera. Finite sums of \(L^3\) profiles remain in \(L^3\). Hence
\(\Phi_{\mathfrak s}\in L^3_\sigma\) and
\(N_{\mathfrak s}<\infty\). \(\square\)
:::

::::{prf:theorem} U3b local critical mass on every retained stratum
:label: thm-u3b-local-critical-mass-each-stratum

Assume

```{math}
K_{\mathrm{StateStratExh}}^+
\wedge
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+.
```

Then

```{math}
K_{\mathrm{StratCritMass}}^+.
```

::::

:::{prf:proof}
By \(K_{\mathrm{StateStratExh}}^+\), every terminal piece is assigned to a
unique stratum. Consider a retained active terminal stratum
\(\mathfrak s\in\mathcal S_{\mathrm{act}}\). Lemma
{prf:ref}`lem-u3b-active-strata-positive` gives \(N_{\mathfrak s}>0\), and
Lemma {prf:ref}`lem-u3b-active-strata-finite` gives
\(N_{\mathfrak s}<\infty\). Since \(\mathfrak s\) was arbitrary, every retained
active stratum has positive finite critical mass. This is
\(K_{\mathrm{StratCritMass}}^+\). \(\square\)
:::

## Finite active packet bounds

::::{prf:theorem} Bounded terminal sequence gives finite active packet
:label: thm-u3b-finite-active-packet

Assume a terminal packet is extracted from a bounded terminal critical sequence
with

```{math}
\sup_n\|u_{0,n}\|_{L^3}\le M_0,
```

and active profiles are those with critical norm at least
\(\varepsilon_{\mathrm{sd}}>0\). Then the number of active retained strata in
that packet is finite and satisfies

```{math}
\#\mathcal S_{\mathrm{act}}
\le
\left\lfloor M_0^3/\varepsilon_{\mathrm{sd}}^3\right\rfloor.
```

Moreover each active stratum satisfies

```{math}
\varepsilon_{\mathrm{sd}}
\le N_{\mathfrak s}
\le M_0.
```

Thus the packet emits

```{math}
K_{\mathrm{StratCritPacket}}^+(\varepsilon_{\mathrm{sd}},M_0,
\lfloor M_0^3/\varepsilon_{\mathrm{sd}}^3\rfloor).
```

::::

:::{prf:proof}
Critical \(L^3\)-mass decoupling in the terminal profile decomposition gives,
for every finite active subset \(A\),

```{math}
\sum_{\mathfrak s\in A}N_{\mathfrak s}^3
\le M_0^3.
```

Since each active stratum has
\(N_{\mathfrak s}\ge\varepsilon_{\mathrm{sd}}\),

```{math}
\#A\,\varepsilon_{\mathrm{sd}}^3\le M_0^3.
```

This holds for every finite active subset, so the active set is finite with
the displayed cardinality bound. The upper bound
\(N_{\mathfrak s}\le M_0\) follows from the same decoupling inequality applied
to the singleton \(A=\{\mathfrak s\}\). \(\square\)
:::

## Replacement for global C3 in the terminal backend

::::{prf:corollary} Stratified replacement for the C3 terminal critical-mass slot
:label: cor-u3b-stratified-replacement-for-c3

In S12--S14 and C18, whenever the argument uses critical mass only to control
retained active terminal profile states, the global certificate

```{math}
K_{L^3\mathrm{Norm}}^+
```

may be replaced by the stratified packet certificate

```{math}
K_{\mathrm{StratCritPacket}}^+(\eta,M,J),
```

or, for pointwise stratum arguments, by

```{math}
K_{\mathrm{StratCritMass}}^+.
```

::::

:::{prf:proof}
The terminal profile and decoupling arguments use the critical norm only for
three local purposes: nonzero active profile mass, finite active profile mass,
and finite active profile count in a bounded packet. These are exactly the
outputs of Theorems {prf:ref}`thm-u3b-local-critical-mass-each-stratum` and
{prf:ref}`thm-u3b-finite-active-packet`. No step in S12--S14 requires a global
\(L^3\) estimate for the full physical solution once the state-space partition
has assigned each terminal piece to a stratum. \(\square\)
:::

## Residual bookkeeping

The remaining U3 work is now bookkeeping, not a global a priori estimate:

1. verify that every C18 use of \(K_{L^3\mathrm{Norm}}^+\) is a stratum-local
   use and replace it by
   \(K_{\mathrm{StratCritMass}}^+\) or
   \(K_{\mathrm{StratCritPacket}}^+\);
2. keep global \(L^3\)-unboundedness out of the terminal finite-stratum
   ledger; it belongs to the radiation, multistrata, cascade, or rough-core
   classification routes;
3. retain \(K_{\mathrm{StateStratExh}}^+\) as the explicit exhaustive
   partition certificate.

No global \(L^3\) estimate is used or claimed in U3b.
