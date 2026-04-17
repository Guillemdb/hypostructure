# C1 Type II branch exhaustion

This note implements C1 in the classification-completeness program. Its role is
to make precise the statement:

```{math}
\text{every declared Navier-Stokes Type II candidate enters the Type II sieve branch.}
```

This is not a global regularity theorem. It is an exhaustion theorem for the
declared Type II backend: once the backend has identified a candidate as Type
II, C1 proves that the candidate is represented by the certificates consumed by
C2--C4.

The output certificate is

```{math}
K_{\mathrm{TypeIIExhaust}}^+.
```

## Declared Type II universe

::::{prf:definition} Declared Navier-Stokes Type II universe
:label: def-declared-ns-typeII-universe

The **declared Navier-Stokes Type II universe** is the class
\(\mathcal U_{\mathrm{II}}^{NS}\) of candidate terminal branches
\((u,p,T^*)\) in the Navier-Stokes dataset backend satisfying:

1. \(u,p\) solve the 3D incompressible Navier-Stokes equations on
   \([0,T^*)\) in the declared analytic class of the dataset;
2. \(T^*<\infty\) is a terminal time for the declared branch;
3. the branch is not classified as Type I by the dataset's Type I criterion;
4. the branch is routed by the dataset as a Type II candidate rather than as a
   continuation-success, Type I, boundary, or non-Navier-Stokes domain failure.

Membership in \(\mathcal U_{\mathrm{II}}^{NS}\) is therefore a backend
declaration. C1 does not assert that such branches exist. It only classifies
what the sieve must do if one is declared.

::::

## Exhaustion payloads

C1 is the bridge from the declared Type II universe to the sieve/profile
certificates.

::::{prf:definition} Type II exhaustion payload
:label: def-typeII-exhaustion-payload

A **Type II exhaustion payload** is a tuple

```{math}
\mathsf{Exh}_{\mathrm{II}}^{NS}
=
\bigl(
\mathcal U_{\mathrm{II}}^{NS},
\mathsf{ConcentrationExtract},
\mathsf{ScaleRoute},
\mathsf{ProfileComplete},
\mathsf{NoLostTypeII}
\bigr)
```

with the following components.

1. **Concentration extraction.** For every
   \((u,p,T^*)\in\mathcal U_{\mathrm{II}}^{NS}\), the compactness interface
   emits
   ```{math}
   K_{C_\mu}^+.
   ```
   This means the candidate has a concentration profile or blow-up germ modulo
   the declared Navier-Stokes symmetry group.

2. **Scale routing.** For every
   \((u,p,T^*)\in\mathcal U_{\mathrm{II}}^{NS}\), the scaling interface emits
   ```{math}
   K_{\mathrm{SC}_\lambda}^-.
   ```
   This records that the branch lies in the supercritical scaling route that
   triggers `BarrierTypeII`.

3. **Profile backend completeness.** For every emitted concentration profile,
   the profile backend emits
   ```{math}
   K_{\mathrm{Prof}_{NS}}^+.
   ```

4. **No lost Type II branch.** The backend emits no Type II candidate outside
   the above three outputs. Equivalently, every declared Type II candidate is
   either routed to the triple
   ```{math}
   K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-\wedge K_{\mathrm{Prof}_{NS}}^+
   ```
   or the backend emits one of the explicit defects in Definition
   {prf:ref}`def-typeII-exhaustion-defects`.

The positive certificate for this payload is denoted

```{math}
K_{\mathrm{ExhPayload}}^+.
```

::::

## Type II branch exhaustion certificate

::::{prf:definition} Candidate Type II route certificate
:label: def-candidate-typeII-route-certificate

For a fixed declared Type II candidate
\(\omega=(u,p,T^*)\in\mathcal U_{\mathrm{II}}^{NS}\), define

```{math}
K_{\mathrm{TypeIIRoute}}^+(\omega)
```

to mean that the candidate emits the triple

```{math}
K_{C_\mu}^+(\omega)
\wedge
K_{\mathrm{SC}_\lambda}^-(\omega)
\wedge
K_{\mathrm{Prof}_{NS}}^+(\omega).
```

This is the per-candidate route certificate. The global exhaustion certificate
below is the universal closure of these per-candidate certificates.

::::

::::{prf:definition} Type II branch exhaustion certificate
:label: def-typeII-exhaustion-certificate

The certificate

```{math}
K_{\mathrm{TypeIIExhaust}}^+
```

means:

```{math}
\forall (u,p,T^*)\in\mathcal U_{\mathrm{II}}^{NS},\qquad
K_{\mathrm{TypeIIRoute}}^+(u,p,T^*)\text{ is emitted.}
```

This is an exhaustion certificate for the declared Type II backend. It is not a
claim that \(\mathcal U_{\mathrm{II}}^{NS}\) is nonempty, nor a claim that all
Navier-Stokes singularities are Type II.

::::

## Main C1 theorem

::::{prf:theorem} C1 Type II branch exhaustion
:label: thm-c1-typeII-branch-exhaustion

Assume the Navier-Stokes Type II backend supplies

```{math}
K_{\mathrm{ExhPayload}}^+.
```

Then it emits

```{math}
K_{\mathrm{TypeIIExhaust}}^+.
```

::::

:::{prf:proof}
Let \((u,p,T^*)\in\mathcal U_{\mathrm{II}}^{NS}\). By the concentration
extraction component of \(K_{\mathrm{ExhPayload}}^+\), the compactness
interface emits \(K_{C_\mu}^+\) for this branch. By the scale-routing component,
the scaling interface emits \(K_{\mathrm{SC}_\lambda}^-\). By profile backend
completeness, the extracted profile emits \(K_{\mathrm{Prof}_{NS}}^+\).

Since the candidate was arbitrary in \(\mathcal U_{\mathrm{II}}^{NS}\), the
triple

```{math}
K_{C_\mu}^+
\wedge
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{Prof}_{NS}}^+
```

is emitted for every declared Type II branch, i.e.
\(K_{\mathrm{TypeIIRoute}}^+(u,p,T^*)\) is emitted for every branch. This is exactly
\(K_{\mathrm{TypeIIExhaust}}^+\). \(\square\)
:::

## NS3D dataset payload

We now instantiate the abstract payload above for the concrete 3D
Navier-Stokes dataset in
[../dataset/navier_stokes_3d.md](../dataset/navier_stokes_3d.md).

::::{prf:definition} NS3D Type II exhaustion payload
:label: def-ns3d-typeII-exhaustion-payload

The **NS3D Type II exhaustion payload** is

```{math}
\mathsf{Exh}_{\mathrm{II}}^{NS3D}
=
\bigl(
\mathcal U_{\mathrm{II}}^{NS},
\mathsf{ConcentrationExtract}_{NS3D},
\mathsf{ScaleRoute}_{NS3D},
\mathsf{ProfileComplete}_{NS3D},
\mathsf{NoLostTypeII}_{NS3D}
\bigr).
```

Its components are:

1. **Concentration extraction from Node 3.**
   The dataset's compactness interface \(C_\mu\) emits
   ```{math}
   K_{C_\mu}^+,
   ```
   whose payload is a concentration profile / blow-up germ modulo the declared
   Navier-Stokes symmetry group.

2. **Scale routing from Node 4.**
   The dataset's scaling interface emits
   ```{math}
   K_{\mathrm{SC}_\lambda}^-,
   ```
   recording the supercritical scaling route that triggers `BarrierTypeII`.

3. **Profile completeness from the singularity module.**
   After Node 3, the dataset's singularity module emits
   ```{math}
   K_{\mathrm{Prof}_{NS}}^+,
   \qquad
   K_{\mathrm{Prof}_{NS}}^+\in
   \{K_{\mathrm{lib}}^+,K_{\mathrm{strat}}^+\}.
   ```

4. **No lost Type II branch by declared-universe convention.**
   The universe \(\mathcal U_{\mathrm{II}}^{NS}\) consists only of branches
   routed by the NS3D dataset as Type II candidates. Therefore the NS3D Type II
   backend sends every element of \(\mathcal U_{\mathrm{II}}^{NS}\) either to
   the triple
   ```{math}
   K_{C_\mu}^+\wedge
   K_{\mathrm{SC}_\lambda}^-\wedge
   K_{\mathrm{Prof}_{NS}}^+
   ```
   or to one of the ordered exhaustion defects
   ```{math}
   K_{\mathrm{CmuExtract}}^-,
   \quad
   K_{\mathrm{ScaleRoute}}^-,
   \quad
   K_{\mathrm{ProfComplete}}^-,
   \quad
   K_{\mathrm{LostTypeII}}^-.
   ```

The positive certificate for this concrete payload is denoted

```{math}
K_{\mathrm{ExhPayload},NS3D}^+.
```

::::

::::{prf:lemma} NS3D dataset supplies the Type II exhaustion payload
:label: lem-ns3d-supplies-exhpayload

For the declared Type II universe \(\mathcal U_{\mathrm{II}}^{NS}\) of the
NS3D dataset backend, the payload

```{math}
K_{\mathrm{ExhPayload},NS3D}^+
```

is available.

::::

:::{prf:proof}
The NS3D dataset records the core sieve outputs

```{math}
K_{C_\mu}^+,
\qquad
K_{\mathrm{SC}_\lambda}^-,
\qquad
K_{\mathrm{Prof}_{NS}}^+.
```

The first is the Node 3 compactness/concentration output. The second is the
Node 4 supercritical scaling output. The third is the singularity-module
profile output, with branch
\(K_{\mathrm{Prof}_{NS}}^+\in\{K_{\mathrm{lib}}^+,K_{\mathrm{strat}}^+\}\).

By Definition {prf:ref}`def-declared-ns-typeII-universe`, a branch belongs to
\(\mathcal U_{\mathrm{II}}^{NS}\) only if the NS3D backend has routed it as a
Type II candidate. The declared NS3D Type II backend therefore has no additional
untyped Type II output channel: a declared Type II candidate either carries the
three positive outputs above or emits one of the typed exhaustion defects listed
in Definition {prf:ref}`def-typeII-exhaustion-defects`.

These are exactly the four components of Definition
{prf:ref}`def-ns3d-typeII-exhaustion-payload`. Hence
\(K_{\mathrm{ExhPayload},NS3D}^+\) is available. \(\square\)
:::

::::{prf:corollary} NS3D Type II branch exhaustion
:label: cor-ns3d-typeII-branch-exhaustion

For the declared Type II universe of the NS3D dataset backend,

```{math}
K_{\mathrm{TypeIIExhaust}}^+
```

is emitted.

::::

:::{prf:proof}
Apply Theorem {prf:ref}`thm-c1-typeII-branch-exhaustion` to the concrete
payload \(K_{\mathrm{ExhPayload},NS3D}^+\) supplied by Lemma
{prf:ref}`lem-ns3d-supplies-exhpayload`. \(\square\)
:::

## Exhaustion defects

If the exhaustion certificate cannot be emitted, the failure is classified.

::::{prf:definition} Type II exhaustion defects
:label: def-typeII-exhaustion-defects

For a declared Type II candidate, the ordered exhaustion defects are:

1. **Concentration extraction defect**
   ```{math}
   K_{\mathrm{CmuExtract}}^-:
   \quad
   \text{the candidate is declared Type II but }K_{C_\mu}^+
   \text{ is not emitted.}
   ```
2. **Scale routing defect**
   ```{math}
   K_{\mathrm{ScaleRoute}}^-:
   \quad
   K_{C_\mu}^+\text{ is emitted, but }K_{\mathrm{SC}_\lambda}^-
   \text{ is not emitted.}
   ```
3. **Profile completeness defect**
   ```{math}
   K_{\mathrm{ProfComplete}}^-:
   \quad
   K_{C_\mu}^+\wedge K_{\mathrm{SC}_\lambda}^-
   \text{ is emitted, but }K_{\mathrm{Prof}_{NS}}^+
   \text{ is not emitted.}
   ```
4. **Backend leakage defect**
   ```{math}
   K_{\mathrm{LostTypeII}}^-:
   \quad
   \text{the backend declares a Type II branch outside its certified
   concentration/scale/profile route.}
   ```

The first applicable defect in this order is the emitted exhaustion-failure
certificate.

::::

::::{prf:corollary} Ordered C1 exhaustion classification
:label: cor-c1-exhaustion-classification

For every fixed candidate \(\omega=(u,p,T^*)\) declared by the Navier-Stokes
backend as Type II, exactly one ordered output certificate is emitted:

1. \(K_{\mathrm{TypeIIRoute}}^+(\omega)\);
2. \(K_{\mathrm{CmuExtract}}^-\);
3. \(K_{\mathrm{ScaleRoute}}^-\);
4. \(K_{\mathrm{ProfComplete}}^-\);
5. \(K_{\mathrm{LostTypeII}}^-\).

Moreover, \(K_{\mathrm{TypeIIExhaust}}^+\) holds if and only if
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\) holds for every
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\).

::::

:::{prf:proof}
Run the exhaustion checks in the order of Definition
{prf:ref}`def-typeII-exhaustion-defects` for the fixed candidate \(\omega\). If
the concentration, scale, and profile checks all succeed, then
\(K_{\mathrm{TypeIIRoute}}^+(\omega)\) is emitted. If a check fails, the first
failed check emits the corresponding defect certificate. The outputs are
disjoint by first-failure ordering and exhaustive by typed certificate
evaluation. The final equivalence is the definition of
\(K_{\mathrm{TypeIIExhaust}}^+\). \(\square\)
:::

## Coupling C1 with C2--C4

::::{prf:theorem} Declared Type II candidates enter the bridge ledger
:label: thm-c1-enters-bridge-ledger

Assume:

```{math}
K_{\mathrm{TypeIIExhaust}}^+.
```

Then every declared Navier-Stokes Type II candidate enters the ordered bridge
ledger. For each candidate, exactly one of the following occurs:

1. it reaches the compact barrier hypotheses after the tightness and
   windowed-\(H^1\) checks and, by C4, emits `BarrierTypeII`;
2. it emits one of the explicit downstream defects

```{math}
K_{\mathrm{ProfOrb}}^-,
K_{\mathrm{GaugeReal}}^-,
K_{\mathrm{PressureRep}}^-,
K_{\mathrm{ModParams}}^-,
K_{L^3\mathrm{Zero}}^-,
K_{L^3\mathrm{Inf}}^-,
K_{L^3\mathrm{Dom}}^-,
K_{L^3\mathrm{Tight}}^-,
K_{\mathrm{WinH1}}^-.
```

::::

:::{prf:proof}
By \(K_{\mathrm{TypeIIExhaust}}^+\), every declared Type II candidate enters the
sieve branch

```{math}
K_{C_\mu}^+
\wedge
K_{\mathrm{SC}_\lambda}^-
\wedge
K_{\mathrm{Prof}_{NS}}^+.
```

C2 then either emits \(K_{\mathrm{RepBridge}}^+\) or one of the ordered
representation defects. If \(K_{\mathrm{RepBridge}}^+\) is emitted, C3 then
either emits \(K_{L^3\mathrm{Norm}}^+\) or one of the ordered critical-mass
defects. If \(K_{L^3\mathrm{Norm}}^+\) is emitted, C4 supplies the default cost
bridge. The only remaining compact-barrier inputs are tightness and local
windowed \(H^1\) control; their failures are exactly
\(K_{L^3\mathrm{Tight}}^-\) and \(K_{\mathrm{WinH1}}^-\). If both are present,
the compact barrier and C4 emit `BarrierTypeII`. This is an ordered exhaustive
ledger for each declared Type II candidate. \(\square\)
:::

## What C1 discharges

C1 removes the last vague phrase "every actual Type II candidate should enter
the sieve" and replaces it by the certificate

```{math}
K_{\mathrm{TypeIIExhaust}}^+.
```

If the declared backend cannot emit that certificate, it must emit one of:

```{math}
K_{\mathrm{CmuExtract}}^-,
\qquad
K_{\mathrm{ScaleRoute}}^-,
\qquad
K_{\mathrm{ProfComplete}}^-,
\qquad
K_{\mathrm{LostTypeII}}^-.
```

Thus C1 does not hide the hard global profile-extraction problem. It packages it
as an auditable backend completeness certificate. Once this certificate is
present, C2--C4 apply to every declared Type II branch rather than merely to an
individual represented candidate.
